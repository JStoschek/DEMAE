"""The geomorphon label pass `freeze --task landform` runs: census, labels, fingerprints, weights (#201).

:mod:`benchmark.geomorphon` labels one tile. This module walks Task 1's committed lists with it,
in the order ADR-0014 and plan §4.4 step 1 give, and writes what Task 2 commits about its labels:

1. **The census** (D5), before any read: every COG stem each tile's window reaches at the widest
   L, against the release frame. An absent stem that is on none of the committed lists
   (``verified_ocean``, ``absent_land``) refuses the pass, naming it, so no absence is assumed
   to be ocean *or* land without someone checking it. It costs no network.
2. **The labels**: every tile of every variant's splits, read once at the widest L its variants
   need (L = 100 for a test tile), each variant cropped from that read (D4, D7). Rasters and a
   resume log go under ``paths.labels_dir/<label config hash>/``; neither is committed. A tile
   already in the log with its raster on disk is skipped, so an interrupted pass resumes. A
   failed read raises and stops the pass: there is no fallback window.
3. **The fingerprints** (D8): one CSV per variant (``tile_id, content_hash, input_hash`` and the
   tile's per-class pixel counts), and ``geomorphon_config.json``: the label config and its
   hash, each variant's set fingerprint, the census, and the class weights (D6). Both through
   :func:`benchmark.artifacts.write_committed`, so a moved label set refuses rather than
   overwrites.
4. **The fixed sample** (story 18): a few test tiles regenerated from the COGs, every variant,
   and required byte-identical to the committed fingerprints. When fingerprints are already
   committed it runs *first*, so a different WBT build or a moved upstream is caught before it
   labels anything.

The WBT call is reached through :func:`make_labeller`, which is substitution point A: tests
swap in a deterministic function of the window and the four parameters.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np

from . import artifacts
from . import geomorphon as gm
from .config import _hash

if TYPE_CHECKING:
    from .landform import LandformResolution
    from .landform_config import Geomorphon, LandformConfig

#: ``geomorphon_config.json``: the label config, its hash, the census, set fingerprints, weights.
CONFIG_NAME: Final = "geomorphon_config.json"

#: One fingerprint CSV per variant, in this directory under ``paths.artifacts_dir``.
FINGERPRINT_DIR: Final = "geomorphon_fingerprints"

#: Task 1's committed lists Task 2 labels, by split (story 32: reused unchanged).
LIST_NAMES: Final = {
    "train": artifacts.TILE_LIST_NAMES["20k"],
    "val": artifacts.VAL_TILES_NAME,
    "test": artifacts.TEST_TILES_NAME,
}

FINGERPRINT_COLUMNS: Final = ("tile_id", "content_hash", "input_hash")


def fingerprint_path(artifacts_dir: Path | str, variant: str) -> Path:
    return Path(artifacts_dir) / FINGERPRINT_DIR / f"{variant}.csv"


# --- The label config and its hash (D8) -----------------------------------------------------------


def label_config(g: Geomorphon) -> dict:
    """What a label is a function of: the WBT pin, the four parameters, and the three rules.

    D8's config hash covers exactly this. The exception lists are part of the ignore rule. The
    fixed-sample size is not: it decides what is checked, not what is labelled.
    """
    return {
        "tool": f"WhiteboxTools v{g.wbt_version} Geomorphons --forms",
        "wbt_sha256": list(g.wbt_sha256),
        "params": {"search": g.search, "threshold": g.threshold, "skip": g.skip, "fdist": g.fdist},
        "variants": {
            v.name: {**v.params.__dict__, "splits": list(v.splits)} for v in gm.variants(g)
        },
        "buffer_rule": gm.BUFFER_RULE,
        "spacing_rule": gm.SPACING_RULE,
        "ignore_rule": gm.IGNORE_RULE,
        "ignore_label": g.ignore_label,
        "classes": list(_classes()),
        "verified_ocean": list(g.verified_ocean),
        "absent_land": list(g.absent_land),
        "antimeridian_wrap": list(g.antimeridian_wrap),
    }


def label_config_hash(g: Geomorphon) -> str:
    return _hash(label_config(g))


def _classes() -> tuple[str, ...]:
    from .landform_config import GEOMORPHON_CLASSES

    return GEOMORPHON_CLASSES


# --- Task 1's lists -------------------------------------------------------------------------------


def read_lists(task1_dir: Path | str) -> dict[str, tuple[str, ...]]:
    """``split -> tile ids`` from Task 1's committed lists, refusing a tile in two of them."""
    from pretrain.artifact import read_tile_strata

    out: dict[str, tuple[str, ...]] = {}
    for split, name in LIST_NAMES.items():
        path = Path(task1_dir) / name
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} is not frozen. Task 2 labels Task 1's committed lists unchanged "
                f"(story 32); run `python -m benchmark.launch freeze` first"
            )
        out[split] = tuple(t for t, _ in read_tile_strata(path, what=f"{split} list"))
    seen = Counter(t for ids in out.values() for t in ids)
    twice = sorted(t for t, n in seen.items() if n > 1)
    if twice:
        raise ValueError(f"{len(twice)} tiles are in two of Task 1's lists, e.g. {twice[:3]}")
    return out


def _read_search(g: Geomorphon, split: str) -> int:
    """The L a tile of ``split`` is read at: the widest of the variants that label it (D4)."""
    return max(v.params.search for v in gm.variants(g) if split in v.splits)


# --- 1. The census --------------------------------------------------------------------------------


@dataclass(frozen=True)
class Census:
    """Every absent COG the windows reach at the widest L, and how D5 treats it."""

    search: int
    tiles: int
    #: ``stem -> number of tiles whose window reaches it``, absent stems only.
    absent: Mapping[str, int]
    #: Tiles whose window wraps across the antimeridian.
    wrapped: int

    def treatment(self, g: Geomorphon) -> dict[str, str]:
        out = {}
        for stem in self.absent:
            if stem in g.verified_ocean:
                out[stem] = "verified ocean, read as 0 m"
            elif stem in g.absent_land:
                out[stem] = "land, absent from the release: its reach is ignored (255)"
            else:
                out[stem] = "unclassified"
        return out

    def to_dict(self, g: Geomorphon) -> dict:
        treat = self.treatment(g)
        return {
            "search": self.search,
            "tiles": self.tiles,
            "tiles_wrapping_the_antimeridian": self.wrapped,
            "absent": {s: {"tiles": self.absent[s], "treated_as": treat[s]}
                       for s in sorted(self.absent)},
        }


def census(lists: Mapping[str, Iterable[str]], frame: frozenset[str], g: Geomorphon) -> Census:
    """D5's census at the widest buffer, over every tile, from the frame alone — no reads.

    Refuses an absent stem on no committed list, and a listed exception that is not absent.
    """
    ex = gm.Exceptions.of(g)
    search = g.widest_search
    reached: Counter[str] = Counter()
    tiles = wrapped = 0
    for ids in lists.values():
        for tile_id in ids:
            tiles += 1
            header = gm.spacing(tile_id)
            cols = header.column_buffer(search)
            sources, wraps = gm.window_sources(tile_id, search, cols, ex)
            wrapped += wraps
            reached.update({s.stem for s in sources} - frame)
    present = sorted((set(g.verified_ocean) | set(g.absent_land)) & frame)
    if present:
        raise ValueError(
            f"geomorphon lists {present} as absent from the release, but the frame has them"
        )
    result = Census(search=search, tiles=tiles, absent=dict(sorted(reached.items())),
                    wrapped=wrapped)
    unclassified = [s for s, t in result.treatment(g).items() if t == "unclassified"]
    if unclassified:
        listing = "\n".join(f"  {s}  ({reached[s]:,} tiles)" for s in unclassified)
        raise ValueError(
            f"the census at L = {search} found {len(unclassified)} absent COG(s) that no "
            f"committed list classifies:\n{listing}\nCheck each one and add it to "
            f"geomorphon.verified_ocean (open ocean, read as 0 m) or geomorphon.absent_land "
            f"(its reach is ignored). No absence is assumed to be either (ADR-0014 D5)"
        )
    return result


# --- 2. The labels, with resume -------------------------------------------------------------------


def make_labeller(cfg: LandformConfig) -> gm.Labeller:
    """Substitution point A: the pinned WBT binary, refused here if it is unset or unpinned."""
    return gm.WbtLabeller(cfg.paths.wbt_binary, tuple(cfg.geomorphon.wbt_sha256))


def _build(labeller: gm.Labeller) -> str:
    """Which labeller wrote a label: the WBT binary's sha256, or a stand-in's name."""
    return getattr(labeller, "build", None) or getattr(labeller, "__qualname__", repr(labeller))


class LabelStore:
    """Label rasters and the resume log under ``labels_dir/<label config hash>/`` (gitignored).

    A raster is written (atomically) before its log line, so a line implies a raster. A torn
    last line from an interrupted write is dropped on load.
    """

    def __init__(self, root: Path | str, build: str) -> None:
        self.root = Path(root)
        self.build = build
        self.log = self.root / "progress.jsonl"
        self.records: dict[tuple[str, str], dict] = {}
        if self.log.is_file():
            for line in self.log.read_text().splitlines():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self.records[(rec["variant"], rec["tile_id"])] = rec
        mixed = sorted({r["build"] for r in self.records.values()} - {build})
        if mixed:
            raise ValueError(
                f"{self.log} holds labels written by {mixed}, and this pass labels with "
                f"{build}. Two builds' labels are never mixed unchecked (ADR-0014): move the "
                f"directory aside, or pin this build only once the fixed sample agrees"
            )

    def raster_path(self, variant: str, tile_id: str) -> Path:
        return self.root / variant / f"{tile_id}.npy"

    def done(self, variant: str, tile_id: str) -> bool:
        return (variant, tile_id) in self.records and self.raster_path(variant, tile_id).is_file()

    def put(self, variant: str, tile_id: str, label: np.ndarray, input_hash: str) -> dict:
        path = self.raster_path(variant, tile_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npy")
        np.save(tmp, label)
        os.replace(tmp, path)
        rec = {
            "variant": variant,
            "tile_id": tile_id,
            "content_hash": gm.content_hash(label),
            "input_hash": input_hash,
            "counts": gm.class_counts(label),
            "build": self.build,
        }
        with self.log.open("a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
        self.records[(variant, tile_id)] = rec
        return rec


@dataclass
class TileJob:
    """Everything one tile's job needs: where COGs are, the shards, the rules, the WBT call."""

    source: gm.CogSource
    store_shards: object  # benchmark.evaluate.TileStore
    g: Geomorphon
    labeller: gm.Labeller

    def label(self, tile_id: str, split: str, todo: Iterable[gm.Variant]) -> tuple[
        str, list[tuple[gm.Variant, np.ndarray, str]]
    ]:
        """One read at the split's widest L, then every variant in ``todo`` cropped from it."""
        header = gm.spacing(tile_id)
        search = _read_search(self.g, split)
        window = gm.read_buffered(
            self.source, tile_id, search, header.column_buffer(search),
            gm.Exceptions.of(self.g), header,
        )
        shard = self.store_shards.read(tile_id).raw_i16
        out = []
        for v in todo:
            label, ih = gm.label_tile(window, v.params, self.labeller, shard_raw_i16=shard,
                                      ignore_label=self.g.ignore_label)
            out.append((v, label, ih))
        return tile_id, out


def run_pass(
    lists: Mapping[str, tuple[str, ...]],
    job: TileJob,
    store: LabelStore,
    *,
    threads: int,
    report_every: int,
    say: Callable[[str], None] = print,
) -> tuple[int, int]:
    """Label every (variant, tile) not already in ``store``. Returns ``(labelled, skipped)`` tiles.

    Reads and labels run on ``threads`` workers; results are written from this thread. Any
    worker's exception cancels what is queued and propagates — the pass stops, and what was
    written stays for the resume.
    """
    vs = gm.variants(job.g)
    todo: list[tuple[str, str, list[gm.Variant]]] = []
    skipped = 0
    for split, ids in lists.items():
        for tile_id in ids:
            need = [v for v in vs if split in v.splits and not store.done(v.name, tile_id)]
            if need:
                todo.append((tile_id, split, need))
            else:
                skipped += 1
    total = len(todo) + skipped
    say(f"{'labels':<16} {total:,} tiles: {skipped:,} already fingerprinted, "
        f"{len(todo):,} to read on {threads} threads")
    if not todo:
        return 0, skipped
    started = time.monotonic()
    done = 0
    queue = enumerate(todo)
    with ThreadPoolExecutor(max_workers=threads) as pool:
        pending: dict = {}  # future -> submission order
        # Finished results held until every earlier one is written, so the store's log is in
        # submission order whatever order the workers finish in.
        ready: dict[int, tuple[str, list]] = {}
        next_write = 0

        def submit() -> None:
            # Bounded by what is in flight *or* held, so one slow tile cannot let the buffer
            # grow without limit behind it.
            for order, (tile_id, split, need) in queue:
                pending[pool.submit(job.label, tile_id, split, need)] = order
                if len(pending) + len(ready) >= threads * 2:
                    return

        submit()
        try:
            while pending:
                finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    ready[pending.pop(future)] = future.result()
                while next_write in ready:
                    tile_id, labels = ready.pop(next_write)
                    next_write += 1
                    for v, label, ih in labels:
                        store.put(v.name, tile_id, label, ih)
                    done += 1
                    if done % report_every == 0 or done == len(todo):
                        say(_progress(done, len(todo), skipped, time.monotonic() - started))
                submit()
        except BaseException:
            for future in pending:
                future.cancel()
            raise
    return done, skipped


def _progress(done: int, todo: int, skipped: int, elapsed: float) -> str:
    rate = elapsed / max(done, 1)
    eta = rate * (todo - done)
    return (f"{'labels':<16} {done:,}/{todo:,} tiles read ({skipped:,} skipped), "
            f"{rate:.2f} s/tile, {eta / 60:.0f} min to go")


# --- 3. Fingerprints and class weights (D6, D8) ---------------------------------------------------


def render_fingerprints(records: Iterable[dict]) -> bytes:
    """One variant's committed CSV: ``tile_id, content_hash, input_hash``, then the tile's
    per-class pixel counts and its ignored count, sorted by tile id.

    The counts ride along because every later consumer of "how much of each class" — the class
    weights, the ladders' per-rung coverage report (#200, #202) — is a sum over tiles, and the
    rasters are not committed.
    """
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow([*FINGERPRINT_COLUMNS, *_classes(), "ignore"])
    for r in sorted(records, key=lambda r: r["tile_id"]):
        writer.writerow([r["tile_id"], r["content_hash"], r["input_hash"], *r["counts"]])
    return buf.getvalue().encode()


def read_fingerprints(path: Path | str) -> dict[str, dict]:
    """``tile_id -> {content_hash, input_hash, counts}`` from a committed fingerprint CSV."""
    rows = list(csv.reader(Path(path).read_text().splitlines()))
    want = [*FINGERPRINT_COLUMNS, *_classes(), "ignore"]
    if not rows or rows[0] != want:
        raise ValueError(f"{path} is not a geomorphon fingerprint file (header {rows[:1]})")
    return {
        r[0]: {"tile_id": r[0], "content_hash": r[1], "input_hash": r[2],
               "counts": [int(x) for x in r[3:]]}
        for r in rows[1:]
    }


def set_fingerprint(records: Iterable[dict]) -> str:
    """D8's per-variant set fingerprint: BLAKE2b over the sorted ``(tile_id, content_hash)``."""
    h = hashlib.blake2b(digest_size=16)
    for tile_id, ch in sorted((r["tile_id"], r["content_hash"]) for r in records):
        h.update(f"{tile_id},{ch}\n".encode())
    return h.hexdigest()


def class_weights(counts: Iterable[Iterable[int]]) -> tuple[dict[str, int], dict[str, float]]:
    """D6: inverse-sqrt-frequency weights over non-ignore pixels, scaled to mean 1 over classes.

    ``counts`` is each train tile's :func:`benchmark.geomorphon.class_counts`. Computed once on
    the full train list and used unchanged at every rung. A class with no train pixel would get
    an infinite weight, which is refused rather than clipped.
    """
    classes = _classes()
    total = np.zeros(len(classes), dtype=np.int64)
    for c in counts:
        total += np.asarray(list(c)[: len(classes)], dtype=np.int64)
    if (total == 0).any():
        empty = [classes[i] for i in np.flatnonzero(total == 0)]
        raise ValueError(f"no train pixel is labelled {empty}: its class weight would be infinite")
    inv = 1.0 / np.sqrt(total / total.sum())
    weights = inv / inv.mean()
    return (
        {name: int(n) for name, n in zip(classes, total)},
        {name: float(w) for name, w in zip(classes, weights)},
    )


def render_config(g: Geomorphon, cen: Census, sets: Mapping[str, dict], pixels: dict,
                  weights: dict) -> bytes:
    """``geomorphon_config.json``: what labels are a function of, and what they came out as."""
    body = {
        "label_config_hash": label_config_hash(g),
        "label_config": label_config(g),
        "census": cen.to_dict(g),
        "variants": dict(sets),
        "class_pixel_counts_train": pixels,
        "class_weights": weights,
        "class_weights_rule": (
            "inverse-sqrt frequency over the primary labels' non-ignore pixels on the full "
            "train list, scaled to mean 1 over the ten classes; used unchanged at every rung"
        ),
    }
    return (json.dumps(body, indent=2) + "\n").encode()


# --- 4. The fixed sample (story 18) ---------------------------------------------------------------


def fixed_sample(test_ids: Iterable[str], n: int) -> list[str]:
    """The ``n`` test tiles first in keyed-BLAKE2b order — the same tiles on every box."""
    from dataset.splits import tie_rank

    ids = sorted(test_ids)
    if not ids:
        return []
    rank = tie_rank(np.array(ids))
    return [ids[i] for i in np.argsort(rank)[:n]]


def check_fixed_sample(
    job: TileJob, sample: list[str], committed: Mapping[str, Mapping[str, dict]]
) -> str:
    """Regenerate ``sample`` from the COGs, every test variant, and require byte identity.

    ``committed`` is ``variant -> tile_id -> row``. A disagreement says which side moved: the
    input hash (upstream data) or only the label hash (the pipeline, or the WBT build).
    """
    test_variants = [v for v in gm.variants(job.g) if "test" in v.splits]
    for tile_id in sample:
        _, labels = job.label(tile_id, "test", test_variants)
        for v, label, ih in labels:
            want = committed[v.name].get(tile_id)
            if want is None:
                raise ValueError(f"fixed-sample tile {tile_id} has no committed {v.name} row")
            got = gm.content_hash(label)
            if ih != want["input_hash"]:
                raise ValueError(
                    f"fixed sample: {tile_id} ({v.name}) read a different window — input hash "
                    f"{ih}, committed {want['input_hash']}. The upstream COGs moved under the "
                    f"committed labels"
                )
            if got != want["content_hash"]:
                raise ValueError(
                    f"fixed sample: {tile_id} ({v.name}) labels the same window differently — "
                    f"label hash {got}, committed {want['content_hash']}. The pipeline or the "
                    f"WBT build moved, not the data; labels from the two are not mixed"
                )
    return (f"{'fixed sample':<16} {len(sample)} test tiles x {len(test_variants)} variants "
            f"regenerated from the COGs, byte-identical to the committed fingerprints")


# --- The freeze stage -----------------------------------------------------------------------------


def write(resolution: LandformResolution) -> list[str]:
    """``freeze --task landform``'s geomorphon stage: census, pass, fingerprints, weights, sample.

    Progress is printed as it goes (the pass is ~1.5 h); the returned lines are the summary.
    """
    from .evaluate import TileStore

    cfg = resolution.config
    g, paths = cfg.geomorphon, cfg.paths
    say = _say
    lists = read_lists(paths.task1_artifacts_dir)
    source = gm.RasterioCogSource(paths.cog_frame, paths.cog_root)
    chash = label_config_hash(g)
    say(f"{'geomorphon':<16} label config {chash}: WBT v{g.wbt_version} --forms, search "
        f"{g.search}, threshold {g.threshold}, skip {g.skip}, fdist {g.fdist}")
    cen = census(lists, source.frame(), g)
    treat = Counter(t.split(",")[0] for t in cen.treatment(g).values())
    say(f"{'census':<16} L = {cen.search} over {cen.tiles:,} tiles: {len(cen.absent)} absent "
        f"COG(s) reached ({', '.join(f'{n} {t}' for t, n in treat.items()) or 'none'}); "
        f"{cen.wrapped} tile(s) wrap the antimeridian")
    labeller = make_labeller(cfg)
    ids = [t for split_ids in lists.values() for t in split_ids]
    shards = TileStore(paths.corpus_dir, index=TileStore.index_for(paths.corpus_dir, ids))
    missing = [t for t in ids if t not in shards]
    if missing:
        raise KeyError(f"{len(missing)} listed tiles are not in {paths.corpus_dir}, e.g. "
                       f"{missing[:3]}; the labels' centre check reads each one's shard tile")
    job = TileJob(source=source, store_shards=shards, g=g, labeller=labeller)
    art = Path(paths.artifacts_dir)
    sample = fixed_sample(lists["test"], g.fixed_sample)
    vs = gm.variants(g)

    # Fingerprints already committed: the sample runs first, so a different WBT build or a moved
    # upstream is caught before this pass labels a single tile with it.
    committed_config = art / CONFIG_NAME
    sample_checked = committed_config.is_file()
    if sample_checked:
        committed = {v.name: read_fingerprints(fingerprint_path(art, v.name)) for v in vs}
        say(check_fixed_sample(job, sample, committed))

    store = LabelStore(Path(paths.labels_dir) / chash, _build(labeller))
    labelled, skipped = run_pass(lists, job, store, threads=cfg.labelling.threads,
                                 report_every=cfg.labelling.report_every, say=say)

    sets: dict[str, dict] = {}
    by_variant: dict[str, list[dict]] = {}
    for v in vs:
        records = [store.records[(v.name, t)] for split in v.splits for t in lists[split]]
        by_variant[v.name] = records
        path, md5 = artifacts.write_committed(
            fingerprint_path(art, v.name), lambda r=records: render_fingerprints(r),
            what=f"{v.name} geomorphon label set",
        )
        sets[v.name] = {"tiles": len(records), "set_fingerprint": set_fingerprint(records),
                        "fingerprints_md5": md5}
        say(f"{'labels':<16} {v.name:<9} {len(records):>7,} tiles, set fingerprint "
            f"{sets[v.name]['set_fingerprint']}")
    train = set(lists["train"])
    pixels, weights = class_weights(
        r["counts"] for r in by_variant["primary"] if r["tile_id"] in train
    )
    _, md5 = artifacts.write_committed(
        committed_config, lambda: render_config(g, cen, sets, pixels, weights),
        what="geomorphon label config",
    )
    say(f"{'class weights':<16} " + ", ".join(f"{k} {w:.3f}" for k, w in weights.items()))
    if not sample_checked:
        say(check_fixed_sample(job, sample, {v: {r["tile_id"]: r for r in recs}
                                             for v, recs in by_variant.items()}))
    return [f"{'geomorphon':<16} {committed_config} md5 {md5} ({labelled:,} tiles labelled, "
            f"{skipped:,} resumed from {store.log})"]


def _say(line: str) -> None:
    print(line, flush=True)
