"""§10.3's kNN probe — the cheap proxy for representation quality (#99).

MAE gives few in-flight signals, and the reconstruction loss is the weakest of them: it says
how well the decoder inpaints, not whether the encoder learned terrain. §10.3 asks for the one
other thing that can be measured without training anything — a **frozen encoder, features over
a fixed set of val tiles, and a kNN over them** — and #85 adjudicates the release recipe on it
jointly with the fixed-mask val loss.

Three probes, all supervised for free by columns the manifest already carries:

- **Classification of ``stratum_id``** — the k=20 terrain cells the corpus was stratified into
  (ADR-0004/ADR-0005). A label, not a rank.
- **Regression of ``slope_mean`` and ``geomorphon_entropy``** — two axes of the frozen metric
  vector, so the probe reports both a discrete and a continuous view of the same question.

## The caveat is binding, and it is emitted rather than documented

SSL4EO-S12 measured MAE representations as **weak under linear and kNN probes** relative to
contrastive methods, on benchmarks where MAE fine-tuning ultimately wins. So these numbers
compare **trends** — this checkpoint against the last, this arm against the release recipe —
and are never an absolute quality bar. The ticket asks for output shaped so that is hard to
misread, and three things here do that work:

1. :data:`SSL4EO_CAVEAT` is logged **beside the numbers at every probe point**, not written in
   a docstring. A W&B panel someone opens six months from now carries it.
2. The headline quantity is the **delta against the epoch-0 baseline** — the same probe run on
   the untrained encoder — so the natural thing to read off a chart is movement, not level.
   That baseline is also why this has to run on a randomly initialized model without erroring.
3. Nothing here emits a "pass". There is no threshold, no verdict, and no colour.

## What it costs, measured

One probe point over the shipped 5,000-tile set: **~1.5 minutes on a laptop CPU** (ViT-S/16, 4
loader workers, fp32; 77 s and 92 s on two runs), of which the kNN over those features is a
fraction of a second — the cost is the encoder
pass, and a rented GPU does it in a small fraction of that. Against a training epoch the ratio
is what matters, and it is structural rather than incidental: the probe encodes 5,000 tiles at
197 tokens with no backward pass, while an R1 epoch trains 475,030 tiles at 50 tokens with one,
so the probe is **~1.4% of an epoch's encoder work**, once every 20 epochs — under 0.1% of the
run. §10.3's "minutes, not a meaningful fraction of an epoch" holds with room to spare, and
:attr:`ProbeReport.seconds` re-measures it on every point rather than trusting this paragraph.

## Why the probe set is a committed artifact

Same contract as the selection list and the split assignment (ADR-0005 §5): **the file is what
later runs re-read**, not this code re-run. Every probe number is a comparison, and a probe set
that redrew itself — because a tier's val split grew, because a seed moved — would make each of
those comparisons a measurement of two different tile populations. :func:`draw_probe_set` runs
once per tier; :func:`load_probe_tiles` is what every run afterwards calls.

The draw is stratified over ``stratum_id`` and ordered by :func:`dataset.splits.tie_rank`, the
project's keyed-BLAKE2b tie-break — never a NumPy ``Generator``, whose distribution streams are
not guaranteed across releases.

## Why the kNN excludes a query's own COG

The corpus card §4.6 is blunt: two windows from one source COG share geology, climate, the same
radar pass and the same void-filling history — they are neighbours, not independent samples. A
plain leave-one-out kNN over val tiles would answer most queries with the *overlapping window
next door*, and would report a high accuracy for an encoder holding no terrain knowledge at all,
because the nearest tile in feature space is the one whose pixels nearly are the query's. So the
bank excludes every tile from the query's own COG (:func:`knn_classify`, :func:`knn_regress`).
It is the same blocking unit #77 drew the val split at, applied one level down.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch

from dataset import config as dataset_config

from .artifact import read_tile_strata, write_tile_strata
from .config import PROBE_POOLINGS, RunConfig
from .runtime import eager_module
from .sink import RunLogger

#: What SSL4EO-S12 found, in the form it has to travel in. Logged beside every probe number
#: (:meth:`ProbeReport.scalars`) and printed by :func:`format_probe`, because a caveat that
#: lives only in a ticket is one nobody reading the chart will ever see.
SSL4EO_CAVEAT: Final = (
    "kNN probe numbers compare TRENDS across checkpoints and ablation arms — never an "
    "absolute quality bar. SSL4EO-S12 measured MAE representations as weak under linear and "
    "kNN probes relative to contrastive methods, on benchmarks where MAE fine-tuning wins."
)

#: The two continuous axes of the frozen metric vector §10.3 regresses. Both are manifest
#: columns (ADR-0004), so the supervision costs nothing: no labelling, no second pass.
PROBE_REGRESSION_TARGETS: Final = ("slope_mean", "geomorphon_entropy")

#: The classification label — the k=20 k-means cell (ADR-0005). A label, not a rank, which is
#: why the probe reports accuracy and never an ordinal error.
PROBE_CLASS_LABEL: Final = "stratum_id"

#: §10.3's "~5k val tiles". Big enough that a 20-class accuracy is estimated to about a point
#: and small enough that the pass is minutes; the shipped val split is ~10k, so this is half of
#: it rather than an arbitrary cap.
PROBE_SIZE: Final = 5_000

#: The manifest columns the probe reads. Nothing else, and no pixel — feature extraction reads
#: the shards through ADR-0003's own loader.
PROBE_COLUMNS: Final = (
    "tile_id",
    "split",
    "source_cog",
    PROBE_CLASS_LABEL,
    *PROBE_REGRESSION_TARGETS,
)

#: How many offending tiles an error message names before it stops listing — `pretrain.data`'s
#: number, for the same reason: enough to tell one relabelled tile from a whole stale artifact.
_MAX_NAMED: Final = 5


# --- The committed probe set --------------------------------------------------


@dataclass(frozen=True)
class ProbeSet:
    """The committed artifact: which val tiles, and the stratum each was committed under.

    The stratum rides along rather than being read fresh every time, and that redundancy is
    the point — it is what lets :func:`load_probe_tiles` *catch* a manifest whose labels have
    moved under a committed probe set, instead of silently scoring the arms against two
    different label vectors. The two regression targets are floats and are not committed: a
    CSV round-trip of a float is a formatting decision, and the discrete label is enough to
    detect a re-labelled corpus.
    """

    tile_id: tuple[str, ...]
    stratum_id: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.tile_id) != len(self.stratum_id):
            raise ValueError(
                f"{len(self.tile_id)} tiles carry {len(self.stratum_id)} strata"
            )
        if len(set(self.tile_id)) != len(self.tile_id):
            raise ValueError("a probe set names a tile twice; it is a set, not a sample")

    def __len__(self) -> int:
        return len(self.tile_id)


def _read_columns(manifest_path: Path | str, columns: Sequence[str]) -> dict[str, np.ndarray]:
    """The named manifest columns as plain arrays. No pixel is touched."""
    import pyarrow.parquet as pq

    table = pq.read_table(Path(manifest_path), columns=list(columns))
    return {c: table[c].to_numpy(zero_copy_only=False) for c in columns}


def draw_probe_set(
    manifest_path: Path | str,
    *,
    split: str = "val",
    size: int = PROBE_SIZE,
    seed: int = dataset_config.RANDOM_SEED,
) -> ProbeSet:
    """Draw §10.3's probe set: ``size`` tiles of ``split``, stratified over ``stratum_id``.

    Run **once per tier**, and its output committed. Two properties, and each is one of the
    ways a probe number stops being comparable:

    - **Stratified proportionally**, through :func:`dataset.sampling.band_quotas` — the
      project's one largest-remainder apportionment, reused rather than rewritten. A probe set
      that over-weighted a stratum would report an accuracy about a population the corpus does
      not have; capped per stratum, so a split short on a stratum yields fewer rather than
      raising.
    - **Ordered by keyed hash**, through :func:`dataset.splits.tie_rank`. The house rule
      (`dataset.sampling`): NumPy guarantees its bit-generator streams across releases but not
      the distributions layered on them, so every tie-break in this project is a BLAKE2b of the
      block's name. Here the "block" is a tile id.

    Asked for more than the split holds, it returns the split. The shipped 500k tier has ~10k
    val tiles against a 5k budget, but a debug tier has fewer, and a draw that raised would
    make the probe unusable exactly where it is cheapest to exercise.
    """
    from dataset.sampling import band_quotas
    from dataset.splits import tie_rank

    cols = _read_columns(manifest_path, ("tile_id", "split", PROBE_CLASS_LABEL))
    rows = np.flatnonzero(cols["split"] == split)
    if rows.size == 0:
        raise ValueError(
            f"no tile carries split {split!r}; the corpus holds "
            f"{sorted(set(map(str, cols['split'].tolist())))}"
        )
    tile_id = cols["tile_id"][rows].astype(str)
    stratum = cols[PROBE_CLASS_LABEL][rows].astype(np.int64)

    # Keys are the stratum ids as sortable strings: `band_quotas` breaks its own ties on the
    # key name, so zero-padding is what keeps the apportionment independent of `str` ordering.
    caps = {f"{s:03d}": int((stratum == s).sum()) for s in np.unique(stratum)}
    quotas = band_quotas(caps, size, allocation="proportional")

    rank = tie_rank(tile_id, seed)
    chosen: list[int] = []
    for key, quota in quotas.items():
        within = np.flatnonzero(stratum == int(key))
        chosen.extend(within[np.argsort(rank[within], kind="stable")][:quota].tolist())

    order = np.argsort(tile_id[chosen], kind="stable")
    picked = np.asarray(chosen)[order]
    return ProbeSet(
        tile_id=tuple(tile_id[picked].tolist()),
        stratum_id=tuple(int(s) for s in stratum[picked]),
    )


def write_probe_set(probe_set: ProbeSet, path: Path | str) -> Path:
    """Write the committed CSV, sorted by ``tile_id`` and LF-terminated.

    Sorted for the same reason :func:`dataset.splits.write_assignment` sorts: two runs that
    agree on the set produce a byte-identical file whatever order the rows arrived in, which
    is what makes its md5 a contract rather than a timestamp. Sorted by ``tile_id`` alone,
    where §10.2's panel sorts by ``(stratum_id, tile_id)``: the panel has a render order to
    preserve and a probe set has none, so the plainest thing to diff wins.

    The format is :mod:`pretrain.artifact`'s, shared with that panel — the two committed
    artifacts are the same two columns in two roles.
    """
    return write_tile_strata(sorted(zip(probe_set.tile_id, probe_set.stratum_id)), path)


def read_probe_set(path: Path | str) -> ProbeSet:
    """Read the committed CSV back. Neither column ever contains a comma or a quote."""
    rows = read_tile_strata(path, what="probe set")
    return ProbeSet(
        tile_id=tuple(t for t, _ in rows), stratum_id=tuple(s for _, s in rows)
    )


# --- The probe set, located in a corpus ---------------------------------------


@dataclass(frozen=True)
class ProbeTiles:
    """A committed probe set resolved against a corpus: rows, labels, targets, groups.

    :attr:`indices` addresses the **manifest**, in the committed file's own order, which is
    what lets the feature pass be an ordinary ``Subset`` of ADR-0003's read path rather than a
    second reader. Everything else here is a column of that same manifest, gathered once.
    """

    #: Manifest row of each probe tile, in committed order.
    indices: np.ndarray
    tile_id: np.ndarray
    #: The classification label — checked against the committed file, not merely read.
    stratum_id: np.ndarray
    #: ``{name: values}`` for :data:`PROBE_REGRESSION_TARGETS`.
    targets: dict[str, np.ndarray]
    #: The kNN's exclusion group. See the module docstring on why it is the source COG.
    source_cog: np.ndarray
    #: md5 of the committed file these came from, and the reason it is carried: it is logged
    #: with every probe point (``probe/set_md5``), so a chart comparing two runs' trends can be
    #: checked to be comparing them over the *same tiles*. Two 5,000-tile probe sets drawn at
    #: different times agree on every other number this report carries.
    md5: str
    #: Where it came from, for a message.
    path: str

    def __len__(self) -> int:
        return len(self.indices)

    @property
    def n_cogs(self) -> int:
        return len(set(self.source_cog.tolist()))


def load_probe_tiles(
    corpus_dir: Path | str,
    path: Path | str = dataset_config.PROBE_SET_PATH,
    *,
    split: str = "val",
) -> ProbeTiles:
    """Resolve the committed probe set against a corpus, refusing three disagreements.

    A committed artifact is only a contract if it is *checked* against what it names, and
    these are the three ways it can stop describing this corpus:

    - **A tile the corpus does not hold** — a stale or cross-tier probe set. Same refusal, and
      the same reason, as :func:`pretrain.data.load_blocks` makes of a block list naming
      geocells this corpus does not contain.
    - **A stratum that disagrees with the manifest.** The arms are compared under the committed
      label; a manifest re-labelled beneath it would make every cross-arm comparison a
      comparison of two label vectors, and nothing in the numbers would show it.
    - **A tile outside ``split``.** The probe is a measurement on held-out data. A probe set
      that had drifted onto train tiles would report the encoder's memory as its representation
      quality — the one failure mode that flatters exactly the thing being measured.
    """
    corpus_dir = Path(corpus_dir)
    committed = read_probe_set(path)
    cols = _read_columns(corpus_dir / "manifest.parquet", PROBE_COLUMNS)
    row_of = {t: i for i, t in enumerate(cols["tile_id"].astype(str).tolist())}

    absent = [t for t in committed.tile_id if t not in row_of]
    if absent:
        raise ValueError(
            f"{path} names {len(absent):,} tile(s) of {len(committed):,} that {corpus_dir} "
            f"does not hold — a stale or cross-tier probe set. First: {absent[:_MAX_NAMED]}"
        )

    indices = np.array([row_of[t] for t in committed.tile_id], dtype=np.int64)
    manifest_stratum = cols[PROBE_CLASS_LABEL][indices].astype(np.int64)
    committed_stratum = np.array(committed.stratum_id, dtype=np.int64)
    moved = np.flatnonzero(manifest_stratum != committed_stratum)
    if moved.size:
        named = [
            f"{committed.tile_id[i]}: committed {committed_stratum[i]} vs manifest "
            f"{manifest_stratum[i]}"
            for i in moved[:_MAX_NAMED]
        ]
        raise ValueError(
            f"{path} disagrees with {corpus_dir}/manifest.parquet on the stratum of "
            f"{moved.size:,} of {len(committed):,} tiles — the committed label is what every "
            f"arm is compared under. First: " + "; ".join(named)
        )

    labels = cols["split"][indices].astype(str)
    outside = np.flatnonzero(labels != split)
    if outside.size:
        named = [f"{committed.tile_id[i]} ({labels[i]})" for i in outside[:_MAX_NAMED]]
        raise ValueError(
            f"{path} names {outside.size:,} tile(s) outside the {split!r} split — the probe "
            f"measures held-out representation quality, and a probe tile the encoder trained "
            f"on reports its memory instead. First: " + "; ".join(named)
        )

    from dataset.materialize import file_md5

    return ProbeTiles(
        indices=indices,
        tile_id=cols["tile_id"][indices].astype(str),
        stratum_id=manifest_stratum,
        targets={t: cols[t][indices].astype(np.float64) for t in PROBE_REGRESSION_TARGETS},
        source_cog=cols["source_cog"][indices].astype(str),
        # The project's one implementation of "hash a committed artifact" — the corpus card,
        # the ledger and the split all pin their inputs with it (`pretrain.fingerprint`).
        md5=file_md5(Path(path)),
        path=str(path),
    )


# --- The frozen feature pass --------------------------------------------------


def build_probe_loader(cfg: RunConfig, tiles: ProbeTiles):
    """ADR-0003's read path over the probe tiles, in committed order and unaugmented.

    Three deliberate differences from :func:`pretrain.train.build_loader`, and each of them is
    a way the probe would otherwise stop being a measurement:

    - **No augmentation.** A fresh dihedral draw per read would make a tile's feature move
      between probe points for reasons that have nothing to do with the encoder. §10.1's val
      loader is built the same way, for the same reason.
    - **No shuffle, no dropped tail.** The row order *is* the contract between a feature and
      its label, and every probe tile is scored exactly once.
    - **Through** :class:`~pretrain.data.SplitDataset`. The probe could address the manifest
      directly — it holds the row indices — but routing through the split keeps plan §15.2's
      containment assertion on this path too. :func:`load_probe_tiles` already refuses a tile
      outside the split; this is the same refusal from the other side, and the one that would
      still catch a probe set committed against a *differently split* corpus.

    And one thing that is **not** a difference and has to be here anyway: the loader carries an
    **explicit generator**. A ``DataLoader`` given none draws its per-iterator base seed from
    the ambient torch RNG — once per pass, even unshuffled with no workers — which is the same
    stream #90's masking draws from. Without this, taking a probe point would shift every mask
    the run drew afterwards, so a run's trajectory would depend on whether anyone was measuring
    it. §10.1's val pass avoids the same trap by passing its own generator, and
    `test_the_probe_does_not_perturb_the_run_it_measures` is what holds it here.
    """
    from torch.utils.data import Subset

    from dataset.loader import build_dataloader

    from .data import SplitDataset
    from .train import seeded_generator

    split = SplitDataset(
        cfg.data.corpus_dir,
        cfg.data.val_split,
        blocks_path=cfg.data.split_blocks_path,
        assignment_path=cfg.data.split_assignment_path,
        normalization=cfg.data.normalization,
    )
    position = {row: i for i, row in enumerate(split.indices)}
    outside = [t for t, row in zip(tiles.tile_id, tiles.indices) if row not in position]
    if outside:
        raise ValueError(
            f"{len(outside):,} probe tile(s) are not in the {cfg.data.val_split!r} split of "
            f"{cfg.data.corpus_dir}. First: {outside[:_MAX_NAMED]}"
        )
    return build_dataloader(
        Subset(split, [position[r] for r in tiles.indices]),
        batch_size=cfg.optimization.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=False,
        drop_last=False,
        generator=seeded_generator(cfg),
    )


def _fingerprint(params: Sequence[torch.nn.Parameter]) -> tuple:
    """A cheap fingerprint of every parameter's value and gradient.

    Sum *and* sum-of-squares, in fp64, because a plain sum is blind to the change that matters
    most here: an optimizer step moves parameters in both directions and can leave the total
    almost exactly where it was. Two reductions over the encoder cost microseconds against a
    pass that costs minutes.
    """
    def summarize(t: torch.Tensor) -> tuple[float, float]:
        return float(t.sum(dtype=torch.float64)), float(t.square().sum(dtype=torch.float64))

    return tuple(
        (summarize(p.detach()), None if p.grad is None else summarize(p.grad))
        for p in params
    )


@contextmanager
def frozen(model: torch.nn.Module) -> Iterator[torch.nn.Module]:
    """§10.3's frozen encoder, as a checked guarantee rather than a convention.

    "Frozen" is three claims, and each is independently breakable by a plausible edit:

    - **No gradient reaches the encoder.** The body runs under :func:`torch.no_grad`, and every
      parameter's ``.grad`` is fingerprinted before and after. A probe that accumulated into
      the gradients a training step is midway through building would corrupt that step — and
      the loss curve would show a blip nobody could trace back to the epoch's monitoring.
    - **No parameter moves.** Fingerprinted the same way. Nothing here would move one today;
      the check is for the edit that adds a normalization layer with running statistics, or a
      caller that hands in a model with something fitted inside it.
    - **The mode the caller had is the mode it gets back.** ``eval()`` is set for the pass and
      restored after. A probe that left a model in eval would silently disable §6's dropout and
      drop-path for every remaining epoch, which trains a different recipe than the one recorded.

    The checks run on the way out and **raise**, because the acceptance criterion is that the
    freeze is *asserted*. They are skipped when the body itself raised: the caller's exception
    is the more informative one, and a guard that replaced it would be a guard that hides bugs.
    """
    params = list(model.parameters())
    was_training = model.training
    before = _fingerprint(params)
    model.eval()
    try:
        with torch.no_grad():
            yield model
    finally:
        model.train(was_training)
    if _fingerprint(params) != before:
        raise RuntimeError(
            "the encoder was not frozen: a parameter or its gradient moved during the probe. "
            "§10.3 measures a checkpoint, and a probe that trains is measuring itself"
        )


def extract_features(
    model: torch.nn.Module,
    loader,
    *,
    pooling: str = "mean",
    device: torch.device | str | None = None,
) -> np.ndarray:
    """One frozen inference pass: ``(n_tiles, encoder_dim)``, in the loader's order.

    Goes through :meth:`MaskedAutoencoder.encode
    <pretrain.model.MaskedAutoencoder.encode>` — every token, nothing hidden — which is the one
    model path with **no RNG in it at all**, and therefore the only one a measurement can be
    built on. The masked path would give a different feature every call.

    Run in fp32 rather than under §7.1's bf16 autocast. The probe is a measurement taken once
    every 20 epochs and its cost is already minutes; trading precision for speed here would buy
    nothing and would make two arms' features differ by their autocast policy.

    Returns plain float32 NumPy, not a tensor: everything downstream of this is kNN arithmetic
    over a matrix that fits in RAM, and handing back a tensor would leave a device and a
    ``requires_grad`` flag for each of those call sites to get right.
    """
    if pooling not in PROBE_POOLINGS:
        raise ValueError(f"unknown pooling {pooling!r} (one of {PROBE_POOLINGS})")
    module = eager_module(model)
    if device is None:
        device = next(module.parameters()).device
    device = torch.device(device)

    rows: list[np.ndarray] = []
    with frozen(module):
        for batch in loader:
            encoded = module.encode(batch["image"].to(device, non_blocking=True))
            pooled = encoded[:, 0, :] if pooling == "cls" else encoded[:, 1:, :].mean(dim=1)
            rows.append(pooled.to(torch.float32).cpu().numpy())
    return np.concatenate(rows, axis=0) if rows else np.zeros((0, 0), dtype=np.float32)


# --- The estimators -----------------------------------------------------------

#: Neighbours each query votes over. 20 is DINO's own kNN-monitor value and is small enough
#: that a 20-class vote is not dominated by the largest stratum; the number matters far less
#: than its being fixed, since every probe number is read as a trend.
DEFAULT_K: Final = 20

#: Query rows per distance block. The full 5k x 5k similarity matrix is 100 MB in fp32, which
#: would fit — but the probe runs on the training box mid-run, beside a model and its optimizer
#: state, and a monitoring pass that competes for memory with the run it is monitoring is a
#: monitoring pass that eventually kills one.
_CHUNK: Final = 512


@dataclass(frozen=True)
class KnnClassification:
    """§10.3's ``stratum_id`` probe: what a kNN recovers, and what guessing would have."""

    k: int
    accuracy: float
    #: Mean per-class recall. Reported beside plain accuracy because the strata are not equal
    #: sized — the shipped probe set runs 69 to 372 tiles — so accuracy alone can rise by
    #: getting better at the big cells and worse at the rest, which is the opposite of what
    #: §10.3 is asking about.
    balanced_accuracy: float
    #: What the same leave-one-COG-out setup scores by always answering the bank's most common
    #: class. The number that makes the accuracy readable at all.
    baseline_accuracy: float
    #: ``(accuracy - baseline) / (1 - baseline)``: 0 at no skill, 1 at perfect. Not a quality
    #: bar — see :data:`SSL4EO_CAVEAT` — but it is the scale on which "the trend moved" means
    #: something, since raw accuracy moves with the class balance of whatever set is probed.
    skill: float
    n_scored: int
    n_classes: int


@dataclass(frozen=True)
class KnnRegression:
    """§10.3's ``slope_mean`` / ``geomorphon_entropy`` probe: error, and error against what."""

    k: int
    mae: float
    rmse: float
    #: The leave-one-COG-out **constant** predictor's MAE — every query answered with the mean
    #: of its eligible bank. Reported because "MAE 3.4" on `slope_mean` says nothing on its own:
    #: the two targets are in different units and the val split's spread differs per tier.
    baseline_mae: float
    #: ``1 - mae / baseline_mae``. 0 at no skill, 1 at perfect, negative below the constant.
    skill: float
    n_scored: int


def _group_codes(groups: Sequence | np.ndarray) -> np.ndarray:
    """Group labels as dense integer codes, order-independent."""
    return np.unique(np.asarray(groups), return_inverse=True)[1]


def _unit(features: np.ndarray) -> np.ndarray:
    """L2-normalized rows, so a dot product is a cosine similarity.

    Cosine rather than Euclidean because a ViT feature's *norm* carries how confidently the
    encoder responded rather than what it responded to, and it drifts over training — an
    unnormalized kNN would report that drift as a change in representation quality.

    A zero row (possible on a randomly initialized encoder at the epoch-0 baseline) is left at
    zero rather than divided: it then sits at similarity 0 to everything, which is the honest
    answer, instead of producing a NaN that would silently take a query out of every count.
    """
    x = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norms > 0, norms, 1.0)


def assert_bank_holds(groups: Sequence | np.ndarray, k: int) -> None:
    """Refuse a ``k`` the exclusion cannot supply — **before** a run spends anything.

    Called at :class:`KnnProbe` construction rather than only from :func:`neighbour_bank`, and
    that timing is the point. Whether a bank is big enough is fixed by the committed probe set
    and the configured ``k``, both known before the first optimizer step; letting it surface at
    epoch 0 instead would take a 40-hour rented run down inside its own monitoring, for a
    condition that was decidable at startup. Same shape as
    :func:`pretrain.train.assert_a_step_fits`.
    """
    codes = _group_codes(groups)
    if k < 1:
        raise ValueError(f"k {k} must be >= 1")
    eligible = len(codes) - np.bincount(codes)[codes]
    if eligible.min() < k:
        raise ValueError(
            f"k = {k} exceeds the smallest eligible bank ({int(eligible.min())} tiles): a "
            f"query's own source COG is excluded (corpus card §4.6), and one COG contributes "
            f"{int(np.bincount(codes).max())} of the {len(codes):,} probe tiles. Lower "
            f"monitoring.probe_k, or draw a probe set spread over more COGs — a silent clamp "
            f"would let two arms run different k off one config"
        )


@dataclass(frozen=True)
class NeighbourBank:
    """The k nearest non-own-COG tiles of every probe tile, and how near they were.

    Built **once** per probe point and read by all three estimators, because it is the
    expensive part — an ``O(n^2 d)`` matmul — and the classification and the two regressions
    ask the same question of the same features. Computing it per estimator would triple the
    only part of the probe that is not the encoder pass, and would leave three chances for the
    three of them to disagree about which neighbours they scored.

    :attr:`similarity` is carried rather than recomputed for the same reason, and at a cost
    that is nothing: ``(5000, 20)`` floats is 400 KB, against the 146 MB a gather of the same
    neighbours' *feature vectors* would have taken — which would have undone the whole reason
    the matmul is blocked in the first place.
    """

    #: ``(n, k)`` bank rows, nearest first.
    indices: np.ndarray
    #: ``(n, k)`` cosine similarity of each of those, descending along the row.
    similarity: np.ndarray
    #: Dense group code per query — the source COG, as :func:`_group_codes` numbers them.
    codes: np.ndarray

    @property
    def k(self) -> int:
        return self.indices.shape[1]

    def __len__(self) -> int:
        return len(self.indices)


def neighbour_bank(
    features: np.ndarray, *, groups: Sequence | np.ndarray, k: int = DEFAULT_K
) -> NeighbourBank:
    """The k nearest tiles of every query, with the query's own source COG excluded.

    Blocked over queries (see :data:`_CHUNK`) and selected with :func:`numpy.argpartition`, so
    the cost is ``O(n^2 d)`` in the matmul and linear in the selection rather than ``O(n^2 log
    n)`` in a sort of every row. At 5,000 x 384 that is a fraction of a second — the probe's
    cost is the encoder pass, not this.
    """
    codes = _group_codes(groups)
    assert_bank_holds(codes, k)
    n = len(codes)

    unit = _unit(features)
    indices = np.empty((n, k), dtype=np.int64)
    similarity = np.empty((n, k), dtype=np.float32)
    for start in range(0, n, _CHUNK):
        stop = min(start + _CHUNK, n)
        block = unit[start:stop] @ unit.T
        # -inf rather than a mask array: it survives argpartition without a second gather, and
        # an excluded neighbour can then never be selected however tied the real ones are.
        block[codes[start:stop][:, None] == codes[None, :]] = -np.inf
        top = np.argpartition(-block, k - 1, axis=1)[:, :k]
        # argpartition leaves the k unordered; sort them so "nearest" means the same thing to
        # the vote's tie-break and to the regression's mean on any NumPy build.
        rows = np.arange(stop - start)[:, None]
        order = np.argsort(-block[rows, top], axis=1)
        indices[start:stop] = np.take_along_axis(top, order, axis=1)
        similarity[start:stop] = np.take_along_axis(block[rows, top], order, axis=1)
    return NeighbourBank(indices=indices, similarity=similarity, codes=codes)


def knn_classify(bank: NeighbourBank, labels: np.ndarray) -> KnnClassification:
    """§10.3's classification probe: majority vote over the k nearest non-own-COG tiles.

    A plain majority vote, with two tie-breaks that only ever decide a coin flip: the class
    whose **nearest** neighbour is nearest, then the lower class id. Similarity-*weighted*
    voting was the alternative and was not taken — cosine similarity goes negative, so a
    weighted vote would have neighbours arguing *against* their own class, which is a strange
    thing to explain in a number #85 adjudicates on. The tie-break avoids that too: it reads
    the neighbour list's own ordering (position 0 is nearest) rather than summing signed
    similarities, so no quantity here is ever made worse by a distant neighbour.

    The baseline is computed under the identical exclusion: each query answered with the most
    common class in *its own* eligible bank. A global majority rate would be a slightly
    different question, and the whole point of reporting a baseline is that the accuracy above
    it is the only part that means anything.
    """
    labels = np.asarray(labels)
    classes, encoded = np.unique(labels, return_inverse=True)
    neighbours, codes, k = bank.indices, bank.codes, bank.k
    n, n_classes = len(encoded), len(classes)
    if len(neighbours) != n:
        raise ValueError(f"{len(neighbours)} probe tiles carry {n} labels")

    rows = np.repeat(np.arange(n), k)
    voted = encoded[neighbours].ravel()
    votes = np.zeros((n, n_classes), dtype=np.int64)
    np.add.at(votes, (rows, voted), 1)
    # Rank of each class's *nearest* representative: the smallest position it holds in the
    # neighbour list, or k when it holds none. `np.minimum.at` keeps the smallest.
    nearest = np.full((n, n_classes), k, dtype=np.int64)
    np.minimum.at(nearest, (rows, voted), np.tile(np.arange(k), n))
    # lexsort ranks by the last key first: most votes, then nearest representative (negated so
    # "smaller is better" sorts last), then the lowest class id (same trick).
    keys = (-np.arange(n_classes)[None, :].repeat(n, 0), -nearest, votes)
    predicted = np.lexsort(keys)[:, -1]

    correct = predicted == encoded
    accuracy = float(correct.mean())
    recalls = [
        float(correct[encoded == c].mean()) for c in range(n_classes) if (encoded == c).any()
    ]

    # The no-skill answer, under the same exclusion: class counts over the whole set, minus the
    # query's own group's contribution. Vectorized as a (groups x classes) table.
    per_group = np.zeros((codes.max() + 1, n_classes), dtype=np.int64)
    np.add.at(per_group, (codes, encoded), 1)
    bank_counts = per_group.sum(axis=0)[None, :] - per_group
    baseline = float((bank_counts[codes].argmax(axis=1) == encoded).mean())

    return KnnClassification(
        k=k,
        accuracy=accuracy,
        balanced_accuracy=float(np.mean(recalls)) if recalls else float("nan"),
        baseline_accuracy=baseline,
        skill=(accuracy - baseline) / (1.0 - baseline) if baseline < 1.0 else float("nan"),
        n_scored=n,
        n_classes=n_classes,
    )


def knn_regress(bank: NeighbourBank, values: np.ndarray) -> KnnRegression:
    """§10.3's regression probe: the unweighted mean of the k nearest non-own-COG targets.

    Unweighted rather than similarity- or distance-weighted, and for the same reason the vote
    above is: a weighting scheme is a second thing that can differ between two arms' numbers,
    and there is no evidence it would move the trend these numbers exist to report.
    """
    values = np.asarray(values, dtype=np.float64)
    codes = bank.codes
    if len(bank) != len(values):
        raise ValueError(f"{len(bank)} probe tiles carry {len(values)} target values")
    predicted = values[bank.indices].mean(axis=1)
    error = predicted - values

    # The constant predictor, under the same exclusion: each query answered with the mean of
    # everything outside its own COG.
    per_group_sum = np.bincount(codes, weights=values)
    per_group_n = np.bincount(codes)
    bank_mean = (values.sum() - per_group_sum[codes]) / (len(values) - per_group_n[codes])
    baseline_mae = float(np.abs(values - bank_mean).mean())

    mae = float(np.abs(error).mean())
    return KnnRegression(
        k=bank.k,
        mae=mae,
        rmse=float(np.sqrt(np.square(error).mean())),
        baseline_mae=baseline_mae,
        skill=1.0 - mae / baseline_mae if baseline_mae > 0 else float("nan"),
        n_scored=len(values),
    )


# --- The probe point ----------------------------------------------------------

#: The prefix every probe key sits under, so a W&B page groups them into one panel — beside
#: #97's ``train/``, ``val/`` and ``throughput/``, and #98's grids.
PREFIX: Final = "probe"


@dataclass(frozen=True)
class ProbeReport:
    """One probe point: three estimators, what they were run over, and what they cost."""

    epoch: int
    pooling: str
    k: int
    n_tiles: int
    n_cogs: int
    #: md5 of the committed probe set these numbers were scored over. The one field that makes
    #: "comparable across runs and arms" checkable rather than asserted: two probe sets of the
    #: same size over the same corpus agree on every other field here.
    set_md5: str
    #: Wall-clock of the whole point — the encoder pass plus the kNN. §10.3's cost has to stay
    #: in minutes rather than a meaningful fraction of an epoch, and the only way that stays
    #: true is by being reported every time rather than measured once in a ticket.
    #:
    #: Measured and reported, **not alarmed on**. §10.3 sets no threshold, the measured cost is
    #: ~1.4% of an epoch (see the module docstring) with two orders of magnitude of headroom,
    #: and inventing a limit here would put a number nobody argued in front of a run that could
    #: then be stopped by it. A regression shows up as this series moving.
    seconds: float
    classification: KnnClassification
    regression: dict[str, KnnRegression]

    def headline(self) -> dict[str, float]:
        """The numbers a trend is read off. Keys are the suffixes of the logged names."""
        values = {
            "stratum_acc": self.classification.accuracy,
            "stratum_skill": self.classification.skill,
        }
        for name, result in self.regression.items():
            values[f"{name}_mae"] = result.mae
            values[f"{name}_skill"] = result.skill
        return values

    def scalars(self, baseline: ProbeReport | None = None) -> dict[str, Any]:
        """The whole point as one sink payload, under keys that do not move between runs.

        ``baseline`` is the epoch-0 report. Given one, every headline number also emits a
        ``…_vs_epoch0`` series, which is the form §10.3's numbers are meant to be read in: the
        baseline's own point is exactly 0, so a chart of the delta starts at the origin and
        shows movement rather than level. Where the two disagree about what was measured —
        a different pooling, a different k, a different probe set — the delta is **not** emitted
        at all, because subtracting two incomparable numbers produces a plausible one.
        """
        classification = self.classification
        values: dict[str, Any] = {
            f"{PREFIX}/stratum_acc": classification.accuracy,
            f"{PREFIX}/stratum_balanced_acc": classification.balanced_accuracy,
            f"{PREFIX}/stratum_baseline_acc": classification.baseline_accuracy,
            f"{PREFIX}/stratum_skill": classification.skill,
            f"{PREFIX}/stratum_classes": classification.n_classes,
            f"{PREFIX}/seconds": self.seconds,
            f"{PREFIX}/tiles": self.n_tiles,
            f"{PREFIX}/cogs": self.n_cogs,
            f"{PREFIX}/k": self.k,
            f"{PREFIX}/pooling": self.pooling,
            f"{PREFIX}/set_md5": self.set_md5,
            # Beside the numbers, at every point. See the module docstring.
            f"{PREFIX}/caveat": SSL4EO_CAVEAT,
        }
        for name, result in self.regression.items():
            values[f"{PREFIX}/{name}_mae"] = result.mae
            values[f"{PREFIX}/{name}_rmse"] = result.rmse
            values[f"{PREFIX}/{name}_baseline_mae"] = result.baseline_mae
            values[f"{PREFIX}/{name}_skill"] = result.skill

        if baseline is not None and self.comparable_to(baseline):
            mine, theirs = self.headline(), baseline.headline()
            values.update(
                {
                    f"{PREFIX}/{key}_vs_epoch0": mine[key] - theirs[key]
                    for key in mine
                    if key in theirs
                }
            )
        return values

    def comparable_to(self, other: ProbeReport) -> bool:
        """Whether a difference between these two points is about the encoder and nothing else.

        The probe set is compared by **md5**, not by tile count: two 5,000-tile sets drawn at
        different times agree on every count this report carries and are not the same
        measurement. That is the whole reason :attr:`set_md5` exists.
        """
        return (
            self.pooling == other.pooling
            and self.k == other.k
            and self.set_md5 == other.set_md5
            and set(self.regression) == set(other.regression)
        )


def format_probe(report: ProbeReport) -> str:
    """One probe point, as console lines — the numbers, then the caveat under them.

    The caveat is on the printed form and not only in the sink because the person most likely
    to over-read a probe number is the one watching the run it came out of.
    """
    regressions = "  ".join(
        f"{name} mae {r.mae:.3f} (skill {r.skill:+.3f})"
        for name, r in report.regression.items()
    )
    return (
        f"probe epoch {report.epoch:4d}  stratum acc {report.classification.accuracy:.3f} "
        f"(chance {report.classification.baseline_accuracy:.3f}, "
        f"skill {report.classification.skill:+.3f})   {regressions}\n"
        f"                   {report.n_tiles:,} tiles / {report.n_cogs:,} COGs, k={report.k}, "
        f"{report.pooling}-pooled, {report.seconds:.1f}s\n"
        f"                   {SSL4EO_CAVEAT}"
    )


class KnnProbe:
    """§10.3's probe, on its cadence, writing into #97's sink.

    Driven rather than driving, like :class:`~pretrain.monitor.EpochMonitor`: something else
    owns the epoch loop and calls :meth:`maybe_run` at each boundary. That is what lets the
    epoch loop, #94's resumed loop and a one-shot CLI point the same object at a model.

    It holds the loader across probe points rather than rebuilding it, because building one
    re-reads the manifest and re-runs plan §15.2's containment check over the whole split —
    seconds, twenty times over a 400-epoch run, for an answer that cannot have changed.

    **Constructing one validates ``k`` against the probe set** (:func:`assert_bank_holds`). That
    is a startup-time refusal for a startup-time fact, and the alternative is the shape a
    monitoring harness must never have: a run that trains for an epoch and is then killed by its
    own diagnostics over a config value that was wrong before the first step.
    """

    def __init__(
        self,
        cfg: RunConfig,
        tiles: ProbeTiles,
        logger: RunLogger,
        *,
        every: int | None = None,
        k: int | None = None,
        pooling: str | None = None,
    ) -> None:
        monitoring = cfg.monitoring
        self.cfg = cfg
        self.tiles = tiles
        self.logger = logger
        self.every = monitoring.probe_every_epochs if every is None else every
        self.k = monitoring.probe_k if k is None else k
        self.pooling = monitoring.probe_pooling if pooling is None else pooling
        if self.pooling not in PROBE_POOLINGS:
            raise ValueError(f"unknown pooling {self.pooling!r} (one of {PROBE_POOLINGS})")
        assert_bank_holds(tiles.source_cog, self.k)
        self._loader = None
        #: Every point taken, in order. The first is the epoch-0 baseline.
        self.reports: list[ProbeReport] = []

    @classmethod
    def from_config(cls, cfg: RunConfig, logger: RunLogger, **kwargs) -> KnnProbe:
        """Load the committed probe set off the config's paths and build the probe."""
        return cls(
            cfg,
            load_probe_tiles(
                cfg.data.corpus_dir, cfg.data.probe_set_path, split=cfg.data.val_split
            ),
            logger,
            **kwargs,
        )

    @property
    def loader(self):
        if self._loader is None:
            self._loader = build_probe_loader(self.cfg, self.tiles)
        return self._loader

    @property
    def baseline(self) -> ProbeReport | None:
        """The epoch-0 point every later one is reported as a delta against."""
        return self.reports[0] if self.reports else None

    def due(self, epoch: int) -> bool:
        """§10.3's cadence. ``every = 0`` turns the probe off; epoch 0 is the baseline."""
        return self.every > 0 and epoch % self.every == 0

    def run(self, model, epoch: int) -> ProbeReport:
        """Take a probe point unconditionally, log it, and return it."""
        import time

        started = time.perf_counter()
        features = extract_features(model, self.loader, pooling=self.pooling)
        # One bank, read by all three estimators: they ask the same question of the same
        # features, and the search is the only part of a probe point that is not the encoder
        # pass. Three banks would also be three chances to disagree about which neighbours the
        # classification and the regressions were scored over.
        bank = neighbour_bank(features, groups=self.tiles.source_cog, k=self.k)
        report = ProbeReport(
            epoch=epoch,
            pooling=self.pooling,
            k=self.k,
            n_tiles=len(self.tiles),
            n_cogs=self.tiles.n_cogs,
            set_md5=self.tiles.md5,
            classification=knn_classify(bank, self.tiles.stratum_id),
            regression={
                name: knn_regress(bank, values) for name, values in self.tiles.targets.items()
            },
            seconds=time.perf_counter() - started,
        )
        # The baseline point is its own baseline, so the ``…_vs_epoch0`` series starts at
        # exactly 0 rather than starting one point late — a chart missing its origin is a
        # chart whose first visible movement is unattributable.
        self.logger.log(report.scalars(self.baseline or report), step=epoch)
        self.reports.append(report)
        return report

    def maybe_run(self, model, epoch: int) -> ProbeReport | None:
        """Take a point iff ``epoch`` is on the cadence. What an epoch loop calls."""
        return self.run(model, epoch) if self.due(epoch) else None


def probe_for(cfg: RunConfig, logger) -> KnnProbe | None:
    """The run's probe, or ``None`` with the reason said out loud.

    Two ways to get ``None``, and they are not the same thing:

    - **The cadence is 0** — the probe is off on purpose (a unit test, R0's smoke run). Silent.
    - **No probe set is committed at** :attr:`Data.probe_set_path
      <pretrain.config.Data>` — a tier that has never had one drawn. That warns, because it is
      §7's failure shape applied to monitoring: the run trains perfectly, reports every other
      number, and simply has no probe trends at the end of forty hours. `python -m
      pretrain.probe --draw` is the fix, and it takes a second.

    A probe set that *is* there and disagrees with the corpus does not return ``None``. It
    raises, out of :func:`load_probe_tiles`, because that is a wrong measurement rather than an
    absent one.
    """
    import warnings

    if cfg.monitoring.probe_every_epochs <= 0:
        return None
    if not Path(cfg.data.probe_set_path).exists():
        warnings.warn(
            f"§10.3's kNN probe is configured every {cfg.monitoring.probe_every_epochs} "
            f"epochs, but no committed probe set exists at {cfg.data.probe_set_path}. This run "
            f"will report no probe trends — draw one with `python -m pretrain.probe --draw "
            f"--corpus {cfg.data.corpus_dir}`",
            stacklevel=2,
        )
        return None
    return KnnProbe.from_config(cfg, logger)


# --- The entrypoint -------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m pretrain.probe`` — draw the committed set, or take one probe point.

    Two jobs, because they are the two halves of §10.3 that happen outside a training run:

    - ``--draw`` runs the tier's **one** stratified draw and commits it. Run once, then the
      file is what every run re-reads.
    - Without it, the command takes a **single probe point** and prints it: the encoder from
      ``--checkpoint``, or an untrained one, which is the epoch-0 baseline every later point is
      a delta against. That is also how §10.3's cost is measured on real hardware before anyone
      wires it into a 400-epoch run.
    """
    import argparse

    from .header import add_config_arguments, config_from_args
    from .model import MaskedAutoencoder
    from .sink import open_run_logger

    p = add_config_arguments(
        argparse.ArgumentParser(
            prog="python -m pretrain.probe",
            description="Draw §10.3's probe set, or take one probe point (#99).",
        )
    )
    p.add_argument("--probe-set", default=None, help="committed probe set CSV (#99)")
    p.add_argument("--draw", action="store_true", help="draw and commit the probe set, then stop")
    p.add_argument("--size", type=int, default=PROBE_SIZE, help="tiles to draw (--draw only)")
    p.add_argument("--out", default=None, help="where to write the draw (default: --probe-set)")
    p.add_argument("--checkpoint", default=None, help="encoder to probe (default: untrained)")
    p.add_argument("--epoch", type=int, default=0, help="epoch to log this point at")
    p.add_argument("--device", default=None, help="cuda | cpu | … (default: cuda if present)")
    args = p.parse_args(argv)

    cfg = config_from_args(args)
    if args.probe_set is not None:
        cfg = cfg.replace(data=cfg.data.replace(probe_set_path=args.probe_set))

    if args.draw:
        path = write_probe_set(
            draw_probe_set(
                Path(cfg.data.corpus_dir) / "manifest.parquet",
                split=cfg.data.val_split,
                size=args.size,
            ),
            args.out or cfg.data.probe_set_path,
        )
        from dataset.materialize import file_md5

        drawn = read_probe_set(path)
        counts = np.bincount(np.array(drawn.stratum_id))
        # The md5 is printed because it is the thing that makes the artifact a contract: it is
        # what every probe point logs beside its numbers, and what says two arms were scored
        # over the same tiles. `pretrain.panel`'s draw prints its own for the same reason.
        print(
            f"drew {len(drawn):,} {cfg.data.val_split} tiles over {int((counts > 0).sum())} "
            f"strata -> {path}\nmd5 {file_md5(path)}"
        )
        return 0

    model = MaskedAutoencoder.from_config(cfg)
    if args.checkpoint is not None:
        from .checkpoint import load_checkpoint

        model.load_state_dict(load_checkpoint(args.checkpoint).model)
    model.to(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    with open_run_logger(cfg) as logger:
        probe = KnnProbe.from_config(cfg, logger)
        print(format_probe(probe.run(model, args.epoch)), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
