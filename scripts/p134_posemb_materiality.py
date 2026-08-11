"""Encode 300 DEMAE tiles under three position embeddings, and measure how far the features move.

Issue #134, under map #117; the reading of it is `docs/research/p134-posemb-materiality.md`.
ADR-0011 replaced terratorch's bicubic `pos_emb` resize with the centre-anchored 14x14 crop on a
geometric argument and a formula check — no weights, no forward pass. This script is the missing
half: whether TerraMind's *features* move enough for §2.2's pre-registered ablation row to be
worth a line in the results table.

**It is a materiality check, not an adjudication.** Feature drift cannot say which variant is
better: "better" needs a metric, a metric needs the trained head, and no arm runs on this map. No
number here sends the spec back to bicubic — bicubic is geometrically wrong whatever the drift is
(ADR-0011). The only thing at stake is the ablation row, and its threshold was fixed in ADR-0011
§Decision 5 **before** any of these numbers existed: mean per-token cosine >= 0.99 between the
bicubic and centre features means immaterial, and the row is dropped.

Three arms, differing in exactly one tensor:

    bicubic   F.interpolate(pos_emb, (14, 14), mode="bicubic", align_corners=False)  <- library default
    centre    pos_emb[:, :, 1:15, 1:15]                                              <- ADR-0011's spec
    top_left  pos_emb[:, :, :14, :14]                                                <- == build_2d_sincos_posemb(14, 14)

All three are installed as the encoder's `pos_emb` at 14x14, which leaves terratorch's resize in
the path as a 14->14 no-op — bit-identical, checked at run time and pinned by a test — so the
three feature maps compare elementwise with no resampling confound.

Two metrics per pair. **Mean per-token cosine** is what ADR-0011 pre-registered, taken over the
channel axis of every one of the 300 x 196 tokens. **Linear CKA** is reported beside it because
cosine is a statement about one fixed basis: a rotation of the 768-d space would leave every
downstream linear head equally able to read the features while dragging cosine down, and CKA is
invariant to exactly that. Where the two disagree, the disagreement is the finding.

**terratorch is not a project dependency and must not become one** — see the module docstring of
`scripts/p118_terramind_probe.py`, whose tile reader and normalization this script reuses rather
than re-copying. Run it from the same throwaway venv:

    uv venv --python 3.12 /tmp/tt-venv
    VIRTUAL_ENV=/tmp/tt-venv uv pip install -r docs/research/p118-terramind-requirements.txt
    /tmp/tt-venv/bin/python scripts/p134_posemb_materiality.py \
        --corpus /path/to/data/corpus_500k --out docs/research/p134-posemb-materiality.json

Everything above the terratorch import is importable without it, which is what
`tests/test_p134_posemb_materiality.py` exercises.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

_SCRIPTS = Path(__file__).resolve().parent


def _sibling(name: str):
    """Load another `scripts/` module by path.

    Neither #118's probe nor #133's geometry is importable as a package — they are scripts, and
    map #117 keeps them that way — but both own facts this script must not restate. #118 owns the
    corpus reader and TerraMind's global affine, whose failure mode is silent (#118 §7); #133 owns
    the closed-form embedding builder. A second copy of either is a second thing to keep in sync.
    """
    spec = importlib.util.spec_from_file_location(f"p134_dep_{name}", _SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


p118 = _sibling("p118_terramind_probe")
p133 = _sibling("p133_posemb_geometry")

TERRAMIND_MODEL = p118.TERRAMIND_MODEL
NATIVE_GRID = p133.NATIVE_GRID  # 16 — the tokenizer trained at 256 px
TARGET_GRID = p133.TARGET_GRID  # 14 — DEMAE's primary regime, 224 px at patch 16
TOKENS_PER_TILE = p133.N_TOKENS  # 196
EMBED_DIM = p133.EMBED_DIM  # 768

#: 15 tiles in each of the corpus's 20 strata (ADR-0005) — 300, the ticket's "few hundred".
TILES_PER_STRATUM = 15

VARIANTS = ("bicubic", "centre", "top_left")
PAIRS = (("bicubic", "centre"), ("bicubic", "top_left"), ("centre", "top_left"))

#: ADR-0011 §Decision 5, fixed before this script produced a number.
THRESHOLD = 0.99

#: What picks the tiles, then what describes them in the record. One read serves both.
SELECTION_COLUMNS = ("tile_id", "split", "land_frac", "stratum_id")
MANIFEST_COLUMNS = SELECTION_COLUMNS + ("shard_id", "row_index", "lon", "lat", "local_relief", "source_cog")


# --------------------------------------------------------------------------------------
# Which tiles
# --------------------------------------------------------------------------------------


def read_selection_frame(corpus: Path) -> dict[str, list]:
    """One manifest read, columnar, covering both what picks the tiles and what describes them.

    `select_tile_ids` needs four columns and `selection_rows` needs eight; reading the union once
    is a column-projected pass over half a million rows either way, and two reads would leave the
    two callers able to disagree about what "the manifest" said.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(corpus / "manifest.parquet", columns=list(MANIFEST_COLUMNS))
    return {name: table[name].to_pylist() for name in MANIFEST_COLUMNS}


def select_tile_ids(frame: Mapping[str, Sequence], per_stratum: int = TILES_PER_STRATUM) -> list[str]:
    """`per_stratum` evenly-spaced tiles from each stratum's sorted eligible ids, all-land, test.

    Stated rather than sampled, following #118's rule and for the same reason: a seeded draw is
    reproducible only for someone holding the seed, and this list has to be re-derivable by anyone
    holding the manifest.

    Balanced across strata rather than drawn proportionally, because the question is whether the
    drift is *uniform*. The corpus's 20 strata are its own terrain partition, so an equal quota
    buys the widest spread of terrain the manifest can express for a fixed tile budget; a
    proportional draw would spend a third of the sample on the three largest strata and leave the
    small ones — the rugged ones void filling actually lives in — represented by a handful.

    **Evenly spaced, not the lowest ids.** #118 took the single lowest eligible `tile_id`, which is
    unimpeachable for one tile and wrong for three hundred: a Copernicus tile id begins with its
    latitude band (`..._N00_00_E030_00_...`), so sorting puts every low id in the equatorial band
    and "the 15 lowest per stratum" lands the whole sample in one corner of Africa. Taking indices
    spread across each stratum's sorted list keeps the rule deterministic and one line long while
    spanning the full latitude range the corpus covers. That matters here because the strata
    partition *terrain*, not geography, and a sample confined to one region would leave the
    within-stratum result resting on one region's landforms.

    `land_frac == 1.0` keeps the sample off coastlines, where the ocean mask would be doing part
    of the work, and `split == "test"` keeps it off the geography Phase-1 trained on.
    """
    if per_stratum < 2:
        raise ValueError(f"per_stratum must be at least 2 to span a stratum, got {per_stratum}")

    tile_ids = np.asarray(frame["tile_id"], dtype=object)
    eligible = (np.asarray(frame["split"], dtype=object) == "test") & (
        np.asarray(frame["land_frac"], dtype=np.float64) >= 1.0
    )
    strata = np.asarray(frame["stratum_id"], dtype=np.int64)

    selected: list[str] = []
    for stratum in np.unique(strata):
        in_stratum = sorted(tile_ids[eligible & (strata == stratum)])
        if len(in_stratum) < per_stratum:
            raise ValueError(
                f"stratum {stratum} has only {len(in_stratum)} eligible tiles, need {per_stratum}"
            )
        # Endpoints included, so the span is the stratum's whole id range and not its interior.
        picks = np.round(np.linspace(0, len(in_stratum) - 1, per_stratum)).astype(int)
        selected.extend(in_stratum[i] for i in picks)
    return selected


def selection_rows(frame: Mapping[str, Sequence], tile_ids: Sequence[str]) -> list[dict[str, Any]]:
    """`frame` pivoted to one row per id in `tile_ids`, in that order."""
    index = {tile_id: i for i, tile_id in enumerate(frame["tile_id"])}
    return [{name: frame[name][index[tile_id]] for name in frame} for tile_id in tile_ids]


def iter_batches(
    corpus: Path, rows: Sequence[Mapping[str, Any]], batch_size: int
) -> Iterator[Tensor]:
    """Tiles out of the shards as `[B, 1, 224, 224]` in TerraMind's units.

    Raw int16 metres, then TerraMind's *global* affine — never the pretraining loader's per-tile
    standardization, which destroys absolute elevation and fails silently (#118 §7).
    """
    for lo in range(0, len(rows), batch_size):
        chunk = rows[lo : lo + batch_size]
        metres = np.stack(
            [p118.load_tile_metres(corpus, row["shard_id"], row["row_index"]) for row in chunk]
        )
        yield torch.from_numpy(p118.terramind_normalize(metres))[:, None]


# --------------------------------------------------------------------------------------
# The three embeddings
# --------------------------------------------------------------------------------------


def posemb_variants(native: Tensor) -> dict[str, Tensor]:
    """The ticket's three arms from one `[1, 768, 16, 16]` embedding, all at 14x14, in its dtype.

    Returned at 14x14 rather than as instructions for `forward` to follow, so the only difference
    between arms is the tensor: `ViTEncoder.forward` resizes whatever it finds to the grid the
    input implies, and a 14->14 bicubic resize is bit-identity (`resize_error_at`).
    """
    grid, (row, col) = TARGET_GRID, p133.CENTRE_ANCHOR
    variants = {
        "bicubic": p133.bicubic_resize(native, grid),
        "centre": native[:, :, row : row + grid, col : col + grid],
        "top_left": native[:, :, :grid, :grid],
    }
    return {name: tensor.contiguous() for name, tensor in variants.items()}


def resize_error_at(sample: Tensor) -> float:
    """Max abs difference a same-size bicubic resize introduces. Must be exactly 0.0.

    terratorch resizes on every pass, so the centre and top-left arms still go through
    `F.interpolate` — at 14->14 with `align_corners=False` the sample index is
    `(i + 0.5) * 1 - 0.5 == i`, the kernel lands on integer offsets, and the resize returns its
    input bit-for-bit. Reported as the measured number rather than as a bool, so the record carries
    a fact that could have come out otherwise: a bool that is `True` or an exception is not
    evidence of anything. If a future torch changes bicubic at unit scale, this stops being 0.0 and
    the centre and top-left arms have silently acquired a resampling confound.
    """
    resized = F.interpolate(sample, size=tuple(sample.shape[-2:]), mode="bicubic", align_corners=False)
    return float((resized - sample).abs().max())


# --------------------------------------------------------------------------------------
# The two metrics
# --------------------------------------------------------------------------------------


def per_token_cosine(a: Tensor, b: Tensor) -> Tensor:
    """`[B, D, H, W]` x2 -> `[B, H*W]`, cosine over the channel axis of each token."""
    return F.cosine_similarity(a, b, dim=1).flatten(1)


class TokenMoments:
    """Streaming sums and cross-products, enough for exact linear CKA over every token.

    CKA (Kornblith et al. 2019) wants `||Xc^T Yc||_F` over the whole sample, where a row is one
    token's 768-d feature. Materializing that is 300 x 196 x 768 floats per variant — affordable
    here, and the wrong shape to write: the sample size is the one knob a reader will want to turn,
    and `S_xy = X^T Y` is only 768x768 whatever `n` is. Centring after the fact is exact, since
    `Xc^T Yc == X^T Y - (1^T X)^T (1^T Y) / n`.

    Accumulated in float64 — the features are float32, and `X^T X` over 58,800 rows is where a
    float32 accumulation would start to show.
    """

    def __init__(self, names: Sequence[str], dim: int = EMBED_DIM):
        self.names = tuple(names)
        self.dim = dim
        self.n = 0
        self._sum = {name: torch.zeros(dim, dtype=torch.float64) for name in self.names}
        self._cross = {
            (a, b): torch.zeros(dim, dim, dtype=torch.float64)
            for i, a in enumerate(self.names)
            for b in self.names[i:]
        }

    def update(self, batch: Mapping[str, Tensor]) -> None:
        """`batch` maps every name to an `(n_rows, dim)` block of token features."""
        missing = set(self.names) - set(batch)
        unknown = set(batch) - set(self.names)
        if missing or unknown:
            raise KeyError(f"expected exactly {self.names}, missing={sorted(missing)} unknown={sorted(unknown)}")

        rows = {name: batch[name].detach().to(torch.float64).reshape(-1, self.dim) for name in self.names}
        counts = {tensor.shape[0] for tensor in rows.values()}
        if len(counts) != 1:
            raise ValueError(f"all variants must contribute the same number of rows, got {counts}")

        self.n += counts.pop()
        for name, tensor in rows.items():
            self._sum[name] += tensor.sum(dim=0)
        for (a, b), acc in self._cross.items():
            acc += rows[a].T @ rows[b]

    def _centred_cross(self, a: str, b: str) -> Tensor:
        key = (a, b) if (a, b) in self._cross else (b, a)
        cross = self._cross[key]
        if key != (a, b):
            cross = cross.T
        return cross - torch.outer(self._sum[a], self._sum[b]) / self.n

    def cka(self, a: str, b: str) -> float:
        """Linear CKA: invariant to rotation and isotropic scaling of either representation."""
        hsic_ab = float(self._centred_cross(a, b).pow(2).sum())
        hsic_aa = float(self._centred_cross(a, a).pow(2).sum())
        hsic_bb = float(self._centred_cross(b, b).pow(2).sum())
        return hsic_ab / float(np.sqrt(hsic_aa * hsic_bb))


def summarize_cosines(cosines: np.ndarray, strata: Sequence[int]) -> dict[str, Any]:
    """The distribution and its two structures, not just the mean.

    `cosines` is `(n_tiles, 196)`. The pre-registered rule is on the grand mean, but a mean is
    only a verdict if nothing interesting is hiding under it, and there are exactly two places
    something could hide:

    - **Position.** ADR-0011's argument says the index error accumulates along each row, so the
      damage should be concentrated at the far edge of the tile rather than spread evenly. That is
      a falsifiable prediction, and `cosine_by_position` is what falsifies or confirms it.
    - **Terrain.** If the drift were much larger in rugged strata than flat ones, the grand mean
      would understate the cost for exactly the regime void filling lives in.
    """
    flat = cosines.ravel()
    p01, p50, p99 = np.percentile(flat, [1, 50, 99])
    by_stratum: dict[str, float] = {}
    for stratum in sorted(set(strata)):
        rows = [i for i, s in enumerate(strata) if s == stratum]
        by_stratum[str(stratum)] = float(cosines[rows].mean())
    return {
        "n_token_cosines": int(flat.size),
        "mean_per_token_cosine": float(flat.mean()),
        "cosine_std": float(flat.std()),
        "cosine_min": float(flat.min()),
        "cosine_p01": float(p01),
        "cosine_median": float(p50),
        "cosine_p99": float(p99),
        "cosine_max": float(flat.max()),
        # Row-major over the 14x14 grid, averaged across tiles.
        "cosine_by_position": cosines.mean(axis=0).reshape(TARGET_GRID, TARGET_GRID).tolist(),
        "cosine_by_stratum": by_stratum,
    }


# --------------------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------------------


def encode(model, x: Tensor, pos_emb: Tensor) -> Tensor:
    """One forward pass with `pos_emb` installed. `[B, 1, 224, 224]` -> `[B, 768, 14, 14]`."""
    model.encoder.pos_emb = torch.nn.Parameter(pos_emb, requires_grad=False)
    with torch.no_grad():
        return model.encoder(x)


def run(corpus: Path, device: str, batch_size: int, per_stratum: int) -> dict[str, Any]:
    frame = read_selection_frame(corpus)
    tile_ids = select_tile_ids(frame, per_stratum=per_stratum)
    rows = selection_rows(frame, tile_ids)

    constants = p118.probe_constants()
    if not constants["matches_this_script"]:
        raise AssertionError(
            f"installed terratorch says mean={constants['mean']} std={constants['std']}, "
            f"#118 recorded {p118.TERRAMIND_DEM_MEAN} / {p118.TERRAMIND_DEM_STD}"
        )

    model = p118.build_tokenizer().to(device)
    model_pos_emb = model.encoder.pos_emb.detach()  # the checkpoint's own 16x16, left as it came
    native = model_pos_emb.cpu()

    # ADR-0011's tamper-evidence consequence, run rather than promised: the embedding that came
    # down with the checkpoint is the closed form, so cropping it *is* re-deriving it. If a
    # terratorch upgrade ever ships a learned embedding here, this flag flips and the whole
    # argument — #133's and this script's — needs redoing rather than silently continuing.
    closed_form = p133.build_2d_sincos_posemb(NATIVE_GRID, NATIVE_GRID)
    loaded_is_closed_form = bool(
        native.shape == closed_form.shape
        and torch.allclose(native.to(torch.float64), closed_form, atol=1e-6, rtol=0)
    )

    # Built from the checkpoint's own float32 tensor, in float32 — the library's own arithmetic,
    # not a more accurate float64 rendering of it. On a non-CPU device the resize is still computed
    # on the host and the result moved, which is a difference from what terratorch would do
    # in-place; `bicubic_arm_matches_untouched_model` below is what would catch it mattering.
    variants = {k: v.to(device) for k, v in posemb_variants(native).items()}
    resize_error = resize_error_at(variants["centre"])
    if resize_error != 0.0:
        raise AssertionError(f"bicubic 14->14 is not identity (max abs diff {resize_error})")

    moments = TokenMoments(VARIANTS)
    cosines: dict[tuple[str, str], list[np.ndarray]] = {pair: [] for pair in PAIRS}
    self_consistency = 1.0
    bicubic_matches_untouched = 1.0
    feature_shape_per_tile: list[int] | None = None

    for batch in iter_batches(corpus, rows, batch_size):
        x = batch.to(device)

        # The control for the whole design: encode once with the checkpoint's *untouched* 16x16
        # embedding, letting `ViTEncoder.forward` do its own resize, and check the bicubic arm
        # reproduces it. Without this, "bicubic is terratorch's default" rests on reading
        # `forward` rather than on running it — and on nothing else in `forward` keying off the
        # embedding's spatial size, which reading is a poor way to establish.
        untouched = encode(model, x, model_pos_emb)
        feats = {name: encode(model, x, variants[name]) for name in VARIANTS}
        feature_shape_per_tile = [1, *feats["centre"].shape[1:]]
        bicubic_matches_untouched = min(
            bicubic_matches_untouched, float(per_token_cosine(feats["bicubic"], untouched).min())
        )

        moments.update({name: f.permute(0, 2, 3, 1).reshape(-1, EMBED_DIM).cpu() for name, f in feats.items()})
        for pair in PAIRS:
            cosines[pair].append(per_token_cosine(feats[pair[0]], feats[pair[1]]).cpu().numpy())

        # The noise floor, measured rather than assumed: encode one variant a second time and
        # keep the *worst* token. It is not bit-identity — float32 reduction order on a threaded
        # CPU matmul is not reproducible pass to pass — so the honest control is "how far apart
        # do two runs of the *same* embedding land", against which the between-arm drift is read.
        repeat = encode(model, x, variants["centre"])
        self_consistency = min(self_consistency, float(per_token_cosine(feats["centre"], repeat).min()))

    import importlib.metadata as md

    strata = [int(row["stratum_id"]) for row in rows]
    pairs = {}
    for a, b in PAIRS:
        stacked = np.concatenate(cosines[(a, b)], axis=0)
        pairs[f"{a}_vs_{b}"] = summarize_cosines(stacked, strata) | {"cka": moments.cka(a, b)}

    headline = pairs["bicubic_vs_centre"]["mean_per_token_cosine"]
    immaterial = headline >= THRESHOLD

    return {
        "issue": 134,
        "model": TERRAMIND_MODEL,
        "device": device,
        "versions": {p: md.version(p) for p in ("terratorch", "diffusers", "torch", "timm", "numpy")},
        "normalization": "terramind_global_z_score",
        "constants": constants,
        # The sample.
        "n_tiles": len(tile_ids),
        "tiles_per_stratum": per_stratum,
        "n_strata": int(len({row["stratum_id"] for row in rows})),
        "first_tile_id": tile_ids[0],
        "last_tile_id": tile_ids[-1],
        # Recorded because the first draft of the selection rule got this wrong and nothing in the
        # metrics would have shown it: "the lowest ids per stratum" is a latitude filter in
        # disguise, and it put all 300 tiles in one corner of Africa.
        "n_source_cogs": len({row["source_cog"] for row in rows}),
        "lat_range": [min(row["lat"] for row in rows), max(row["lat"] for row in rows)],
        "lon_range": [min(row["lon"] for row in rows), max(row["lon"] for row in rows)],
        "local_relief_range_metres": [
            min(row["local_relief"] for row in rows),
            max(row["local_relief"] for row in rows),
        ],
        "tokens_per_tile": TOKENS_PER_TILE,
        "n_token_rows": moments.n,
        # The mechanism.
        "loaded_pos_emb_shape": list(native.shape),
        "loaded_pos_emb_is_the_closed_form": loaded_is_closed_form,
        "resize_error_at_14x14": resize_error,
        "feature_shape_per_tile": feature_shape_per_tile,
        "self_consistency_cosine": self_consistency,
        "bicubic_arm_matches_untouched_model": bicubic_matches_untouched,
        # The answer.
        "pairs": pairs,
        "threshold": THRESHOLD,
        "verdict": "immaterial" if immaterial else "material",
        "ablation_row": "drop" if immaterial else "keep",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, default=None, help="path to data/corpus_500k")
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda"))
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--per-stratum", type=int, default=TILES_PER_STRATUM)
    parser.add_argument("--out", type=Path, default=None, help="where to write the JSON record")
    args = parser.parse_args(argv)

    record = run(
        corpus=args.corpus or p118.find_corpus(),
        device=args.device,
        batch_size=args.batch_size,
        per_stratum=args.per_stratum,
    )

    text = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
