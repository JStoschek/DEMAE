"""Drive TerraMind's DEM tokenizer encoder over one real DEMAE tile, and record what it returns.

Issue #118, under map #117. Five arms of the Task-1 table (A5, A9 and the ablations hanging off
them) assume `terratorch` can be installed and driven to produce a 768-d feature map from a
single-channel DEM, and that its token grid co-registers with DEMAE's 14x14. Nobody in this repo
had ever run it. This script is that run, kept in the tree so the answer is reproducible rather
than a number somebody once saw; the reading of it is
`docs/research/p118-terramind-encoder-probe.md`.

Two things it deliberately does *not* do. It does not build anything under `benchmark/` — map #117
puts that out of scope, and a probe that grows into a module is how a plan-only map stops being
plan-only. And it does not decode: the tokenizer's decode path is diffusion, which is the
expensive half and answers no question #118 asks.

**terratorch is not a project dependency and must not become one.** It resolves 113 packages —
lightning, lightly==1.5.22, torchgeo, geopandas — none of which the corpus ETL has any reason to
carry, and `numpy>=2.2` / `timm>=1.0.15` sit close enough to DEMAE's own floors to make a shared
lockfile a standing hazard. Run this from a throwaway venv and point `--corpus` at the real tree:

    uv venv --python 3.12 /tmp/tt-venv
    VIRTUAL_ENV=/tmp/tt-venv uv pip install 'terratorch>=1.2.5'
    /tmp/tt-venv/bin/python scripts/p118_terramind_probe.py --corpus /path/to/data/corpus_500k

Everything above the `terratorch` import is importable without it, which is what
`tests/test_p118_terramind_probe.py` exercises.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

# TerraMind's global DEM normalization, from `terramind_register.v1_pretraining_{mean,std}`
# keyed by the modality below. `probe_constants()` re-reads them from the installed package and
# refuses to continue if they have moved — #118 asks for the installed source, not this memo.
TERRAMIND_DEM_MEAN = 670.665
TERRAMIND_DEM_STD = 951.272
TERRAMIND_DEM_MODALITY = "tok_dem@224"

TERRAMIND_MODEL = "terramind_v1_tokenizer_dem"

TILE_PX = 224

# The tile this probe was recorded on, re-derivable by `select_tile_id` (see its docstring).
# Rwenzori Mountains, on the Uganda/DRC border: 1165 m of local relief at 1789 m mean elevation.
PROBE_TILE_ID = "Copernicus_DSM_COG_10_N00_00_E030_00_DEM_1344_448"

SELECTION_COLUMNS = ("tile_id", "shard_id", "row_index", "split", "land_frac", "local_relief")


# --------------------------------------------------------------------------------------
# The DEMAE side: a stored tile, in the units TerraMind expects
# --------------------------------------------------------------------------------------


def terramind_normalize(metres: np.ndarray) -> np.ndarray:
    """Apply TerraMind's *global* DEM affine — not DEMAE's per-tile standardization.

    The distinction is load-bearing and silent if you get it wrong. DEMAE's corpus is
    per-tile standardized, so a tile leaves the loader with mean 0 / std 1 and its absolute
    elevation destroyed. TerraMind was trained on one fixed affine over metres, so absolute
    elevation is signal to it. Both produce a tensor the encoder accepts.
    """
    return ((metres.astype(np.float32) - TERRAMIND_DEM_MEAN) / TERRAMIND_DEM_STD).astype(np.float32)


def to_model_input(metres: np.ndarray):
    """A (224, 224) metre array to the [1, 1, 224, 224] float32 tensor the encoder takes."""
    if metres.shape != (TILE_PX, TILE_PX):
        raise ValueError(f"expected a {TILE_PX}x{TILE_PX} tile, got {metres.shape}")

    import torch

    return torch.from_numpy(terramind_normalize(metres))[None, None]


def find_corpus(start: Path | None = None) -> Path:
    """Locate a `data/corpus_500k` that has its **shards**, walking up from `start`.

    The manifest is not the thing to test for. `.gitignore` excludes `data/*` and then
    deliberately re-includes the corpus index (issue #51), so every clone and every worktree has
    `manifest.parquet` and only the machine holding the 47 GB build has `shards/`. Probing for
    the manifest finds an empty shell and fails a hundred lines later inside `np.load`.
    """
    here = (start or Path(__file__).resolve().parent).resolve()
    for base in [here, *here.parents]:
        candidate = base / "data" / "corpus_500k"
        if (candidate / "shards").is_dir() and any(candidate.glob("shards/shard_*.npy")):
            return candidate
    raise FileNotFoundError("no data/corpus_500k with materialized shards above " + str(here))


def read_selection_frame(corpus: Path) -> dict[str, list]:
    import pyarrow.parquet as pq

    table = pq.read_table(corpus / "manifest.parquet", columns=list(SELECTION_COLUMNS))
    return {name: table[name].to_pylist() for name in SELECTION_COLUMNS}


def select_tile_id(frame: Mapping[str, Sequence], quantile: float = 0.90) -> str:
    """The lowest `tile_id` among all-land test tiles in the top relief decile.

    Deterministic on purpose, and stated rather than sampled: a seeded draw would be
    reproducible too, but only for someone holding the seed. "Rugged" is the regime the
    void-filling benchmark lives in, and `land_frac == 1.0` keeps the probe off a coastline
    where the ocean mask would be doing half the work.
    """
    relief = np.asarray(frame["local_relief"], dtype=np.float64)
    land = np.asarray(frame["land_frac"], dtype=np.float64)
    split = np.asarray(frame["split"], dtype=object)
    tile_ids = np.asarray(frame["tile_id"], dtype=object)

    is_test = split == "test"
    if not is_test.any():
        raise ValueError("no tile in the test split")

    threshold = float(np.quantile(relief[is_test], quantile))
    eligible = is_test & (land >= 1.0) & (relief >= threshold)
    if not eligible.any():
        raise ValueError("no tile passes the selection rule")

    return min(tile_ids[eligible])


def manifest_row(corpus: Path, tile_id: str) -> dict[str, Any]:
    import pyarrow.parquet as pq

    table = pq.read_table(corpus / "manifest.parquet")
    ids = table["tile_id"].to_pylist()
    try:
        index = ids.index(tile_id)
    except ValueError:
        raise ValueError(f"{tile_id} is not in {corpus}/manifest.parquet") from None
    return {name: table[name][index].as_py() for name in table.column_names}


def load_tile_metres(corpus: Path, shard_id: int, row_index: int) -> np.ndarray:
    """One tile back out of the shards, as float32 metres above the ellipsoid."""
    shard = np.load(corpus / "shards" / f"shard_{shard_id:05d}.npy", mmap_mode="r")
    return np.asarray(shard[row_index], dtype=np.float32)


# --------------------------------------------------------------------------------------
# The TerraMind side
# --------------------------------------------------------------------------------------


def probe_constants() -> dict[str, Any]:
    """Read the normalization constants out of the installed package, not out of this file."""
    from terratorch.models.backbones.terramind.model.terramind_register import (
        v1_pretraining_mean,
        v1_pretraining_std,
    )

    mean = v1_pretraining_mean[TERRAMIND_DEM_MODALITY]
    std = v1_pretraining_std[TERRAMIND_DEM_MODALITY]
    if len(mean) != 1 or len(std) != 1:
        raise AssertionError(f"expected single-channel constants, got {mean} / {std}")
    return {
        "source": "terratorch.models.backbones.terramind",
        "modality": TERRAMIND_DEM_MODALITY,
        "mean": mean[0],
        "std": std[0],
        "matches_this_script": (mean[0] == TERRAMIND_DEM_MEAN and std[0] == TERRAMIND_DEM_STD),
    }


def build_tokenizer():
    from terratorch.registry import FULL_MODEL_REGISTRY

    model = FULL_MODEL_REGISTRY.build(TERRAMIND_MODEL, pretrained=True)
    return model.eval()


def probe_encoder(model, x) -> dict[str, Any]:
    """The four shape claims #118 asks about, measured on one forward pass each."""
    import torch

    with torch.no_grad():
        # `prepare_input` is in the extraction snippet at §3.4 of the selection memo, annotated
        # "no-op for DEM". Run it rather than skipping it, so this probe drives the documented
        # path exactly — and record whether the annotation is true instead of trusting it.
        prepared = model.prepare_input(x)
        prepare_input_is_identity = bool(torch.equal(prepared, x))

        feats = model.encoder(prepared)
        latent = model.quant_proj(feats)
        intermediates = model.encoder(prepared, return_intermediates=True)

    # 14x14 is what §2.2's co-registration premise needs, but it is not the grid this encoder
    # was trained on: the tokenizer's own training args say `input_size=256`, i.e. a 16x16 grid,
    # and `ViTEncoder.forward` bicubically interpolates `pos_emb` to whatever grid the input
    # implies. So the premise holds — and holds by resampling, which is worth recording rather
    # than discovering later. A second pass at 256 px pins that the grid follows the input.
    with torch.no_grad():
        native = model.encoder(
            torch.nn.functional.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
        )

    return {
        "input_shape": list(x.shape),
        "prepare_input_is_identity": prepare_input_is_identity,
        "encoder_output_shape": list(feats.shape),
        "pos_emb_shape": list(model.encoder.pos_emb.shape),
        "pos_emb_is_interpolated_at_224": list(model.encoder.pos_emb.shape[-2:]) != list(feats.shape[-2:]),
        "encoder_output_shape_at_256": list(native.shape),
        "encoder_output_dtype": str(feats.dtype).removeprefix("torch."),
        "encoder_feature_stats": {
            "mean": float(feats.mean()),
            "std": float(feats.std()),
            "min": float(feats.min()),
            "max": float(feats.max()),
        },
        "quant_proj_output_shape": list(latent.shape),
        "quant_proj_module": repr(model.quant_proj),
        "n_intermediates": len(intermediates),
        "intermediate_shape": list(intermediates[-1].shape),
        "intermediates_all_same_shape": len({tuple(t.shape) for t in intermediates}) == 1,
        "last_intermediate_is_encoder_output": bool(
            torch.allclose(intermediates[-1], feats, atol=0, rtol=0)
        ),
    }


# --------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, default=None, help="path to data/corpus_500k")
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda"))
    parser.add_argument("--out", type=Path, default=None, help="where to write the JSON record")
    args = parser.parse_args(argv)

    import torch

    corpus = args.corpus or find_corpus()
    row = manifest_row(corpus, PROBE_TILE_ID)
    tile = load_tile_metres(corpus, row["shard_id"], row["row_index"])
    x = to_model_input(tile).to(args.device)

    constants = probe_constants()
    if not constants["matches_this_script"]:
        raise AssertionError(
            f"installed terratorch says mean={constants['mean']} std={constants['std']}, "
            f"this script says {TERRAMIND_DEM_MEAN} / {TERRAMIND_DEM_STD}"
        )

    model = build_tokenizer().to(args.device)
    record = probe_encoder(model, x)

    import importlib.metadata as md

    record |= {
        "issue": 118,
        "model": TERRAMIND_MODEL,
        "constants": constants,
        "device": args.device,
        "versions": {
            p: md.version(p) for p in ("terratorch", "diffusers", "torch", "timm", "numpy")
        },
        "tile": {
            "tile_id": PROBE_TILE_ID,
            "lon": row["lon"],
            "lat": row["lat"],
            "stratum_id": row["stratum_id"],
            "split": row["split"],
            "mean_metres": row["mean"],
            "std_metres": row["std"],
            "local_relief_metres": row["local_relief"],
            "content_hash": str(row["content_hash"]),
        },
        "encoder_params": sum(p.numel() for p in model.encoder.parameters()),
        "input_stats": {"mean": float(x.mean()), "std": float(x.std())},
    }

    text = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
