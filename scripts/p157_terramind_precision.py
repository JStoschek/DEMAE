"""Measure the TerraMind arm's fp32-versus-compute-dtype agreement, and record it.

Issue [#157](https://github.com/JStoschek/DEMAE/issues/157), under map
[#117](https://github.com/JStoschek/DEMAE/issues/117).
[ADR-0012](../docs/adr/0012-frozen-feature-extractor-contract.md) §6 pins fp32 as the reference
tensor and lets the *compute* dtype be a per-task declared field, and it rests that on one
number: mean per-token cosine **0.999968**, three orders of magnitude the kind side of
[#134](https://github.com/JStoschek/DEMAE/issues/134)'s 0.99 materiality bar. That number is
DEMAE-family weights on MPS, and §6 is explicit that the TerraMind half **does not exist** and is
to be measured on first run rather than inherited. This is that run.

**It drives `benchmark.extract`'s own measurement rather than reimplementing it.** What comes out
is literally what :meth:`benchmark.extract.Extractor.precision_agreement` returns after one
`extract` call, so the recorded number is the mechanism a real run records and not a parallel
computation that could agree with it by luck. The reading of it is
`docs/research/p157-terramind-precision.md`, and `tests/test_p157_terramind_precision.py` pins
the headline claims against the JSON so neither drifts out of that prose silently.

Needs terratorch (`pip install -e '.[terramind]'`), the 1.15 GB tokenizer, and the 47 GB corpus,
so it is not reproducible from a fresh clone — which is why the record is committed:

    PYTHONPATH=src .venv/bin/python scripts/p157_terramind_precision.py \
        --device mps --out docs/research/p157-terramind-precision.json

The tile rule is **p134's, imported rather than restated**: the same 20-stratum, evenly-spaced,
all-land, test-split draw, at a smaller per-stratum quota. Two answers to "which tiles" is how a
second sample quietly becomes a second population.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO = Path(__file__).resolve().parent.parent

#: Three per stratum over the corpus's 20 strata: 60 tiles, 11,760 tokens. Chosen against
#: ADR-0012 §6's own DEMAE-side sample — 64 tiles, 12,544 tokens — so the two figures the ADR
#: ends up carrying side by side are measured at comparable scale rather than one at ten times
#: the other. Balanced across strata for p134's reason: the question is whether the disagreement
#: is *uniform*, and a proportional draw would spend the sample on the three largest strata.
TILES_PER_STRATUM = 3

#: The arm ADR-0012 §6 names. A9 reads the same weights through a separate registry entry, and
#: gets its own figure on its own first run for exactly the reason this script exists.
ARM = "A5"


def _p134():
    """p134's driver, loaded by path — `scripts/` is not a package and should not become one."""
    path = REPO / "scripts" / "p134_posemb_materiality.py"
    spec = importlib.util.spec_from_file_location("p134_posemb_materiality", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_tiles(corpus: Path, per_stratum: int = TILES_PER_STRATUM) -> list[dict[str, Any]]:
    """The manifest rows for p134's draw, in its order."""
    p134 = _p134()
    frame = p134.read_selection_frame(corpus)
    tile_ids = p134.select_tile_ids(frame, per_stratum=per_stratum)
    return p134.selection_rows(frame, tile_ids)


def load_tiles(corpus: Path, rows: Sequence[dict]) -> np.ndarray:
    """The raw int16 metres for those rows — what the extractor's door accepts (ADR-0012 §1)."""
    from dataset.shards import shard_path

    out = []
    for row in rows:
        shard = np.load(shard_path(corpus / "shards", row["shard_id"]), mmap_mode="r")
        out.append(np.asarray(shard[row["row_index"]], dtype=np.int16))
    return np.stack(out)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, default=REPO / "data" / "corpus_500k")
    parser.add_argument("--checkpoint", type=Path, default=None, help="TerraMind_Tokenizer_DEM.pt")
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda"))
    parser.add_argument("--compute-dtype", default="bf16", choices=("bf16", "fp16"))
    parser.add_argument("--per-stratum", type=int, default=TILES_PER_STRATUM)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    from benchmark.arms import ENCODERS
    from benchmark.config import BenchmarkConfig
    from benchmark.extract import Extractor, load_spec, spec_path
    from dataset.normalize import land_mask

    checkpoint = args.checkpoint
    if checkpoint is None:
        from huggingface_hub import hf_hub_download

        checkpoint = Path(
            hf_hub_download(
                "ibm-esa-geospatial/TerraMind-1.0-Tokenizer-DEM", "TerraMind_Tokenizer_DEM.pt"
            )
        )

    base = BenchmarkConfig()
    cfg = base.replace(
        arm=ARM,
        paths=base.paths.replace(terramind_checkpoint=str(checkpoint)),
        extract=base.extract.replace(compute_dtype=args.compute_dtype),
    )
    spec = load_spec(spec_path(cfg))
    extractor = Extractor(spec, cfg, device=args.device)

    rows = select_tiles(args.corpus, args.per_stratum)
    raw = load_tiles(args.corpus, rows)
    # The mask is land-only here, and it makes no difference to what is being measured: A5's diet
    # is TerraMind's global affine, which has no per-tile term a mask could enter (ADR-0012
    # Decision 1). Passing the real one anyway keeps this on the contract's one code path.
    extractor.extract(raw, land_mask(raw), arm=ARM)
    agreement = extractor.precision_agreement(ARM)

    import importlib.metadata as md

    entry = spec.encoder_for(ARM)
    record = {
        "issue": 157,
        "adr": "ADR-0012",
        "arm": ARM,
        "agreement": agreement,
        "encoder": {
            "name": entry.name,
            "checkpoint_md5": entry.checkpoint_md5,
            "pos_emb": entry.pos_emb,
            "normalization": entry.normalization,
            "registry_params": ENCODERS[entry.name].params,
            # p118 §8 counted 90,172,416 on the encoder *as terratorch builds it*, with its
            # native 16x16 `pos_emb`. ADR-0011 substitutes the 14x14 crop, so the model that
            # actually runs holds `768 * (16**2 - 14**2)` = 46,080 fewer — all of them frozen
            # position, none of them trainable. Recorded with the arithmetic checked rather than
            # left as a number that looks like it contradicts p118.
            "measured_params": sum(p.numel() for p in extractor.encoder(ARM).parameters()),
            "params_delta_is_the_crop": (
                ENCODERS[entry.name].params
                - sum(p.numel() for p in extractor.encoder(ARM).parameters())
                == entry.width
                * (entry.pos_emb["native_grid_px"] ** 2 - entry.pos_emb["grid_px"] ** 2)
            ),
        },
        "extractor_spec_md5": spec.md5(),
        "tiles": [
            {
                "tile_id": row["tile_id"],
                "stratum_id": row["stratum_id"],
                "split": row["split"],
                "lon": row["lon"],
                "lat": row["lat"],
                "local_relief_metres": row["local_relief"],
            }
            for row in rows
        ],
        "selection": {
            "rule": "p134's: evenly-spaced, all-land, test-split, per stratum",
            "per_stratum": args.per_stratum,
        },
        "versions": {
            p: md.version(p) for p in ("terratorch", "diffusers", "torch", "timm", "numpy")
        },
    }
    text = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
