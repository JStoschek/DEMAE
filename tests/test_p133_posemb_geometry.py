"""Pin the four numbers ADR-0011 rests on, so they cannot drift out of its prose silently.

Unlike #118's probe, none of this needs TerraMind installed, or any weights at all: TerraMind's
DEM-tokenizer position embedding is built by `build_2d_sincos_posemb` with `learnable_pos_emb=False`,
so it is a closed-form function of the token's integer grid index and lives in the checkpoint only
because it is wrapped in an `nn.Parameter`. Every claim below follows from the formula.

**This is not `benchmark/` arriving early.** Map #117 puts building that module out of scope. What
this leaves behind for it is a written-down constraint and a failing test if that constraint is
violated — specifically, that ADR-0011's centre anchor is the crop it says it is, and that
terratorch's default resize is the stretch ADR-0011 says it is.

The port in `scripts/p133_posemb_geometry.py` is the thing most at risk of drifting: if a
terratorch upgrade changes `build_2d_sincos_posemb`, `test_port_matches_terratorch_formula` is
what should fail. It re-derives the embedding independently rather than calling the port, so a
copy-paste error in the port cannot hide behind itself.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO / "scripts" / "p133_posemb_geometry.py"
RECORD_PATH = REPO / "docs" / "research" / "p133-posemb-geometry.json"


def _load_script():
    spec = importlib.util.spec_from_file_location("p133_posemb_geometry", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return _load_script()


@pytest.fixture(scope="module")
def record():
    return json.loads(RECORD_PATH.read_text())


def _numeric_leaves(value, prefix=""):
    """Flatten a nested JSON-ish structure to ``{path: number}`` so approx can compare it."""
    if isinstance(value, dict):
        return {k: v for key, item in value.items() for k, v in _numeric_leaves(item, f"{prefix}.{key}").items()}
    if isinstance(value, list):
        return {k: v for i, item in enumerate(value) for k, v in _numeric_leaves(item, f"{prefix}[{i}]").items()}
    return {prefix: value}


def test_record_matches_a_fresh_run(script, record):
    """The committed JSON is what the script produces today, not what it produced once."""
    fresh = _numeric_leaves(script.measure())
    stored = _numeric_leaves(record)
    assert fresh.keys() == stored.keys()
    assert fresh == pytest.approx(stored, rel=1e-9), "re-run scripts/p133_posemb_geometry.py --out"


def test_port_matches_terratorch_formula(script):
    """Re-derive the embedding from scratch and check the port agrees, to 1e-12.

    Deliberately independent of `build_2d_sincos_posemb` — a port that drifted would otherwise
    validate itself.

    **The axes are transposed, and that is terratorch's behaviour, not a bug here.** The builder
    calls `torch.meshgrid(grid_w, grid_h, indexing="ij")`, so the *w* coordinate varies along the
    first meshgrid axis, and the subsequent `b (h w) d -> b d h w` regroup lands it on the output's
    **row** axis. Hence `sin(row * omega)` in the first quarter below, where a naive reading of the
    variable names would predict `sin(col * omega)`. DEMAE's own
    `sincos2d_pos_embed` carries the mirror-image quirk and documents it at
    `src/pretrain/model.py:48`.

    None of ADR-0011's four claims depend on which axis is which — they are all statements about
    cropping and resizing one fixed tensor — but a silent axis flip in a future terratorch would
    change the embedding, so it is pinned here.
    """
    dim, grid = script.EMBED_DIM, script.NATIVE_GRID
    pos_dim = dim // 4
    expected = torch.zeros(1, dim, grid, grid, dtype=torch.float64)
    for row in range(grid):
        for col in range(grid):
            for k in range(pos_dim):
                omega = 1.0 / (10000.0 ** (k / pos_dim))
                expected[0, k, row, col] = math.sin(row * omega)
                expected[0, pos_dim + k, row, col] = math.cos(row * omega)
                expected[0, 2 * pos_dim + k, row, col] = math.sin(col * omega)
                expected[0, 3 * pos_dim + k, row, col] = math.cos(col * omega)

    actual = script.build_2d_sincos_posemb(grid, grid)
    assert (actual - expected).abs().max() < 1e-12


def test_rederivation_is_exactly_a_crop(script, record):
    """Claim 1 — the grid index is absolute, so a 14x14 embedding *is* the corner crop of a 16x16.

    Bit-identical, not approximately equal. This is what makes ADR-0011's fix free and exact:
    there is nothing to approximate, only a resize to stop doing.
    """
    assert record["rederived_minus_top_left_crop_max_abs"] == 0.0

    grid = script.TARGET_GRID
    native = script.build_2d_sincos_posemb(script.NATIVE_GRID, script.NATIVE_GRID)
    assert torch.equal(script.build_2d_sincos_posemb(grid, grid), native[:, :, :grid, :grid])


def test_bicubic_stretches_the_index_spacing(record):
    """Claim 2 — `align_corners=False` preserves normalized position, so spacing becomes 16/14."""
    assert record["bicubic_index_spacing"] == pytest.approx(16 / 14, rel=1e-12)
    assert record["crop_index_spacing"] == 1.0
    # It reaches nearly the full extent of the 16-grid across only 14 tokens.
    assert record["bicubic_source_index_last"] > 14.9


def test_the_stretch_costs_most_tokens_their_identity(record):
    """Claim 3 — and why a 0.97 cosine is not the reassurance it looks like.

    Adjacent *correct* positions already sit at ~0.986 cosine, so the resize's error lands inside
    the gap between neighbouring patches. The consequence is positional, not just numerical: the
    error accumulates monotonically along each row until a token ranks its neighbour's embedding
    above its own.
    """
    assert record["bicubic_vs_top_left_rel_rms"] == pytest.approx(0.2625, abs=5e-4)
    assert record["bicubic_vs_top_left_identity_retained"] == 20
    assert record["n_tokens"] == 196
    assert record["distinct_token_cosine_max"] > 0.98


def test_centre_is_the_anchor_nearest_current_behaviour(record):
    """Claim 4 — the anchor is a decision. Nine crops, spread over ~44% relative RMS.

    ADR-0011 picks centre to keep the intervention single-purpose: bicubic centres *and* stretches,
    and only the stretch is wrong. The numbers back that reading — centre is the closest honest
    crop to what terratorch does today, by a clear margin on both measures.
    """
    anchors = record["anchors"]
    assert tuple(record["adr_0011_anchor"]) == (1, 1)

    centre = anchors["1,1"]
    others = [v for k, v in anchors.items() if k != "1,1"]
    assert all(centre["rel_rms_vs_bicubic"] < o["rel_rms_vs_bicubic"] for o in others)
    assert all(centre["identity_retained"] > o["identity_retained"] for o in others)

    spread = max(a["rel_rms_vs_top_left"] for a in anchors.values())
    assert spread > record["bicubic_vs_top_left_rel_rms"], (
        "anchors should differ from each other by more than the error being fixed"
    )
