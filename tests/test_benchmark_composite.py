"""§4.2's inference composite, as arithmetic — the half of #166 that needs no checkpoint.

The composite is the part of A3 that has to be *built* rather than configured, and the reason is
a measurement artifact rather than a fill quality: the decoder predicts each 16 px patch through
its own token, so two patches that disagree at their shared edge print a straight line on the
16 px grid. §5's seam-gap column measures exactly that kind of discontinuity — so without the
composite, the column would charge A3 for the inference scheme rather than for the fill, and
#124's "auditable rather than assumed" would have nothing to audit.

Everything here runs on arrays. The fade weights, the partition of unity and the grid-energy
statistic are all functions of the tile geometry alone, so the property that matters — *the
patch grid is suppressed, and the void boundary is left alone* — is checkable without loading
85M parameters. `tests/test_benchmark_a3.py` carries the half that needs a model.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmark import finetune
from benchmark.config import BenchmarkConfig

TILE = 224
PATCH = 16


# --- The fade weights ---------------------------------------------------------


def test_the_fade_weights_are_a_partition_of_unity():
    """Every pixel's four weights sum to 1 — so the blend moves no level anywhere.

    The one property the whole construction rests on. A blend whose weights summed to 1.02 in the
    patch interiors and 0.98 at the boundaries would *print the grid it was built to remove*, as
    a 4% ripple, and every number downstream would still look plausible.
    """
    weights = finetune.fade_weights(tile_px=TILE, patch_px=PATCH, crossfade_px=PATCH)

    assert weights.shape == (4, TILE, TILE)
    np.testing.assert_allclose(weights.sum(axis=0), 1.0, atol=1e-12)


def test_each_grid_carries_no_weight_at_its_own_patch_boundary():
    """The mechanism, stated as a number: a field contributes least exactly where it seams.

    `fade_weights`' rows are ordered by the (dy, dx) origins `crossfade_origins` returns, and row
    0 is the native grid — the one whose patch boundaries fall at multiples of 16. Read along one
    row at a patch centre, so the y factor is held fixed and the comparison is about x alone.
    """
    weights = finetune.fade_weights(tile_px=TILE, patch_px=PATCH, crossfade_px=PATCH)
    row = weights[0][PATCH // 2]

    boundary = np.arange(PATCH, TILE, PATCH)  # the columns just past a native seam
    centre = boundary - PATCH // 2

    assert row[boundary].max() < row[centre].min()


def test_a_fade_narrower_than_a_patch_is_still_normalized():
    """`crossfade_px` is a config knob, so the construction has to hold away from its default."""
    weights = finetune.fade_weights(tile_px=TILE, patch_px=PATCH, crossfade_px=8)

    np.testing.assert_allclose(weights.sum(axis=0), 1.0, atol=1e-12)


def test_a_fade_wider_than_a_patch_is_refused():
    """A fade that outruns the patch it is fading across has no boundary left to centre on."""
    with pytest.raises(ValueError, match="crossfade"):
        finetune.fade_weights(tile_px=TILE, patch_px=PATCH, crossfade_px=PATCH * 2)


# --- The grid statistic the acceptance criterion is stated on -----------------


def test_grid_energy_sees_a_patch_grid_and_ignores_smooth_terrain():
    """The statistic itself, before it is used to judge anything.

    §5's seam-gap column is not what this measures and must not be: the criterion #166 states is
    that the composite *suppresses the patch grid* rather than that some other column absorbs it,
    so the evidence has to be a direct reading of the 16 px grid. This is that reading — the mean
    absolute step across 16 px-boundary-crossing pixel pairs, minus the mean absolute step across
    the pairs just inside the patches, so smooth terrain scores ~0 and a per-patch offset does
    not.
    """
    y, x = np.mgrid[0:TILE, 0:TILE].astype(np.float64)
    smooth = 0.01 * y + 0.02 * x

    rng = np.random.default_rng(0)
    offsets = rng.normal(scale=5.0, size=(TILE // PATCH, TILE // PATCH))
    gridded = smooth + np.repeat(np.repeat(offsets, PATCH, axis=0), PATCH, axis=1)

    assert finetune.grid_energy(smooth, patch_px=PATCH) < 1e-9
    assert finetune.grid_energy(gridded, patch_px=PATCH) > 3.0


# --- The composite, over fields a model would have produced -------------------


def _patchy_fields(rng, *, scale: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """Four decoded fields with the artifact #166 is about, each on its own grid.

    One smooth truth, plus a constant offset per patch **of that field's own grid** — which is
    what a per-token decoder produces when neighbouring patches disagree about the level. The
    four grids are `crossfade_origins`', so the blend has somewhere to fade to.
    """
    y, x = np.mgrid[0:TILE, 0:TILE].astype(np.float64)
    truth = 40.0 * np.sin(y / 60.0) + 25.0 * np.cos(x / 45.0)

    fields = []
    for dy, dx in finetune.crossfade_origins(PATCH):
        offsets = rng.normal(scale=scale, size=(TILE // PATCH + 1, TILE // PATCH + 1))
        grid = np.repeat(np.repeat(offsets, PATCH, axis=0), PATCH, axis=1)
        # Roll the per-patch offsets onto this field's own grid, then crop to the tile.
        fields.append(truth + np.roll(grid, (dy, dx), axis=(0, 1))[:TILE, :TILE])
    return truth, np.stack(fields)


def test_the_crossfade_suppresses_the_patch_grid():
    """#166's fifth criterion, as arithmetic: the grid statistic drops, and by a lot.

    Measured directly on the blended field rather than inferred from a score, which is the
    distinction the criterion draws — "suppresses the patch grid rather than the seam-gap column
    absorbing it".
    """
    rng = np.random.default_rng(1)
    _, fields = _patchy_fields(rng)

    blended = finetune.crossfade(fields, patch_px=PATCH, crossfade_px=PATCH)
    before = finetune.grid_energy(fields[0], patch_px=PATCH)
    after = finetune.grid_energy(blended, patch_px=PATCH)

    assert before > 3.0
    assert after < before / 3.0


def test_the_crossfade_is_not_a_blur():
    """The grid goes and the terrain stays — otherwise "suppresses" would just mean "smooths".

    A 16 px box blur would also flatten the grid statistic, and would cost the fill every ridge
    it was asked to reconstruct. So the blend is held to the truth it was built from: averaging
    four independent per-patch offsets around one smooth field must land *closer* to that field
    than any single one of them.
    """
    rng = np.random.default_rng(2)
    truth, fields = _patchy_fields(rng)

    blended = finetune.crossfade(fields, patch_px=PATCH, crossfade_px=PATCH)

    single = np.sqrt(np.mean((fields[0] - truth) ** 2))
    composite = np.sqrt(np.mean((blended - truth) ** 2))
    assert composite < single


def test_the_composite_restores_every_pixel_outside_the_void():
    """§4.2's first clause — *predicted patches only where voided, original pixels elsewhere*.

    The void boundary is the one discontinuity the composite must **not** touch: it is the fill
    meeting the terrain, which is what §5's seam-gap column is for. A composite that faded across
    it would be the arm marking its own homework.
    """
    rng = np.random.default_rng(3)
    _, fields = _patchy_fields(rng)
    original = np.full((TILE, TILE), 123.0)
    void = np.zeros((TILE, TILE), dtype=bool)
    void[64:160, 64:160] = True

    out = finetune.composite(fields, original, void, patch_px=PATCH, crossfade_px=PATCH)

    np.testing.assert_array_equal(out[~void], original[~void])
    assert not np.allclose(out[void], original[void])


def test_the_declared_crossfade_is_sixteen_pixels():
    """#166's third criterion is a number, and it is §4.2's own rather than this module's."""
    assert BenchmarkConfig().finetune.crossfade_px == 16


# --- §4.2's near-tile-scale clause --------------------------------------------


def test_only_near_tile_scale_voids_take_the_shifted_encodings():
    """#166's fourth criterion, and the trigger read off §3.2's own size vocabulary.

    "Near tile scale" is not a constant this module invents: a void reaches it when its
    equivalent diameter reaches the **large** bucket's floor, which is the size axis §3.2 already
    measures every draw on and the one the results rows carry.
    """
    cfg = BenchmarkConfig()
    small = np.zeros((TILE, TILE), dtype=bool)
    small[100:120, 100:120] = True
    large = np.zeros((TILE, TILE), dtype=bool)
    large[40:190, 40:190] = True

    assert not finetune.near_tile_scale(small, masks_cfg=cfg.masks)
    assert finetune.near_tile_scale(large, masks_cfg=cfg.masks)


def test_the_shift_offsets_are_the_unshifted_encoding_alone():
    """§4.2 as amended by #177: the fade's four half-patch grids are the composite, and no
    further shifted encoding is declared. The clause that named 0 and 112 px is struck — 112 is
    seven whole patches (identically phased grid) and three-quarters of a 224 px window
    (fabricated canvas), and it measured 45.0 m against 21.0 m without."""
    cfg = BenchmarkConfig()

    assert cfg.finetune.shift_offsets == (0,)
    assert finetune.shift_origins(cfg.finetune) == ((0, 0),)
