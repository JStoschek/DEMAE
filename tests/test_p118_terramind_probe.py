"""The half of #118's probe that runs without TerraMind installed.

The probe script is mostly a driver for someone else's model, and there is nothing to unit-test
about `FULL_MODEL_REGISTRY.build` — that claim is settled by running it, and the run's record is
`docs/research/p118-terramind-encoder-probe.md`. What *is* ours, and what every arm of the §3.4
comparison will reuse, is the two lines either side of the forward pass: getting a stored tile
back out as metres, and normalizing it the way TerraMind expects rather than the way DEMAE does.

That second one is the whole reason this file exists. DEMAE's corpus is **per-tile standardized**
(`corpus_meta.json: normalization=per_tile_standardize`) — every tile leaves the loader with mean
0 and std 1, and its absolute elevation is *gone*. TerraMind was trained on a single global affine
with fixed constants, so a tile at 1789 m and a tile at 200 m must stay 1589 m apart after
normalization. Feed the TerraMind encoder DEMAE-standardized tiles and it still returns a
plausible-looking [1, 768, 14, 14] — no shape error, no warning, just features computed from an
input distribution the model never saw. `test_offset_survives` is the test that fails if anyone
wires the two normalizations together later.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
PROBE_PATH = REPO / "scripts" / "p118_terramind_probe.py"

def _load_probe():
    spec = importlib.util.spec_from_file_location("p118_probe", PROBE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = _load_probe()

# The real corpus is local but gitignored (47 GB of shards), so it is absent from a fresh clone
# and from any worktree. The tile-reading tests are conditional on it; the normalization tests —
# the ones carrying the claim in this module's docstring — never are.
try:
    CORPUS = probe.find_corpus(REPO)
except FileNotFoundError:
    CORPUS = None

requires_corpus = pytest.mark.skipif(CORPUS is None, reason="data/corpus_500k not present")


class TestConstants:
    def test_constants_are_terramind_s_own(self):
        """The values #118 asks to confirm, pinned here so a typo in the script is a red test."""
        assert probe.TERRAMIND_DEM_MEAN == 670.665
        assert probe.TERRAMIND_DEM_STD == 951.272

    def test_modality_key_is_the_224_variant(self):
        assert probe.TERRAMIND_DEM_MODALITY == "tok_dem@224"


class TestNormalize:
    def test_applies_the_fixed_affine(self):
        metres = np.array([[670.665, 1621.937], [-280.607, 0.0]], dtype=np.float32)
        out = probe.terramind_normalize(metres)
        expected = (metres - 670.665) / 951.272
        np.testing.assert_allclose(out, expected, rtol=0, atol=1e-6)
        # And the hand-checked values, so the test does not merely restate the implementation.
        np.testing.assert_allclose(out, [[0.0, 1.0], [-1.0, -0.705022]], rtol=0, atol=1e-5)

    def test_offset_survives(self):
        """A constant elevation offset must reach the model. Per-tile standardization eats it.

        This is the failure the docstring describes: both tiles below have identical *shape* and
        identical per-tile statistics, and differ only in where they sit above sea level. Under
        TerraMind's global affine they stay 1589 m / 951.272 apart. Under DEMAE's per-tile
        standardization they would come out identical, and the encoder would have no way to say so.
        """
        rng = np.random.default_rng(0)
        relief = rng.normal(0, 120, size=(224, 224)).astype(np.float32)
        low = probe.terramind_normalize(relief + 200.0)
        high = probe.terramind_normalize(relief + 1789.0)

        delta = high - low
        np.testing.assert_allclose(delta, (1789.0 - 200.0) / 951.272, rtol=0, atol=1e-4)
        assert not np.allclose(low, high)

    def test_does_not_rescale_by_tile_std(self):
        """Relief is carried in TerraMind's units, not renormalized to unit variance."""
        flat = np.full((224, 224), 1000.0, dtype=np.float32) + np.zeros((224, 224), np.float32)
        flat[0, 0] = 1010.0
        out = probe.terramind_normalize(flat)
        assert out.std() == pytest.approx(flat.std() / 951.272, rel=1e-5)

    def test_returns_float32(self):
        out = probe.terramind_normalize(np.zeros((224, 224), dtype=np.int16))
        assert out.dtype == np.float32


class TestModelInput:
    def test_shape_is_bchw_with_one_channel(self):
        x = probe.to_model_input(np.zeros((224, 224), dtype=np.float32))
        assert tuple(x.shape) == (1, 1, 224, 224)
        assert str(x.dtype) == "torch.float32"

    def test_rejects_a_tile_that_is_not_224(self):
        with pytest.raises(ValueError, match="224"):
            probe.to_model_input(np.zeros((256, 256), dtype=np.float32))

    def test_normalizes_on_the_way_in(self):
        metres = np.full((224, 224), 670.665, dtype=np.float32)
        x = probe.to_model_input(metres)
        assert float(x.abs().max()) == pytest.approx(0.0, abs=1e-6)


class TestTileSelection:
    def test_rule_is_a_pure_function_of_the_frame(self):
        """Lowest tile_id among full-land test tiles in the top relief decile — no RNG, no clock."""
        frame = {
            "tile_id": ["b", "a", "c", "d", "e"],
            "split": ["test", "test", "test", "train", "test"],
            "land_frac": [1.0, 1.0, 0.4, 1.0, 1.0],
            "local_relief": [900.0, 900.0, 900.0, 900.0, 10.0],
        }
        assert probe.select_tile_id(frame, quantile=0.5) == "a"

    def test_is_stable_under_row_order(self):
        frame = {
            "tile_id": ["z", "a"],
            "split": ["test", "test"],
            "land_frac": [1.0, 1.0],
            "local_relief": [900.0, 900.0],
        }
        reversed_frame = {k: list(reversed(v)) for k, v in frame.items()}
        assert probe.select_tile_id(frame, quantile=0.0) == probe.select_tile_id(
            reversed_frame, quantile=0.0
        )

    def test_raises_when_nothing_qualifies(self):
        frame = {"tile_id": ["a"], "split": ["train"], "land_frac": [1.0], "local_relief": [1.0]}
        with pytest.raises(ValueError, match="no tile"):
            probe.select_tile_id(frame, quantile=0.5)


@requires_corpus
class TestAgainstTheRealCorpus:
    def test_shards_hold_raw_metres(self):
        """The assumption the whole normalization rests on.

        If shards held DEMAE-standardized values rather than int16 metres, `terramind_normalize`
        would be mapping already-normalized data through a second affine and every feature in
        the writeup would be computed from nonsense. The manifest's per-tile `mean`/`std` are in
        metres, so agreement between them and the decoded tile pins the storage contract.
        """
        row = probe.manifest_row(CORPUS, probe.PROBE_TILE_ID)
        tile = probe.load_tile_metres(CORPUS, row["shard_id"], row["row_index"])

        assert tile.shape == (224, 224)
        # int16 storage rounds; the tolerance is the rounding, not a fudge factor.
        assert float(tile.mean()) == pytest.approx(row["mean"], abs=0.5)
        assert float(tile.std()) == pytest.approx(row["std"], abs=0.5)
        assert tile.min() >= -500.0 and tile.max() <= 9000.0

    def test_the_committed_tile_is_still_the_one_the_rule_picks(self):
        """A record of a run is only reproducible if its input can be re-derived."""
        assert probe.select_tile_id(probe.read_selection_frame(CORPUS)) == probe.PROBE_TILE_ID


@pytest.mark.skipif(
    not (REPO / "docs" / "research" / "p118-terramind-probe.json").exists(),
    reason="probe has not been run yet",
)
class TestRecordedRun:
    """The recorded run is an artifact of this repo, so its shape claims are pinned like any other.

    #118 asks four shape questions. Answering them in prose only would let the numbers drift out
    of the writeup silently; asserting them against the committed JSON means re-running the probe
    on a newer terratorch and getting a different grid is a test failure, which is exactly the
    signal §2.2's co-registration premise needs.
    """

    @property
    def record(self):
        path = REPO / "docs" / "research" / "p118-terramind-probe.json"
        return json.loads(path.read_text())

    def test_encoder_grid_co_registers_with_demae(self):
        assert self.record["encoder_output_shape"] == [1, 768, 14, 14]

    def test_the_grid_is_interpolated_rather_than_native(self):
        """14x14 holds, and holds by resampling — the caveat A5/A9 have to be read against."""
        record = self.record
        assert record["pos_emb_shape"] == [1, 768, 16, 16]
        assert record["pos_emb_is_interpolated_at_224"] is True

    def test_the_grid_follows_the_input_size(self):
        assert self.record["encoder_output_shape_at_256"] == [1, 768, 16, 16]

    def test_quant_proj_gives_the_five_dim_latent(self):
        assert self.record["quant_proj_output_shape"] == [1, 5, 14, 14]

    def test_twelve_intermediates(self):
        assert self.record["n_intermediates"] == 12
        assert self.record["intermediate_shape"] == [1, 768, 14, 14]

    def test_prepare_input_is_the_no_op_the_memo_says_it_is(self):
        assert self.record["prepare_input_is_identity"] is True

    def test_constants_came_from_the_installed_package(self):
        assert self.record["constants"]["source"] == "terratorch.models.backbones.terramind"
        assert self.record["constants"]["mean"] == 670.665
        assert self.record["constants"]["std"] == 951.272
