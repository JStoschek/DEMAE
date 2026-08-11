"""Guard #140's GSD diagnostic: its four arms, its two statistics, and the record it writes.

The run itself needs terratorch, 1.15 GB of weights, the 47 GB corpus and ~12 MB fetched from
TerraMesh, so it is not reproducible from a fresh clone and nothing here re-runs it. What *is*
checkable without any of that is everything the run's answer depends on:

1. **The arms.** Four inputs that differ in exactly the way the ticket says they differ —
   bandwidth and ground extent — and in nothing else. Each is pinned against an analytic
   expectation (a ramp has a known value at a known pixel; a Nyquist checkerboard has a known
   fate under a 3x round trip) rather than against a previous run's output.
2. **The statistics.** `fsq_assign` re-derives the pre-rounding value that TerraMind's own
   `FiniteScalarQuantizer` throws away, so it has to agree with that module about everything the
   module *does* keep — the code and the index. `CodeUsage` estimates an entropy from 58,800
   draws over 15,360 bins, which is exactly the regime where a plug-in estimator is biased, so
   both the estimator and its correction are checked against closed forms.
3. **The fetch.** The TerraMesh side decodes a zarr-in-zip out of a webdataset tar by hand. The
   test builds one of those by hand too, so the decoder is checked against a store this file
   constructed rather than against a byte blob nobody can read.
4. **The record.** #140 pre-registers a reading rule and forbids a threshold. The tests on the
   JSON assert the numbers the writeup quotes, and assert that no field in the record encodes a
   pass/fail — the one thing the ticket rules out by name.

**This is not `benchmark/` arriving early.** Map #117 puts building that module out of scope; the
script is loaded by file path here and is importable by nothing.
"""

from __future__ import annotations

import importlib.util
import io
import json
import math
import sys
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402  (after the importorskip guard)

REPO = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO / "scripts" / "p140_gsd_quantizer.py"
RECORD_PATH = REPO / "docs" / "research" / "p140-gsd-quantizer.json"
MPS_RECORD_PATH = REPO / "docs" / "research" / "p140-gsd-quantizer-mps.json"


def _load_script():
    spec = importlib.util.spec_from_file_location("p140_gsd_quantizer", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    # `@dataclass` resolves annotations through `sys.modules[cls.__module__]`, so the module has
    # to be registered before it executes — a path-loaded module cannot otherwise define one.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


script = _load_script()


@pytest.fixture(scope="module")
def record():
    return json.loads(RECORD_PATH.read_text())


# --------------------------------------------------------------------------------------
# 1. The arms
# --------------------------------------------------------------------------------------


def _column_ramp(px: int = 224) -> torch.Tensor:
    """`x[0, 0, i, j] == j`. Linear, so bilinear resampling reproduces it exactly."""
    return torch.arange(px, dtype=torch.float32).expand(px, px)[None, None].clone()


def _nyquist_checkerboard(px: int = 224) -> torch.Tensor:
    """+-1 at the pixel Nyquist — the highest frequency a 224 px grid can carry."""
    i = torch.arange(px)
    return ((-1.0) ** (i[:, None] + i[None, :])).to(torch.float32)[None, None]


class TestArms:
    def test_the_four_arms_are_the_ones_the_ticket_names(self):
        arms = script.demae_arms(_column_ramp())
        assert set(arms) == {"native_30m", "upsampled_10m", "bandwidth_matched_30m"}
        assert all(tuple(a.shape) == (1, 1, 224, 224) for a in arms.values())

    def test_native_is_the_tile_untouched(self):
        """The benchmark's own regime is the control, so it must not be resampled at all."""
        x = _column_ramp()
        assert torch.equal(script.demae_arms(x)["native_30m"], x)

    def test_upsampled_10m_is_the_centre_2_24_km_at_10_m(self):
        """The arm's geometry, checked against arithmetic rather than a golden array.

        A 3x bilinear upsample of a 224 px window maps output pixel `k` to input coordinate
        `(k + 0.5)/3 - 0.5`; cropping 224 of the resulting 672 starts at `k = 224`. So the first
        column of the arm sits at input column 74.3333 and the columns step by 1/3 — which is
        what "the centre third of each axis, at 10 m" means in numbers.
        """
        arm = script.demae_arms(_column_ramp())["upsampled_10m"]
        cols = arm[0, 0, 112]  # any row; the ramp varies only along j

        assert float(cols[0]) == pytest.approx((224 + 0.5) / 3 - 0.5, abs=1e-4)
        assert float(cols[-1]) == pytest.approx((447 + 0.5) / 3 - 0.5, abs=1e-4)
        # 224 samples at 1/3 of a native pixel spans 74.667 native px = 2.24 km at 30 m.
        assert float(cols[-1] - cols[0]) == pytest.approx(223 / 3, abs=1e-4)

    def test_upsampled_10m_is_centred_on_the_native_window(self):
        """Centred, not anchored: the arm must not translate the terrain it claims to fix."""
        arm = script.demae_arms(_column_ramp())["upsampled_10m"]
        centre_of_arm = float(arm[0, 0, 112].mean())
        assert centre_of_arm == pytest.approx((224 - 1) / 2, abs=1e-3)

    def test_bandwidth_matched_keeps_the_extent_and_drops_the_detail(self):
        """The added arm's whole purpose: same 6.72 km window, a third of the bandwidth.

        A pixel-Nyquist checkerboard is the signal a 3x down-up round trip is supposed to
        destroy; a linear ramp is the signal it is supposed to leave alone. Both are checked,
        because an arm that flattened everything would pass the first test on its own.

        The checkerboard bound is what caught the first version of this arm. Without
        `antialias=True` on the downsample the checkerboard came through at std 0.129 — folded
        back in rather than removed — and the arm was adding high frequencies where it claimed to
        be taking them away. With the low-pass it lands at 0.0014, a 700x suppression.
        """
        crushed = script.demae_arms(_nyquist_checkerboard())["bandwidth_matched_30m"]
        assert float(crushed.std()) < 0.01, "Nyquist detail survived the bandwidth match"

        ramp = _column_ramp()
        kept = script.demae_arms(ramp)["bandwidth_matched_30m"]
        interior = slice(8, 216)
        assert torch.allclose(
            kept[..., interior, interior], ramp[..., interior, interior], atol=0.01
        ), "a linear ramp should survive a resample that only removes high frequencies"

    def test_bandwidth_matched_reduces_by_the_recorded_factor(self):
        """224 is not divisible by 3, so the factor is 224/75 and the record says so.

        Pinned because the arm's entire claim is that it reproduces TerraMesh's 3:1 ratio of
        source spacing to pixel spacing. 2.9867 is close enough and 2 or 4 would not be, so the
        number belongs in a test rather than in a comment.
        """
        assert script.BANDWIDTH_MATCH_PX == 75
        assert 224 / script.BANDWIDTH_MATCH_PX == pytest.approx(2.9867, abs=1e-3)

    def test_terramesh_arm_is_a_centre_crop_and_nothing_else(self):
        """TerraMesh tiles are already at 10 m, so resampling them would be the confound."""
        tile = torch.arange(264 * 264, dtype=torch.float32).reshape(1, 1, 264, 264)
        arm = script.terramesh_arm(tile)
        assert tuple(arm.shape) == (1, 1, 224, 224)
        assert torch.equal(arm, tile[..., 20:244, 20:244])

    def test_both_10_m_arms_have_the_same_geometry(self):
        """The comparison the ticket rests on: the two references differ in geography only."""
        upsampled = script.demae_arms(_column_ramp())["upsampled_10m"]
        terramesh = script.terramesh_arm(torch.zeros(1, 1, 264, 264))
        assert upsampled.shape == terramesh.shape
        assert script.ARM_GROUND_METRES["upsampled_10m"] == script.ARM_GROUND_METRES["terramesh_10m"]
        assert script.ARM_GROUND_METRES["native_30m"] == 3 * script.ARM_GROUND_METRES["upsampled_10m"]

    def test_normalizing_before_or_after_resampling_is_the_same_thing(self):
        """Why the script is free to normalize once, at one site.

        TerraMind's affine is linear and bilinear resampling is a convex combination, so the two
        commute. That is what lets #118 §7's silent trap have exactly one place to go wrong in
        this script instead of four. Verified rather than argued, because "they commute" is the
        kind of claim that is true until someone adds a clamp.
        """
        metres = torch.randn(1, 1, 224, 224) * 400 + 900

        after = {k: v for k, v in script.demae_arms(metres).items()}
        after = {k: torch.from_numpy(script.p118.terramind_normalize(v.numpy())) for k, v in after.items()}
        before = script.demae_arms(torch.from_numpy(script.p118.terramind_normalize(metres.numpy())))

        for name in after:
            assert torch.allclose(after[name], before[name], atol=1e-5), name


# --------------------------------------------------------------------------------------
# 2. The statistics
# --------------------------------------------------------------------------------------


class _FSQStub(torch.nn.Module):
    """The arithmetic of terratorch's `FiniteScalarQuantizer`, written out independently.

    The point of `fsq_assign` is to recover the pre-rounding value the real module discards, so
    it cannot be checked against the real module's residual — there isn't one. What it *can* be
    checked against is everything the module does return, and this stub is that contract written
    from the source's definition rather than imported from it, so a misreading of the source
    could not validate itself.
    """

    def __init__(self, levels=(8, 8, 8, 6, 5)):
        super().__init__()
        self._levels = torch.tensor(levels, dtype=torch.int)
        self._basis = torch.cumprod(torch.tensor([1, *levels[:-1]]), dim=0, dtype=torch.int)

    def forward(self, x):
        h, w = x.shape[-2:]
        z = x.permute(0, 2, 3, 1)
        d = self._levels - 1
        number = (torch.sigmoid(z) * d).round()
        code = (number / d).permute(0, 3, 1, 2)
        indice = (number * self._basis).sum(dim=-1).to(torch.int)
        return code, torch.tensor([0.0]), indice.reshape(x.shape[0], h, w)


class TestFsqAssign:
    def test_it_agrees_with_the_quantizer_about_code_and_index(self):
        quantizer = _FSQStub()
        latent = torch.randn(3, 5, 14, 14) * 4

        code, _, indices = quantizer(latent)
        assignment = script.fsq_assign(latent, script.fsq_levels(quantizer))

        assert torch.equal(assignment.indices, indices.flatten(1).to(torch.long))
        assert torch.allclose(
            assignment.number / (script.fsq_levels(quantizer) - 1),
            code.permute(0, 2, 3, 1).reshape(3, 196, 5),
            atol=1e-6,
        )

    def test_the_residual_is_the_rounding_error_and_nothing_else(self):
        quantizer = _FSQStub()
        latent = torch.randn(2, 5, 14, 14) * 3
        assignment = script.fsq_assign(latent, script.fsq_levels(quantizer))

        d = (script.fsq_levels(quantizer) - 1).to(torch.float32)
        expected = torch.sigmoid(latent).permute(0, 2, 3, 1).reshape(2, 196, 5) * d
        assert torch.allclose(assignment.residual, expected - assignment.number, atol=1e-6)
        assert float(assignment.residual.abs().max()) <= 0.5 + 1e-6

    def test_a_latent_already_on_the_grid_has_no_residual(self):
        """The zero the statistic has to be able to reach, or it is not measuring distance."""
        levels = script.fsq_levels(_FSQStub())
        d = (levels - 1).to(torch.float32)
        # Interior levels only — 1..d-1. The terminal levels are exactly where `sigmoid` cannot
        # be inverted, and they are the subject of the saturation tests rather than this one.
        number = torch.stack(
            [1 + torch.arange(196) % (int(k) - 1) for k in d], dim=-1
        ).to(torch.float32)
        latent = torch.logit(number / d)[None].reshape(1, 14, 14, 5).permute(0, 3, 1, 2)

        assignment = script.fsq_assign(latent, levels)
        assert float(assignment.residual.abs().max()) < 1e-4

    def test_saturation_counts_the_terminal_bins(self):
        """A residual near zero can mean "well represented" or "squashed flat against an end".

        The two are opposite readings and `sigmoid` makes the second one free, so the fraction
        of assignments in a terminal level is reported beside the residual. This pins that it
        counts both ends and only the ends.
        """
        levels = torch.tensor([8, 8, 8, 6, 5], dtype=torch.int)
        # dim 0 pinned to level 0, dim 1 to the top level, dims 2-4 to interior levels
        number = torch.zeros(1, 4, 5)
        number[..., 1] = 7
        number[..., 2] = 3
        number[..., 3] = 2
        number[..., 4] = 1
        assert script.saturated_fraction(number, levels) == pytest.approx(2 / 5)

    def test_saturation_is_reported_per_dimension_too(self):
        levels = torch.tensor([8, 8, 8, 6, 5], dtype=torch.int)
        number = torch.zeros(1, 10, 5)
        number[:, :5, 0] = 7  # half of dim 0's tokens at the top level
        per_dim = script.saturated_fraction_per_dim(number, levels)
        assert per_dim == pytest.approx([1.0, 1.0, 1.0, 1.0, 1.0])  # zeros are terminal too

        number[:, :, 1:] = 1.0  # an interior level for every other dimension
        per_dim = script.saturated_fraction_per_dim(number, levels)
        assert per_dim == pytest.approx([1.0, 0.0, 0.0, 0.0, 0.0])


class TestCodeUsage:
    def test_a_uniform_spread_reaches_the_ceiling(self):
        usage = script.CodeUsage(script.CODEBOOK_LEVELS)
        usage.update(torch.arange(script.CODEBOOK_SIZE)[None], numbers=None)
        assert usage.n_codes_used == script.CODEBOOK_SIZE
        assert usage.entropy_bits() == pytest.approx(script.ENTROPY_CEILING_BITS, abs=1e-9)

    def test_one_code_is_zero_bits(self):
        usage = script.CodeUsage(script.CODEBOOK_LEVELS)
        usage.update(torch.full((1, 500), 77), numbers=None)
        assert usage.n_codes_used == 1
        assert usage.entropy_bits() == pytest.approx(0.0, abs=1e-12)

    def test_it_matches_the_plug_in_formula_on_a_known_mix(self):
        counts = {3: 50, 11: 30, 12: 20}
        draws = torch.tensor([k for k, n in counts.items() for _ in range(n)])[None]
        usage = script.CodeUsage(script.CODEBOOK_LEVELS)
        usage.update(draws, numbers=None)

        p = np.array([50, 30, 20]) / 100
        assert usage.n == 100
        assert usage.entropy_bits() == pytest.approx(float(-(p * np.log2(p)).sum()), abs=1e-12)
        assert usage.perplexity() == pytest.approx(2 ** usage.entropy_bits(), rel=1e-12)

    def test_streaming_in_ragged_batches_gives_the_same_answer(self):
        rng = np.random.default_rng(0)
        draws = rng.integers(0, script.CODEBOOK_SIZE, size=5000)

        whole = script.CodeUsage(script.CODEBOOK_LEVELS)
        whole.update(torch.from_numpy(draws)[None], numbers=None)

        streamed = script.CodeUsage(script.CODEBOOK_LEVELS)
        for lo in range(0, 5000, 137):
            streamed.update(torch.from_numpy(draws[lo : lo + 137])[None], numbers=None)

        assert streamed.n == whole.n
        assert streamed.entropy_bits() == pytest.approx(whole.entropy_bits(), rel=1e-12)

    def test_the_miller_madow_correction_is_the_textbook_one(self):
        """58,800 draws over 15,360 bins is where a plug-in entropy is biased low, so the
        correction is reported beside it. `(K_used - 1) / (2 n ln 2)`, from the definition."""
        counts = {3: 50, 11: 30, 12: 20}
        draws = torch.tensor([k for k, n in counts.items() for _ in range(n)])[None]
        usage = script.CodeUsage(script.CODEBOOK_LEVELS)
        usage.update(draws, numbers=None)

        expected = usage.entropy_bits() + (3 - 1) / (2 * 100 * math.log(2))
        assert usage.entropy_bits_miller_madow() == pytest.approx(expected, rel=1e-12)

    def test_the_correction_never_exceeds_the_ceiling(self):
        """Miller-Madow is a bias correction, not a bound, and it can overshoot log2(K).

        It is clamped, because an entropy above the ceiling would be reported in the writeup as
        a codebook using more than all of itself.
        """
        usage = script.CodeUsage(script.CODEBOOK_LEVELS)
        usage.update(torch.arange(script.CODEBOOK_SIZE)[None], numbers=None)
        assert usage.entropy_bits_miller_madow() == pytest.approx(script.ENTROPY_CEILING_BITS, abs=1e-9)

    def test_per_dimension_marginals_bound_the_joint(self):
        """Sub-additivity, which is what makes the marginals worth reporting.

        If the joint entropy collapses, the marginals say *which* of the five axes collapsed;
        the bound is what guarantees they are talking about the same thing.
        """
        rng = np.random.default_rng(1)
        levels = np.array(script.CODEBOOK_LEVELS)
        numbers = torch.from_numpy(
            np.stack([rng.integers(0, k, size=4000) for k in levels], axis=-1)
        ).to(torch.float32)[None]
        basis = np.cumprod([1, *levels[:-1]])
        indices = torch.from_numpy((numbers[0].numpy() * basis).sum(-1).astype(np.int64))[None]

        usage = script.CodeUsage(script.CODEBOOK_LEVELS)
        usage.update(indices, numbers=numbers)

        marginals = usage.entropy_bits_per_dim()
        assert len(marginals) == 5
        assert usage.entropy_bits() <= sum(marginals) + 1e-9
        for bits, k in zip(marginals, levels):
            assert bits <= math.log2(k) + 1e-9

    def test_the_ceiling_is_the_codebook_the_tokenizer_actually_has(self):
        assert math.prod(script.CODEBOOK_LEVELS) == script.CODEBOOK_SIZE == 15360
        assert script.ENTROPY_CEILING_BITS == pytest.approx(13.9069, abs=5e-5)


class TestBootstrap:
    @staticmethod
    def _tiles(rng, n_tiles=40, tokens=196, n_codes=200):
        return rng.integers(0, n_codes, size=(n_tiles, tokens)).astype(np.int16)

    def test_identical_tiles_have_no_spread(self):
        """The degenerate case that says the resample is over tiles and not over noise."""
        one = np.arange(196, dtype=np.int16)
        tiles = np.repeat(one[None], 30, axis=0)
        replicates = script.bootstrap_entropies({"a": tiles}, [], n_boot=50, seed=0)
        assert replicates["a"].std() == pytest.approx(0.0, abs=1e-12)

    def test_a_paired_group_shares_its_tile_draw(self):
        """Why the DEMAE arms are grouped: their differences must not carry terrain.

        Two arms with *identical* per-tile codes are the sharpest possible statement of it — if
        they were resampled independently, every replicate would differ and the difference's
        interval would be spuriously wide.
        """
        rng = np.random.default_rng(0)
        tiles = self._tiles(rng)
        paired = script.bootstrap_entropies({"a": tiles, "b": tiles}, [("a", "b")], n_boot=64, seed=3)
        np.testing.assert_allclose(paired["a"], paired["b"], rtol=0, atol=0)

        apart = script.bootstrap_entropies({"a": tiles, "b": tiles}, [], n_boot=64, seed=3)
        assert not np.allclose(apart["a"], apart["b"])

    def test_the_replicates_sit_just_below_the_point_estimate(self):
        """A property of the estimator, pinned so the writeup reads the intervals correctly.

        A bootstrap draw of n tiles from n contains only ~63% distinct tiles, so it sees fewer
        distinct codes, so its plug-in entropy is biased *low* against the point estimate. The
        offset is small and it is common to every arm, which is why the record's entropy
        intervals are for **differences** — where it cancels — and why the absolute
        `entropy_bits_bootstrap_mean` is reported beside the point estimate rather than instead
        of it.
        """
        rng = np.random.default_rng(1)
        tiles = self._tiles(rng, n_tiles=80)
        point = script.entropy_bits_from_counts(
            np.bincount(tiles.ravel(), minlength=script.CODEBOOK_SIZE)
        )
        replicates = script.bootstrap_entropies({"a": tiles}, [], n_boot=300, seed=7)

        offset = replicates["a"].mean() - point
        assert offset < 0, "resampling tiles with replacement cannot raise the plug-in entropy"
        assert abs(offset) < 0.05, "the offset should be small against any difference being read"

    def test_the_offset_cancels_out_of_a_paired_difference(self):
        """What the intervals are actually used for, checked directly.

        Two arms whose codes differ by a fixed relabelling have identical entropies. If the
        bootstrap's downward offset did not cancel, their measured difference would drift off
        zero replicate by replicate; it does not.
        """
        rng = np.random.default_rng(5)
        tiles = self._tiles(rng, n_tiles=60, n_codes=300)
        relabelled = (tiles.astype(np.int32) + 1000).astype(np.int16)

        replicates = script.bootstrap_entropies(
            {"a": tiles, "b": relabelled}, [("a", "b")], n_boot=80, seed=13
        )
        difference = replicates["b"] - replicates["a"]
        assert np.abs(difference).max() < 1e-12

    def test_it_is_reproducible_from_the_seed(self):
        rng = np.random.default_rng(2)
        tiles = self._tiles(rng)
        first = script.bootstrap_entropies({"a": tiles}, [], n_boot=32, seed=11)
        again = script.bootstrap_entropies({"a": tiles}, [], n_boot=32, seed=11)
        np.testing.assert_allclose(first["a"], again["a"], rtol=0, atol=0)

    def test_the_interval_brackets_the_replicates(self):
        samples = np.linspace(0.0, 1.0, 1001)
        interval = script.bootstrap_interval(samples)
        assert interval["ci95"][0] == pytest.approx(0.025, abs=1e-3)
        assert interval["ci95"][1] == pytest.approx(0.975, abs=1e-3)
        assert interval["sd"] == pytest.approx(samples.std(ddof=1))

    def test_entropy_from_counts_matches_the_accumulator(self):
        counts = np.zeros(script.CODEBOOK_SIZE, dtype=np.int64)
        counts[[3, 11, 12]] = [50, 30, 20]
        usage = script.CodeUsage(script.CODEBOOK_LEVELS)
        usage.update(torch.tensor([k for k, n in zip([3, 11, 12], [50, 30, 20]) for _ in range(n)])[None], None)
        assert script.entropy_bits_from_counts(counts) == pytest.approx(usage.entropy_bits(), rel=1e-12)


class TestPairedDifference:
    def test_it_reports_the_mean_its_spread_and_an_interval(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([1.5, 2.0, 3.5, 5.0])
        out = script.paired_difference(a, b)

        diff = b - a
        assert out["mean"] == pytest.approx(diff.mean())
        assert out["sd"] == pytest.approx(diff.std(ddof=1))
        assert out["se"] == pytest.approx(diff.std(ddof=1) / 2)
        assert out["ci95"][0] == pytest.approx(diff.mean() - 1.959964 * out["se"], rel=1e-5)
        assert out["ci95"][1] == pytest.approx(diff.mean() + 1.959964 * out["se"], rel=1e-5)

    def test_it_refuses_unpaired_inputs(self):
        """The whole point of the paired form is that the two arms are the same 300 tiles."""
        with pytest.raises(ValueError, match="paired"):
            script.paired_difference(np.zeros(4), np.zeros(5))


# --------------------------------------------------------------------------------------
# 3. The fetch
# --------------------------------------------------------------------------------------


def _zarr_zip(metres: np.ndarray, lat: float, lon: float, sample: str) -> bytes:
    """A minimal `*.zarr.zip` in TerraMesh's own layout, built here rather than downloaded.

    Only the four entries the decoder reads are written. That is deliberate: if the decoder ever
    starts depending on something else in the store, this fixture stops supplying it and the test
    fails, rather than the decoder silently acquiring a dependency on a real download.
    """
    numcodecs = pytest.importorskip("numcodecs")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "bands/.zarray",
            json.dumps(
                {
                    "chunks": [1, 1, *metres.shape],
                    "compressor": {"blocksize": 0, "clevel": 5, "cname": "zstd", "id": "blosc", "shuffle": 1},
                    "dtype": "<i2",
                    "fill_value": None,
                    "filters": None,
                    "order": "C",
                    "shape": [1, 1, *metres.shape],
                    "zarr_format": 2,
                }
            ),
        )
        zf.writestr("bands/0.0.0.0", numcodecs.Blosc().encode(metres.astype("<i2").tobytes()))
        zf.writestr("center_lat/0", np.array([lat], dtype="<f8").tobytes())
        zf.writestr("center_lon/0", np.array([lon], dtype="<f8").tobytes())
        zf.writestr("sample/0", np.array([sample], dtype=f"<U{len(sample)}").tobytes())
    return buf.getvalue()


def _tar_of(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))
    return buf.getvalue()


class TestTerraMeshDecode:
    def test_it_reads_the_elevation_and_the_centroid_back_out(self):
        metres = (np.arange(264 * 264).reshape(264, 264) % 3000 - 500).astype(np.int16)
        tile = script.decode_dem_zarr_zip(_zarr_zip(metres, 41.669, 140.328, "463U_1168R_2_1"))

        assert tile.metres.dtype == np.int16
        assert tile.metres.shape == (264, 264)
        np.testing.assert_array_equal(tile.metres, metres)
        assert tile.lat == pytest.approx(41.669)
        assert tile.lon == pytest.approx(140.328)
        assert tile.sample_id == "463U_1168R_2_1"

    def test_a_tile_that_is_not_264_px_is_an_error_not_a_reshape(self):
        """TerraMesh's patch size is 264 and the arm's crop assumes it. A store that says
        otherwise is a change in the dataset, and it should stop the run rather than be padded."""
        metres = np.zeros((256, 256), dtype=np.int16)
        with pytest.raises(ValueError, match="264"):
            script.decode_dem_zarr_zip(_zarr_zip(metres, 0.0, 0.0, "x"))

    def test_members_come_out_of_a_truncated_tar_stream(self):
        """The fetch range-reads a prefix of a 388 MB shard, so the tar *will* end mid-member.

        That has to be an ordinary stop, not an exception: the whole design is "take the first k
        and hang up".
        """
        payloads = {f"majortom_val_{i:07d}.zarr.zip": bytes([i]) * 5000 for i in range(1, 7)}
        truncated = _tar_of(payloads)[:14000]

        got = list(script.iter_tar_members(truncated))
        assert 1 <= len(got) < 6
        assert got[0][0] == "majortom_val_0000001.zarr.zip"
        assert all(len(payload) == 5000 for _, payload in got)

    def test_round_robin_deals_one_from_each_stream_in_turn(self):
        """Why the sample is not the head of one shard: it is a slice across all nine."""
        streams = [iter(range(100, 110)), iter(range(200, 210)), iter(range(300, 310))]
        assert script.round_robin(streams, 7) == [100, 200, 300, 101, 201, 301, 102]

    def test_round_robin_drains_short_streams_without_stalling(self):
        streams = [iter([1]), iter([2, 3, 4]), iter([])]
        assert script.round_robin(streams, 10) == [1, 2, 3, 4]

    def test_the_sample_is_drawn_from_every_val_dem_shard(self):
        """Nine shards, eight MajorTOM and one SSL4EO-S12 — TerraMesh's own val composition."""
        assert len(script.TERRAMESH_VAL_DEM_SHARDS) == 9
        assert sum("ssl4eos12" in s for s in script.TERRAMESH_VAL_DEM_SHARDS) == 1
        assert all(s.startswith("val/DEM/") for s in script.TERRAMESH_VAL_DEM_SHARDS)


# --------------------------------------------------------------------------------------
# 4. The record, and the rule it is read under
# --------------------------------------------------------------------------------------


class TestRecord:
    def test_it_is_the_run_the_ticket_asked_for(self, record):
        assert record["issue"] == 140
        assert record["model"] == "terramind_v1_tokenizer_dem"
        assert record["n_tiles_per_arm"] == 300
        assert record["tokens_per_tile"] == 196
        assert set(record["arms"]) == {
            "native_30m",
            "upsampled_10m",
            "bandwidth_matched_30m",
            "terramesh_10m",
        }
        for name, arm in record["arms"].items():
            assert arm["n_tokens"] == 300 * 196, name

    def test_every_arm_saw_the_same_number_of_tokens(self, record):
        """The design constraint the entropy comparison rests on.

        A plug-in entropy over 15,360 bins from 58,800 draws is biased low, and the bias depends
        on the sample size. Matching `n` across arms is what makes the *differences* readable
        without trusting the correction — so if the arms ever stop being matched, the comparison
        needs redoing and this test is what says so.
        """
        assert len({arm["n_tokens"] for arm in record["arms"].values()}) == 1

    def test_the_run_used_terramind_s_global_normalization(self, record):
        """#118 §7's silent failure, re-guarded on both corpora."""
        assert record["normalization"] == "terramind_global_z_score"
        assert record["constants"]["mean"] == script.p118.TERRAMIND_DEM_MEAN
        assert record["constants"]["std"] == script.p118.TERRAMIND_DEM_STD
        assert record["constants"]["matches_this_script"] is True

    def test_the_residual_derivation_was_checked_against_the_quantizer(self, record):
        """`fsq_assign` re-derives a value the module discards. The run checks the half it
        cannot discard — the code and the index — on every batch, and records the result."""
        assert record["quantizer"]["levels"] == [8, 8, 8, 6, 5]
        assert record["quantizer"]["codebook_size"] == 15360
        assert record["quantizer"]["parameters"] == 0
        assert record["quantizer"]["code_matches_module"] is True
        assert record["quantizer"]["index_matches_module"] is True

    def test_no_field_encodes_a_pass_or_a_fail(self, record):
        """#140's own instruction, made structural.

        "No threshold and no drop condition — deliberately unlike #134." #134's record carries
        `threshold`, `verdict` and `ablation_row`; this one must not, because these statistics
        have no familiar scale and any cutoff would be invented to look principled.
        """
        forbidden = {"threshold", "verdict", "ablation_row", "passes", "material", "immaterial"}
        assert not forbidden & set(record)
        for arm in record["arms"].values():
            assert not forbidden & set(arm)

    def test_the_limit_is_carried_in_the_record_not_only_in_the_prose(self, record):
        """"The limit is reported wherever the number is" — so it travels with the JSON."""
        assert "limit" in record and "reading_rule" in record
        assert "122" in record["reading_rule"]
        assert record["measures"] and record["does_not_measure"]

    def test_the_terramesh_sample_is_global_and_licence_clean(self, record):
        """The hole the second reference exists to close, checked rather than assumed.

        TerraMesh is global and this benchmark's tiles are strata-selected, so the reference only
        does its job if the draw actually spans the globe — a sample that landed in one region
        would silently turn "varies geography" into "varies one region".
        """
        mesh = record["terramesh"]
        assert mesh["repo_id"] == "ibm-esa-geospatial/TerraMesh"
        assert mesh["licence"] == "cc-by-sa-4.0"
        assert mesh["gated"] is False
        assert mesh["split"] == "val"
        assert mesh["n_tiles"] == 300
        assert mesh["n_shards"] == 9
        assert mesh["tiles_committed"] == 0, "#128: statistics only, never tiles"

        lat_lo, lat_hi = mesh["lat_range"]
        lon_lo, lon_hi = mesh["lon_range"]
        assert lat_hi - lat_lo > 90
        assert lon_hi - lon_lo > 300

    def test_the_demae_sample_is_the_pre_registered_one(self, record):
        """The same 300 tiles #134 used, so the two DEMAE arms are paired and comparable."""
        assert record["demae"]["tiles_per_stratum"] == 15
        assert record["demae"]["n_strata"] == 20
        assert record["demae"]["selection_rule"] == "p134"
        assert record["demae"]["n_source_cogs"] > 150

    def test_every_arm_reports_both_statistics_over_the_full_sample(self, record):
        for name, arm in record["arms"].items():
            residual = arm["residual"]
            assert residual["min"] <= residual["p01"] <= residual["mean"], name
            assert residual["mean"] <= residual["p99"] <= residual["max"], name
            assert 0.0 <= residual["max"] <= math.sqrt(5) / 2 + 1e-6, name
            assert len(residual["mean_per_dim"]) == 5, name
            assert len(arm["saturated_fraction_per_dim"]) == 5, name

            usage = arm["code_usage"]
            assert 1 <= usage["n_codes_used"] <= script.CODEBOOK_SIZE, name
            assert 0.0 <= usage["entropy_bits"] <= script.ENTROPY_CEILING_BITS, name
            assert usage["entropy_bits_miller_madow"] >= usage["entropy_bits"], name
            assert usage["entropy_bits_miller_madow"] <= script.ENTROPY_CEILING_BITS + 1e-9, name
            assert len(usage["entropy_bits_per_dim"]) == 5, name

    def test_every_entropy_comes_with_a_resampled_interval(self, record):
        """The entropy is one number over pooled tokens, so its spread has to be resampled.

        Over **tiles**: 196 tokens of one mountain range share a code distribution, so a
        token-level resample would report an interval several times too narrow.
        """
        assert record["bootstrap"]["resampled"] == "tiles"
        assert record["bootstrap"]["n_replicates"] >= 1000

        for name, arm in record["arms"].items():
            interval = arm["code_usage"]["entropy_bits_bootstrap"]["ci95"]
            assert interval[0] < arm["code_usage"]["entropy_bits"] < interval[1], name
            # Biased low by the ~37% of tiles a resample omits — small, common to every arm,
            # and the reason the intervals below are read on differences and not on levels.
            assert arm["code_usage"]["entropy_bits_bootstrap_mean"] < arm["code_usage"]["entropy_bits"]

    def test_the_headline_numbers_the_writeup_quotes(self, record):
        """Pinned so the prose cannot drift from the JSON silently — #118's and #134's precedent."""
        arms = record["arms"]
        assert arms["native_30m"]["residual"]["mean"] == pytest.approx(0.4660, abs=5e-4)
        assert arms["bandwidth_matched_30m"]["residual"]["mean"] == pytest.approx(0.4662, abs=5e-4)
        assert arms["upsampled_10m"]["residual"]["mean"] == pytest.approx(0.4749, abs=5e-4)
        assert arms["terramesh_10m"]["residual"]["mean"] == pytest.approx(0.4676, abs=5e-4)

        assert arms["native_30m"]["code_usage"]["entropy_bits"] == pytest.approx(7.140, abs=5e-3)
        assert arms["bandwidth_matched_30m"]["code_usage"]["entropy_bits"] == pytest.approx(7.131, abs=5e-3)
        assert arms["upsampled_10m"]["code_usage"]["entropy_bits"] == pytest.approx(6.808, abs=5e-3)
        assert arms["terramesh_10m"]["code_usage"]["entropy_bits"] == pytest.approx(7.123, abs=5e-3)

        for arm in arms.values():
            assert arm["saturated_fraction"] == pytest.approx(0.68, abs=0.02)

    def test_native_30_m_is_not_the_off_distribution_arm(self, record):
        """#140's actual question, reduced to the two comparisons that answer it.

        Native 30 m sits inside TerraMesh's own val distribution on both statistics — the
        entropy gap's interval spans zero, and the residual is *smaller* than TerraMesh's. If a
        rerun ever inverted this, §2.2's third bullet would be stating the opposite of its
        evidence, which is precisely what pre-registering the measurement was for.
        """
        arms = record["arms"]
        gap = record["entropy_difference"]["native_30m_vs_terramesh_10m"]
        assert gap["ci95"][0] < 0 < gap["ci95"][1], "the native-vs-TerraMesh entropy gap resolved"
        assert arms["native_30m"]["residual"]["mean"] < arms["terramesh_10m"]["residual"]["mean"]

    def test_bandwidth_is_the_smallest_effect_measured(self, record):
        """What the fourth arm was added to establish, stated as the comparison it rests on.

        `bandwidth_matched_30m` holds the ground extent and moves only the source-to-pixel
        spacing ratio to TerraMesh's 3:1. Against it, the two things the ticket's references
        confound bandwidth with — ground extent and geography — are both an order of magnitude
        larger.
        """
        bandwidth = abs(record["entropy_difference"]["native_30m_vs_bandwidth_matched_30m"]["point"])
        extent = abs(record["entropy_difference"]["upsampled_10m_vs_bandwidth_matched_30m"]["point"])
        geography = abs(record["entropy_difference"]["upsampled_10m_vs_terramesh_10m"]["point"])

        assert bandwidth < extent / 10
        assert bandwidth < geography / 10
        assert bandwidth < 0.01 * record["entropy_ceiling_bits"]

        residual = record["paired_residual"]["native_30m_vs_bandwidth_matched_30m"]
        assert residual["ci95"][0] < 0 < residual["ci95"][1], "the bandwidth residual shift resolved"

    def test_the_codebook_is_barely_used_by_any_arm(self, record):
        """Context without which every number above reads larger than it is.

        Every arm — TerraMesh's own tiles included — reaches about 7.1 of 13.91 bits and roughly
        140 effective codes of 15,360, with two thirds of latent dimensions pinned to a terminal
        level. That is a property of the DEM tokenizer, not of this benchmark's inputs, and it
        is the reason no arm's absolute number is interpretable on its own.
        """
        for name, arm in record["arms"].items():
            usage = arm["code_usage"]
            assert usage["entropy_bits"] < 0.55 * record["entropy_ceiling_bits"], name
            assert usage["perplexity"] < 200, name
            assert usage["n_codes_used"] < 0.05 * script.CODEBOOK_SIZE, name
            assert arm["saturated_fraction"] > 0.5, name

    def test_the_encoder_only_claim_is_counted_not_quoted(self, record):
        """#140 says "encoder-only: 90.2M of the 1.15 GB checkpoint". Counted, and matching #118.

        The count is taken before anything installs a `pos_emb`; the control pass below swaps a
        16x16 embedding for a 14x14 one, which is 46,080 parameters, and a count taken afterwards
        would silently disagree with #118 §8 by exactly that.
        """
        assert record["quantizer"]["encoder_parameters"] == 90_172_416
        assert record["quantizer"]["quant_proj_parameters"] == 3845
        assert record["quantizer"]["decoder_parameters_unused"] == 196_140_560

    def test_the_adr_0011_departure_is_measured_not_argued(self, record):
        """The one ADR this run does not follow, and the second pass that prices it.

        ADR-0011 Decision 2 replaces terratorch's bicubic 16->14 `pos_emb` resize with the
        centre-anchored crop "on every frozen-encoder pass", and #134 measured the two feature
        sets 0.9708 apart. These arms drive the checkpoint's own path instead, because
        "off-distribution for this tokenizer" is a question about the tokenizer as released — so
        every arm is encoded a second time under ADR-0011's embedding.

        What must hold is not that the levels agree. They do not: the control sits ~0.10 bits
        higher on every arm. What must hold is that the **differences** — the only thing this
        record reads — survive the choice.
        """
        control = record["posemb_control"]
        assert control["adr"] == "0011"
        assert set(control["arms"]) == set(record["arms"])

        # The embedding moves every arm's level, and by more than some of the differences read
        # off them. That is the reason it is priced rather than assumed away.
        assert control["max_abs_shift_in_residual_mean"] > 0.005

        for name, arm in record["arms"].items():
            ckpt = arm["code_usage"]["entropy_bits"]
            assert control["arms"][name]["entropy_bits"] > ckpt, name

        # Every conclusion §7 of the memo draws, re-derived under the other embedding.
        diffs = control["entropy_difference"]
        bandwidth = abs(diffs["native_30m_vs_bandwidth_matched_30m"])
        assert bandwidth < abs(diffs["upsampled_10m_vs_bandwidth_matched_30m"]) / 10
        assert bandwidth < abs(diffs["upsampled_10m_vs_terramesh_10m"]) / 10

        # The native-vs-TerraMesh gap changes sign between the embeddings — which is the sharpest
        # possible statement that it is indistinguishable from zero, since it is smaller than the
        # influence of a choice neither arm is affected by differently. Both readings sit well
        # inside the bootstrap interval measured on the checkpoint path.
        gap = record["entropy_difference"]["native_30m_vs_terramesh_10m"]
        assert gap["ci95"][0] < diffs["native_30m_vs_terramesh_10m"] < gap["ci95"][1]

    def test_the_mps_run_agrees_with_the_cpu_run(self, record):
        """Both devices committed, as #118 did — and here the agreement is not a formality.

        `round()` is discontinuous, so a token whose pre-rounding value sits near a half-integer
        can be assigned a different code on a different device, and both statistics are built on
        that assignment. Measured rather than assumed: the two runs land within 2e-4 bits and
        pick the same number of distinct codes.
        """
        mps = json.loads(MPS_RECORD_PATH.read_text())
        assert record["device"] == "cpu" and mps["device"] == "mps"

        for name, arm in record["arms"].items():
            other = mps["arms"][name]
            assert other["residual"]["mean"] == pytest.approx(arm["residual"]["mean"], abs=1e-5), name
            assert other["code_usage"]["entropy_bits"] == pytest.approx(
                arm["code_usage"]["entropy_bits"], abs=2e-4
            ), name
            assert other["code_usage"]["n_codes_used"] == arm["code_usage"]["n_codes_used"], name

    def test_the_uniform_scale_marker_is_a_scale_and_not_a_cutoff(self, record):
        """`sqrt(5/12)` is what the residual would be if the latents ignored the grid entirely.

        Recorded so a reader can place the measured numbers on a scale that has none, and
        labelled in the record itself as carrying no decision — the distinction #140 draws
        between a reference and a threshold.
        """
        assert record["uniform_residual_rms"] == pytest.approx(math.sqrt(5 / 12), abs=1e-9)
        assert "no decision" in record["uniform_residual_rms_note"].lower()
