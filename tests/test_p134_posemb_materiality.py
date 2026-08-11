"""Guard #134's materiality check: its mechanism, its statistics, and the number it turns on.

The run itself needs terratorch, 1.15 GB of weights and the 47 GB corpus, so it is not
reproducible from a fresh clone and nothing here re-runs it. What *is* checkable without any of
that is everything the run's answer depends on:

1. **The mechanism.** All three arms differ in exactly one tensor. That rests on a fact about
   `F.interpolate`, not on a promise: terratorch's `ViTEncoder.forward` resizes `pos_emb` to the
   grid the input implies on *every* pass, so installing an already-14x14 embedding leaves a
   14->14 bicubic resize in the path. `test_installing_a_14x14_posemb_makes_the_resize_a_no_op`
   is the test that fails if that ever stops being bit-identity — at which point the "elementwise,
   no resampling confound" claim in the ticket would be false for two of the three arms.
2. **The statistics.** `TokenMoments` computes linear CKA from streamed cross-products rather
   than from a materialized (n x 768) matrix, so it is arithmetic worth checking against the
   textbook form. The reference implementations below are written out independently — a
   streaming bug that also lived in the reference would otherwise validate itself.
3. **The decision.** #134's threshold was pre-registered in ADR-0011 *before* the numbers
   existed, so the one thing that must not drift is the tie between the recorded cosine and the
   verdict written into the plan. `test_verdict_follows_from_the_recorded_number` recomputes the
   rule from the JSON rather than trusting the `verdict` field.

**This is not `benchmark/` arriving early.** Map #117 puts building that module out of scope; the
script is loaded by file path here and is importable by nothing.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402  (after the importorskip guard)

REPO = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO / "scripts" / "p134_posemb_materiality.py"
RECORD_PATH = REPO / "docs" / "research" / "p134-posemb-materiality.json"

#: ADR-0011 §Decision 5, fixed before any of these numbers existed.
PREREGISTERED_THRESHOLD = 0.99


def _load_script():
    spec = importlib.util.spec_from_file_location("p134_posemb_materiality", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load_script()


@pytest.fixture(scope="module")
def record():
    return json.loads(RECORD_PATH.read_text())


try:
    CORPUS = script.p118.find_corpus(REPO)
except FileNotFoundError:
    CORPUS = None

requires_corpus = pytest.mark.skipif(CORPUS is None, reason="data/corpus_500k not present")


# --------------------------------------------------------------------------------------
# 1. The mechanism: three arms, one tensor apart
# --------------------------------------------------------------------------------------


class TestVariants:
    def test_the_three_arms_are_the_ones_the_ticket_names(self):
        native = script.p133.build_2d_sincos_posemb(16, 16)
        variants = script.posemb_variants(native)

        assert set(variants) == {"bicubic", "centre", "top_left"}
        assert all(tuple(v.shape) == (1, 768, 14, 14) for v in variants.values())

        # centre is ADR-0011's spec, top-left is the corner crop, bicubic is the library default.
        assert torch.equal(variants["centre"], native[:, :, 1:15, 1:15])
        assert torch.equal(variants["top_left"], native[:, :, :14, :14])
        assert torch.equal(
            variants["bicubic"],
            F.interpolate(native, size=(14, 14), mode="bicubic", align_corners=False),
        )

    def test_top_left_is_the_rederivation_bit_for_bit(self):
        """#133's claim 1, restated here because #134's third arm *is* that claim run forward.

        If this ever fails, "top-left" stops being "the embedding terratorch would build for a
        14x14 grid" and the third arm loses its meaning.
        """
        native = script.p133.build_2d_sincos_posemb(16, 16)
        rederived = script.p133.build_2d_sincos_posemb(14, 14)
        assert torch.equal(script.posemb_variants(native)["top_left"], rederived)

    def test_installing_a_14x14_posemb_makes_the_resize_a_no_op(self):
        """The load-bearing fact: two of the three arms still go through terratorch's resize.

        `ViTEncoder.forward` calls `F.interpolate(self.pos_emb, size=(N_H, N_W), ...)`
        unconditionally, so the centre and top-left arms are resized 14->14 before use. With
        `align_corners=False` the sample index is `(i + 0.5) * 1 - 0.5 == i`, exactly, so the
        bicubic kernel lands on integer offsets and the resize returns its input bit-for-bit.
        Verified rather than argued, in both dtypes that appear in the run.
        """
        for dtype in (torch.float32, torch.float64):
            x = torch.randn(1, 768, 14, 14, dtype=dtype)
            assert script.resize_error_at(x) == 0.0, f"14->14 bicubic is not identity in {dtype}"

    def test_the_variants_keep_the_checkpoints_own_dtype(self):
        """Built in the checkpoint's float32, so the bicubic arm is the library's own arithmetic.

        Computing it in float64 and casting would be *more* accurate and therefore less faithful:
        the arm is supposed to be what terratorch does, not a better version of it.
        """
        for dtype in (torch.float32, torch.float64):
            native = script.p133.build_2d_sincos_posemb(16, 16).to(dtype)
            assert all(v.dtype == dtype for v in script.posemb_variants(native).values())


# --------------------------------------------------------------------------------------
# 2. The statistics
# --------------------------------------------------------------------------------------


def _reference_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Textbook linear CKA on materialized matrices (Kornblith et al. 2019, eq. 6-7).

    Deliberately written from the definition and against materialized arrays — the whole point of
    `TokenMoments` is that it never materializes these, so a shared bug is not possible.
    """
    xc = x - x.mean(axis=0, keepdims=True)
    yc = y - y.mean(axis=0, keepdims=True)
    hsic_xy = np.linalg.norm(xc.T @ yc, ord="fro") ** 2
    hsic_xx = np.linalg.norm(xc.T @ xc, ord="fro") ** 2
    hsic_yy = np.linalg.norm(yc.T @ yc, ord="fro") ** 2
    return float(hsic_xy / np.sqrt(hsic_xx * hsic_yy))


class TestTokenMoments:
    def test_matches_the_textbook_form_when_streamed_in_batches(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(120, 17))
        y = 0.3 * x + rng.normal(size=(120, 17))

        moments = script.TokenMoments(("a", "b"), dim=17)
        for lo in range(0, 120, 7):  # ragged batches, so the running mean is actually exercised
            moments.update({"a": torch.from_numpy(x[lo : lo + 7]), "b": torch.from_numpy(y[lo : lo + 7])})

        assert moments.n == 120
        assert moments.cka("a", "b") == pytest.approx(_reference_cka(x, y), rel=1e-10)

    def test_a_representation_is_perfectly_similar_to_itself(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(64, 9))
        moments = script.TokenMoments(("a",), dim=9)
        moments.update({"a": torch.from_numpy(x)})
        assert moments.cka("a", "a") == pytest.approx(1.0, abs=1e-12)

    def test_it_is_invariant_to_rotation_and_isotropic_scale(self):
        """The property that makes CKA worth reporting beside cosine.

        Cosine is a per-token statement about *this* basis; CKA asks whether the two feature sets
        induce the same geometry over tokens. A rotation of the 768-d space would leave every
        downstream linear head equally able to read the features, and CKA says so where a naive
        elementwise comparison would not.
        """
        rng = np.random.default_rng(2)
        x = rng.normal(size=(80, 6))
        q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
        y = 4.5 * (x @ q) + 7.0  # rotate, scale, translate

        moments = script.TokenMoments(("a", "b"), dim=6)
        moments.update({"a": torch.from_numpy(x), "b": torch.from_numpy(y)})
        assert moments.cka("a", "b") == pytest.approx(1.0, abs=1e-10)

    def test_it_is_symmetric(self):
        rng = np.random.default_rng(3)
        moments = script.TokenMoments(("a", "b"), dim=5)
        moments.update(
            {"a": torch.from_numpy(rng.normal(size=(40, 5))), "b": torch.from_numpy(rng.normal(size=(40, 5)))}
        )
        assert moments.cka("a", "b") == pytest.approx(moments.cka("b", "a"), rel=1e-12)

    def test_it_refuses_a_name_it_was_not_given(self):
        moments = script.TokenMoments(("a",), dim=3)
        with pytest.raises(KeyError):
            moments.update({"b": torch.zeros(2, 3)})


class TestPerTokenCosine:
    def test_matches_a_naive_per_token_loop(self):
        rng = np.random.default_rng(4)
        a = torch.from_numpy(rng.normal(size=(3, 8, 4, 4)).astype(np.float32))
        b = torch.from_numpy(rng.normal(size=(3, 8, 4, 4)).astype(np.float32))

        actual = script.per_token_cosine(a, b)
        assert tuple(actual.shape) == (3, 16)

        expected = torch.zeros(3, 16)
        for i in range(3):
            for r in range(4):
                for c in range(4):
                    u, v = a[i, :, r, c], b[i, :, r, c]
                    expected[i, r * 4 + c] = torch.dot(u, v) / (u.norm() * v.norm())
        assert torch.allclose(actual, expected, atol=1e-6)

    def test_a_feature_map_is_identical_to_itself(self):
        a = torch.randn(2, 8, 4, 4)
        assert torch.allclose(script.per_token_cosine(a, a), torch.ones(2, 16), atol=1e-6)


# --------------------------------------------------------------------------------------
# 3. The tiles
# --------------------------------------------------------------------------------------


class TestSelection:
    @staticmethod
    def _frame(rows):
        keys = ("tile_id", "split", "land_frac", "stratum_id")
        return {k: [r[i] for r in rows] for i, k in enumerate(keys)}

    def test_spans_each_stratum_endpoint_to_endpoint(self):
        """Evenly spaced over the sorted ids, endpoints included.

        The spacing is the whole point: Copernicus tile ids lead with their latitude band, so
        "the lowest N" is a latitude filter wearing a sort's clothing. Spanning the range is what
        keeps a stratum's quota from collapsing onto one region.
        """
        rows = [(f"tile_{s}_{i:03d}", "test", 1.0, s) for s in range(2) for i in range(8, -1, -1)]
        selected = script.select_tile_ids(self._frame(rows), per_stratum=3)
        assert selected == [
            "tile_0_000", "tile_0_004", "tile_0_008",
            "tile_1_000", "tile_1_004", "tile_1_008",
        ]

    def test_a_full_quota_takes_the_whole_stratum(self):
        rows = [(f"tile_{i:03d}", "test", 1.0, 0) for i in range(4)]
        assert script.select_tile_ids(self._frame(rows), per_stratum=4) == [
            "tile_000", "tile_001", "tile_002", "tile_003",
        ]

    def test_a_quota_below_two_cannot_span_anything(self):
        rows = [(f"tile_{i:03d}", "test", 1.0, 0) for i in range(4)]
        with pytest.raises(ValueError, match="at least 2"):
            script.select_tile_ids(self._frame(rows), per_stratum=1)

    def test_it_excludes_train_val_and_any_coastline(self):
        rows = [
            ("keep_a", "test", 1.0, 0),
            ("keep_b", "test", 1.0, 0),
            ("drop_train", "train", 1.0, 0),
            ("drop_val", "val", 1.0, 0),
            ("drop_partial_land", "test", 0.98, 0),
        ]
        assert script.select_tile_ids(self._frame(rows), per_stratum=2) == ["keep_a", "keep_b"]

    def test_a_short_stratum_is_an_error_not_a_silent_shortfall(self):
        """A quiet 297-instead-of-300 would make the sample no longer stratum-balanced."""
        rows = [("a", "test", 1.0, 0), ("b", "test", 1.0, 0), ("c", "test", 1.0, 1)]
        with pytest.raises(ValueError, match="stratum 1"):
            script.select_tile_ids(self._frame(rows), per_stratum=2)

    def test_pivoting_the_frame_preserves_the_requested_order(self):
        frame = self._frame([("b", "test", 1.0, 0), ("a", "test", 1.0, 1), ("c", "test", 1.0, 2)])
        assert [row["tile_id"] for row in script.selection_rows(frame, ["c", "a"])] == ["c", "a"]
        assert script.selection_rows(frame, ["a"])[0]["stratum_id"] == 1

    @requires_corpus
    def test_the_real_manifest_yields_the_recorded_sample(self, record):
        """Re-derive the exact tile list from the manifest, so "300 tiles" is checkable."""
        frame = script.read_selection_frame(CORPUS)
        selected = script.select_tile_ids(frame, per_stratum=record["tiles_per_stratum"])
        assert len(selected) == record["n_tiles"] == len(set(selected))
        assert selected[0] == record["first_tile_id"]
        assert selected[-1] == record["last_tile_id"]

        rows = script.selection_rows(frame, selected)
        assert len({row["source_cog"] for row in rows}) == record["n_source_cogs"]

    def test_the_sample_is_spread_over_the_globe_not_one_region(self, record):
        """The check that would have caught the first draft of the rule.

        Nothing in the cosines or the CKA would have flagged a sample confined to one corner of
        Africa — the numbers looked perfectly reasonable — so the guard has to be on the sample.
        """
        assert record["n_source_cogs"] > 150, "300 tiles drawn from too few source scenes"
        lat_lo, lat_hi = record["lat_range"]
        lon_lo, lon_hi = record["lon_range"]
        assert lat_hi - lat_lo > 120, "latitude span too narrow to call this a global sample"
        assert lon_hi - lon_lo > 300, "longitude span too narrow to call this a global sample"


# --------------------------------------------------------------------------------------
# 4. The record, and the decision hanging off it
# --------------------------------------------------------------------------------------


class TestRecord:
    def test_it_is_the_run_the_ticket_asked_for(self, record):
        assert record["issue"] == 134
        assert record["model"] == "terramind_v1_tokenizer_dem"
        assert record["n_tiles"] == 300
        assert record["tiles_per_stratum"] == 15
        assert record["n_strata"] == 20
        assert record["tokens_per_tile"] == 196
        assert record["n_token_rows"] == 300 * 196
        assert set(record["pairs"]) == {"bicubic_vs_centre", "bicubic_vs_top_left", "centre_vs_top_left"}

    def test_the_run_used_terramind_s_global_normalization(self, record):
        """#118 §7's silent failure, re-guarded: per-tile standardization would erase elevation."""
        assert record["normalization"] == "terramind_global_z_score"
        assert record["constants"]["mean"] == script.p118.TERRAMIND_DEM_MEAN
        assert record["constants"]["std"] == script.p118.TERRAMIND_DEM_STD
        assert record["constants"]["matches_this_script"] is True

    def test_the_arms_differed_in_nothing_but_pos_emb(self, record):
        """Recorded from the run, not asserted in prose: the confounds the ticket rules out."""
        assert record["resize_error_at_14x14"] == 0.0
        assert record["loaded_pos_emb_is_the_closed_form"] is True
        assert record["loaded_pos_emb_shape"] == [1, 768, 16, 16]
        assert record["feature_shape_per_tile"] == [1, 768, 14, 14]

    def test_the_bicubic_arm_is_the_library_default_and_was_checked_against_it(self, record):
        """The control the whole design rests on, run rather than reasoned.

        The bicubic arm installs a pre-resized 14x14 embedding instead of leaving the checkpoint's
        16x16 in place. That is only "terratorch's default" if `ViTEncoder.forward` keys nothing
        else off the embedding's spatial size — which is a claim about someone else's code, so it
        is measured: every batch is also encoded with the untouched model, and the arm must land on
        top of it.

        The bar is the noise floor's *order of magnitude*, not the floor itself. Both quantities
        are worst-token minima over the same float32 nondeterminism, so which of the two samples
        the larger deviation is a coin toss — the run this record came from has the control at
        1.8e-6 against a floor of 1.6e-6. What must hold is that the control sits down at that
        scale rather than anywhere near the 0.029 the run is reporting.
        """
        floor = 1.0 - record["self_consistency_cosine"]
        deviation = 1.0 - record["bicubic_arm_matches_untouched_model"]
        drift = 1.0 - record["pairs"]["bicubic_vs_centre"]["mean_per_token_cosine"]
        assert deviation < 10 * floor
        assert deviation < drift / 1000

    def test_every_pair_reports_both_metrics_over_the_full_sample(self, record):
        for name, pair in record["pairs"].items():
            assert pair["n_token_cosines"] == record["n_token_rows"], name
            assert 0.0 <= pair["cka"] <= 1.0 + 1e-9, name
            assert pair["cosine_min"] <= pair["cosine_p01"] <= pair["mean_per_token_cosine"], name
            assert pair["mean_per_token_cosine"] <= pair["cosine_p99"] <= pair["cosine_max"], name

    def test_verdict_follows_from_the_recorded_number(self, record):
        """The pre-registered rule, recomputed rather than trusted.

        ADR-0011 fixed >= 0.99 as "immaterial, drop the row" before the run existed. This test is
        what stops the JSON's `verdict`, the ADR's prose and plan §2.2 drifting apart later.
        """
        cosine = record["pairs"]["bicubic_vs_centre"]["mean_per_token_cosine"]
        assert record["threshold"] == PREREGISTERED_THRESHOLD
        assert record["verdict"] == ("immaterial" if cosine >= PREREGISTERED_THRESHOLD else "material")
        assert record["ablation_row"] == ("drop" if cosine >= PREREGISTERED_THRESHOLD else "keep")

    def test_the_headline_numbers_the_writeup_quotes(self, record):
        """Pinned so the prose cannot drift from the JSON silently — #118's and #133's precedent."""
        pairs = record["pairs"]
        assert pairs["bicubic_vs_centre"]["mean_per_token_cosine"] == pytest.approx(0.9708, abs=5e-4)
        assert pairs["bicubic_vs_centre"]["cka"] == pytest.approx(0.9815, abs=5e-4)
        assert pairs["bicubic_vs_centre"]["cosine_median"] == pytest.approx(0.9932, abs=5e-4)
        assert pairs["bicubic_vs_top_left"]["mean_per_token_cosine"] == pytest.approx(0.9258, abs=5e-4)
        assert pairs["bicubic_vs_top_left"]["cka"] == pytest.approx(0.9533, abs=5e-4)
        assert pairs["centre_vs_top_left"]["mean_per_token_cosine"] == pytest.approx(0.9363, abs=5e-4)
        assert pairs["centre_vs_top_left"]["cka"] == pytest.approx(0.9610, abs=5e-4)
        assert record["verdict"] == "material"

    def test_centre_is_still_the_crop_nearest_current_behaviour(self, record):
        """#133 established this in embedding space; it survives into feature space.

        Not a decision — ADR-0011 already made it, and on a different argument (single-purpose
        intervention, not proximity to the library default). Worth pinning because if the ordering
        had *inverted* here, the ADR's "bicubic centres and stretches, and only the stretch is
        wrong" reading of the mechanism would be in trouble.
        """
        pairs = record["pairs"]
        assert (
            pairs["bicubic_vs_centre"]["mean_per_token_cosine"]
            > pairs["bicubic_vs_top_left"]["mean_per_token_cosine"]
        )
        assert pairs["bicubic_vs_centre"]["cka"] > pairs["bicubic_vs_top_left"]["cka"]

    def test_the_drift_is_orders_of_magnitude_outside_the_noise_floor(self, record):
        """The control that makes the headline mean readable as signal.

        Re-encoding the *same* embedding is not bit-identical — float32 reduction order on a
        threaded CPU matmul varies pass to pass — so "1 - cosine" has a floor, and the finding is
        only a finding if it clears that floor by a wide margin. It clears it by ~4 orders of
        magnitude, which is what makes the between-arm numbers the embedding and nothing else.
        """
        floor = 1.0 - record["self_consistency_cosine"]
        drift = 1.0 - record["pairs"]["bicubic_vs_centre"]["mean_per_token_cosine"]
        assert floor < 1e-4
        assert drift > 1000 * floor

    def test_the_grand_mean_is_a_tail_not_a_shift(self, record):
        """The shape of the distribution, which is the reason the row is worth a line.

        The median token barely moves. The mean is dragged below the threshold by a minority of
        tokens that move a great deal — the 1st percentile is around half a radian of cosine and
        the worst token is negative. A uniform 0.97 would be a different (and more ignorable)
        finding than a 0.993 median with that tail, so the two must not be allowed to be confused
        in the prose.
        """
        pair = record["pairs"]["bicubic_vs_centre"]
        assert pair["cosine_median"] > pair["mean_per_token_cosine"] + 0.02
        assert pair["cosine_p01"] < 0.6
        assert pair["cosine_min"] < 0.0

    def test_the_structure_under_the_mean_is_recorded(self, record):
        """Position and terrain, recorded so a reader can check the mean is not hiding either.

        No assertion here about the *shape* of the positional pattern. ADR-0011's edge-ward
        accumulation is a statement about embedding space, and §5 of the writeup reports that
        feature space does not reproduce it — pinning a shape would pin an interpretation this
        ticket did not establish. What is pinned is that the variation is real: far larger than
        the run-to-run noise floor, so the grid is signal and not jitter.
        """
        for name, pair in record["pairs"].items():
            grid = np.asarray(pair["cosine_by_position"])
            assert grid.shape == (14, 14), name
            assert grid.mean() == pytest.approx(pair["mean_per_token_cosine"], abs=1e-6), name
            assert sorted(pair["cosine_by_stratum"]) == sorted(str(s) for s in range(20)), name

        grid = np.asarray(record["pairs"]["bicubic_vs_centre"]["cosine_by_position"])
        strata = np.asarray(list(record["pairs"]["bicubic_vs_centre"]["cosine_by_stratum"].values()))
        floor = 1.0 - record["self_consistency_cosine"]
        assert np.ptp(grid) > 1000 * floor
        assert np.ptp(strata) > 1000 * floor
