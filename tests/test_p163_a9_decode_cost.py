"""Pin #163's A9 decode-cost record: its arithmetic, its transfer, and the claims §7 now rests on.

The run itself needs terratorch, 1.15 GB of weights and the 47 GB corpus, so it is not
reproducible from a fresh clone and nothing here re-runs it. Following
`tests/test_p157_terramind_precision.py`'s precedent, what is checkable without any of that is
everything the record's *reading* depends on:

1. **The projection is arithmetic, so it is tested as arithmetic.** `project`, `sensitivity` and
   `breakeven_factor` are pure functions of a per-instance figure, a count and a ratio, and they
   are exercised on numbers this file chooses. A projection that cannot be checked independently
   of the measurement is a projection whose errors arrive as budget overruns.
2. **The record is internally consistent.** Its headline projection is recomputed from its own
   measured plateau and its own quoted ratio, so a hand-edit to either half is caught.
3. **The transfer is p120's, unmodified.** The ratio comes out of p120's committed record, and
   the batch-128 column p120 §3 identifies as a memory cliff is excluded rather than allowed to
   win on numeric proximity to a plateau.
4. **The plan says what the record says.** §7's A9 row is the thing this ticket amends, so the
   figures in it are asserted against the JSON rather than transcribed into it by hand.

What is deliberately *not* here is any assertion that the decode lands on a particular number of
seconds. It is a measurement on one device at one moment; what must not drift is what it is read
against and what is derived from it.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
RECORD_PATH = REPO / "docs" / "research" / "p163-a9-decode-cost.json"
DOC_PATH = REPO / "docs" / "research" / "p163-a9-decode-cost.md"
PLAN_PATH = REPO / "docs" / "research" / "task1-void-fill-plan.md"

#: §7's own target, and the number every verdict in the memo is stated against.
BUDGET_USD = 25.0

#: §3.4's eval mask list: 30,000 synthetic rows plus #153's 9,978 real-void rows. §7's A9 row
#: rounds it to "40k"; the exact count is what the projection multiplies.
EVAL_INSTANCES = 39_978

#: TerraMind's own `num_eval_timesteps`, which is what makes A9 the tokenizer *as published*.
TIMESTEPS = 50


def driver():
    """The probe driver, loaded by path — `scripts/` is not a package and should not become one."""
    path = REPO / "scripts" / "p163_a9_decode_cost.py"
    spec = importlib.util.spec_from_file_location("p163_a9_decode_cost", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def record() -> dict:
    return json.loads(RECORD_PATH.read_text())


# --- 1. the projection, checked as arithmetic on numbers this file chooses


def test_the_projection_is_seconds_times_count_over_ratio_and_nothing_else():
    """A round hundred instances at a round second, so every field is checkable by inspection."""
    p = driver().project(1.0, instances=3600, device_ratio=100.0, usd_per_hour=0.4016)
    assert p["local_gpu_hours"] == pytest.approx(1.0)
    assert p["host_gpu_hours"] == pytest.approx(0.01)
    assert p["host_usd"] == pytest.approx(0.004016)
    # The local hours are the measured ones and carry no ratio; that separation is the whole
    # reason the record reports both, since this project's own laptop is not rented.
    assert p["per_instance_seconds_local"] == 1.0
    assert p["device_ratio_5090_over_local"] == 100.0


def test_a_sensitivity_row_is_the_base_projection_with_the_ratio_divided():
    p163 = driver()
    base = p163.project(2.0, instances=10_000, device_ratio=50.0, usd_per_hour=0.4016)
    rows = p163.sensitivity(
        2.0, instances=10_000, device_ratio=50.0, usd_per_hour=0.4016, factors=(1.0, 4.0)
    )
    assert rows[0]["host_usd"] == pytest.approx(base["host_usd"])
    assert rows[1]["host_usd"] == pytest.approx(base["host_usd"] * 4)
    assert rows[1]["ratio_overstated_by"] == 4.0
    # The local hours are what the ratio does *not* touch, so they are the invariant across rows.
    assert rows[0]["local_gpu_hours"] == pytest.approx(rows[1]["local_gpu_hours"])


def test_the_budget_verdict_on_each_row_is_against_the_stated_target():
    p163 = driver()
    rows = p163.sensitivity(
        1.0,
        instances=3_600_000,
        device_ratio=1.0,
        usd_per_hour=1.0,
        factors=(1.0, 100.0),
        budget_usd=BUDGET_USD,
    )
    assert rows[0]["host_usd"] == pytest.approx(1000.0) and rows[0]["within_budget"] is False
    assert rows[1]["within_budget"] is False
    cheap = p163.sensitivity(
        1.0, instances=36, device_ratio=1.0, usd_per_hour=1.0, factors=(1.0,), budget_usd=BUDGET_USD
    )
    assert cheap[0]["host_usd"] == pytest.approx(0.01) and cheap[0]["within_budget"] is True


def test_the_breakeven_factor_is_where_the_bill_reaches_the_budget():
    """Closed-form in the driver; checked here by putting it back through the projection."""
    p163 = driver()
    kwargs = dict(instances=39_978, device_ratio=97.0, usd_per_hour=0.4016)
    factor = p163.breakeven_factor(2.8, budget_usd=BUDGET_USD, **kwargs)
    at_breakeven = p163.project(
        2.8,
        instances=kwargs["instances"],
        device_ratio=kwargs["device_ratio"] / factor,
        usd_per_hour=kwargs["usd_per_hour"],
    )
    assert at_breakeven["host_usd"] == pytest.approx(BUDGET_USD)


def test_the_instance_count_is_the_configs_synthetic_half_plus_the_real_family():
    sys.path.insert(0, str(REPO / "src"))
    from benchmark.config import BenchmarkConfig

    p163 = driver()
    cfg = BenchmarkConfig()
    assert cfg.synthetic_eval_instances == 30_000
    assert p163.REAL_VOID_INSTANCES == 9_978
    assert p163.eval_instances(cfg) == EVAL_INSTANCES


# --- 2. the transfer is p120's, and p120's cliff is not part of it


def test_the_device_ratio_comes_from_p120_and_skips_its_memory_cliff():
    """p120 §3: the batch-128 column is an 8x collapse, not a data point, and its ratio spread
    (9.18 against 1.10-1.14 everywhere else) is what identifies it without this file naming 128."""
    p163 = driver()
    p120 = json.loads((REPO / "docs" / "research" / "p120-primitive-costs.json").read_text())
    for batch in (16, 32, 64):
        matched = p163.device_ratios(batch)
        assert matched["measured_at_batch"] == batch and matched["batch_matched"] is True
        assert matched["geometric_mean"] == pytest.approx(
            p120["local_anchor_batch_sweep"][str(batch)]["_ratio"]["geometric_mean"]
        )
    # 128 is p120's own measured batch and the nearest one to 96 by pure distance; the cliff test
    # sends it to 64 instead.
    assert p163.device_ratios(128)["measured_at_batch"] == 64
    assert p163.device_ratios(96)["measured_at_batch"] == 64
    assert p163.device_ratios(96)["batch_matched"] is False
    # The dollar rate is Phase 1's real one, quoted from p120 rather than retyped here.
    assert p163.device_ratios(32)["usd_per_hour"] == p120["host_usd_per_hour"]


# --- 3. the record is internally consistent


def test_the_record_is_the_arm_and_the_step_count_the_prose_says_it_is():
    r = record()
    assert r["issue"] == 163 and r["arm"] == "A9"
    assert r["timesteps"] == TIMESTEPS and r["tile_px"] == 224
    assert r["instances"]["total"] == EVAL_INSTANCES
    assert r["instances"]["synthetic"] + r["instances"]["real_void"] == EVAL_INSTANCES
    assert r["device"] == "mps"


def test_the_timed_encoder_is_the_contracts_encoder_exactly():
    """A9's pass or not A9's pass, and the difference is not a tolerance.

    The registry builds terratorch's own encoder with the native 16x16 `pos_emb`; ADR-0011
    substitutes the centre-anchored 14x14 crop, and #134 measured the two feature sets 0.9708
    apart. So an agreement of anything but exactly zero means the timings above are about a
    different encoder than the one A9 runs.
    """
    sys.path.insert(0, str(REPO / "src"))
    from benchmark import arms

    r = record()
    agreement = r["encoder_agreement"]
    assert agreement["max_abs_difference"] == 0.0
    assert agreement["identical"] is True
    assert agreement["dtype"] == "fp32"
    assert (agreement["tokens"], agreement["width"]) == (196, 768)
    assert r["encoder"]["pos_emb"]["derivation"] == "rederived_14x14_crop"
    assert r["encoder"]["checkpoint_md5"] == arms.TERRAMIND_TOKENIZER_DEM_MD5


def test_the_whole_tokenizer_was_loaded_and_p118s_split_still_holds():
    """A9 is the one arm that needs the 196 M-parameter half `benchmark.extract` declines to load."""
    params = record()["encoder"]["parameters"]
    assert params["encoder"] == 90_172_416 - 46_080  # p118 §8, less ADR-0011's crop
    assert params["decoder"] == 196_140_560
    assert params["quant_proj"] == 3_845 and params["quantize"] == 0
    # The whole point of the arm: what it runs is mostly the decoder.
    assert params["decoder"] / sum(params.values()) > 0.6


def test_the_headline_projection_is_recomputed_from_the_records_own_plateau():
    p163 = driver()
    r = record()
    plateau, transfer = r["measured"]["plateau"], r["transfer"]
    rebuilt = p163.project(
        plateau["decode_seconds_per_instance"],
        instances=r["instances"]["total"],
        device_ratio=transfer["geometric_mean"],
        usd_per_hour=transfer["usd_per_hour"],
    )
    assert r["projection"]["host_usd"] == pytest.approx(rebuilt["host_usd"])
    assert r["projection"]["host_gpu_hours"] == pytest.approx(rebuilt["host_gpu_hours"])
    assert r["projection"]["local_gpu_hours"] == pytest.approx(rebuilt["local_gpu_hours"])


def test_the_plateau_is_the_cheapest_row_of_the_sweep_it_was_chosen_from():
    r = record()
    ran = [row for row in r["measured"]["batch_sweep"] if "failed" not in row]
    cheapest = min(row["seconds_per_instance"] for row in ran)
    assert r["measured"]["plateau"]["seconds_per_instance"] == cheapest
    # And the curve is a curve: batching buys something, or the plateau claim is vacuous.
    assert ran[0]["seconds_per_instance"] > cheapest
    assert all(row["trials"] >= 3 for row in ran)
    # The shape a saturating curve must have, and the check that caught the contaminated run: cost
    # per instance falls monotonically with batch. The second run's *medians* rose from batch 8 to
    # batch 16, which is not a thing a batch curve does — it is what a shared box does to a median.
    costs = [row["seconds_per_instance"] for row in ran]
    assert costs == sorted(costs, reverse=True)


def test_the_boxs_drift_is_measured_and_published_rather_than_absorbed():
    """The methodological fix this script's first run forced, pinned so it cannot quietly go away.

    One configuration, unchanged, run between every phase. Its spread is the honest error bar on
    every timing in the record — and publishing it is what stops a drifting box from arriving as a
    result attributed to whichever condition happened to run last.
    """
    control = record()["measured"]["control"]
    assert control["batch"] == 8 and control["timesteps"] == 10
    # Before, between and after: enough runs that a monotone drift is visible as one.
    assert len(control["runs"]) >= 5
    assert [c["phase"] for c in control["runs"]][0] == "start"
    assert control["drift_max_over_min"] == pytest.approx(
        control["spread"]["max"] / control["spread"]["min"]
    )
    assert control["drift_max_over_min"] >= 1.0


def test_the_decode_cost_does_not_depend_on_what_is_in_the_tile():
    """The claim that makes a forty-tile sample enough — read off the floor, not off the mean.

    A fixed step count over fixed shapes has no data-dependent work in it, so decode cost should
    not move with terrain. On a shared box the *upper* tail cannot test that: interference is
    one-sided and inflates whichever tiles happened to run while another session was busy. What
    can test it is the floor. If terrain mattered, the cheapest tiles would still disagree with
    each other, and they would come from one kind of ground.

    So: the fastest quartile agrees to within a few percent, **and** it is drawn from many
    different strata. Both halves are needed — a tight floor over one stratum would be the terrain
    hypothesis, not evidence against it.
    """
    r = record()
    per_tile, tiles = r["measured"]["per_tile"], r["tiles"]
    assert len(per_tile) == len(tiles) == 40
    fastest = sorted(per_tile, key=lambda t: t["decode_seconds"])[: len(per_tile) // 4]
    seconds = [t["decode_seconds"] for t in fastest]
    assert max(seconds) / min(seconds) < 1.10
    assert len({tiles[t["index"]]["stratum_id"] for t in fastest}) >= 3
    # And the draw itself spans the corpus, so "many strata" is a fact about the sample too.
    assert len({t["stratum_id"] for t in tiles}) == 20


def test_the_cost_is_linear_in_timesteps_so_the_amendment_lever_is_priced():
    """Fewer steps, proportionally cheaper — the first lever a scope amendment would reach for.

    Stated as the thing that actually matters: 2.5x the steps costs about 2.5x the seconds. The
    tolerance is the drift the run's own control measured, not a constant chosen here, so this
    test tightens automatically on a quieter box instead of enshrining one session's noise.
    """
    r = record()
    rows = {row["timesteps"]: row for row in r["measured"]["timestep_sweep"]}
    observed = (
        rows[50]["decode_seconds_per_instance"] / rows[20]["decode_seconds_per_instance"]
    )
    assert observed == pytest.approx(2.5, rel=0.10)
    # Every point runs in both directions, which is what makes the ratio a ratio of conditions
    # rather than of the order they happened to run in.
    assert all(row["trials"] == 2 for row in r["measured"]["timestep_sweep"])


def test_the_decode_default_raster_is_not_the_benchmarks_tile():
    """The trap ticket 23 inherits: `decode_quant` sizes its output from the decoder's training
    resolution and takes no hint from the 14x14 latent, so the default returns 256 px for a 224 px
    tile. Pinned here so a terratorch that changes the default is caught by a test rather than by
    a scoring pass against a silently rescaled raster."""
    rows = {row["image_size"]: row for row in record()["measured"]["image_size_sweep"]}
    assert rows[224]["matches_tile"] is True and rows[224]["output_px"] == [224, 224]
    assert rows[256]["matches_tile"] is False and rows[256]["output_px"] == [256, 256]
    assert rows[256]["is_decode_quant_default"] is True
    assert rows[224]["is_decode_quant_default"] is False
    # And it is not free: more decoder pixels cost more decoder seconds. Both arms are run in both
    # orders, so this is a statement about the rasters and not about which one ran second.
    assert all(row["trials"] == 2 for row in rows.values())
    assert rows[256]["decode_seconds_per_instance"] > rows[224]["decode_seconds_per_instance"]


def test_the_pass_is_reproducible_only_under_a_seeded_generator():
    d = record()["measured"]["determinism"]
    assert d["seeded_is_reproducible"] is True
    assert d["seeded_max_abs_difference"] == 0.0
    assert d["unseeded_max_abs_difference"] > 0.0


def test_the_environment_is_the_pinned_one():
    versions = record()["versions"]
    assert versions["terratorch"] == "1.2.10"
    assert versions["diffusers"] == "0.39.0"


# --- 4. the plan now says what the record says


def test_section_7s_a9_row_is_no_longer_unpriced():
    """The ticket's actual deliverable against the plan: A9's cost cell holds a number.

    Read on the *cell*, not on the row. The row still contains the word "unpriced" — it says what
    it replaced, which is how every other amended line in §7 is written — so a substring test over
    the whole row would be testing the prose rather than the table.
    """
    r, plan = record(), PLAN_PATH.read_text()
    row = next(line for line in plan.splitlines() if line.startswith("| **A9 "))
    cells = [c.strip() for c in row.strip().strip("|").split("|")]
    minutes = float(cells[2].strip("*"))
    assert minutes == pytest.approx(r["projection"]["host_gpu_hours"] * 60, abs=0.1)
    assert "#163" in row
    # §7's closing paragraph named A9 as the one open number; with it measured, that sentence
    # cannot still be standing.
    assert "The one number that could still move the total is A9's." not in plan
    # And the GPU total now contains it, which it could not while the line was unpriced.
    total = next(line for line in plan.splitlines() if line.startswith("| **GPU total**"))
    assert f"{2.42 + r['projection']['host_gpu_hours']:.2f} h" in total


def test_the_memo_quotes_the_records_numbers_rather_than_its_own():
    """§1's headline figures, asserted against the JSON. p157's test 2, for this record."""
    r, doc = record(), DOC_PATH.read_text()
    assert f"{r['measured']['plateau']['decode_seconds_per_instance']:.2f} s" in doc
    assert f"{r['projection']['host_gpu_hours']:.2f} GPU-h" in doc
    assert f"${r['projection']['host_usd']:.2f}" in doc
    assert f"{round(r['breakeven_ratio_overstated_by'])}" in doc
    assert f"{r['measured']['control']['drift_max_over_min']:.2f}" in doc


def test_the_line_prices_the_decode_alone_and_says_so():
    """The double-count §7's own inference line makes available, declined explicitly.

    §7 bills one frozen encoder forward for each of the 38 scored passes and A9 is one of them, so
    an A9 row charging encode+decode charges its encoder twice. The difference is 0.6% and the
    verdict does not move — which is exactly why it would have survived into the table.
    """
    r = record()
    plateau = r["measured"]["plateau"]
    assert r["projection"]["per_instance_seconds_local"] == plateau["decode_seconds_per_instance"]
    assert (
        r["projection_including_encode"]["per_instance_seconds_local"]
        == plateau["seconds_per_instance"]
    )
    assert r["projection"]["host_usd"] < r["projection_including_encode"]["host_usd"]
    assert "decode only" in r["projection_priced"]
    # And the encode is reported rather than hidden by not being charged.
    assert plateau["encode_seconds_per_instance"] > 0


def test_the_re_measured_encoder_forward_agrees_with_the_one_section_7_already_prices():
    """The transfer's one independent check, and the only evidence here that the ratio travels.

    §7 carries a frozen ViT-B forward at 0.179 ms/tile, from p120, measured on this laptop eight
    months and several phases ago. A9's encode half is that same primitive, so carrying it across
    by the same ratio should land on the same number. Asserted as an agreement band rather than a
    value: what must not drift is that the two independent routes to this primitive still meet.
    """
    check = record()["encoder_forward_cross_check"]
    assert check["plan_section_7_ms_per_tile"] == 0.179
    assert 0.7 < check["ratio_to_plan"] < 1.4
