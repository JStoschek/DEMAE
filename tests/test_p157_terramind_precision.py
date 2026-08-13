"""Pin #157's precision record: its numbers, its verdict rule, and the arithmetic it reconciles.

The run itself needs terratorch, 1.15 GB of weights and the 47 GB corpus, so it is not
reproducible from a fresh clone and nothing here re-runs it. Following
`tests/test_p134_posemb_materiality.py`'s precedent, what is checkable without any of that is
everything the record's *reading* depends on:

1. **The verdict follows from the number.** ADR-0011 §Decision 5 fixed 0.99 as the materiality
   bar before any of these numbers existed, and `benchmark.extract` carries it as a constant. The
   record's `immaterial` field is recomputed from its own `mean_cosine` rather than trusted, so
   the tie between the recorded figure and the sentence in `p157-terramind-precision.md` §1
   cannot come apart.
2. **The prose's table is the record's numbers.** §1 puts four TerraMind figures beside four
   DEMAE ones. The DEMAE column is ADR-0012 §6's own, quoted; the TerraMind column is this JSON,
   and it is asserted rather than transcribed.
3. **The parameter-count reconciliation.** The record says the 46,080 gap between p118's count
   and this one *is* the `pos_emb` crop. That is arithmetic, so it is checked here from the
   declared geometry rather than believed from a boolean the script wrote about itself.

What is deliberately *not* here is any assertion that the agreement lands on a particular value.
It is a measurement; the thing that must not drift is what it is read against.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RECORD_PATH = REPO / "docs" / "research" / "p157-terramind-precision.json"
DOC_PATH = REPO / "docs" / "research" / "p157-terramind-precision.md"

#: ADR-0011 §Decision 5 / #134, fixed on 2026-08-10 — before this run and before p134's.
PREREGISTERED_THRESHOLD = 0.99

#: ADR-0012 §Decision 6's DEMAE-family figures, which §1's table sets this arm's beside. Quoted
#: from the ADR rather than recomputed: they are that document's measurement, not this one's.
DEMAE_MEAN_COSINE = 0.999968
DEMAE_MIN_COSINE = 0.999431
DEMAE_P01_COSINE = 0.999809


def record() -> dict:
    return json.loads(RECORD_PATH.read_text())


def test_the_record_is_the_arm_and_the_contract_the_prose_says_it_is():
    r = record()
    assert r["issue"] == 157 and r["adr"] == "ADR-0012" and r["arm"] == "A5"
    assert r["encoder"]["name"] == "terramind"
    assert r["encoder"]["normalization"] == "terramind_v1"
    # The derivation in force when the number was taken. A figure measured under the bicubic
    # resize would be a figure about a different set of features (#134: 0.9708 between them).
    assert r["encoder"]["pos_emb"]["derivation"] == "rederived_14x14_crop"
    assert r["encoder"]["pos_emb"]["anchor"] == [1, 1]


def test_the_recorded_md5s_are_the_ones_this_code_publishes():
    """The record names a checkpoint and a contract; both are things this repository pins."""
    import sys

    sys.path.insert(0, str(REPO / "src"))
    from benchmark import arms

    assert record()["encoder"]["checkpoint_md5"] == arms.TERRAMIND_TOKENIZER_DEM_MD5


def test_the_threshold_in_the_record_is_the_one_that_was_pre_registered():
    import sys

    sys.path.insert(0, str(REPO / "src"))
    from benchmark import extract

    assert record()["agreement"]["threshold"] == PREREGISTERED_THRESHOLD
    assert extract.PRECISION_THRESHOLD == PREREGISTERED_THRESHOLD


def test_the_verdict_follows_from_the_recorded_number():
    """Recomputed from the rule, not read off the field the script wrote about itself."""
    agreement = record()["agreement"]
    assert agreement["immaterial"] is (agreement["mean_cosine"] >= PREREGISTERED_THRESHOLD)
    # And the direction the document actually claims in §1.
    assert agreement["immaterial"] is True


def test_the_measurement_is_fp32_against_the_declared_compute_dtype():
    agreement = record()["agreement"]
    assert agreement["reference_dtype"] == "fp32"
    assert agreement["compute_dtype"] == "bf16"
    # Device-matched with ADR-0012 §6's DEMAE figure, which is what makes §1's table a comparison.
    assert agreement["device"] == "mps"


def test_the_sample_is_the_one_the_document_describes():
    r = record()
    agreement = r["agreement"]
    assert r["selection"]["per_stratum"] == 3
    assert len(r["tiles"]) == agreement["tiles"] == 60
    assert agreement["tokens"] == 60 * 196 == 11_760
    assert len({t["stratum_id"] for t in r["tiles"]}) == 20
    assert all(t["split"] == "test" for t in r["tiles"])


def test_the_tails_are_wider_than_the_demae_figure_and_the_prose_says_so():
    """§1's actual finding, and the reason ADR-0012 §6 declined to inherit the DEMAE number.

    The verdict is the same on both arms; the shape is not. The 1st percentile is the row that
    matters — it sits *below* the bar the mean clears — so it is asserted as an ordering against
    the DEMAE figures rather than as a value, which is what makes it a claim about the two arms
    rather than a transcription of one run.
    """
    agreement = record()["agreement"]
    assert agreement["mean_cosine"] < DEMAE_MEAN_COSINE
    assert agreement["p01_cosine"] < DEMAE_P01_COSINE
    assert agreement["min_cosine"] < DEMAE_MIN_COSINE
    assert agreement["p01_cosine"] < PREREGISTERED_THRESHOLD < agreement["mean_cosine"]
    assert "below the 0.99 bar the mean clears" in DOC_PATH.read_text()


def test_the_parameter_gap_against_p118_is_exactly_the_cropped_position_embedding():
    """p118 §8 counted 90,172,416 on the encoder as published; this one holds 90,126,336.

    Recomputed from the declared geometry rather than taken from the record's own boolean: a
    script that got the arithmetic wrong would have written `false` and a script that got the
    *field* wrong would have written `true` regardless.
    """
    r = record()
    encoder = r["encoder"]
    width, block = 768, encoder["pos_emb"]
    cropped_away = width * (block["native_grid_px"] ** 2 - block["grid_px"] ** 2)
    assert cropped_away == 46_080
    assert encoder["registry_params"] - encoder["measured_params"] == cropped_away
    assert encoder["params_delta_is_the_crop"] is True


def test_the_environment_is_the_pinned_one():
    versions = record()["versions"]
    assert versions["terratorch"] == "1.2.10"
    assert versions["diffusers"] == "0.39.0"
