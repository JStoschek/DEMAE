"""#161's two A5 rows: the arm the table prints, and ADR-0011 D4's pre-registered ablation.

The run itself needs terratorch, 1.15 GB of tokenizer weights and the 47 GB corpus, so nothing
here scores an instance. What *is* checkable from a fresh clone is everything that decides how
the two rows are told apart once they exist, and that is the whole risk this ticket carries:
both rows train the same head, on the same tiles, over the same checkpoint, and differ in one
tensor nobody can see in a results file. A pair that keyed the same would not produce a wrong
number — it would produce **one** number where the ADR pre-registered two, silently, because
`benchmark.results.append_rows` replaces by `(arm, size, seed)`.

So what is pinned here is the keying, the derivations, the freeze, and R4's citation:

1. **The ablation is its own arm over its own encoder entry**, so the parquet's `arm` column
   separates the cells and the replacement rule cannot fold one into the other.
2. **The two entries differ in exactly one field** — `pos_emb_derivation` — because an ablation
   that moved a second thing would not be ADR-0011 D4's.
3. **The run record says which derivation produced its rows**, which is the only place the
   distinction is legible after the fact.
4. **R4's 20 DEM-tokenizer epochs ride with the encoder**, so the per-modality citation is
   carried wherever the arm is described rather than remembered by whoever writes the memo.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark import arms

REPO = Path(__file__).resolve().parent.parent

#: The arm ADR-0011 Decision 4 pre-registers, and ADR-0013 files as R11.
ABLATION_ARM = "A5-BICUBIC"

#: Its encoder entry — the same 1,145,511,376-byte file as `terramind`, read under terratorch's
#: own default derivation instead of ADR-0011 Decision 2's crop.
ABLATION_ENCODER = "terramind-bicubic"


def test_the_ablation_arm_is_registered_and_is_not_a_printed_row():
    """R5 fixes eleven printed rows; an ablation must not widen a pre-registered table."""
    assert ABLATION_ARM in arms.ARMS
    assert ABLATION_ARM not in arms.PRINTED_ROWS
    assert "A5" in arms.PRINTED_ROWS


def test_the_ablation_keys_separately_from_A5_in_the_results_parquet():
    """`append_rows` replaces by (arm, size, seed) — so the two cells must differ in `arm`.

    The failure this pins is not a wrong number. It is A5's three seeds being *deleted* by the
    ablation's three seeds at the same size, leaving one row where the ADR registered two and
    leaving no trace in the parquet that anything was replaced.
    """
    from benchmark import results

    assert "arm" in results.KEY_COLUMNS
    assert ABLATION_ARM != "A5"


def test_the_two_A5_arms_differ_in_the_derivation_and_in_nothing_else():
    """ADR-0011 D4 registers *one* contrast. A second moved field would confound it."""
    a5, ablation = arms.ARMS["A5"], arms.ARMS[ABLATION_ARM]
    assert (a5.trains, a5.input_convention, a5.normalization) == (
        ablation.trains,
        ablation.input_convention,
        ablation.normalization,
    )

    base, variant = arms.ENCODERS["terramind"], arms.ENCODERS[ABLATION_ENCODER]
    assert base.pos_emb_derivation == "rederived_14x14_crop"
    assert variant.pos_emb_derivation == "bicubic_resize"
    # Same weights, same width, same diet, same native grid — the file is byte-identical and
    # the md5 says so, which is what makes this an ablation on the derivation alone.
    for attr in (
        "family",
        "width",
        "depth",
        "pos_encoding",
        "pos_emb_native_grid_px",
        "checkpoint_field",
        "checkpoint_format",
        "checkpoint_md5",
        "params",
        "output_is_normed",
        "has_cls",
    ):
        assert getattr(base, attr) == getattr(variant, attr), attr


def test_both_derivations_are_registered_vocabulary():
    assert set(arms.POS_EMB_DERIVATIONS) >= {"rederived_14x14_crop", "bicubic_resize"}


def test_the_frozen_spec_carries_both_derivations():
    """The committed contract is what a box re-checks itself against, so the ablation has to be
    *in* it — a derivation the frozen spec cannot express is an ablation nobody can run."""
    spec = json.loads((REPO / "artifacts" / "task1" / "extractor_spec.json").read_text())
    assert spec["arms"][ABLATION_ARM] == ABLATION_ENCODER
    assert spec["encoders"]["terramind"]["pos_emb"]["derivation"] == "rederived_14x14_crop"
    assert spec["encoders"][ABLATION_ENCODER]["pos_emb"]["derivation"] == "bicubic_resize"


def test_the_bicubic_entry_carries_no_ADR_0011_anchor():
    """The anchor is Decision 2's; a bicubic block claiming one would read as a crop."""
    from benchmark.extract import pos_emb_block

    block = pos_emb_block(arms.ENCODERS[ABLATION_ENCODER], grid_px=14)
    assert block["derivation"] == "bicubic_resize"
    assert "anchor" not in block


def test_R4s_twenty_tokenizer_epochs_ride_with_every_terramind_encoder():
    """R4: the per-modality figure, never the paper's 100 — carried, not remembered."""
    from benchmark import prereg

    registered = {rule.rule: rule for rule in prereg.REGISTER}["R4"]
    for name in ("terramind", ABLATION_ENCODER, "terramind-tokenizer"):
        encoder = arms.ENCODERS[name]
        assert encoder.pretraining_epochs == registered.value == 20
        citation = encoder.pretraining_epochs_citation
        assert citation and "DEM" in citation
        # R4's statement is a *disambiguation*, so the citation names the figure it is not.
        # A citation that merely said "20 epochs" would leave the 5x-larger headline standing
        # as the thing a reader supplies for themselves, which is the failure R4 registers.
        assert "100" in citation and "S-2/S-1" in citation

    # Every non-TerraMind encoder leaves it unstated rather than guessing: R4 is a fact about
    # one published artifact, and a default of 20 on DEMAE's own encoders would be an invention.
    for name in ("demae-r5", "demae-r4", "imagenet-mae"):
        assert arms.ENCODERS[name].pretraining_epochs is None


def _resolution(arm_name: str):
    """One cell's resolution, built the way `launch.resolve` builds it and reading no corpus."""
    from benchmark import artifacts
    from benchmark.config import BenchmarkConfig

    cfg = BenchmarkConfig.from_dict({"arm": arm_name, "size": "20k", "seed": 0})
    return artifacts.resolve(cfg, arms.ARMS[arm_name])


def test_the_run_record_distinguishes_the_two_pos_emb_derivations():
    """The acceptance criterion, as the field a reader actually checks it in.

    Both cells train the same head over the same checkpoint on the same tiles. What separates
    them is one tensor, and this is where the separation is written down.
    """
    a5 = _resolution("A5").to_dict()
    ablation = _resolution(ABLATION_ARM).to_dict()

    assert a5["pos_emb_derivation"] == "rederived_14x14_crop"
    assert ablation["pos_emb_derivation"] == "bicubic_resize"
    # And the identity the parquet's own columns carry, so a rows-only reader is not stranded.
    assert (a5["arm"], a5["encoder"]) == ("A5", "terramind")
    assert (ablation["arm"], ablation["encoder"]) == (ABLATION_ARM, ABLATION_ENCODER)


def test_the_two_cells_do_not_share_a_run_id():
    """`run_id` hashes the provenance payload, so a shared one would mean shared rows."""
    from benchmark import results

    assert results.run_id(_resolution("A5")) != results.run_id(_resolution(ABLATION_ARM))


def test_R4s_citation_reaches_the_run_record_and_the_printed_block():
    """"Carried ... anywhere the arm is described" — the two surfaces that describe it."""
    record = _resolution("A5").to_dict()
    assert record["pretraining_epochs"] == 20
    assert "S-2/S-1" in record["pretraining_epochs_citation"]

    printed = "\n".join(_resolution("A5").lines(verb="run"))
    assert "20 epochs over the DEM modality" in printed
    # And the derivation, on the same block, because the ablation's own line must differ.
    assert "rederived_14x14_crop" in printed
    assert "bicubic_resize" in "\n".join(_resolution(ABLATION_ARM).lines(verb="run"))


def test_the_ablation_trains_the_head_frozen_in_ticket_14():
    """§4.1's recipe is tuned once on A6 and frozen; a new arm must not escape that check.

    `launch.resolve` keys the refusal on `trains_head or trains_everything`, so an ablation arm
    registered as a head arm inherits it — this pins that it *is* one, which is the property the
    check reads.
    """
    assert arms.ARMS[ABLATION_ARM].trains_head
    # The head differs between frozen arms only by input width (§4.1), and TerraMind's 768 is
    # DEMAE's — so A5 and its ablation run a head byte-identical to A4's, not merely alike.
    assert arms.ARMS[ABLATION_ARM].width == arms.ARMS["A5"].width == arms.ARMS["A4-R5"].width


def test_the_frozen_encoder_assertion_covers_the_ablation_arm():
    """"The encoder is asserted frozen on the way out" — the assertion, run against a stub.

    `benchmark.train.assert_encoder_frozen` is the mechanism, and what matters for a new arm is
    that it refuses the two states a frozen-encoder pass must never end in.
    """
    torch = pytest.importorskip("torch")
    from benchmark import train

    model = torch.nn.Linear(4, 4)
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()
    before = train.encoder_checksums(model)
    train.assert_encoder_frozen(model, name=ABLATION_ENCODER, before=before)

    model.train()
    with pytest.raises(train.TrainError, match="training mode"):
        train.assert_encoder_frozen(model, name=ABLATION_ENCODER, before=before)

    model.eval()
    model.weight.requires_grad_(True)
    with pytest.raises(train.TrainError, match="requires_grad"):
        train.assert_encoder_frozen(model, name=ABLATION_ENCODER, before=before)
