"""The TerraMind rows of Task 2's arm table: B4/B5, B8, B4-ALT/B5-ALT and B9 (#211).

**Seam 1** is :func:`benchmark.launch.main` on #206's fixture box. Each TerraMind row resolves
under ``--dry-run`` with its encoder, its pin and its diet; B8 and the ALT rows are refused at
any rung but 100%; a B9 cell's run record carries its bf16-against-fp32 agreement, the figure
ADR-0012 Amendment 2 leaves to be measured on B9's first run.

**Seam 2** is the extractor on the real, cached ``TerraMind_v1_base.pt``: the agreement measured
there, under the declared ``bf16`` compute dtype, and read against #134's 0.99 bar. It skips where
the weights are not cached.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from benchmark import arms, landform_arms, launch  # noqa: E402
from tests.test_p206_linear_probe_cell import (  # noqa: E402,F401  (fixtures)
    _flags,
    box,
    corpus,
    stub,
)

#: Plan §1's rows, each with the registry encoder it names and the heads it trains.
ROWS = {
    "B4": ("terramind", ("linear",)),
    "B5": ("terramind", ("upernet",)),
    "B8": ("terramind-latent", ("linear",)),
    "B4-ALT": ("terramind-per-tile", ("linear",)),
    "B5-ALT": ("terramind-per-tile", ("upernet",)),
    "B9": ("terramind-backbone", ("linear", "upernet")),
}

#: §1's pins, quoted from the plan and ADR-0012 Amendment 2 rather than read off the registry.
TOKENIZER_MD5 = "66b7fbcb81e04684df31873623d7a3de"
BACKBONE_MD5 = "690349bd1328e50b64979847d4f362e7"
PINS = {
    "B4": TOKENIZER_MD5, "B5": TOKENIZER_MD5, "B8": TOKENIZER_MD5,
    "B4-ALT": TOKENIZER_MD5, "B5-ALT": TOKENIZER_MD5, "B9": BACKBONE_MD5,
}
#: §2.1's diets: TerraMind's global z-score under each checkpoint's own key, or DEMAE's per-tile
#: standardization for the altitude-passthrough control.
DIETS = {
    "B4": "terramind_v1 under tok_dem@224",
    "B5": "terramind_v1 under tok_dem@224",
    "B8": "terramind_v1 under tok_dem@224",
    "B4-ALT": "per_tile",
    "B5-ALT": "per_tile",
    "B9": "terramind_v1 under untok_dem@224",
}

#: The rows plan §1 runs at 100% only.
ONLY_100 = ("B8", "B4-ALT", "B5-ALT")


def _mounts(box) -> list[str]:
    checkpoint = box / "release" / "encoder.pt"
    return [
        "--set", f'paths.terramind_checkpoint="{checkpoint}"',
        "--set", f'paths.terramind_backbone_checkpoint="{checkpoint}"',
    ]


def test_the_table_names_each_terramind_row_with_its_encoder_and_heads():
    for name, (encoder, heads) in ROWS.items():
        arm = landform_arms.get(name)
        assert (arm.encoder, arm.heads, arm.trains) == (encoder, heads, "head"), name


@pytest.mark.parametrize("name", sorted(ROWS))
def test_each_terramind_row_resolves_under_dry_run_with_its_encoder_pin_and_diet(
    name, box, corpus, capsys
):
    rung = "100%"
    for head in ROWS[name][1]:
        argv = ["run", "--dry-run", "--arm", name, "--head", head, "--rung", rung,
                "--seed", "0", *_flags(box, corpus), *_mounts(box)]
        assert launch.main(argv) == 0
        out = capsys.readouterr().out
        encoder = ROWS[name][0]
        assert f"encoder          {encoder}, pinned md5 {PINS[name]}" in out
        assert f"diet             {DIETS[name]}" in out
        assert f"head {head}" in out
        assert "nothing was written" in out
        # L9: the controls say they are insulated before anything is spent; the core rows do not.
        assert ("insulated        L9" in out) == (name in ONLY_100)


@pytest.mark.parametrize("name", ONLY_100)
@pytest.mark.parametrize("rung", ["1%", "10%"])
def test_b8_and_the_alt_rows_are_refused_at_any_rung_but_100(name, rung, box, corpus, capsys):
    argv = ["run", "--dry-run", "--arm", name, "--rung", rung, "--seed", "0",
            *_flags(box, corpus), *_mounts(box)]
    with pytest.raises(ValueError, match=rf"arm {name} runs at \['100%'\] only, so rung {rung} "):
        launch.main(argv)


def test_the_core_terramind_rows_resolve_at_every_rung(box, corpus, capsys):
    for name in ("B4", "B5", "B9"):
        for rung in ("1%", "10%"):
            argv = ["run", "--dry-run", "--arm", name, "--rung", rung, "--seed", "0",
                    *_flags(box, corpus), *_mounts(box)]
            assert launch.main(argv) == 0, (name, rung)


# --- Seam 1: B9's first run records its bf16 agreement ------------------------------

from tests.test_p205_b9_backbone import synthetic_backbone  # noqa: E402,F401  (fixture)

#: The shortest cell the fixture box can train: batch 4, one epoch over the 8-tile floor ->
#: 2 steps, so a real ViT-B forward on a CPU stays cheap.
SHORT = ["--set", "linear.epochs=1", "--set", "upernet.epochs=1",
         "--set", "upernet.batch_size=4", "--set", "upernet.channels=32"]


def _real_b9_box(box, corpus, synthetic_backbone, monkeypatch) -> list[str]:
    """#206's box with B9 mounted on the synthetic backbone, pinned and in the contract, and
    the short recipe frozen as B6's line for both heads at the 1% rung."""
    import hashlib
    from dataclasses import replace

    from benchmark import extract as extract_module
    from benchmark import landform_recipes
    from benchmark.config import BenchmarkConfig
    from benchmark.landform_config import build_config

    md5 = hashlib.md5(synthetic_backbone.read_bytes()).hexdigest()
    monkeypatch.setitem(arms.ENCODERS, "terramind-backbone",
                        replace(arms.ENCODERS["terramind-backbone"], checkpoint_md5=md5))
    spec = box / "task1" / "extractor_spec.json"
    spec.unlink()  # B9's pin moved, so the contract is re-frozen, as a real pin change would be
    base = BenchmarkConfig()
    extract_module.write_spec(base.replace(paths=base.paths.replace(
        artifacts_dir=str(box / "task1"))))
    argv = [*_flags(box, corpus), *SHORT,
            "--set", f'paths.terramind_backbone_checkpoint="{synthetic_backbone}"']
    (box / "task2" / landform_recipes.RECIPE_LINES_NAME).unlink()
    for head in ("linear", "upernet"):
        cfg = build_config(arm="B6", head=head, rung="1%", overrides=argv[2:][1::2])
        landform_recipes.write_line(cfg, lr=1e-3, scores={"0.001": 0.5})
    return argv


def test_b9s_two_heads_are_two_cells_and_each_run_records_its_bf16_agreement(
    box, corpus, synthetic_backbone, monkeypatch, capsys
):
    from benchmark import extract as extract_module
    from benchmark import results

    argv = _real_b9_box(box, corpus, synthetic_backbone, monkeypatch)
    for head in ("linear", "upernet"):
        assert launch.main(["run", "--arm", "B9", "--head", head, "--rung", "1%", "--seed", "0",
                            *argv]) == 0
        out = capsys.readouterr().out
        assert "precision        terramind-backbone bf16 against fp32: mean cosine" in out
        assert "#134's 0.99 bar" in out

    records = results.read_run_records(results.run_record_path(box / "results.parquet"))
    assert [(r["cell"]["arm"], r["cell"]["head"]) for r in records] == [
        ("B9", "linear"), ("B9", "upernet")]
    for record in records:
        precision = record["precision"]
        assert precision["encoder"] == "terramind-backbone"
        assert (precision["reference_dtype"], precision["compute_dtype"]) == ("fp32", "bf16")
        assert precision["threshold"] == extract_module.PRECISION_THRESHOLD == 0.99
        assert precision["immaterial"] == (precision["mean_cosine"] >= 0.99)
        assert precision["tokens"] == 196 * precision["tiles"] > 0


# --- Seam 1: every tokenizer row runs its cell -------------------------------------

from tests.test_p206_linear_probe_cell import StubExtractor, pin_stub_checkpoint  # noqa: E402


class _LatentAwareStub(StubExtractor):
    """#206's stub, with ``output`` at the encoder's declared `output` width: 5 for B8's
    pre-FSQ latent, the block width for every other encoder (ADR-0012 §3, #204)."""

    def extract(self, raw_i16, valid_mask, *, arm=None, encoder_name=None):
        features = super().extract(raw_i16, valid_mask, arm=arm, encoder_name=encoder_name)
        width = arms.ENCODERS[encoder_name].output_width
        return features._replace(output=features.output[..., :width])


def test_b4_b5_b8_and_the_alt_rows_each_train_their_head_at_100(box, corpus, monkeypatch):
    from benchmark import extract as extract_module
    from benchmark import landform_recipes, results
    from benchmark import upernet as head
    from benchmark.config import BenchmarkConfig
    from benchmark.landform_config import build_config

    monkeypatch.setattr(extract_module, "Extractor", _LatentAwareStub)
    pin_stub_checkpoint(monkeypatch, "terramind", "terramind-latent", "terramind-per-tile")
    (box / "task1" / "extractor_spec.json").unlink()
    base = BenchmarkConfig()
    extract_module.write_spec(base.replace(paths=base.paths.replace(
        artifacts_dir=str(box / "task1"))))
    argv = [*_flags(box, corpus), *_mounts(box), "--set", "upernet.batch_size=4",
            "--set", "upernet.epochs=5", "--set", "upernet.channels=32"]
    cfg = build_config(arm="B6", head="upernet", rung="100%", overrides=argv[2:][1::2])
    landform_recipes.write_line(cfg, lr=1e-3, scores={"0.001": 0.5})

    for name in ("B4", "B5", "B8", "B4-ALT", "B5-ALT"):
        assert launch.main(["run", "--arm", name, "--rung", "100%", "--seed", "0", *argv]) == 0

    records = results.read_run_records(results.run_record_path(box / "results.parquet"))
    trained = {r["cell"]["arm"]: r["parameters"]["trainable"] for r in records}
    upernet_768 = head.parameter_count(768, channels=32)
    assert trained == {
        "B4": 9_226,  # LayerNorm(768) 1,536 + a 1x1 conv to ten classes 7,690
        "B5": upernet_768,
        # The linear probe reads B8's 5-d latent: LayerNorm(5) 10 + a 1x1 conv to ten classes 60
        "B8": 70,
        "B4-ALT": 9_226,
        "B5-ALT": upernet_768,
    }
    assert {r["encoder"]["name"] for r in records if r["cell"]["arm"].endswith("-ALT")} == {
        "terramind-per-tile"}


# --- Seam 2: the agreement on the real backbone, where a box has it -----------------

from tests.test_p205_b9_backbone import BACKBONE  # noqa: E402

REPO = __import__("pathlib").Path(__file__).resolve().parents[1]


def _corpus():
    """The real corpus's directory, in this checkout or — from a worktree, whose ``data/`` holds
    only the committed manifests — in the main checkout the worktree shares a git dir with."""
    import subprocess
    from pathlib import Path

    here = REPO / "data" / "corpus_500k"
    if (here / "shards").is_dir():
        return here
    try:
        common = subprocess.run(["git", "rev-parse", "--git-common-dir"], cwd=REPO,
                                capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    main = (REPO / common).resolve().parent / "data" / "corpus_500k"
    return main if (main / "shards").is_dir() else None


CORPUS = _corpus()


def _real_tiles(n: int = 8):
    """``n`` all-land test tiles, evenly spaced over the sorted test list.

    Real terrain and never synthetic metres: on seeded Gaussian tiles the backbone's agreement
    reads 0.99998, and on real tiles it does not (#211), so a synthetic stand-in would measure
    the wrong thing and look reassuring doing it.
    """
    import numpy as np
    import pyarrow.parquet as pq

    from dataset.normalize import land_mask
    from dataset.shards import shard_path

    rows = [r for r in pq.read_table(CORPUS / "manifest.parquet", columns=[
        "tile_id", "split", "land_frac", "shard_id", "row_index"]).to_pylist()
        if r["split"] == "test" and r["land_frac"] == 1.0]
    rows = sorted(rows, key=lambda r: r["tile_id"])[:: max(1, len(rows) // n)][:n]
    raw = np.stack([
        np.asarray(np.load(shard_path(CORPUS / "shards", r["shard_id"]), mmap_mode="r")
                   [r["row_index"]], dtype=np.int16)
        for r in rows
    ])
    return raw, land_mask(raw)


@pytest.mark.skipif(BACKBONE is None, reason="TerraMind_v1_base.pt is not in the HF cache")
@pytest.mark.skipif(CORPUS is None, reason="the real corpus's shards are not on this box")
def test_b9s_bf16_agreement_is_measured_on_the_pinned_file_under_the_declared_dtype(tmp_path):
    """Amendment 2: the backbone's bf16-against-fp32 agreement, measured rather than inherited
    from the tokenizer's 0.99888, and read against #134's 0.99 bar.

    No assertion on where the number lands, on p157's precedent: it is a measurement, and what
    must not drift is what it is read against and that the verdict follows from it.
    """
    from benchmark.extract import PRECISION_THRESHOLD, Extractor, load_spec, write_spec
    from benchmark.landform_config import LandformConfig

    base = LandformConfig()
    assert base.extract.compute_dtype == "bf16", "Amendment 2 keeps the declared bf16"
    cfg = base.replace(paths=base.paths.replace(
        artifacts_dir=str(tmp_path), terramind_backbone_checkpoint=str(BACKBONE)))
    extractor = Extractor(load_spec(write_spec(cfg)), cfg)
    raw, valid = _real_tiles()
    extractor.extract(raw, valid, encoder_name="terramind-backbone")

    agreement = extractor.precision_agreement(encoder_name="terramind-backbone")
    print(f"\nB9 bf16 agreement: {agreement}")
    assert agreement["encoder"] == "terramind-backbone"
    assert (agreement["reference_dtype"], agreement["compute_dtype"]) == ("fp32", "bf16")
    assert agreement["tiles"] == len(raw) and agreement["tokens"] == 196 * len(raw)
    assert agreement["threshold"] == PRECISION_THRESHOLD == 0.99
    assert agreement["immaterial"] == (agreement["mean_cosine"] >= 0.99)


def test_the_insulated_rows_are_l9s_and_no_core_row_is_one():
    """ADR-0015 L9's list, quoted: B4-ALT/B5-ALT, B8, B1-S/B2-S and B1-R4/B2-R4 (the
    concatenation row is a suite cell, not an arm)."""
    l9 = {"B4-ALT", "B5-ALT", "B8", "B1-S", "B2-S", "B1-R4", "B2-R4"}
    assert {name for name, arm in landform_arms.ARMS.items() if arm.insulated} == l9


# --- #197 review: what a stored row says about its arm and its encoder's precision ----

#: A verdict below #134's bar, as :meth:`benchmark.extract.Extractor._measure_precision` writes
#: one: the case B9's bf16 hit, which the rows must let a reader tell apart.
FAILED_VERDICT = {
    "encoder": "terramind-per-tile", "reference_dtype": "fp32", "compute_dtype": "bf16",
    "device": "cpu", "tiles": 1, "tokens": 196, "mean_cosine": 0.95, "min_cosine": 0.9,
    "p01_cosine": 0.91, "relative_rms": 0.3, "threshold": 0.99, "immaterial": False,
}


class _VerdictStub(_LatentAwareStub):
    """A failed agreement for the ALT encoder, and none measured for B4's."""

    def precision_agreement(self, arm=None, *, encoder_name=None):
        return FAILED_VERDICT if encoder_name == "terramind-per-tile" else None


def test_each_row_carries_its_l9_insulation_and_its_encoders_precision_verdict(
    box, corpus, monkeypatch
):
    """L9's `insulated` left the dry-run line for the run record and the parquet; the encoder's
    measured agreement rides on every row, null where none was measured, so a failing encoder's
    rows are distinguishable without the run record."""
    import pyarrow.parquet as pq

    from benchmark import extract as extract_module
    from benchmark import results
    from benchmark.config import BenchmarkConfig

    monkeypatch.setattr(extract_module, "Extractor", _VerdictStub)
    pin_stub_checkpoint(monkeypatch, "terramind", "terramind-per-tile")
    (box / "task1" / "extractor_spec.json").unlink()
    base = BenchmarkConfig()
    extract_module.write_spec(base.replace(paths=base.paths.replace(
        artifacts_dir=str(box / "task1"))))
    argv = [*_flags(box, corpus), *_mounts(box)]
    for name in ("B4", "B4-ALT"):
        assert launch.main(["run", "--arm", name, "--rung", "100%", "--seed", "0", *argv]) == 0

    rows = pq.read_table(box / "results.parquet").to_pylist()
    by_arm = {}
    for row in rows:
        by_arm.setdefault(row["arm"], set()).add((
            row["insulated"], row["compute_dtype"], row["precision_mean_cosine"],
            row["precision_immaterial"],
        ))
    assert by_arm == {
        "B4": {(False, None, None, None)},
        "B4-ALT": {(True, "bf16", 0.95, False)},
    }
    records = results.read_run_records(results.run_record_path(box / "results.parquet"))
    assert {r["cell"]["arm"]: r["insulated"] for r in records} == {"B4": False, "B4-ALT": True}


def test_a_task_2_parquet_under_an_older_schema_is_refused_as_a_schema_mismatch(tmp_path):
    """#197 review: the refusal used to guess "perhaps Task 1's" for any column mismatch,
    including Task 2's own file written before a column was added."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from benchmark import landform_results

    older = pa.schema([f for f in landform_results.schema() if f.name != "insulated"])
    path = tmp_path / "results.parquet"
    pq.write_table(older.empty_table(), path)
    with pytest.raises(ValueError, match="schema mismatch") as raised:
        landform_results.scored_rows(path)
    assert "Task 1" not in str(raised.value)
    assert "insulated" in str(raised.value)
