"""PANGAEA's UperNet under `run --task landform`: the B2 rows and B6's second head (#208).

**Seam 1** is :func:`benchmark.launch.main`, on #206's fixture box. A B2 cell trains §4.2's
UperNet over a stub encoder through the linear probe's own cell path and appends rows to the
same parquet; the head is shown reading ``intermediates[3, 5, 7, 11]`` and nothing else; and
``--tune --head upernet`` freezes a line on B6.

**Below the seam**, as the spec allows: the head's trainable count against PANGAEA's 39.3M for a
768-d encoder. The command cannot show it, and it is the one external anchor these rows have.

The fixture runs the head at 32 channels (``--set upernet.channels=32``) so a CPU cell takes
seconds. The recipe line freezes that knob like any other, so a real cell at PANGAEA's 512 is a
different line, and the parameter test below builds the head at 512.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from benchmark import upernet  # noqa: E402

#: PANGAEA's Table 4 reports 39.34-39.35M trainable parameters for every 768-d encoder.
PANGAEA_768 = 39.3e6


# --- Below the seam: PANGAEA's count --------------------------------------------


def test_the_head_trains_pangaea_s_39_3m_parameters_for_a_768_d_encoder():
    """Neck 7,081,728 (two ConvTranspose2d and a BatchNorm for the 4x level, one ConvTranspose2d
    for the 2x), then UperNet at 512 channels with pool scales 1/2/3/6: 32,267,274 including
    the 10-class 1x1 classifier. 39,349,002 in all, inside Table 4's 39.34-39.35M."""
    built = upernet.build(768)
    assert built.parameters == 39_349_002
    assert round(built.parameters / 1e6, 1) == PANGAEA_768 / 1e6
    assert 39.34e6 <= built.parameters <= 39.35e6
    assert built.parameters == sum(p.numel() for p in built.module.parameters()
                                   if p.requires_grad)


# --- Seam 1: a B2 cell end to end -----------------------------------------------

import json  # noqa: E402

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from benchmark import landform_recipes, launch, results  # noqa: E402
from benchmark.landform_config import build_config  # noqa: E402
from tests.test_p206_linear_probe_cell import (  # noqa: E402,F401  (fixtures)
    TILES_PER_COG,
    _flags,
    box,
    corpus,
    stub,
)

#: The UperNet at test scale: batch 4, 5 epochs over the 8-tile floor -> 10 steps at 1%, and
#: 32 channels so the 56 px FPN fuse is cheap on a CPU.
UPERNET_TEST = [
    "--set", "upernet.batch_size=4",
    "--set", "upernet.epochs=5",
    "--set", "upernet.channels=32",
]
STEPS = 10


@pytest.fixture
def flags(box, corpus):
    """#206's box with an UperNet line at every rung, as though B6 had tuned them."""
    argv = [*_flags(box, corpus), *UPERNET_TEST]
    cfg = build_config(arm="B6", head="upernet", overrides=argv[2:][1::2])
    for rung in ("1%", "10%", "100%"):
        landform_recipes.write_line(cfg.replace(rung=rung), lr=1e-3, scores={"0.001": 0.5})
    return argv


def b2(*argv: str) -> list[str]:
    return ["run", "--arm", "B2", "--rung", "1%", "--seed", "0", *argv]


def records_of(box) -> list[dict]:
    return results.read_run_records(results.run_record_path(box / "results.parquet"))


def test_a_b2_cell_trains_the_upernet_and_appends_rows_beside_the_linear_ones(box, flags, stub):
    assert launch.main(["run", "--rung", "1%", "--seed", "0", *flags]) == 0  # B1, linear
    assert launch.main([*b2(), *flags]) == 0
    rows = pq.read_table(box / "results.parquet").to_pylist()
    b2_rows = [r for r in rows if r["arm"] == "B2"]
    assert len(b2_rows) == 2 * TILES_PER_COG, "one row per test tile"
    assert {r["head"] for r in b2_rows} == {"upernet"}
    assert {r["head"] for r in rows if r["arm"] == "B1"} == {"linear"}
    assert all(np.asarray(r["confusion"]).shape == (100,) for r in b2_rows)

    record = records_of(box)[-1]
    assert record["cell"]["arm"] == "B2" and record["cell"]["head"] == "upernet"
    assert record["recipe_line"]["head"] == "upernet" and record["recipe_line"]["lr"] == 1e-3
    assert record["training"]["kind"] == "upernet" and record["training"]["steps"] == STEPS
    assert record["timing"]["head"] == "upernet" and record["timing"]["seconds_per_step"] > 0
    from benchmark import upernet as head
    assert record["parameters"]["trainable"] == head.parameter_count(768, channels=32)


class _Watched(tuple):
    """The extractor's twelve blocks, noting which ones a consumer indexes."""

    seen: set = set()

    def __getitem__(self, i):
        _Watched.seen.add(i)
        return super().__getitem__(i)

    def __iter__(self):
        raise AssertionError("the head iterated every block rather than indexing its four")


class _WatchedFeatures:
    """Features whose ``output`` refuses to be read, so a head reading it fails loudly."""

    def __init__(self, features):
        self.intermediates = _Watched(features.intermediates)
        self.cls = None

    @property
    def output(self):
        raise AssertionError("the UperNet read `output`; §4.2 reads intermediates only")


def test_the_head_reads_intermediates_3_5_7_and_11_and_never_output(box, flags, stub,
                                                                    monkeypatch):
    real = stub.extract
    monkeypatch.setattr(stub, "extract",
                        lambda self, *a, **k: _WatchedFeatures(real(self, *a, **k)))
    _Watched.seen = set()
    assert launch.main([*b2(), *flags]) == 0
    assert _Watched.seen == {3, 5, 7, 11}


# --- Seam 1: `--tune --head upernet` on B6 ---------------------------------------


def test_tune_head_upernet_freezes_b6_s_upernet_line_and_a_b2_cell_then_runs_at_it(
    box, corpus, stub, capsys
):
    (box / "task2" / landform_recipes.RECIPE_LINES_NAME).unlink()
    argv = [*_flags(box, corpus), *UPERNET_TEST]
    assert launch.main(["run", "--tune", "--head", "upernet", "--rung", "1%", *argv]) == 0
    assert "arm B6" in capsys.readouterr().out
    lines = json.loads((box / "task2" / landform_recipes.RECIPE_LINES_NAME).read_text())["lines"]
    line = lines["upernet"]["1%"]
    assert line["tuned_on"] == {"arm": "B6", "seed": 0, "ladder": 0}
    assert line["recipe"]["channels"] == 32 and line["recipe"]["layers"] == [3, 5, 7, 11]
    assert line["recipe"]["weight_decay"] == 0.05
    record = json.loads(landform_recipes.tuning_record_path_for(box / "task2", "upernet", "1%")
                        .read_text())
    assert tuple(record["passes"][0]["grid"]) == (3e-5, 1e-4, 3e-4, 1e-3, 3e-3)
    assert record["steps_per_point"] == STEPS // 5

    assert launch.main([*b2(), *argv]) == 0
    assert records_of(box)[-1]["recipe_line"]["lr"] == line["lr"]


# --- Seam 1: the other UperNet rows -----------------------------------------------


def test_b6_upernet_b2_r4_and_b2_s_run_through_the_same_command_beside_b6_s_linear_cell(
    box, flags, stub, monkeypatch
):
    """B6 is one arm with two heads: its linear and UperNet cells are two cells in one parquet.
    B2-R4 and B2-S are the identical UperNet on their own encoders, at their own widths."""
    from benchmark import extract as extract_module
    from benchmark.config import BenchmarkConfig
    from tests.test_p206_linear_probe_cell import pin_stub_checkpoint

    pin_stub_checkpoint(monkeypatch, "demae-r4", "demae-vits")
    spec = box / "task1" / "extractor_spec.json"
    spec.unlink()  # the pins moved, so the contract is re-frozen, as a real pin change would be
    base = BenchmarkConfig()
    extract_module.write_spec(base.replace(paths=base.paths.replace(
        artifacts_dir=str(box / "task1"))))
    checkpoint = box / "release" / "encoder.pt"
    mounts = [
        "--set", f'paths.demae_r4_checkpoint="{checkpoint}"',
        "--set", f'paths.demae_vits_checkpoint="{checkpoint}"',
    ]
    cell = ["run", "--rung", "1%", "--seed", "0"]
    assert launch.main([*cell, "--arm", "B6", *flags, *mounts]) == 0
    assert launch.main([*cell, "--arm", "B6", "--head", "upernet", *flags, *mounts]) == 0
    for arm in ("B2-R4", "B2-S"):
        assert launch.main([*cell, "--arm", arm, *flags, *mounts]) == 0

    from benchmark import upernet as head
    trained = {(r["cell"]["arm"], r["cell"]["head"]): r["parameters"]["trainable"]
               for r in records_of(box)}
    assert trained == {
        ("B6", "linear"): 9_226,
        ("B6", "upernet"): head.parameter_count(768, channels=32),
        ("B2-R4", "upernet"): head.parameter_count(768, channels=32),
        ("B2-S", "upernet"): head.parameter_count(384, channels=32),
    }
    rows = pq.read_table(box / "results.parquet").to_pylist()
    assert {(r["arm"], r["head"]) for r in rows} == set(trained)
