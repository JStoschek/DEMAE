"""§10.3's kNN probe: the cheap proxy for representation quality (#99).

Five things are on trial, and they fail in five different ways.

**The probe set is committed, or the arms are not comparable.** Every number this harness
emits is a comparison — this checkpoint against that one, this ablation arm against the
release recipe — and a probe set that redrew itself would make each of those comparisons a
measurement of two different tile populations. So the artifact, not the draw, is the contract:
the tests here are about what the *file* pins and what a file disagreeing with the corpus is
refused for.

**The encoder is frozen, or the probe is training.** "Frozen" is three separate claims — no
gradient, no parameter moved, and the training/eval mode the caller had is the mode it gets
back — and each is independently breakable by a plausible edit.

**A kNN over neighbouring tiles measures the corpus, not the model.** The corpus card (§4.6)
is explicit that two windows from one COG are neighbours rather than independent samples, and
a leave-*one*-out kNN over val tiles would answer every query with the overlapping window next
door. The exclusion is leave-one-**COG**-out, and the test that holds it is the one that would
otherwise report a near-perfect accuracy on features carrying no signal at all.

**The estimators are estimators.** kNN accuracy and kNN error are tested against features that
*are* the label (recover it) and features that are noise (fall back to the no-skill baseline),
because a probe that silently reported chance as skill would be worse than no probe.

**The SSL4EO caveat travels with the number.** MAE representations read weak under kNN relative
to contrastive methods even where fine-tuning wins, so the output is shaped to be read as a
trend: the caveat is logged beside the numbers at every probe point, and the headline quantity
is the *delta against the untrained epoch-0 baseline* rather than the level.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from pretrain import probe as probe_mod
from pretrain.probe import (
    PROBE_REGRESSION_TARGETS,
    ProbeSet,
    draw_probe_set,
    load_probe_tiles,
    read_probe_set,
    write_probe_set,
)
from tests.conftest import build_corpus

#: Enough COGs that a val split holds several of them (the kNN exclusion needs more than one),
#: and enough tiles per COG that the same COG's tiles really are each other's neighbours. The
#: two ``test`` COGs sit in a geocell of their own, because ``test`` blocks whole geocells and
#: `pretrain.data` refuses a train tile inside one.
PROBE_COGS = [
    (
        f"Copernicus_DSM_COG_10_N{40 + i:02d}_00_E00{i % 8}_00_DEM",
        "+010_+050" if split == "test" else "+000_+040",
        split,
    )
    for i, split in enumerate(["val"] * 8 + ["train"] * 6 + ["test"] * 2)
]
TILES_PER_COG = 5


def probe_metrics(c: int, j: int) -> dict:
    """A metric vector that varies per tile, so a regression target is not a constant.

    ``stratum_id`` varies with the COG rather than with the tile index, which is what makes
    the leave-one-COG-out exclusion observable: hold out a query's own COG and its label has to
    be recovered from *elsewhere*, which is the honest question.
    """
    return {
        "stratum_id": c % 4,
        "slope_mean": 0.5 + 0.37 * c + 0.11 * j,
        "geomorphon_entropy": 1.0 + 0.09 * c - 0.03 * j,
    }


@pytest.fixture(scope="module")
def probe_corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("probe_corpus")
    return build_corpus(
        root, PROBE_COGS, tiles_per_cog=TILES_PER_COG, metrics_fn=probe_metrics
    )


# --- The committed probe set --------------------------------------------------


def test_the_probe_set_is_drawn_from_the_val_split_and_nothing_else(probe_corpus):
    """A probe tile outside `val` would be a tile the encoder trained on."""
    import pyarrow.parquet as pq

    drawn = draw_probe_set(probe_corpus / "manifest.parquet", size=20)
    table = pq.read_table(probe_corpus / "manifest.parquet", columns=["tile_id", "split"])
    split_of = dict(zip(table["tile_id"].to_pylist(), table["split"].to_pylist()))
    assert {split_of[t] for t in drawn.tile_id} == {"val"}


def test_the_draw_is_stratified_over_the_splits_own_composition(probe_corpus):
    """Proportional over `stratum_id`, so the probe set looks like the split it came from."""
    drawn = draw_probe_set(probe_corpus / "manifest.parquet", size=20)
    counts = np.bincount(np.array(drawn.stratum_id), minlength=4)
    # 8 val COGs over 4 strata, 5 tiles each: an even split, so 20 draws is 5 per stratum.
    assert counts.tolist() == [5, 5, 5, 5]


def test_a_probe_set_asked_for_more_tiles_than_the_split_holds_takes_the_split(probe_corpus):
    """The shipped 500k val split is ~10k tiles; a tier's may be smaller than the budget."""
    drawn = draw_probe_set(probe_corpus / "manifest.parquet", size=10_000)
    assert len(drawn.tile_id) == len(PROBE_COGS[:8]) * TILES_PER_COG


def test_two_draws_of_one_manifest_write_the_same_bytes(probe_corpus, tmp_path):
    """The house rule: keyed-hash order, never a NumPy Generator (`dataset.sampling`)."""
    manifest = probe_corpus / "manifest.parquet"
    once = write_probe_set(draw_probe_set(manifest, size=20), tmp_path / "a.csv")
    twice = write_probe_set(draw_probe_set(manifest, size=20), tmp_path / "b.csv")
    assert once.read_bytes() == twice.read_bytes()


def test_a_committed_probe_set_round_trips(probe_corpus, tmp_path):
    drawn = draw_probe_set(probe_corpus / "manifest.parquet", size=20)
    assert read_probe_set(write_probe_set(drawn, tmp_path / "probe.csv")) == drawn


# --- What the reader refuses --------------------------------------------------


def test_a_probe_set_naming_a_tile_the_corpus_does_not_hold_is_refused(probe_corpus, tmp_path):
    """A stale or cross-tier artifact — the same refusal `pretrain.data` makes of blocks."""
    stale = ProbeSet(tile_id=("no_such_tile_000000_000000",), stratum_id=(0,))
    path = write_probe_set(stale, tmp_path / "stale.csv")
    with pytest.raises(ValueError, match="does not hold"):
        load_probe_tiles(probe_corpus, path)


def test_a_committed_stratum_that_disagrees_with_the_manifest_is_refused(probe_corpus, tmp_path):
    """The committed label is what the arms are compared under; the manifest may not move."""
    drawn = draw_probe_set(probe_corpus / "manifest.parquet", size=20)
    relabelled = ProbeSet(
        tile_id=drawn.tile_id, stratum_id=(drawn.stratum_id[0] + 1,) + drawn.stratum_id[1:]
    )
    path = write_probe_set(relabelled, tmp_path / "relabelled.csv")
    with pytest.raises(ValueError, match="disagrees"):
        load_probe_tiles(probe_corpus, path)


def test_a_probe_set_holding_a_tile_outside_the_probe_split_is_refused(probe_corpus, tmp_path):
    """The check the draw cannot make for a *committed* file it did not write."""
    import pyarrow.parquet as pq

    columns = ["tile_id", "split", "stratum_id"]
    table = pq.read_table(probe_corpus / "manifest.parquet", columns=columns)
    rows = zip(*(table[c].to_pylist() for c in columns))
    tile_id, _, stratum = next(row for row in rows if row[1] == "train")
    leak = ProbeSet(tile_id=(tile_id,), stratum_id=(stratum,))
    path = write_probe_set(leak, tmp_path / "leak.csv")
    with pytest.raises(ValueError, match="val"):
        load_probe_tiles(probe_corpus, path)


# --- What a loaded probe set carries ------------------------------------------


def test_the_loaded_probe_set_carries_the_labels_and_the_targets_the_probe_scores(
    probe_corpus, tmp_path
):
    """§10.3 asks for one classification and two regressions; all three come off the manifest."""
    drawn = draw_probe_set(probe_corpus / "manifest.parquet", size=20)
    path = write_probe_set(drawn, tmp_path / "p.csv")
    tiles = load_probe_tiles(probe_corpus, path)

    assert len(tiles) == 20
    assert tiles.stratum_id.dtype.kind == "i"
    assert set(tiles.targets) == set(PROBE_REGRESSION_TARGETS)
    for values in tiles.targets.values():
        assert values.shape == (20,) and np.isfinite(values).all()
    # The group the kNN excludes on: several COGs, or the exclusion would empty the bank.
    assert len(set(tiles.source_cog.tolist())) > 1


def test_the_loaded_rows_are_the_manifest_rows_the_committed_ids_name(probe_corpus, tmp_path):
    """`indices` addresses the manifest, so a probe loader and a tile loader agree on a row."""
    import pyarrow.parquet as pq

    drawn = draw_probe_set(probe_corpus / "manifest.parquet", size=20)
    path = write_probe_set(drawn, tmp_path / "p.csv")
    tiles = load_probe_tiles(probe_corpus, path)
    table = pq.read_table(probe_corpus / "manifest.parquet", columns=["tile_id"])
    ids = np.array(table["tile_id"].to_pylist())
    assert ids[tiles.indices].tolist() == list(tiles.tile_id)


def test_the_module_states_the_ssl4eo_caveat(probe_corpus):
    """It is binding, and it belongs in what the harness emits — not only in the ticket."""
    assert "trend" in probe_mod.SSL4EO_CAVEAT.lower()


# --- The frozen feature pass --------------------------------------------------


@pytest.fixture(scope="module")
def probe_path(tmp_path_factory, probe_corpus):
    """A committed probe set over the whole val split of the fixture corpus."""
    path = tmp_path_factory.mktemp("probe_artifact") / "probe_set.csv"
    return write_probe_set(draw_probe_set(probe_corpus / "manifest.parquet", size=40), path)


@pytest.fixture(scope="module")
def probe_cfg(probe_corpus):
    from tests.test_pretrain_train import tiny_config

    base = tiny_config(probe_corpus)
    return base.replace(monitoring=base.monitoring.replace(wandb_mode="disabled"))


@pytest.fixture(scope="module")
def tiles(probe_corpus, probe_path):
    return load_probe_tiles(probe_corpus, probe_path)


@pytest.fixture(scope="module")
def model(probe_cfg):
    from pretrain.model import MaskedAutoencoder

    return MaskedAutoencoder.from_config(probe_cfg)


def test_the_probe_loader_reads_the_committed_tiles_in_committed_order(probe_cfg, tiles):
    """Order is the contract between a feature row and its label; nothing may reshuffle it."""
    loader = probe_mod.build_probe_loader(probe_cfg, tiles)
    read = [t for batch in loader for t in batch["tile_id"]]
    assert read == list(tiles.tile_id)


def test_the_probe_loader_does_not_augment(probe_cfg, tiles):
    """A fresh dihedral draw per read would make a feature a moving measurement."""
    from pretrain.augment import DihedralDataset

    loader = probe_mod.build_probe_loader(probe_cfg, tiles)
    assert not isinstance(loader.dataset, DihedralDataset)


def test_the_probe_loader_goes_through_the_split_dataset(probe_cfg, tiles):
    """§15.2's containment assertion covers the probe path too, not only the training one."""
    from pretrain.data import SplitDataset

    loader = probe_mod.build_probe_loader(probe_cfg, tiles)
    assert isinstance(loader.dataset.dataset, SplitDataset)


def test_features_come_back_one_row_per_probe_tile(probe_cfg, tiles, model):
    features = probe_mod.extract_features(model, probe_mod.build_probe_loader(probe_cfg, tiles))
    assert features.shape == (len(tiles), probe_cfg.architecture.encoder_dim)
    assert np.isfinite(features).all()  # on an *untrained* encoder — the epoch-0 baseline


def test_two_extractions_of_one_model_agree_exactly(probe_cfg, tiles, model):
    """`encode` is the one model path with no RNG in it; a probe has to be a measurement."""
    loader = probe_mod.build_probe_loader(probe_cfg, tiles)
    first = probe_mod.extract_features(model, loader)
    second = probe_mod.extract_features(model, loader)
    assert np.array_equal(first, second)


def test_mean_pooling_averages_the_patch_tokens_and_cls_takes_the_cls_token(
    probe_cfg, tiles, model
):
    """Independently: run the model's own `encode` and pool it by hand."""
    import torch

    loader = probe_mod.build_probe_loader(probe_cfg, tiles)
    batch = next(iter(loader))
    with torch.no_grad():
        encoded = model.encode(batch["image"]).numpy()

    n = len(batch["image"])
    mean = probe_mod.extract_features(model, loader, pooling="mean")[:n]
    cls = probe_mod.extract_features(model, loader, pooling="cls")[:n]
    assert np.allclose(mean, encoded[:, 1:, :].mean(axis=1), atol=1e-5)
    assert np.allclose(cls, encoded[:, 0, :], atol=1e-5)


def test_an_unknown_pooling_is_refused(probe_cfg, tiles, model):
    with pytest.raises(ValueError, match="pooling"):
        probe_mod.extract_features(
            model, probe_mod.build_probe_loader(probe_cfg, tiles), pooling="attentive"
        )


# --- Frozen means three separate claims ---------------------------------------


def test_extraction_reaches_no_gradient_into_the_encoder(probe_cfg, tiles, model):
    """The acceptance criterion, as a check the harness makes rather than a hope."""
    import torch

    # Give the model a gradient first, so "no gradient reached it" is distinguishable from
    # "there were no gradients anywhere". This is what a probe mid-run actually sees.
    model.zero_grad(set_to_none=True)
    model(torch.randn(2, 1, 224, 224)).pred.sum().backward()
    before = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    assert before, "the fixture failed to produce any gradient to protect"

    probe_mod.extract_features(model, probe_mod.build_probe_loader(probe_cfg, tiles))

    after = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    assert set(after) == set(before)
    assert all(torch.equal(after[n], before[n]) for n in before)
    model.zero_grad(set_to_none=True)


def test_extraction_moves_no_parameter(probe_cfg, tiles, model):
    import torch

    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    probe_mod.extract_features(model, probe_mod.build_probe_loader(probe_cfg, tiles))
    assert all(torch.equal(p.detach(), before[n]) for n, p in model.named_parameters())


@pytest.mark.parametrize("training", [True, False])
def test_extraction_gives_back_the_mode_it_found(probe_cfg, tiles, model, training):
    """A probe that left the model in eval would silently disable §6's dropout for the run."""
    model.train(training)
    probe_mod.extract_features(model, probe_mod.build_probe_loader(probe_cfg, tiles))
    assert model.training is training


def test_a_model_that_moves_under_the_frozen_guard_is_refused(probe_cfg, tiles, model):
    """The guard is a check, not a comment: it fires when something does move.

    Nothing in this package moves a parameter during a probe — which is exactly why the guard
    would rot unexercised. So the test moves one itself, inside the guarded region.
    """
    import torch

    with pytest.raises(RuntimeError, match="frozen"), probe_mod.frozen(model), torch.no_grad():
        next(iter(model.parameters())).add_(1.0)


# --- The estimators -----------------------------------------------------------

#: 20 groups of 8, which is the shape that matters: enough tiles that a kNN has a bank, and
#: enough groups that excluding one leaves most of it.
N_GROUPS, PER_GROUP = 20, 8
N_SYNTHETIC = N_GROUPS * PER_GROUP


def synthetic_groups() -> np.ndarray:
    return np.repeat([f"cog{g:02d}" for g in range(N_GROUPS)], PER_GROUP)


def one_hot(values: np.ndarray, *, width: int, noise: float = 0.01) -> np.ndarray:
    """Features that literally spell out ``values``, plus enough noise to break exact ties."""
    rng = np.random.default_rng(0)
    features = np.zeros((len(values), width), dtype=np.float32)
    features[np.arange(len(values)), values % width] = 1.0
    return features + noise * rng.standard_normal(features.shape).astype(np.float32)


def test_knn_classification_recovers_a_label_the_features_spell_out():
    labels = np.arange(N_SYNTHETIC) % 4
    bank = probe_mod.neighbour_bank(one_hot(labels, width=4), groups=synthetic_groups(), k=5)
    result = probe_mod.knn_classify(bank, labels)
    assert result.accuracy == pytest.approx(1.0)
    assert result.skill == pytest.approx(1.0)
    assert result.n_scored == N_SYNTHETIC


def test_knn_classification_of_noise_lands_at_its_own_no_skill_baseline():
    """The failure this rules out: chance reported as skill."""
    rng = np.random.default_rng(1)
    labels = np.arange(N_SYNTHETIC) % 4
    features = rng.standard_normal((N_SYNTHETIC, 16)).astype(np.float32)
    bank = probe_mod.neighbour_bank(features, groups=synthetic_groups(), k=5)
    result = probe_mod.knn_classify(bank, labels)
    assert abs(result.skill) < 0.2
    assert result.baseline_accuracy == pytest.approx(0.25, abs=0.02)


def bearing_features(values: np.ndarray) -> np.ndarray:
    """The target written into the feature's *direction*, not its length.

    The probe measures cosine similarity — a ViT feature's norm carries how strongly the
    encoder responded rather than what it responded to — so a feature that spelled the target
    out as a magnitude would normalize to the same unit vector for every tile. Laying the
    target on an arc is the version of "the features carry the answer" this metric can see.
    """
    theta = (values - values.min()) / np.ptp(values) * (np.pi / 2)
    return np.stack([np.cos(theta), np.sin(theta)], axis=1).astype(np.float32)


def test_knn_regression_recovers_a_target_the_features_spell_out():
    values = np.linspace(0.0, 4.0, N_SYNTHETIC)
    bank = probe_mod.neighbour_bank(bearing_features(values), groups=synthetic_groups(), k=3)
    result = probe_mod.knn_regress(bank, values)
    assert result.skill > 0.9
    assert result.mae < 0.1 * result.baseline_mae


def test_knn_regression_of_noise_scores_no_better_than_the_constant_predictor():
    rng = np.random.default_rng(2)
    values = rng.standard_normal(N_SYNTHETIC)
    features = rng.standard_normal((N_SYNTHETIC, 16)).astype(np.float32)
    bank = probe_mod.neighbour_bank(features, groups=synthetic_groups(), k=5)
    result = probe_mod.knn_regress(bank, values)
    assert result.skill < 0.2


def test_the_bank_excludes_the_querys_own_cog():
    """Corpus card §4.6, as the one test that would fail if the exclusion were dropped.

    Every tile's feature spells out *its own COG*, and its label is that COG. Allowed to answer
    from its own COG, a kNN scores a perfect 1.0 while knowing nothing transferable; made to
    answer from elsewhere, it scores chance. A probe over neighbouring windows measures the
    corpus's geography, not the encoder — and the numbers look their best exactly then.
    """
    groups = synthetic_groups()
    labels = np.repeat(np.arange(N_GROUPS), PER_GROUP)
    features = one_hot(labels, width=N_GROUPS)

    # `groups=arange(n)` puts every tile in its own group, i.e. plain leave-one-out.
    loo = probe_mod.neighbour_bank(features, groups=np.arange(N_SYNTHETIC), k=5)
    by_cog = probe_mod.neighbour_bank(features, groups=groups, k=5)
    permissive = probe_mod.knn_classify(loo, labels)
    blocked = probe_mod.knn_classify(by_cog, labels)
    assert permissive.accuracy == pytest.approx(1.0)
    assert blocked.accuracy < 0.2


def test_a_k_larger_than_the_smallest_eligible_bank_is_refused():
    """Silently clamping would let two arms run different k off one config."""
    with pytest.raises(ValueError, match="k"):
        probe_mod.neighbour_bank(
            one_hot(np.zeros(N_SYNTHETIC, dtype=int), width=4),
            groups=synthetic_groups(),
            k=N_SYNTHETIC,
        )


def test_the_regression_baseline_is_the_leave_one_cog_out_constant_predictor():
    """Computed here the long way round, so the assertion is not the code's own arithmetic."""
    groups = synthetic_groups()
    values = np.linspace(0.0, 4.0, N_SYNTHETIC)
    bank = probe_mod.neighbour_bank(bearing_features(values), groups=groups, k=3)
    result = probe_mod.knn_regress(bank, values)

    expected = np.mean(
        [abs(values[i] - values[groups != groups[i]].mean()) for i in range(N_SYNTHETIC)]
    )
    assert result.baseline_mae == pytest.approx(expected)


# --- The probe point: what a run logs every 20 epochs -------------------------


@pytest.fixture
def logger():
    from pretrain.sink import MemoryLogger

    return MemoryLogger()


@pytest.fixture
def knn_probe(probe_cfg, tiles, logger):
    # k is 3 here, not §10.3's 20: the fixture corpus has 8 val COGs of 5 tiles, so excluding
    # a query's own COG leaves 35 — plenty for a bank, but the real k would swamp it.
    return probe_mod.KnnProbe(probe_cfg, tiles, logger, k=3, every=20)


def test_the_probe_runs_on_an_untrained_encoder(knn_probe, model):
    """The epoch-0 baseline point — and the acceptance criterion it comes from."""
    report = knn_probe.run(model, epoch=0)
    assert np.isfinite(report.classification.accuracy)
    assert set(report.regression) == set(PROBE_REGRESSION_TARGETS)
    assert all(np.isfinite(r.mae) for r in report.regression.values())


def test_the_probe_logs_its_numbers_at_the_epoch_step_axis(knn_probe, model, logger):
    """One step axis for the whole run — #97's losses, #98's grids and this, all on the epoch."""
    knn_probe.run(model, epoch=20)
    logged = logger.at(20)
    assert logged["probe/stratum_acc"] == pytest.approx(
        knn_probe.reports[-1].classification.accuracy
    )
    assert "probe/slope_mean_mae" in logged
    assert "probe/geomorphon_entropy_mae" in logged


def test_the_caveat_travels_with_every_probe_point(knn_probe, model, logger):
    """Not once at the top of a run: on the page, beside the numbers, at every point."""
    knn_probe.run(model, epoch=0)
    knn_probe.run(model, epoch=20)
    assert logger.at(0)["probe/caveat"] == probe_mod.SSL4EO_CAVEAT
    assert logger.at(20)["probe/caveat"] == probe_mod.SSL4EO_CAVEAT


def test_the_headline_series_is_the_delta_against_the_epoch_zero_baseline(
    knn_probe, model, logger
):
    """Shaping the output so the natural read is movement rather than level."""
    baseline = knn_probe.run(model, epoch=0)
    later = knn_probe.run(model, epoch=20)
    assert logger.at(0)["probe/stratum_skill_vs_epoch0"] == pytest.approx(0.0)
    assert logger.at(20)["probe/stratum_skill_vs_epoch0"] == pytest.approx(
        later.classification.skill - baseline.classification.skill
    )


def test_the_probe_measures_its_own_cost(knn_probe, model, logger):
    """§10.3 has to stay minutes, not a meaningful fraction of an epoch — so it is measured."""
    report = knn_probe.run(model, epoch=0)
    assert report.seconds > 0
    assert logger.at(0)["probe/seconds"] == pytest.approx(report.seconds)


def test_two_probe_points_on_one_model_report_the_same_numbers(knn_probe, model):
    """A probe is a measurement: the same checkpoint has to score the same thing twice."""
    first = knn_probe.run(model, epoch=0)
    second = knn_probe.run(model, epoch=20)
    assert first.classification.accuracy == second.classification.accuracy
    assert first.regression["slope_mean"].mae == second.regression["slope_mean"].mae


def test_the_cadence_is_every_n_epochs_starting_at_the_baseline(knn_probe, model, logger):
    assert [knn_probe.due(e) for e in (0, 1, 19, 20, 21, 40)] == [
        True, False, False, True, False, True
    ]
    assert knn_probe.maybe_run(model, epoch=1) is None
    assert knn_probe.maybe_run(model, epoch=20) is not None
    assert logger.trend("probe/stratum_acc") == [(20, pytest.approx(
        knn_probe.reports[-1].classification.accuracy
    ))]


def test_a_cadence_of_zero_turns_the_probe_off(probe_cfg, tiles, logger, model):
    off = probe_mod.KnnProbe(probe_cfg, tiles, logger, k=3, every=0)
    assert off.maybe_run(model, epoch=0) is None
    assert off.maybe_run(model, epoch=20) is None
    assert logger.history == []


def test_every_point_says_which_probe_set_it_was_scored_over(knn_probe, model, logger, probe_path):
    """"Comparable across runs and arms" is checkable only if the tiles are identified."""
    from dataset.materialize import file_md5

    knn_probe.run(model, epoch=0)
    assert logger.at(0)["probe/set_md5"] == file_md5(probe_path)


def test_no_delta_is_emitted_across_two_different_probe_sets(knn_probe, model):
    """Two 5,000-tile sets agree on every count and are not the same measurement."""
    point = knn_probe.run(model, epoch=20)
    elsewhere = replace(point, epoch=0, set_md5="a-different-probe-set")
    assert not point.comparable_to(elsewhere)
    assert not any(key.endswith("_vs_epoch0") for key in point.scalars(elsewhere))
    # And the control: the same set, and the deltas are there.
    assert any(key.endswith("_vs_epoch0") for key in point.scalars(replace(point, epoch=0)))


def test_a_k_the_probe_set_cannot_supply_is_refused_before_the_run_starts(
    probe_cfg, tiles, logger
):
    """A monitoring harness must not be what kills a rented run at epoch 0 over a startup fact."""
    with pytest.raises(ValueError, match="probe_k"):
        probe_mod.KnnProbe(probe_cfg, tiles, logger, k=len(tiles), every=20)


def test_the_printed_line_carries_the_caveat(knn_probe, model):
    """An alarm nobody sees is not an alarm; a caveat nobody sees is worse — it flatters."""
    printed = probe_mod.format_probe(knn_probe.run(model, epoch=20))
    assert "trend" in printed.lower()
    assert "epoch   20" in printed


# --- Wired into the epoch loop ------------------------------------------------


@pytest.fixture
def probing_cfg(probe_cfg, probe_path):
    """The fixture config with §10.3's probe on, at a cadence a 4-epoch test can see."""
    return probe_cfg.replace(
        data=probe_cfg.data.replace(probe_set_path=str(probe_path)),
        monitoring=probe_cfg.monitoring.replace(probe_every_epochs=2, probe_k=3),
    )


def test_a_run_probes_on_its_cadence_and_not_in_between(probing_cfg, logger):
    """End to end: #97's epoch loop drives the probe, and both land on one step axis."""
    from pretrain.monitor import train_epochs

    train_epochs(probing_cfg, epochs=4, device="cpu", logger=logger)
    assert [epoch for epoch, _ in logger.trend("probe/stratum_acc")] == [0, 2]
    # The same epoch carries #97's loss and #99's probe — two callers, one point.
    assert {"val/loss", "probe/stratum_acc"} <= set(logger.at(2))


def test_a_run_with_the_probe_off_logs_none_of_it(probing_cfg, logger):
    from pretrain.monitor import train_epochs

    off = probing_cfg.replace(monitoring=probing_cfg.monitoring.replace(probe_every_epochs=0))
    train_epochs(off, epochs=2, device="cpu", logger=logger)
    assert logger.trend("probe/stratum_acc") == []


def test_a_configured_probe_with_no_committed_probe_set_warns_rather_than_silently_skipping(
    probing_cfg, logger
):
    """§7's failure mode, and §10.3 shares it: a lever that fails by being quietly absent."""
    nowhere = str(probing_cfg.data.corpus_dir) + "/none.csv"
    absent = probing_cfg.replace(data=probing_cfg.data.replace(probe_set_path=nowhere))
    with pytest.warns(UserWarning, match="probe set"):
        assert probe_mod.probe_for(absent, logger) is None


def test_the_probe_does_not_perturb_the_run_it_measures(probing_cfg):
    """A monitoring pass that moved the trajectory would be measuring itself."""
    from pretrain.monitor import train_epochs

    off = probing_cfg.replace(monitoring=probing_cfg.monitoring.replace(probe_every_epochs=0))
    with_probe = [r.train_loss for r in train_epochs(probing_cfg, epochs=3, device="cpu")]
    without = [r.train_loss for r in train_epochs(off, epochs=3, device="cpu")]
    assert with_probe == without


# --- The entrypoint -----------------------------------------------------------


def test_the_cli_draws_and_commits_a_probe_set(probe_corpus, tmp_path, capsys):
    out = tmp_path / "drawn.csv"
    argv = ["--corpus", str(probe_corpus), "--draw", "--size", "20", "--out", str(out)]
    assert probe_mod.main(argv) == 0
    assert len(read_probe_set(out)) == 20
    assert str(out) in capsys.readouterr().out


def test_the_cli_takes_the_epoch_zero_baseline_point(probing_cfg, probe_corpus, tmp_path, capsys):
    """The acceptance criterion, as the command that demonstrates it."""
    import json

    config = tmp_path / "tiny.json"
    config.write_text(json.dumps(probing_cfg.to_dict()))
    assert probe_mod.main(["--config", str(config), "--device", "cpu"]) == 0
    printed = capsys.readouterr().out
    assert "probe epoch    0" in printed
    assert "trend" in printed.lower()
