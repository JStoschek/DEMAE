"""What a command resolved before it took a step — the committed inputs, by content (#147).

`freeze` writes these files and `run` reads them, and between those two moments the only
honest description of an artifact is **its md5**. That is what this module produces: for one
config and one arm, where every input is on this box, whether it is there, and what it hashes
to — printed before the first step and recorded in the run record beside the table it produced.

The failure this closes is the one §15's pre-flight exists for: *a run scoring the wrong arm
should not look like a normal run*. An encoder fed the wrong checkpoint returns a plausible
tensor, a head trained against a re-drawn mask bank converges normally, and a table computed
against a stale eval mask list prints exactly as wide as a good one. None of those raise. So
each is named, out loud, before anything is spent.

**Nothing here reads a schema.** Resolution is presence and content hash and nothing else,
which is what lets it work before the code that writes these files lands — the same property
:mod:`pretrain.fingerprint` was built for. A missing artifact is a *state*, recorded as one,
because an absent entry and a forgotten one look identical otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from dataset.provenance import git_sha

from . import arms
from .config import BenchmarkConfig

#: What `freeze` writes into ``paths.artifacts_dir``. The train mask bank is **not** here — it
#: is 740 MB and lives under ``paths.mask_bank_path``, committed as a hash rather than as bytes
#: (§3.4).
TILE_LIST_NAMES: Final = {
    "20k": "head_train_tiles.csv",
    "5k": "head_train_5k.csv",
    "1k": "head_train_1k.csv",
}

#: The other two §3.1 draws, which have no label-efficiency ladder and so no size key. Named
#: here beside the ladder rather than typed twice: :mod:`benchmark.tiles` writes these files and
#: this module resolves them, and a writer and a resolver that disagreed about a name would
#: produce a `run` that reports "not frozen yet" about a file `freeze` had just written.
VAL_TILES_NAME: Final = "val_tiles.csv"
TEST_TILES_NAME: Final = "test_tiles.csv"


def file_md5(path: Path | str | None) -> str | None:
    """The file's md5, or ``None`` if there is no path or nothing at it.

    :func:`dataset.materialize.file_md5` is the project's **one** implementation of "hash a
    committed artifact" — the corpus card, the ledger, the split and every Phase-1 fingerprint
    pin their inputs with it, and a second chunked loop here would be a second answer to the
    same question. Imported inside the function so resolving a config never drags in the ETL,
    exactly as :func:`pretrain.fingerprint._md5` does it.

    Absence returns ``None`` rather than raising, because before `freeze` runs it is the
    expected state — see the module docstring.
    """
    if path is None:
        return None
    path = Path(path)
    if not path.is_file():
        return None
    from dataset.materialize import file_md5 as hash_artifact

    return hash_artifact(path)


@dataclass(frozen=True)
class Resolved:
    """One input, as this box has it: what it is called, where it is, and what it hashes to."""

    #: Machine key, and what the run record files it under.
    key: str
    #: What the resolution line calls it.
    label: str
    #: ``None`` when no path is configured at all — an unmounted checkpoint, say.
    path: str | None
    #: ``None`` when the file is not there. Absence is a state, not an error.
    md5: str | None
    #: Why it is absent, in the words that say what to do about it.
    absent_note: str | None = None

    @property
    def present(self) -> bool:
        """Is it on this box? ``md5`` is the only evidence that counts — see the module docstring."""
        return self.md5 is not None

    def line(self) -> str:
        """The resolution line's row for this input."""
        if self.path is None:
            return f"{self.label:<16} {self.absent_note}"
        if self.md5 is None:
            return f"{self.label:<16} {self.path}  — {self.absent_note}"
        return f"{self.label:<16} {self.path}  md5 {self.md5}"

    def to_dict(self) -> dict:
        """What the run record files this input under: where it was, and what it hashed to."""
        return {"path": self.path, "md5": self.md5}


@dataclass(frozen=True)
class Resolution:
    """Everything one command resolved: the cell, the arm, and every input by content."""

    config: BenchmarkConfig
    arm: arms.Arm
    #: The arm's encoder checkpoint, or ``None`` for the two arms that run no encoder.
    checkpoint: Resolved | None
    #: The committed inputs, in the order the resolution line prints them.
    inputs: tuple[Resolved, ...]

    @property
    def encoder(self) -> arms.Encoder | None:
        """This arm's encoder, or ``None`` for the two arms that run none (A1 and A7)."""
        return None if self.arm.encoder is None else arms.ENCODERS[self.arm.encoder]

    @property
    def missing(self) -> tuple[str, ...]:
        """The committed inputs `freeze` has not written yet, by label."""
        return tuple(item.label for item in self.inputs if not item.present)

    # --- what a human reads

    def lines(self, *, verb: str) -> list[str]:
        """The whole resolution block, for the person about to spend money on this command.

        The verb is read **once**, here, rather than at each of the four places its answer
        differs. What it selects is honest rather than cosmetic: ``freeze`` is CPU-only,
        torch-free and touches no encoder (§3.4), so an encoder block and a checkpoint md5
        there would be noise at best and, read as a resolution, misleading; and only ``run``
        and ``report`` have anything to do with the results parquet.
        """
        block = [*self.header_lines(verb=verb), ""]
        if verb == "run":
            block += [*self.arm_lines(), ""]
        block += self.input_lines()
        if verb != "freeze":
            block.append(self.results_line())
        return [*block, "", *self.scale_lines()]

    def header_lines(self, *, verb: str) -> list[str]:
        """The first thing on screen — what this command is about, and its hash.

        ``run``'s subject is a **cell**, so it names the arm, the size and the seed and prints
        the config hash those three move. ``freeze`` and ``report`` have no cell: freeze writes
        inputs every cell shares, so it prints §3.4's narrower generator hash instead, which
        does not move with ``--arm``. Naming an arm there would claim the artifacts had one.
        """
        cfg = self.config
        if verb == "run":
            return [
                f"DEMAE Task 1 — run   arm {cfg.arm}, {cfg.size}, seed {cfg.seed}   "
                f"config {cfg.config_hash}",
                f"{'':<16} {self.arm.what}",
            ]
        return [f"DEMAE Task 1 — {verb}   artifacts {cfg.artifact_hash} (§3.4's generator hash)"]

    def arm_lines(self) -> list[str]:
        """Which encoder, which checkpoint, what trains, on what diet, under what contract."""
        encoder = self.encoder
        if encoder is None:
            return [
                f"{'encoder':<16} none — this arm runs no encoder",
                f"{'trains':<16} {self.arm.trains}, input {self.arm.input_convention}",
            ]
        params = "" if encoder.params is None else f", {encoder.params:,} params"
        lines = [
            f"{'encoder':<16} {encoder.name} — {encoder.family} family, {encoder.width}d, "
            f"pos_emb {encoder.pos_encoding}{params}",
        ]
        if self.checkpoint is not None:
            lines.append(self.checkpoint.line())
        lines.append(
            f"{'trains':<16} {self.arm.trains}, input {self.arm.input_convention}, "
            f"{self.arm.normalization} normalization"
        )
        # The contract this arm's features come out of, said where the arm is said. §2.1's trap
        # is that the TerraMind encoder accepts either diet silently, so "which normalization"
        # is a pre-flight fact rather than an implementation detail.
        extract = self.config.extract
        lines.append(
            f"{'extractor':<16} ADR-0012 — {extract.layers} blocks + `output`, "
            f"(N, {extract.tokens}, {encoder.width}), {extract.reference_dtype} reference / "
            f"{extract.compute_dtype} compute, valid = {extract.valid_mask}"
        )
        return lines

    def input_lines(self) -> list[str]:
        """Every committed input, by content — what `freeze` writes and `run` reads."""
        return [item.line() for item in self.inputs]

    def results_line(self) -> str:
        """Where the numbers go, and where the table comes back from.

        Deliberately not one of :attr:`inputs`: it is what ``run`` appends to and ``report``
        reads, so it does not exist before the first cell and its absence is not a missing
        input. **Computed and printed are different questions** — nothing that collapses an
        axis for print discards a number, which is what makes `report` a pure function of this
        file and keeps re-rendering the table from ever re-running an arm.
        """
        return f"{'results':<16} {self.config.paths.results_path}"

    def scale_lines(self) -> list[str]:
        """How much of the corpus this cell touches, and how many instances it will score."""
        cfg = self.config
        nested = (
            ""
            if cfg.size == "20k"
            else f" (nested in the {cfg.tiles.head_train:,} draw)"
        )
        return [
            f"{'tiles':<16} {cfg.train_tiles:,} head-train{nested} / {cfg.tiles.val:,} val / "
            f"{cfg.tiles.test:,} test",
            f"{'bank':<16} {cfg.bank_rows:,} rows "
            f"({cfg.masks.bank_draws_per_tile} draws x {cfg.tiles.head_train:,} tiles)",
            # The synthetic half is derivable and the real half is not: the real family has no
            # bucket axis and is drawn at its natural size distribution (§3.2), so its count is
            # a property of the artifact rather than a number this config can compute.
            f"{'instances':<16} {cfg.synthetic_eval_instances:,} synthetic "
            f"({cfg.tiles.test:,} test tiles x {len(cfg.masks.synthetic_families)} families "
            f"x {len(cfg.masks.buckets)} buckets), plus the real family's own rows",
        ]

    # --- what the run record carries

    def to_dict(self) -> dict:
        """The provenance payload a results table is answerable to — plain JSON throughout.

        The extractor spec's md5 is in here and deliberately **not** in Phase 1's frozen
        four-field data fingerprint (ADR-0012 §7): a results table should say which contract
        produced it without moving a hash four committed artifacts quote.

        This is the provenance block, not the run record. ``benchmark.results`` owns the record
        and the parquet schema beside it, and embeds this — the same split
        :func:`pretrain.header.run_record` and :mod:`pretrain.fingerprint` have, and for the
        same reason: what a run *read* is a different question from what it *produced*.
        """
        return {
            "config": self.config.to_dict(),
            "config_hash": self.config.config_hash,
            # §3.4's generator hash beside the cell's: the row says both which recipe scored it
            # and which draw of the committed inputs it was scored against.
            "artifact_hash": self.config.artifact_hash,
            "arm": self.arm.name,
            "encoder": None if self.encoder is None else self.encoder.name,
            "checkpoint": None if self.checkpoint is None else self.checkpoint.to_dict(),
            "artifacts": {item.key: item.to_dict() for item in self.inputs},
            "git_sha": git_sha(),
        }


def _committed(key: str, label: str, path: Path | str) -> Resolved:
    """One artifact `freeze` owns — so its absence is "not frozen yet" rather than "missing"."""
    return Resolved(
        key=key,
        label=label,
        path=str(path),
        md5=file_md5(path),
        absent_note="not frozen yet (`python -m benchmark.launch freeze`)",
    )


def _checkpoint_for(arm: arms.Arm, cfg: BenchmarkConfig) -> Resolved | None:
    """The arm's encoder checkpoint, resolved — or the fact that this box has not mounted one.

    An unset path is named rather than omitted. A resolution line that simply left out an
    encoder nobody had downloaded is one a build session reads as "resolved".
    """
    if arm.encoder is None:
        return None
    field = arms.ENCODERS[arm.encoder].checkpoint_field
    path = getattr(cfg.paths, field)
    if path is None:
        return Resolved(
            key="checkpoint",
            label="checkpoint",
            path=None,
            md5=None,
            absent_note=f"not mounted on this box — set paths.{field}",
        )
    return Resolved(
        key="checkpoint",
        label="checkpoint",
        path=str(path),
        md5=file_md5(path),
        absent_note=f"missing — check paths.{field}",
    )


def resolve(cfg: BenchmarkConfig, arm: arms.Arm | None = None) -> Resolution:
    """Everything this cell reads, by content. Stats and hashes; opens no schema and no corpus."""
    arm = arms.get(cfg.arm) if arm is None else arm
    directory = Path(cfg.paths.artifacts_dir)
    inputs = (
        _committed(
            "head_train_tiles",
            "head-train list",
            directory / TILE_LIST_NAMES[cfg.size],
        ),
        _committed("val_tiles", "val list", directory / VAL_TILES_NAME),
        _committed("test_tiles", "test list", directory / TEST_TILES_NAME),
        _committed("eval_masks", "eval masks", directory / "eval_masks.parquet"),
        _committed("mask_bank", "mask bank", cfg.paths.mask_bank_path),
        _committed("extractor_spec", "extractor spec", directory / "extractor_spec.json"),
        _committed("prereg_bar", "prereg bar", directory / "task1-prereg-bar.json"),
    )
    return Resolution(
        config=cfg, arm=arm, checkpoint=_checkpoint_for(arm, cfg), inputs=inputs
    )
