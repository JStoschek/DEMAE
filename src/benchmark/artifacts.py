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

The one thing here that *writes* is :func:`write_committed`, and it is here for the same reason
the names are: §3.4's "do not overwrite a committed input whose content hash has moved" is one
rule over the whole set, and a rule each stage restates in its own words is a rule three stages
can end up disagreeing about. The stages keep what only they can know — how to render their
artifact, and what counts as the same one — and this module keeps the refusal.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from dataset.provenance import git_sha

from . import arms, prereg
from .config import BenchmarkConfig

#: What `freeze` writes into ``paths.artifacts_dir``. The train mask bank is **not** here — it
#: is 104 MB and lives under ``paths.mask_bank_path``, committed as a hash rather than as bytes
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

#: §3.4's eval mask list — ``tile_id, family, bucket, mask_rle, seed``, one row per eval
#: instance. Named here for the same reason the tile lists are: :mod:`benchmark.masks` writes it
#: and this module resolves it, and a writer and a resolver disagreeing about a name would
#: produce a `run` reporting "not frozen yet" about a file `freeze` had just written.
EVAL_MASKS_NAME: Final = "eval_masks.parquet"

#: §3.4's train mask bank, as the only thing about it that can be committed. The bank itself is
#: 1,000,000 RLE rows and **104 MB** as the zstd parquet `freeze` writes it, and it lives at
#: ``paths.mask_bank_path``, out of git on the same
#: rule the corpus's shards follow rather than on any size threshold. This file holds its md5,
#: its draw hash and the generator config hash, and `run` refuses a bank whose md5 has moved
#: rather than reading it. Named here for the reason the rest are: :mod:`benchmark.masks` writes
#: it and this module resolves it.
MASK_BANK_SIDECAR_NAME: Final = "train_mask_bank.json"


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


def write_committed(
    path: Path | str,
    body: Callable[[], bytes],
    *,
    what: str,
    holds: Callable[[Path], bool] | None = None,
) -> tuple[Path, str]:
    """Write a committed input, idempotently, refusing to overwrite a moved one (§3.4).

    Returns ``(path, md5)``, the md5 being **the committed file's** — what a run record quotes —
    rather than the one this call would have written, which are the same number except in the
    case ``holds`` exists to allow.

    Three things happen here, and each closes a different failure:

    *Re-freezing the same artifact is a no-op.* The bytes on disk are left exactly where they
    were, which is what makes `freeze` idempotent rather than merely repeatable, and what keeps
    the md5 every run record already quotes from moving underneath it.

    *An existing file holding something else is a refusal.* Re-running `freeze` after a code
    change must not silently re-draw an input a half-finished run is already scoring against.
    Deleting the file is the deliberate act that says the recipe really did change, and the
    refusal names both md5s so the tables produced under the old one stay attributable to it.

    *What counts as "something else" is the caller's to say.* ``holds`` is handed the committed
    path and answers whether it holds what this call would write; the default reads the bytes,
    which is right for an artifact whose serialization is a function of its content alone. It is
    not right for parquet: the footer carries the writing library's own version string, so byte
    equality there would answer "the same content, written by this pyarrow?" and a library
    upgrade would refuse a `freeze` that had re-derived an identical artifact — a refusal about
    the environment wearing the words of one about the recipe.

    ``what`` is the noun the refusal uses for the thing that moved — a "contract", a "draw" —
    because the person reading it is being told to delete a committed file, and whether that is
    the right thing to do is a question about which one it is.
    """
    path = Path(path)
    rendered: list[bytes] = []

    def render() -> bytes:
        """``body`` at most once: the byte comparison and the write want the same bytes."""
        if not rendered:
            rendered.append(body())
        return rendered[0]

    def same_bytes(committed: Path) -> bool:
        return committed.read_bytes() == render()

    same_content = same_bytes if holds is None else holds
    if path.is_file():
        committed_md5 = file_md5(path)
        if same_content(path):
            return path, committed_md5
        raise FileExistsError(
            f"{path} is already frozen at md5 {committed_md5}, holding a different {what} from "
            f"the one this config produces, which would be written at md5 "
            f"{hashlib.md5(render()).hexdigest()}. `freeze` does not overwrite a committed input "
            f"whose content hash has moved (§3.4); delete it deliberately if the {what} really "
            f"has changed, remembering that every result already scored against it quotes the "
            f"old md5"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(render())
    return path, file_md5(path)


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
    #: Whether **this arm** reads it. Resolved and printed either way — what is on this box is a
    #: fact about the box, not about the cell — but only a *required* input's absence stops a
    #: run. See :func:`_required_for`.
    required: bool = True

    @property
    def present(self) -> bool:
        """Is it on this box? ``md5`` is the only evidence that counts — see the module docstring."""
        return self.md5 is not None

    def line(self) -> str:
        """The resolution line's row for this input.

        An input this arm does not read is *said* to be one rather than dropped, on the same rule
        the unmounted checkpoint follows: a resolution line that silently omitted an artifact is
        one a build session reads as resolved.
        """
        note = "" if self.required else "  (not read by this arm)"
        if self.path is None:
            return f"{self.label:<16} {self.absent_note}{note}"
        if self.md5 is None:
            return f"{self.label:<16} {self.path}  — {self.absent_note}{note}"
        return f"{self.label:<16} {self.path}  md5 {self.md5}{note}"

    def to_dict(self) -> dict:
        """What the run record files this input under: where it was, what it hashed to, and
        whether this arm read it at all.

        ``required`` is in here rather than only on the printed line, and it has to be: without
        it a null ``md5`` has two readings that a reader cannot tell apart — *this arm does not
        read it* and *`freeze` never wrote it*. An A1 record filed before #151 lands carries a
        null mask-bank md5 for the first reason, and a table whose provenance cannot say which
        one it was is exactly the ambiguity this module exists to remove.
        """
        return {"path": self.path, "md5": self.md5, "required": self.required}


@dataclass(frozen=True)
class Resolution:
    """Everything one command resolved: the cell, the arm, and every input by content."""

    config: BenchmarkConfig
    arm: arms.Arm
    #: The arm's encoder checkpoint, or ``None`` for the two arms that run no encoder.
    checkpoint: Resolved | None
    #: The committed inputs, in the order the resolution line prints them.
    inputs: tuple[Resolved, ...]
    #: The **pretraining** checkpoint — encoder and decoder — for the arm that runs the decoder
    #: as pretrained, or ``None`` for every arm that does not (:attr:`benchmark.arms.Arm.
    #: runs_pretrained_decoder`). Resolved beside :attr:`checkpoint` rather than replacing it:
    #: they are two files, and which of them a box is missing is the difference between "mount
    #: the release" and "mount the run this release was exported from".
    mae_checkpoint: Resolved | None = None

    @property
    def encoder(self) -> arms.Encoder | None:
        """This arm's encoder, or ``None`` for the two arms that run none (A1 and A7)."""
        return None if self.arm.encoder is None else arms.ENCODERS[self.arm.encoder]

    def reads(self, key: str) -> bool:
        """Does this arm read the committed input filed under ``key``?

        The same arm-aware question :attr:`missing` asks, exposed for the checks that go
        *beyond* presence. §3.4's bank is the one input whose being on this box is not enough —
        it is committed as an md5 rather than as bytes — and the arm that does not read it must
        not be refused for a sidecar it has no use for.
        """
        return any(item.key == key and item.required for item in self.inputs)

    @property
    def missing(self) -> tuple[str, ...]:
        """The committed inputs **this arm reads** that `freeze` has not written yet, by label.

        Arm-aware, and that is what lets A1 run first (§4.5 step 2). §3.4's train mask bank is
        1,000,000 rows a head trains against; A1 trains nothing, so a `run --arm A1` that refused
        for want of it would gate *the first number in this benchmark* — the one every learned
        arm is a residual over — on the artifact only the learned arms consume.

        The unrequired inputs are still resolved and still printed, because "what is on this
        box" is a fact about the box; what changes is only whether their absence stops a run.
        """
        return tuple(item.label for item in self.inputs if item.required and not item.present)

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
            lines = [
                f"{'encoder':<16} none — this arm runs no encoder",
                f"{'trains':<16} {self.arm.trains}, input {self.arm.input_convention}",
            ]
            if self.arm.trains_everything:
                # A7's trunk and the half of §4.1's recipe it shares, printed for
                # `trains_head`'s reason one arm over: a UNet trained at a moved learning rate
                # converges normally and prints a row exactly as wide as a good one.
                #
                # `SHARED_WITH_HEAD` rather than `tune.format_frozen`, which prints the whole
                # recipe including `proj_dim` and `stage_dims`. Those two are the *head's shape*
                # and A7 has no head; printing them here would put a 256-d projection on the
                # resolution line of an arm that has none. What is printed is exactly what A7
                # trains under, which is the claim §4.4 makes and the one #162 asserts.
                from . import unet
                from .config import SHARED_WITH_HEAD

                shared = ", ".join(
                    f"{name}={getattr(self.config.head, name)!r}"
                    for name in SHARED_WITH_HEAD
                )
                lines.append(f"{'unet':<16} {unet.format_parameters(self.config.unet)}")
                lines.append(f"{'§4.1 recipe':<16} {shared}")
                lines.append(
                    f"{'':<16} A7 trains under the frozen arms' loss, schedule and seeds "
                    f"(§4.4) — read from `head`, not configured here (#162)"
                )
            return lines
        params = "" if encoder.params is None else f", {encoder.params:,} params"
        lines = [
            f"{'encoder':<16} {encoder.name} — {encoder.family} family, {encoder.width}d, "
            f"pos_emb {encoder.pos_encoding}{params}",
        ]
        # ADR-0011's derivation, printed rather than left to the committed spec (#161). The
        # encoder's *name* already separates `terramind` from `terramind-bicubic`, so this adds
        # no distinction a reader could not reach — but it reaches it through `arms.ENCODERS` at
        # whatever commit `git_sha` names, and the two cells this ticket runs are identical in
        # every other respect. `pos_encoding` above says what the shipped tensor **is**; this
        # says what the model actually **runs with**, and #134 measured those two derivations
        # 0.9708 apart. The same argument #160 made for the axis order, one field along.
        if encoder.pos_emb_derivation != "as_published":
            lines.append(
                f"{'pos_emb':<16} {encoder.pos_emb_derivation} from a "
                f"{encoder.pos_emb_native_grid_px}x{encoder.pos_emb_native_grid_px} grid "
                f"(ADR-0011)"
            )
        # ADR-0013 R4, carried where the arm is described rather than remembered (#161). The
        # rule is that the per-modality figure travels with the encoder wherever it is stated,
        # and a resolution block that names the checkpoint is exactly such a place.
        if encoder.pretraining_epochs_citation is not None:
            lines.append(f"{'pretraining':<16} {encoder.pretraining_epochs_citation}")
        if self.checkpoint is not None:
            lines.append(self.checkpoint.line())
        if self.mae_checkpoint is not None:
            lines.append(self.mae_checkpoint.line())
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
        if self.arm.trains_head:
            # §4.1's frozen recipe, printed before anything is spent, for the same reason the
            # checkpoint md5 is: a head trained at a moved learning rate converges normally and
            # prints a row exactly as wide as a good one. The import is deferred because
            # `benchmark.tune` reaches `benchmark.train`, which reaches this module — a cycle,
            # not a dependency this module is avoiding.
            from . import head, tune

            lines.append(f"{'head':<16} {head.format_parameters(encoder.width)}")
            lines.extend(tune.format_frozen(self.config.head))
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
            # A ceiling on the synthetic half, and nothing at all on the real half: the real
            # family has no bucket axis and is drawn at its natural size distribution (§3.2),
            # and the synthetic half loses the part-ocean tiles `masks.scoreable_tiles` filters
            # out, which is a property of the corpus rather than a number this config computes.
            f"{'instances':<16} at most {cfg.synthetic_eval_instances:,} synthetic "
            f"({cfg.tiles.test:,} test tiles x {len(cfg.masks.synthetic_families)} families "
            f"x {len(cfg.masks.buckets)} buckets, less the part-ocean tiles), plus the real "
            f"family's own rows",
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
            # The axis order by name, beside the encoder that chose it (#160). The encoder's name
            # already separates `demae-r4` from `demae-r5`, so this adds no distinction a reader
            # could not have reached — but it reaches it through `arms.ENCODERS` at whatever
            # commit `git_sha` names, and the failure this record exists to catch is precisely a
            # `pos_embed` loaded in the wrong axis order, which trains fine and is invisible
            # (ADR-0011, P102 §4). A record that names the mistake's own axis states the fact
            # where the mistake would show, rather than one lookup away from it.
            "pos_encoding": None if self.encoder is None else self.encoder.pos_encoding,
            "pos_emb_derivation": (
                None if self.encoder is None else self.encoder.pos_emb_derivation
            ),
            # ADR-0013 R4's per-modality citation, beside the encoder it is about (#161). Both
            # halves, because the rule governs the *sentence*: a bare 20 in a JSON record is a
            # number a reader has no way to tell from TerraMind's 100-epoch headline, and that
            # headline is the one they will supply for themselves if this does not.
            #
            # `None` for every encoder this plan publishes no figure for, which is not the same
            # fact as "not recorded" — the key is always present, and its absence would be a
            # record written before R4 was carried at all.
            "pretraining_epochs": (
                None if self.encoder is None else self.encoder.pretraining_epochs
            ),
            "pretraining_epochs_citation": (
                None if self.encoder is None else self.encoder.pretraining_epochs_citation
            ),
            "checkpoint": None if self.checkpoint is None else self.checkpoint.to_dict(),
            "mae_checkpoint": (
                None if self.mae_checkpoint is None else self.mae_checkpoint.to_dict()
            ),
            "artifacts": {item.key: item.to_dict() for item in self.inputs},
            "git_sha": git_sha(),
        }


#: The committed inputs only a **training** arm reads: §3.1's head-train draw, the val draw its
#: early stopping and LR pick are made on, and §3.4's train mask bank. Everything else — the
#: test list, the eval mask list, the registered bar — is read by every arm that scores anything,
#: and the extractor spec by every arm that runs an encoder.
#:
#: A set rather than a branch inside :func:`_required_for` so the membership is one readable
#: line: the question "does A1 need the mask bank?" is answered by looking, which is the property
#: that matters when the answer decides whether the benchmark's first number can be produced.
TRAINING_ONLY_INPUTS: Final = frozenset(
    {"head_train_tiles", "val_tiles", "mask_bank", "mask_bank_sidecar"}
)


def _required_for(key: str, arm: arms.Arm) -> bool:
    """Does this arm read this committed input?

    Read off the registry's declared fields rather than off the arm's name: ``trains ==
    "nothing"`` is what makes an arm inference-only (A1, A2, A9) and ``encoder is None`` is what
    makes it contract-free (A1, A7). An arm added later inherits the right answer from what it
    declares, which is the whole reason those are fields.
    """
    if key in TRAINING_ONLY_INPUTS:
        return not arm.is_inference_only
    if key == "extractor_spec":
        # ADR-0012's contract governs what comes *out of a frozen encoder* — twelve
        # intermediates and an `output`, at a declared dtype, under a declared diet. A2 asks an
        # encoder for none of that: it runs the whole pretrained autoencoder and reads pixels off
        # the decoder, so the committed spec says nothing about it and gating it on one would
        # refuse an arm for a contract it never consults (#167). It still applies §2.1's declared
        # diet, and that arrives through `benchmark.extract`'s functions rather than through the
        # spec file.
        return arm.encoder is not None and not arm.runs_pretrained_decoder
    return True


def _committed(key: str, label: str, path: Path | str, *, required: bool = True) -> Resolved:
    """One artifact `freeze` owns — so its absence is "not frozen yet" rather than "missing"."""
    return Resolved(
        key=key,
        label=label,
        path=str(path),
        md5=file_md5(path),
        absent_note="not frozen yet (`python -m benchmark.launch freeze`)",
        required=required,
    )


def _checkpoint_for(arm: arms.Arm, cfg: BenchmarkConfig) -> Resolved | None:
    """The arm's encoder checkpoint, resolved — or the fact that this box has not mounted one.

    An unset path is named rather than omitted. A resolution line that simply left out an
    encoder nobody had downloaded is one a build session reads as "resolved".
    """
    if arm.encoder is None:
        return None
    field = arms.ENCODERS[arm.encoder].checkpoint_field
    # A2 declares this encoder — it is what says *whose* weights the arm runs — and reads the
    # pretraining checkpoint instead of the release export of them. Resolved and printed either
    # way, because what is on this box is a fact about the box; what changes is that its absence
    # does not stop the run (#167, and `missing`'s own rule).
    required = not arm.runs_pretrained_decoder
    path = getattr(cfg.paths, field)
    if path is None:
        return Resolved(
            key="checkpoint",
            label="checkpoint",
            path=None,
            md5=None,
            absent_note=f"not mounted on this box — set paths.{field}",
            required=required,
        )
    return Resolved(
        key="checkpoint",
        label="checkpoint",
        path=str(path),
        md5=file_md5(path),
        absent_note=f"missing — check paths.{field}",
        required=required,
    )


def _mae_checkpoint_for(arm: arms.Arm, cfg: BenchmarkConfig) -> Resolved | None:
    """The pretraining checkpoint A2 runs the decoder out of, resolved — or why there is none.

    Three states, and each is *said* rather than left to an absence, on
    :func:`_checkpoint_for`'s rule:

    * this arm does not run a pretrained decoder — ``None``, and nothing is printed;
    * its encoder's family publishes no pretraining checkpoint at all (ViT-S has no released
      run, so it has no source run either) — named, so a reader is not left to infer it;
    * the file is declared and this box has not mounted it — named, with the field to set.

    The absent note points at the release manifest because that is where the answer is: each
    release records the ``source_checkpoint`` it was exported from, by path and by md5.
    """
    if not arm.runs_pretrained_decoder:
        return None
    encoder = arms.ENCODERS[arm.encoder]
    field = encoder.mae_checkpoint_field
    if field is None:
        return Resolved(
            key="mae_checkpoint",
            label="mae checkpoint",
            path=None,
            md5=None,
            absent_note=(
                f"this plan publishes no pretraining checkpoint for {encoder.name}, and "
                f"zero-shot inpainting is the pretrained decoder (§4.3)"
            ),
        )
    path = getattr(cfg.paths, field)
    if path is None:
        return Resolved(
            key="mae_checkpoint",
            label="mae checkpoint",
            path=None,
            md5=None,
            absent_note=(
                f"not mounted on this box — set paths.{field}. §14 exports encoder-only "
                f"weights, so the release `encoder.pt` cannot drive this arm; the release "
                f"manifest's `source_checkpoint` names the file that can"
            ),
        )
    return Resolved(
        key="mae_checkpoint",
        label="mae checkpoint",
        path=str(path),
        md5=file_md5(path),
        absent_note=f"missing — check paths.{field}",
    )


def resolve(cfg: BenchmarkConfig, arm: arms.Arm | None = None) -> Resolution:
    """Everything this cell reads, by content. Stats and hashes; opens no schema and no corpus."""
    arm = arms.get(cfg.arm) if arm is None else arm
    directory = Path(cfg.paths.artifacts_dir)
    at = {
        "head_train_tiles": (
            "head-train list",
            directory / TILE_LIST_NAMES[cfg.size],
        ),
        "val_tiles": ("val list", directory / VAL_TILES_NAME),
        "test_tiles": ("test list", directory / TEST_TILES_NAME),
        "eval_masks": ("eval masks", directory / EVAL_MASKS_NAME),
        # The bank and the file that says what the bank should be, resolved as two inputs
        # because on a given box they are two states: the sidecar is in git and always present
        # on a frozen tree, and the 104 MB it describes may not be. Which of them is missing is
        # the difference between "freeze first" and "regenerate the bank from the sidecar".
        "mask_bank": ("mask bank", cfg.paths.mask_bank_path),
        "mask_bank_sidecar": ("bank sidecar", directory / MASK_BANK_SIDECAR_NAME),
        "extractor_spec": ("extractor spec", directory / "extractor_spec.json"),
        "prereg_bar": ("prereg bar", directory / prereg.BAR_NAME),
    }
    inputs = tuple(
        _committed(key, label, path, required=_required_for(key, arm))
        for key, (label, path) in at.items()
    )
    return Resolution(
        config=cfg,
        arm=arm,
        checkpoint=_checkpoint_for(arm, cfg),
        inputs=inputs,
        mae_checkpoint=_mae_checkpoint_for(arm, cfg),
    )
