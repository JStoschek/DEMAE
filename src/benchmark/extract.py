"""ADR-0012's frozen-feature contract — raw metres in, all twelve blocks out (#156).

The one thing Phase 2's two benchmarks share, and it is a **committed file Task 2 re-reads**
rather than a description it re-implements. The failure that shape exists to prevent is precise:
Task 2 re-runs "the same extractor" from prose, gets something subtly different, and both tasks'
C1 claims silently stop being about the same features.
[ADR-0012](../../docs/adr/0012-frozen-feature-extractor-contract.md) is the decision; this module
is the file, its writer, and the loader that re-checks it and **refuses rather than warns**.

## The interface, and the bug it makes unrepresentable

``extract(raw_i16, valid_mask, *, arm) -> Features`` — int16 metres and a boolean valid-pixel
mask, never a normalized tensor. p118 §7 measured the trap: TerraMind wants
``(metres − 670.665) / 951.272``, DEMAE wants per-tile standardization, and the TerraMind encoder
**accepts either silently**, returning a plausible ``[1, 768, 14, 14]`` both ways. Nothing raises.
So the extractor applies its own declared normalization and the pretraining loader's output — a
float32 torch tensor of per-tile-standardized units — is refused at the door by dtype rather than
caught by a test that has to be remembered.

``valid = land ∧ ¬void``. Task 1 passes the void-aware mask (§2.1's non-void statistics, which is
what stops the leak), Task 2 passes land-only, and it is one code path with no branch whose two
sides belong to different tasks. Statistics are **always recomputed from the passed mask**
(:func:`tile_statistics`) and never read from the manifest; ADR-0012 Decision 2 ships that as a
*checked claim* rather than a comment, so an ETL land-definition drift surfaces in Task 1 rather
than in Task 2.

## `output` and `intermediates[11]` are separately named

DEMAE runs its twelve blocks and then applies ``self.norm``
([`pretrain.model`](../pretrain/model.py) ``_transform``), so its own final tensor is *not* its
last block's output. p118 §5.3 measured that TerraMind's twelve intermediates carry no separate
post-norm entry and its last is bit-identical to ``encoder(x)``. "The last layer" and "block
twelve" are therefore two different things on one side and the same thing on the other, and a
shared index would quietly compare one to the other. :class:`Features` names both, and
:attr:`EncoderSpec.output_is_normed` is the declared field that says which encoder is which.

No Task 1 arm reads ``intermediates``. ADR-0012 Decision 4 requires Task 1 to test the path
anyway — shapes, ordering, count, and the layer-12-versus-``output`` relation — because the
narrow version is the one under which "reusable by Task 2 unchanged" quietly fails, and the test
is what makes it a feature rather than a plausible-looking function.

## What is checked, and when

Two moments, because the checks cost two very different things.

:func:`load_spec` is **torch-free and cheap**: it parses the committed file and re-checks it
against the ADR's invariants and against the live :mod:`benchmark.arms` registry. A spec that has
drifted from the code it claims to describe is refused there, which is the check Task 2 gets for
free by loading the file at all.

:meth:`Extractor.encoder` is where §7.1's *"declared checkpoint md5 and architecture against the
model actually loaded"* happens, because it is the one that reads 340 MB. Four refusals: the
file's md5 against the declared one; the checkpoint's own block count against the declared layer
count; its width and token grid against the declared shape; and its ``pos_embed`` against
:func:`pretrain.model.sincos2d_pos_embed` under the **declared axis order**. That last one is
§2.3's R4-versus-R5 precondition enforced rather than remembered — R4 learned mmselfsup's
row-first sin-cos and R5 MAE's own, every name and every shape matches under either, and the two
orders differ by 2.0, the full range of a sinusoid.

## What this ticket does not cover

TerraMind is [#12](https://github.com/JStoschek/DEMAE/issues/12): the named
``v1_pretraining_mean/std["tok_dem@224"]`` lookup, its value check against ``670.665 / 951.272``
(#138 found three DEM constant pairs shipping across IBM's artifacts, so the *name* is what
selects the pair and the *value* is what catches a terratorch upgrade retuning it), and
ADR-0011's ``pos_emb`` derivation as a declared field. It is **named as owed** in the spec's
:attr:`ExtractorSpec.pending` block rather than left out, and A5 and A9 refuse by naming the
ticket — an absent contract that read as a resolved one is the thing the whole family of
committed artifacts exists to prevent.

## Where the spec's md5 goes, and where it deliberately does not

The **benchmark's run record** (:meth:`benchmark.artifacts.Resolution.to_dict`), so a results
table says which contract produced it. Explicitly **not** a fifth field in Phase 1's data
fingerprint (:mod:`pretrain.fingerprint`), which is a frozen four-field block four committed
artifacts quote (ADR-0012 §7).

torch, timm and :mod:`pretrain` are imported **inside** the functions that need them, the way
:mod:`dataset.loader` does it (ADR-0003): writing and reading the spec is CPU-only, torch-free
and runs on a box with no CUDA stack, which is what `freeze` is.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Final, NamedTuple

import numpy as np

from dataset import config as dataset_config

from . import arms
from .config import BenchmarkConfig

#: What `freeze` writes into ``paths.artifacts_dir``, and what
#: :mod:`benchmark.artifacts` resolves under the key ``extractor_spec``.
SPEC_FILENAME: Final = "extractor_spec.json"

#: The decision this file is the executable half of.
ADR: Final = "ADR-0012"

#: ADR-0012 §3 — all twelve block outputs, plus a separately named ``output``. Here rather than
#: read off :class:`benchmark.config.Extract` because :func:`load_spec` is what a *consumer*
#: calls and must not depend on a run config existing; the two are pinned equal by
#: ``test_the_contracts_layer_count_is_the_configs``.
LAYERS: Final = 12

#: ADR-0012 §5, in the one sentence a consumer needs.
CANONICAL_LAYOUT: Final = "(N, 196, D) patch tokens, row-major; CLS an optional side field"

#: ADR-0012 §1's mask definition, spelled the way the config knob spells it.
VALID_MASK: Final = "land_and_not_void"

#: ADR-0012 §2, as a string the committed file carries — so a spec that read the manifest would
#: have to say so in the file rather than in a code path nobody diffs.
STATISTICS: Final = "recomputed from the passed valid mask; never read from the manifest"

#: The std guard, and it is :data:`dataset.loader._STD_FLOOR`'s value rather than a second
#: opinion: the ETL already drops tiles under 3 m of land std, so this only ever fences a
#: degenerate mask. ``test_the_std_floor_is_the_loaders_own`` pins the two together.
STD_FLOOR: Final = 1e-6

#: The checkpoint layouts :meth:`Extractor.encoder` can read. ``terratorch`` is declared by the
#: registry and refused here — see the module docstring on #12.
READABLE_FORMATS: Final = ("timm", "mmselfsup_rgb")

#: What #12 owes, named in the committed file rather than absent from it.
PENDING: Final = {
    "terramind": {
        "ticket": 12,
        "encoders": ["terramind", "terramind-tokenizer"],
        "owes": [
            "the named v1_pretraining_mean/std[\"tok_dem@224\"] lookup, and its value checked "
            "against 670.665 / 951.272 (ADR-0012 §7.2, #138)",
            "ADR-0011's rederived_14x14_crop pos_emb derivation as a checked declared field",
            "the fp32-versus-compute-dtype agreement, which ADR-0012 §6 names as unmeasured "
            "for this arm rather than inheriting the DEMAE family's 0.999968",
        ],
    }
}


# --- What comes back ----------------------------------------------------------


class Features(NamedTuple):
    """One forward pass through a frozen encoder, in ADR-0012 §5's canonical layout.

    A :class:`~typing.NamedTuple` so it survives `torch.compile` and so a consumer that only
    wants ``output`` can unpack it without learning a class — the same choice
    :class:`pretrain.model.MaeOutput` makes.
    """

    #: ``(N, 196, D)`` — **this encoder's own final tensor**, which Task 1's §4.1 head and Task
    #: 2's linear probes read. Normed for DEMAE, identical to block twelve for TerraMind, and
    #: never assumed to be either: :attr:`EncoderSpec.output_is_normed` declares which.
    output: Any
    #: Twelve ``(N, 196, D)`` tensors in block order, ``intermediates[i]`` being the output of
    #: block ``i + 1``. Task 2's UperNet selects its own indices; they are deliberately *not*
    #: narrowed to its current {3, 6, 9, 12}, because §4.2 calls layer choice a known
    #: sensitivity and hard-coding four indices here would foreclose Task 2's own sweep.
    intermediates: tuple
    #: ``(N, D)`` where the architecture has a CLS token and ``None`` where it structurally has
    #: none. ADR-0012 §5: it never enters a cross-encoder comparison, because a statistic pooled
    #: over it would compare one model's token to the other's absence.
    cls: Any = None


# --- The committed file -------------------------------------------------------


def _plain(value: Any) -> Any:
    """JSON-native view of a field: tuples become lists, dataclasses their own dicts."""
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, EncoderSpec):
        return value.to_dict()
    return value


def _fields_of(cls: type, d: dict, *, where: str) -> dict:
    """Check a plain dict names exactly ``cls``'s fields, and hand it back.

    :func:`benchmark.config._from_dict`'s rule, one artifact along: a spec key that is silently
    dropped is a contract clause Task 2 thinks it re-read and did not, and a *missing* one is a
    clause it never read at all.
    """
    if not isinstance(d, dict):
        raise ValueError(f"{where} is a block of fields, not a value; got {d!r}")
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(d) - known)
    if unknown:
        raise ValueError(f"unknown {where} field(s) {unknown}; known: {sorted(known)}")
    missing = sorted(known - set(d))
    if missing:
        raise ValueError(f"{where} is missing field(s) {missing}")
    return d


@dataclass(frozen=True)
class EncoderSpec:
    """One encoder's half of the contract — its identity, its diet, and its output shape.

    Every field is something a wrong value for would produce a **plausible tensor and no
    exception**, which is the selection rule: the axis order trains fine wrong (P102 §4), the
    normalization returns a well-shaped result from a distribution the encoder has never seen
    (p118 §7), and a swapped checkpoint scores a whole arm under another arm's weights.
    """

    name: str
    family: str
    #: Feature width ``D``.
    width: int
    #: Transformer blocks, and therefore :attr:`Features.intermediates`' length.
    depth: int
    #: ADR-0011 / P102's axis order, checked against the checkpoint's own ``pos_embed``.
    pos_encoding: str
    #: §2.1's declared diet. ``per_tile`` recomputes from the passed mask; ``terramind_v1`` is
    #: the global affine #12 owns.
    normalization: str
    #: How the file is laid out — one of :data:`READABLE_FORMATS`, or a format this ticket
    #: declares and does not read.
    checkpoint_format: str
    #: The published architecture, as a consumer types it into ``timm.create_model``.
    timm_model: str
    #: The authoritative file's md5, quoted from the registry. ``None`` means **unpinned** — this
    #: plan publishes no md5 for this encoder yet — and extraction refuses rather than running
    #: against whatever happens to sit at the path.
    #:
    #: Deliberately *not* "whatever this box had at freeze time". An observed md5 would make the
    #: committed file's bytes a fact about the box that froze it, and the spec's md5 goes into
    #: every run record — a hash that identified the freezing box as much as the contract is the
    #: recipe/environment line (:mod:`benchmark.config`) crossed in the one artifact that must
    #: not cross it. Pinning a new encoder is therefore a **registry change**: one reviewed line
    #: in :data:`benchmark.arms.ENCODERS`, which is what "authoritative" ought to cost.
    checkpoint_md5: str | None
    #: Whether ``output`` is ``intermediates[-1]`` put through a final norm. **The field-level
    #: note ADR-0012 §3 asks for**, so nobody discovers DEMAE's final norm by reading tensors.
    output_is_normed: bool
    #: Whether the architecture has a CLS token at all.
    has_cls: bool
    #: What the resolution line and this file's reader see about that distinction.
    note: str

    def to_dict(self) -> dict:
        return {f.name: _plain(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_dict(cls, d: dict, *, where: str) -> EncoderSpec:
        return cls(**_fields_of(cls, d, where=where))


def _encoder_note(encoder: arms.Encoder) -> str:
    """The field-level note, written from the fact rather than copied per entry."""
    if encoder.output_is_normed:
        return (
            f"`output` is {ADR} §3's own final tensor: block {encoder.depth} put through this "
            f"encoder's final norm. It is **not** `intermediates[{encoder.depth - 1}]`, and the "
            f"two are separately named so nobody discovers the norm by reading tensors."
        )
    return (
        f"`output` is bit-identical to `intermediates[{encoder.depth - 1}]`: this encoder "
        f"applies no norm after its last block (p118 §5.3). The two fields are still separate, "
        f"because on the DEMAE side they are two different tensors."
    )


@dataclass(frozen=True)
class ExtractorSpec:
    """The whole committed contract. ``freeze`` writes it; :func:`load_spec` re-checks it.

    Round-trips through plain JSON, hashes to a stable md5 (canonical separators and sorted
    keys, so re-freezing on another box is byte-identical), and refuses at load rather than
    warning — which is the property that makes it a member of the contract family the selection
    list, the split assignment, the val panel and the probe set are in.
    """

    #: ADR-0012 §3/§5's shape, at the top level because it is one number for every encoder.
    layers: int
    tokens: int
    #: The patch grid's side, 14 — :func:`pretrain.model.sincos2d_pos_embed`'s ``grid_px``, and
    #: named after it. Not to be read as a *width*: :attr:`EncoderSpec.width` is ``D``.
    grid_px: int
    img_px: int
    patch_px: int
    #: ADR-0012 §6 — fp32 is the reference tensor, always. The *compute* dtype is a per-run
    #: declared field and lives in :class:`benchmark.config.Extract`, not here: it may differ
    #: per task, because the measured difference is cosine 0.999968 against #134's 0.99 bar.
    reference_dtype: str
    layout: str
    valid_mask: str
    statistics: str
    #: One entry per registered encoder this contract covers, keyed by registry name.
    encoders: dict[str, EncoderSpec]
    #: Which encoder each registered arm reads — the mapping Task 2 needs and would otherwise
    #: have to re-derive from a plan section.
    arms: dict[str, str | None]
    #: What is named as owed rather than absent. See the module docstring.
    pending: dict = field(default_factory=lambda: copy.deepcopy(PENDING))

    # --- serialization

    def to_dict(self) -> dict:
        out: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name == "encoders":
                out[f.name] = {k: v.to_dict() for k, v in value.items()}
            else:
                out[f.name] = _plain(value)
        out["adr"] = ADR
        return out

    @classmethod
    def from_dict(cls, d: dict) -> ExtractorSpec:
        d = {k: v for k, v in d.items() if k != "adr"}
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(d) - known)
        if unknown:
            raise ValueError(f"unknown spec field(s) {unknown}; known: {sorted(known)}")
        missing = sorted(known - set(d) - {"pending"})
        if missing:
            raise ValueError(f"{SPEC_FILENAME} is missing field(s) {missing}")
        encoders = {
            name: EncoderSpec.from_dict(body, where=f"encoders.{name}")
            for name, body in d["encoders"].items()
        }
        return cls(**{**d, "encoders": encoders})

    def to_json(self) -> str:
        """The committed bytes: sorted keys, two-space indent, LF-terminated.

        Byte-identical on a re-freeze, which is what makes :meth:`md5` a contract rather than a
        timestamp — the property :mod:`pretrain.artifact` states for its own two-column siblings.
        """
        return json.dumps(self.to_dict(), sort_keys=True, indent=2) + "\n"

    def md5(self) -> str:
        """This contract's md5 — what the run record carries beside the table it produced."""
        return hashlib.md5(self.to_json().encode("utf-8")).hexdigest()

    # --- the self-check

    def check(self) -> ExtractorSpec:
        """Re-check the contract against the ADR and against the live registry, or refuse.

        Torch-free and cheap, so a consumer pays nothing to have it run on every load. What it
        catches is the *drift* case: a spec whose declared shape, arm map or encoder identity no
        longer describes the code that claims to implement it. Warning there would leave two
        tasks computing different features and both of them reporting success.
        """
        if self.layers != LAYERS:
            raise ValueError(
                f"{SPEC_FILENAME} declares {self.layers} layers; {ADR} §3 emits all {LAYERS} "
                f"block outputs plus a separate `output`"
            )
        if self.tokens != dataset_config.TOKENS_PER_TILE:
            raise ValueError(
                f"{SPEC_FILENAME} declares {self.tokens} tokens; {ADR} §5's canonical layout is "
                f"(N, {dataset_config.TOKENS_PER_TILE}, D) at the corpus's "
                f"{dataset_config.TILE_PX}/{dataset_config.PATCH_PX} grid"
            )
        if (self.img_px, self.patch_px) != (dataset_config.TILE_PX, dataset_config.PATCH_PX):
            raise ValueError(
                f"{SPEC_FILENAME} declares a {self.img_px}/{self.patch_px} tile; the corpus is "
                f"{dataset_config.TILE_PX}/{dataset_config.PATCH_PX}"
            )
        if self.grid_px**2 != self.tokens:
            raise ValueError(
                f"{SPEC_FILENAME}'s {self.grid_px}x{self.grid_px} grid is not "
                f"{self.tokens} tokens"
            )
        if self.reference_dtype != "fp32":
            raise ValueError(f"{ADR} §6: fp32 is the reference tensor, not {self.reference_dtype!r}")
        # The clause Task 2 re-reads most directly, and therefore the one least worth emitting
        # unchecked: a spec declaring a `(D, 14, 14)` layout would load clean and every consumer
        # would unpack the tensors the other way round.
        if self.layout != CANONICAL_LAYOUT:
            raise ValueError(
                f"{ADR} §5's canonical layout is {CANONICAL_LAYOUT!r}; this spec says "
                f"{self.layout!r}"
            )
        if self.valid_mask != VALID_MASK:
            raise ValueError(f"{ADR} §1: valid = land ∧ ¬void, not {self.valid_mask!r}")
        if self.statistics != STATISTICS:
            raise ValueError(
                f"{ADR} §2 recomputes statistics from the passed mask; this spec says "
                f"{self.statistics!r}"
            )

        # Against the live registry. A spec that has drifted from `benchmark.arms` describes an
        # extractor this code does not implement, and Task 2 loading it would be re-reading a
        # contract nothing honours.
        if set(self.arms) != set(arms.ARMS):
            raise ValueError(
                f"{SPEC_FILENAME}'s arm map is not the registry's: "
                f"missing {sorted(set(arms.ARMS) - set(self.arms))}, "
                f"extra {sorted(set(self.arms) - set(arms.ARMS))}"
            )
        for name, encoder_name in self.arms.items():
            if encoder_name != arms.ARMS[name].encoder:
                raise ValueError(
                    f"{SPEC_FILENAME} has arm {name} reading {encoder_name!r}; the registry has "
                    f"it reading {arms.ARMS[name].encoder!r}"
                )
        for name, entry in self.encoders.items():
            registered = arms.ENCODERS.get(name)
            if registered is None:
                raise ValueError(f"{SPEC_FILENAME} declares unregistered encoder {name!r}")
            declared = (
                entry.family, entry.width, entry.depth, entry.pos_encoding,
                entry.checkpoint_format, entry.timm_model, entry.output_is_normed, entry.has_cls,
            )
            actual = (
                registered.family, registered.width, registered.depth, registered.pos_encoding,
                registered.checkpoint_format, registered.timm_model,
                registered.output_is_normed, registered.has_cls,
            )
            if declared != actual:
                raise ValueError(
                    f"{SPEC_FILENAME}'s {name} entry has drifted from the registry: "
                    f"declared {declared}, registered {actual}"
                )
            if entry.depth != self.layers:
                raise ValueError(
                    f"{SPEC_FILENAME}'s {name} declares {entry.depth} blocks against the "
                    f"contract's {self.layers}"
                )
            if entry.normalization != arms.normalization_for(name):
                raise ValueError(
                    f"{SPEC_FILENAME} feeds {name} a {entry.normalization!r} diet; the arms "
                    f"reading it declare {arms.normalization_for(name)!r} (§2.1)"
                )
            # The md5 is the registry's or it is nothing — see `EncoderSpec.checkpoint_md5` on
            # why a spec is never allowed to pin one a box observed for itself.
            if entry.checkpoint_md5 != registered.checkpoint_md5:
                raise ValueError(
                    f"{SPEC_FILENAME} pins {name} at {entry.checkpoint_md5!r}; the registry "
                    f"publishes {registered.checkpoint_md5!r}"
                )
        return self

    # --- what a consumer asks it

    def encoder_for(self, arm: str) -> EncoderSpec:
        """The contract clause for one arm's encoder, or the refusal that says why there is none."""
        if arm not in self.arms:
            raise ValueError(f"unknown arm {arm!r}; registered: {sorted(self.arms)}")
        name = self.arms[arm]
        if name is None:
            raise ValueError(
                f"arm {arm} runs no encoder ({arms.ARMS[arm].what}), so it has no frozen-feature "
                f"contract to read"
            )
        entry = self.encoders.get(name)
        if entry is None:
            owed = next(
                (block for block in self.pending.values() if name in block.get("encoders", ())),
                None,
            )
            if owed is not None:
                raise NotImplementedError(
                    f"arm {arm} reads the {name} encoder, whose half of {ADR} is ticket "
                    f"#{owed['ticket']} and is named as owed in {SPEC_FILENAME} rather than "
                    f"absent from it: {'; '.join(owed['owes'])}"
                )
            raise ValueError(f"{SPEC_FILENAME} has no entry for encoder {name!r}")
        return entry


# --- Writing it ---------------------------------------------------------------


def build_spec(cfg: BenchmarkConfig | None = None) -> ExtractorSpec:
    """The contract, assembled from the registry. **Its content is a function of the code.**

    Nothing a box happens to have mounted reaches these bytes, which is what lets the committed
    file be byte-identical everywhere and lets its md5 mean *which contract* rather than *whose
    freeze* — see :attr:`EncoderSpec.checkpoint_md5`.

    A mounted file *is* read, and for one thing only: a published md5 that the file at the
    configured path does not match is refused here rather than at load. A `freeze` that pinned
    the wrong checkpoint would hand Task 2 a contract naming weights nobody ran, and the moment
    to catch that is while the file is being written.
    """
    cfg = BenchmarkConfig() if cfg is None else cfg
    from .artifacts import file_md5

    entries: dict[str, EncoderSpec] = {}
    for name, encoder in arms.ENCODERS.items():
        if encoder.checkpoint_format not in READABLE_FORMATS:
            continue  # named in `pending` instead — see the module docstring on #12
        path = getattr(cfg.paths, encoder.checkpoint_field)
        observed = None if encoder.checkpoint_md5 is None else file_md5(path)
        if observed is not None and observed != encoder.checkpoint_md5:
            raise ValueError(
                f"paths.{encoder.checkpoint_field} is {path}, whose md5 is {observed}; the "
                f"{name} encoder this plan benchmarks is {encoder.checkpoint_md5}. Freezing "
                f"this would commit a contract naming weights nobody ran ({ADR} §7.1)."
            )
        entries[name] = EncoderSpec(
            name=name,
            family=encoder.family,
            width=encoder.width,
            depth=encoder.depth,
            pos_encoding=encoder.pos_encoding,
            normalization=arms.normalization_for(name),
            checkpoint_format=encoder.checkpoint_format,
            timm_model=encoder.timm_model,
            checkpoint_md5=encoder.checkpoint_md5,
            output_is_normed=encoder.output_is_normed,
            has_cls=encoder.has_cls,
            note=_encoder_note(encoder),
        )
    return ExtractorSpec(
        layers=cfg.extract.layers,
        tokens=cfg.extract.tokens,
        grid_px=dataset_config.TILE_PX // dataset_config.PATCH_PX,
        img_px=dataset_config.TILE_PX,
        patch_px=dataset_config.PATCH_PX,
        reference_dtype=cfg.extract.reference_dtype,
        layout=CANONICAL_LAYOUT,
        valid_mask=cfg.extract.valid_mask,
        statistics=STATISTICS,
        encoders=entries,
        arms={name: arm.encoder for name, arm in arms.ARMS.items()},
    ).check()


def spec_path(cfg: BenchmarkConfig) -> Path:
    """Where this config's committed contract lives."""
    return Path(cfg.paths.artifacts_dir) / SPEC_FILENAME


def write_spec(cfg: BenchmarkConfig | None = None) -> Path:
    """Write `extractor_spec.json`, idempotently, refusing to overwrite a moved one (§3.4).

    Re-running `freeze` after a code change must not silently re-draw an input a half-finished
    run is already scoring against, so an existing file whose content differs is a **refusal**
    naming both md5s. Deleting it is the deliberate act that says the contract really did change
    — and since the spec's md5 is in every run record, the tables produced under the old one
    stay attributable.

    The refusal only ever fires on a *code* change, because :func:`build_spec`'s output does not
    depend on the box. That is the point of the split: mounting a checkpoint never puts a build
    session in the position of being told to delete a committed artifact that is still correct.
    """
    cfg = BenchmarkConfig() if cfg is None else cfg
    target = spec_path(cfg)
    body = build_spec(cfg).to_json()
    if target.is_file():
        existing = target.read_text()
        if existing == body:
            return target
        raise FileExistsError(
            f"{target} is already frozen at md5 "
            f"{hashlib.md5(existing.encode('utf-8')).hexdigest()} and this config would write "
            f"{hashlib.md5(body.encode('utf-8')).hexdigest()}. `freeze` does not overwrite a "
            f"committed input whose content hash has moved (§3.4); delete it deliberately if "
            f"the contract really has changed."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body)
    return target


def load_spec(path: Path | str) -> ExtractorSpec:
    """Read the committed contract back and re-check it — the whole point of the file.

    This is the call Task 2 makes. It refuses rather than warning, on every clause
    :meth:`ExtractorSpec.check` covers.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is not on this box — run `python -m benchmark.launch freeze` (§4.5 step 1; "
            f"it is CPU-only and local)"
        )
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not a readable {SPEC_FILENAME}: {exc}") from None
    return ExtractorSpec.from_dict(payload).check()


# --- ADR-0012 Decision 2: the statistics, recomputed --------------------------


def tile_statistics(raw_i16: np.ndarray, valid_mask: np.ndarray) -> tuple[float, float]:
    """Per-tile ``(mean, std)`` in metres over the passed valid pixels. The only code path.

    float64 accumulation, which is how [p137](../../docs/research/p137-extractor-contract-checks.md)
    §1 measured the gap against the manifest's float32-accumulated columns: max **3.637e-4 m**
    on the mean over 500 real tiles, max **1.059e-7 relative** on the std. Float32 accumulation
    order, not a definitional gap — recomputing from the land mask *is* the ETL's land
    definition — and ADR-0012 Decision 2 ships that equivalence as a test so an ETL drift
    surfaces in Task 1 rather than in Task 2.
    """
    land = raw_i16[valid_mask].astype(np.float64)
    if land.size == 0:
        raise ValueError(
            "no valid pixels in this tile: valid = land ∧ ¬void (ADR-0012 §1), and a tile whose "
            "mask is empty has no per-tile statistic to normalize by"
        )
    return float(land.mean()), float(land.std())


def _as_batch(raw_i16, valid_mask) -> tuple[np.ndarray, np.ndarray]:
    """The door ADR-0012 Decision 1 closes: int16 metres and a boolean mask, or a refusal.

    Type-checked rather than tested for, because the failure it prevents is completely silent —
    the encoder returns a well-shaped, plausible tensor from an input distribution it has never
    seen (p118 §7). The pretraining loader's ``image`` is a float32 torch tensor of per-tile
    standardized units, so it does not get past this function, and "handing the extractor
    pre-normalized input" stops being a mistake anyone can make.
    """
    if not isinstance(raw_i16, np.ndarray) or raw_i16.dtype != np.int16:
        kind = (
            f"a {type(raw_i16).__name__}"
            if not isinstance(raw_i16, np.ndarray)
            else f"a numpy array of dtype {raw_i16.dtype}"
        )
        raise TypeError(
            f"{ADR} §1: the extractor takes **raw int16 metres** and applies its own declared "
            f"normalization; it was handed {kind}. If this is `dataset.loader`'s `image`, it is "
            f"already per-tile standardized and an encoder would accept it silently (p118 §7) — "
            f"pass the shard's raw int16 tile and the valid-pixel mask instead."
        )
    if not isinstance(valid_mask, np.ndarray) or valid_mask.dtype != np.bool_:
        raise TypeError(
            f"{ADR} §1: valid = land ∧ ¬void, a boolean mask; got "
            f"{getattr(valid_mask, 'dtype', type(valid_mask).__name__)}"
        )
    raw = raw_i16[None] if raw_i16.ndim == 2 else raw_i16
    mask = valid_mask[None] if valid_mask.ndim == 2 else valid_mask
    if raw.ndim != 3:
        raise ValueError(f"expected a (H, W) tile or an (N, H, W) batch; got {raw_i16.shape}")
    if mask.shape != raw.shape:
        raise ValueError(f"mask {valid_mask.shape} does not match tiles {raw_i16.shape}")
    tile = dataset_config.TILE_PX
    if raw.shape[1:] != (tile, tile):
        raise ValueError(f"expected {tile}x{tile} tiles; got {raw.shape[1:]}")
    return raw, mask


def normalize(raw: np.ndarray, mask: np.ndarray, *, normalization: str) -> np.ndarray:
    """Apply the encoder's **declared** diet — the whole of what §2.1 says is per-arm.

    ``per_tile`` standardizes by statistics recomputed from the passed mask (Decision 2), one
    code path for Task 1's non-void statistics and Task 2's clean-tile statistics.
    """
    if normalization != "per_tile":
        raise NotImplementedError(
            f"normalization {normalization!r} is not this ticket's — see {SPEC_FILENAME}'s "
            f"`pending` block and the module docstring"
        )
    out = np.empty(raw.shape, dtype=np.float32)
    for i in range(raw.shape[0]):
        mean, std = tile_statistics(raw[i], mask[i])
        out[i] = (raw[i].astype(np.float32) - np.float32(mean)) / np.float32(max(std, STD_FLOOR))
    return out


# --- The extractor ------------------------------------------------------------


def _torch_dtype(name: str):
    import torch

    try:
        return {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[name]
    except KeyError:
        raise ValueError(f"unknown dtype {name!r} (fp32 | bf16 | fp16)") from None


def _checkpoint_depth(state) -> int:
    """How many blocks the *checkpoint* actually has, off its own keys.

    Read before the model is built, so a declared layer count that does not match the weights
    is refused **by name** rather than surfacing as two dozen missing keys from a strict load —
    which names every tensor and not the one fact that is wrong.
    """
    indices = set()
    for key in state:
        head, _, rest = key.partition(".")
        if head == "blocks" and rest:
            index, _, _ = rest.partition(".")
            if index.isdigit():
                indices.add(int(index))
    return 1 + max(indices) if indices else 0


class Extractor:
    """The contract, applied: ``extract(raw_i16, valid_mask, *, arm) -> Features``.

    Holds the committed spec and materializes an encoder the first time an arm asks for one, so
    a build session that resolves the contract pays nothing until it actually scores something,
    and a box mounting one checkpoint of three is not blocked by the other two.

    Built as ``Extractor(load_spec(spec_path(cfg)), cfg)`` — two calls rather than a classmethod
    wrapping them, because :func:`load_spec` is the call this module is *about* and hiding it
    behind a constructor makes the re-check look like an implementation detail of building an
    object rather than the thing the committed file exists to do.
    """

    def __init__(
        self,
        spec: ExtractorSpec,
        cfg: BenchmarkConfig | None = None,
        *,
        device: str = "cpu",
    ) -> None:
        self.spec = spec
        self.config = BenchmarkConfig() if cfg is None else cfg
        self.device = device
        self._models: dict[str, Any] = {}

    # --- the encoder, and §7.1's four refusals

    def encoder(self, arm: str):
        """The frozen model this arm reads, checked against the contract on the way in."""
        entry = self.spec.encoder_for(arm)
        model = self._models.get(entry.name)
        if model is None:
            model = self._models[entry.name] = self._build(entry)
        return model

    def _build(self, entry: EncoderSpec):
        import timm
        import torch

        from pretrain.init import read_state_dict
        from pretrain.model import AXIS_ORDERS, sincos2d_pos_embed

        registered = arms.ENCODERS[entry.name]
        path = getattr(self.config.paths, registered.checkpoint_field)
        if path is None:
            raise FileNotFoundError(
                f"the {entry.name} encoder is not mounted on this box — set "
                f"paths.{registered.checkpoint_field}"
            )
        if not Path(path).is_file():
            raise FileNotFoundError(
                f"the {entry.name} encoder is missing at {path} — check "
                f"paths.{registered.checkpoint_field}"
            )
        if entry.checkpoint_md5 is None:
            raise ValueError(
                f"this plan publishes no md5 for {entry.name}, so nothing distinguishes the file "
                f"at {path} from any other. Pin it in `benchmark.arms.ENCODERS` and re-freeze — "
                f"an encoder's identity is a reviewed line, not something a box certifies for "
                f"itself by having mounted it ({ADR} §7.1)."
            )
        from .artifacts import file_md5

        observed = file_md5(path)
        if observed != entry.checkpoint_md5:
            raise ValueError(
                f"{path} has md5 {observed}; {SPEC_FILENAME} declares {entry.name} as "
                f"{entry.checkpoint_md5}. An encoder handed the "
                f"wrong checkpoint returns a plausible tensor and raises nothing, so this "
                f"refuses ({ADR} §7.1)."
            )

        state, _ = read_state_dict(path)
        if entry.checkpoint_format == "mmselfsup_rgb":
            state = _timm_state_from_mmselfsup(state, entry)
        elif entry.checkpoint_format != "timm":
            raise NotImplementedError(
                f"{entry.name}'s checkpoint format {entry.checkpoint_format!r} is not one this "
                f"ticket reads ({', '.join(READABLE_FORMATS)}) — see {SPEC_FILENAME}'s `pending`"
            )

        depth = _checkpoint_depth(state)
        if depth != entry.depth:
            raise ValueError(
                f"{path} has {depth} transformer blocks; {SPEC_FILENAME} declares "
                f"{entry.depth} for {entry.name}. {ADR} §3 emits one intermediate per block, so "
                f"a layer count that does not match the model loaded is refused."
            )
        pos_embed = state.get("pos_embed")
        if pos_embed is None:
            raise ValueError(
                f"{path} has no `pos_embed`: it is the one tensor this loader verifies rather "
                f"than trusts, and without it the spatial convention is unchecked (P102 §4)"
            )
        expected_shape = (1, 1 + self.spec.tokens, entry.width)
        if tuple(pos_embed.shape) != expected_shape:
            raise ValueError(
                f"{path}'s `pos_embed` is {tuple(pos_embed.shape)}; {SPEC_FILENAME} declares "
                f"{entry.name} at {expected_shape} — {ADR} §5's (N, {self.spec.tokens}, "
                f"{entry.width}) grid"
            )
        # §2.3's R4-vs-R5 precondition, enforced rather than remembered: the two axis orders are
        # each other's transpose over the patch grid, every name and every shape matches under
        # either, and a wrong one makes the encoder read every tile as its own transpose.
        if entry.pos_encoding not in AXIS_ORDERS:
            # ADR-0011's `rederived_14x14_crop` is the case, and it has a different oracle: a
            # centre-anchored crop of a 16x16 grid, not a 14x14 sin-cos. Naming it beats a
            # `KeyError` from a dict lookup one line down.
            raise NotImplementedError(
                f"{entry.name} declares pos_emb derivation {entry.pos_encoding!r}, which is not "
                f"one of {sorted(AXIS_ORDERS)} and needs its own oracle — see "
                f"{SPEC_FILENAME}'s `pending` block"
            )
        reference = sincos2d_pos_embed(
            entry.width, self.spec.grid_px, axis_order=AXIS_ORDERS[entry.pos_encoding]
        )
        if not torch.allclose(pos_embed.float(), reference.float(), atol=1e-5):
            other = {v: k for k, v in AXIS_ORDERS.items()}
            matched = next(
                (
                    name
                    for order, name in other.items()
                    if torch.allclose(
                        pos_embed.float(),
                        sincos2d_pos_embed(
                            entry.width, self.spec.grid_px, axis_order=order
                        ).float(),
                        atol=1e-5,
                    )
                ),
                None,
            )
            raise ValueError(
                f"{path}'s `pos_embed` is not sin-cos under the declared {entry.pos_encoding!r} "
                f"axis order"
                + (f" — it is {matched!r}" if matched else "")
                + f". The orders are each other's transpose and a wrong one trains fine "
                f"(ADR-0011, P102 §4), so it is checked here ({ADR} §7.1)."
            )

        model = timm.create_model(
            entry.timm_model,
            pretrained=False,
            in_chans=1,
            # `pretrain.release.TIMM_CREATE_KWARGS`'s pair, and the same reason: the released
            # artifact is a headless backbone, and `global_pool=""` is what keeps
            # `forward_features`' sequence from being pooled away.
            num_classes=0,
            global_pool="",
        )
        model.load_state_dict(dict(state), strict=True)
        if len(model.blocks) != entry.depth:
            raise ValueError(
                f"timm's {entry.timm_model} built {len(model.blocks)} blocks against "
                f"{SPEC_FILENAME}'s declared {entry.depth}"
            )
        return model.eval().to(self.device)

    # --- ADR-0012's one call

    def extract(self, raw_i16, valid_mask, *, arm: str) -> Features:
        """Raw int16 metres and a valid-pixel mask in; ADR-0012 §5's canonical layout out.

        The compute dtype is :attr:`benchmark.config.Extract.compute_dtype` and the returned
        tensors are always fp32 — §6's reference — because a task that moves its compute dtype
        re-checks it under #134's 0.99 materiality bar rather than changing what the contract
        hands back.
        """
        import torch

        # The input is checked and normalized *before* the encoder is materialized, so a
        # pre-normalized tensor costs a type check rather than a 340 MB load. Which is also the
        # order that makes the refusal legible: it is about the argument, not about the box.
        entry = self.spec.encoder_for(arm)
        raw, mask = _as_batch(raw_i16, valid_mask)
        x = normalize(raw, mask, normalization=entry.normalization)
        model = self.encoder(arm)
        tensor = torch.from_numpy(x).unsqueeze(1).to(self.device)

        compute = self.config.extract.compute_dtype
        with torch.no_grad():
            if compute == self.spec.reference_dtype:
                tokens, intermediates = _forward_blocks(model, tensor)
            else:
                with torch.autocast(torch.device(self.device).type, dtype=_torch_dtype(compute)):
                    tokens, intermediates = _forward_blocks(model, tensor)

        # The prefix is the CLS token where there is one and nothing where there is not, so the
        # canonical (N, 196, D) slice is the same expression on both sides rather than a branch.
        prefix = tokens.shape[1] - self.spec.tokens
        if prefix != int(entry.has_cls):
            raise ValueError(
                f"{entry.name} returned {tokens.shape[1]} tokens against {ADR} §5's "
                f"{self.spec.tokens} patch tokens plus {int(entry.has_cls)} CLS"
            )
        return Features(
            output=tokens[:, prefix:].float(),
            intermediates=tuple(block[:, prefix:].float() for block in intermediates),
            cls=tokens[:, 0].float() if entry.has_cls else None,
        )


def _forward_blocks(model, x):
    """timm's ``forward_features``, unrolled so every block's output is kept.

    Deliberately unrolled rather than routed through ``get_intermediate_layers``: this is the
    exact composition :meth:`pretrain.model.MaskedAutoencoder._transform` runs — patch embed,
    CLS and position, twelve blocks, final norm — so ``output`` is the tensor Phase 1 measured
    and not a helper's interpretation of it. Pinned bit-identical to ``forward_features`` by
    ``test_the_unrolled_pass_is_timms_own_forward_features``.
    """
    x = model.patch_embed(x)
    x = model._pos_embed(x)
    x = model.patch_drop(x)
    x = model.norm_pre(x)
    intermediates = []
    for block in model.blocks:
        x = block(x)
        intermediates.append(x)
    return model.norm(x), tuple(intermediates)


def _timm_state_from_mmselfsup(state, entry: EncoderSpec) -> dict:
    """The published ImageNet MAE, as the single-channel timm encoder A6 reads.

    The two things worth not having twice — P102's key map and §8's patch-embed collapse — are
    **imported from** :mod:`pretrain.init`: R4 warm-started through exactly this translation,
    every row of that map is a checked claim in P102 §3, and a second copy would be a second
    place for `backbone.ln1` to be mistaken for a block's.

    The two guards around them *are* restated, and the honest reason is that their originals in
    :func:`pretrain.init.load_mmselfsup_mae_encoder` are written against a
    :class:`~pretrain.model.MaskedAutoencoder` — they check a live model's parameters, and there
    is no live model here, only a state dict on its way into timm. The wording is deliberately
    the same so a reader who has seen one recognizes the other.

    ``sum`` and not ``mean``: for an input identical in every channel — which is what a
    grayscale image replicated to RGB is — the summed kernel reproduces the original
    convolution exactly, and the mean reproduces a third of it.
    """
    from pretrain.init import (
        ENCODER_PREFIX,
        POS_EMBED_KEY,
        collapse_input_channels,
        encoder_key_map,
    )

    mapping = encoder_key_map(entry.depth)
    missing = sorted(key for key in mapping if key not in state)
    if missing:
        raise ValueError(
            f"the {entry.name} checkpoint is not the mmselfsup MAE this contract declares: "
            f"{len(missing)} mapped key(s) absent, first {missing[:3]}"
        )
    patch_key = f"{ENCODER_PREFIX}patch_embed.projection.weight"
    out = {}
    for theirs, ours in mapping.items():
        tensor = state[theirs]
        if theirs == patch_key:
            channels = int(tensor.shape[1])
            if channels != 3:
                raise ValueError(
                    f"{patch_key} reads {channels} channels; §8's collapse is written for the "
                    f"3-channel RGB kernel an ImageNet model has"
                )
            tensor = collapse_input_channels(tensor, adapt="sum")
        out[ours] = tensor
    if POS_EMBED_KEY not in state:
        raise ValueError(
            f"the {entry.name} checkpoint has no {POS_EMBED_KEY}: it is the one tensor this "
            f"loader verifies rather than copies (P102 §4)"
        )
    out["pos_embed"] = state[POS_EMBED_KEY]
    return out
