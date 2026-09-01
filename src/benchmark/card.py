"""§3.4's dataset card — what `freeze` produced, written for a reader (#154).

The last artifact `freeze` writes, and the only one written for a person rather than for a later
stage. Everything else in ``artifacts_dir`` is read back by code: the tile lists by the mask
stages and by every arm, the extractor spec by Task 2, the bar by `run`. This one is read by
whoever arrives at the benchmark and has to work out what it *is* before they can trust a number
it produced.

## Two things it states rather than lets a reader infer

**The module commits tile lists and masks, and never tiles.** The artifacts index the corpus;
they do not copy it. That is what keeps them freely shippable under all three sets of terms
below, so it is load-bearing rather than a nicety — nothing here is a redistribution question,
and a reader who assumed otherwise would be asking one.

**The train mask bank is not in this repository.** It is 1,000,000 RLE rows and 104 MB, out of
git on the same rule the corpus's shards follow, and what git carries in its place is
``train_mask_bank.json`` — the bank's md5, a parquet-independent hash of the draw, and the
generator config hash. `run` refuses a bank whose md5 has moved rather than reading it. A reader
who assumes the bytes are in git draws the wrong conclusion about the determinism contract:
what is committed is the mask sequence's **hash**, not its bytes, and §3.4 resolved that
explicitly rather than leaving it implicit for exactly this reason.

## Why this is not written through :func:`benchmark.artifacts.write_committed`

Every other committed input goes through that guard, whose contract is *this file already holds
what I would write, or it is a refusal*. The card cannot honour it, and the reason is the same
one :class:`benchmark.masks.BankSidecar` gives for its own local write: **two of the md5s the
card carries are this box's encoding of a draw rather than the draw**. Both mask artifacts are
parquet, whose footer carries the writing library's own version string, so a bank regenerated
under a newer pyarrow hashes differently having drawn the identical 1,000,000 masks. Under the
shared guard that would be a refusal about the environment wearing the words of one about the
recipe; under a ``holds`` that looked past it, the committed card would keep an md5 the bank no
longer has, which is worse — a document whose whole job is to say what the artifacts hash to.

So the card **re-renders**, and the refusal §3.4 asks for is not lost: it has already fired. The
card runs last, downstream of five stages that each carry the guard, so by the time it renders,
`freeze` has proven every artifact it describes is the one committed. What is left for this
module is to not churn — an unchanged box rewrites nothing — and to refuse the one disagreement
only it can see: two mask artifacts carrying **different** generator config hashes, which would
otherwise be published under a single hash belonging to neither.

## What is deliberately absent

No path, no timestamp, no generation time, no git SHA. Every one of them would move the card's
md5 on a box that changed nothing, and the file git carries would become a record of whose
laptop ran `freeze` last. Two boxes freezing one config write the same bytes — save for
the two parquet md5s, which are this box's encoding of the draw rather than the draw itself.
That property is asserted, because it is the one this module's write semantics rest on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Final

from pretrain.artifact import read_tile_strata

from . import masks
from .artifacts import (
    EVAL_MASKS_NAME,
    TEST_TILES_NAME,
    TILE_LIST_NAMES,
    VAL_TILES_NAME,
    file_md5,
)
from .config import BenchmarkConfig

#: What `freeze` writes into ``paths.artifacts_dir``. Named here rather than in
#: :mod:`benchmark.artifacts` beside the inputs `run` resolves, because the card is not one:
#: nothing a results table is answerable to changes with it, and gating a rented run on a
#: document would be a refusal about documentation. It is committed for the reader's sake and
#: read by nobody in this package.
CARD_NAME: Final = "dataset_card.md"

#: §3.1's four draws in their five committed files, in the order the card lists them: what each
#: file is called, what draw it is, and which split it came from. A table rather than five
#: paragraphs, and the names are :mod:`benchmark.artifacts`' own so a rename cannot leave the
#: card describing a file that is no longer there.
TILE_LISTS: Final = (
    (TILE_LIST_NAMES["20k"], "head-train (and the label-efficiency ladder's top rung)", "train"),
    (TILE_LIST_NAMES["5k"], "label-efficiency, 5k rung — nested in the 20k draw", "train"),
    (TILE_LIST_NAMES["1k"], "label-efficiency, 1k rung — nested in the 5k rung", "train"),
    (VAL_TILES_NAME, "val — early stopping and the head LR pick", "val"),
    (TEST_TILES_NAME, "test — every scored instance", "test"),
)


@dataclass(frozen=True)
class CommittedList:
    """One tile list as the card found it: what it is, how many rows, and what it hashes to."""

    name: str
    what: str
    split: str
    rows: int
    md5: str


def _read_lists(directory: Path) -> tuple[CommittedList, ...]:
    """§3.1's five files, read back off disk rather than taken from the draw or the config.

    The contract family's one rule (ADR-0005 §5): **the committed file is what later phases
    re-read**. The card is the first thing to re-read them, so it obeys the rule its own text
    states — a card that printed ``tiles.head_train`` would be describing the recipe while
    claiming to describe the draw, and the two are only equal because #149's draw refuses a
    thin pool rather than accepting one short.

    A missing file is a refusal naming it, not a row left out: a partly-frozen box must not
    produce a document that reads as a complete one.
    """
    committed = []
    for name, what, split in TILE_LISTS:
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} is not on this box, and §3.4's card documents §3.1's four draws in "
                f"their five files. `python -m benchmark.launch freeze` writes them before it "
                f"reaches this stage, so this is a partly-frozen box rather than a missing step"
            )
        # A short noun, not this row's `what`: the parameter names the artifact in the
        # refusal, and the card's table prose would make one unreadable.
        rows = read_tile_strata(path, what="tile list")
        committed.append(
            CommittedList(
                name=name, what=what, split=split, rows=len(rows), md5=file_md5(path)
            )
        )
    return tuple(committed)


@dataclass(frozen=True)
class DatasetCard:
    """Everything the card says, assembled — and the one disagreement only it can see.

    Holds the two mask stages' own return values rather than copying numbers out of them, so
    there is one description of each artifact and the stage that wrote it owns it.
    """

    config: BenchmarkConfig
    tile_lists: tuple[CommittedList, ...]
    eval_masks: masks.EvalMaskList
    bank: masks.TrainMaskBank

    def __post_init__(self) -> None:
        # §3.4's determinism contract is **one** contract: the two mask artifacts are generated
        # from the same knobs and hash together. A card is the one place that single hash gets
        # published, so a card over two draws that do not share it would put a reader's name to
        # a claim neither artifact makes. Refused here rather than rendered with two hashes,
        # because two hashes is not a card — it is a bug report.
        drawn = self.bank.committed.draw
        if self.eval_masks.artifact_hash != drawn.artifact_hash:
            raise ValueError(
                f"the eval mask list was frozen under artifact_hash "
                f"{self.eval_masks.artifact_hash} and the train mask bank under "
                f"{drawn.artifact_hash}. §3.4's two mask artifacts carry **one generator config "
                f"hash**, and a dataset card cannot publish one hash over two draws that do not "
                f"share it — re-freeze the artifact that moved"
            )

    @property
    def artifact_hash(self) -> str:
        """The generator config hash, once — the value both mask artifacts carry."""
        return self.eval_masks.artifact_hash

    def render(self) -> str:
        """The committed document — LF-terminated, and a function of the draw alone.

        Written as Markdown blocks rather than assembled from fragments, because the thing being
        got right here is *prose a person reads*, and prose is only reviewable at the width it
        will be read at. Every quoted licence phrase is kept whole on its own line for the same
        reason one level down: a licence sentence is what a reader is most likely to arrive at by
        grepping, and a quote broken over a newline is a quote that does not answer.
        """
        return "\n".join(
            [
                self._heading(),
                self._what_is_committed(),
                self._tile_lists(),
                self._mask_artifacts(),
                self._real_family(),
                self._data_terms(),
            ]
        )

    # --- the sections

    def _heading(self) -> str:
        return dedent(f"""\
            # DEMAE Task 1 — the void-filling benchmark's dataset card

            What `python -m benchmark.launch freeze` produced, by content. Every md5 and row
            count below is read off the committed artifacts rather than typed beside them, and
            nothing about *where* the freeze ran appears here — no path, no timestamp, no host.
            Two boxes freezing this config write this file byte for byte, save for the two
            parquet md5s: a parquet footer carries the writing library's own version, so those
            two values can move while the draw underneath them does not.

            **Generator config hash — `{self.artifact_hash}`.** Plan §3.4's determinism
            contract is one contract and not two: the eval mask list and the train mask bank
            are generated by `{masks.GENERATOR}` from the same knobs and hash together, so the
            value appears once here and is carried by both.

            The extractor contract (`extractor_spec.json`, ADR-0012) and the pre-registration bar
            (`task1-prereg-bar.json`, ADR-0013) are committed beside these and documented in their
            own decision records. This card is about the data.
            """)

    def _what_is_committed(self) -> str:
        bank, draw = self.bank.committed, self.bank.committed.draw
        return dedent(f"""\
            ## What is committed, and what is not

            **This module commits tile lists and masks, and never tiles.** The artifacts below
            *index* the corpus — a tile list names tile ids, a mask is a run-length encoding of a
            shape — and none of them carries a DEM pixel. Nothing here is therefore a
            redistribution question, which is what keeps the whole set freely shippable under all
            three sets of terms at the foot of this card.

            **The {bank.megabytes:,.0f} MB train mask bank is not in this repository.** It is
            {draw.rows:,} RLE rows, out of git on the same rule the corpus's shards follow rather
            than on any size threshold. `freeze` generates it once and commits
            `{self.bank.sidecar_path.name}` in its place — the bank's md5, a parquet-independent
            hash of the draw itself, and the generator config hash — and `run`
            **refuses a bank whose md5 has moved** rather than reading it.

            The determinism contract is preserved intact by that indirection: the mask sequence a
            run sees is still a committed input, committed as a *hash rather than as bytes*.
            Nothing about getting it wrong is loud otherwise — a head trained against a re-drawn
            bank converges normally and prints a table exactly as wide as a good one.
            """)

    def _tile_lists(self) -> str:
        rows = "\n".join(
            f"| `{lst.name}` | {lst.what} | {lst.split} | {lst.rows:,} | `{lst.md5}` |"
            for lst in self.tile_lists
        )
        return dedent(f"""\
            ## §3.1's four tile draws, in five committed files

            | file | draw | split | rows | md5 |
            | --- | --- | --- | ---: | --- |
            {{rows}}

            The 1k, 5k and 20k rungs are **nested** — 1k ⊂ 5k ⊂ 20k — so the label-efficiency
            sweep measures data volume rather than draw luck. Every draw is proportional over
            the corpus's k=20 strata and written sorted, so a redraw is byte-identical and the
            md5 a run record quotes is a contract rather than a timestamp.
            """).replace("{rows}", rows)

    def _mask_artifacts(self) -> str:
        evaluated, bank = self.eval_masks, self.bank.committed
        draw, geometry = bank.draw, self.config.masks
        synthetic = evaluated.rows - len(evaluated.real.rows)
        return dedent(f"""\
            ## §3.4's two mask artifacts

            ### Eval mask list — `{EVAL_MASKS_NAME}`, committed

            - **md5** `{evaluated.md5}`
            - {evaluated.rows:,} rows of `tile_id, family, bucket, mask_rle, seed`, one per
              scored eval instance
            - {synthetic:,} synthetic, being {evaluated.tiles:,} test tiles ×
              {len(geometry.synthetic_families)} families × {len(geometry.buckets)} size buckets
            - {len(evaluated.real.rows):,} real — §3.3's void shapes transplanted onto
              stratum-matched clean test tiles, cropped and pasted and never resampled

            ### Train mask bank — `{self.bank.path.name}`, **not committed**

            - **md5** `{bank.md5}`, recorded in `{self.bank.sidecar_path.name}` — which *is*
              committed, and is what `run` checks the bank against
            - **draw hash** `{bank.draw_hash}`, a parquet-independent hash of the draw itself, so
              two boxes agree about the bank without either holding the other's bytes
            - {draw.rows:,} rows of `{draw.columns}` — K = {draw.draws_per_tile} draws per
              head-train tile over {draw.tiles:,} tiles, {bank.megabytes:,.0f} MB
            - Written once and read by every arm, seed and hyperparameter config, so per-epoch
              mask variation is preserved exactly while head training does not become loader-bound
            """)

    def _real_family(self) -> str:
        real, geometry = self.eval_masks.real, self.config.masks
        # The real family's bands stay at 128–160 where the synthetic large bucket tops at 158
        # (`BenchmarkConfig.masks`): they describe sizes real voids *have*, measured on SRTM,
        # rather than draw targets the area cap constrains.
        small, medium = geometry.small_px, geometry.medium_px
        large, top = geometry.large_px, geometry.real_band_top_px
        return dedent(f"""\
            ## §3.3's real-void family: what the transplant excluded

            Counts, beside the data, so the published size histogram's cut-offs are documented
            where a reader meets them rather than only in the code that made them. The published
            sub-rows are **{small[0]}–{small[1]}**, **{medium[0]}–{medium[1]}** and
            **{large[0]}–{top}** px of equivalent diameter, so a component that places cleanly
            can still fall into one of two gaps — {small[1]}–{medium[0]} px or
            {medium[1]}–{large[0]} px. The counts are an attrition ladder and subtract exactly,
            in this order:

            - **{real.components_at_or_above_floor:,}** components at or above the floor of
              {geometry.small_px[0]} px, before any exclusion
            - less **{real.excluded_above_top:,}** above the published top of
              {geometry.real_band_top_px} px
            - less **{real.failed_placement:,}** inside the published range that failed
              §3.2's placement tests
            - = **{real.scored_components:,}** scored components
            - less **{real.excluded_in_gaps:,}** that placed cleanly and fell in the gaps
              *between* the published size sub-rows — excluded rather than silently dropped,
              because those boundaries are what keep real-small size-comparable to synthetic small
            - = **{len(real.rows):,}** published rows, transplanted from {real.cells} SRTMGL1N
              granules onto {real.hosts:,} stratum-matched clean test tiles, at most
              {real.max_reuse} on any one host

            The ceiling on large real voids is **this benchmark's tile, not nature**: voids over
            400 px exist in abundance and cannot fit a 224 px tile under §3.2's area cap of
            {geometry.max_area_frac:.0%}. A histogram that simply stopped at the top would imply
            a scarcity that is not there, which is why the tail is counted rather than left as
            silence.
            """)

    def _data_terms(self) -> str:
        return dedent("""\
            ## Data terms, as the distributors state them

            A record of what the distributors publish, and **not legal advice** — the precedent
            `docs/research/p101-mae-checkpoint-licence.md` set for this repository. None of it is
            a redistribution question, for the reason at the top of this card: the module commits
            tile lists and masks, and never tiles.

            **Copernicus GLO-30** — every benchmark tile. Available
            "on a free basis for the general public" under the ESA Copernicus Contributing
            Missions user licence, which requires acceptance and carries an attribution
            requirement. It is *not a standard open licence* such as CC BY. This is the primary
            data behind every arm.

            **SRTMGL1N v003** — the real-void family's mask shapes (§3.3). LP DAAC states that
            the dataset "is openly shared, without restriction, in accordance with" NASA's
            EOSDIS data use policy. That is the distributor's own sentence, quoted rather than
            summarised because the conclusion a summary reaches for is one the distributor does
            not make.

            **The synthetic masks, the mask bank and the tile lists** — this repository's own,
            under its licence.
            """)


def build(
    cfg: BenchmarkConfig,
    *,
    eval_masks: masks.EvalMaskList,
    bank: masks.TrainMaskBank,
) -> DatasetCard:
    """Assemble the card: the tile lists off disk, the two mask artifacts as `freeze` made them.

    The asymmetry is deliberate. The tile lists are **re-read**, because that is the contract
    family's rule and the card is the first consumer to follow it. The mask artifacts are handed
    in, because what the card owes about them — the real family's attrition, the bank's draw hash
    — lives in the stage's own return value and in schema metadata, and a second reader of those
    would be a second answer to what `freeze` just did.
    """
    return DatasetCard(
        config=cfg,
        tile_lists=_read_lists(Path(cfg.paths.artifacts_dir)),
        eval_masks=eval_masks,
        bank=bank,
    )


@dataclass(frozen=True)
class WrittenCard:
    """Where the card is, what it hashes to, and whether this `freeze` moved it."""

    path: Path
    md5: str
    #: ``False`` when the committed card already said exactly this. Reported rather than implied
    #: for the reason :attr:`benchmark.masks.TrainMaskBank.generated` is: the whole point of the
    #: check is that a re-freeze of an unchanged box leaves the tree clean.
    rewritten: bool


def write(rendered: str, path: Path | str) -> WrittenCard:
    """Write the card, leaving an identical one exactly where it was.

    Not :func:`benchmark.artifacts.write_committed` — see the module docstring for why the shared
    guard is the wrong tool here and what fires in its place.
    """
    path = Path(path)
    if path.is_file() and path.read_text() == rendered:
        return WrittenCard(path=path, md5=file_md5(path), rewritten=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered)
    return WrittenCard(path=path, md5=file_md5(path), rewritten=True)


def freeze(
    cfg: BenchmarkConfig,
    *,
    eval_masks: masks.EvalMaskList,
    bank: masks.TrainMaskBank,
) -> WrittenCard:
    """`freeze`'s last stage: the document that says what the other five produced."""
    card = build(cfg, eval_masks=eval_masks, bank=bank)
    return write(card.render(), Path(cfg.paths.artifacts_dir) / CARD_NAME)


def format_written(written: WrittenCard) -> list[str]:
    """What `freeze` says it committed — by content, as every stage above it does."""
    state = "written" if written.rewritten else "unchanged — the box already said this"
    return [
        f"{'committed':<16} the dataset card into {written.path}",
        f"{'':<16} {written.path.name:<21} md5 {written.md5}  ({state})",
        f"{'':<16} §3.1's four tile draws, both mask artifacts and one generator config hash, "
        f"plus the three sets of data terms",
    ]
