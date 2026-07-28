# DEMAE corpus card — what this dataset is, and what it will get wrong

**Ticket:** [Record the E3 honesty notes against the shipped scale corpus (#75)](https://github.com/JStoschek/DEMAE/issues/75) ·
**Map:** [Phase F (#67)](https://github.com/JStoschek/DEMAE/issues/67) ·
**Numbers:** [`f-honesty-notes.json`](f-honesty-notes.json) ·
**Driver:** [`notebooks/f_honesty_notes.py`](../../notebooks/f_honesty_notes.py)

This document is written for someone using the corpus or a model trained on it — not for
the project's own record. Every number in it is derived by the driver above from committed
artifacts, so it can be re-checked rather than trusted, and links point at the evidence
behind each claim.

A Phase-4 model card should be able to lift from this document directly.

---

## 1. What this corpus is

499,934 tiles of Copernicus GLO-30 elevation data, each 224 × 224 pixels — about **6.7 km
across** at roughly 30 m per pixel. One channel: elevation in metres, stored as `int16`.

| | |
|---|---|
| tiles shipped | **499,934** |
| windows selected | 500,000 |
| `dataset_version` | `bcbd6484d8116199c5c1160e21ea0284` |
| selection list | [`selection_scale_500k.csv`](../../artifacts/phase_f/selection_scale_500k.csv), md5 `90d74c896d1fa04f7079b8491781c21e` |
| source snapshot | 2026-07-27 |
| elevation clip | −500 m to 9,000 m |
| default normalization | per-tile standardize |

**The count is not round, and that is not a rounding error.** 500,000 windows were selected;
499,934 tiles shipped. The missing **66** were dropped during the build, when the builder's
own recomputation of a window's statistics disagreed with the catalog's and pushed it across
a quality threshold — 62 by the flatness screen (§4.3) and 4 by a water-fill check. They
were **recorded, not backfilled**: the corpus does not silently substitute a replacement
window, because doing so would break the guarantee that the committed selection list
describes what actually shipped.

**Every tile has been checked against its own pixels.** The manifest claims six terrain
metrics per tile; a validation criterion (`F1d`) opened all 47 GB of stored imagery and
recomputed them. **0 of 499,934** tiles disagreed beyond tolerance, at a median disagreement
of 8.1 × 10⁻⁵. Worth stating explicitly because it is easy to check the wrong thing here — a
metadata join can be perfectly consistent while the imagery behind it is misaligned, and
earlier validation in this project verified only the join.

**Geographic reach is complete over the eligible pool.** Land is divided into 10° × 10°
cells; 472 contain data, and after quality screening **425** still contain eligible windows.
The corpus reaches **all 425** — zero unreached, median 1,117 tiles per cell, maximum 4,098.

> If you are using the smaller **50 k debug tier** rather than this one, note that it misses
> **25** of those 425 cells. That tier is a fast validation corpus, not a scaled-down version
> of this one, and the two selection lists genuinely differ — neither contains the other.

---

## 2. The thing most likely to mislead you

**This corpus is not a uniform sample of Earth's land surface, and it was never meant to be.**

Every candidate window was scored on six terrain descriptors, then sorted by those scores
into **20 terrain families** ("strata"). The corpus draws a quota from each family. The point
is to give a model a balanced diet of terrain types — so that rare landscapes like high
alpine or deeply dissected badlands are not drowned out by the plains and gentle hills that
dominate the planet by area.

The consequence is that **rare terrain is deliberately over-represented relative to how much
of the world it covers.** Any geographic statistic computed from this corpus will be wrong:

- how much of Earth is mountainous — wrong
- where terrain of a given type occurs, or how common it is — wrong
- anything treating tile counts per region as a measure of that region's extent — wrong

This is the single most misreadable property of the design, which is why it sits ahead of
everything else here rather than in the caveat list. The design and its reasoning are in
[ADR-0005](../adr/0005-sampling-design.md).

You will also need this section to read the next one: §3 is about how deeply the sampler
reached *within* one of those 20 families, which is not a meaningful sentence until you know
the families exist.

---

## 3. Validation — the corpus did not pass its own quality gate

Before this corpus was built, the project wrote down a list of pass/fail criteria and
committed them — **18 criteria, 13 of them gating** — specifically so that no threshold could
be chosen after seeing the numbers. That bar is [`f-gate-bar.md`](f-gate-bar.md); it ran blind
in [`f-validation-report.md`](f-validation-report.md), and the verdict is
[`f-gate-verdict.md`](f-gate-verdict.md).

**The verdict was FAIL.**

**What that does and does not mean.** The roll-up rule gives no partial credit, so one
criterion firing fails the whole gate. What fired was a single **Finding** about *which
terrain got sampled*. **No defect was found in how the corpus was built** — the fidelity
checks in §1 are the same run's output.

Two further criteria fired at the report stage and were set aside at the verdict stage, in
both cases because the problem was in the measuring apparatus rather than the corpus:

- **`F8`** compared a recorded corpus-wide statistic against a re-derivation and found them
  4 ULP apart — roughly, the last four bits of a floating-point number. The cause was
  addition order: the builder accumulates as it writes tiles, the checker read them back in
  512-tile blocks. Read back the builder's way, the number reproduces **bit-identically**.
- **`F1b`** tested whether the 66 dropped tiles were random with respect to terrain. They
  are not — 62 came from a *flatness* screen, and a flatness screen is by definition a
  terrain threshold. That contradiction was already written into another committed criterion
  before the corpus existed, so it is two clauses disagreeing rather than a result being
  explained away after the fact.

Neither tolerance was loosened and neither criterion was re-read to make it pass. The
reasoning for both is in [`f-gate-verdict.md`](f-gate-verdict.md).

### 3.1 The finding: drainage-bearing terrain is under-sampled in two terrain families

In **strata 2 and 17** — both gently-sloping, low-relief families — the corpus contains less
*dissected* terrain than those families actually hold. Dissected terrain means the tiles
carrying drainage structure: valleys, gullies, stream channels cut into an otherwise gentle
surface.

The bar asked that at least **25%** of each family's tiles come from that drainage-bearing
group, which was the group's own share of the terrain before quality screening.

| stratum | tiles shipped | drainage-bearing | realized share `r_h` | bar |
|---|---|---|---|---|
| 2 | 23,153 | 4,856 | **0.2097** | 0.25 |
| 17 | 18,459 | 3,811 | **0.2065** | 0.25 |

**What a model trained on this corpus saw:** in gently-sloping landscapes, fewer valley and
channel networks than those landscapes really contain. If your downstream task is
hydrological — drainage extraction, flow accumulation, channel delineation — this is the
caveat most likely to matter to you, and it matters most in low-relief terrain, which is
where drainage is hardest to detect anyway.

**The obvious explanation is the wrong one.** The natural reading is that the quality screen
(§4.3) stripped the drainage tiles out and left too few to draw from. The measurement says
otherwise. Alongside the shortfall, the bar recorded **exhaustion** — the share of the
*surviving* drainage tiles the sampler actually took:

| stratum | exhaustion `e_h` | surviving tail left untouched |
|---|---|---|
| 2 | 0.185 | **81.5%** |
| 17 | 0.213 | **78.7%** |

Roughly four-fifths of the eligible drainage terrain was **never drawn at all**. There was
enough left untouched to cover the shortfall **23.0×** over in stratum 2 and **17.5×** in
stratum 17. The corpus is not short of drainage tiles because the screen removed them — it is
short because the sampler did not reach that deep into the pool.

That distinction was published deliberately rather than left to inference, because the two
causes point at completely different fixes.

### 3.2 What this means for the project

*This subsection is for the project's own record; a reader using the corpus can stop at §3.1.*

[ADR-0007](../adr/0007-accept-degenerate-flatness-gate-flat-strata.md) accepted the flatness
screen's effect on these strata at the 50 k debug tier, on the explicit condition that it be
re-tested at scale — its own words were "sub-threshold and unstarved at debug scale —
re-verify at scale," not "fine forever." `F3` **is** that re-test, and it fired. **ADR-0007's
accept is therefore not confirmed at scale**, and its revisit trigger follows this corpus
into Phase 1 as *fired*. The ADR has been amended to say so.

[ADR-0006](../adr/0006-accept-edm-flm-hollowing-stratum19.md) is untouched — nothing here
re-tested it.

**What to do about it is not decided here, and deliberately so.** Sizing the fix belongs to
a fresh effort with its own evidence; the attribution above is its starting point, and it
says the flatness screen is not the binding constraint. The knob that is — how deeply the
sampler reaches within a stratum — had no entry in the project's knob table until this phase
added one.

---

## 4. Caveats

### 4.1 The elevations include vegetation and buildings

GLO-30 is a **Digital Surface Model**: it measures the top of whatever is there, not the
ground beneath it. Over forest, the recorded elevation is roughly canopy height above true
terrain; over cities, rooftops.

Published validation of this product finds that **land cover drives elevation error more
than slope does** — so error in this corpus is structured by vegetation, not scattered
randomly.

**Nothing in this corpus corrects, filters, or flags this.** A vegetation-cover column was
considered and deliberately **not built**: promoting it requires evidence from Phase-2 probes
showing canopy-biased reconstructions, and that evidence cannot exist yet. **This note is the
entire mitigation.** There is no canopy column in the manifest to filter on and no windows
were removed on canopy grounds.

### 4.2 Heavily edited and void-filled terrain was removed — and it was not evenly spread

The source product marks pixels that were edited or filled from other data rather than
measured directly. Windows over the caps are excluded, because interpolated surface is
exactly the kind of thing a reconstruction model would learn as genuine terrain.

That screen bites hardest on **high-alpine terrain (stratum 19)**, removing **11,787 of
65,225** would-be members — **18.1%**, about **12.1×** the corpus-wide rate of 1.5%.

This was reviewed and **consciously accepted**
([ADR-0006](../adr/0006-accept-edm-flm-hollowing-stratum19.md)). The dropped terrain is real
high-alpine landscape, and losing it is a genuine cost. It was accepted because **94% of the
drops** sit in the severe external-infill class — surfaces substantially synthesized rather
than measured. The judgement was that a model learning invented alpine terrain as real is
worse than a model seeing less alpine terrain.

**What this means for you:** high-alpine terrain is under-represented, and the specific
alpine terrain missing is the terrain that was hardest to measure — steep, shadowed,
snow-covered. Those are not a random sample of mountains.

### 4.3 The flatness screen, and where it concentrates

Any window whose land pixels vary by less than **3 m** is excluded. These are
near-featureless surfaces — flat sea-level plain, ice-sheet interior — that carry almost no
terrain structure for a reconstruction objective to learn from.

Naturally this removes more from flat terrain families than from steep ones. Measured at the
debug tier, within-stratum drop rates ran **30.1% / 51.9% / 31.7% / 29.7%** in strata
2 / 5 / 7 / 17 against a **9.8%** corpus rate — **3.04× to 5.32×**.

That was examined and **accepted** as structural rather than biased
([ADR-0007](../adr/0007-accept-degenerate-flatness-gate-flat-strata.md)): flat strata contain
more flat windows, so a flatness screen removing more of them is the expected outcome, not
evidence of a defect.

**But in strata 2 and 17 the effect ran the other way, and that is the finding in §3.1** — the
tiles removed there were the *more dissected* ones, not the flattest. See §3.1 for the
outcome at scale; the numbers are not repeated here.

This screen is also the cause of 62 of the 66 build-time drops in §1. Those are recompute
disagreements landing across the 3 m line, not a second screen: the smallest dropped window
measured 3.0010 m and the median 3.071 m, against a retained median of 24.2 m.

### 4.4 A few countries are missing, and we cannot tell you how many tiles

The public edition of GLO-30 **withholds tiles over a small number of specific countries**,
available only to authorised users.

**We cannot measure this gap.** A withheld area is simply an absent file in the source
bucket — and fully-ocean areas are absent for exactly the same reason. The two are
indistinguishable without an external list of withheld countries, which this project does not
hold. So: the gap is real, its size is unknown, and **no number here bounds it.**

Stating that plainly is the whole of this note. Any figure would be invented.

### 4.5 Low-lying atoll and small-island terrain is effectively absent

Of the 472 occupied 10° cells, **47 kept nothing at all**. These are not the withheld
countries — they are named, and they are all ocean: the Marshall Islands, Micronesia, Tuvalu,
French Polynesia, the Seychelles.

The mechanism: a 1° source tile exists there because a speck of land falls inside it, but at
224 pixels a window is ~6.7 km across, so essentially every window is ≥75% sea and the ocean
screen removed all of them.

**This is a missing terrain family, not a gap in a map.** Low-lying coral atoll and small
tropical island terrain is not in this corpus, and no larger budget would recover it — those
windows never entered the eligible pool. If your application involves small island states or
low-lying coastal morphology, this is a hard limitation.

### 4.6 Windows are not independent — do not split this corpus randomly

**This is the caveat you can most easily act on, and most easily be hurt by.**

The corpus is drawn from **1° source tiles** (about 110 km across), each cut into a grid of at
most 256 non-overlapping windows. The draw reached deep into many of them:

| | |
|---|---|
| mean share of a source tile's inventory taken | **23%** |
| median windows per source tile | 14 |
| source tiles with **half or more** taken | **2,788** |
| source tiles **drawn completely dry** | **423** |
| deepest single draw | 233 of a 256 ceiling |

At the 50 k debug tier, *no* source tile was drawn past half. This is entirely an effect of
the larger budget.

Tiles do not share pixels — the grid is non-overlapping. But two windows from the same 1°
tile share geology, climate, the same radar acquisition pass, and the same void-filling and
editing history. **They are neighbours, not copies, and they are not independent samples.**

Two consequences:

1. **Effective dataset size.** 499,934 tiles do not carry 499,934 tiles' worth of independent
   information. Reasoning about sample efficiency or scaling behaviour on the raw count will
   over-estimate what you have.

2. **Train/validation leakage — the concrete hazard.** The manifest ships a `split` column
   that is **declared and null**; filling it with a geographic hold-out is Phase 2's job and
   has not been done. So the obvious thing to do is split randomly — and with 423 source
   tiles fully drawn and 2,788 more past half, a random split puts immediate neighbours on
   both sides of the boundary. **Your validation loss will read better than it should, and
   nothing in the numbers will show you why.**

   **Split by `geocell_id` or by `source_cog`, not at random.** Both columns are in the
   manifest for this purpose.

### 4.7 High-latitude pixels are not square, by decision

Pixels are not reprojected and no latitude cap is applied, so ground pixels are
mildly rectangular at high latitude. Worst case is about **1.55 : 1**, and it occurs near
**50°** — not at the pole, because Copernicus coarsens longitude sampling poleward, which
keeps ground resolution roughly constant (~24–31 m east–west against ~30.9 m north–south).

This was a deliberate choice ([ADR-0001](../adr/0001-corpus-construction-recipe.md)):
reprojection introduces its own artifacts, and a latitude cap would discard glaciated and
tundra terrain to solve a problem that per-tile standardization largely absorbs.

Every tile records its **`lat_band`** in the manifest, so resolution-aware encoding remains a
configuration change rather than a re-tile.

### 4.8 Under `global` normalization, tiles are nearly flat — this is the data, not a bug

The corpus ships two normalization modes. **Per-tile** (the default) subtracts each tile's own
mean and divides by its own standard deviation, keeping terrain shape and discarding absolute
altitude. **Global** subtracts one corpus-wide mean and divides by one corpus-wide standard
deviation, keeping altitude.

Measured on the shipped corpus:

| | |
|---|---|
| median tile's own elevation spread | **24.2 m** |
| the same, normalized under `global` | **0.024** |
| tiles under 0.10 normalized spread | **82.0%** |
| between-tile altitude spread ÷ typical within-tile spread | **41.6×** |

The variation *between* tiles' absolute altitudes is roughly **42 times** the variation
*within* a typical tile. Under `global` mode with a pixel-wise squared-error loss, most of
the loss is therefore the model guessing a tile's average elevation, not reconstructing its
terrain.

Both modes are legitimate — per-tile discards altitude by design, global keeps it — and
comparing them is a scheduled Phase-3 ablation. This is recorded so that **a normalized tile
with a standard deviation around 0.02 is not mistaken for a loader bug**, and so the
ablation's likely outcome has a mechanism stated in advance rather than one invented
afterwards.

These figures were measured on the debug tier first ([#70](https://github.com/JStoschek/DEMAE/issues/70),
[ADR-0003 Amendment 1](../adr/0003-dataloader-design.md#amendments)) and are recomputed here
on the shipped corpus: 0.024 / 82% / 42× there, 0.024 / 82.0% / 41.6× here. The finding
reproduces at scale.

**One constraint that follows:** the recorded global mean and standard deviation
(955.67 m / 1012.16 m) belong to **this corpus only**. Each tier accumulates its own. They
are not portable — normalizing a different corpus against these numbers would be normalizing
against a statistic measured on different data.

---

## 5. What isn't here

Three things are declared but deliberately unbuilt. Each has a written trigger that would
promote it, and none of those triggers can fire until pretraining runs.

| | state | what would build it |
|---|---|---|
| **`split` column** | declared, all null | Phase 2 fills a geographic hold-out. Until then see §4.6 — do not split randomly. |
| **vegetation / canopy column** | not built | Phase-2 probes showing canopy-biased reconstructions (§4.1) |
| **anisotropy metric** | not built | downstream tasks involving oriented terrain fabrics — dunes, drumlins (§4.7) |

---

## 6. Where the evidence is

| claim | source |
|---|---|
| every number in this document | [`f-honesty-notes.json`](f-honesty-notes.json) · [driver](../../notebooks/f_honesty_notes.py) |
| the pre-registered quality bar | [`f-gate-bar.md`](f-gate-bar.md) / [`.json`](f-gate-bar.json) |
| the validation run | [`f-validation-report.md`](f-validation-report.md) / [`.json`](f-validation-report.json) |
| the verdict and its reasoning | [`f-gate-verdict.md`](f-gate-verdict.md) / [`.json`](f-gate-verdict.json) |
| how the corpus was drawn | [`f-scale-draw.md`](f-scale-draw.md) · [ADR-0005](../adr/0005-sampling-design.md) |
| terrain families, 12 tiles each | [`f-gallery.md`](f-gallery.md) |
| the edited-pixel decision | [ADR-0006](../adr/0006-accept-edm-flm-hollowing-stratum19.md) |
| the flatness-screen decision | [ADR-0007](../adr/0007-accept-degenerate-flatness-gate-flat-strata.md) |
| storage format and manifest schema | [ADR-0002](../adr/0002-storage-format-and-manifest.md) |
| normalization modes | [ADR-0003](../adr/0003-dataloader-design.md) |
