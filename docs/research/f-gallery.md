# F4 stratum gallery — 20 strata × 12 tiles from the 500k corpus

**Ticket:** [Run the Gate F validation report + gallery (#73)](https://github.com/JStoschek/DEMAE/issues/73) ·
**Map:** [Phase F (#67)](https://github.com/JStoschek/DEMAE/issues/67) ·
**Bar:** [`f-gate-bar.json`](f-gate-bar.json) → `F4_gallery_published` ·
**Labels recorded at:** [Gate F (#74)](https://github.com/JStoschek/DEMAE/issues/74)

> **Published, non-gating.** [#69](https://github.com/JStoschek/DEMAE/issues/69) retired
> E2.4's pass/fail: its registered question was whether `k=20` is terrain-meaningful,
> its knob was `k`, and the same persisted centroids produce the same partition — so
> [#53](https://github.com/JStoschek/DEMAE/issues/53)/[#54](https://github.com/JStoschek/DEMAE/issues/54)'s
> **20/20 coherent** already answered it, and `k` is out of scope for map #67. There is no
> coherence gate here and no fail condition to trip.

What this gallery is *for* at Gate F: [#75](https://github.com/JStoschek/DEMAE/issues/75)
needs a gallery of the corpus that actually shipped, the terrain-family labels are recorded
for continuity, and this is the only look at scale imagery that would catch a corruption
preserving land-std — which `F1d_pixel_integrity`, the whole-corpus pixel check that took
over F4's former gating role, cannot see.

Corpus `bcbd6484d8116199c5c1160e21ea0284` · 499,934 tiles · seed `20260723` · rendered 2026-07-27 at
`a92bb79-dirty`. The exact tile ids behind every sheet are in
[`gallery.json`](f-gallery/gallery.json), so this gallery is reconstructible after the
shards — which are gitignored, and at 47 GB will not be kept — are gone.

## How the tiles are drawn

Unchanged from [#53](https://github.com/JStoschek/DEMAE/issues/53), so the two galleries are
comparable: grey hillshade, north-west sun at 315°/45°, σ=1.0 px Gaussian
before shading, per-tile vertical exaggeration towards a 60 m land-std, true
anisotropic pixel spacing. **No colour carries terrain** — magnitude is printed in the
caption, and the coloured bar left of each tile carries absolute elevation on a shared ramp.
Exaggeration is per tile, so **apparent contrast means nothing across tiles**; the captioned
`relief` and `elev` are the magnitudes. The full argument for each choice is in
[`e2-gallery-protocol.md`](e2-gallery-protocol.md), which still describes the *rendering*
faithfully even though its pass/fail clause no longer applies.

## What is new at 500k

A ten-times-deeper draw reaches further from each centroid, so more **peripheral cluster
members** appear than at 50k. The bar records that twelve tiles per stratum has almost no
power to detect that, and that blending in a continuum is expected rather than a failure
([#41](https://github.com/JStoschek/DEMAE/issues/41)) — which is part of why these marks do
not gate. Read the sheets for *what shipped*, not for a verdict.

## The sheets

| stratum | n_h shipped | 12 uniform tiles | diagnostic (interior→edge) |
|---|---:|---|---|
| 0 | 32,196 | [`stratum-00.png`](f-gallery/stratum-00.png) | [`stratum-00-spanning.webp`](f-gallery/stratum-00-spanning.webp) |
| 1 | 28,071 | [`stratum-01.png`](f-gallery/stratum-01.png) | [`stratum-01-spanning.webp`](f-gallery/stratum-01-spanning.webp) |
| 2 | 23,153 | [`stratum-02.png`](f-gallery/stratum-02.png) | [`stratum-02-spanning.webp`](f-gallery/stratum-02-spanning.webp) |
| 3 | 21,847 | [`stratum-03.png`](f-gallery/stratum-03.png) | [`stratum-03-spanning.webp`](f-gallery/stratum-03-spanning.webp) |
| 4 | 36,614 | [`stratum-04.png`](f-gallery/stratum-04.png) | [`stratum-04-spanning.webp`](f-gallery/stratum-04-spanning.webp) |
| 5 | 19,905 | [`stratum-05.png`](f-gallery/stratum-05.png) | [`stratum-05-spanning.webp`](f-gallery/stratum-05-spanning.webp) |
| 6 | 7,343 | [`stratum-06.png`](f-gallery/stratum-06.png) | [`stratum-06-spanning.webp`](f-gallery/stratum-06-spanning.webp) |
| 7 | 26,043 | [`stratum-07.png`](f-gallery/stratum-07.png) | [`stratum-07-spanning.webp`](f-gallery/stratum-07-spanning.webp) |
| 8 | 36,997 | [`stratum-08.png`](f-gallery/stratum-08.png) | [`stratum-08-spanning.webp`](f-gallery/stratum-08-spanning.webp) |
| 9 | 27,238 | [`stratum-09.png`](f-gallery/stratum-09.png) | [`stratum-09-spanning.webp`](f-gallery/stratum-09-spanning.webp) |
| 10 | 28,173 | [`stratum-10.png`](f-gallery/stratum-10.png) | [`stratum-10-spanning.webp`](f-gallery/stratum-10-spanning.webp) |
| 11 | 10,148 | [`stratum-11.png`](f-gallery/stratum-11.png) | [`stratum-11-spanning.webp`](f-gallery/stratum-11-spanning.webp) |
| 12 | 32,032 | [`stratum-12.png`](f-gallery/stratum-12.png) | [`stratum-12-spanning.webp`](f-gallery/stratum-12-spanning.webp) |
| 13 | 33,236 | [`stratum-13.png`](f-gallery/stratum-13.png) | [`stratum-13-spanning.webp`](f-gallery/stratum-13-spanning.webp) |
| 14 | 24,672 | [`stratum-14.png`](f-gallery/stratum-14.png) | [`stratum-14-spanning.webp`](f-gallery/stratum-14-spanning.webp) |
| 15 | 20,975 | [`stratum-15.png`](f-gallery/stratum-15.png) | [`stratum-15-spanning.webp`](f-gallery/stratum-15-spanning.webp) |
| 16 | 30,859 | [`stratum-16.png`](f-gallery/stratum-16.png) | [`stratum-16-spanning.webp`](f-gallery/stratum-16-spanning.webp) |
| 17 | 18,459 | [`stratum-17.png`](f-gallery/stratum-17.png) | [`stratum-17-spanning.webp`](f-gallery/stratum-17-spanning.webp) |
| 18 | 30,919 | [`stratum-18.png`](f-gallery/stratum-18.png) | [`stratum-18-spanning.webp`](f-gallery/stratum-18-spanning.webp) |
| 19 | 11,054 | [`stratum-19.png`](f-gallery/stratum-19.png) | [`stratum-19-spanning.webp`](f-gallery/stratum-19-spanning.webp) |

**[`coastal-floor.webp`](f-gallery/coastal-floor.webp)** — the 6,696
`floor:coastline` tiles as a reporting axis (plan §133).
[#40](https://github.com/JStoschek/DEMAE/issues/40) measured coastline as *not* a stratum
(max purity 4.9%); there is no coastal criterion in the Gate F bar and this sheet does not
create one.

## Terrain-family labels — for #74 to complete

One short terrain-family label per stratum, plus any note worth carrying into #75. **No
mark column**: `coherent` / `mixed` / `incoherent` was E2.4's registered vocabulary and it
belonged to a gate that no longer exists, so recording one here would manufacture a verdict
the bar declines to ask for. Judge each stratum's tiles on their own; do not compare strata.

| stratum | terrain-family label | note |
|---|---|---|
| 0 |  |  |
| 1 |  |  |
| 2 |  |  |
| 3 |  |  |
| 4 |  |  |
| 5 |  |  |
| 6 |  |  |
| 7 |  |  |
| 8 |  |  |
| 9 |  |  |
| 10 |  |  |
| 11 |  |  |
| 12 |  |  |
| 13 |  |  |
| 14 |  |  |
| 15 |  |  |
| 16 |  |  |
| 17 |  |  |
| 18 |  |  |
| 19 |  |  |

*Left blank deliberately — the labels are #74's, and #73 does not pre-empt them.*
