# E2.4 stratum gallery — 20 strata x 12 tiles

**Ticket:** [Render the stratum gallery (#53)](https://github.com/JStoschek/DEMAE/issues/53) ·
**Map:** [Phase E (#46)](https://github.com/JStoschek/DEMAE/issues/46) ·
**Reviewed at:** [Gate E (#54)](https://github.com/JStoschek/DEMAE/issues/54) ·
**Protocol:** [`e2-gallery-protocol.md`](e2-gallery-protocol.md)

The human backstop that k=20 strata are terrain-meaningful. Read the protocol
first — it fixes what is being judged (**internal coherence**, one stratum at a time),
what is explicitly **not** (distinguishability from a neighbour: #41 found the descriptor
space is a continuum, so adjacent strata blending is expected), and the pre-registered
fail condition (**any stratum `incoherent`, or ≥ 3 `mixed`**).

Corpus `6fde6a580fdebb24b20987181748ad7f` · 49,998 tiles · seed `20260723` · rendered 2026-07-23 at
`fd14d6d-dirty`. The exact tile ids behind every sheet are in
[`gallery.json`](gallery.json), so this gallery is reconstructible after the shards —
which are gitignored — are gone.

## How the tiles are drawn

Grey hillshade, north-west sun at 315°/45°, σ=1.0 px Gaussian before
shading, per-tile vertical exaggeration towards a 60 m land-std, true
anisotropic pixel spacing. **No colour carries terrain**: a shared elevation ramp flattens
every low-relief stratum into one block, and a per-tile ramp gives a 10 m-relief tile the
full alpine treatment — so magnitude is *printed*, and the coloured bar left of each tile
carries absolute elevation without touching the terrain pixels. Exaggeration is per tile,
so **apparent contrast means nothing across tiles**; the captioned `relief` and `elev` are
the magnitudes. σ=1 px matters: hillshading is a derivative, and at the exaggeration a flat
tile needs, the stored int16's 1 m quantum becomes stipple that reads as texture.

## The sheets

| stratum | n_h | gating sheet (12 uniform) | diagnostic (interior→edge) |
|---|---:|---|---|
| 0 | 3,220 | [`stratum-00.png`](e2-gallery/stratum-00.png) | [`stratum-00-spanning.webp`](e2-gallery/stratum-00-spanning.webp) |
| 1 | 2,807 | [`stratum-01.png`](e2-gallery/stratum-01.png) | [`stratum-01-spanning.webp`](e2-gallery/stratum-01-spanning.webp) |
| 2 | 2,316 | [`stratum-02.png`](e2-gallery/stratum-02.png) | [`stratum-02-spanning.webp`](e2-gallery/stratum-02-spanning.webp) |
| 3 | 2,185 | [`stratum-03.png`](e2-gallery/stratum-03.png) | [`stratum-03-spanning.webp`](e2-gallery/stratum-03-spanning.webp) |
| 4 | 3,662 | [`stratum-04.png`](e2-gallery/stratum-04.png) | [`stratum-04-spanning.webp`](e2-gallery/stratum-04-spanning.webp) |
| 5 | 1,991 | [`stratum-05.png`](e2-gallery/stratum-05.png) | [`stratum-05-spanning.webp`](e2-gallery/stratum-05-spanning.webp) |
| 6 | 734 | [`stratum-06.png`](e2-gallery/stratum-06.png) | [`stratum-06-spanning.webp`](e2-gallery/stratum-06-spanning.webp) |
| 7 | 2,604 | [`stratum-07.png`](e2-gallery/stratum-07.png) | [`stratum-07-spanning.webp`](e2-gallery/stratum-07-spanning.webp) |
| 8 | 3,699 | [`stratum-08.png`](e2-gallery/stratum-08.png) | [`stratum-08-spanning.webp`](e2-gallery/stratum-08-spanning.webp) |
| 9 | 2,724 | [`stratum-09.png`](e2-gallery/stratum-09.png) | [`stratum-09-spanning.webp`](e2-gallery/stratum-09-spanning.webp) |
| 10 | 2,817 | [`stratum-10.png`](e2-gallery/stratum-10.png) | [`stratum-10-spanning.webp`](e2-gallery/stratum-10-spanning.webp) |
| 11 | 1,015 | [`stratum-11.png`](e2-gallery/stratum-11.png) | [`stratum-11-spanning.webp`](e2-gallery/stratum-11-spanning.webp) |
| 12 | 3,203 | [`stratum-12.png`](e2-gallery/stratum-12.png) | [`stratum-12-spanning.webp`](e2-gallery/stratum-12-spanning.webp) |
| 13 | 3,324 | [`stratum-13.png`](e2-gallery/stratum-13.png) | [`stratum-13-spanning.webp`](e2-gallery/stratum-13-spanning.webp) |
| 14 | 2,467 | [`stratum-14.png`](e2-gallery/stratum-14.png) | [`stratum-14-spanning.webp`](e2-gallery/stratum-14-spanning.webp) |
| 15 | 2,098 | [`stratum-15.png`](e2-gallery/stratum-15.png) | [`stratum-15-spanning.webp`](e2-gallery/stratum-15-spanning.webp) |
| 16 | 3,087 | [`stratum-16.png`](e2-gallery/stratum-16.png) | [`stratum-16-spanning.webp`](e2-gallery/stratum-16-spanning.webp) |
| 17 | 1,847 | [`stratum-17.png`](e2-gallery/stratum-17.png) | [`stratum-17-spanning.webp`](e2-gallery/stratum-17-spanning.webp) |
| 18 | 3,092 | [`stratum-18.png`](e2-gallery/stratum-18.png) | [`stratum-18-spanning.webp`](e2-gallery/stratum-18-spanning.webp) |
| 19 | 1,106 | [`stratum-19.png`](e2-gallery/stratum-19.png) | [`stratum-19-spanning.webp`](e2-gallery/stratum-19-spanning.webp) |

**[`coastal-floor.webp`](e2-gallery/coastal-floor.webp)** — the 669 `floor:coastline`
tiles as a reporting axis (plan §133). #40 measured coastline as *not* a stratum (max
purity 4.9%); there is no coastal-coherence criterion in the bar and this sheet does not
create one.

The `-spanning` sheets are **published, not gating**. They walk each stratum from its
centroid outward, which is where a continuum's cluster boundaries are really tested — but
over-representing the boundary would make the gate stricter than the one pre-registered,
so they inform the reviewer without moving the bar.

## Marks — for #54 to complete

Per the protocol: one mark per stratum (`coherent` / `mixed` / `incoherent`) plus a short
terrain-family label. Record all 20, **including on a pass** — they are the evidence
behind the verdict and the seed for the E3 notes (#55) and ADR-0005 (#56). Judge each
stratum's tiles on their own; do not compare strata.

| stratum | mark | terrain-family label | note |
|---|---|---|---|
| 0 |  |  |  |
| 1 |  |  |  |
| 2 |  |  |  |
| 3 |  |  |  |
| 4 |  |  |  |
| 5 |  |  |  |
| 6 |  |  |  |
| 7 |  |  |  |
| 8 |  |  |  |
| 9 |  |  |  |
| 10 |  |  |  |
| 11 |  |  |  |
| 12 |  |  |  |
| 13 |  |  |  |
| 14 |  |  |  |
| 15 |  |  |  |
| 16 |  |  |  |
| 17 |  |  |  |
| 18 |  |  |  |
| 19 |  |  |  |

*Left blank deliberately — the marks are #54's, and #53 does not pre-empt them.*
