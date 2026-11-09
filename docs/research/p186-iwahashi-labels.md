# P186: The Iwahashi & Pike map is free to fetch; 45% of the test tiles survive the 60% rule

**Ticket:** [#186](https://github.com/JStoschek/DEMAE/issues/186), under map
[#183](https://github.com/JStoschek/DEMAE/issues/183).
**Script:** [`scripts/p186_iwahashi_survival.py`](../../scripts/p186_iwahashi_survival.py). It
runs in about 20 s and reads the raster through one small window per tile.
**Amends:** [`task2-landform-plan.md`](task2-landform-plan.md) §3.2.

## The answer in five lines

1. **Availability:** yes. One 466 MB zip from the distributor's page, over plain HTTPS. There is
   no login, registration, licence click-through or cookie step.
2. **Licence:** the distributor says it "believe[s]" the data is CC BY 4.0, and asks users to cite
   Iwahashi et al. (2018). That wording is hedged, and the DEM it was computed from carries
   different terms (see §2).
3. **Grouping:** the 15 groups are in the zip as a CSV legend. The GeoTIFF holds **only the 15
   groups**, not the 40 classes. The 40 classes do not map onto the 15 groups as a function:
   mountains and hills are split into a/b subgroups by texture *within* a class.
4. **Survival:** **2,240 of 5,000** committed test tiles clear 60%, and **3,053** clear 50%. On
   the 4,523 clean tiles (#150's filter) the counts are 1,985 and 2,712. **639 tiles can never
   get a label**: 555 lie in Antarctica, south of the map's edge, and 84 lie on Arctic ice the
   map leaves unclassified.
5. **Lineage:** the map was computed from MERIT DEM, which is built from SRTM, AW3D and
   Viewfinder Panoramas. GLO-30 (TanDEM-X) is not in that chain, so §3.2's independence claim
   holds.

## 1. Availability

The page at <https://gisstar.gsi.go.jp/terrain2018/> (last modified 2024-04-30, fetched
2026-10-15) links three kinds of download, all as plain `<a href>` files on the same host:

| What | Link | Form |
| --- | --- | --- |
| Global GeoTIFF | `7.5_arc_second_compressedGeoTIFF.zip` | one zip, **466,197,923 bytes**, md5 `5ae98dc7782c13384a1df4d8d609418f` |
| Polygon shapefiles | `01_70N180W.zip` … `96_70S150E.zip` | 75 tiles of 30° × 20°, in Lambert azimuthal equal-area; they carry both `CLASS` (40) and `GROUP` (15) |
| Legend image | `Legend.jpg` | |

A `HEAD` request and a plain `curl` both return `200` with `content-type: application/zip`. No
cookie or credential is involved, so **no human step is needed**.

What the zip holds:

- `GlobalTerrainClassification_Iwahashi_etal_2018.tif`: 486,772,725 bytes, md5
  `d604f9b57dbb12d85eeb48c011ebcf3f`.
  - **Grid:** EPSG:4326, 172,801 × 68,683 pixels, pixel size 0.0020833° (7.5″, about 232 m at
    the equator, not 280 m).
  - **Extent:** 83.63° N to **59.46° S**.
  - **Data:** `uint8`, values 1–15 are the groups and 255 is nodata. Value **0** also appears and
    is not in the legend (see §4).
  - **Storage:** deflate-compressed strips one row tall, with overviews. A windowed read touches
    only the rows it needs.
  - **Provenance:** the page says Tomislav Hengl (ISRIC) converted it from the 15-group
    shapefiles. It is a rasterization of the polygons, not the authors' own raster product.
- `Landform_legend.csv`: maps raster value to group, overview class, landform name and colour.
- `Color_ArcGIS.lyr`: an ArcGIS style file.

The zip is unpacked locally under `data/iwahashi2018/` (1.1 GB with three sample shapefile
zips). That directory is gitignored by `.gitignore`'s `data/*` line, and nothing from it is
committed.

## 2. Licence, as the distributor states it

These are the distribution terms as written. This section reports them and draws no legal
conclusion. Quoting the GSI page's "License" section, which was added on 2024-05-01 per its
update log:

> Based on the concept of JSPS KAKENHI which encourages open access, we believe the data
> published on this site is licensed under a Creative Commons CC-BY 4.0.

The page also asks users to cite Iwahashi, Kamiya, Matsuoka & Yamazaki (2018), *Progress in
Earth and Planetary Science* 5:1, doi:10.1186/s40645-017-0157-2. The shapefile README repeats
the citation request and states no licence.

Three facts a reader of the dataset card should have:

- **The grant is hedged.** The page says "we believe … is licensed under", not "is licensed
  under". No LICENSE file ships in the zip.
- **The upstream DEM carries different terms.** MERIT DEM's own page
  (<https://global-hydrodynamics.github.io/MERIT_DEM/>) dual-licenses it, and users choose one:
  CC BY-NC 4.0, or ODbL 1.0, which allows commercial use but requires derived databases to be
  released under the same licence. The terrain classification is a product derived from MERIT.
  Whether those upstream terms reach a classification computed from it is a question for a
  human, and this note does not answer it.
- **The GeoTIFF is a third party's conversion.** Hengl made it. The page publishes it under the
  same License section.

The plan's "CC-BY 4.0" therefore matches what the distributor states. The dataset card should
quote that statement, carry the citation, and note the MERIT upstream terms beside it.

## 3. The 40 → 15 grouping

**The 15 groups are machine-readable.** `Landform_legend.csv` ships in the GeoTIFF zip, and an
identical `Terrain_GROUP_legend.csv` (without the `Value` column) ships in every shapefile zip.
The raster stores values 1–15, which map to groups as follows:

| Value | Group | Overview class | Landform pattern |
| ---: | --- | --- | --- |
| 1 | 1a | Bedrock mountain | Steep mountain (rough) |
| 2 | 1b | Bedrock mountain | Steep mountain (smooth) |
| 3 | 2a | Bedrock mountain | Moderate mountain (rough) |
| 4 | 2b | Bedrock mountain | Moderate mountain (smooth) |
| 5 | 3a | Hills | Hills (rough in small and large scale) |
| 6 | 3b | Hills | Hills (smooth in small scale, rough in large scale) |
| 7 | 4 | Large highland slope | Upper large slope |
| 8 | 5 | Large highland slope | Middle large slope |
| 9 | 6 | Plateau, terrace, large lowland slope | Dissected terrace, moderate plateau |
| 10 | 7 | Plateau, terrace, large lowland slope | Slope in and around terrace or plateau |
| 11 | 8 | Plateau, terrace, large lowland slope | Terrace, smooth plateau |
| 12 | 9 | Plateau, terrace, large lowland slope | Alluvial fan, pediment, bajada, pediplain |
| 13 | 10 | Plain | Alluvial plain, pediplain |
| 14 | 11 | Plain | Alluvial or coastal plain, pediplain |
| 15 | 12 | Plain | Alluvial or coastal plain (gentlest), lake plain, playa |

**The 40-class to group mapping is not a lookup table, and §3.2's wording overstates it.** I
cross-tabulated `CLASS` against `GROUP` over 600,429 polygons in three shapefile tiles
(`19_50N000E`, `28_30N090W`, `34_30N090E`):

- **Groups 4–12** are a pure function of the class:
  - group 4 ← class 18
  - group 5 ← classes 16, 34
  - group 6 ← classes 2, 21, 28, 37, 38
  - group 7 ← class 9
  - group 8 ← classes 3, 26, 30
  - group 9 ← classes 1, 11, 12, 13, 36
  - group 10 ← class 35
  - group 11 ← classes 27, 32, 33
  - group 12 ← classes 8, 23
- **Groups 1–3** are not. Each of their classes appears in **both** subgroups:
  - 1a/1b ← classes 6, 15, 22, 40
  - 2a/2b ← classes 5, 17, 19, 29
  - 3a/3b ← classes 4, 7, 10, 14, 20, 24, 25, 31, 39
- **Class 0 → group 0** is a residual polygon class with no legend entry.

This matches the paper's abstract as indexed (I could not reach the article itself; see below).
The authors reclassified the 40 clusters into 12 groups using Japanese reference maps, then split
mountains and hills into rough and smooth by a combined texture measure. So "40 classes grouped
to 15" is really "40 → 12 by table, then 3 of the 12 split by texture", which gives 15.

Consequences:

- The 15-group GeoTIFF is the right input and needs no grouping step.
- A label set on the 40 classes would need the shapefiles rasterized. There are 75 zips of
  0.2–81 MB each.

The paper's own grouping table (Table 3) was not read. The paper's landing page on SpringerOpen,
and its PDF, redirect to a Springer cookie-authorization endpoint that neither `curl` nor
WebFetch gets past. A published correction
([doi:10.1186/s40645-018-0173-x](https://doi.org/10.1186/s40645-018-0173-x)) amends Table 3's
group 10–12 names ("natural level" → "natural levee"). That change affects the Japanese-example
column only, and the shipped legend CSV does not carry that column.

## 4. Survival under §3.2's majority rule

### Method

**Tiles.** The input is `artifacts/task1/test_tiles.csv`, 5,000 tiles (md5
`e842b31d02df74e81260dffd03499378`). They are joined to `data/corpus_500k/manifest.parquet` for
`source_cog`, `win_row0`, `win_col0`, `lat`, `slope_mean`, `land_frac` and `water_affected`.

**Footprint.** Each tile is a 224 × 224 full-resolution window of a 1° Copernicus COG, so:

- its north and south edges are 224/3600° apart, about 30 label rows;
- its east–west span is 224/cols° wide, with cols taken from
  `dataset.geometry.COLS_PER_DEGREE` (3600 below 50°, fewer poleward). That gives about 30 label
  columns at the equator and 60 at 60°N.

The script asserts that every footprint's centre equals the manifest's `lon`/`lat` to within 1e-6°.

**Fraction.** Each label pixel is weighted by its overlap area with the footprint. Two fractions
are computed per tile:

- `frac_footprint`: area of the majority group ÷ **whole footprint**. Nodata and value 0 count
  against the majority. This is §3.2's rule as written, and the headline.
- `frac_valid`: area of the majority group ÷ **labelled area only**.

**Ruggedness strata.** The four strata are §5's bands R1 (flattest) to R4: quartiles of
`slope_mean` over the eligible test pool, from `benchmark.voids.Bands.from_manifest`. These are
the same bands Task 1's R12 reads.

### Headline counts (`frac_footprint`)

| Population | n | ≥ 60% | ≥ 50% | can never be labelled |
| --- | ---: | ---: | ---: | ---: |
| All committed test tiles | 5,000 | **2,240** (44.8%) | **3,053** (61.1%) | 639 |
| Clean tiles (`voids.eligible`, #150) | 4,523 | **1,985** (43.9%) | **2,712** (60.0%) | 622 |

By ruggedness band (rows sum across bands to the table above):

| Band | all: n | ≥ 60% | ≥ 50% | clean: n | ≥ 60% | ≥ 50% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R1 (flattest) | 1,264 | 624 | 757 | 1,127 | 544 | 652 |
| R2 | 1,275 | 539 | 731 | 1,127 | 450 | 617 |
| R3 | 1,295 | 593 | 832 | 1,158 | 523 | 735 |
| R4 (steepest) | 1,166 | 484 | 733 | 1,111 | 468 | 708 |

By majority group, over all 5,000 tiles. "n" counts the tiles whose plurality is that group,
whatever its fraction.

| Group | n | ≥ 60% | ≥ 50% |
| --- | ---: | ---: | ---: |
| 1a Steep mountain (rough) | 206 | 105 | 145 |
| 1b Steep mountain (smooth) | 80 | 22 | 37 |
| 2a Moderate mountain (rough) | 206 | 78 | 126 |
| 2b Moderate mountain (smooth) | 68 | 17 | 27 |
| 3a Hills (rough) | 728 | 314 | 485 |
| 3b Hills (smooth/rough) | 486 | 303 | 376 |
| 4 Upper large slope | 54 | **10** | 24 |
| 5 Middle large slope | 130 | 29 | 54 |
| 6 Dissected terrace, moderate plateau | 670 | 398 | 524 |
| 7 Slope around terrace/plateau | 139 | 23 | 48 |
| 8 Terrace, smooth plateau | 530 | 292 | 393 |
| 9 Alluvial fan, pediment | 517 | 288 | 368 |
| 10 Alluvial plain | 116 | 48 | 79 |
| 11 Alluvial/coastal plain | 377 | 273 | 319 |
| 12 Gentlest plain, playa | 54 | 40 | 48 |
| none (no labelled pixel) | 639 | 0 | 0 |

The same table over the clean 4,523, and every row under `frac_valid`, are in the script's
output. The per-tile table is written to `data/iwahashi2018/p186_test_tile_fractions.csv`, which
is gitignored.

Measuring against labelled area only (`frac_valid`) moves little: 2,310 tiles clear 60% and 3,118
clear 50%. Few surviving tiles are partly unlabelled; only 32 of the 2,240 have more than 5% of
their footprint as nodata or 0.

### What the numbers say

- **The 60% rule keeps less than half the test set: 2,240 tiles.** That is 51% of the 4,361
  tiles that have any label at all. §9 anticipated relaxing to 50%; that rule keeps 3,053.
- **639 tiles (12.8%) are unlabellable whatever the threshold.** The losses fall unevenly across
  bands (R1 368, R2 179, R3 73, R4 19), which skews the surviving set away from flat ground.
  - **555 tiles in Antarctica:** they lie south of 59.46° S, where the raster ends.
  - **84 tiles on Arctic ice** (Svalbard, Greenland, around 61–80° N): they read value 0. The
    shapefiles carry this as class 0 / group 0 with no legend entry. The page says GlobCover 2009
    masked water bodies. That ice falls under the same mask is an inference from where the 0s
    are, not something the distributor states.
- **After the 60% rule, every band keeps 41–49% of its tiles.** A per-band report is feasible.
- **Per-group counts are thin for five groups.** At 60%: group 4 has 10 tiles, 2b has 17, 1b 22,
  7 23, and 5 29. A 15-way per-class accuracy will be noisy on those rows. Two remedies, both left
  to the decision ticket and neither chosen here:
  - report at the 5-way overview level (bedrock mountain / hills / large highland slope /
    plateau-terrace / plain) beside the 15 groups;
  - pool the a/b subgroups.

The rule was applied to the **test list only**, as asked. Train and val survival will be similar
in proportion but is not measured here.

## 5. Lineage

The GSI page says, under "Additional information":

- the classification was "created from 280 m DEMs using MERIT DEM";
- GlobCover 2009 "was used to mask water bodies in most areas".

The paper's abstract, as search engines index it, says the 280 m DEMs were "interpolated from"
MERIT DEM. MERIT DEM's page says it removes error components "from multiple existing DEM
products, including NASA SRTM, JAXA AW3D, and Viewfinder Panoramas", at 3″ (~90 m). It names
neither TanDEM-X nor the Copernicus DEM. GLO-30 is a TanDEM-X product. **No GLO-30 pixel is
upstream of the Iwahashi map**, so §3.2's claim holds as written.

Two caveats for the writeup. They do not weaken the claim, but they stop it being overstated:

- **Independent is not uncorrelated.** SRTM (C-band, 2000) and TanDEM-X (X-band, 2011–15) sample
  the same terrain, so the labels agree with GLO-30's geomorphology wherever the ground is real.
  C3's claim is that the label is not a *function* of the encoder's input, and that is what
  holds.
- **SRTM is already in this repo.** Task 1 transplants real SRTM void *shapes* (#127). Only the
  void masks' geometry is taken from SRTM, never its elevations, so it does not reach Task 2's
  labels.

## Sources

- GSI distributor page, <https://gisstar.gsi.go.jp/terrain2018/>, fetched 2026-10-15: the
  download links, the License, Citation and Additional information sections, and the update log.
- The distributed zip's `Landform_legend.csv`. The shapefile zips' `README.txt` and DBF
  attributes (`19_50N000E`, `28_30N090W`, `34_30N090E`).
- MERIT DEM page, <https://global-hydrodynamics.github.io/MERIT_DEM/>: source DEMs and licence.
- Iwahashi et al. (2018), doi:10.1186/s40645-017-0157-2, and its correction,
  doi:10.1186/s40645-018-0173-x. These were seen only as search-index abstracts, because the
  full text sits behind Springer's cookie redirect.
- In-repo: `artifacts/task1/test_tiles.csv`, `data/corpus_500k/manifest.parquet`,
  `src/dataset/geometry.py` (footprint geometry), `src/benchmark/voids.py` (bands, `eligible`).
