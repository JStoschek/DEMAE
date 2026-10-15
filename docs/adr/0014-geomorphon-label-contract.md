# ADR 0014 — The geomorphon label contract: WBT v2.4.0 at L = 50 and 1.0°, ground spacing in the header, an anisotropic buffer, and an ignore label

- **Status:** Accepted
- **Date:** 2026-10-15
- **Context ticket:** "Settle the geomorphon label contract" ([issue #188](https://github.com/JStoschek/DEMAE/issues/188)), part of the Task-2 landform map ([#183](https://github.com/JStoschek/DEMAE/issues/183))
- **Evidence:** the timing pass in [#187](https://github.com/JStoschek/DEMAE/issues/187) ([`p187-geomorphon-pass.md`](https://github.com/JStoschek/DEMAE/blob/research/p187-geomorphon-pass/docs/research/p187-geomorphon-pass.md)); the parameter sweep in [`p188-geomorphon-sweep.md`](https://github.com/JStoschek/DEMAE/blob/research/p188-geomorphon-sweep/docs/research/p188-geomorphon-sweep.md); WhiteboxTools v2.4.0 source (`whitebox-tools-app/src/tools/terrain_analysis/geomorphons.rs`)
- **Amends:** [`docs/research/task2-landform-plan.md`](../research/task2-landform-plan.md) §3.1, §3.4, §4.4 step 1

## Context

Task 2's dense labels are geomorphons computed from GLO-30. Every head trains against them and
every C-claim is read off them, so a label that changes after training starts invalidates the
grid. The plan's §3.1 gave the recipe in one paragraph, and three parts of it did not hold:

- **"Literature-default settings."** L = 50 cells is WhiteboxTools' command-line default, and
  1.0° is GRASS `r.geomorphon`'s default (GRASS's own L default is 3). No source sets both. WBT's
  `--threshold` is also inconsistent with itself: its help text says 0.0° and its code initialises
  it to 1.0°.
- **"324×324 buffered window."** WBT measures the search radius as L·dy metres, so along a row it
  reaches ⌈L·dy/dx⌉ cells. A fixed +50 px leaves 2.6–14.5% of interior labels wrong (#187). The
  column buffer is 58 px at the median and 224 px at worst (85–90° band).
- **GLO-30's native CRS.** On EPSG:4326 input, WBT treats degrees as metres. Every line of sight
  then clears the threshold, and flat, shoulder and footslope never appear (#187).

The sweep (#188) then tested whether 1.0° suits our flat strata, given a published precedent of
0.0001° on 10 m lidar plains. It does not transfer. On GLO-30, thresholds under 0.5° remove flat,
shoulder and footslope from every ruggedness band. 1.0° is the only value on the grid that keeps
all ten classes and holds the flattest band under half flat (45.9%; 73.3% at 1.5°). L and fdist
barely move the labels: 1.8% and 1.4% of the class mix, respectively.

## Decision

1. **Tool.** WhiteboxTools v2.4.0 `Geomorphons --forms`, pinned by the binary's sha256
   (`63ce77174e2abc32590df2b4868b6fb4123d8598a540290ef9669b2cb919647b` for the macOS M-series
   build; each platform build the pipeline runs on is pinned by its own hash in the config). GRASS
   `r.geomorphon` is not used. The two tools differ in distance handling and form lookup, and one
   label set is what the benchmark needs.
2. **Parameters.** `--search=50 --threshold=1.0 --skip=0 --fdist=0`. **All four are passed
   explicitly on every call** and never left to WBT's defaults. They are a measured choice on
   GLO-30 (the sweep), not a literature default, and the dataset card says so.
3. **Spacing.** WBT reads the **same unresampled array**, written with a projected-CRS header whose
   cell size is the WGS84-ellipsoid ground spacing (dx, dy) at the tile's centre latitude. The
   elevations are never reprojected or resampled, so labels stay on the shard grid. An EPSG:4326
   raster is never passed to WBT.
4. **Edge fix: an anisotropic buffered re-read, with no fallback.** Each tile is re-read from the
   source COGs with **L rows** and **⌈L·dy/dx⌉ columns** of buffer on each side, mosaicked across
   neighbouring COGs on the tile COG's own grid (pixel-is-point offset, nearest resampling across a
   column-width band). The centre 224² is cropped after WBT runs. A test tile is read once at the
   widest buffer any variant needs (L = 100: 100 rows, up to 447 columns), and each variant crops
   from that read. Ring-masking is not used: it would score only the centre 124², and the buffered
   pass costs about 1.5 h of laptop wall-clock (#187). WBT's diagonal reach (~L/√2 cells) is
   shorter than the axial one, so the axes set the buffer.
5. **Label values and the ignore label.** Labels are uint8: 0–9 in WBT's form order (flat, peak,
   ridge, shoulder, spur, slope, hollow, footslope, valley, pit) and **255 = ignore**. Ignored
   pixels are excluded from the loss, every metric and the class weights. A pixel is ignored when:
   - it is **ocean**: exact-zero fill, the same definition the corpus's land fraction uses (0.23–0.34%
     of pixels across Task 1's lists). Inland water that GLO-30 flattened keeps its label, as the DEM
     presents it; or
   - its search window reaches **data absent from the GLO-30 release**.

   Two named exceptions read absent data instead of ignoring it:
   - an absent neighbour COG that is verified open ocean is filled with 0 m (S44 E171 today; the
     build's census at the widest buffer extends this list, each entry checked, never assumed);
   - S85 E179 and S86 E179 wrap across the antimeridian to W180.

   The land absences (#187: the N41 neighbours of N42 E043 and N42 E046, all in the train list) get
   the ignore rule. Task 2's tile lists stay identical to Task 1's.
6. **Class weights.** Inverse-sqrt-frequency weights are computed **once**, over non-ignore pixels of
   the primary labels on the full 20k train list. They are committed in the geomorphon config and
   used unchanged at every label fraction. This passes ten class priors from unlabelled train tiles
   into the 1% and 10% regimes, which the dataset card states.
7. **Sensitivity variants (test set only).** Each variant moves one parameter and holds the rest at
   the primary: **L ∈ {25, 100}** and **flatness ∈ {0.75°, 1.25°}**. Already-trained probes are
   re-scored against them, with no retraining. Flatness is included because the sweep shows the
   labels are more sensitive to it than to L.
8. **Determinism and fingerprints**, in ADR-0002's style. Per tile and variant, record a
   `content_hash` of the uint8 label raster and an **input hash** of the buffered elevation window
   fed to WBT. Per variant, record a set fingerprint over sorted `(tile_id, content_hash)`. A **config
   hash** covers the WBT sha256, the four parameters, the buffer formula, the spacing rule and the
   ignore rule. Rebuildability is tested by regenerating a fixed sample of tiles and requiring
   byte-identical labels. If an input hash changes, the upstream data moved; if only a label hash
   changes, the pipeline did. Label rasters are gitignored; the config, the weights and the
   fingerprints are committed under `benchmark/`.

## Consequences

- **The flattest quarter of terrain is about half flat.** At 1.0°, 45.9% of the flattest band's
  pixels are flat. Macro-F1 and mIoU (§6) keep that from dominating the headline, and §6's per-band
  reporting shows it.
- **Some classes are absent from whole bands by construction.** Steep terrain has no flat or
  footslope at any threshold, so per-band class presence is not a valid check. Class presence is
  judged over the whole dataset. How per-band macro metrics treat a class a band lacks is left to
  the pre-registration register.
- **The flatness variants are part of the result, not a robustness footnote.** If the encoder
  ranking flips between 0.75° and 1.25°, that is a finding about the labels and is reported as one.
- **A second WBT build for a rented box needs its own pinned hash**, and the byte-identical rebuild
  check has to pass across the two builds before their labels are mixed.

## Alternatives considered

- **GRASS `r.geomorphon`.** It computes geodesic distances on lat/long rasters, which would remove
  decision 3. It was rejected because WBT is the tool #187 measured on these tiles, and running both
  gives two label sets that nothing reads.
- **Reprojecting to UTM or an equal-area grid.** That gives true metric cells, but it resamples the
  elevations and moves the labels off the shard grid the encoders read.
- **Ring-masking instead of re-reading.** It scores only 31% of each tile. It was the plan's fallback
  for an unaffordable re-read, and the re-read turned out to cost about 1.5 h.
- **A low flatness threshold for plains (≤ 0.2°).** This follows lidar practice, but on GLO-30 it
  erases flat, shoulder and footslope everywhere (the sweep).
- **Dropping tiles with missing source data.** That changes the tile lists Task 2 shares with
  Task 1, where ignoring the affected pixels does not.
- **Per-fraction class weights.** They are more literally "low-label", but at 1% they are noisy, and
  a class absent from the subset gets an infinite weight.
