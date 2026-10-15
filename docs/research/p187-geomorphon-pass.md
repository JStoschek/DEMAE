# #187: one buffered geomorphon pass, timed on real tiles

Throwaway branch, never merged. Measures what §3.1's buffered label pass costs and whether
its edge fix works. Nothing here picks parameters; that is the label-contract ticket (#188).

Scripts: [`p187/geomorphon_pass.py`](p187/geomorphon_pass.py) (20 tiles, every variant and
control), [`p187/read_bench.py`](p187/read_bench.py) (240-tile cold-read benchmark plus the
anisotropy census over all 27k tiles). Per-tile numbers: [`p187/records.json`](p187/records.json).
Run on the laptop (8 cores, 6 performance), over its home network to `eu-central-1`, on 2026-10-15.

## Tool

WhiteboxTools **v2.4.0**, the official `WhiteboxTools_darwin_m_series.zip` that the PyPI
`whitebox` package fetches. Binary sha256 `63ce7717…19647b`. Invoked as `Geomorphons
--search=L --threshold=1.0 --skip=0 --fdist=0 --forms`. §3.1 does not mention `--fdist`, the
distance at which the flatness threshold starts to shrink. It is a fourth parameter, and the
run pins it at 0.

## Where the 324 px window comes from

- **Only `s3://copernicus-dem-30m`.** The shards hold 224 px tiles and nothing else. No local copy of
  the source COGs exists.
- **`dataset.tiling` does not do it yet.** `RasterioCogReader.read_windows` reads in-bounds windows
  from one COG at a time. A buffered read needs (a) a mosaic of up to four COGs, (b) the
  pixel-is-point offset (each COG's origin sits half a pixel north-west of its degree corner), and
  (c) resampling where a neighbour sits in a different column-width band. The script uses
  `rasterio.merge` over the needed neighbours, on the tile COG's grid, with nearest resampling.
- **Alignment is exact.** For every one of the 20 tiles, the buffered window's centre 224×224,
  rounded, equals the shard tile.
- **8,408 of the 27,000 tiles (31%) need a neighbouring COG** at a 50 px buffer: 740 are
  corners and 273 cross a column-width band. With the anisotropic buffer below, the count grows.
- **Five tiles need a COG that does not exist.** Filling it with 0 m is correct for only one of them:
  - S44 E171: the missing neighbour is open ocean, so 0 m is right.
  - S85 E179 and S86 E179: their east neighbour is across the antimeridian. The read has to wrap to W180.
  - N42 E043 and N42 E046: the missing N41 tiles are on land in the Caucasus and absent from the
    public GLO-30 release. 0 m is wrong there, and those two tiles need a rule.

## The edge fix works, with one correction

Bare-224 labels against buffered-crop labels, L = 50, 20 tiles. A pixel's distance is to the
nearest tile edge.

| distance from edge (px) | 0–5 | 5–10 | 10–25 | 25–40 | 40–50 | 50–60 | 60–75 | 75–112 |
|---|---|---|---|---|---|---|---|---|
| labels that disagree | 78% | 73% | 74% | 69% | 42% | 14.5% | 7.6% | 2.6% |

Over the whole tile, 50% of the bare-tile labels are wrong. §3.1 predicted a 50 px ring holding
60% of the area.

**The disagreement past 50 px is anisotropy, and it runs along columns.** WhiteboxTools treats
L as a ground distance, L·dy, so it searches ⌈L·dy/dx⌉ cells along a row. Once a pixel sits at
least ⌈50·dy/dx⌉ columns in from the edge, disagreement is exactly 0 for every tile up to
dy/dx = 1.43. Rows need only 50 px.

So a fixed **+50 px each side does not make labels valid everywhere**. The column buffer has to
be ⌈L·dy/dx⌉, computed per tile:

| over all 27k tiles | median | p99 | max | tiles needing more than 60 / 75 / 100 / 150 px |
|---|---|---|---|---|
| L = 50 column buffer (px) | 58 | 102 | 224 | 11,208 / 2,174 / 294 / 14 |
| L = 100 column buffer (px) | 115 | 203 | 447 | n/a |

The tail comes from the 85–90° band (457 tiles, 360-column COGs). The L = 100 variant also
needs a 100 px row buffer, not 50.

## Two controls the contract needs

- **Do not feed a geographic raster.** On EPSG:4326 input, v2.4.0 reads the cell size as degrees,
  as though they were metres. Every line of sight clears 1°, so **flat, shoulder and footslope never
  appear** in any of the 20 tiles. Labels disagree with the metric run on up to 98% of a tile.
  Every number above comes from the metric run, which is the same array written with a projected
  CRS and per-tile ground `(dx, dy)`.
- **Anisotropy matters to the labels.** Forcing dx = dy changes 0.1–45% of a tile's labels,
  growing with dy/dx (1.43 → 13–32%, 2.09 → 45%). The true per-axis spacing has to go in.

Class shares, metric run, 20 tiles, L = 50: flat 10.4, peak 3.4, ridge 12.6, shoulder 2.9,
spur 13.4, slope 25.1, hollow 10.3, footslope 4.5, valley 14.0, pit 3.4 (%).

## Cost

| per tile, seconds | median | mean | p90 |
|---|---|---|---|
| cold buffered read, sequential (n = 40) | 0.80 | 1.00 | 2.08 |
| WBT L = 50 on 324², one process | 0.102 | 0.106 | 0.125 |
| WBT L = 50 on bare 224² | 0.053 | 0.055 | 0.062 |
| WBT L = 25 on 324² | 0.066 | 0.065 | 0.071 |
| WBT L = 100 on 424² | 0.257 | 0.265 | 0.307 |

The compute times include process startup. Threaded reads ran at 6.4 windows/s with 16 threads
(100 windows) and 4.9 windows/s with 32. Single reads varied 20× between runs on this network.

**Extrapolated:**

- **Reads:** 27k windows take ~70 min at 16 threads, or ~6 h sequentially.
- **Compute:** L = 50 over 27k tiles is ~46 CPU-min, ~8 min across 6 cores.
- **Test-set variants (5k):** L = 25 adds ~5.5 CPU-min and L = 100 ~21 CPU-min.

Reading each test tile once at the widest buffer and cropping for every L avoids a second read.
The wider anisotropic windows add some compute; at worst it is about 1.5×.

**Total: about 1.5 h of wall-clock on the laptop, bound by network reads.** No GPU, no rented
box. §3.1's "~27k windowed COG reads, CPU + network" holds in order of magnitude.
