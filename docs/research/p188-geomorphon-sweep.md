# #188: geomorphon parameter sweep on GLO-30

Throwaway branch, never merged. This is the evidence behind ADR-0014's parameter choice. The
contract itself lives in the ADR.

Scripts: [`p188/p188_sweep.py`](p188/p188_sweep.py) (the sweep) and
[`p188/summarize.py`](p188/summarize.py). Raw per-tile class counts are in
[`p188/sweep.json`](p188/sweep.json) and [`p188/sweep_hi.json`](p188/sweep_hi.json) (1.5° and
2.0°, L = 50, fdist 0). The full table is [`p188/summary.txt`](p188/summary.txt). Run on the
laptop on 2026-10-15.

## Setup

- **Tool:** WhiteboxTools v2.4.0, `WhiteboxTools_darwin_m_series.zip` from whiteboxgeo.com.
  Binary sha256 `63ce77174e2abc32590df2b4868b6fb4123d8598a540290ef9669b2cb919647b`, the same
  binary #187 timed. The GitHub v2.4.0 release ships no binaries.
- **Tiles:** 50 from Task 1's committed test list, taken in keyed-BLAKE2b order. The first
  10 fully-land tiles (`land_frac ≥ 0.999`) are drawn from each of §5's four ruggedness bands
  (quartiles of `slope_mean`), plus 10 more from the flat k-means strata 2, 5, 7 and 17. Bands
  0–3 hold 18, 12, 10 and 10 tiles.
- **Read:** each tile is read once at the widest anisotropic buffer: 50 rows each side and
  ⌈50·dy/dx⌉ columns each side. The L = 25 runs crop that window down to their own buffer.
- **Spacing:** the same array, written with a projected-CRS header. dx and dy are the WGS84
  ellipsoid ground spacing at the tile's centre (pyproj `Geod`).
- **Shares:** class shares over the centre 224², with exact-zero (ocean) pixels excluded.
- **Grid:** flatness {0.1, 0.2, 0.5, 1.0}° × fdist {0, L/2} × L {25, 50}, then 1.5° and 2.0°
  at L = 50 with fdist 0.

## Result

Flat share by flatness threshold (L = 50, fdist 0):

| flatness | 0.1° | 0.2° | 0.5° | 1.0° | 1.5° | 2.0° |
|---|---|---|---|---|---|---|
| flat %, bands 0 / 1 / 2 / 3 | 0.0 / 0.0 / 0.0 / 0.0 | 0.1 / 0.1 / 0.0 / 0.0 | 10.4 / 2.3 / 0.0 / 0.0 | 45.9 / 6.1 / 0.0 / 0.5 | 73.3 / 16.3 / 0.1 / 2.5 | 86.6 / 33.5 / 1.0 / 4.5 |
| flat %, flat-strata tiles (median) | 0.0 | 0.0 | 1.7 | 43.3 | 73.2 | 84.6 |
| shoulder / footslope %, all tiles | 0.0 / 0.0 | 0.1 / 0.3 | 2.2 / 3.3 | 5.0 / 6.1 | 5.2 / 6.3 | 4.7 / 5.9 |

- **Below 0.5°, flat, shoulder and footslope disappear in every band.** GLO-30 is a radar
  surface model, and its decimetre-scale noise across one 30 m cell is enough to break the
  flatness test. The North Carolina statewide map's 0.0001° for its Coastal Plain was set on
  10 m lidar and does not transfer.
- **1.0° is the only threshold on the grid that keeps all ten classes** while holding the
  flattest band under half flat (45.9%).
- **Flatness is the parameter the labels are most sensitive to.** Between 0.5° and 1.5° the
  flattest band's flat share runs from 10% to 73%.
- **L and fdist barely move the labels.** At 1.0°, the total-variation distance from
  L = 50, fdist 0 over all tiles is 0.018 for L = 25 and 0.014 for fdist = L/2.
- **Steep bands have no flat or footslope at any threshold** (band 2: 0.0% at 1.0°). Class
  presence can only be judged across the whole dataset, not per band.

Class shares at the chosen setting (L = 50, 1.0°, fdist 0), all 50 tiles: flat 18.1,
peak 2.9, ridge 11.3, shoulder 5.0, spur 11.2, slope 22.1, hollow 8.7, footslope 6.1,
valley 12.0, pit 2.7 (%). This sample leans toward flat terrain on purpose, so these shares
are not the test set's.
