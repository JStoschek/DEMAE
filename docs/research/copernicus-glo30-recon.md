# Reconnaissance — Copernicus DEM GLO-30 corpus

Resolves the Phase-0 wayfinder ticket **"Reconnoiter the Copernicus GLO-30 corpus"**
(GitHub issue #3). Combines empirical probing of `s3://copernicus-dem-30m`
(rasterio 1.5.0 / GDAL 3.12.1, anonymous access) with authoritative facts from the
bucket `readme.html`, the AWS Open Data registry, and the Copernicus DEM Product
Handbook i5.0. Findings feed the corpus-recipe, storage/manifest, and dataloader
decision tickets.

---

## 1. Bucket layout, naming, discovery, scale

- **Bucket:** `s3://copernicus-dem-30m`, region **eu-central-1**, public / no credentials.
  Anonymous access needs `AWS_NO_SIGN_REQUEST=YES` (env var for GDAL/rasterio; `--no-sign-request` for the CLI).
- **Scale:** **26,454** top-level tile prefixes — i.e. 26,454 released 1°×1° land-containing cells
  (fully-ocean cells simply have no prefix; see §2).
- **Naming:** one prefix per 1°×1° cell:
  `Copernicus_DSM_COG_10_{N|S}<lat>_00_{E|W}<lon>_00_DEM/`
  (token `10` = GLO-30 / 30 m; GLO-90 uses `30`. lat 2-digit, lon 3-digit, `_00` decimals.)
- **Per-prefix objects:** the DEM COG `..._DEM.tif` (~43 MB for a full 3600² tile) plus a
  `Copernicus_DSM_10_..._.xml` metadata sidecar and `AUXFILES/`, `INFO/`, `PREVIEW/` subfolders.
- **Programmatic discovery:** STAC v1.0.0 catalog at
  `https://copernicus-dem-30m-stac.s3.amazonaws.com/`. Bucket listing (`aws s3 ls`) also works
  and is simplest for "which 1° cells exist."

## 2. No-data & ocean encoding  *(→ corpus-recipe / land masking)*

- **`nodata` is `None` on every tile probed** — there is **no no-data sentinel**. dtype `float32`.
- **Ocean appears two ways:**
  1. **Fully-ocean 1° cells have no tile at all** — opening an absent key raises `RasterioIOError`.
     (Handbook: *"ocean areas do not have tiles, there one can assume height values equal to zero."*)
     So the primary "ocean filter" is *free*: only sample cells whose prefix exists.
  2. **Within coastal tiles, sea reads as elevation exactly `0.0`.** Measured zero-fractions on the
     center 224² window: Gulf-of-Guinea 1.00, a mostly-sea Denmark cell 1.00, California coast 0.80.
- **⚠️ Zero ≠ always ocean.** Genuine land sits at or below 0 m: the **Netherlands** tile reads
  **−12 m** (polders below sea level), Svalbard coast dips to −1 m. A naive "drop all zeros" mask
  would delete real low-lying land and mislabel it. Land masking needs zero-**fraction** /
  connected-component logic, not a per-pixel `==0` test. **This is a corpus-recipe decision input.**

## 3. Restricted / non-released tiles  *(→ corpus-recipe)*

- GLO-30 **Public** withholds a small subset of tiles over **specific countries** (authorised-user-only).
  A "list of non-released tiles / countries" is referenced by the registry and product docs but is not
  needed operationally: **a withheld cell is simply an absent prefix**, indistinguishable from ocean at
  the bucket level, and handled by the same "only sample existing prefixes" rule.
- Consequence for the project's honesty framing: the corpus has **coverage gaps over a few countries**;
  worth a one-line note in the model card. No code needed to "skip" them — they're already absent.

## 4. Elevation values, datum, artifacts, clipping  *(→ corpus-recipe / normalization)*

- **Vertical datum:** heights are metres above the **EGM2008 geoid** (EPSG 3855). Horizontal CRS
  **WGS84 / EPSG:4326**. Product is a **DSM** — surface incl. buildings & vegetation (not bare earth).
- **Observed range:** ≈ **−12 m** (Netherlands) to **8,737 m** (Everest tile; true summit ~8,849 m).
  Everest full-tile: min 187, p1 327, median 2162, p99 6215, max 8738; **0 NaNs**, 0 zero-pixels.
- **Source is radar (TanDEM-X)**, so fewer lidar-style single-pixel spikes than 3DEP, but known
  artifacts exist: water-surface flattening, coastline/edge effects, and infilled voids. A **robust
  per-tile clip (≈1st–99th percentile) before standardizing** is prudent (confirms plan §2.4.4).
- Supports the plan's default: **per-tile standardization** (subtract tile mean / divide tile std),
  storing mean+std for inversion. Watch **near-constant tiles** (std ≈ 0: open sea, flat ice interiors
  like the Greenland tile at ~3040 m ±9) — division blows up; needs an explicit guard.

## 5. COG windowed-read ergonomics & timing  *(→ storage/manifest, dataloader, throughput)*

- **Windowed reads work great:** a 224×224 window is fetched without downloading the 43 MB tile,
  via COG tiling + HTTP range requests. Internal **overviews at ×2/×4/×8** are present on every tile.
- **Measured latency (US client → eu-central-1 bucket):** `open` ≈ 0.45–1.2 s (first/cold highest),
  `read(224²)` ≈ 0.2–0.48 s. **Opening dominates**; window reads are cheap once open.
- **Amortization:** an equatorial 3600² tile yields up to ~256 non-overlapping 224² windows, so reusing
  one open handle spreads the open cost across many windows. Effective cost ≈ *open-once-per-COG +
  cheap-window-reads*, dominated by transatlantic latency and per-COG opens.
- **Throughput signal for local-vs-cloud:** naive ~0.3 s/window ⇒ 50k windows ≈ ~4 h single-threaded;
  parallelism + handle reuse cut this to minutes. The bucket is **eu-central-1** and the dev machine is
  US, so a **same-region cloud burst (or a local COG/tile cache) would slash the open latency** — the
  main lever if local proves slow. *(Precise cloud-burst trigger still needs the real pipeline
  measurement in the build ticket; recon gives only the order of magnitude.)*
- **Cheap TRI pre-screen:** the ×8 overview (≈450² at low latitude) lets stratification rank candidate
  windows for ruggedness without full-res reads — a concrete efficiency lever for the recipe ticket.

## 6. Pixel geometry & latitude-dependent dimensions  *(→ corpus-recipe / tiling geometry)*

**The single most consequential finding for tiling.** Rows are always 3600 (latitude sampled at a
constant 1″); **columns shrink poleward** as Copernicus coarsens longitude sampling. Authoritative
table (bucket `readme.html`), **confirmed empirically** across a latitude ladder:

| Latitude band | Lon spacing | COG width × height | Overviews (width) | Empirically confirmed |
|---|---|---|---|---|
| 0–50°  | 1″    | **3600** × 3600 | 1800/900/450 | ✅ Alps, Himalaya, Sahara, Andes, California |
| 50–60° | 1.5″  | **2400** × 3600 | 1200/600/300 | ✅ Netherlands 52°, Denmark 55° |
| 60–70° | 2″    | **1800** × 3600 | 900/450/225  | ✅ Norway 60°, Iceland 65° |
| 70–80° | 3″    | **1200** × 3600 | 600/300/150  | ✅ N-Norway 70°, Greenland 72°, Svalbard 78° |
| 80–85° | 5″    | **720**  × 3600 | 360/180/90   | ✅ Ellesmere 82° |
| 85–90° | 10″   | **360**  × 3600 | 180/90/45    | (table; not probed) |

**Implications for the tiling-geometry decision (#4):**
- A 224×224 **pixel** window is **not** a square **ground** patch. N–S ground spacing is ~30 m
  everywhere; E–W ground spacing = `lon_spacing″ · 30.9 m · cos(lat)`, so within 0–50° pixels get
  progressively *taller-than-wide* in ground terms (up to ~1.6:1 near 50°), then reset at each band
  boundary above 50°. Ground area per 224² tile therefore varies substantially with latitude.
- **The crux decision for #4:** accept pixel-square windows (simplest, MAE-native, per-tile
  standardization + shape-learning likely absorb the anisotropy) **vs.** reproject to a metric/equal-area
  grid per window (expensive) **vs.** cap the corpus latitude (e.g. ≤ 60°, bounding distortion and
  dropping mostly low-information ice — but sacrificing high-arctic terrain diversity).
- All bands are ≥ 224 wide, so 224² windows always fit; only the **count per tile** and **ground scale**
  differ by band.

---

## Handoff to downstream tickets

- **Decide the corpus-construction recipe (#4):** land-mask via existing-prefix + zero-**fraction**
  (not `==0`); the pixel-vs-ground / latitude-distortion crux above; robust per-tile clip + std≈0 guard;
  TRI pre-screen on the ×8 overview; consider a latitude cap for the debug corpus.
- **Decide storage format + manifest (#5):** manifest must record source COG, lon/lat + latitude band
  (for the geo-split *and* to reconstruct ground scale), stored mean+std, TRI/stratum. Windowed-read
  latency argues for **pre-materializing** normalized tiles rather than streaming COGs at train time.
- **Decide the dataloader (#6):** confirms pre-materialize-before-GPU; overviews/COG range-reads are the
  fallback path, not the hot path.
- **Fog (unchanged):** the concrete cloud-burst throughput threshold still awaits the build ticket's
  real measurement; recon supplies only the order of magnitude and the eu-central-1-vs-US lever.

## Sources

- Bucket README (authoritative sampling table, ocean statement): https://copernicus-dem-30m.s3.amazonaws.com/readme.html
- AWS Open Data registry (bucket, region, STAC, license): https://registry.opendata.aws/copernicus-dem/
- Copernicus DEM Product Handbook i5.0 (datum EGM2008, resolution): https://dataspace.copernicus.eu/sites/default/files/media/files/2024-06/geo1988-copernicusdem-spe-002_producthandbook_i5.0.pdf
- Empirical probes: `s3://copernicus-dem-30m` via rasterio 1.5.0 / GDAL 3.12.1, July 2026.
