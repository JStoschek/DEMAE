# Mask spec — Copernicus DEM GLO-30 quality layers (EDM / FLM / WBM)

Resolves the wayfinder ticket **"Establish where the EDM/FLM/WBM masks live and what access
they require"** (GitHub issue #14). Sibling to `copernicus-glo30-recon.md`, and follows the same
method: authoritative facts from the **Copernicus DEM Product Handbook i5.0** (the document
that *owns* the code vocabularies, §1.2.5, Tables 5–8) and the **licence PDF the bucket
itself ships** (`INFO/eula_F.pdf`), cross-checked against empirical probes of
`s3://copernicus-dem-30m` (rasterio 1.5.0 / GDAL 3.12.1, anonymous). Feeds the
mask-prevalence measurement and the corpus-recipe / model-card tickets.

**Bottom line:** ✅ **The masks are already in the bucket the pipeline reads.** Every tile
prefix in `s3://copernicus-dem-30m` carries `AUXFILES/{EDM,FLM,WBM}.tif` — anonymous, no
registration, no account, no new distribution channel (§1.1). The ticket's premise that they
live behind an ESA/Copernicus gate is **falsified**. All three vocabularies are **fully pinned
from the primary spec** — no guessing, no secondary sources needed. Masks are **bit-exactly
co-registered** with the DEM, and their COG overviews are **categorical-safe** (§7).
Redistribution of derived statistics is **permitted** under an explicit adaptation +
distribution grant, subject to one mandatory attribution notice.

---

## 1. What ships, and which spec owns it

- The quality layers are defined in **Product Handbook i5.0** §1.2.5 (Campaign ID
  GEO.2018-1988-2, doc GEO1988-CopernicusDEM-SPE-002, version 5.0, dated 29.11.2022 —
  the current issue). Tables 6/7/8 are the **normative** code lists used below.
- **Handbook Table 5** lists six quality layers: EDM, FLM, **HEM** (32-bit float height
  error), WBM, **SRC** (source data layer, KML), **ACM** (accuracy layer, KML).
- **The AWS bucket ships five of them per tile** ✅ (confirmed by listing
  `Copernicus_DSM_COG_10_N45_00_E007_00_DEM/`): `AUXFILES/` carries `_EDM.tif`, `_FLM.tif`,
  `_HEM.tif`, `_WBM.tif`, `_ACM.kml`; `PREVIEW/` carries 8-bit quicklooks (`_EDM_QL.tif`
  etc.) and `_SRC.kml`. **Only the three 8-bit thematic masks (EDM/FLM/WBM) are in scope here.**
- **Encoding (Handbook Table 5, confirmed empirically):** EDM/FLM/WBM are **8-bit unsigned
  integer GeoTIFF, LZW-compressed**. Probes report `dtype=uint8`, **`nodata=None`** — i.e.
  **no GDAL no-data sentinel**; "void" is carried *in-band* as **code 0** (see §2–§4). This
  mirrors the DEM's own `nodata=None` (recon §2) — the same trap, same shape: **you cannot
  rely on a nodata mask, you must test the code value.**
- ⚠️ **One spec-vs-reality gap worth naming.** The Handbook says quality layers "are
  available for the instances of the Copernicus DEM with **DGED & INSPIRE** format" — it does
  **not** enumerate a COG instance, because the AWS COG bucket is an AWS/ESA **repackaging**
  that postdates/sits outside the Handbook's instance taxonomy. The bucket's per-tile layout
  (`INFO/eula_F.pdf` + `DEM/` + `AUXFILES/{EDM,FLM,WBM,HEM,ACM}` + `PREVIEW/`) matches the
  Handbook's **DGED product structure** (Figure 11) almost exactly, so the bucket is best read
  as *COG-converted GLO-30 Public DGED*. **This is an inference, not a spec statement** — but
  it is not load-bearing: the masks are empirically present and their codes empirically fall
  inside the Handbook vocabulary (§6), which is the only thing the prevalence table needs.

## 1.1 Distribution channel, access gate, and addressing  *(→ `masks.py`)*

**The ticket asked where the masks live and what access they require. Answer: they live in the
bucket we already read, and require none.**

- ✅ **Channel: `s3://copernicus-dem-30m` — the same AWS Open Data bucket the pipeline already
  streams the DEM from** (`src/dataset/config.py:COPERNICUS_GLO30_BUCKET`). **No ESA/Copernicus
  PRISM/panda account, no OpenTopography, no mirror, no registration, no cost, no terms to
  accept beyond the EULA already shipped in-band (§5).** The plan's §10 "long-pole external
  dependency" and its §10 *"mask access friction"* risk **do not exist**.
- ✅ **Access gate: none.** Verified anonymously two ways: `aws s3 ls --no-sign-request` and
  GDAL `/vsis3/` with `AWS_NO_SIGN_REQUEST=YES`; **anonymous HTTPS range reads also work**
  (`curl -r 0-1023` → `HTTP 206`) against
  `https://copernicus-dem-30m.s3.amazonaws.com/...`, so `/vsicurl/` is an equally valid path.
- ✅ **Universality — this is not a per-tile lottery.** A **latitude-stratified sample of 216
  tiles across all 18 ten-degree bands** (−90 … +80) found **zero tiles missing any of the
  three masks**; a separate uniform 30-tile sample likewise found 30/30 complete. The bucket
  holds **26,450 tile prefixes**. Treat mask presence as an invariant — but see the fog note.

**Addressing.** Given a 1° cell stem `S = Copernicus_DSM_COG_10_N45_00_E007_00`:

| Layer | Key |
|---|---|
| DEM | `s3://copernicus-dem-30m/{S}_DEM/{S}_DEM.tif` |
| EDM | `s3://copernicus-dem-30m/{S}_DEM/AUXFILES/{S}_EDM.tif` |
| FLM | `s3://copernicus-dem-30m/{S}_DEM/AUXFILES/{S}_FLM.tif` |
| WBM | `s3://copernicus-dem-30m/{S}_DEM/AUXFILES/{S}_WBM.tif` |

⚠️ **Two naming traps for `masks.py`.** (1) The **parent directory keeps the `_DEM` suffix**
even for mask keys — the prefix is `{S}_DEM/`, not `{S}/`. (2) The mask **filename** swaps only
the trailing token (`_EDM`/`_FLM`/`_WBM`) while **retaining `COG_10` in the stem** — unlike the
sidecar metadata XML, which drops it (`Copernicus_DSM_10_N45_00_E007_00.xml`). So the mask key
is derivable from the existing DEM key by a **single trailing-token substitution plus an
`AUXFILES/` segment** — no separate manifest, no new resolver, no lookup table.

**Cost: negligible.** Across the 30-tile sample, **median EDM ≈ 106 KB, FLM ≈ 103 KB,
WBM ≈ 49 KB** (range 10 KB–0.6 MB) — all LZW-compressed uint8. Reading **all three masks for a
tile costs ~0.25 MB against the DEM's ~42 MB**, i.e. **well under 1% overhead**. ⚠️ Do **not**
casually pull `_HEM.tif` alongside them — it is float32 and **~47 MB, larger than the DEM
itself**; it is out of scope here and would dominate the read budget.

## 2. EDM — Edit Data Mask  *(→ prevalence table)*

Handbook §1.2.5.1: the EDM *"indicates all DEM pixels that were modified during the terrain
and hydro editing process"*, and **"represents the last editing process that was applied to a
pixel."** ⚠️ **That last clause is the one to internalize: EDM is *not* a bitfield and codes
are *not* composable — it is a single-valued "last operation wins" label.** A pixel that was
both filled and smoothed reports only whichever came last. Prevalence-per-code is therefore a
partition (fractions sum to 1), **not** an accounting of everything done to a pixel.

**Table 6: Edit Data Mask – Pixel Values** — complete, verbatim from Handbook i5.0 p.18:

| Code | Meaning |
|---|---|
| 0 | Void (no data) |
| 1 | Not edited |
| 2 | Infill of external elevation data |
| 3 | Interpolated pixels |
| 4 | Smoothed pixels |
| 5 | Airport editing |
| 6 | Raised negative elevation pixels |
| 7 | Flattened pixels |
| 8 | Ocean pixels |
| 9 | Lake pixels |
| 10 | River pixels |
| 11 | Shoreline pixels |
| 12 | Morphed pixels (series of pixels manually set) |
| 13 | Shifted pixels |

- **Codes 0–13 is the whole vocabulary** — the table has no "reserved"/"other" row and no
  values above 13.
- **The "pristine" code is `1` (Not edited).** Everything else marks *some* intervention.

## 3. FLM — Filling Mask  *(→ prevalence table; the TanDEM-X-native question)*

Handbook §1.2.5.2: the FLM *"is created primarily during the terrain editing process. All
edited and filled pixels are flagged in this mask. For filled pixels, the fill source is
specified."*

**Table 7: Filling Mask – Pixel Values** — complete, verbatim from Handbook i5.0 p.19:

| Code | Meaning | Native TanDEM-X? |
|---|---|---|
| 0 | Void (no data) | — |
| 1 | Edited (except filled pixels) | **yes** (edited, not replaced) |
| 2 | Not edited / not filled | **yes** (pristine) |
| 3 | ASTER | no — filled |
| 4 | SRTM90 | no — filled |
| 5 | SRTM30 | no — filled |
| 6 | GMTED2010 | no — filled |
| 7 | SRTM30plus | no — filled |
| 8 | TerraSAR-X Radargrammetric DEM | no — filled |
| 9 | AW3D30 | no — filled |
| 100 | Norway DEM | no — filled |
| 101 | DSM05 Spain | no — filled |
| 102 | Norway DEM v2 | no — filled |

- ⚠️ **The value space is non-contiguous: 0–9, then a jump to 100–102.** A prevalence table
  keyed on `range(0, 10)` would **silently drop the three national-DEM fill sources** (Norway,
  Spain). Bin on the explicit value list, or on `np.unique`.
- ⚠️ **Codes 1 and 2 are the trap for "is this pixel TanDEM-X?".** `2` = untouched. `1` =
  **edited but not replaced** — still TanDEM-X-derived elevation. So:
  **native ⇔ `FLM ∈ {1, 2}`; externally-filled ⇔ `FLM ≥ 3`.** Reading "not filled" as only
  code 2 would misclassify every *edited* native pixel as non-native.
- **Code 8 (TerraSAR-X Radargrammetric DEM) is a judgement call for the model card**: it is a
  *fill from a different sensor/technique*, not TanDEM-X interferometry, so it is grouped as
  "filled" above — but it is not a legacy third-party DEM in the way 3–7/9 are. Worth a
  footnote if its prevalence turns out non-trivial.
- **The Handbook footnotes the provenance of each fill source** (p.19, footnotes 2–8):
  ASTER GDEM (NASA/METI/AIST/Japan Space Systems + U.S./Japan ASTER Science Team); SRTM
  (USGS / NASA LP DAAC, `10.5067/MEaSUREs/SRTM/SRTMGL1.003`); GMTED2010 (USGS); AW3D30
  (JAXA); Norway DEM v1 & v2 (©Kartverket, **CC BY 4.0**); DSM05 Spain (CNIG). ⚠️ **If the
  prevalence work ever escalates to redistributing *pixels* rather than *statistics*, these
  upstream attributions ride along** — see §5.

## 4. WBM — Water Body Mask  *(→ prevalence table; land-masking cross-check)*

Handbook §1.2.5.4: the WBM *"shows all DEM pixels, which are classified as water and edited
according to the categories Ocean, Lake or River."*

**Table 8: Water Body Mask – Pixel Values** — complete, verbatim from Handbook i5.0 p.21:

| Code | Meaning |
|---|---|
| 0 | No water |
| 1 | Ocean |
| 2 | Lake |
| 3 | River |

- **Four values only.** Note the asymmetry with EDM/FLM: **WBM has no "void" code** — `0` is
  the *substantive* class "No water", not no-data.
- **This is the principled fix for recon §2's zero-fraction hack.** Recon flagged that
  `elevation == 0` is an unreliable ocean test (the Netherlands tile reads −12 m; genuine land
  sits at/below 0 m). **`WBM == 1` is the spec's own ocean label** — a direct, per-pixel,
  datum-independent answer that the zero-fraction heuristic was approximating. **Strong
  recommendation for the corpus-recipe ticket: replace zero-fraction land masking with WBM.**

## 5. Licensing — verdict: derived statistics are **redistributable**  *(→ repo, model card)*

The governing document is **the licence the bucket ships in every tile's `INFO/eula_F.pdf`**,
titled *"Licence for Copernicus DEM instance **COP-DEM-GLO-30-F Global 30m Full, Free & Open**
— Licence for the use of the Copernicus WorldDEM-30"*. ✅ **This is the primary licence
instrument for exactly the instance we read**, fetched directly from
`s3://copernicus-dem-30m/.../INFO/eula_F.pdf` — not a registry paraphrase.

**Rights granted (Article 4, verbatim).** The Licensor grants **non-exclusive** rights of:
*(a) reproduction; (b) distribution; (c) communication to the General Public; (d) adaptation,
modification and combination with other data and information.* Article 5: **free of charge**.
Article 3: **worldwide and without limitation in time.**

**Verdict on the ticket's actual question** ✅: a committed table of per-tile mask-code
fractions is an **adaptation/derivation** (Art. 4(d)) that we then **distribute /
communicate to the public** (Art. 4(b),(c)). **All three are expressly granted. Committing
the prevalence table to this repo — public — is permitted.** There is **no non-commercial
clause, no share-alike, and no redistribution prohibition** in the F instance. (The Preamble
excludes only **WorldDEM-10**, the higher-resolution product, from public distribution —
irrelevant to GLO-30.)

**Mandatory attribution (Article 6) — the one obligation with teeth.** Because our output is
**adapted/modified** data, **Art. 6(b)** is the applicable notice, verbatim:

> "produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space
> GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved"

Also required when distributing:
- **Art. 6(c)** — a liability disclaimer must be passed to downstream users, verbatim:
  *"The organisations in charge of the Copernicus programme by law or by delegation do not
  incur any liability for any use of the Copernicus WorldDEM-30".*
- **Art. 6(d)** — must not imply endorsement by Airbus/ESA/the Copernicus programme.
- **Art. 6(e)** — downstream users must be bound by the same obligations.
- **Art. 9** — we receive **no IPR title**; but *"Neither the Licensor nor the Provider will
  claim any IPRs … as obtained by the User through his own work while exercising the use
  rights"* — i.e. **our derived statistics and trained weights are our own work.** This is
  the clause that makes a DEM-pretrained model's weights clean.
- **Art. 6(a)** — the *unmodified*-redistribution notice (`© DLR e.V. 2010-2014 and © Airbus
  Defence and Space GmbH 2014-2018 …`) applies **if we ever ship raw GLO-30 tiles** (e.g. a
  cached corpus artifact), as distinct from statistics. Worth pre-deciding.
- **AWS registry citation** (separate from, and additive to, the licence): *"Copernicus
  Digital Elevation Model (DEM) was accessed on `DATE` from
  https://registry.opendata.aws/copernicus-dem."* **ESA's citation DOI:**
  `https://doi.org/10.5270/ESA-c5d3d65`.

⚠️ **One honest interpretive caveat.** Art. 2 defines *'Copernicus WorldDEM-30'* as *"a
Digital Surface Model providing a horizontal sampling of 30m"* — **the definition names the
elevation model and does not explicitly enumerate the auxiliary masks.** A literalist could
ask whether EDM/FLM/WBM are "the Copernicus WorldDEM-30" at all. **Practically this is a
non-issue in both directions:** the masks are shipped inside the same product unit, under the
same per-tile EULA, and the Handbook calls them *"part of the Copernicus DEM data product"*
(§1.2.5) — so they are either covered by the grant (permitted, with Art. 6 notice) or not
licensed material at all. **The safe and clearly-correct action is identical either way:
treat them as covered and carry the Art. 6(b) notice.** Flagged only so nobody later
"discovers" it and thinks it was missed. This is the single point in this document where the
primary text does not answer the question with total crispness.

## 6. Co-registration — spec statement **and** empirical confirmation  *(→ prevalence table)*

**Spec statement (Handbook §1.2.5), verbatim:** *"The grid spacing corresponds with the DEM
data (see Table 5)."* The Handbook also notes the masks' provenance: *"All available 8 Bit
data (EDM, FLM, WBM) are obtained from the auxiliary information of the edited WorldDEM (grid
spacing: 0.4"), by applying a **majority value resampling**. A **consistent coding of thematic
classes throughout all 8 Bit data** has been taken into account."*

⚠️ The spec sentence pins **grid spacing** but is thin on **extent/origin**, and "majority
value resampling" is a *downsampling* statement (0.4″ → 1″), not an alignment guarantee. So
this was verified directly rather than assumed. ✅ **Confirmed exactly, on two tiles across
two different latitude bands:**

| Tile | Layer | Size | dtype | Affine transform | Bounds |
|---|---|---|---|---|---|
| N45E007 (Alps) | DEM | 3600×3600 | float32 | `(0.00027778, 0, 6.99986111, 0, −0.00027778, 46.00013889)` | identical |
| | EDM / FLM / WBM | 3600×3600 | uint8 | **byte-identical to DEM** | identical |
| N65W019 (Iceland) | DEM | **1800**×3600 | float32 | `(0.00055556, 0, −19.00027778, 0, −0.00027778, 66.00013889)` | identical |
| | EDM / FLM / WBM | **1800**×3600 | uint8 | **byte-identical to DEM** | identical |

**Same width, height, CRS (EPSG:4326), affine transform, and bounds — including the poleward
column-shrink** (recon §6: 1800 columns at 65° from 2″ longitude sampling). **The masks track
the DEM's latitude-band geometry.** ✅ **Verdict: a mask window read with the *same*
`rasterio` window as the DEM window is pixel-aligned by construction. No reprojection, no
resampling, no offset handling. The prevalence measurement can reuse the DEM's exact window
logic verbatim.**

### 6.1 Internal consistency — the vocabularies cross-validate ✅

The Handbook's *"consistent coding of thematic classes"* claim is **measurably true**, and it
gives us a **free correctness check on any prevalence code we write**. Measured full-tile
fractions:

- **`EDM == 1` (Not edited) ≡ `FLM == 2` (Not edited / not filled)** — to 4 decimal places.
  Alps: both **82.1089%**. Iceland: both **93.8992%**.
- **`EDM == 2` (Infill of external elevation data) ≡ `Σ FLM ≥ 3` (all fill sources)** — exactly.
  Alps: EDM 16.4535% vs FLM (3:4.2901 + 4:0.0676 + 5:9.4740 + 6:0.0120 + 7:2.6098) =
  **16.4535%**. Iceland: EDM 1.5723% vs FLM (3:1.5708 + 6:0.0015) = **1.5723%**.
- **`WBM ∈ {1,2,3}` ≡ `EDM ∈ {8,9,10}`** (ocean/lake/river), class-for-class. Iceland:
  ocean **4.2165%**, lake **0.0053%**, river **0.0407%** in both masks.

**Every observed code fell inside the Handbook vocabulary; no undocumented values appeared.**
**Recommendation: assert these three identities in the prevalence-table test** — they are a
cheap tripwire that catches a mis-keyed bin or an off-by-one window before it corrupts a
committed table. Observed codes were EDM {1,2,3,4,5,8,9,10,11}, FLM {1,2,3,4,5,6,7},
WBM {0,1,2,3} — note **`EDM == 0` / `FLM == 0` (void) did not appear on either tile**, so the
void path is untested here (expected: voids are rare in edited GLO-30).

## 7. Overview pyramids — categorical-safe, but rare codes evaporate  *(→ prevalence, Gate A3)*

The masks ship the **same `[2, 4, 8]` overview pyramid as the DEM**, with the same 1024×1024
blocking. That matters because ADR-0001 budgets reads at the **×8 overview**, and averaging a
*categorical* raster would produce codes that mean nothing (an "EDM 6.5" between *Raised
negative elevation* and *Flattened*). **This was tested, not assumed.**

✅ **The overviews are categorical-safe.** Reading EDM/FLM/WBM at ×8 produced **zero codes
outside the full-resolution value set** on every layer and tile probed — consistent with the
Handbook's *"majority value resampling"* / *"consistent coding of thematic classes"* posture
(§6). ⚠️ They are **not** plain corner-decimation either: ×8 agrees with a nearest-neighbour
subsample only **89–98%** of the time, which is the signature of a **mode** resampler — the
right choice, and *better* than nearest for this data. **Verdict: `masks.py` may read masks at
×8 without a full-res fallback.**

⚠️ **But the ×8 error is class-dependent, and it is not symmetric.** Measured full-tile,
per-code, full-res vs ×8:

| Tile | Layer | Code | Full-res | ×8 | Abs err |
|---|---|---|---|---|---|
| N45E007 | EDM | 1 *Not edited* | 0.821089 | 0.845007 | 0.023918 |
| | EDM | 2 *External infill* | 0.164535 | 0.152825 | 0.011710 |
| | EDM | **3** *Interpolated* | 0.001934 | **0.000000** | **vanishes** |
| | EDM | 4 *Smoothed* | 0.008960 | 0.000286 | 0.008674 |
| | EDM | **11** *Shoreline* | 0.000061 | **0.000000** | **vanishes** |
| | FLM | 5 *SRTM30* | 0.094740 | 0.081877 | 0.012864 |
| | WBM | 0 *No water* | 0.996683 | 0.998212 | 0.001529 |
| N60E010 | EDM | **3, 4, 5, 11** | 0.001045 … 0.000020 | **0.000000** | **all vanish** |
| | EDM | 9 *Lake* | 0.080447 | 0.080000 | 0.000447 |
| | WBM | 2 *Lake* | 0.080447 | 0.079674 | 0.000773 |

**The rule: mode resampling is a majority vote, so a code that is never locally dominant is
erased.** Dominant classes are accurate to **~0.01–0.03**; classes below ~1% are systematically
**under-counted or driven to exactly zero**. EDM **3 / 4 / 5 / 11** and FLM **3** disappeared
entirely on tiles where they demonstrably exist.

**Two consequences, and they point opposite ways:**

1. ✅ **For quality *gating*, ×8 is fine.** The plan's thresholds (WBM water **>0.30** drop,
   EDM/FLM **>0.25** drop) sit an order of magnitude above the ×8 error on the classes that
   drive them. Water especially: total water (`WBM ∈ {1,2,3}`) tracked full-res to
   **~0.001**. The **0.05 flag** threshold has thinner margin (worst observed error 0.027) and
   is the one to re-check at full res.
2. ⚠️ **For the *prevalence table*, ×8 is not fine — and that is this ticket's warning to its
   successor.** A per-code table built on ×8 would report **hard zeros for real classes** and
   silently understate every rare edit. The table is a **committed artifact** meant to *set*
   thresholds; baking a resampling artifact into it would corrupt the very numbers the Phase-C
   knobs are tuned against. **Recommendation: measure prevalence at full resolution.** This is
   affordable precisely because the masks are tiny (§1.1) — a ~500-tile full-res pass over
   ~0.25 MB/tile is **~125 MB total**, versus ~21 GB for the equivalent DEM pass. **The
   read-budget argument that motivates ×8 for the DEM simply does not transfer to the masks.**

---

## Handoff to downstream tickets

- **`masks.py` (#18): unblocked now, and cheaper than planned.** No acquisition step, no
  credentials, no new channel — derive the mask key from the DEM key by trailing-token
  substitution + `AUXFILES/` (§1.1) and reuse the existing `/vsis3` reader and window logic
  verbatim (§6). The **HITL signup-checklist ticket the ticket anticipated is not needed.**
- **Mask-prevalence measurement (#19):** vocabularies are pinned — bin on
  the **explicit value lists** (EDM 0–13; **FLM 0–9 *plus* 100–102**; WBM 0–3), never on a
  contiguous range. Report fraction-per-code. `nodata=None`, so **void is code 0, in-band**.
  Reuse the DEM's window logic unchanged (§6). Assert the three §6.1 identities as a tripwire.
  ⚠️ **Measure at full resolution, not ×8** (§7) — the masks are small enough that the
  overview budget is unnecessary, and ×8 zeroes out real rare classes.
- **Corpus recipe (#4):** two concrete levers land here. (1) **`WBM == 1` replaces the
  zero-fraction ocean heuristic** from recon §2 — spec-labelled, and immune to the
  Netherlands −12 m false-positive. (2) **`FLM ≥ 3` is a principled "this pixel is not
  TanDEM-X" filter** — a candidate quality gate for excluding windows contaminated by legacy
  ASTER/SRTM texture, which would otherwise teach the MAE another sensor's artifacts.
  ⚠️ Note the Alps tile is **16.45% externally filled** — this is *not* a rare edge case, and
  a strict `FLM ≥ 3` exclusion could bite hard in exactly the rugged terrain we most want.
  Prevalence numbers should set the threshold, not intuition.
- **Model card / repo licensing:** committing derived statistics is **permitted** (§5). Ship
  the **Art. 6(b)** notice + the **Art. 6(c)** liability sentence wherever the table or the
  weights go. Pre-decide the Art. 6(a) question **before** caching raw tiles into any
  published artifact.
- **Fog:** (1) whether the AWS COG instance is formally the DGED instance is **inferred, not
  spec-stated** (§1) — harmless for the prevalence table, but do not cite it as spec. (2) The
  Art. 2 "does the EULA name the masks" caveat (§5) — safe either way, but unresolved in the
  primary text. (3) The **void (code 0)** path is spec-known but **empirically unobserved** on
  every tile probed here, so the void branch of any binning code is **untested by real data** —
  #19's full-res pass over ~500 tiles is the natural place to find out whether voids exist at
  all in GLO-30. (4) Mask presence is **sampled (216/216 across all 18 latitude bands), not
  exhaustively proven** over all 26,450 prefixes — `masks.py` should still fail loudly on a
  missing mask rather than assume the invariant. (5) The bucket lists **26,450** tile prefixes
  where the curation plan says **26,454** COGs; a 4-tile discrepancy nobody has explained yet.

## Sources

- **Copernicus DEM Product Handbook i5.0** (GEO1988-CopernicusDEM-SPE-002, v5.0, 29.11.2022)
  — **owns Tables 5/6/7/8**: quality-layer list & encoding p.17; EDM §1.2.5.1 + Table 6 p.18;
  FLM §1.2.5.2 + Table 7 & fill-source footnotes p.19; WBM §1.2.5.4 + Table 8 p.21; DGED
  product structure Figure 11; grid-spacing/majority-resampling statement §1.2.5.
  https://dataspace.copernicus.eu/sites/default/files/media/files/2024-06/geo1988-copernicusdem-spe-002_producthandbook_i5.0.pdf
- **Licence, COP-DEM-GLO-30-F ("Full, Free & Open") — `eula_F.pdf` as shipped in the bucket**
  (Articles 3/4/5/6/9; the authoritative licence for the data we actually read):
  https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N45_00_E007_00_DEM/INFO/eula_F.pdf
- **ESA Copernicus Data Space — COP-DEM collection description** (attribution notices, licence
  PDFs, citation DOI `10.5270/ESA-c5d3d65`):
  https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM
- **ESA User Licence for Copernicus Contributing Mission (CCM) / COP-DEM data**:
  https://dataspace.copernicus.eu/sites/default/files/media/files/2025-06/copernicus_contributing_mission_data_access_v2_cop_dem_licenses.pdf
- **AWS Open Data registry — Copernicus DEM** (licence pointer + required access citation):
  https://registry.opendata.aws/copernicus-dem/
- **Bucket README** (GLO-30 Public licence pointer, sampling table):
  https://copernicus-dem-30m.s3.amazonaws.com/readme.html
- **Empirical probes** (co-registration, dtype/nodata, observed code fractions, AUXFILES
  listing): `s3://copernicus-dem-30m`, tiles N45E007 & N65W019, via rasterio 1.5.0 /
  GDAL 3.12.1 anonymous, July 2026.
- **Empirical probes, access & overview behaviour** (§1.1, §7): anonymous `aws s3api`
  listing of all 26,450 tile prefixes; **latitude-stratified mask-presence sample, 216 tiles
  across 18 bands**, plus a uniform 30-tile size sample; anonymous HTTPS range read
  (`HTTP 206`); ×8-overview vs full-res per-code comparison on N45E007 & N60E010; oceanic
  water-code confirmation on N50W002, N38W010, N33W119. rasterio 1.5.0 / GDAL 3.12.1,
  July 2026.
- **Sibling recon doc** (bucket layout, `nodata=None` on the DEM, zero≠ocean, latitude bands):
  `docs/research/copernicus-glo30-recon.md`
