# P125 — real SRTM void geometry: what it costs, and what the fallback is actually made of

Ticket [#125](https://github.com/JStoschek/DEMAE/issues/125) · map [#117](https://github.com/JStoschek/DEMAE/issues/117) · plan [`task1-void-fill-plan.md`](task1-void-fill-plan.md) §3.2, §3.3 · selection memo [`downstream-task-selection.md`](downstream-task-selection.md) §2.2

> **What this document is.** A scoping study for the fourth eval family. No bulk data was
> downloaded. Everything numeric below §4 was **measured on real SRTM tiles** — 17 of them,
> 58 MB total, all pulled from no-login sources — not estimated from published summaries.
> Two kinds of statement are kept apart: **read from a specification** (a claim with a URL to
> the document that owns it) and **measured** (a number produced by running code over an actual
> raster, with the tile named). Where the plan's text is contradicted, the contradiction is
> stated plainly and the source is given.

## 1. Bottom line

**The data is cheap, the login is unavoidable, and the family cannot be built the way §3.2
specifies.** In order:

1. **SRTMGL1N exists, is current, and is exactly the right product.** 1 arc-second, 1°×1°
   granules, a 1-byte class raster per tile. The three named regions are **975 granules,
   388 MB compressed** (§5.1). A sufficient, geographically-spread subset is **24 granules,
   28.4 MB** (§9). This is not a data-volume problem at any scale.
2. **Earthdata Login is genuinely required, and there is no no-login route to it.** Confirmed by
   HTTP probe, not by reading the marketing copy: the distribution URL 302s to a URS OAuth
   authorize endpoint (§4.1). Every alternative was probed and every one failed (§4.3). A human
   must register and click through an application authorization once. Cost: a free web form,
   no eligibility gate, no fee (§4.2).
3. **The archive.org v2.1 mirror is not a fallback.** It is 3 arc-second data — one 13.7 GB
   tarball, all-or-nothing (§6.1) — and SRTM v2.1 at 1 arc-second was **United States only**,
   so it has no Himalaya, Andes or Atlas coverage at the resolution the plan wants (§6.2). The
   plan's "the same geometry" is wrong, and §7 measures by how much: a v2.1 3-arc-second void
   mask upsampled to 30 m recovers **IoU 0.60** against the true 1 arc-second mask, missing
   **39% of void area** and **~39% of usable components**. The reason is structural — v2.1's
   3 arc-second voids are an *erosion* of the 1 arc-second voids (§7.1), which is measured, not
   assumed.
4. **There is a better no-login fallback than the one the plan names**, and it is what this
   study ran on: the Wayback Machine holds the retired USGS `dds.cr.usgs.gov` tree
   **per tile**, ~2 MB each (§6.3). It is still 3 arc-second, so it inherits every defect in (3)
   — but it makes the fallback *testable* rather than a 13.7 GB gamble, and it is how §7's
   measurement was possible at all.
5. **The plan's void predicate is wrong.** §3.3 says "pixels with scene count 0 are true voids."
   There is no 0 in SRTMGL1N's class table; SRTM-observed pixels are **201–224**, and the value
   0 belongs to a *different product's* `.num` file (§4.4). Left uncorrected this yields an
   empty mask.
6. **NASADEM_SHHP is a cleaner product measuring a different thing.** NASADEM's reprocessing
   **filled 52–63% of SRTM's voids** (§8.1, NASADEM's own table). Its voids are therefore
   roughly half the phenomenon, and it costs **~100× the bytes** (§8.2). It is the right source
   only if the benchmark's claim changes from "real SRTM voids" to "voids that survive modern
   InSAR reprocessing."
7. **The resampling the plan worries about does not exist on the recommended path.** SRTM
   1 arc-second and Copernicus GLO-30 below 50° latitude are the *same lattice* (§10). The mask
   is cropped, never resampled. The resampling risk is entirely a property of the 3 arc-second
   fallback — which is the fallback's whole problem, not a side note.
8. **The binding constraint is not access, bytes, or compute — it is the size distribution.**
   Real voids are ~94% small. Measured at true 1 arc-second, two high-relief cells yielded
   **139 and 175** voids in the 16–48 px bucket, **2 and 6** in 64–96, and **zero** in 128–160
   (§11.2). §3.2 requires every tile to appear "once per bucket per family." **The large bucket
   starves, and no amount of downloading fixes it.**

**Recommendation (§12): build the family, at ~28 MB and one human login, but redefine it as a
small-void family at its natural size distribution and drop it from the bucket-balanced table.**
The "not worth it" verdict is not warranted — the marginal cost is genuinely near zero — but the
"~2,000 instances, bucketed like the synthetic families" shape in §3.3 is not achievable and
should be amended before a `wayfinder:task` is cut against it.

## 2. What was asked

§3.3 proposes the fourth eval family: real SRTM void shapes, transplanted onto clean GLO-30
tiles of matching ruggedness. It is the only part of the core scope with an external dependency
and a possible login, so #125 asks whether a human has to be in the loop, and #117's fog patch
asks whether the family earns its place at all.

Two claims in the plan were flagged for verification rather than inheritance, because each one
decides something: that SRTMGL1N needs only a "free login", and that the archive.org v2.1 mirror
gives "the same geometry" without one. Both are answered below, and one of them is wrong.

## 3. Method, and what was and was not fetched

Primary sources only: LP DAAC / NASA Earthdata product pages, NASA's CMR metadata API, the
USGS/NASA product specifications, and real HTTP responses from the distribution endpoints.

**Nothing was downloaded that required an account or accepting terms.** Where a fact could be
established by an unauthenticated HTTP probe (does this URL 302 to a login?), it was probed
rather than read about. Where a fact required looking at actual pixels, tiles were taken from
sources that need no login:

| Fetched | Source | Size | Login? |
|---|---|---|---|
| 13 × SRTM v2.1 SRTM3 tiles (Himalaya ×6, Andes ×4, Atlas ×3) | Wayback capture of `dds.cr.usgs.gov` | 25 MB | no |
| 2 × SRTM v2.1 SRTM1 tiles (`N46W122`, `N48W121`) | Wayback capture of `dds.cr.usgs.gov` | 29 MB | no |
| 2 × SRTM v2.1 SRTM3 siblings of those two | Wayback capture of `dds.cr.usgs.gov` | 4 MB | no |
| `SRTM_User_Guide_V3.pdf`, `SRTM_Topo.pdf`, `NASADEM_User_Guide_V1.pdf` | LP DAAC / Wayback | 6 MB | no |
| Granule inventories for 3 regions | CMR search API, anonymous | — | no |

The two SRTM1 tiles are the load-bearing ones: **the only place on Earth where SRTM v2.1 exists
at both 1 and 3 arc-seconds is the United States**, which makes them the only way to measure the
resolution penalty directly instead of arguing about it. That is §7.

## 4. SRTMGL1N: the product, and the login

### 4.1 The product still exists, and the login is real

`lpdaac.usgs.gov/products/srtmgl1nv003/` 301s to
`https://www.earthdata.nasa.gov/data/catalog/lpcloud-srtmgl1n-003`. The product is live:

> **NASA Shuttle Radar Topography Mission Global 1 arc second number**, short name `SRTMGL1N`,
> version `003`, resolution 30 m × 30 m, granule size "210.1 KB to 3,003.5 MB",
> DOI `10.5067/MEaSUREs/SRTM/SRTMGL1N.003`.

CMR gives the granule layout precisely (`C2763266368-LPCLOUD`). One granule per 1°×1° cell,
`GranuleUR` of the form `N28E086.SRTMGL1N.num`, distributed as
`…/SRTMGL1N.003/N28E086.SRTMGL1N.num/N28E086.SRTMGL1N.num.zip`. Uncompressed the `.num` raster is
3601 × 3601 × 1 byte = **12,967,201 bytes**; the zips compress to 0.2–2.8 MB depending on how
much structure the tile has.

*The old on-premise pool is gone.* `https://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1N.003/…`
returns **404** — the collection moved to Earthdata Cloud. Anything in the plan or a downstream
task that assumes an `e4ftl01` path is stale.

**The login is confirmed by response, not by inference.** Unauthenticated `GET` of a real
granule URL:

```
GET https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1N.003/
      N28E088.SRTMGL1N.num/N28E088.SRTMGL1N.num.zip

HTTP/2 302
location: https://urs.earthdata.nasa.gov/oauth/authorize?client_id=FtSFfbOeuxDcdf4px-elGw
          &response_type=code&redirect_uri=https://data.lpdaac.earthdatacloud.nasa.gov/login
          &state=%2Flp-prod-protected%2F…&app_type=401
```

The bucket is literally named `lp-prod-protected`. `app_type=401` is the DAAC telling URS this is
an authentication challenge. **The plan's "free login" is correct as far as it goes** — but it is
an OAuth authorization against a named application, not a bare password prompt, and the first
authorization is an explicit consent click. That makes it a human step, not an agent step.

*Note the useful asymmetry:* the **CMR metadata API is anonymous**. Granule names, sizes,
bounding boxes and download URLs can all be enumerated by an agent; only the bytes are gated.
The entire checklist in §13 was produced without credentials.

### 4.2 What the login costs a human

`https://urs.earthdata.nasa.gov/users/new` asks for: username, password, first/last name,
middle initial, email, country, affiliation, study area, user type, organization, and an
optional email-notification opt-in. **No fee, no eligibility statement, no per-collection EULA
on the registration form.** The page carries only the standard
`US Govt Property. Unauthorized use subject to prosecution` notice.

The one-time application authorization (the OAuth consent screen above) is a separate click and
is the reason this cannot be delegated: accepting terms on someone's behalf is out of scope for
an agent.

### 4.3 There is no no-login route. Each candidate was probed.

| Candidate | Result | Verdict |
|---|---|---|
| `data.lpdaac.earthdatacloud.nasa.gov` (official) | 302 → URS OAuth | login |
| `e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1N.003/` | **404** | retired |
| `dds.cr.usgs.gov/srtm/` (historic anonymous USGS tree) | **404** | dead (see §6.3) |
| OpenTopography `portal.opentopography.org/API/globaldem` | **401** | API key ⇒ account |
| ESA STEP `step.esa.int/auxdata/dem/SRTMGL1/` | **200**, 15 MB, for `…SRTMGL1.hgt.zip` | serves the **elevation** only; `…SRTMGL1N.num.zip` → **404** |
| `datenatlas.de/geodata/public/srtmgl1/` | **404** | gone |
| Earthdata Search / Harmony / `s3://lp-prod-protected` | same URS gate | login |

The ESA STEP result is the near-miss worth naming: a genuinely open, no-login mirror of SRTM v3
at 1 arc-second — **but of `SRTMGL1`, the void-*filled* DEM.** The elevations there have had the
voids removed; the mask is exactly the thing that was thrown away. It cannot substitute.

### 4.4 The plan's void predicate is wrong, and the fix is exact

§3.3 says "pixels with scene count 0 are true voids." The SRTM Collection User Guide's Table 2
("Fill Values of the '.NUM' files") has **no 0**:

| Value | Meaning |
|---|---|
| 1 | Water-masked SRTM void |
| 2 | Water-masked SRTM non-void |
| 5 | GDEM elevation = 0 in SRTM void |
| 11 | NGA-interpolated SRTM (very small voids in SRTMv1) |
| 21 | GMTED2010 oversampled from 7.5 arc-second postings |
| 25 | SRTM within GDEM |
| 31 | NGA fill of SRTM via GDEM |
| 51 / 52 / 53 | USGS NED / NED via GDEM / Alaska NED via GDEM |
| 72 | Canadian CDED via GDEM |
| 101–200 | ASTER scene count (count limited to 100) |
| **201–224** | **SRTM swath count (non-voided swaths), actual maximum = 24** |

Source: `https://lpdaac.usgs.gov/documents/179/SRTM_User_Guide_V3.pdf`, Table 2, p. 6. The same
page states the encoding rule: *"NUMs less than 6 are water and those greater than 10 are land."*

So the correct predicates are:

```
srtm_observed  ⟺  201 ≤ num ≤ 224
water          ⟺  num ∈ {1, 2, 5}
void (on land) ⟺  num > 10  AND  num < 201
```

**Where the plan's "0" came from.** The same User Guide, §2.2.2, describing a *different*
product — `SRTMIMGM`, the Combined Image Data — says its `.num` file gives *"the number of
pixels averaged to create the output pixel. This value is typically 0 (for a void pixel) to 10
or more."* That is the file where 0 means void. It is not SRTMGL1N. Two products, two `.num`
files, two encodings; the plan merged them.

*One judgement call to record rather than bury:* value **11** ("were very small voids in
SRTMv1") and value **25** ("SRTM within GDEM") are pixels whose relationship to a v2.1 radar
void is ambiguous — 11 was a v1 void that NGA interpolated before v2, and 25 is GDEM fill that
was itself SRTM-derived. Both are rare and small. The clean, defensible mask is the complement
of `201–224` on land; if a downstream task wants strict "never observed by radar", excluding 11
and 25 is a one-line change and should be a documented flag, not a silent choice.

## 5. How many granules, how many GB

### 5.1 The three regions, from CMR

Anonymous CMR queries against `C2763266368-LPCLOUD`, granule sizes summed from
`DataGranule.ArchiveAndDistributionInformation`:

| Region | Bounding box (W,S,E,N) | Granules | Compressed |
|---|---|---|---|
| Himalaya | 72, 26, 98, 37 | 364 | 216.2 MB |
| Andes | −79, −40, −64, −8 | 371 | 102.6 MB |
| Atlas | −11, 28, 12, 37 | 240 | 69.5 MB |
| **Union (deduplicated)** | | **975** | **388.2 MB** |

Uncompressed, all 975 `.num` rasters are 12.6 GB — but nothing needs to hold them all at once;
the extraction is one tile in, a component list out (§10.2).

**388 MB is the answer to "how many GB".** It is small enough that the question is not really
about volume. The cost of this family is one login and one afternoon, not bandwidth.

### 5.2 Yield per cell — and the Atlas problem

Measured on the 13 v2.1 3-arc-second tiles (§3), counting 8-connected void components and
converting each component's equivalent diameter to 30 m-pixel units. `frac` is the tile's void
fraction; `in band` is components with 16 ≤ d < 160 px at 30 m:

| Tile | Region | void frac | components | in band |
|---|---|---|---|---|
| N29E095 | Himalaya (Namcha Barwa) | 22.52% | 3,758 | 689 |
| N30E080 | Himalaya (Kumaon) | 19.58% | 2,276 | 390 |
| N35E075 | Karakoram | 18.74% | 2,965 | 519 |
| N28E085 | Himalaya (Langtang) | 13.76% | 1,959 | 355 |
| N27E088 | Himalaya (Kangchenjunga) | 11.62% | 2,352 | 415 |
| N28E086 | Himalaya (Everest) | 3.70% | 988 | 174 |
| S10W077 | Andes (Cordillera Blanca) | 1.98% | 2,047 | 214 |
| S33W070 | Andes (Aconcagua) | 1.36% | 1,024 | 134 |
| S16W068 | Andes (Cordillera Real) | 0.40% | 855 | 43 |
| **N31W008** | **Atlas (High Atlas)** | **0.16%** | 263 | **24** |
| **N31W006** | **Atlas** | **0.14%** | 228 | **23** |
| S24W068 | Andes (Puna, arid) | 0.03% | 27 | 5 |
| **N33W005** | **Atlas (Middle Atlas)** | **0.03%** | 52 | **4** |

Because these are 3 arc-second counts they **understate** the 1 arc-second yield by a factor
measured in §7.2 as ~1.6×.

**The Atlas is not a void-rich region.** The plan names it in the same breath as the Himalaya
and the Andes; it is two orders of magnitude poorer — 0.03–0.16% void against the Himalaya's
3.7–22.5%, and 4–24 usable components per cell against 174–689. Its 240 granules would
contribute a few hundred small voids at best. It should be dropped or demoted to a diversity
garnish, and the plan's phrasing corrected.

## 6. The archive.org v2.1 mirror: what it actually holds

### 6.1 One 13.7 GB tarball, 3 arc-second, all-or-nothing

`archive.org/details/srtm3v2.1`, metadata read from `archive.org/metadata/srtm3v2.1`:

> **Title:** "USGS SRTM-3 Version 2.1Elevation Models"
> **Description:** *"This is a 14.7 GB GZipped TAR file of the world-wide SRTM-3 v2.1
> topographical data which was originally available at:
> http://dds.cr.usgs.gov/srtm/version2_1/SRTM3/"* — listing Africa, Australia, Eurasia,
> Islands, North_America, South_America. MD5 `444b45eaebd40edfde1b73679f074530`.

The item's entire payload is **a single file**: `srtm3v2.1.tgz`, **14,684,236,758 bytes**. There
are no per-tile files. It is gzip, so it is not range-seekable — extracting one Himalaya cell
means transferring 13.7 GB.

**"SRTM-3" is 3 arc-second.** The name says so, the source path says so, and the fetched sample
confirms it: a tile from that tree unzips to exactly **2,884,802 bytes = 1201 × 1201 × 2**.
The SRTM Collection User Guide's Table 1 is explicit — `SRTMGL3`, 1201/1201 rows/columns,
"3 arc-second"; `SRTMGL1`, 3601/3601, "1 arc-second".

### 6.2 v2.1 at 1 arc-second was United States only

This is the sentence that ends the fallback argument. From the v2.1 specification,
`SRTM_Topo.pdf` §2.2, retrieved from the Wayback capture of
`dds.cr.usgs.gov/srtm/version2_1/Documentation/SRTM_Topo.pdf`:

> *"Edited SRTM1 data for the United States and its territories and possessions are also being
> released and can be found in the directory /United_States_1arcsec./ Cells that straddle the
> border with neighboring countries have been masked with quarter degree quantization such that
> data outside the U.S. have the void value."*

The same document's availability table lists LP DAAC's Version 2 holdings as *1" U.S., 3" world
— averaged, 30" world — averaged*. Global 1 arc-second SRTM did not exist publicly until
Version 3 (2014), which is void-**filled** — which is why SRTMGL1N, the class raster, is the
only way to recover the mask at 1 arc-second.

**Therefore: there is no 1 arc-second SRTM v2.1 for the Himalaya, the Andes, or the Atlas, on
archive.org or anywhere else.** The plan's fallback is not a lower-convenience path to the same
data. It is different data at a third of the resolution.

### 6.3 The better no-login fallback the plan missed

`dds.cr.usgs.gov` is dead (`/srtm/` → 404), but **the Wayback Machine captured the tree at file
granularity.** A CDX query returns individual tile zips with `statuscode:200`:

```
gov,usgs,cr,dds)/srtm/version2_1/srtm3/eurasia/n28e086.hgt.zip 20170105071905
  https://dds.cr.usgs.gov/srtm/version2_1/SRTM3/Eurasia/N28E086.hgt.zip application/zip 200 … 1974536
```

Fetch pattern: `https://web.archive.org/web/<timestamp>id_/<original-url>`. **5,876 distinct
Eurasia tile URLs** are archived, plus North_America, South_America, Africa, Australia, Islands
— and, critically, the US-only `SRTM1/Region_0*` 1 arc-second tiles (§7).

This is strictly better than the archive.org tarball on every axis: per-tile (~2 MB vs 13.7 GB),
no login, same bytes, same provenance. **It is still 3 arc-second, so it does not rescue the
fallback** — but it makes the fallback measurable, which is how §7 exists. If the plan keeps a
no-login fallback at all, this is the one to name.

## 7. Measuring the resolution mismatch, not arguing about it

The US is the only place SRTM v2.1 published both resolutions, so `N46W122` (Mt Rainier,
Cascades) and `N48W121` (North Cascades) were fetched at 1 arc-second (3601²) and 3 arc-second
(1201²) and compared directly.

### 7.1 v2.1's 3 arc-second voids are an *erosion* of the 1 arc-second voids

The v2.1 spec states the derivation but not the void rule:

> *"Unlike DTED, however, 3 arc-second data are generated in each case by 3x3 averaging of the 1
> arc-second data - thus 9 samples are combined in each 3 arc-second data point."*
> — `SRTM_Topo.pdf` §3.0

What happens when some of the nine are void (`-32768`) is unstated. It was recovered by
exhaustive test — predicting the actual 3 arc-second mask from the 1 arc-second mask under four
candidate rules, over the 1200×1200 interior of both cells:

| Rule | `N46W122` predicted / actual voids | pixel agreement | `N48W121` predicted / actual | agreement |
|---|---|---|---|---|
| any ≥ 1 of 9 | 20,048 / 9,023 | 99.23% | 30,018 / 13,729 | 98.87% |
| ≥ 2 of 9 | 18,380 / 9,023 | 99.35% | 27,573 / 13,729 | 99.04% |
| majority ≥ 5 of 9 | 13,973 / 9,023 | 99.65% | 21,132 / 13,729 | 99.48% |
| **all 9 of 9** | **9,020 / 9,023** | **99.79%** | **13,660 / 13,729** | **99.70%** |

**A 3 arc-second post is void only if all nine of its 1 arc-second parents are void.** The count
match is exact to within 0.03% and 0.5%. Every other rule overshoots by 50–120%.

This is the mechanism, and it is the worst possible one for this purpose: it systematically
shaves the ragged edge off every void and deletes every void narrower than three 1-arcsecond
posts — precisely the fine radar-shadow filigree that made real voids interesting.

### 7.2 What that costs, in the units the plan cares about

Taking the 3 arc-second mask, upsampling ×3 to the 30 m grid, and scoring it against the true
1 arc-second mask for the same cell:

| | `N46W122` | `N48W121` |
|---|---|---|
| void fraction, 1 arc-second | 0.985% | 1.482% |
| void fraction, 3 arc-second | 0.626% | 0.952% |
| ratio | **0.64** | **0.64** |
| components, 1 as / 3 as | 691 / 628 | 894 / 847 |
| components in 16–160 px band, 1 as / 3 as | 146 / 93 (**64%**) | 191 / 117 (**61%**) |
| **IoU of upsampled 3 as vs true 1 as** | **0.593** | **0.601** |
| recall (of true void area) | 0.609 | 0.617 |
| precision | 0.958 | 0.959 |
| median perimeter / (π · eqdiam), 1 as / 3 as | 2.02 / 1.89 | 2.23 / 2.04 |

**Read the recall and precision together.** Precision 0.96 means the 3 arc-second mask does not
invent voids — what it marks really is void. Recall 0.61 means it **misses two fifths of the
real void area**. Combined with the ×3 boundary quantisation, IoU lands at **0.60**.

An IoU of 0.60 between two masks is not "the same geometry" by any reading. For comparison, that
is the sort of agreement one would reject as a segmentation match. The plan's sentence — *"SRTM
v2.1 (archive.org mirror, no login) is the fallback source for the same geometry"* — is
**false**, and now quantitatively so.

The perimeter ratios add the qualitative half: real 1 arc-second voids are strikingly
non-compact (perimeter ≈ 2.0–2.2× that of an equal-area disc), and the 3 arc-second versions are
measurably smoother (1.89–2.04). The fallback does not just lose voids, it **rounds off the
thing that distinguishes a real void from a random blob** — which is the entire argument for the
family's existence.

*One honest caveat.* Both cells are Cascades terrain at ~1–1.5% void, an order of magnitude less
void-rich than the Himalaya cells in §5.2. The IoU ratio is a property of the 3×3 all-9 erosion
rule and of void boundary roughness, both of which are scale-free enough to travel; the absolute
void fractions are not, and no claim is made that they do.

### 7.3 Consequence for §5.2's numbers

The 3 arc-second in-band counts in §5.2 are **61–64% of the 1 arc-second truth**. Dividing
through by 0.62, a void-rich Himalaya cell yields roughly **280–1,110 usable components at
1 arc-second**, mean ≈ 680. That is the number §11 uses.

## 8. NASADEM_SHHP: cleaner, more expensive, and a different phenomenon

### 8.1 NASADEM filled half of SRTM's voids — so its voids are not SRTM's voids

The selection memo names `NASADEM_SHHP` as distinguishing SRTM-observed from interpolated
pixels. It does — but NASADEM reprocessed the raw radar, and the reprocessing worked:

> *"Overall, more than 50% of the void areas in the original strip data results were successfully
> unwrapped using SNAPHU and the multi-burst approach."*
> — `NASADEM_User_Guide_V1.pdf` §4.1.2

Table 3 of the same document, "% SRTM Voids Now Filled … + Using Multi-Burst Approach", by
continent: **North America 51.87, South America 53.47, Eurasia 52.28, Africa 62.94, Australia
57.48.**

**Between 52% and 63% of SRTM's voids are simply not voids in NASADEM.** A mask built from
NASADEM_SHHP is a mask of *the voids that survived a better phase unwrapper* — a smaller, more
severe, differently-shaped population. That is a legitimate and arguably more modern target, but
it is not what §3.3 says the family is, and the two must not be swapped silently. If the
benchmark's sentence is "real SRTM void geometry", SRTMGL1N is correct. If it becomes "voids
that defeat state-of-the-art InSAR", SHHP is correct and the writeup must say so.

### 8.2 The price, from CMR

| Product | Per granule | 975 granules | Contents |
|---|---|---|---|
| `SRTMGL1N` | 0.2–2.8 MB (~0.4 MB avg) | **388 MB** | 1-byte class raster |
| `NASADEM_HGT` | ~5.5 MB | ~5.4 GB | `hgt` + `num` + `swb` |
| `NASADEM_NUMNC` | 24.8 MB | ~24 GB | source layer, netCDF |
| `NASADEM_SHHP` | ~60–64 MB | **~58 GB** | 4-byte float SRTM-only DEM + 2-byte error |

SHHP is ~150× SRTMGL1N's bytes because it ships a float32 DEM and a precision layer when the
task needs one bit per pixel. `NASADEM_HGT`'s bundled `num` is the efficient way to get a
NASADEM-source mask if that is ever wanted — but note its class table is a **different
encoding** (`NASADEM_User_Guide_V1.pdf` Table 2: `0` = water, **`1–23` = SRTM data takes**,
`41–94` = PRISM, `110–160` = GDEM3, `170–220` = GDEM2, `250` = interpolation). Neither NASADEM's
`1–23` nor SRTMGL1N's `201–224` is the plan's "0". Any code must be told which product it is
reading.

**Verdict: SRTMGL1N, not SHHP.** Cheaper by two orders of magnitude, and it answers the question
the plan actually asked.

## 9. The recommended draw

24 granules, spanning three massifs in the Himalaya/Karakoram and three in the Andes, chosen
around the cells whose void richness was measured in §5.2. Sizes are from CMR:

| Cell | MB | | Cell | MB | | Cell | MB |
|---|---|---|---|---|---|---|---|
| N28E086 | 0.803 | | N30E080 | 1.525 | | N29E095 | 1.632 |
| N28E085 | 1.406 | | N29E080 | 1.386 | | N29E094 | 1.140 |
| N29E085 | 0.405 | | N30E081 | 0.865 | | N28E095 | 1.080 |
| N28E087 | 0.785 | | N35E075 | 2.187 | | N30E095 | 1.207 |
| N27E088 | 1.530 | | N35E076 | 2.833 | | S10W077 | 0.866 |
| N28E088 | 0.320 | | N36E074 | 2.473 | | S09W077 | 0.635 |
| S13W072 | 0.452 | | N36E075 | 2.044 | | S16W068 | 0.452 |
| S17W068 | 1.135 | | S33W070 | 0.841 | | S32W070 | 0.432 |

**Total: 28.4 MB.** Expected yield ≈ 17 Himalaya/Karakoram cells × ~680 + 7 Andes cells × ~210
≈ **13,000 in-band components** — far more than the 2,000 §3.3 asks for, which is the point: the
draw can then be *subsampled* for size and geographic balance instead of scraped for volume.

If more is ever wanted, the full three-region sweep is 975 granules / 388 MB and needs no new
decision — just a wider CMR bounding box (§13.3).

## 10. The extraction path

### 10.1 There is no resampling step

§3.3 says "rasterize to 30 m", and #125 asks whether resampling changes the geometry enough to
matter. **On the SRTMGL1N path it is a non-step.**

- SRTM 1 arc-second tiles are 3601 × 3601 with *sample centres on whole arc-second multiples*;
  the file name coordinates "refer to the geometric center of the lower-left pixel"
  (`SRTM_User_Guide_V3.pdf` §2.1.4).
- Copernicus GLO-30 COGs below 50° latitude are 3600 × 3600 with affine
  `(0.00027778, 0, 6.99986111, 0, −0.00027778, 46.00013889)` — verified on `N45E007` in
  [`copernicus-mask-spec.md`](copernicus-mask-spec.md) §6. `0.00027778 = 1/3600°` is exactly one
  arc-second, and the origin `7 − 0.5/3600` places pixel *centres* on whole arc-second multiples.

**Same lattice.** The mask is cropped and pasted, never interpolated. The 3601-vs-3600 difference
is SRTM's one-row/one-column overlap with its neighbours, which is a bookkeeping detail, not a
regridding one.

The one boundary condition: GLO-30 widens its longitude spacing poleward (1800 × 3600 at 65°N,
same source). All three source regions and all sensible transplant targets sit below 50°, so it
does not bite — but a transplant onto a high-latitude test tile would, and the benchmark code
should assert `|lat| < 50` rather than discover it.

**The resampling risk #125 asks about is real, large, and belongs entirely to the 3 arc-second
fallback** (§7.2: IoU 0.60). On the recommended path it is zero.

### 10.2 Compute cost, measured

Timed on a real 3601 × 3601 raster (5 runs, single-threaded, this laptop): predicate +
8-connected labelling + component areas + bounding boxes = **120 ms per tile**.

- 24 granules → **3 seconds**.
- All 975 → **117 seconds**.

Peak memory is one tile: 13 MB for the `uint8` raster, ~52 MB for the `int32` label array.
Nothing here needs a rented box, a shard format, or a plan.

The full path: unzip → apply §4.4's predicate → `ndimage.label` → equivalent diameter
`2·√(area/π)` → keep 16 ≤ d < 160 → store each component as an offset RLE. Output is a few
thousand small masks — kilobytes — which is what §3.5's `mask_rle` parquet already expects.

### 10.3 Attrition against §3.2's constraints, measured

§3.2 requires voids fully interior (≥ 8 px from a 224 px tile edge, so a 208 × 208 box) and
≤ 40% of tile area. Real voids are elongated, so this was checked rather than assumed, on the
two true-1-arcsecond cells:

| | `N46W122` | `N48W121` |
|---|---|---|
| in-band components | 146 | 191 |
| bounding box fits 208 × 208 | 146 (100%) | 190 (99%) |
| area ≤ 40% of a 224² tile | 146 (100%) | 191 (100%) |
| median max(bbox) / eqdiam | 1.72 | 1.64 |
| p90 max(bbox) / eqdiam | 2.24 | 2.39 |

**Attrition is negligible.** Real voids are elongated — a median component's longest bounding-box
side is 1.7× its equivalent diameter — but because the in-band population sits near the small end
of the band, essentially all of them fit. This was the one place a problem was expected and did
not appear; it is reported as a null result rather than dressed up.

## 11. The real constraint: the size distribution

### 11.1 Per-bucket yield

Component counts in §3.2's three buckets. Himalaya/Andes/Atlas rows are 3 arc-second measurements
scaled to 1 arc-second by §7.2's 0.62 factor; the last two rows are **direct 1 arc-second
measurements** and are the anchor:

| Source | small 16–48 | medium 64–96 | large 128–160 |
|---|---|---|---|
| Himalaya, per cell (6 cells, scaled) | ~571 | ~38 | **~9** |
| Andes, per cell (4 cells, scaled) | ~148 | ~4 | **~1** |
| Atlas, per cell (3 cells, scaled) | ~27 | 0 | **0** |
| `N46W122`, **measured at 1 arc-second** | 139 | 2 | **0** |
| `N48W121`, **measured at 1 arc-second** | 175 | 6 | **0** |

The distribution is a steep power law. Across the six Himalaya cells the raw 3 arc-second totals
were **2,125 small : 142 medium : 33 large** — roughly **64 : 4 : 1**.

### 11.2 Why that breaks §3.2

§3.2 states: *"every eval tile appears once per bucket per family, so bucket scores are computed
over identical tile sets."* With ~2,000 instances that is ~667 per bucket.

- **667 small voids:** trivial. ~2 void-rich Himalaya cells.
- **667 medium voids:** ~18 void-rich Himalaya cells. Achievable but consumes most of the good
  cells.
- **667 large voids:** ~74 void-rich Himalaya cells, at ~9 per cell. The *entire* Himalaya
  bounding box holds 364 granules and only a fraction are as void-rich as the six sampled here —
  those were chosen for known relief, not drawn at random. Adding the Andes contributes ~1 per
  cell and the Atlas contributes zero.

**The large bucket cannot be filled from real SRTM voids in these regions, and downloading more
does not fix it — the voids do not exist.** The two directly-measured 1 arc-second cells returned
*zero* large components between them, out of 1,585 total components.

This is the finding #117's fog patch was looking for, and it is not about access or cost. Real
radar-shadow voids in mountains are mostly narrow, elongated, and small; the few enormous ones
blow straight past the 160 px ceiling (`N30E080`'s largest is ~1,289 px equivalent diameter, and
every void-rich cell has several over 400). The 128–160 px window sits in a genuine trough of the
natural distribution — big enough to be rare, small enough that the giants overshoot it.

## 12. Recommendation

**Build the family. Change its shape. Cut the Atlas.**

1. **Take the SRTMGL1N path.** One human, one Earthdata Login, 24 granules, 28.4 MB, ~3 seconds
   of compute. Every alternative is worse: the archive.org fallback is 13.7 GB of wrong-resolution
   data, and NASADEM_SHHP is 58 GB of a different phenomenon.
2. **Amend §3.3's void predicate** to `num > 10 AND num < 201` before any task is cut against it.
   As written it produces an empty mask.
3. **Amend §3.3's fallback sentence.** "SRTM v2.1 (archive.org mirror, no login) is the fallback
   source for the same geometry" is false on two counts: it is 3 arc-second only (v2.1's
   1 arc-second release was US-only), and it recovers IoU 0.60 against the truth. If a no-login
   fallback is retained for resilience, name the Wayback per-tile route (§6.3), not the 13.7 GB
   tarball, and label it *degraded*, not equivalent.
4. **Drop the Atlas** from the void-rich region list, or demote it explicitly. At 0.03–0.16%
   void it does not belong in the same sentence as the Himalaya.
5. **Report the real-void family at its natural size distribution, and take it out of the
   bucket-balanced table.** Concretely: ~2,000 instances drawn as measured (≈ 94% small, 5%
   medium, 1% large), reported as its own row, with the size histogram published alongside the
   score. Forcing 667-per-bucket would require either fabricating large voids — which destroys
   the only thing the family contributes — or silently reweighting, which makes it
   non-comparable to the synthetic families anyway.
6. **Say what the family is for.** It is not a fourth point of comparison against the synthetic
   buckets; it is a *distribution-shift check* — do methods tuned on synthetic blobs hold up on
   real ones, at the sizes real ones actually come in? Framed that way the small-void skew is the
   experiment rather than a defect, and the family earns its place at a marginal cost of one
   login and 28 MB.

**Does the family earn its price?** Yes, but not at the price the plan quotes. The *data* is
nearly free and the compute is three seconds. What it actually costs is a human in the loop
(irreducible) and an amendment to §3.2/§3.3 (cheap now, expensive after a task has been built
against the current text). "Not worth it" would be the wrong call — the marginal cost is
genuinely near zero and the scientific content is real. "Worth it, as specified" would also be
wrong.

## 13. Human checklist

A login is unavoidable. Steps 1–2 need a person; steps 3–5 do not.

### 13.1 One-time account setup — human

1. Register at **`https://urs.earthdata.nasa.gov/users/new`**. Free. Fields: username, password,
   first/last name, email, country, affiliation, study area, user type, organization.
2. Authorize the LP DAAC application **once**, by opening any granule URL from §13.2 in a browser
   while logged in and clicking through the consent screen at
   `https://urs.earthdata.nasa.gov/oauth/authorize?client_id=FtSFfbOeuxDcdf4px-elGw…`.
   *This consent click is the step that cannot be delegated to an agent.*
3. Write credentials to `~/.netrc` so `curl`/`wget` can follow the redirect chain:
   ```
   machine urs.earthdata.nasa.gov login <username> password <password>
   ```
   `chmod 600 ~/.netrc`.

### 13.2 Fetch — 24 granules, 28.4 MB

**Destination:** `data/srtm_voids/num/` (under the repo's gitignored `data/*`; nothing here is
committed — the committed artifact is the extracted mask list under §3.5's `benchmark/`).

URL template, exact:

```
https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1N.003/
  <CELL>.SRTMGL1N.num/<CELL>.SRTMGL1N.num.zip
```

```sh
mkdir -p data/srtm_voids/num
for CELL in \
  N28E086 N28E085 N29E085 N28E087 N27E088 N28E088 \
  N30E080 N29E080 N30E081 N35E075 N35E076 N36E074 N36E075 \
  N29E095 N29E094 N28E095 N30E095 \
  S10W077 S09W077 S13W072 S16W068 S17W068 S33W070 S32W070
do
  curl -sSL -n -c /tmp/edl-cookies -b /tmp/edl-cookies \
    -o "data/srtm_voids/num/${CELL}.SRTMGL1N.num.zip" \
    "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1N.003/${CELL}.SRTMGL1N.num/${CELL}.SRTMGL1N.num.zip"
done
```

The cookie jar is required — the URS redirect chain sets a session cookie that the DAAC's
final redirect reads. `-n` picks up `~/.netrc`.

**Verify before trusting:** each zip must contain one file named `<CELL>.num` of exactly
**12,967,201 bytes** (3601 × 3601 × 1). Anything smaller is an HTML error page that followed a
302 into a login form.

### 13.3 Expand the draw — no human needed

The CMR metadata API is anonymous. To widen to the full sweep (975 granules, 388 MB):

```sh
curl -s "https://cmr.earthdata.nasa.gov/search/granules.umm_json\
?collection_concept_id=C2763266368-LPCLOUD&page_size=2000&bounding_box=72,26,98,37"
```

(then `-79,-40,-64,-8` for the Andes; the Atlas box `-11,28,12,37` is not recommended — §5.2.)
Granule names, byte sizes and download URLs all come back without credentials; only the fetch in
§13.2 needs the login.

### 13.4 Extract — no human needed

Per tile: unzip → read as `uint8` 3601 × 3601 → `void = (num > 10) & (num < 201)` → 8-connected
`ndimage.label` → equivalent diameter `2·√(area/π)` → keep `16 ≤ d < 160` → emit offset RLE.
120 ms/tile; 3 seconds for the 24. No resampling (§10.1).

### 13.5 No-login fallback, if the login stalls — degraded, not equivalent

Per-tile v2.1 3 arc-second from the Wayback capture of the retired USGS tree:

```
https://web.archive.org/web/<timestamp>id_/
  https://dds.cr.usgs.gov/srtm/version2_1/SRTM3/<Continent>/<CELL>.hgt.zip
```

Resolve `<timestamp>` per file from the CDX API:

```sh
curl -s "http://web.archive.org/cdx/search/cdx?url=dds.cr.usgs.gov/srtm/version2_1/\
SRTM3/Eurasia/N28E086.hgt.zip&output=text&filter=statuscode:200&limit=1"
```

Continents: `Eurasia` (Himalaya), `South_America` (Andes), `Africa` (Atlas). Each zip is ~2 MB
and unzips to `<CELL>.hgt`, **2,884,802 bytes** = 1201 × 1201 int16 big-endian, voids flagged
`-32768`. **Label this path degraded in any output that uses it: IoU 0.60 against 1 arc-second
truth, 39% of void area missing, and no large-bucket components at all** (§7.2).

## 14. Sources

- **SRTMGL1N product page** — https://www.earthdata.nasa.gov/data/catalog/lpcloud-srtmgl1n-003
  (redirected from `lpdaac.usgs.gov/products/srtmgl1nv003/`). DOI
  https://doi.org/10.5067/MEaSUREs/SRTM/SRTMGL1N.003
- **SRTM Collection User Guide** (rev. October 2015) — https://lpdaac.usgs.gov/documents/179/SRTM_User_Guide_V3.pdf
  · Table 1 (product resolutions), Table 2 (`.NUM` fill values), §2.1.4 (tile geometry),
  §2.2.2 (`SRTMIMGM`'s different `.num`)
- **SRTM Topography, v2.1 specification** — `dds.cr.usgs.gov/srtm/version2_1/Documentation/SRTM_Topo.pdf`,
  retrieved via https://web.archive.org/web/20091229164305id_/http://dds.cr.usgs.gov/srtm/version2_1/Documentation/SRTM_Topo.pdf
  · §2.2 (US-only 1 arc-second), §3.0 (3×3 averaging), §3.1 (void flag −32768), availability table
- **NASADEM User Guide V1** (January 2020, JPL) — https://lpdaac.usgs.gov/documents/592/NASADEM_User_Guide_V1.pdf
  · §4.1.2 and Table 3 (void reduction by continent), Table 1 (product contents), Table 2 (NUM index)
- **CMR search API** (anonymous) — `https://cmr.earthdata.nasa.gov/search/granules.umm_json?collection_concept_id=C2763266368-LPCLOUD`
  · granule names, sizes, bounding boxes, download URLs. Collections `C2763266322-LPCLOUD`
  (`NASADEM_SHHP`), `C2763264762-LPCLOUD` (`NASADEM_HGT`), `C2763264768-LPCLOUD` (`NASADEM_NUMNC`)
- **Earthdata Login registration** — https://urs.earthdata.nasa.gov/users/new
- **archive.org `srtm3v2.1`** — https://archive.org/details/srtm3v2.1 · metadata via
  https://archive.org/metadata/srtm3v2.1
- **Wayback CDX API** — `http://web.archive.org/cdx/search/cdx?url=dds.cr.usgs.gov/srtm/version2_1/*`
- **Copernicus GLO-30 grid geometry** — [`copernicus-mask-spec.md`](copernicus-mask-spec.md) §6
  (measured affine transforms), [`copernicus-glo30-recon.md`](copernicus-glo30-recon.md) §6
- **Negative probes** (each returning the status quoted in §4.3):
  `e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1N.003/`, `dds.cr.usgs.gov/srtm/`,
  `portal.opentopography.org/API/globaldem`, `step.esa.int/auxdata/dem/SRTMGL1/`,
  `datenatlas.de/geodata/public/srtmgl1/`

**Tiles measured** (all from the Wayback capture of `dds.cr.usgs.gov/srtm/version2_1/`):
3 arc-second — `N28E086 N28E085 N30E080 N35E075 N27E088 N29E095` (Eurasia),
`S10W077 S16W068 S33W070 S24W068` (South_America), `N31W008 N31W006 N33W005` (Africa),
`N46W122 N48W121` (North_America); 1 arc-second — `N46W122 N48W121` (`SRTM1/Region_01`).
