# P140 — Is native 30 m off-distribution for TerraMind's DEM tokenizer?

Ticket [#140](https://github.com/JStoschek/DEMAE/issues/140) · map [#117](https://github.com/JStoschek/DEMAE/issues/117) · plan [`task1-void-fill-plan.md`](task1-void-fill-plan.md) §2.2 · arms A5, A9

> **What this document is.** A record of an actual run, not a reading of a model card. Every
> number below came out of forward passes over 1,200 real tiles on 2026-08-11, and the run is
> reproducible from the tree: [`scripts/p140_gsd_quantizer.py`](../../scripts/p140_gsd_quantizer.py)
> writes [`p140-gsd-quantizer.json`](p140-gsd-quantizer.json) (CPU) and
> [`p140-gsd-quantizer-mps.json`](p140-gsd-quantizer-mps.json) (MPS), and
> `tests/test_p140_gsd_quantizer.py` asserts the headline figures, the four arms' geometry and
> the two estimators against those records so this prose cannot drift from them silently.

## 0. What this measures, and what it does not

**Measures:** whether native 30 m inputs are off-distribution for TerraMind's DEM tokenizer.

**Does not measure:** what that costs at void filling. Only a retrained head answers the second
question, and §2.2 declines to build one. Every number here is an input-side diagnostic on the
tokenizer's own quantizer; none of them is a metric, and none of them prices an arm.

**Read under [#122](https://github.com/JStoschek/DEMAE/issues/122)'s symmetry rule.** This result
is not available to explain a TerraMind loss unless it is also priced as a discount on a
TerraMind win. **No threshold and no drop condition** are attached, deliberately unlike
[#134](https://github.com/JStoschek/DEMAE/issues/134): these statistics have no familiar scale,
so any cutoff would be invented to look principled, and a cutoff would imply the instrument can
settle whether the shift *costs* TerraMind anything. It cannot.

## 1. Bottom line

**The GSD asymmetry is not visible in the tokenizer's quantizer. Native 30 m sits inside
TerraMesh's own distribution on both statistics, and the arm that "corrects" the GSD makes both
worse.**

| Arm | GSD | Ground | Residual | Code entropy | Effective codes |
|---|---:|---:|---:|---:|---:|
| **`native_30m`** — the benchmark's regime | 30 m | 6.72 km | **0.4660** | **7.140 bits** | 141 |
| `bandwidth_matched_30m` — added; 3:1 source:pixel at native geometry | 30 m | 6.72 km | 0.4662 | 7.131 bits | 140 |
| `upsampled_10m` — same windows on TerraMind's grid | 10 m | 2.24 km | 0.4749 | 6.808 bits | 112 |
| `terramesh_10m` — real TerraMesh val DEM | 10 m | 2.24 km | 0.4676 | 7.123 bits | 139 |

Three readings, in the order they matter.

1. **Native 30 m is indistinguishable from TerraMesh's own tiles.** The entropy gap is
   **−0.018 bits, 95% CI [−0.175, +0.136]** — an interval that spans zero and is 1.3% of the
   13.907-bit ceiling. The residual is *smaller* for native 30 m than for TerraMesh (0.4660 vs
   0.4676), i.e. native inputs land marginally closer to the tokenizer's own rounding grid than
   the data it was trained on does.
2. **Bandwidth is the smallest effect measured, by a factor of 35.** Holding the ground extent
   and moving only the ratio of source post spacing to pixel spacing to TerraMesh's 3:1 costs
   **0.0093 bits** of entropy and **+0.00017** of residual (CI [−0.0005, +0.0008], spanning
   zero). The two things the ticket's references confound bandwidth with are both ~0.32 bits:
   ground extent (0.323) and geography (0.315).
3. **Putting DEMAE's tiles onto TerraMind's native 10 m grid makes both statistics worse, not
   better.** `upsampled_10m` — which is TerraMesh's recipe applied to our windows — is the only
   arm that moves: entropy falls 0.332 bits and the residual rises 0.0089. §2.2's mechanism
   sentence predicted the *direction* of this ("it is fed sharper inputs than it ever trained on,
   not coarser ones"); what the run adds is that the 10 m presentation is not a correction to
   chase, and the drop is an extent effect rather than a bandwidth one (§4).

**What none of this licenses.** Every arm — TerraMesh's own tiles included — reaches about
**7.1 of 13.907 bits** and roughly **140 effective codes of 15,360**, with **two thirds of latent
dimensions pinned to a terminal level** (§5). That is a property of this tokenizer on elevation,
not of this benchmark's inputs, and it is why no arm's absolute number means anything on its own.
The comparison between arms is the finding; the levels are not.

## 2. The instrument

TerraMind quantizes with **FSQ**, and [`p118-terramind-encoder-probe.md`](p118-terramind-encoder-probe.md)
§5.2 established the shape of it on a real tile: `quant_proj` is a `Conv2d(768, 5, 1x1)` with
**3,845 parameters**, and `quantize` is a `FiniteScalarQuantizer` holding **zero**. Re-confirmed
by this run against the installed package:

```
quant_proj   Conv2d(768 -> 5)         3,845 parameters
quantize     FiniteScalarQuantizer        0 parameters,  levels 8-8-8-6-5,  15,360 codes
```

The rounding itself, from `quantize_finite_scalar.py`:

```python
d = self._levels - 1                       # (7, 7, 7, 5, 4)
number = round_ste(F.sigmoid(latent) * d)  # <- the rounding
code   = number / d                        # what the decoder gets
indice = (number * self._basis).sum(-1)    # which of the 15,360
```

Two statistics fall out of that and out of nothing else. Neither needs a target, a head or any
training, which is what makes them comparable across GSDs in a way metres are not.

- **Quantization residual** — `sigmoid(latent) * d − number`, per latent dimension, reported as
  the per-token L2. Measured **in level units**, where every dimension's grid spacing is exactly
  1; the `[0, 1]` code space the decoder reads gives the five dimensions spacings of 1/7, 1/7,
  1/7, 1/5 and 1/4, so a distance there means five different things. The code-space figure is in
  the JSON beside it.
- **Code-usage entropy** over the 15,360 codes, against the **13.907-bit** ceiling.

**The residual is a derived quantity, and the derivation is checked rather than trusted.** The
unrounded `sigmoid(latent) * d` never leaves the module — the only thing a residual can be
measured against is a value TerraMind discards — so `fsq_assign` reconstructs it from the
source. What the module *does* return is then checked against that reconstruction on **every
batch**: the code and the index must match, and the record carries both flags
(`code_matches_module`, `index_matches_module`, both `true`). A reconstruction that agrees on the
rounded value and the index is not free to be wrong about the unrounded one.

## 3. The four arms

All four are **224 px, 196 tokens, 58,800 tokens per arm**, normalized with TerraMind's global
affine `(metres − 670.665) / 951.272` read out of the installed package — never DEMAE's per-tile
standardization, whose failure mode is silent (#118 §7).

| Arm | Construction | Holds | Varies |
|---|---|---|---|
| `native_30m` | the stored tile, untouched | — | — |
| `upsampled_10m` | bilinear ×3 to 672 px, centre-crop 224 | geography | bandwidth **and extent** |
| `terramesh_10m` | real TerraMesh 264 px tile, centre-crop 224 | bandwidth, extent | geography |
| `bandwidth_matched_30m` | anti-aliased ↓ to 75 px, bilinear ↑ to 224 | geography, extent | bandwidth |

The first three are the ticket's. **The fourth is added here**, and the reason is a hole the
ticket's two references leave open between them: fixing the pixel count at 224 makes ground
extent and GSD the *same knob*, so `upsampled_10m` covers a ninth of `native_30m`'s area and any
difference between them could be bandwidth or extent. Neither reference separates those.
`bandwidth_matched_30m` holds the extent and the pixel grid and moves only the ratio of source
post spacing to pixel spacing to TerraMesh's 3:1 — the spectral condition without the geometry.
It costs one forward pass per tile and no new data. §4 is what it bought.

Two construction notes, both of which changed the answer:

- **The down-leg is anti-aliased and the up-leg is not, and the asymmetry is deliberate.**
  `F.interpolate` point-samples, so a plain bilinear *downsample* aliases: measured, before the
  flag existed, a pixel-Nyquist checkerboard came through the 224 → 75 → 224 round trip at
  **std 0.129** instead of ~0 — the arm was folding high frequencies back in where it claimed to
  be removing them. With the low-pass it lands at **0.0014**, a 700× suppression, and a linear
  ramp still survives to 0.004. TanDEM-X's 30 m posts are a band-limited product, not a
  point-sampling of something finer, so a ~90 m stand-in has to be band-limited too. The up-leg
  is plain bilinear because that is TerraMesh's own step verbatim
  ([`p138-terramind-corpus.md`](p138-terramind-corpus.md) §3).
- **224 is not divisible by 3**, so the down-leg goes through 75 px and the bandwidth reduction is
  **224/75 = 2.9867**, within 0.5% of TerraMesh's 3:1. The figure is in the record and in a test.

**The 10 m arms are not the tokenizer's training crop.** That was a random 0.8–1.0 scale crop of
the 264 px tile resized to 256 (p138 §3). A centre crop is used instead because what these arms
have to match is *each other* — 224 px over 2.24 km — so that `upsampled_10m` and `terramesh_10m`
differ in geography and nothing else.

### The samples

**DEMAE: the 300 tiles #134 already used**, re-derived through the same rule rather than re-drawn
— 15 per stratum across all 20, evenly spaced over each stratum's sorted ids, all-land, test
split. 208 source COGs, latitude −88.0 to +80.0, longitude −168.3 to +159.8. Reusing #134's rule
is what makes the three DEMAE arms *paired*: three views of one tile set, so terrain drops out of
their differences.

**TerraMesh: 300 val DEM tiles**, dealt round-robin across all nine `val/DEM` shards (eight
MajorTOM, one SSL4EO-S12 — the split's own source mix). Latitude −45.2 to +70.6, longitude −161.7
to +170.9; elevations −5 m to 6,448 m.

Three choices worth stating:

- **val, not train.** The reference has to be in-distribution *by construction* — same product,
  same preprocessing, same grid — without being literally in the tokenizer's training set, or a
  low residual there could be read as memorization rather than as bandwidth.
- **Round-robin across nine shards, not the head of one.** Taking the first 300 members of one
  shard would rest on an assumption about how that shard is ordered. Dealing across all nine
  holds whatever ordering each has, and picks up the val split's source mix rather than one side
  of it. (The shards do turn out to be globally scattered — consecutive members land in Malaysia,
  Poland, Japan, Uganda — but that is an observation, not something the rule leans on.)
- **37.7 MB fetched, and zero tiles committed.** The shards are 388 MB each and this needs their
  first 4 MB, so the fetch is an HTTP range request per shard, cached under `data/` which
  `.gitignore` excludes. [#128](https://github.com/JStoschek/DEMAE/issues/128)'s standing rule —
  statistics only, never tiles — is what makes the share-alike licence a non-issue. The record
  carries the dataset revision (`e144fef5`) and a SHA-256 of the ordered sample-id list, so the
  draw is re-derivable exactly without 300 ids being committed to describe 300 tiles nobody can
  see.

**The format is tar-of-zarr, not parquet.** #140's text says parquet; the only parquet files in
the repo are `{train,val}_metadata.parquet`. The modality data is webdataset `.tar` shards whose
members are `*.zarr.zip` — a zipped zarr v2 store holding one blosc-zstd `(1, 1, 264, 264)` int16
chunk. Decoding is four zip reads and a `frombuffer` via `numcodecs`, which is why the script
does that rather than going through xarray/zarr (the stores are zarr v2, the resolvable zarr is
v3, and a `ZipStore` there needs a real file on disk). `numcodecs` is the one package this run
adds to #118's environment; it is **not** a project dependency, and neither is terratorch.

## 4. What the fourth arm bought: bandwidth is 35× smaller than either confound

Entropy differences, in bits, with 95% intervals from a **tile-level bootstrap** (1,000
replicates, seed 140). The three DEMAE arms share each replicate's tile draw, so terrain drops
out of their comparisons; TerraMesh is a different 300 tiles and is resampled on its own, which
is why its intervals are ~5× wider.

| Comparison | Isolates | Δ entropy | 95% CI |
|---|---|---:|---|
| `native_30m` → `bandwidth_matched_30m` | **bandwidth** | **−0.0093** | [−0.0146, −0.0033] |
| `upsampled_10m` → `bandwidth_matched_30m` | **ground extent** | +0.3230 | [+0.2889, +0.3590] |
| `upsampled_10m` → `terramesh_10m` | **geography** | +0.3147 | [+0.1486, +0.4778] |
| `native_30m` → `terramesh_10m` | everything at once | −0.0176 | [−0.1745, +0.1359] |

And the residual, as a paired mean difference over the 300 tiles:

| Comparison | Δ residual | 95% CI |
|---|---:|---|
| `native_30m` → `bandwidth_matched_30m` | **+0.00017** | [−0.00047, +0.00080] |
| `native_30m` → `upsampled_10m` | +0.00885 | [+0.00652, +0.01118] |
| `upsampled_10m` → `bandwidth_matched_30m` | −0.00869 | [−0.01101, −0.00637] |

**The decomposition is clean.** `bandwidth_matched_30m` and `upsampled_10m` have the *same* 3:1
source-to-pixel ratio and differ only in ground extent, and they are 0.323 bits apart. Native and
`bandwidth_matched_30m` have the same extent and differ only in that ratio, and they are 0.0093
bits apart. So essentially all of `upsampled_10m`'s drop is extent, and almost none of it is
bandwidth — which is the question the ticket's two references, on their own, could not answer.

The bandwidth effect is *resolvable* — its interval excludes zero — and it is 0.07% of the
codebook's ceiling. Both facts belong in the same sentence. Resolvable is not the same as large,
and this instrument fixes no threshold at which either word would become a decision.

**On the last row.** `native_30m` → `terramesh_10m` moves GSD, extent and geography at once, and
lands at −0.018 bits with an interval spanning zero. That is the ticket's actual question, and
the answer is that the benchmark's own regime is not distinguishable from the tokenizer's own
data. It is worth being clear that this is a *coincidence of two larger effects that happen to
cancel* — the extent gain and the geography loss are each ~0.32 bits — and not evidence that
neither matters. What it does establish is that no GSD-shaped penalty is visible at the point
where the benchmark actually runs.

## 5. The context that keeps every number above small

**Two thirds of all latent assignments sit at a terminal level**, on every arm:

| Arm | Saturated | Per-dimension (levels 8, 8, 8, 6, 5) |
|---|---:|---|
| `native_30m` | 0.679 | 0.560, 0.512, 0.722, 0.795, 0.809 |
| `bandwidth_matched_30m` | 0.679 | 0.559, 0.511, 0.722, 0.796, 0.809 |
| `upsampled_10m` | 0.678 | 0.558, 0.487, 0.709, 0.813, 0.823 |
| `terramesh_10m` | 0.694 | 0.653, 0.384, 0.806, 0.761, 0.864 |

This is reported beside the residual because the two are easy to confuse and point opposite ways.
`sigmoid` squashes, so a latent driven far from zero is *pinned* to an end of the grid and its
residual goes to zero for free — which looks like a well-represented input and is the opposite of
one. That is why the residual is a blunt instrument here: it is dominated by the third of
assignments that are not pinned, and every arm sits at ~0.466 against **0.6455**, the value the
residual would take if the latents ignored the grid entirely. The arms are only modestly better
placed than indifference, and they are all equally so.

The same picture in the entropy. Every arm reaches ~7.1 of 13.907 bits and ~140 effective codes
of 15,360, TerraMesh's own tiles included. The Miller–Madow correction is **0.006–0.007 bits** on
every arm — negligible, because only 472–575 distinct codes are ever used, so the plug-in
estimator is nowhere near its bad regime. Every arm draws exactly 58,800 tokens by construction,
so what bias remains is common to all four and cancels out of the differences in §4.

**The per-dimension entropies say the arms differ in composition even where the totals agree.**
TerraMesh reaches the same joint entropy as native 30 m by a visibly different route — dimension
0 at 2.07 bits against native's 2.61, dimension 1 at 2.85 against 2.55, dimension 2 at 1.23
against 1.93. That is a geography signature, not a GSD one, and it is the sort of thing a single
scalar would have hidden. (The five marginals sum to 10.4 bits against a joint of 7.14, so the
axes are strongly dependent; nothing here turns on that.)

## 6. Cost, and where it runs

| | |
|---|---|
| Full run, CPU (1,200 forward passes + 1,000 bootstrap replicates) | **47 s** |
| Full run, MPS | **29 s** |
| TerraMesh fetch | 37.7 MB, once, cached |
| Weights | the same 90.2 M encoder A5 and A9 already load |

No new regime, no retrained head, no scope reopen, and nothing rented. Both device records are
committed, and the agreement is **not** a formality here: `round()` is discontinuous, so a token
whose pre-rounding value sits near a half-integer can be assigned a different code on a different
device, and both statistics are built on that assignment. Measured — the two runs agree to
**4.2e-5 bits** of entropy on the native arm (worst arm 1.7e-4), land within 1.8e-7 on the
residual, and pick the **same number of distinct codes** on all four arms. The statistic is not
bit-stable across devices; it is stable three orders of magnitude below the smallest difference
this document reads.

## 7. What this changes

- **§2.2's third bullet** stops saying "measured under #140" and states the figures: the shift is
  not visible in the quantizer, native 30 m is indistinguishable from TerraMesh's own tiles
  (−0.018 bits, CI spanning zero), and the bandwidth component alone is 0.0093 bits against a
  13.907-bit ceiling.
- **§2.2 gains the direction it did not have**: moving to TerraMind's native 10 m grid makes both
  statistics *worse*, so the out-of-scope secondary 10 m regime the next bullet declines to build
  is not a correction the plan is passing up.
- **Nothing else moves.** The caveat stays, still symmetric, still without a threshold. A
  measurement that finds no shift is not a licence to drop the caveat — it is what stops the
  caveat being used, in either direction, as an unpriced explanation later.

## 8. Sources

- The runs: [`p140-gsd-quantizer.json`](p140-gsd-quantizer.json) (CPU),
  [`p140-gsd-quantizer-mps.json`](p140-gsd-quantizer-mps.json) (MPS)
- Quantizer source: `terratorch/models/backbones/terramind/tokenizer/quantizers/quantize_finite_scalar.py`,
  terratorch 1.2.10
- TerraMesh: [`ibm-esa-geospatial/TerraMesh`](https://huggingface.co/datasets/ibm-esa-geospatial/TerraMesh),
  revision `e144fef5`, CC-BY-SA-4.0, `gated: false`, `private: false`
- Prior runs this one builds on: [`p118-terramind-encoder-probe.md`](p118-terramind-encoder-probe.md)
  (the encoder, the constants, the normalization trap),
  [`p134-posemb-materiality.md`](p134-posemb-materiality.md) (the 300-tile selection rule),
  [`p138-terramind-corpus.md`](p138-terramind-corpus.md) (TerraMesh's DEM source, resampling and
  tile size)
- Environment: #118's resolved 113 packages
  ([`p118-terramind-requirements.txt`](p118-terramind-requirements.txt)) plus `numcodecs` 0.16.5
