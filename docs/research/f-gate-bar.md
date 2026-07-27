# Gate F's pre-registered bar — the 500,000-window scale tier

**Ticket:** [Pin the Gate F bar — before the 500k corpus exists (#69)](https://github.com/JStoschek/DEMAE/issues/69) ·
**Map:** [Phase F (#67)](https://github.com/JStoschek/DEMAE/issues/67) ·
**Machine-readable bar:** [`f-gate-bar.json`](f-gate-bar.json) ·
**Rules:** [`dataset/validate_f.py`](../../src/dataset/validate_f.py) ·
**Applied at:** [Gate F review (#74)](https://github.com/JStoschek/DEMAE/issues/74)

Committed **before** the 500k draw ([#71](https://github.com/JStoschek/DEMAE/issues/71)),
the corpus ([#72](https://github.com/JStoschek/DEMAE/issues/72)) and the report
([#73](https://github.com/JStoschek/DEMAE/issues/73)). The reason is the one
[#47](https://github.com/JStoschek/DEMAE/issues/47) and
[#63](https://github.com/JStoschek/DEMAE/issues/63) both gave: a criterion chosen after
seeing the numbers can be argued either way.

This document is the argument. The JSON is the contract, and
[`validate_f.py`](../../src/dataset/validate_f.py) is the executable half; neither decides
a threshold that is not written down here first.

## Amendment — 2026-07-27: the `n_h` table was evaluated on a rounded input

One field has been corrected after the draw:
`committed_inputs.allocation.n_h`. Twelve of its twenty entries (strata 1, 3, 4, 8, 9, 10,
11, 12, 14, 15, 16, 18) were off by **±1 or ±2 windows**. The record lives in
[`f-gate-bar.json`](f-gate-bar.json)'s `post_commit_corrections`, per this bar's own rule
that a post-hoc change arrives as a separate, dated commit that justifies itself.

**The cause.** The registered table was produced by feeding the **4-decimal display
`sigma_h`** that [`d2-allocation-nh.json`](d2-allocation-nh.json) prints for legibility
back into the allocation rule. `dataset.select.allocate` recomputes `sigma_h` from the
frame at full precision, so largest-remainder rounding lands on slightly different
integers. [#44](https://github.com/JStoschek/DEMAE/issues/44) documented this exact trap in
d2's own notes, and `tests/test_select.py::test_allocation_reproduces_committed_table_on_real_catalog`
asserts the full-precision path. #71 reproduced the wrong table bit-for-bit from the
rounded input and published the deltas in
[`f-scale-draw.json`](f-scale-draw.json) → `gate_f_bar_crosscheck` rather than editing a
pre-registered file it had just drawn against.

That report has since been regenerated against the corrected table, so at `HEAD` it
reports agreement on all 20 strata and its delta table is empty. The disagreement is a
fact about a state that really existed, and git is where that state is kept —
`git show 8560ff9:docs/research/f-scale-draw.json` is the reproduction as #71 published it.

**Why this is not a threshold moved to fit an outcome.** The corrected table depends on
exactly three things — the kept frame, the persisted centroids/scaler, and ADR-0005's
allocation rule. All three were committed *before* this bar was written and none has been
touched since, so the number could have been produced at pre-registration time and would
have been identical. No Gate F measurement, and nothing about the shipped list or the
corpus, enters the computation. What changed is an input-precision mistake in how a
registered value was evaluated.

**What it would have cost.** `F2a` compares the shipped list's per-stratum counts against
this table. Left as committed, it would have fired a **Defect on a correct draw** when
[#73](https://github.com/JStoschek/DEMAE/issues/73) executes the gate — an INCOMPLETE
verdict chasing a bug that does not exist.

Nothing else moved. `sum_n_h` is still exactly 500,000, the coastal provenance tags still
sum to `coast_floor = 6,700`, `floor_clamp` is still 5,000 with zero strata below it, and
the smallest `n_h` is still **7,343** in stratum 6 — so every argument below stands
unchanged, including F2c's inertness and F3's stratum 2 / 17 figures (23,159 / 18,472 were
among the eight entries that were already right).

## Gate F is not Gate E with a bigger number

The temptation was to re-register E's nine criteria at 500k and add one. That would have
been mostly theatre. Three of E's criteria are **structurally inert** at this budget — they
cannot return anything new — and one is *arithmetically guaranteed to fire on a healthy
corpus*. Carrying them as gates would have produced a Gate F whose verdict was decided
before the draw.

### E2.5 is bit-identical, not merely reproducible

The attrition-by-stratum criterion reads the 4,386,720-row catalog and the Phase-C gates.
**The budget never enters it.** Same catalog, same gates, same centroids ⇒ the same
3.04–5.32× ratios, the same fired pairs, the same everything. It cannot even surface a
*new* gate × stratum pair. And its firing is already dispositioned in full by
[ADR-0006](../adr/0006-accept-edm-flm-hollowing-stratum19.md) and
[ADR-0007](../adr/0007-accept-degenerate-flatness-gate-flat-strata.md), so re-gating it
would re-raise a settled FAIL — which map #67 places out of scope.

Carried as **`F5_gate_attrition_published`**: the table is reprinted with a disposition
pointer, and gates nothing.

### The gallery's question was answered by the centroids, not the budget

`E2.4`'s registered question is whether the k=20 strata are terrain-meaningful, and its
knob is `k`. The same persisted centroids produce the same partition, so
[#53](https://github.com/JStoschek/DEMAE/issues/53)/[#54](https://github.com/JStoschek/DEMAE/issues/54)'s
**20/20 coherent, 0 mixed, 0 incoherent** already answered it — and `k` is out of scope for
this map. Re-rendering draws different *tiles* from identical strata.

There is one genuinely new effect: a 10×-deeper draw reaches further from each centroid, so
more peripheral cluster members appear. Twelve tiles per stratum has almost no power to
detect it, and the protocol's own `disavowed` clause says blending in a continuum is
expected rather than a failure.

Carried as **`F4_gallery_published`**: still rendered, still labelled — #75's honesty notes
need a gallery of the corpus actually shipped — but no pass/fail.

### The per-COG cap of 64 would fire on a perfect corpus

A 1° COG holds at most **256** kept windows (mean 126.6, median 122, across 24,118 COGs).
At 500k the average draw is **16.4%** of the kept pool, putting mean windows/COG near
**21** — and at 50k a single COG already drew **44** at one-tenth that rate.

The cap sits *inside* the healthy range. No budget-scaled replacement helps either, because
the ceiling is a COG's own inventory rather than a number we choose. And the plan's
§215–228 knob table has **no per-COG row**, so a firing could never have been a Finding —
the guard was always a Defect wearing a cap's clothes. The nearest knob, `dedup`, defaults
to grid-guaranteed zero overlap, which holds at any budget.

Retired, and replaced by the two checks it was proxying for (below).

## What Gate E never checked

### It never read a pixel

`E2.1a_metric_fidelity` compares the manifest's frozen metrics to the catalog's — but
[`manifest.py`](../../src/dataset/manifest.py) *copies* that vector from the catalog. So
E2.1a verifies the **join**, not the imagery. Nothing in Gate E's seven code criteria opens
a shard.

There is a whole-corpus substitute sitting in the schema. The manifest's `std` column **is**
pixel-derived — [`normalize.prepare_tile`](../../src/dataset/normalize.py) computes it on
the stored int16 array — while the catalog's `land_std` was computed independently at full
res in Phase C. Cross-joining the two over all 49,998 debug-tier tiles:

| | as-built | off-by-one window mixup | full shuffle |
|---|---|---|---|
| median relative disagreement | **0.000084** | 0.477 | 0.867 |
| tiles over 0.5 relative | **0 (0.000%)** | 48.1% | 76.2% |

Median agreement is at the int16 quantization floor. The tail is benign and explained — 18
tiles exceed 0.1, worst **0.46** — the plausibility clip and the differing land-mask
definition, not mis-joins. A wrong-window shard lights up roughly half the corpus.

**`F1d_pixel_integrity`** (Defect): fire if more than **0.1%** of tiles disagree by more
than **0.5** relative. `0.5` sits above the entire measured benign tail and below the mixup
signature; `0.001` allows a handful of benign cases at 500k where the debug tier had zero.
Zero tolerance was rejected as brittle against known-benign causes.

It does **not** test plausibility. A corruption preserving land-std — a horizontal flip —
passes it, and would be obvious by eye. That is the residual value of keeping F4's gallery
rendered.

### A duplicate could hide in plain sight

A repeated `window_id` **inside** the selection list passes `F2a` (md5, row count,
per-stratum counts and provenance all still match) and passes `F2b` (`realized ≤ selected`
still holds). Nothing in Gate E would have caught it.

**`F6a_distinctness`** (Defect, zero tolerance): every selection `window_id` distinct,
every manifest `tile_id` distinct. `tile_id` is derived from `source_cog` + `win_row0` +
`win_col0`, so a repeat is definitionally a bug. Debug-tier baseline: 50,000/50,000 and
49,998/49,998.

**`F6b_cog_inventory_bound`** (Defect, zero tolerance): no COG yields more windows than it
holds. This is the mechanical failure the retired cap was watching for — a looping read, a
collapsed stratum — in the one form that survives a budget change.

## The one number the budget actually changes

Everything above is bookkeeping. `F3` is the criterion this map exists for.

[ADR-0007](../adr/0007-accept-degenerate-flatness-gate-flat-strata.md) accepted the
flatness gate's concentration on the flat strata and attached exactly one string: a
**Phase-F revisit trigger** on the strata **2/17 drainage loss**, "accept-for-now, re-test
at scale." Its leg 3 leaned partly on *"each keeps its would-be pool ≥ 53× its
allocation"* — a **50k** fact.

Neyman shares are budget-invariant, so at 500k those same pools sit at **≥ 5.3×**, and FSCS
reaches ten times deeper into each one. Quantified with `texture` as the drainage proxy,
thresholded at each stratum's **pre-gate p75**:

| | stratum 2 | stratum 17 |
|---|---|---|
| pre-gate p75 `texture` | 0.017817 | 0.026826 |
| pre-gate drainage tail | 54,421 | 42,163 |
| survives the flatness gate | 26,271 (**48.3%**) | 17,930 (**42.5%**) |
| `n_h` @500k | 23,159 | 18,472 |
| **tail ÷ `n_h`** | **1.13×** | **0.97×** |

At 50k the allocation was a *tenth* of the surviving drainage tail, so FSCS could
cherry-pick the most dissected survivors and the corpus looked fine even though half the
tail was gone. At 500k the allocation **is** the tail — stratum 17's allocation now exceeds
every drainage-bearing tile the gate left standing. The corpus can no longer be better than
its own depleted pool. **That** is the regime change the revisit trigger was pointing at,
and why this is not a re-run of E2.5.

### Why `texture`

It is the only frozen metric that separates dropped from kept in these strata
([#64](https://github.com/JStoschek/DEMAE/issues/64)): stratum 2 dropped median **0.015**
vs kept **0.006**, stratum 17 **0.021** vs **0.009**. `geomorphon_entropy` does **not**
separate — stratum 2 dropped 1.887 vs kept 1.906 — so it cannot serve.

### The criterion, and the conjunction that had to be abandoned

**`F3_drainage_representation`** (Finding, knob = the `degenerate` flatness screen): for
stratum h ∈ {2, 17}, `r_h` — the share of shipped stratum-h tiles with
`texture ≥ pre-gate p75(h)` — must be at least **0.25**. The reference is 0.25 *by
construction*, and it is the drainage share the terrain itself has before the gate.

The first design was a **conjunction**, mirroring ADR-0007's own conjunctive LOOSEN
definition: fire only if the draw both *exhausts* the surviving tail (so the gate, not the
budget, is the binding constraint) **and** leaves the corpus deficient (so it matters).
That attributes correctly — a deficit at *low* exhaustion is the sampler declining
available tiles, which is not the flatness knob's fault.

**It is logically unsatisfiable at this budget.** Because `n_h` ≈ the tail size, realized
share and exhaustion are the same statistic up to a constant:

```
r_h = e_h · tail_h / n_h
```

Requiring `e_h ≥ 0.50` forces `r_h ≥ 0.57` (stratum 2) or `0.49` (stratum 17) — far above
any deficit threshold. The conjunction could never fire. That is the vacuity trap #63's bar
warned about, in a new costume, and it is recorded here so a later reader does not
re-invent it.

So the gate is single-limb, and the attribution is **published instead**: `r_h`, the
exhaustion `e_h`, and the kept-pool proportional share (**0.1741** / **0.1703**) reported
side by side. `r_h` near the kept-pool share means FSCS drew proportionally and declined an
available tail; `r_h` near the exhaustion ceiling means the survivors genuinely ran out. The
fresh map needs that distinction and should not have to re-derive it.

### The bar is genuinely two-sided, and tenfold harder than at 50k

| | @50k | @500k |
|---|---|---|
| drainage tiles needed for `r_h = 0.25` (str. 2) | 579 | 5,790 |
| as a share of the 26,271-tile surviving tail | **2.2%** | **22.0%** |

Same rule, ten times the pressure. And nothing predetermines the outcome: drawing
proportionally from the kept pool gives `r_h` = 0.174/0.170 → **fires**; coverage-seeking
at ~1.5× proportional gives ~0.26 → **passes**. FSCS is coverage-balanced in the
standardized six-metric space and the drainage tail is a sparse region of these strata, so
it may well over-draw it. That is exactly what is being measured.

### Two tiers, copying #63 exactly

Strata **2 and 17 gate**. Strata **5 and 7** are computed identically and reported as
**corroboration only, never independent triggers** — ADR-0007 scoped its trigger to 2/17,
stratum 5 drops genuine low-information smooth plains (dropped `texture` identical to
kept), and stratum 7's `texture` is amplified sub-metre sastrugi that runs the *opposite*
way (kept 0.098 > dropped 0.091).

**If it fires:** Finding. It reopens ADR-0007's flatness knob, which per map #67's
Destination means **record the verdict and stop**. The loosening rule and its blast-radius
analysis are a fresh map.

## Attrition, derived rather than scaled

Both debug-tier drops were `degenerate` re-screen fires (stratum 8, one coastal): the
int16-quantized `land_std` falling under 3.0 m where the catalog's float value sat just
above. So attrition is a **threshold-boundary artifact**, and its density is measurable
rather than guessable.

The kept pool holds **1,449** windows in `land_std ∈ [3.0, 3.01)` — about 145 per 0.001 m. A
50k draw touches ~2.4 of those per 0.001 m of band, and exactly **2** dropped, implying a
flip band ≈**0.001 m** wide. That matches F1d's median relative agreement of 8.4e-5, which
is ~0.00025 m at `land_std` = 3.0. Attrition is therefore **linear in budget**.

- **Pre-registered expectation: ~20 drops** at 500k, Poisson spread √20 ≈ 4.5, so **≈11–31**
  at ±2σ. Written down so the report has a number to be surprised against, not just a
  threshold to clear.
- **`F7_attrition_expectation`** (Defect): fire if the drop *rate* exceeds **0.0004** — 10×
  the debug-tier rate, ~40σ from the expectation. Keyed on the rate so it reads the same at
  any budget. Defect rather than Finding because the re-screens recompute the same
  quantities on the same full-res arrays Phase C scored, so a 10× excursion means a read
  path, mask-cache or clip-bound regression.
- **`F7b_registered_status_set`** (Defect, zero tolerance): every ledger status in
  `{ok, zero_fill, degenerate, no_land}`. A handful of I/O errors and a handful of boundary
  re-screens are the same *number* and completely different news.

`F2c`'s relative limb is disclosed as effectively non-binding here: its 2% limb needs **147**
drops inside the *smallest* stratum, against ~20 expected corpus-wide.

## The rescaled floor, and the one that did not move

`F2c`'s absolute limb goes **500 → 5,000**: the floor is #42's `max(500, 1%·budget)` clamp,
and `select.allocate(budget=500_000)` reports `floor_clamp = 5000`,
`floor_clamp_strata_below = 0`. Still structurally inert — the smallest `n_h` is **7,343**
(stratum 6, which also has the tightest headroom in the corpus at **4.39×**, tighter than
any flat stratum) — and retained for the same reason as before: if attrition pushes a
realized count under it, that inert clamp has become retroactively binding, which is the
event worth naming.

`F2d`'s **0.01337 does not move**. It is coastline's within-frame *rate*
(40,815 / 3,052,735), not the count, so it is budget-invariant. The realized count is
reported as itself, never rounded up to 6,700, never described as "met".

## The normalization statistic is conditionally pre-registered

`dataset_version` is Phase 1's reproducibility contract, so the recorded global statistic is
gated either way — but **which** check is correct depends on
[#70](https://github.com/JStoschek/DEMAE/issues/70), which was unresolved when this bar was
committed. The #47/#50 precedent applies: both branches are written down now, and #70 may
**select**, not invent.

- **`tier_specific`** — `corpus_meta.json`'s `global_mean`/`global_std` must reproduce
  exactly from the shipped 500k tiles.
- **`inherit_debug`** — they must equal the debug tier's committed
  `951.1396151875607` / `1012.9799372491552` exactly, and the values recomputed from the
  shipped corpus are **published** as a divergence diagnostic carrying no threshold.

The report resolves the live branch from `corpus_meta.json:normalization_stat_source` via
`FBar.live_normalization_branch`, which refuses a branch the bar never pre-committed.

## Two mechanical notes on the code

**Gate E's bar is frozen, not superseded.** ADR-0006 and ADR-0007 rest on
[`e2-gate-bar.json`](e2-gate-bar.json)'s registered numbers. Rescaling `THIN_ABSOLUTE` in
place would have silently re-pointed *Gate E's* measurements at 5,000, so
[`validate.py`](../../src/dataset/validate.py) is untouched and
[`validate_f.py`](../../src/dataset/validate_f.py) imports its pure rule functions and
supplies its own constants. `tests/test_validate_f.py` pins that the two modules disagree
about the floor and that both bars still load.

**The anchors now interpolate their constants.** Gate E's `_PROSE_ANCHORS` hard-code the
literal `"500"` rather than deriving it from `THIN_ABSOLUTE`, so the tie between committed
prose and measured constant is by *convention*: an edit to the constant alone would leave
`read_bar` passing while measuring against a number nobody registered. Every Gate F anchor
is built by interpolating the constant it guards, closing that gap.

## What gates, at a glance

| criterion | class | gates | what it tests |
|---|---|---|---|
| `F1a_metric_fidelity` | Defect | yes | manifest frozen metrics == catalog's, exact |
| `F1b_drop_selectivity` | Defect | yes | dropped windows metric-indistinguishable from retained |
| `F1c_pool_ecdf` | published | no | corpus vs kept-pool ECDFs, raw and N_h/n_h-reweighted |
| `F1d_pixel_integrity` | Defect | yes | **the pixels are the window the manifest claims** |
| `F2a_selection_integrity` | Defect | yes | the list as loaded == #71's commit |
| `F2b_materialization_counts` | Defect | yes | `realized ≤ selected` per stratum |
| `F2c_thin_stratum` | Finding | yes | realized `n_h` ≥ 5,000; 3× / 2% conjunction |
| `F2d_coastal_floor` | Finding | yes | realized coastal rate ≥ 0.01337 |
| `F3_drainage_representation` | **Finding** | **yes** | **ADR-0007's 2/17 re-test: `r_h` ≥ 0.25** |
| `F4_gallery_published` | published | no | 12 × 20 gallery + terrain labels |
| `F5_gate_attrition_published` | published | no | attrition table + disposition pointer |
| `F6a_distinctness` | Defect | yes | no window drawn twice, no tile written twice |
| `F6b_cog_inventory_bound` | Defect | yes | no COG yields more than it holds |
| `F6c_cog_spread_published` | published | no | per-COG counts and fraction-of-inventory |
| `F7_attrition_expectation` | Defect | yes | drop rate ≤ 0.0004 (expect ~20) |
| `F7b_registered_status_set` | Defect | yes | no unregistered ledger status |
| `F7c_ledger_status_published` | published | no | full status breakdown |
| `F8_normalization_statistic` | Defect | yes | recorded global stats match the live branch |

**Roll-up (carried from #47 unchanged):** any Defect open → **INCOMPLETE**, fix and stay in
the map, Findings not yet adjudicated. Else any Finding fired → **FAIL**, record the
payloads and stop. Else → **PASS**, and #75's honesty notes proceed against the shipped
scale corpus.
