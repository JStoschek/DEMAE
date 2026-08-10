# DEMAE Phase 1: Pretraining Plan

**Scope:** the MAE pretraining run itself: architecture, recipe, splits, monitoring, ops, and run schedule. Downstream fine-tuning (Phase 2) gets only an interface section.
**Inputs:** the shipped 500k scale corpus (`data/corpus_500k`, 499,934 tiles, `dataset_version bcbd6484d8116199c5c1160e21ea0284`), the ADR-0002 storage format, the ADR-0003 loader, and the corpus card ([`f-honesty-notes.md`](research/f-honesty-notes.md)).
**Hardware:** single rented RTX 5090 (32 GB) with local NVMe, vast.ai.
**Status:** Proposed (v1, 2026-07-28). Every recipe number below is cited; deviations from cited defaults are argued, not assumed.

---

## 0. Decision summary

| Decision | Choice | Basis |
|---|---|---|
| Codebase | Minimal reimplementation (~500 lines) on timm ViT blocks; official MAE repo used as spec only | §1 |
| Backbone | ViT-S/16, single channel; ViT-B/16 as the final release run | §2.1 |
| Positional encoding | Fixed 2D sin-cos (MAE default); GSDPE deferred to Phase 4 | §2.3 |
| Masking | 75% uniform random, per-sample; ablate 85%/90% | §3 |
| Decoder | 4 blocks × 256-dim, 8 heads (ViT-S); 4 × 512 (ViT-B) | §2.2 |
| Loss | MSE on masked patches only; `norm_pix_loss` ON with fp32 target stats and a raised variance floor; ocean pixels zero-weighted | §4 |
| Input normalization | `per_tile` (ADR-0003 default); `global` stays a Phase-3 ablation | §5 |
| Optimizer | AdamW, betas (0.9, 0.95), weight decay 0.05, fused | §6 |
| LR | blr 1.5e-4 scaled by eff_batch/256; eff. batch 2048 → lr 1.2e-3 | §6 |
| Schedule | 20-epoch linear warmup, cosine decay to 0, 400 epochs (extend to 800 on evidence) | §6 |
| Precision / kernels | bf16 autocast (no GradScaler), SDPA/Flash attention, `torch.compile(dynamic=False)` | §7 |
| Init | Main run from scratch; one continual-pretraining arm from ImageNet MAE ViT-B (patch-embed summed to 1 channel) | §8 |
| Split | Fill the manifest `split` column: ~95% train / ~2% val / ~3% test, blocked geographically (never random) | §9 |
| Monitoring | Train + fixed-mask val recon loss, stratified reconstruction grids, kNN probe on `stratum_id` + `slope_mean` + `geomorphon_entropy` (**all three adjudicate**, and val loss only within one arm — §10, [ADR-0010](adr/0010-frozen-phase1-release-recipe.md)), no EMA | §10 |
| Ops | Checkpoint every 30-60 min, sync off-instance, resumable from step 1; on-demand verified host | §11 |

---

## 1. Codebase

Start from the official MAE recipe but **not** the official code as a base:

- `facebookresearch/mae` is the canonical spec (all recipe numbers below trace to it) but it is frozen on `timm==0.3.2`, uses fp16 + GradScaler with documented NaN issues ([issue 65](https://github.com/facebookresearch/mae/issues/65)), predates SDPA/compile, and is **CC-BY-NC 4.0 licensed**, which is a poor foundation for an open-weights release.
  *(Corrected 2026-07-29 — [P101](research/p101-mae-checkpoint-licence.md), [#103](https://github.com/JStoschek/DEMAE/issues/103): "poor foundation" is right, the implied mechanism was wrong. **CC BY-NC 4.0 has no ShareAlike term** — the string appears zero times in the licence — so it is not copyleft and nothing forces a derivative onto the same licence. The two terms that bind are the **NonCommercial** purpose restriction and **§3(a)(4)**, which requires that the licence applied to Adapted Material "must not prevent recipients … from complying with this Public License." Reading that plainly: a permissive licence affirmatively grants the commercial use CC BY-NC forbids, so it is exactly what §3(a)(4) blocks. The practical consequence is narrower than copyleft and sharper: **running** the official code or weights is unproblematic; **shipping derived weights under permissive terms** is what is barred. This is why §8 changes the checkpoint rather than the objective.)*
- `models_mae.py` is small and self-contained. Port it onto current **timm** ViT blocks (Apache-2.0): `PatchEmbed(in_chans=1)`, `Block` with SDPA, fixed 2D sin-cos positional embeddings, and a from-scratch decoder. HuggingFace's `ViTMAEForPreTraining` is a fallback but its example trainer is explicitly not battle-tested for pretraining ([transformers issue 29416](https://github.com/huggingface/transformers/issues/29416)); `lightly` (MIT) is a reasonable reference implementation with published MAE benchmark numbers.
- Single-channel changes are three: `in_chans=1` in the patch embed, prediction dim 16×16×1 = 256 per patch, and the per-patch target normalization computed over 256 values instead of 768.

The training loop consumes `dataset.loader` unchanged (map-style memmap Dataset per ADR-0003). MAE token masking lives in the model, not the loader, exactly as ADR-0003 assumes.

## 2. Architecture

### 2.1 Backbone: ViT-S/16 first, ViT-B/16 for the release run

- **ViT-S/16** (22M params, 384-dim, 6 heads, 12 blocks) is the iteration backbone. SSL4EO-S12 benchmarked exactly this scale on satellite data and found MAE the cheapest method to pretrain by far (7 h for 100 epochs on their setup, vs 24-25 h for MoCo/DINO) ([SSL4EO-S12](https://arxiv.org/abs/2211.07044)). MAE-Lite showed lightweight ViTs do benefit from MAE pretraining ([arXiv 2205.14443](https://arxiv.org/abs/2205.14443)).
- **ViT-B/16** (86M) is the release candidate once the recipe is proven: at this corpus scale the marginal cost is tens of dollars (§12), and B-scale is where the geospatial MAE literature (SatMAE, Scale-MAE at L; SpectralGPT, SelectiveMAE at B) actually operates, so results are more comparable. ViT-L stays out of scope.
- **Input:** 1×224×224, patch 16, 196 tokens. At 75% masking the encoder sees 49 tokens; this is the structural ~4x pretraining discount that makes the budget work ([MAE](https://arxiv.org/abs/2111.06377)).

### 2.2 Decoder: shallow and narrow

MAE's own ablation (Table 1a/1b) shows fine-tune accuracy is essentially flat across decoder depth 1-8 and width 128-1024; only linear probing rewards the big 8×512 default. The remote-sensing lineage goes further: Scale-MAE found depth 3 optimal (degrading at 8), CROMA found a 1-layer 512-dim decoder matched larger ones, SpectralGPT's ablation peaked at depth 4, and Clay ships depth 4 ([Scale-MAE](https://arxiv.org/abs/2212.14532), [CROMA](https://arxiv.org/abs/2311.00566), [SpectralGPT](https://arxiv.org/html/2311.07113), [Clay spec](https://clay-foundation.github.io/model/release-notes/specification.html)).

**Choice:** decoder depth 4, width 256, 8 heads for ViT-S (width scaled to ~2/3 of encoder dim, following MAE-Lite/small-scale practice); depth 4, width 512 for ViT-B. One optional ablation restores the 8×512 default to confirm nothing was lost. Sin-cos positional embeddings are re-added to decoder inputs (MAE default; also the Scale-MAE finding that decoder PE matters).

### 2.3 Positional encoding

Fixed 2D sin-cos (MAE default). GSD-aware positional encoding (Scale-MAE) is deliberately deferred to Phase 4: the corpus is single-GSD by construction (30 m, with the known ~1.55:1 high-latitude pixel anisotropy accepted in ADR-0001 and recorded per tile as `lat_band`), so GSDPE has nothing to condition on yet. The hook costs nothing now and the manifest already carries what it needs.

## 3. Masking

- **Baseline: 75%, uniform random, per-sample.** This is the optimum in the original MAE ablation (Fig. 5) and re-confirmed on RGB satellite imagery by Scale-MAE's ablation (70-80% indistinguishable, 75% chosen). Random beats block/grid sampling decisively (MAE Table 1f).
- **Planned ablation: 85% and 90%.** The evidence that smooth, low-entropy modalities prefer higher ratios is consistent and recent: SpectralGPT's ablation on Sentinel-2 rose monotonically 25→90% (84.64 mAP at 75% vs 85.61 at 90%), SelectiveMAE settled on 85% on a 4M-image optical corpus, and video MAE (extreme temporal redundancy) uses 90% ([SpectralGPT](https://arxiv.org/html/2311.07113), [SelectiveMAE](https://arxiv.org/html/2406.11933v1)). Per-tile-standardized elevation is plausibly the smoothest modality in this list, so 90% is a live candidate, and it is also *cheaper* per epoch (fewer encoder tokens). Decide on the kNN probe + val-loss evidence, not on reconstruction aesthetics. *(Settled 2026-08-05 — [ADR-0010](adr/0010-frozen-phase1-release-recipe.md), [#85](https://github.com/JStoschek/DEMAE/issues/85): see the amendment below.)*

**Amendment 2026-08-05 — the ratio is frozen at 75%, and the 85% arm does not run** ([ADR-0010](adr/0010-frozen-phase1-release-recipe.md) §2, evidence in [#83](https://github.com/JStoschek/DEMAE/issues/83)).

R2 ran 90% for 200 epochs against R1's 75% and **lost on all three probes** at the last epoch they share: `stratum_id` 0.3718 vs 0.3914 (~3.3 noise widths), `slope_mean` MAE 2.2466 vs 2.0344 (10.4% worse), `geomorphon_entropy` 0.25918 vs 0.25265 (2.6% worse) — on a byte-identical probe set, with the schedule asymmetry running in R2's favour (at epoch 180 R2's LR was ~3.3e-05 against R1's ~7.5e-04, so R2 had already collected an anneal R1 had not, and was still behind). **§12's gate for the 85% arm — "85% only if 90% ≈ 75%" — therefore does not open.**

The prediction above was wrong in an informative way: SpectralGPT's monotonic rise **does not transfer**, even though per-tile-standardized elevation is the smoother modality. Smoothness is not the axis that sets the optimum here.

**A below-75% arm is ruled out of Phase 1 on scope, not evidence.** It is affordable (~4.5 GPU-h, ~$2 — at 65% masking the encoder sees ~69 patches against 49, so it is *slower* than R1, not faster) and it was declined anyway: MAE's own sweep peaks at 75% and Scale-MAE found 70–80% indistinguishable, so the prior is low; and the banked finding does not get stronger for it. What the release therefore claims is a **bound, not an optimum**: the optimum is at or below 75%, 90% was ruled out by measurement, downward was not searched.
- Masking is applied after augmentation, inside the model, with a per-step generator so the pattern differs every epoch. Keep the mask count fixed (it is, at a fixed ratio) so `torch.compile` sees static shapes.

## 4. Loss and reconstruction target

**Base objective:** mean squared error between decoder prediction and the (normalized) target patch, **computed on masked patches only** (computing on all patches costs ~0.5% accuracy in MAE's ablation).

**Per-patch target normalization (`norm_pix_loss`) ON, with two DEM-specific guards:**

1. norm_pix improved MAE (+0.5 ft on ImageNet, Table 1d) and is ON in SatMAE and most of the lineage. For elevation it doubles as local detrending: each 480 m × 480 m patch is judged on its internal relief, not its offset within the tile, which is exactly the "learn terrain shape" objective. **But** the reference implementation divides by `(var + 1e-6)^0.5` per patch, and a near-flat 16×16 patch (playa, lake surface, residual ocean pixels) has variance near zero in per-tile-standardized units, amplifying quantization noise up to ~1000x into the loss. The corpus's 3 m degenerate screen operates per *tile*, not per patch, so flat patches absolutely survive. Guards: compute target stats in fp32 regardless of autocast, and **raise the variance floor to 1e-4** (in per-tile-standardized units). Log the per-patch loss distribution; if the top tail is dominated by near-flat patches, raise the floor before concluding anything about the model.
2. **Ablate norm_pix OFF** once (the small-scale MAE reproduction found it unhelpful below ImageNet scale ([ReScience](https://openreview.net/pdf?id=KXfjZPL5pqr)), and no one has measured it on elevation; this is a genuinely open question DEMAE can answer).

**Amendment 2026-08-05 — norm_pix stays ON, and ReScience's finding does not replicate on elevation** ([ADR-0010](adr/0010-frozen-phase1-release-recipe.md) §1, evidence in [#84](https://github.com/JStoschek/DEMAE/issues/84)).

R3 ran norm_pix OFF for 200 epochs and **lost**: `slope_mean` MAE 2.2680 vs 2.0344 (11.5% worse) and `geomorphon_entropy` 0.28843 vs 0.25265 (14.2% worse), with `stratum_id` a 0.67-noise-width tie. §4's own stated mechanism is what the numbers back — per-patch normalization is local detrending, judging each 480 m patch on internal relief rather than tile offset, and **the two probes that measure terrain shape are exactly the ones that degrade without it**, entropy most. So the genuinely-open question is answered in the affirmative for this modality at this scale.

Guard 1's escape hatch was **not needed**, and neither guard fires the way §13 anticipated: `tail_flat_frac` sits at ~0.0014 and *declines* through R1, with p99 settling ~1.49 by epoch 50, so the top tail is real terrain rather than near-flat patches, and §13's norm_pix-blowup row does not fire at 75%. R3 also confirms the guards are inert on its own path (no variance division with norm_pix off) and that its tail is *better*, not worse — `tail_flat_frac` 0.000333, p99 1.127, `collapsed_patch_frac` 6.8e-07. The fp32 stats and the 1e-4 floor stay, unexercised but cheap.

**Note for anyone comparing these two arms on val loss: don't.** R1's 0.4466 and R3's 0.09316 are not the same quantity — norm_pix OFF changes what the loss is computed over. See §10's amended cross-arm rule.

**Ocean / invalid pixels:** the loader ships `land_mask` (`raw != 0`, ADR-0003 §4) and kept windows are <75% ocean by construction. Extend MAE's patch-level loss mask to pixel weights inside patches: invalid pixels contribute zero, the patch loss renormalizes by valid-pixel count, and a masked patch with <25% valid pixels is excluded from the loss entirely. Per-patch target stats (norm_pix) likewise compute over valid pixels only. Precedent is thin here (the published pattern is curation, e.g. MAESSTRO drops any patch containing missing values ([MAESSTRO](https://os.copernicus.org/articles/20/1309/2024/))); this corpus already curated hard, so pixel weighting is a small correctness patch, not a novelty claim.

**Frequency-decomposed loss (Scale-MAE's low-freq L2 + high-freq L1) stays a Phase 3/4 ablation**, exactly as the project plan schedules it. It is well matched to terrain (low-freq = regional slope and basins, high-freq = drainage and breaklines) but it is not part of proving the vanilla recipe.

## 5. Input normalization

`per_tile` mode, the ADR-0003 default: `(x − mean)/std` with manifest stats. The `global` mode remains the scheduled Phase-3 ablation and its expected failure mechanism is already measured and recorded: under global normalization ~42x more variance sits between tiles than within them, so a pixel-MSE objective mostly learns to predict a tile's mean altitude (corpus card §4.8). Do not re-litigate this in Phase 1; do remember that a global-mode tile with std ≈ 0.02 is the data, not a loader bug.

One interaction to record in the training config: with `per_tile` input normalization *and* norm_pix targets, the model neither sees nor reconstructs absolute altitude anywhere. That is the intended Phase-1 design (terrain shape only); the manifest's `mean`/`std` invert reconstructions back to metres for visualization.

## 6. Optimization and hyperparameters

All MAE defaults unless argued otherwise; the deviations are batch size (hardware) and schedule length (corpus size).

| Hyperparameter | Value | Source / note |
|---|---|---|
| Optimizer | AdamW, betas (0.9, 0.95), `fused=True` | MAE appendix; fused is a free few-% step win |
| Weight decay | 0.05, excluded on bias/norm params | MAE default across the lineage |
| Base LR (blr) | 1.5e-4 | MAE default; linear scaling below |
| LR rule | lr = blr × eff_batch / 256 | MAE PRETRAIN.md |
| Physical batch | **1024 (ViT-S); 512 (ViT-B)** — measured, not hedged; see Amendment below | ViT-S fits comfortably in 32 GB at 49 visible tokens; ViT-B at 1024 does not fit at all |
| Grad accumulation | **2 (ViT-S); 4 (ViT-B)** → **effective batch 2048** either way → lr 1.2e-3 | `--accum_iter` is the official path; lightly's benchmark shows MAE trains fine even at 256, so eff. batch is not fragile |
| Warmup | 20 epochs, linear from 0 — **the rule is 5% of the schedule**, 20 is its output at 400 | 5% of schedule; MAE's 40/800 = 5%; small-scale reproduction used 20 |
| Decay | Cosine to 0 | MAE default |
| Epochs | **400**, checkpoints kept at 50/100/200/400; ~~extend to 800 only if the kNN probe and val loss are still improving at 400~~ *— unfollowable, Amendment below* | 400 ep × 500k = 2×10⁸ samples ≈ the sample budget of ~156 ImageNet epochs; mmpretrain's reproduction shows ft-quality saturates early (83.1 at 300 ep vs 83.5 at 1600 for ViT-B) while linear-probe keeps climbing; MIM data-scaling work shows ~500k images is comfortably enough for S/B models ([arXiv 2206.04664](https://arxiv.org/abs/2206.04664)) |
| Grad clipping | None by default (MAE doesn't); ~~add clip 1.0 only if bf16 instability appears~~ *— never fired, Amendment below* | MAE repo |
| Dropout / droppath | 0 in pretraining | MAE default |
| EMA | None | MAE uses no weight EMA; do not add one |

**Augmentation:** horizontal + vertical flips and 90° rotations (the full dihedral group). Overhead terrain has no canonical orientation, and the lineage keeps augmentation geometric-only; intensity jitter is both harmful (MAE Table 1e) and physically meaningless on standardized elevation. **No RandomResizedCrop in the baseline:** MAE's gain from RRC over fixed crops is small (84.9 vs 84.7), and rescaling changes the effective GSD of a modality where scale is physically meaningful; scale augmentation belongs with GSDPE in Phase 4. This is a deliberate deviation from the MAE default, recorded as such, and cheap to ablate if reconstructions look over-fit to a single texture scale.

**Seeds and determinism:** fixed seed for model init and the DataLoader generator (ADR-0003 reproducibility note), `dataset_version` + selection-list md5 + git SHA + full config logged into every checkpoint and into W&B. Two runs from the same config should differ only by nondeterministic kernels. *(Qualified by #95: that holds **within** a compile setting, not across one — inductor generates its own randomness, so a compiled and an eager run from one seed mask different tokens.)*

**Amendment 2026-08-05 — two hedges are dead, and the extension rule is replaced** ([ADR-0010](adr/0010-frozen-phase1-release-recipe.md) §1 and §5).

~~**Physical batch stays 1024 for ViT-B.** R0 ([#81](https://github.com/JStoschek/DEMAE/issues/81)) measured ViT-B at batch 1024 peaking at **17.82 GiB allocated / 19.03 GiB reserved against 31.35 GiB available** — barely half the card. The drop-to-512 contingency does not fire, so effective batch stays 2048, accumulation stays 2, and `lr` stays 1.20e-3 across both backbones.~~

**Amendment 2026-08-06 — ViT-B's physical batch is 512, and the paragraph above is retracted** ([ADR-0010](adr/0010-frozen-phase1-release-recipe.md) §1 Amendment 1, [#113](https://github.com/JStoschek/DEMAE/issues/113); found by R4 OOM'ing on its first step, [#86](https://github.com/JStoschek/DEMAE/issues/86)).

**ViT-B at batch 1024 does not fit the card.** Measured on the rented RTX 5090's 31.36 GiB, `RunConfig.vit_base()` at 1024 peaks at **30.53 GiB in forward and backward alone** and OOMs — with `torch.compile` on or off to two decimal places — and adding AdamW's two fp32 moments over 98.4M parameters (~0.79 GiB) puts its real requirement at ~31.3 GiB, so it had no path to fitting. **17.82 GiB is what batch 512 costs**: a full step at 512 × 4 measures 18.16 GiB, reproducing R0's cited figure to within 2%, while 1024 misses it by 12 GiB.

The number was real and attached to the wrong batch. R0's readout says both backbones ran *"at physical batch 1024 × 2 accum,"* but `RunConfig.vit_base()` still carried the 512 × 4 contingency when R0 ran, so R0's ViT-B memory *and* throughput figures are both 512 × 4 measurements under a 1024 × 2 label. The paragraph above read that mislabel as a stale hedge — correctly observing that the shipping config was one R0 had measured nothing at — and then moved the preset to match the label, which made its own objection true of 1024 instead. R0's checklist ([#81](https://github.com/JStoschek/DEMAE/issues/81)) had asked for exactly this: *"physical batch drops to 512 if 32 GB demands it — record which."*

**So the table above states values rather than a contingency.** 512 is not the "drop to 512 if needed" hedge revived — it is the only split that runs. Effective batch is 2048 on both backbones, so §6's linear rule derives `lr` 1.20e-3 for both and nothing downstream moves. **§12's budget stays on 3,001.2 samples/s**: that measurement was always of this split, and the 512 × 4 re-measurement at 3,130.2 is 4% higher, so re-basing it would only move the budget in the cheap direction. R4 ran 400 epochs at 512 × 4 by override ahead of this amendment, so the arms already agree and nothing is re-run.

**Grad clipping stays off.** The permission to add clip 1.0 was conditional on bf16 instability appearing. Four runs — R0, R1, R2, R3 — across two backbones, two mask ratios and two loss targets produced no loss spike and no §10.1 alarm. bf16 without a scaler is stable on this objective, as §7.1 predicted.

**Warmup is the 5% rule, not the number 20.** R2 ran 10 at a 200-epoch schedule and the `vit_base_continual` preset derives its own; inheriting 20 into a shorter schedule would make warmup a much larger fraction than the rule allows. §7's worked R2/R3 examples carried the mask-ratio flag but *not* the schedule length, and as written would have run 400 epochs — corrected below.

**"Extend to 800 on evidence at 400" is replaced by a pre-launch call, because the schedule cannot honour it.** [`schedule.py`](../src/pretrain/schedule.py)'s cosine reaches exactly 0.0 at the final epoch and is clamped to 0 beyond it. So at epoch 400 the probe and val curves have flattened **by construction** — the rule asks a question its own schedule guarantees the answer to — and training past 400 applies zero-sized updates. A genuine 800-epoch run is a *different run*, launched from step 1 with an 800-epoch cosine; an epoch-400 checkpoint drawn from an 800-epoch run is **not** the same artifact as a finished 400-epoch run, because it has not annealed, and R1 measured its own 200→400 gain as concentrated in exactly that tail.

R5's budget is therefore **400 by default, with the 400-vs-800 choice made before R5 launches**, on R4's ViT-B trajectory (R4 now runs the same 400-epoch schedule — §12). R1's measured doubling is the prior and argues 400: 200→400 bought val 0.0077 (1.7% of the epoch-50 value), `stratum_id` +0.014 (~2 noise widths), `slope_mean` −7.3%. The counter-argument is recorded rather than dismissed: 400 epochs × 473k tiles is ~148 ImageNet-epochs of samples against MAE's 800–1600, and this table's own citation has linear-probe quality climbing long after fine-tune quality saturates — which is the axis §10 adjudicates on. If 800 is chosen, §14's milestone list gains an 800 entry, which is an artifact-contract change rather than a config tweak.

## 7. Efficiency levers

In order of importance:

1. **bf16 autocast, no GradScaler.** Ada/Blackwell-native, removes the fp16 NaN failure mode the official repo documents. Loss and norm_pix statistics in fp32 (§4).
2. **SDPA / FlashAttention** via `F.scaled_dot_product_attention` (what timm blocks use). At 49-196 tokens the win is kernel fusion, not memory IO, and it is real but modest; no need for the separate flash-attn package.
3. **`torch.compile(dynamic=False)`** on the full model. Expect ~1.1-1.4x on a bf16 ViT; fixed mask count keeps shapes static and avoids recompiles.
4. **Fused AdamW** (above).
5. **The dataloader is the actual risk, not FLOPs.** DailyMAE showed baseline MAE pretraining is dataloader-bound and got 5.8x end-to-end by fixing IO ([DailyMAE](https://arxiv.org/html/2404.00509v1)). This project already paid that cost in Phase 0: memmap shards + the ADR-0003 loader are reasoned at 1-2 orders of magnitude over demand. Two ops notes: rent a box with **≥64 GB RAM** so the 47 GB corpus page-caches after epoch 1, and confirm measured samples/s ≥ 2× model demand in the first rented hour before starting the long run.
6. **Not used:** gradient checkpointing (unneeded at S/B scale, costs ~30%), channels_last (CNN optimization, irrelevant), the progressive-resolution schedule from DailyMAE (complexity not justified at this budget).

**Throughput planning numbers (estimates, to be measured in R0):** ViT-S/16 MAE at bf16+SDPA+compile on a 5090 should land in the 3,000-6,000 img/s range if the loader keeps up (extrapolated from DailyMAE's ~2,000 img/s/A100 for ViT-B and 4090/5090 scaling; no published 5090 MAE number exists). Planning uses the conservative end.

## 8. Initialization: scratch main run, one continual arm

- **Main/release run: from scratch.** The clean claim ("elevation-only, from scratch, reproducible") is part of the project's positioning, and the DEM prior art explicitly flags the ImageNet→DEM domain gap ([arXiv 2309.03367](https://arxiv.org/abs/2309.03367)).
- **One continual-pretraining arm, ViT-B only:** initialize the encoder from an ImageNet MAE ViT-B/16 checkpoint, collapse the patch-embed to one channel by **summing** the RGB kernels (timm's `adapt_input_conv` behavior: sum preserves activation scale; mean shrinks it 3x), fresh decoder, then run the same MAE recipe at 100-200 epochs. The case for the arm: GFM showed continual pretraining beats from-scratch at almost exactly this corpus scale (~600k images) at ~8x less compute ([GFM](https://arxiv.org/abs/2302.04476)), grayscale-ImageNet transfer helps single-channel medical targets ([Xie & Richmond](https://openaccess.thecvf.com/content_ECCVW_2018/papers/11134/Xie_Pre-training_on_Grayscale_ImageNet_Improves_Medical_Image_Classification_ECCVW_2018_paper.pdf)), and a frozen ImageNet MAE already beat a scratch UNet on DEM segmentation (2309.03367). The scratch-vs-continual comparison is itself a headline result of the release, whichever way it goes. (No ViT-S MAE checkpoint exists at any provenance — the official release and OpenMMLab's reproduction both stop at ViT-B/L/H — which is why this arm runs at B.)

  *(Qualified 2026-07-29 — [P101](research/p101-mae-checkpoint-licence.md), [#103](https://github.com/JStoschek/DEMAE/issues/103): the GFM citation is sound but this bullet's ordering invites a wrong inference. GFM does **not** continue from a masked-autoencoder init — its teacher is a **supervised ImageNet-22k Swin-B** and its objective is masked image modelling plus feature distillation from that frozen teacher, across a different architecture family. So the cited precedent is warm-starting a *masked* objective from a *supervised* init, not MAE→MAE. Read straight, that weakens any claim that an MAE init is what makes this arm "continual," and it is why §6.2 of P101 treats a permissively-licensed supervised init (DeiT-3, AugReg) as a defensible fallback rather than a consolation prize, should the checkpoint below prove unloadable.)*

- **Amendment 2026-07-29 — the checkpoint is OpenMMLab's Apache-2.0 reproduction, not the official release** ([#103](https://github.com/JStoschek/DEMAE/issues/103), evidence in [P101](research/p101-mae-checkpoint-licence.md)).

  R4 initializes from **`mae_vit-base-p16_8xb512-amp-coslr-1600e_in1k`** — [`…coslr-1600e_in1k_20220825-f7569ca2.pth`](https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-1600e_in1k/mae_vit-base-p16_8xb512-fp16-coslr-1600e_in1k_20220825-f7569ca2.pth), ~485 MB, **Apache-2.0** via `mmpretrain`/`mmselfsup`. The 1600-epoch schedule is chosen over the 800e one because it is closest to the official weights on the linear-probe axis (0.9 pt behind rather than 2.9) and costs nothing extra to fetch.

  **The official checkpoint is rejected on licence ambiguity, not on quality.** P101 established that the ambiguity is unresolvable rather than merely unresolved: Meta said only "this *project* is under the CC-BY-NC 4.0 license" and never scoped it, the three issues asking exactly that got no maintainer reply, the repo is archived, the CDN serves no terms, and the two redistributions of the same file disagree — with the permissive side of that disagreement traceable to a copy-paste from another model's card. Writing to Meta was considered as a fourth option and **declined**: this substitution removes the ambiguity rather than routing around it, so a reply would no longer change the release, and an unanswered enquiry is an open loop whose deadline is R4.

  **Accepted cost, stated rather than waved through.** OpenMMLab's reproduction matches the official weights on ImageNet fine-tuning to within 0.1–0.5 pt but trails on **linear probing**: 65.1 vs 68.0 at matched 800e, 67.1 vs 68.0 at 1600e. Probing is the axis §10 adjudicates with, so this is a confound on the measurement and not a free swap. It is accepted as **second-order** on the following argument: the published delta describes how well the *starting* weights classify ImageNet photographs, whereas R4-vs-R5 is adjudicated by a kNN probe on `stratum_id` after **150 further epochs** of elevation MAE training — roughly 75M tile presentations — and the effect R4 exists to detect is a scratch-vs-warm-start gap that GFM reports as large. A sub-point difference between two ImageNet MAE runs is small against that; and if the R4-vs-R5 gap is ever narrow enough for 0.9 pt to flip it, the honest readout was "no meaningful difference" regardless of provenance. **R4's reporting therefore carries a one-line caveat** (see #86): the init is an independent reproduction, so a *narrow* R4-vs-R5 gap must not be over-read in either direction.

  Should anyone later want to replace that argument with a measurement, the rule agreed in #103 is: probe both candidate inits frozen on val tiles at epoch zero and compute `(official − openmmlab) / (official − random)`, accepting the substitution at ≤ 0.15 — i.e. the reproduction captured ≥85% of what an ImageNet init buys on terrain features. If `official − random` comes back near zero the ratio is meaningless, and the fallback is the argument above. This was **not** made a gate: the measurement is compute-cheap but sits behind the training step, the monitoring layer and the probe harness, which is a poor trade for confirming a second-order effect.

  **Integration cost.** The file uses mmselfsup's parameter naming, so #102's loader needs an explicit key remap that the official file would not have required, and — at ~485 MB against the official ~343 MB — it ships encoder **and** decoder, of which the decoder is discarded (R4 attaches a fresh one either way). Because a remap can populate every name with correctly-shaped tensors that are nonetheless in the wrong order — fused-`qkv` convention, transposition, block ordering — #102 asserts **layout conventions read from the source implementation, never inferred from shapes**. An exact equivalence test against `mmpretrain` was considered and declined: `mmcv`'s compiled ops and version pins cost more than the bug they guard.

  *(Resolved 2026-07-29 by #102 — [P102](research/p102-mmselfsup-mae-layout.md) records the file's 256-entry inventory and one checked claim per layout convention. Two things came out of it that change this section rather than merely implementing it.)*

  **The ordering hole was real, not hypothetical.** mmselfsup's fixed sin-cos positional embedding is the **row/column transpose of MAE's** — `torch.meshgrid` defaults to `'ij'` where `np.meshgrid` defaults to `'xy'`, so each library's obvious code produces the other's mirror, and both name their variables after the wrong axis. Every parameter name and every shape matches under either convention; the difference is 2.0 in a tensor whose values span ±1. So `Architecture.pos_encoding` gains a second value, R4 sets it, and the loader **verifies the model's positional buffer against the checkpoint's own `pos_embed` tensor before copying any weight** — an oracle from the file's own contents that costs nothing and that #102 was written expecting not to have. Note the reason it is config-driven rather than patched after construction: our positional embeddings are non-persistent buffers rebuilt from the config, so a §11 resume on a fresh box would otherwise silently rebuild the wrong one and continue training under a convention the first leg had not used.

  **The behavioural check is weaker than this section promised.** Its power was measured rather than assumed: eight name-and-shape-preserving scrambles against six candidate statistics (P102 §6.1). The two statistics #102 proposed — natural image versus noise, and two crops of one image versus crops of two — have **no discriminating power**; six of the eight scrambles score a *higher* crop margin than the correct load, because 86%-overlapping crops are close under any smooth function and any network separates a smooth field from white noise. What does discriminate is the per-block residual-stream amplification, a 1.3%-stable fingerprint of the loaded weights, and it catches 2 of the 8. The gap is acceptable for a specific reason and not by concession: the six it misses are **unreachable by the loader**, which performs no permutation, transposition or reshape on any tensor, so its layout guarantee is that none is needed — which is what the claim table asserts, and what a 149-pair unit test of the key map pins. This sentence's original promise, "a dependency-free behavioural check that a scrambled encoder cannot pass", is not something anyone should claim without measuring it.

- GFM-style feature distillation from a frozen ImageNet teacher is out of scope (it ties the student to a 3-channel teacher and adds a second model to the budget); plain warm-start + continual MAE is the arm.

## 9. Splits: fill the `split` column, never split randomly

The corpus card is unambiguous (§4.6): 423 source COGs were drawn dry and 2,788 more past half, so a random split puts adjacent windows on both sides and validation reads better than it should. The manifest ships `split` declared-and-null with `geocell_id` and `source_cog` for exactly this purpose. Phase 1 fills it once, commits the assignment, and every later phase inherits it.

**Design (three-way, geographically blocked):**

- **`test` (~3%): whole held-out 10° geocells.** Select 12-15 of the 425 occupied geocells, chosen to span distinct geomorphic provinces and stratified so all 20 strata appear in the held-out pool; every tile inside them goes to `test`. These regions are untouched by pretraining and reserved as the Phase-2 downstream evaluation regions, which is what makes later "generalizes to unseen regions" claims honest. Blocking at the 10° geocell scale bounds the *typical* separation between train and test; the spatial-CV literature is unanimous that the buffer must exceed the autocorrelation range (Roberts et al. 2017; [Ploton et al. 2020](https://www.nature.com/articles/s41467-020-18321-y); [Karasiak et al.](https://www.sciencedirect.com/science/article/pii/S2667393222000072)).
  - **Amended 2026-08-10 (#88, measuring [#77](https://github.com/JStoschek/DEMAE/issues/77)).** This bullet said blocking "puts hundreds of km between train and test, far beyond terrain autocorrelation ranges". [P77 §3](research/p77-split.md) measured the realized split and that is the median's story, not the edge's: tile-to-nearest-train-tile distance has a **median of 97 km** and a p95 of 348 km, but a **minimum of 3.0 km**, with 3.0% of test tiles inside 10 km of training data and **12.8% inside 25 km**. A whole-cell hold-out has a perimeter. The design is unchanged and the claim is narrowed: the guarantee is strong for the interior and absent at the rim, so a Phase-2 claim built here either spends the buffer — dropping train tiles within 10 km costs 2,964 (0.62% of train), within 25 km 12,931 (2.72%) — or reports the rim separately. Amended here rather than left in the memo because the release's evaluation-regions contract was generated from *this sentence* and shipped the uncorrected version to consumers.
- **`val` (~2%, ~10k tiles): held-out whole source COGs** (1° tiles, ~110 km), sampled across strata and drawn from geocells *not* in `test`. Purpose: masked-reconstruction validation loss and probe evaluation during pretraining. COG-level blocking is the minimum honest unit per the corpus card; going coarser than COG here would spend geographic coverage the train set needs.
- **`train` (~95%): everything else.**

**Recorded caveats:** (a) SSL practice trains on ~100% and monitors nothing (MAE, Scale-MAE, SatMAE all pretrain on the full train split with no held-out val), so the 2% val is cheap insurance justified by pretraining-val-loss being a validated predictor of downstream quality ([arXiv 2206.04664](https://arxiv.org/abs/2206.04664)), not a field norm; the exact 95/2/3 ratio is a judgment call with no literature basis. (b) Removing test geocells slightly perturbs the realized Neyman allocation; record the per-stratum train counts next to the split assignment. (c) The shipped global μ/σ (955.67/1012.16) was accumulated over all tiles including val/test; for the Phase-3 global-norm ablation either recompute on `train` only (MAESSTRO's train-split-only practice) or record the ~2-5% contamination as accepted. (d) The split assignment is a committed artifact (same contract philosophy as the selection list): a CSV/parquet of `tile_id, split` plus the geocell/COG lists, with its md5 in the run config.

## 10. Monitoring, evaluation, and failure detection

MAE gives few in-flight signals; build the ones that exist:

1. **Train loss + val loss every epoch.** Val uses a **fixed masking seed** so the number is comparable across epochs. Expect fast warmup drop then slow decay for hundreds of epochs. Alarm conditions: sudden collapse toward zero (a known symptom of a broken norm_pix target, [transformers issue 29416](https://github.com/huggingface/transformers/issues/29416)), val loss rising while train falls (overfitting; plausible at 500k non-independent tiles, see corpus card §4.6 on effective dataset size), or a plateau at high loss in the first 20 epochs (LR/normalization bug).
2. **Reconstruction grids every 10 epochs** on a fixed panel: 2 val tiles from each of the 20 strata (40 tiles), rendered masked → predicted → truth, inverted to metres via manifest stats, plus a hillshade render of each (terrain artifacts that are invisible in heatmaps are obvious in hillshade). The strata are the project's own terrain families; a model that reconstructs alpine ridges but smears low-relief drainage will show it here first, and that specific failure is the one the corpus card predicts (§3.1).
3. **kNN probe every 20 epochs, the cheap proxy for representation quality:** frozen encoder, CLS/mean-pooled features on ~5k val tiles, kNN classification of `stratum_id` (20 classes, free labels the manifest already carries) plus kNN regression of `slope_mean` and `geomorphon_entropy`. Cost is one inference pass, minutes. Caveat from SSL4EO: MAE representations look weak under linear/kNN probes relative to contrastive methods even when fine-tuning wins, so use probe *trends* to compare checkpoints and ablation arms, never as an absolute quality bar.
4. **Throughput dashboard:** samples/s, step time, dataloader wait fraction. If loader wait exceeds ~10%, stop and fix IO before burning GPU hours (the DailyMAE lesson, and the plan's #1 named risk).
5. **What is deliberately absent:** EMA (not part of MAE), online linear probe with labels (no labels), FID-style metrics (meaningless here).

**Decision rule for the 400→800 epoch extension and for ablation winners:** kNN-probe trend on `stratum_id` + val recon loss, jointly. Reconstruction prettiness is explicitly not a criterion (loss down does not equal representations good). *(Amended 2026-08-05 — [ADR-0010](adr/0010-frozen-phase1-release-recipe.md) §3, [#85](https://github.com/JStoschek/DEMAE/issues/85): this one sentence was doing two different jobs and gets R3 wrong. Replaced below; the last clause survives unchanged.)*

**Amendment 2026-08-05 — the rule splits in two** ([ADR-0010](adr/0010-frozen-phase1-release-recipe.md) §3, evidence in [#82](https://github.com/JStoschek/DEMAE/issues/82) / [#83](https://github.com/JStoschek/DEMAE/issues/83) / [#84](https://github.com/JStoschek/DEMAE/issues/84)).

Read literally, the rule above gets R3 wrong. `stratum_id` came in at 0.3874 against R1's 0.3914 — **0.67 noise widths, a tie** — and val loss at 0.09316 against 0.4466, which is **not a comparison at all**, because norm_pix OFF changes what the loss is computed over. Both named instruments report "no difference" on an arm that lost by 11.5% and 14.2% on the metrics this sentence lists *second*.

It is not a quirk of one arm. **`stratum_id` takes ~92% of its total gain by epoch 40 in all three runs** (92.0 / 92.3 / 90.9%), so the saturation is a property of the probe rather than of any run, while both regression probes improve to the end of every run. R2 bounds the correction: there the categorical probe *did* separate the arms, at 3.3 noise widths — just least, 5.0% relative against `slope_mean`'s 10.4%. So `stratum_id` is not demoted; it stops being the lead.

- **Across arms** (ablation winners, R4-vs-R5): **all three probes read jointly at a matched epoch on a byte-identical probe set** (`probe/set_md5`), with no lead metric. **Val loss is admissible only when the loss target and mask ratio are identical** across the arms compared — which excludes it from R2-vs-R1 (0.5878 vs 0.4466 is task difficulty: reconstructing 90% from 10% visible is strictly harder) and from R3-vs-R1 (different loss target), and admits it for R4-vs-R5.
- **Within one arm** (the epoch-budget call): **val loss stays valid**, exactly as §10.1 always claimed — the fixed masking seed makes it comparable across epochs of one run — read alongside the probe trends.

**When the categorical and regression probes disagree, the disagreement goes in the readout** rather than being resolved silently. That is the whole operational content of dropping the lead metric: R3 is what a silent resolution looks like, and it would have shipped as "no difference".

Two corollaries are now load-bearing rather than incidental: arms being compared must **share probe epochs** (the probe fires only on multiples of `probe_every_epochs`), and must be scored on the **same probe set**. R1–R3 satisfied both by luck; §12's R4 schedule change makes the first structural.

Reconstruction prettiness remains not a criterion, unchanged.

## 11. Ops on vast.ai (interruptible reality)

- **Instance:** single RTX 5090, on-demand (~$0.31/hr as of 2026-07-28; spot ~$0.20 but pausable), verified datacenter host, ≥64 GB RAM, NVMe with ≥100 GB free, good download bandwidth for the one-time 47 GB corpus pull. The reliability tax on unverified/spot hosts is real (documented instance deaths at ~9 h, advertised bandwidth not delivered); for a multi-day run, on-demand + verified is worth the ~50% premium. A 4090 fallback changes nothing in the plan except wall-clock (~0.5-0.7x throughput).
- **Corpus transfer:** shards + manifest to the box once, checksum against `dataset_version` before training (the manifest's `content_hash` machinery exists for exactly this).
- **Checkpointing:** every epoch *and* every 30-60 minutes of wall-clock, atomic write (tmp + rename), keep last 3 + the milestone set (50/100/200/400). Every checkpoint carries model, optimizer, epoch/step, RNG states, and the full config + data fingerprints. **Sync each checkpoint off-instance** (Backblaze B2/R2/S3; ~$1/mo at this size) because vast.ai instances can vanish permanently, not just pause. Resume must be exact (tested in the smoke run by kill-and-resume, matching loss curves).
- **W&B (or equivalent) for all metrics and the recon grids**, so the box holds nothing unique except the latest shard of checkpoints.

## 12. Run schedule and budget

Throughput is measured in R0; the table plans on the conservative 3,000 img/s (ViT-S) / 1,300 img/s (ViT-B) and on-demand $0.31/hr. GPU dollars are small; wall-clock and attention are the real budget.

| Run | Config | Epochs | Est. GPU-h | Est. cost | Purpose / gate |
|---|---|---|---|---|---|
| R0 | Smoke: ViT-S, 1-2k steps + kill/resume + throughput profile | ~1 | 2 | ~$1 | Loader ≥2x demand; resume exact; grids render; go/no-go for R1 |
| R1 | **ViT-S/16 scratch baseline** (75%, norm_pix on, 4×256 decoder) | 400 | ~19 | ~$6 | The recipe-proving run; first shareable reconstructions |
| R2 | Mask ratio 90% (else = R1) | 200 | ~7 | ~$2 | The highest-prior ablation for smooth terrain; 85% only if 90% ≈ 75% |
| R3 | norm_pix OFF (else = R1) | 200 | ~9 | ~$3 | The open elevation-specific question |
| R4 | ViT-B continual from ImageNet MAE (patch-embed summed) | 150 | ~17 | ~$6 | The GFM-style arm; compare to R5 at equal probe cost |
| R5 | **ViT-B/16 release run**, winning recipe from R1-R3 | 400-800 | 43-86 | $13-27 | The released checkpoint; extend past 400 only on probe evidence |
| | Slack: reruns, IO debugging, instance churn (~40%) | | | ~$20 | Spot-GPU reality |
| | **Total** | | | **~$50-70** | Well inside the $100-300 envelope; leaves Phase 2+ funded |

Sequencing: R0 → R1 → (R2, R3 in either order, can share the box) → decide recipe → R4 and R5. R2/R3 read out at 200 epochs deliberately: ablation *ordering* is stable early even when absolute numbers keep improving.

**Amendment 2026-08-05 — the table is measured, R4 goes to 400 epochs, and R5 is sequenced behind it** ([ADR-0010](adr/0010-frozen-phase1-release-recipe.md) §4, [#85](https://github.com/JStoschek/DEMAE/issues/85)).

Everything above the R4 row is now measurement rather than plan, on the rented 5090 at its real **$0.4016/hr** (not the ~$0.31 assumed). The planning rates were conservative by roughly 2.5×:

| Run | Epochs | Measured samples/s | Loader wait | GPU-h | Cost | vs planned |
|---|---|---|---|---|---|---|
| R0 | smoke | 7,445.6 (S) / 3,001.2 (B) | 0.4% / 0.1% | ~2 | ~$1 | as planned |
| R1 | 400 | 7,291.5 | 2.3% | 7.2 | **$2.90** | ~$6 |
| R2 | 200 | 9,609.2 | 2.0% | 2.74 | **~$1.10** | ~$2 |
| R3 | 200 | 7,353.0 | 1.6% | 3.57 | **~$1.44** | ~$3 |
| **R4** | **400** *(was 150)* | 3,001.2 (B, measured in R0) | — | ~17.5 | ~$7.00 | ~$6 at 150 |
| **R5** | 400 *(800 = ~35 GPU-h, ~$14.05)* | 3,001.2 (B, measured in R0) | — | ~17.5 | ~$7.00 | $13–27 |

**§7.5's named #1 risk — the dataloader — did not fire on any run**, clearing the ~10% alarm by an order of magnitude at every scale and duration tried, so ADR-0003's headroom argument holds on this host rather than merely in argument.

**R4 moves from 150 to 400 epochs, adopting R5's schedule exactly** (warmup 20 under §6's 5% rule). Reason, in full in [ADR-0010](adr/0010-frozen-phase1-release-recipe.md) §4: under §10's amended cross-arm rule, R4 at 150 against R5 at 400 has no shared final probe epoch (the probe fires on multiples of 20, so R4's endpoint is unmeasured and the last shared epoch is 140), and at epoch 140 R4 would be 93% through its cosine against R5's 35% — **carrying an anneal advantage that has nothing to do with the warm start, in the direction §8's hypothesis predicts**. R2-vs-R1 had the same asymmetry running the *other* way, which is what made it dismissible there; a confound that flatters the hypothesis is not. The fix costs ~11 GPU-h and ~$4.35, makes every 20-epoch probe a valid paired reading rather than one confounded point, and makes the arm's stated premise — hold the recipe fixed, move only the initialization — literally true. GFM's "8× less compute" efficiency claim still reads off the same pair: R4's epoch-150 point against R5's epoch-400 point.

**"Compare to R5 at equal probe cost" in the R4 row above is superseded by that** — equal *schedule*, with the equal-compute reading available as a second cut.

**Sequencing becomes R0 → R1 → (R2, R3) → decide recipe → R4 → R5**, as a tracker blocking edge rather than a convention. R4 is the same recipe at the same scale for a quarter of R5's cost, so it is a near-free canary for a recipe never yet trained to length at ViT-B, and it is the evidence R5's 400-vs-800 call needs (§6).

**R4's only early kill is a §10.1 alarm** — loss collapse, val loss rising while train falls, or a first-20-epoch plateau. A flat probe curve mid-cosine is *not* a kill signal: at epoch 200 of a 400-epoch schedule the LR is still 54% of peak (6.5e-4 of 1.20e-3), R1's 200→400 gains lived in the tail, and a run stopped mid-cosine yields an unannealed checkpoint that is neither comparable to R5 nor releasable. The second half costs 8.75 GPU-h and ~$3.50.

## 13. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Dataloader starves the 5090 (500k = 47 GB, > page cache on small-RAM hosts) | Medium | ≥64 GB RAM host; measured samples/s gate in R0; ADR-0003 headroom argument re-verified empirically before R1 |
| norm_pix blowup on near-flat patches | Medium | fp32 stats, variance floor 1e-4, per-patch loss distribution logged (§4); R3 is the escape hatch |
| bf16 instability / loss spikes | Low | bf16 (not fp16), no scaler; add grad-clip 1.0 if seen; MAE is a stable objective |
| Overfitting: 500k non-independent tiles < ImageNet scale | Medium | Val split exists (unlike field norm) and is watched; effective-size caveat is corpus card §4.6; stop extension at 400 if val turns |
| Spot/instance loss mid-run | High on spot | On-demand verified host for long runs; 30-60 min off-instance checkpoints; exact resume tested in R0 |
| Probe misleads (MAE's known weak linear/kNN showing) | Medium | Probes compare arms/checkpoints only; final adjudication is Phase-2 fine-tuning (SSL4EO finding) |
| Low-relief drainage reconstructed poorly | Known, pre-measured | Corpus card §3.1 + ADR-0008 revisit trigger; the stratified recon grid (§10.2) is built to catch exactly this; if a Phase-2 probe confirms it, ADR-0008 reopens |
| Scooped by a new DEM FM | Realised in part, absorbed | Re-check 2026-07-29 ([P100](research/p100-scoop-recheck.md), #100) found TerraMind's per-modality **DEM tokenizer** ([`ibm-esa-geospatial/TerraMind-1.0-Tokenizer-DEM`](https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-Tokenizer-DEM)) is an Apache-2.0 ViT-B/16 pretrained on ~9M single-channel Copernicus DEM patches, public since 2025-02-20 — so **"first open encoder on elevation alone" is retired**. The 2026-07-28 pass read TerraMind's multimodal *backbone* correctly and missed that it ships seven per-modality tokenizers, one DEM-only. What holds: first **masked-reconstruction** encoder on **raw single-channel** elevation, released and **evaluated as a transferable backbone**, on a **terrain-stratified** corpus (MIM ≠ FSQ-VAE; IBM publishes no probe or transfer number for its tokenizer; TerraMesh samples the 10 m Sentinel grid, not terrain). Related work must name TerraMind-Tokenizer-DEM, DINO-CV, `terrain-diffusion-30m`. Re-check before release — via **full-text card search and registry source-reading**, never model-ID search (HF `?search=dem` returns `demucs`; that is how this was missed). |

## 14. Interface to Phase 2 (out of scope here, pinned now)

- Released artifacts per run: encoder-only weights (timm-loadable) + full MAE checkpoint + config + `dataset_version`/split-assignment fingerprints + **init provenance** + the training curves.
- **Init provenance is a required field on every released run** *(added 2026-07-29 — [#103](https://github.com/JStoschek/DEMAE/issues/103))*: what the encoder was initialized from, and under what licence. `scratch` for R1/R2/R3/R5; for R4, the OpenMMLab checkpoint URL and Apache-2.0 (§8). "What was this initialized from" is the same class of fact as `dataset_version` and the split md5, was previously unrecorded, and is the first thing a consumer auditing the release's provenance will look for. Under the §8 amendment every artifact ships under the **same** terms, so no run needs a separate-licence carve-out — but the field is required regardless of provenance, precisely so that a future arm that *does* differ cannot ship silently.
- **The licence the weights ship under is a required field on every released run** *(added 2026-08-10 — #88)*: the SPDX identifier, the full text beside the weights, and what it was inherited from. The chain was settled by [#103](https://github.com/JStoschek/DEMAE/issues/103) — an Apache-2.0 init means one set of terms across every arm — and the first export still shipped without it, naming every *neighbour's* licence in related work and the *init's* licence under provenance while saying nothing about the files in the directory. It is the first thing a consumer needs and the last thing they can infer. Released weights are **Apache-2.0**; the repo's source is MIT and is a separate grant. A run whose init licence is unverified or incompatible gets **no claim and no licence file** rather than a default, on the same "recognized, never assumed" rule as init provenance.
- The `test` geocells (§9) are the downstream evaluation regions; Phase-2 task datasets (voids, geomorphon labels) are built there, never from `train` geocells.
- Milestone checkpoints (50/100/200/400) are kept so Phase 2 can plot downstream-performance-vs-pretraining-compute, the cheapest interesting plot the field rarely reports.
- The release carries a **positioning line** naming the nearest open elevation-pretrained artifacts and what differs — TerraMind-Tokenizer-DEM first (§13, [P100](research/p100-scoop-recheck.md)). Shipping "first open encoder on elevation" beside a better-resourced IBM/ESA artifact doing the same thing is a credibility problem; naming it and stating the difference is a stronger claim precisely because it is checkable. The head-to-head against TerraMind's ViT-B/16 is a Phase-2 benchmark, not a Phase-1 deliverable — say so rather than leave the gap silent.

## 15. Pre-flight checklist (gate for R1)

1. Corpus on box, `dataset_version` verified, manifest loads, loader benchmark ≥2x model demand.
2. Split column filled and committed; train/val/test counts per stratum recorded; no test geocell in any train batch (asserted in the Dataset).
3. Fixed val panel (40 tiles, 2 per stratum) and fixed val mask seed committed.
4. Kill-and-resume produces bit-identical loss trajectory over ≥50 steps.
5. Recon grid + hillshade render pipeline works on the untrained model (garbage in, garbage rendered).
6. Per-patch loss histogram logged; flat-patch guard verified on a synthetic flat tile.
7. W&B run tagged with git SHA, config hash, selection-list md5, split md5.
8. Off-instance checkpoint sync verified round-trip.

## 16. Key sources

Recipe: [MAE (He et al., CVPR 2022)](https://arxiv.org/abs/2111.06377) · [official repo + PRETRAIN.md](https://github.com/facebookresearch/mae) · [MAE-Lite](https://arxiv.org/abs/2205.14443) · [ReScience small-scale MAE](https://openreview.net/pdf?id=KXfjZPL5pqr) · [DailyMAE](https://arxiv.org/html/2404.00509v1) · [MIM data scaling](https://arxiv.org/abs/2206.04664)
Geospatial lineage: [SatMAE](https://arxiv.org/abs/2207.08051) · [SatMAE++](https://arxiv.org/abs/2403.05419) · [Scale-MAE](https://arxiv.org/abs/2212.14532) · [Prithvi-EO-2.0](https://arxiv.org/abs/2412.02732) · [GFM continual pretraining](https://arxiv.org/abs/2302.04476) · [SSL4EO-S12](https://arxiv.org/abs/2211.07044) · [SpectralGPT](https://arxiv.org/html/2311.07113) · [SelectiveMAE](https://arxiv.org/html/2406.11933v1) · [CROMA](https://arxiv.org/abs/2311.00566) · [Clay](https://clay-foundation.github.io/model/release-notes/specification.html)
DEM prior art (nearest first — see §13 and [P100](research/p100-scoop-recheck.md)): [TerraMind-1.0-Tokenizer-DEM](https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-Tokenizer-DEM) (**the nearest artifact**: Apache-2.0 ViT-B/16, ~9M single-channel Copernicus DEM patches from [TerraMesh](https://huggingface.co/datasets/ibm-esa-geospatial/TerraMesh), public 2025-02-20; FSQ-VAE not MIM, and no published probe or transfer number — [TerraMind paper](https://arxiv.org/abs/2504.11171), [TerraTorch registration](https://github.com/IBM/terratorch/blob/main/terratorch/models/backbones/terramind/tokenizer/tokenizer_register.py)) · [DINO-CV](https://arxiv.org/abs/2510.17644) (CC BY-NC; DINO on DEM *derivatives* — hillshade/VAT — over one 301 km² site) · [terrain-diffusion-30m](https://huggingface.co/xandergos/terrain-diffusion-30m) (MIT; generative, but trained on GLO-30 itself, so it retires "first open model on GLO-30 elevation") · [GeomorPM](https://www.tandfonline.com/doi/full/10.1080/13658816.2024.2414409) (closed) · [MAE-on-DEM segmentation](https://arxiv.org/abs/2309.03367) · [EvaMAE](https://dl.acm.org/doi/10.1145/3748636.3762739) (DEM+RGB; repo ships no weights despite the abstract) · [Copernicus-FM](https://arxiv.org/abs/2503.11849) (no DEM-only checkpoint) · [MAESSTRO](https://os.copernicus.org/articles/20/1309/2024/) (SST, not land — a recipe precedent, not modality prior art) · [Xie & Richmond grayscale transfer](https://openaccess.thecvf.com/content_ECCVW_2018/papers/11134/Xie_Pre-training_on_Grayscale_ImageNet_Improves_Medical_Image_Classification_ECCVW_2018_paper.pdf)
Spatial validation: Roberts et al. 2017 · [Ploton et al. 2020](https://www.nature.com/articles/s41467-020-18321-y) · [Karasiak et al. 2022](https://www.sciencedirect.com/science/article/pii/S2667393222000072) · [spatial-CV buffer sizing](https://arxiv.org/pdf/2303.07334)
Repo grounding: [ADR-0002](adr/0002-storage-format-and-manifest.md) · [ADR-0003](adr/0003-dataloader-design.md) · [ADR-0005](adr/0005-sampling-design.md) · [ADR-0008](adr/0008-accept-f3-drainage-shortfall-strata-2-17.md) · [corpus card](research/f-honesty-notes.md)
