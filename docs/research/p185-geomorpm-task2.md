# P185: can any Task 2 number stand beside GeomorPM's?

**Ticket:** [#185](https://github.com/JStoschek/DEMAE/issues/185), part of the Task 2 map [#183](https://github.com/JStoschek/DEMAE/issues/183). **Builds on:** [`p124-voidfill-comparability.md`](p124-voidfill-comparability.md) §5 and [#128](https://github.com/JStoschek/DEMAE/issues/128)'s resolution, which read the Figshare deposit for Task 1. **Amends:** [`task2-landform-plan.md`](task2-landform-plan.md) §11.

## 1. The answer

There are two ways a Task 2 number could sit beside GeomorPM's. One is to cite their published landform result. The other is to run their released pretrained encoder under our frozen probes.

**Citing their number is closed.** No numeric landform result of theirs could be read anywhere (§3). Even if one turns up, it would not share a column with ours. Their task has five classes decoded from what is very probably China's 1:1,000,000 geomorphological map, a regional and coarse label. Ours has ten geomorphon classes at L = 50 on global GLO-30 tiles. Their number also comes from a full fine-tune that uses the whole VQ-GAN decoder as its head, and it is a per-tile-averaged mIoU on a regional test set. Plan §11's "task-level comparability" citation is wrong and is replaced (§7).

**Running their encoder as a frozen arm is technically viable, with known costs.** The released `vqmae.ckpt` holds a complete frozen extractor: a VQ-GAN f4 conv stem and then a ViT-L/16-equivalent MAE encoder. Its token grid is **14 × 14, the same as ours**, with 1024-d tokens, a CLS token, and its own fixed sin-cos position table at that grid. All 24 blocks are reachable. Nothing makes an ADR-0012 extractor impossible. Three things must change or be built first (§5.4):

- ADR-0012's contract-wide constant of twelve layers refuses a 24-block encoder.
- The released `forward_encoder` silently shuffles token order, so the extractor has to bypass it.
- The checkpoint is a 4.66 GB Lightning file. It needs a two-stage loader.

The confounds to report are a capacity gap of 3.8× in parameters and about 11× in per-tile compute against the DEMAE ViT-B, an ImageNet-MAE ViT-L initialisation, and a pretraining corpus nobody can inspect. **Whether to add the arm is not decided here.** It goes to the arms rewrite, [#190](https://github.com/JStoschek/DEMAE/issues/190).

## 2. Method and what was read

Everything in §4 to §6 comes from primary artifacts. Three kinds were read:

- **The deposit's code, configs and README.** All 19 files under 100 KB, fetched through the Figshare API (`api.figshare.com/v2/articles/25756806`, retrieved 2026-10-15).
- **The two checkpoints' metadata, without downloading their weights.** A torch checkpoint is a ZIP whose first member is `archive/data.pkl`, the pickled tree of key names and tensor shapes. The first 1 MiB of each file was fetched by HTTP range request. The pickle was then read by an unpickler whose `find_class` returns inert stubs, so no code from the file ran. This yields every key, every shape and the Lightning loop counters, but no weight values.
- **`Areas.zip`'s central directory.** It was read the same way #128 read it, by range requests on the ZIP64 end records: 10.07 MB of directory for 85,565 entries, and no member data.

The md5 pins come from Figshare's own `computed_md5` field.

## 3. Their landform numbers: not found, and where we looked

| Where | Result |
|---|---|
| Taylor & Francis article page and supplementary-material URL (`/doi/suppl/10.1080/13658816.2024.2414409`) | HTTP 403 to fetches. A Cloudflare bot check in the browser, which was not bypassed. |
| OpenAlex | `is_oa: false`, `oa_status: closed`, `any_repository_has_fulltext: false` |
| IGSNRR/CAS publication database ([record](https://igsnrr.cas.cn/sourcedb/zw/lw/202606/t20260604_8214383.html)) | Title, authors and abstract only. No numbers and no full text. |
| Semantic Scholar API | HTTP 429 (rate-limited). p124 had already read the abstract there. |
| ResearchGate | HTTP 403 |
| Figshare search for any other deposit by the author (ORCID 0000-0003-3094-4291) holding landform labels | none |
| Chinese-language web search (`GeomorPM 地貌 预训练模型 DEM 地貌分类`) | No Chinese-language follow-up paper by the GeomorPM group. Results returned only the IJGIS paper and the 2023 predecessor below. |
| The 10 works OpenAlex lists as citing GeomorPM | Four are open access. The two most likely to quote a number, *ESPL* doi:10.1002/esp.70295 and SSRN 6546301, returned 403 or a Cloudflare check. None of the titles is a landform reproduction. |
| `Areas.zip` (all 85,565 entries) | **No landform data at all.** There are no `LF/` directory, no label rasters and no `LF` split files. The archive holds `Datas/{8,9,10,13,14}/{30,90,120,240}` DEM tiles, `VF/`, `SR/` and `mask/`. This confirms p124 §5.5 against the full listing rather than a partial read. |

The only quantitative signal outside the paywall is still p124 §5.3's trailing comment in `dem_VF_vqmae_finetune.py`, which belongs to void filling. Nothing landform-shaped exists.

## 4. The label source: very probably China's 1:1,000,000 geomorphological map

**The predecessor paper names it.** Yang, Xu, Lv, Zhou, Zhu & Cheng, "Deep learning-based automated terrain classification using high-resolution DEM data", *Int. J. Appl. Earth Obs. Geoinf.* 118, 103249 (2023), doi:[10.1016/j.jag.2023.103249](https://doi.org/10.1016/j.jag.2023.103249), CC BY-NC-ND 4.0 per Crossref. It has the same first author, the same lab (LREIS/IGSNRR) and the same DEM product (AW3D30). Its abstract, reconstructed from OpenAlex's inverted index, says: *"We used the 1:1,000,000 Chinese landform map as the label."* Its task was classifying "elementary landform types" with an FCN-ResNet. ScienceDirect returned 403 to both the fetch and the browser, so its class count and numbers were not read. A search snippet attributes PA 75.24 and mIoU 63.15 to it. That figure is unverified, belongs to the 2023 paper rather than GeomorPM, and must not be cited.

**GeomorPM's loader fits a rasterised digit-coded map.** `dem_LF_dataloader.py` reads an integer raster and decodes it as follows:

```python
label[label < 100] = 290; label[label >= 255] = 290   # out-of-range values -> ignored
label = (label // 10) % 10                            # tens digit is the class
label[label == 5] = 4; label[label > 5] = 255         # 0-4 kept, 5 merged into 4, 6-9 ignored
```

The class codes are three digits from 100 to 254, and the class is the tens digit. A hierarchical numeric type code is what a rasterised Chinese 1:1M geomorphological map would carry. That map's "basic morphological types" combine a relief class (plain, platform, hill, then mountains by relief amplitude) with an altitude class, under the classification system of Zhou, Cheng, Qian, Li & Zhang, *J. Geo-inf. Sci.* 11(6), 707–724 (2009). **Which digit encodes relief and which encodes altitude was not confirmed.** That paper's page renders no abstract, and its PDF was not read. So the five classes cannot be named.

**Verdict on identity.** The source is the Chinese 1:1M geomorphological map, a **China-only, regional** product. The evidence is the predecessor paper plus the coding fit. It is not confirmed from GeomorPM itself. Its 1:1M cartographic generalisation is far coarser than a 30 m pixel. That is a different kind of label from geomorphons, which are computed per pixel from the DEM.

**Two more facts from the fine-tune code bear on any comparison:**

- **Their landform tiles are almost certainly 448 px, and the pretrained stem is discarded.** `LFModel.__init__` loads `vqmae.ckpt` and *then* replaces `encoder.conv_in` with a freshly initialised `Conv2d(3, 128, 3, stride=2)`. It also replaces the decoder's last layer with a ×2 bilinear upsample and a `Conv2d(128, 5, 1)`. With an extra stride of 2 before the f4 stem, the MAE's fixed `img_size=56` only fits a 448 px input. This is inferred from the shapes, because no landform tile exists to check against. So GeomorPM's own landform result does not run its pretrained first conv.
- **Their metric differs from ours in three ways.** It is computed per tile and averaged over tiles: `mean_iou` returns one tile's mIoU over the classes present in that tile, and `test_step` averages those. It excludes an 18 px border. Training uses focal loss with α = 1 and γ = 2, and the head is the full 33 M-parameter VQ-GAN decoder. Every weight trains: the optimizer covers encoder, quantizer, MAE and decoder. It is a full fine-tune, closest to our B3, and not a probe.

## 5. The pretrained encoder as a frozen arm under ADR-0012

### 5.1 What the checkpoint is

Read from `vqmae.ckpt`'s pickled metadata:

| Part | Keys | Parameters | Used by a frozen extractor? |
|---|---|---:|---|
| VQ-GAN encoder (`encoder.*`) | taming `Encoder`, `ch=128`, `ch_mult=[1,2,4]`, `conv_in` `(128, 3, 3, 3)`, `conv_out` to 3 channels | 22,337,411 | yes |
| `quant_conv` | `(3, 3, 1, 1)` | 12 | yes |
| MAE encoder | `patch_embed.proj` `(1024, 3, 4, 4)`, `pos_embed` `(1, 197, 1024)`, `cls_token`, 24 blocks of width 1024, `norm` | 302,564,352 | yes |
| MAE decoder | 8 blocks of width 512, `decoder_pred` `(48, 512)` | 25,870,896 | no |
| VQ codebook, `post_quant_conv`, VQ-GAN decoder | `quantize.embedding` `(8192, 3)` | 32,985,359 | no |
| LPIPS loss network (`loss.perceptual_loss.*`) | VGG16 | 14,716,166 | no |
| **Total in file** | 636 tensors | **398,474,196** | |

The total matches the `# 398.474217M` comment at the end of the VF fine-tune script to within 21 parameters (398,474,217 against 398,474,196). The difference was not traced. The checkpoint is a full Lightning checkpoint with `optimizer_states`. Adam's two moment buffers for each weight are why the file is 4.66 GB when the weights are about 1.6 GB.

**The frozen extractor holds 324,901,775 parameters**, the stem plus `quant_conv` plus the MAE encoder. The DEMAE ViT-B registered in `src/benchmark/arms.py` holds 85,254,144, so the ratio is **3.81×**. The ticket's 398 M overstates the frozen arm by 73.6 M of decoders and a loss network that no probe reads.

Compute is the bigger gap. A rough count at 2 FLOPs per multiply-add puts the conv stem at about **258 GFLOPs per 224 px tile**, most of it 3 × 3 convolutions at 224² and 112² with 128 to 512 channels. It also has a 3,136-token self-attention block in its mid layer. The ViT-L adds about 123 GFLOPs, so the extractor needs about **380 GFLOPs per tile against about 35 for a ViT-B/16**, roughly **11×**. This is arithmetic on the shapes, not a measurement. It is the number #190 should price the arm with before a laptop timing replaces it.

`vq-f4.ckpt` (723 MB) holds only the VQ-GAN: 70,038,948 parameters, the same stem shapes, and LPIPS. **A frozen arm does not need it.** `vqmae.ckpt` already contains the encoder. In the VQ-MAE stage the encoder is not in the optimizer (`dem_vqmae_pretrain.py`'s `configure_optimizers` lists only MAE, quantizer, `post_quant_conv` and decoder), so it should be the f4 encoder unchanged. That is inferred from the code and was not checked weight by weight.

**Pins, from Figshare's `computed_md5`:** `vqmae.ckpt` is `9503a71f7387d8881596ab8b29b8d5c1` (4,662,292,513 bytes) and `vq-f4.ckpt` is `dd6ebf52b5fc1bac9d7f55bb3da5a4b4`. The supplied and computed md5 values agree for both. So the encoder can be pinned before it is downloaded, which the map's rule that `checkpoint_md5=None` is unrunnable requires.

**Training provenance, from the loop counters:**

- `vqmae.ckpt` stopped at epoch 9, step 81,490, with 8,149 batches per epoch. That is ten epochs, matching `vqmae.yaml`'s `max_epochs: 10`.
- `vq-f4.ckpt` stopped at epoch 49, step 203,750, with 4,075 batches per epoch. That is fifty epochs, matching `vq-f4.yaml`.

The batch size used at run time is not recorded, so the corpus size cannot be derived from these counters. The pretraining corpus itself is unreleased (README).

### 5.2 Input contract

- **Normalisation is a global fixed affine, applied before 3-channel replication.** Both `dem_dataloader.py` (pretraining) and `dem_LF_dataloader.py` set `dem[dem > 9000] = -1000` and `dem[dem < -1000] = -1000`, compute `(dem + 1000) / 10000`, and then `torch.cat([dem, dem, dem])`. Note the first rule: values **above 9000 m are mapped to −1000 m, not clipped to 9000**. No GLO-30 land pixel exceeds 8,849 m, so the rule never fires on our tiles. The extractor should still copy it exactly rather than "fix" it. Under ADR-0012 Decision 1 this works like TerraMind's normaliser, an affine that ignores the mask. GLO-30's ocean value of 0 m maps to 0.1, a valid in-range input.
- **The input is 224 × 224 at the pretraining scale.** The MAE is built for a 56 × 56 latent, and the f4 stem turns 224 px into 56. Our tiles fit without resampling.
- **The ground sample distance does not match, and the size of the mismatch is unknown.** The fine-tune areas are on an EPSG:3395 grid whose "30 m" is about 22–24 m on the ground (p124 §5.3). The pretraining corpus's grid and latitudes are unreleased, so the pretraining GSD cannot be stated. Report it as a GSD caveat in the same family as TerraMind's.

### 5.3 What a frozen feature is in this stack

The fine-tune `forward` defines the path: `encode(x)`, which is `encoder` then `quant_conv`, then `self.mae(latent)`, then `decode`, which quantizes. **The MAE reads the continuous pre-quantization latent, not codebook entries.** So the frozen feature is:

> raw metres → affine → replicate ×3 → VQ-GAN encoder → `quant_conv` → a 3 × 56 × 56 latent → MAE `patch_embed` (4 × 4 patches) → 196 tokens + CLS, 1024-d, plus the fixed sin-cos `pos_embed` → 24 transformer blocks → `norm`.

- **The token grid is 14 × 14 and matches ours exactly.** `pos_embed` is `(1, 197, 1024)`, which is 14 × 14 + CLS. Each token covers a 4 × 4 latent patch, which is a 16 × 16 px input footprint. The geometry is the same as a ViT-B/16 at 224 px. **No ADR-0011-style position derivation is needed**, because the table is native to this grid.
- **The layout is ADR-0012 Decision 5's**, once ordering is fixed. timm's `PatchEmbed` flattens row-major, `has_cls=True`, and `D = 1024`.
- **`output` is normed and `intermediates[23]` is not.** `forward_encoder` applies `self.norm` after the last block, the same pattern as DEMAE. So `output_is_normed=True`, and ADR-0012's field note carries over with 24 in place of 12.
- **All 24 intermediates are reachable.** `mae.blocks` is a plain `nn.ModuleList` of timm `Block`s, so the extractor iterates it directly. A UperNet at "the same relative depths" as {3, 6, 9, 12} would read {6, 12, 18, 24}. That choice belongs to the arms rewrite.

### 5.4 What stops a conformant extractor today, and what it would take

None of these makes ADR-0012 impossible. Each is a named piece of work.

1. **The contract fixes twelve layers for every encoder.** `src/benchmark/extract.py:146` declares `LAYERS: Final = 12`, the spec loader refuses `self.layers != LAYERS`, and `arms.py`'s `depth` field defaults to 12. A 24-block encoder is refused by name. The fix is an **ADR-0012 amendment** that makes the layer count a per-encoder declared field, which `depth` already half is. The ADR's own "twelve intermediates" check in Decision 7.3 changes with it. This is an interface change, so it needs a decision and not a build-time patch.
2. **`forward_encoder` shuffles token order even with no masking.** It always calls `random_masking`. At `mask_ratio=0`, `len_keep = L`, but `x_masked` is still gathered in `argsort(noise)` order: a fresh random permutation of the 196 tokens on every call. The released decoder undoes it with `ids_restore`, but anyone who reads the encoder output as row-major gets scrambled tokens. Attention does not care about order, so each token's *value* is correct, which makes the bug silent: shapes and statistics look fine. The extractor must run `patch_embed`, add `pos_embed`, prepend CLS, run the blocks and apply `norm` itself, never calling `random_masking`. It owes a test that compares against `forward_encoder` with `ids_restore` applied. This is the same class of trap as p118's normalisation finding, a well-shaped wrong tensor.
3. **The loader has two stages and comes from 2023.** The stem is taming-transformers' `Encoder` (`attn_resolutions: []`, though the mid block always has an attention layer). The MAE uses `timm==0.5.4` `Block`s. The environment pin is CUDA 11.3 with torch 1.12 (#128). The weights load into current modules by key: 636 named tensors under a `state_dict` key, read with an explicit allow-list of the `encoder.`, `quant_conv.` and `mae.`-minus-`decoder*` prefixes. **The load must be strict over that subset.** The released code's `strict=False` is #122's silent-wrong-load hazard. The pickle holds Lightning loop and callback state besides the weights. Whether `torch.load(weights_only=True)` accepts it was not tested, and the first load should check that.
4. **4.66 GB must be downloaded to use about 1.3 GB of weights.** That is a cost, not a blocker. The md5 pin above lets the download be checked.

### 5.5 Confounds the row would have to carry

- **Capacity:** 3.81× the parameters and about 11× the per-tile compute of the DEMAE ViT-B (§5.1). A GeomorPM row beating DEMAE says less about DEM pretraining than about size, and a row losing to DEMAE is the interesting outcome. Report both totals as the plan's capacity policy requires.
- **ImageNet initialisation.** `dem_vqmae_latent_pretrain.py` initialises its MAE from `models/mae_vit_large.pth`, reading a `"model"` key and dropping `patch_embed`, `decoder_pred` and head keys. That filename and key layout are FAIR's ImageNet MAE ViT-L release, which is an inference from the format. The VQ-GAN config (`embed_dim 3`, `n_embed 8192`, `ch 128`, `ch_mult [1,2,4]`, `z_channels 3`) and `vq-f4.yaml`'s initial path `models/vq-f4/model.ckpt` match CompVis's latent-diffusion `vq-f4` first-stage model, also inferred. So GeomorPM is ImageNet-to-DEM continued pretraining in a VQ latent, the same family as DEMAE's R4, not DEM-only from scratch.
- **The released scripts do not reproduce the released weights.** `dem_vqmae_pretrain.py` and `dem_vqmae_latent_pretrain.py` build the MAE with `in_chans=4` (and the latent extractor names a `vq-f8.ckpt`), while the checkpoint's `patch_embed` is 3-channel and the fine-tune scripts use 3. The checkpoint is the authority. The pretraining recipe cannot be stated exactly.
- **Corpus overlap with our test geocells cannot be checked.** The pretraining corpus is unreleased and built from global products (AW3D30, SRTM; p124 §5.2). It may cover our test tiles. That is the same unknowable position TerraMind is in, and it should be stated the same way.

## 6. Licence fit

- **The deposit is CC BY 4.0**, item-level, from the Figshare API.
- **Nothing in the deposit narrows it.** The README has no licence text. Of the 15 `.py` files, only `pos_embed.py` carries a header, Meta's "licensed under the license found in the LICENSE file in the root directory". That file was copied from `facebookresearch/mae`, and per p101 the licence in question is CC BY-NC 4.0. An extractor does not need that file: the sin-cos table ships inside the checkpoint as `pos_embed`, and it is a standard formula.
- **Upstream, the weights carry p101's open question.** If the ViT-L started from FAIR's MAE ViT-L (§5.5), the CC BY 4.0 label on the derivative cannot settle the licence of what it was initialised from. p101 found that question unresolved. CompVis's latent-diffusion repo, the likely VQ-GAN source, is MIT.
- **For this benchmark the fit is fine.** We would download the weights, compute features and publish numbers. We would commit tile lists and results and never redistribute the weights (#128's licence section). B6 already rests on the same ImageNet-MAE question. Attribution goes in the dataset card: Yang (2024), *DEM foundation model*, Figshare, doi:10.6084/m9.figshare.25756806.v1, CC BY 4.0.

## 7. Amendment made

`task2-landform-plan.md` §11 cited GeomorPM for "task-level comparability". That claim fails on class scheme, label source, region, tile size and metric, and on the instrument too (full fine-tune against probe). The citation now names what GeomorPM is to this plan: the nearest prior DEM-pretrained model, whose released encoder is a candidate frozen arm, with no comparable landform number, and a pointer to this note.

## 8. Handed on, not decided here

- **Whether GeomorPM becomes a frozen-encoder arm** goes to the arms rewrite, [#190](https://github.com/JStoschek/DEMAE/issues/190). It should weigh: the ADR-0012 layer-count amendment (§5.4 item 1), the order-safe extractor and its test (item 2), about 11× per-tile compute (§5.1), and the ImageNet-initialisation and capacity confounds (§5.5). If adopted, the plan's "frozen-encoder arm with identical probes" becomes possible.
- **Not determined here:** GeomorPM's landform numbers, its five class names, which code digit is relief and which is altitude, the batch size and corpus of its pretraining, and whether `torch.load(weights_only=True)` reads the checkpoint.

## Sources

- Yang, J. et al. (2024). GeomorPM. *Int. J. Geogr. Inf. Sci.* 39(2), 422–451. doi:[10.1080/13658816.2024.2414409](https://doi.org/10.1080/13658816.2024.2414409). Paywalled; abstract only.
- Yang, J. (2024). *DEM foundation model*. Figshare. doi:[10.6084/m9.figshare.25756806.v1](https://doi.org/10.6084/m9.figshare.25756806.v1). CC BY 4.0. Read: all code and config files, the checkpoint metadata of both `.ckpt` files, and `Areas.zip`'s central directory.
- Yang, J., Xu, J., Lv, Y., Zhou, C., Zhu, Y., Cheng, W. (2023). Deep learning-based automated terrain classification using high-resolution DEM data. *Int. J. Appl. Earth Obs. Geoinf.* 118, 103249. doi:[10.1016/j.jag.2023.103249](https://doi.org/10.1016/j.jag.2023.103249). Abstract via OpenAlex; full text blocked.
- IGSNRR publication records: [GeomorPM](https://igsnrr.cas.cn/sourcedb/zw/lw/202606/t20260604_8214383.html), [2023 paper](https://igsnrr.cas.cn/sourcedb/zw/lw/202508/t20250826_7913067.html).
- Zhou, C., Cheng, W., Qian, J., Li, B., Zhang, B. (2009). Research on the classification system of digital land geomorphology of 1:1,000,000 in China. *J. Geo-inf. Sci.* 11(6), 707–724. [Page](https://www.dqxxkx.cn/CN/Y2009/V11/I6/707); title only read.
- OpenAlex: works citing GeomorPM, and open-access status of both papers.
