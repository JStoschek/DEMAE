"""#194: what B9 (TerraMind v1 base backbone, raw DEM) actually is, measured off the pinned file.

Reads the HF checkpoint at a pinned revision, builds terratorch's `terramind_v1_base` with the
single `untok_dem@224` modality, loads it through terratorch's own filter, and records:

- the file's identity (bytes, md5, sha256) and what the model takes from it;
- parameter / buffer counts, for §2's capacity table;
- whether the `untok_dem@224` position table is the native 14x14 sin-cos closed form;
- what the backbone's returned list *is* against hooked raw block outputs: twelve entries,
  the last one replaced by `encoder_norm(block 12)`;
- a forward over p134's balanced test-tile rule (3 per stratum, 60 tiles) at 224 px, and the
  cosine cost of swapping in #138's two other DEM constant pairs.

Run with the main checkout's venv (terratorch 1.2.10), e.g.
`python scripts/p194_b9_backbone_probe.py --corpus data/corpus_500k --device mps`.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

_SCRIPTS = Path(__file__).resolve().parent

REPO_ID = "ibm-esa-geospatial/TerraMind-1.0-base"
FILENAME = "TerraMind_v1_base.pt"
REVISION = "fb96c70d0a5f68dcc44030b89cbfd8ec3fb0c67a"
MODALITY = "untok_dem@224"

#: #138's three DEM pairs across IBM's artifacts.
PAIRS = {
    "v1 untok_dem@224 (terramind_register)": (670.665, 951.272),
    "TerraMesh loader": (651.663, 928.168),
    "v0.1 tok_dem@224": (435.726, 560.326),
}


def _sibling(name: str):
    spec = importlib.util.spec_from_file_location(f"p194_dep_{name}", _SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def file_identity(path: Path) -> dict:
    md5, sha = hashlib.md5(), hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            md5.update(chunk)
            sha.update(chunk)
    return {"bytes": path.stat().st_size, "md5": md5.hexdigest(), "sha256": sha.hexdigest()}


def cos(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    return torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=-1).flatten().cpu().numpy()


def summ(x: np.ndarray) -> dict:
    return {"mean": float(x.mean()), "p01": float(np.quantile(x, 0.01)), "min": float(x.min())}


def run(corpus: Path, device: str, batch_size: int, per_stratum: int) -> dict:
    from huggingface_hub import hf_hub_download
    from terratorch.models.backbones.terramind.model.terramind_register import (
        checkpoint_filter_fn,
        terramind_v1_base,
        v1_pretraining_mean,
        v1_pretraining_std,
    )
    from terratorch.models.backbones.terramind.model.tm_utils import build_2d_sincos_posemb

    rec: dict = {"repo_id": REPO_ID, "filename": FILENAME, "revision": REVISION}
    path = Path(hf_hub_download(REPO_ID, FILENAME, revision=REVISION))
    rec["file"] = file_identity(path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    rec["file_tensors"] = len(state)
    rec["file_numel"] = int(sum(v.numel() for v in state.values()))

    rec["constants_by_name"] = {
        "v1_pretraining_mean[untok_dem@224]": v1_pretraining_mean[MODALITY],
        "v1_pretraining_std[untok_dem@224]": v1_pretraining_std[MODALITY],
        "v1_pretraining_mean[tok_dem@224]": v1_pretraining_mean["tok_dem@224"],
        "v1_pretraining_std[tok_dem@224]": v1_pretraining_std["tok_dem@224"],
    }

    model = terramind_v1_base(pretrained=False, modalities=["DEM"], merge_method=None)
    model_sd = model.state_dict()
    rec["modalities"] = model.modalities
    rec["tokenizers_built"] = list(model.tokenizer.keys()) if hasattr(model.tokenizer, "keys") else []
    rec["model_keys_missing_from_file"] = sorted(set(model_sd) - set(state))
    rec["model_keys_shape_mismatch"] = sorted(
        k for k in model_sd if k in state and state[k].shape != model_sd[k].shape
    )
    model.load_state_dict(checkpoint_filter_fn(state, model), strict=True)
    # every model tensor bit-identical to the file?
    rec["all_model_tensors_from_file"] = all(
        torch.equal(v, state[k]) for k, v in model.state_dict().items()
    )
    model.eval()

    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    rec["params"] = int(sum(p.numel() for p in params.values()))
    rec["buffers"] = {k: int(v.numel()) for k, v in buffers.items() if v.numel() > 0}
    rec["buffers_total"] = int(sum(rec["buffers"].values()))
    rec["state_dict_numel"] = int(sum(v.numel() for v in model.state_dict().values()))
    rec["params_by_group"] = {}
    for k, p in params.items():
        g = k.split(".")[0]
        rec["params_by_group"][g] = rec["params_by_group"].get(g, 0) + p.numel()
    # LayerNorm biases: registered as buffers (bias=False) but present in the file
    ln_bias = {k: float(v.abs().max()) for k, v in buffers.items() if k.endswith(".bias")}
    rec["layernorm_bias_buffers"] = {"count": len(ln_bias), "max_abs": max(ln_bias.values(), default=0.0)}

    emb = model.encoder_embeddings[MODALITY]
    rec["pos_emb_shape"] = list(emb.pos_emb.shape)
    rec["pos_emb_is_buffer"] = "encoder_embeddings." + MODALITY + ".pos_emb" in buffers
    rec["embedding_image_size"] = list(emb.image_size)
    closed = build_2d_sincos_posemb(14, 14, 768)
    rec["pos_emb_vs_sincos14_maxabs"] = float((emb.pos_emb - closed).abs().max())
    w = emb.proj.weight
    rec["proj_weight_std"] = float(w.std())
    rec["xavier_uniform_init_std"] = float((2.0 / (w.shape[0] + w.shape[1])) ** 0.5)
    rec["encoder_depth"] = len(model.encoder)
    rec["num_register_tokens"] = model.num_register_tokens
    rec["encoder_norm"] = type(model.encoder_norm).__name__

    # --- forward over real tiles ---
    p134 = _sibling("p134_posemb_materiality")
    p118 = _sibling("p118_terramind_probe")
    frame = p134.read_selection_frame(corpus)
    ids = p134.select_tile_ids(frame, per_stratum=per_stratum)
    rows = p134.selection_rows(frame, ids)
    rec["tiles"] = len(rows)

    model.to(device)
    raw: list[torch.Tensor] = []
    hooks = [blk.register_forward_hook(lambda m, i, o: raw.append(o.detach())) for blk in model.encoder]

    agree_mid, agree_last_normed, last_vs_raw12 = [], [], {k: [] for k in PAIRS}
    raw12_rms, out_rms = [], []
    shapes = None
    swap = {k: [] for k in PAIRS if not k.startswith("v1")}
    with torch.no_grad():
        for lo in range(0, len(rows), batch_size):
            chunk = rows[lo : lo + batch_size]
            metres = np.stack([p118.load_tile_metres(corpus, r["shard_id"], r["row_index"]) for r in chunk])
            ref_out = None
            for name, (mu, sd) in PAIRS.items():
                x = torch.from_numpy(((metres - mu) / sd).astype(np.float32))[:, None].to(device)
                raw.clear()
                out = model(x)
                if name.startswith("v1"):
                    shapes = {"input": list(x.shape), "returned_len": len(out),
                              "returned_shapes": sorted({tuple(t.shape) for t in out}),
                              "hooked_len": len(raw)}
                    for i in range(11):
                        agree_mid.append(bool(torch.equal(out[i], raw[i])))
                    normed = model.encoder_norm(raw[11])
                    agree_last_normed.append(float((out[11] - normed).abs().max()))
                    last_vs_raw12["v1 untok_dem@224 (terramind_register)"].extend(cos(out[11], raw[11]))
                    raw12_rms.append(float(raw[11].float().pow(2).mean().sqrt()))
                    out_rms.append(float(out[11].float().pow(2).mean().sqrt()))
                    ref_out = out[11]
                else:
                    swap[name].extend(cos(out[11], ref_out))
    for h in hooks:
        h.remove()

    rec["forward"] = {
        **(shapes or {}),
        "blocks_1_to_11_returned_equal_raw_hook": all(agree_mid),
        "returned_12_minus_encoder_norm_raw12_maxabs": max(agree_last_normed),
        "cos_returned12_vs_raw_block12": summ(np.asarray(last_vs_raw12["v1 untok_dem@224 (terramind_register)"])),
        "rms_raw_block12": float(np.mean(raw12_rms)),
        "rms_returned12": float(np.mean(out_rms)),
        "cos_output_wrong_pair_vs_v1": {k: summ(np.asarray(v)) for k, v in swap.items()},
    }
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--per-stratum", type=int, default=3)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    rec = run(a.corpus, a.device, a.batch_size, a.per_stratum)
    text = json.dumps(rec, indent=2, default=str)
    print(text)
    if a.out:
        a.out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
