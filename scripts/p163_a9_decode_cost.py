"""Price A9's diffusion decode — the one line plan §7 leaves unpriced — before the pass is run.

Issue [#163](https://github.com/JStoschek/DEMAE/issues/163), under map
[#145](https://github.com/JStoschek/DEMAE/issues/145).
[`task1-void-fill-plan.md`](../docs/research/task1-void-fill-plan.md) §7 costs every line of the
benchmark from a measured primitive times a count, with one exception: **A9's row is `unpriced`,
carrying a guess — "at 1 s/instance it is 11.1 GPU-h, 4.5x the whole table above" — and the note
that §7 is not final until it is measured.** This is that measurement, and it deliberately runs
out of §4.5's order for the reason #163 gives: it is the only number left that can still move the
whole budget, so it is worth having early and cheaply rather than late and expensively.

**It measures the primitive, not the pass.** A small sample of real test tiles goes through the
full tokenizer — encode, FSQ, diffusion decode — and is timed; the count comes from §7. #163 is
explicit that no full pass runs here, that being ticket 23's, blocked on this so the bill is known
before it is incurred.

Four things are measured rather than assumed, each because assuming it would have cost something:

1. **Per-instance decode seconds, with the spread across tiles.** Diffusion decode should be
   content-independent — a fixed step count over fixed shapes — so the across-tile spread is the
   claim that the sample size does not matter, stated as a number instead of an argument.
2. **The batch curve.** Per-instance cost falls with batch until the device saturates, and
   [`p120-primitive-costs.md`](../docs/research/p120-primitive-costs.md) §3 requires a primitive be
   transferred at the batch where it ran fastest per sample, using the device ratio measured at
   that same batch. So the plateau has to be found, not guessed.
3. **Linearity in timesteps.** Cost is one decoder forward per DDIM step, so halving the steps
   should halve the bill. That is the lever any scope amendment would reach for first, and it is
   worth pricing here so the amendment can be argued against a number.
4. **`image_size`.** :meth:`DiVAE.decode_quant` defaults its output raster to the *decoder's*
   training size — 256 px — and takes no hint from the 14x14 latent it is handed. A 224 px tile
   decoded at the default comes back 256 px, silently, at 1.31x the decoder pixels. Both are
   timed and the record says which one A9 is.
5. **The box's own drift, as an error bar on all four.** This script's first run measured one
   configuration three times over twelve minutes and got 2.775, 3.272 and 3.597 s/instance, rising
   with elapsed time, while another session held a core at 100%. Nothing about that is visible in
   a sweep that walks its conditions once in order — it arrives as a *result*, attributed to
   whichever condition ran last. So a fixed control runs between phases, the sweeps repeat and
   report medians, and the order-sensitive comparisons run in both directions. See :func:`measure`.

**The pass that is timed is A9's pass, and that is checked rather than asserted.** The full
tokenizer comes down through terratorch's registry — the decoder is 68.5% of the checkpoint and
:class:`benchmark.extract.Extractor` deliberately does not load it — but the registry's encoder
half is terratorch's own build, with the native 16x16 `pos_emb` that
[ADR-0011](../docs/adr/0011-terramind-position-embedding.md) replaces with the centre-anchored
14x14 crop. So the crop is substituted here too, through `benchmark.extract`'s own
:func:`~benchmark.extract.derive_pos_emb`, and the record carries the max abs difference between
this encoder's tokens and the ones :meth:`benchmark.extract.Extractor.extract` returns for the
same tiles. It is 0.0 or this is not A9's encoder.

Needs terratorch (`pip install -e '.[terramind]'`), the 1.15 GB tokenizer and the 47 GB corpus, so
it is not reproducible from a fresh clone — which is why the record is committed:

    PYTHONPATH=src .venv/bin/python scripts/p163_a9_decode_cost.py \
        --device mps --out docs/research/p163-a9-decode-cost.json

The tile rule is **p134's, imported rather than restated**, by way of `p157`'s two helpers: the
same 20-stratum, evenly-spaced, all-land, test-split draw at a one-per-stratum quota. Timing does
not need a large sample — item 1 above is what establishes that — but it does need the *same*
sample, because two answers to "which tiles" is how a second sample quietly becomes a second
population.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Sequence

REPO = Path(__file__).resolve().parent.parent

#: The arm §1 defines and §7 leaves unpriced.
ARM = "A9"

#: §3.4's eval mask list: 30,000 synthetic rows (`BenchmarkConfig.synthetic_eval_instances`, which
#: this script cross-checks rather than restates) plus #153's 9,978 real-void rows. §7's A9 row
#: rounds the sum to "40k"; the exact count is used here, because a projection that rounds its own
#: count cannot later be re-derived from the parts.
REAL_VOID_INSTANCES = 9_978

#: TerraMind's own evaluation setting, out of the DEM tokenizer's recorded training args
#: (`num_eval_timesteps=50`, terratorch's `tokenizer_register`), and also
#: :meth:`DiVAE.decode_quant`'s default. A9 is a *reference row* — the tokenizer as published —
#: so the faithful step count is the one its authors evaluated at, not a cheaper one this project
#: picks.
TIMESTEPS = 50

#: The corpus tile, and therefore what A9 must return. Not the decoder's 256 px default — see the
#: module docstring's item 4.
TILE_PX = 224

#: Batches the sweep walks. p120 §3 found this device falls off a memory cliff at ViT-B batch 128;
#: the decoder is twice ViT-B's parameters, so the sweep stops well short and records where it
#: stopped rather than probing for the cliff.
BATCHES: tuple[int, ...] = (1, 2, 4, 8, 16, 32)

#: The timestep sweep, for the linearity claim. 50 is TerraMind's; the rest exist so that an
#: amendment proposing fewer steps is priced rather than imagined.
TIMESTEP_SWEEP: tuple[int, ...] = (5, 10, 20, 50)

#: The batch the timestep and `image_size` sweeps run at. Deliberately *not* the plateau batch:
#: neither sweep is projected from, both need repeating in both directions to be order-balanced,
#: and at the plateau that costs four times the wall clock to answer a question about a ratio.
SWEEP_BATCH = 8

#: The drift control — one cheap configuration, run between phases and never varied. Short enough
#: that running it eight times costs less than one plateau-batch decode, long enough that its own
#: timing noise is well under the drift it exists to detect.
CONTROL_BATCH = 8
CONTROL_TIMESTEPS = 10

#: p120's measured primitives and device ratios, read out of the record rather than retyped. It is
#: the only source in this repository for "what a local second is worth on the rented 5090", and a
#: second copy of those ratios is a second thing to keep in step.
P120_PATH = REPO / "docs" / "research" / "p120-primitive-costs.json"

#: How wrong the transfer can be, in multiples, on the way to a verdict. p120 §3 runs exactly this
#: table for the whole GPU program and reads the answer off where the $25 target breaks.
SENSITIVITY_FACTORS: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0, 20.0, 100.0)

#: §7's own target, quoted so the verdict is against a stated number.
BUDGET_USD = 25.0

#: §7's frozen ViT-B forward, from p120. This run re-measures the same primitive as A9's encode
#: half, so carrying it across by the same ratio is an independent check on the transfer — the
#: kind p120 §3 ran for A3 and had only one instance of.
PLAN_ENCODER_MS_PER_TILE = 0.179


def _script(name: str):
    """Load a sibling driver by path — `scripts/` is not a package and should not become one."""
    path = REPO / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- the projection, which is arithmetic and therefore testable without any of the above


def eval_instances(cfg) -> int:
    """§3.4's eval mask list, from the config's own synthetic count plus #153's real rows."""
    return cfg.synthetic_eval_instances + REAL_VOID_INSTANCES


def project(
    per_instance_seconds: float,
    *,
    instances: int,
    device_ratio: float,
    usd_per_hour: float,
) -> dict:
    """One local per-instance figure to a host bill, by p120 §3's transfer and nothing else.

    ``local_gpu_hours`` is what the pass would cost on the measuring device — the honest number,
    since it is measured — and ``host_gpu_hours`` divides it by the ratio p120 established between
    this laptop's MPS and the rented 5090. The dollars are the host's; this project's own laptop
    is not rented and its hours cost nothing but wall clock, which is the distinction #117's
    standing note draws and §7 repeats.
    """
    local_hours = per_instance_seconds * instances / 3600.0
    host_hours = local_hours / device_ratio
    return {
        "instances": instances,
        "per_instance_seconds_local": per_instance_seconds,
        "device_ratio_5090_over_local": device_ratio,
        "local_gpu_hours": local_hours,
        "host_gpu_hours": host_hours,
        "host_usd": host_hours * usd_per_hour,
        "usd_per_hour": usd_per_hour,
    }


def sensitivity(
    per_instance_seconds: float,
    *,
    instances: int,
    device_ratio: float,
    usd_per_hour: float,
    factors: Sequence[float] = SENSITIVITY_FACTORS,
    budget_usd: float = BUDGET_USD,
) -> list[dict]:
    """p120 §3's question, asked of this line: how wrong can the ratio be before the verdict moves.

    Each row divides the measured ratio by ``factor`` — "the transfer overstates the 5090 by this
    much" — and re-prices. ``within_budget`` is against §7's $25 target, so the row that breaks it
    is readable off the table instead of being argued for.
    """
    rows = []
    for factor in factors:
        row = project(
            per_instance_seconds,
            instances=instances,
            device_ratio=device_ratio / factor,
            usd_per_hour=usd_per_hour,
        )
        row["ratio_overstated_by"] = factor
        row["within_budget"] = row["host_usd"] <= budget_usd
        rows.append(row)
    return rows


def breakeven_factor(
    per_instance_seconds: float,
    *,
    instances: int,
    device_ratio: float,
    usd_per_hour: float,
    budget_usd: float = BUDGET_USD,
) -> float:
    """The multiple by which p120's transfer would have to be wrong for A9 alone to hit $25.

    Closed-form rather than searched: cost is linear in ``1 / ratio``, so the factor is just the
    budget divided by the projected bill. Reported because "the verdict survives a 5x error" is a
    sentence p120 had to write from a table, and this line can state it exactly.
    """
    base = project(
        per_instance_seconds,
        instances=instances,
        device_ratio=device_ratio,
        usd_per_hour=usd_per_hour,
    )
    return budget_usd / base["host_usd"] if base["host_usd"] > 0 else float("inf")


def device_ratios(batch: int) -> dict:
    """p120's ratio at ``batch``, or at its nearest measured batch, said out loud either way.

    p120 swept 16/32/64/128 and this sweep's plateau need not land on one of them. Rather than
    interpolate — the ratio is not smooth, and p120's 128 column is a memory cliff, not a data
    point — the nearest *stable* measured batch is used and the record carries both numbers so the
    mismatch is visible in the record rather than buried in this function.
    """
    p120 = json.loads(P120_PATH.read_text())
    sweep = p120["local_anchor_batch_sweep"]
    # p120 §3: the 128 column is off a memory cliff (spread 9.18 against 1.10-1.14 elsewhere), so
    # it is excluded from the match rather than being allowed to win on numeric proximity.
    stable = {
        int(k): v
        for k, v in sweep.items()
        if not k.startswith("_") and v["_ratio"]["spread"] < 2.0
    }
    nearest = min(stable, key=lambda b: (abs(b - batch), b))
    ratio = stable[nearest]["_ratio"]
    return {
        "source": "p120-primitive-costs.json",
        "measured_at_batch": nearest,
        "requested_batch": batch,
        "batch_matched": nearest == batch,
        "geometric_mean": ratio["geometric_mean"],
        "vit_b": ratio["vit_b"],
        "vit_s": ratio["vit_s"],
        "spread": ratio["spread"],
        # The smallest stable ratio p120 measured anywhere, which is the conservative transfer:
        # it makes the 5090 look as slow as p120's own evidence allows.
        "conservative_floor": min(
            min(v["_ratio"]["vit_b"], v["_ratio"]["vit_s"]) for v in stable.values()
        ),
        "usd_per_hour": p120["host_usd_per_hour"],
    }


# --- the measurement


def _sync(device: str) -> None:
    """Make the device's queue empty before the clock is read, whatever the device is."""
    import torch

    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


class Timer:
    """One decode, timed, on a device whose queue is drained on both sides of the clock."""

    def __init__(self, model, device: str):
        self.model = model
        self.device = device

    def encode(self, tensor):
        import torch

        with torch.no_grad():
            _sync(self.device)
            t0 = time.perf_counter()
            quant, _, tokens = self.model.encode(tensor)
            _sync(self.device)
            return quant, tokens, time.perf_counter() - t0

    def decode(self, quant, *, timesteps: int, image_size: int, generator=None):
        import torch

        with torch.no_grad():
            _sync(self.device)
            t0 = time.perf_counter()
            out = self.model.decode_quant(
                quant,
                timesteps=timesteps,
                image_size=image_size,
                verbose=False,
                generator=generator,
            )
            _sync(self.device)
            return out, time.perf_counter() - t0


def _stats(values: Sequence[float]) -> dict:
    """Mean, spread and range — #163 asks for the variance, so it is reported, not summarized.

    ``median_over_min`` is the robust one, and the one to read on a shared box. Interference is a
    **one-sided** error: another process can make a timed interval longer and can never make it
    shorter. So the minimum is the best available estimate of the uncontended cost, the spread
    above it is contention rather than variance in the thing being measured, and the ratio between
    them says how much of the sample was clean. A standard deviation over a contaminated tail
    describes the other sessions on this laptop, not the decoder.
    """
    values = list(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    mean = statistics.fmean(values)
    median = statistics.median(values)
    return {
        "n": len(values),
        "mean": mean,
        "median": median,
        "stdev": stdev,
        "min": min(values),
        "max": max(values),
        "coefficient_of_variation": stdev / mean if mean else float("nan"),
        "median_over_min": median / min(values) if min(values) else float("nan"),
    }


def build_tokenizer(checkpoint: Path, spec, entry, *, device: str):
    """The whole tokenizer, with ADR-0011's `pos_emb` substituted into its encoder half.

    Everything :meth:`benchmark.extract.Extractor._build_terratorch` refuses on, this refuses on
    too — the checkpoint's md5, TerraMind's named constants, the native `pos_emb` shape — but the
    *decoder* is why this exists: the contract's builder loads the encoder half alone, strictly,
    and A9 is the one arm that needs the other 196 M parameters.
    """
    import torch
    from torch import nn

    from benchmark.extract import derive_pos_emb, terramind_constants
    from pretrain.init import checkpoint_md5
    from terratorch.registry import FULL_MODEL_REGISTRY

    md5 = checkpoint_md5(checkpoint)
    if md5 != entry.checkpoint_md5:
        raise ValueError(
            f"{checkpoint} is md5 {md5}; extractor_spec.json declares {entry.checkpoint_md5} for "
            f"{entry.name}. A9 reads the same file A5 does, and a different one is a different "
            f"tokenizer with the same shape"
        )
    terramind_constants()

    model = FULL_MODEL_REGISTRY.build("terramind_v1_tokenizer_dem", pretrained=True)
    native = model.encoder.pos_emb.detach()
    native_px = entry.pos_emb["native_grid_px"]
    expected = (1, entry.width, native_px, native_px)
    if tuple(native.shape) != expected:
        raise ValueError(
            f"terratorch's registry built `pos_emb` {tuple(native.shape)} against the declared "
            f"{expected}; ADR-0011's crop has no subject at another shape"
        )
    model.encoder.pos_emb = nn.Parameter(
        derive_pos_emb(native, entry.pos_emb).clone(), requires_grad=False
    )
    if model.image_size != native_px * spec.patch_px:
        raise ValueError(
            f"the tokenizer's declared image_size {model.image_size} is not "
            f"{native_px}x{spec.patch_px}; p118 §3's 256 px training resolution is what the "
            f"native grid is a fact about"
        )
    return model.eval().to(device)


def encoder_agreement(model, extractor, raw, mask) -> dict:
    """Is the encoder half that is about to be timed the one the contract builds? A number, not a claim.

    Both sides run at fp32 — the extractor's reference dtype — so this is an exact comparison and
    not a tolerance. Any difference at all means the substituted `pos_emb` did not take, or that
    terratorch's registry build and its `vit_b_enc` build have diverged, and either one would make
    every figure below a figure about a different encoder.
    """
    import torch

    from benchmark.extract import normalize

    features = extractor.extract(raw, mask, arm=ARM)
    x = normalize(raw, mask, normalization="terramind_v1")
    tensor = torch.from_numpy(x).unsqueeze(1).to(extractor.device)
    with torch.no_grad():
        tokens = model.encoder(model.prepare_input(tensor)).flatten(2).transpose(1, 2)
    delta = (tokens.float() - features.output.to(tokens.device).float()).abs().max()
    return {
        "tiles": int(features.output.shape[0]),
        "tokens": int(features.output.shape[1]),
        "width": int(features.output.shape[2]),
        "max_abs_difference": float(delta),
        "identical": bool(delta == 0),
        "dtype": extractor.config.extract.compute_dtype,
    }


def measure(model, tensor, *, device: str, batches: Sequence[int], out_px: int, repeats: int) -> dict:
    """The sweeps, drift-controlled, all off the one loaded tokenizer and the one draw of tiles.

    **The first run of this script measured the same configuration three times and got 2.775,
    3.272 and 3.597 s/instance** — a 30% spread, rising monotonically with elapsed time, on a box
    that was sharing a core with another session's `freeze`. A sweep that walks its conditions in
    order on a drifting device does not measure the conditions; it measures the order. So:

    * a cheap **control** configuration runs before and after every phase, and its spread across
      the run is published as the drift, rather than the drift being distributed silently over the
      conditions that happened to run late;
    * every sweep point is repeated, and reported by its **minimum** with the median beside it —
      see :func:`_stats` for why the minimum is the estimator on a shared box, and note that the
      second run's medians were not even monotone in batch, which is not a shape a saturating
      curve can have;
    * the two order-sensitive comparisons — timesteps, and `image_size` — are run **ascending then
      descending**, so drift enters both arms of the comparison rather than landing entirely on
      whichever arm ran second.
    """
    import torch

    timer = Timer(model, device)
    n = int(tensor.shape[0])
    controls: list[dict] = []

    def control(phase: str) -> None:
        """A fixed, cheap reference: whatever else changed, this did not, so its drift is the box's."""
        idx = [i % n for i in range(CONTROL_BATCH)]
        quant, _, _ = timer.encode(tensor[idx])
        _, seconds = timer.decode(quant, timesteps=CONTROL_TIMESTEPS, image_size=out_px)
        controls.append(
            {
                "phase": phase,
                "batch": CONTROL_BATCH,
                "timesteps": CONTROL_TIMESTEPS,
                "seconds_per_instance": seconds / CONTROL_BATCH,
                "milliseconds_per_instance_per_step": (
                    seconds / CONTROL_BATCH / CONTROL_TIMESTEPS * 1e3
                ),
            }
        )

    # Warmup. The first decode on this device pays for allocation and kernel specialization, and
    # p120 §3 records its warmups separately for the same reason: a first call folded into a mean
    # is a measurement of the loader.
    quant, _, _ = timer.encode(tensor[:1])
    _, warmup_seconds = timer.decode(quant, timesteps=TIMESTEPS, image_size=out_px)
    control("start")

    # 1. Per-tile, at batch 1: the across-content spread that says the sample size does not matter.
    # Not repeated — forty tiles *is* forty samples of the same configuration, so its own spread
    # already carries whatever drift happened while it ran.
    per_tile = []
    for i in range(n):
        quant, _, encode_seconds = timer.encode(tensor[i : i + 1])
        _, decode_seconds = timer.decode(quant, timesteps=TIMESTEPS, image_size=out_px)
        per_tile.append(
            {"index": i, "encode_seconds": encode_seconds, "decode_seconds": decode_seconds}
        )
    control("after_per_tile")

    # 2. The batch curve. Each batch is one chunk of the same tiles, cycled if the batch is larger
    # than the sample — the decode's cost is fixed by shapes and step count, which sweep 1 is what
    # establishes, so repeating a tile costs nothing but is recorded anyway. The repeats walk the
    # whole sweep each time rather than repeating each batch in place, so a batch is never measured
    # only at one moment of the run.
    trials: dict[int, list[dict]] = {batch: [] for batch in batches}
    stopped_at = None
    for _ in range(repeats):
        for batch in batches:
            if stopped_at is not None and batch >= stopped_at:
                continue
            idx = [i % n for i in range(batch)]
            chunk = tensor[idx]
            try:
                quant, _, encode_seconds = timer.encode(chunk)
                _, decode_seconds = timer.decode(quant, timesteps=TIMESTEPS, image_size=out_px)
            except RuntimeError as exc:  # pragma: no cover - device memory, not behaviour
                trials[batch].append({"failed": str(exc)})
                stopped_at = batch
                continue
            trials[batch].append(
                {
                    "distinct_tiles": len(set(idx)),
                    "encode_seconds_per_instance": encode_seconds / batch,
                    "decode_seconds_per_instance": decode_seconds / batch,
                    "seconds_per_instance": (encode_seconds + decode_seconds) / batch,
                }
            )
    control("after_batch_sweep")

    batch_rows = []
    for batch in batches:
        ok = [t for t in trials[batch] if "failed" not in t]
        if not ok:
            failure = next((t["failed"] for t in trials[batch] if "failed" in t), "not run")
            batch_rows.append({"batch": batch, "failed": failure})
            continue
        batch_rows.append(
            {
                "batch": batch,
                "trials": len(ok),
                "distinct_tiles": ok[0]["distinct_tiles"],
                # The projected figure. Minimum, not median, for the reason `_stats` gives: on a
                # box shared with other sessions the error is one-sided, and the pass this
                # projects onto a rented 5090 will not be sharing anything.
                "seconds_per_instance": min(t["seconds_per_instance"] for t in ok),
                "encode_seconds_per_instance": min(t["encode_seconds_per_instance"] for t in ok),
                "decode_seconds_per_instance": min(t["decode_seconds_per_instance"] for t in ok),
                # Published beside it so the choice is visible rather than silent, and so a run on
                # a quiet box can be told from this one.
                "seconds_per_instance_median": statistics.median(
                    t["seconds_per_instance"] for t in ok
                ),
                "seconds_per_instance_spread": _stats([t["seconds_per_instance"] for t in ok]),
            }
        )

    ran = [r for r in batch_rows if "failed" not in r]
    plateau = min(ran, key=lambda r: r["seconds_per_instance"])

    # 3. Linearity in timesteps. The lever an amendment would pull, so it is the one comparison
    # worth protecting hardest from order: the sweep runs up and back down, and each step count's
    # two passes straddle the midpoint of the phase.
    idx = [i % n for i in range(SWEEP_BATCH)]
    quant, _, _ = timer.encode(tensor[idx])
    timestep_trials: dict[int, list[float]] = {t: [] for t in TIMESTEP_SWEEP}
    for timesteps in (*TIMESTEP_SWEEP, *reversed(TIMESTEP_SWEEP)):
        _, seconds = timer.decode(quant, timesteps=timesteps, image_size=out_px)
        timestep_trials[timesteps].append(seconds / SWEEP_BATCH)
    control("after_timestep_sweep")

    timestep_rows = [
        {
            "timesteps": timesteps,
            "batch": SWEEP_BATCH,
            "trials": len(seconds),
            "decode_seconds_per_instance": min(seconds),
            "milliseconds_per_instance_per_step": min(seconds) / timesteps * 1e3,
            "ascending_then_descending": list(seconds),
        }
        for timesteps, seconds in timestep_trials.items()
    ]

    # 4. `image_size`. The default is the decoder's 256, whatever the latent says. Alternated for
    # the same reason as sweep 3, and this one needs it most: the arms differ by only (256/224)^2 =
    # 1.31x of decoder pixels, which a drifting box can manufacture or erase on its own.
    sizes = (TILE_PX, int(model.image_size))
    size_trials: dict[int, list[float]] = {px: [] for px in sizes}
    shapes: dict[int, tuple] = {}
    for image_size in (*sizes, *reversed(sizes)):
        decoded, seconds = timer.decode(quant, timesteps=TIMESTEPS, image_size=image_size)
        size_trials[image_size].append(seconds / SWEEP_BATCH)
        shapes[image_size] = tuple(int(d) for d in decoded.shape[-2:])
    control("after_image_size_sweep")

    size_rows = [
        {
            "image_size": px,
            "batch": SWEEP_BATCH,
            "trials": len(seconds),
            "is_decode_quant_default": px == int(model.decoder.sample_size),
            "output_px": list(shapes[px]),
            "matches_tile": shapes[px] == (TILE_PX, TILE_PX),
            "decode_seconds_per_instance": min(seconds),
            "ascending_then_descending": list(seconds),
        }
        for px, seconds in size_trials.items()
    ]

    # 5. Is the pass reproducible? A9 trains nothing and scores once, so a stochastic decode is a
    # result that cannot be re-derived. Checked at the cheapest step count that still runs the
    # loop, because what is being established is whether the generator is honoured, not a cost.
    generator = torch.Generator().manual_seed(0)
    first, _ = timer.decode(quant[:1], timesteps=2, image_size=out_px, generator=generator)
    generator = torch.Generator().manual_seed(0)
    second, _ = timer.decode(quant[:1], timesteps=2, image_size=out_px, generator=generator)
    free, _ = timer.decode(quant[:1], timesteps=2, image_size=out_px)

    drift = _stats([c["seconds_per_instance"] for c in controls])
    return {
        "warmup_seconds": warmup_seconds,
        "per_tile": per_tile,
        "per_tile_decode": _stats([r["decode_seconds"] for r in per_tile]),
        "per_tile_encode": _stats([r["encode_seconds"] for r in per_tile]),
        "batch_sweep": batch_rows,
        "plateau": plateau,
        "timestep_sweep": timestep_rows,
        "image_size_sweep": size_rows,
        "control": {
            "batch": CONTROL_BATCH,
            "timesteps": CONTROL_TIMESTEPS,
            "runs": controls,
            "spread": drift,
            # The honest error bar on every figure above: one unchanging configuration, run
            # repeatedly through the same session, moved by this much.
            "drift_max_over_min": drift["max"] / drift["min"],
        },
        "determinism": {
            "timesteps": 2,
            "seeded_max_abs_difference": float((first - second).abs().max()),
            "unseeded_max_abs_difference": float((first - free).abs().max()),
            "seeded_is_reproducible": bool((first - second).abs().max() == 0),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, default=REPO / "data" / "corpus_500k")
    parser.add_argument("--checkpoint", type=Path, default=None, help="TerraMind_Tokenizer_DEM.pt")
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda"))
    # Two, which is p134's own floor — its draw is evenly spaced *within* a stratum and needs both
    # endpoints to span one. Forty tiles is more content than a timing measurement needs, and that
    # is the point: the across-tile spread is what establishes it did not need them.
    parser.add_argument("--per-stratum", type=int, default=2)
    parser.add_argument(
        "--batches",
        type=lambda s: tuple(int(b) for b in s.split(",")),
        default=BATCHES,
    )
    # Five passes over the whole batch sweep, not five of each batch in place — a batch measured
    # five times back to back is measured five times at one moment of a drifting run. Five rather
    # than three because the estimator is a minimum, and a minimum over a contended sample gets
    # better with draws in a way a mean does not.
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    import torch

    from benchmark.config import BenchmarkConfig
    from benchmark.extract import Extractor, load_spec, normalize, spec_path
    from dataset.normalize import land_mask

    checkpoint = args.checkpoint
    if checkpoint is None:
        from huggingface_hub import hf_hub_download

        checkpoint = Path(
            hf_hub_download(
                "ibm-esa-geospatial/TerraMind-1.0-Tokenizer-DEM", "TerraMind_Tokenizer_DEM.pt"
            )
        )

    base = BenchmarkConfig()
    cfg = base.replace(
        arm=ARM,
        paths=base.paths.replace(terramind_checkpoint=str(checkpoint)),
        # fp32 so `encoder_agreement` below is an equality and not a tolerance. It is also what
        # A9 would run at: the arm has no head to absorb a rounding, and ADR-0012 §6's licence to
        # move the compute dtype is granted against a measured agreement this arm has not taken.
        extract=base.extract.replace(compute_dtype="fp32"),
    )
    spec = load_spec(spec_path(cfg))
    entry = spec.encoder_for(ARM)
    extractor = Extractor(spec, cfg, device=args.device)

    p157 = _script("p157_terramind_precision")
    rows = p157.select_tiles(args.corpus, args.per_stratum)
    raw = p157.load_tiles(args.corpus, rows)
    mask = land_mask(raw)

    model = build_tokenizer(checkpoint, spec, entry, device=args.device)
    agreement = encoder_agreement(model, extractor, raw, mask)

    x = normalize(raw, mask, normalization="terramind_v1")
    tensor = torch.from_numpy(x).unsqueeze(1).to(args.device)
    measured = measure(
        model,
        tensor,
        device=args.device,
        batches=args.batches,
        out_px=TILE_PX,
        repeats=args.repeats,
    )

    instances = eval_instances(cfg)
    plateau = measured["plateau"]
    ratios = device_ratios(plateau["batch"])
    # **Decode alone**, deliberately. §7's inference line already prices one frozen encoder forward
    # for each of the 38 scored passes and A9 is one of them, so billing A9's encode here would
    # count it twice. At 14 ms/instance the double-count would move this line by 0.6% and the
    # verdict by nothing — which is exactly the size of error that survives into a table and gets
    # re-derived by someone later. The encode is measured and reported; it is just not charged
    # twice, and `projection_including_encode` is what it would have been.
    per_instance = plateau["decode_seconds_per_instance"]
    projection = project(
        per_instance,
        instances=instances,
        device_ratio=ratios["geometric_mean"],
        usd_per_hour=ratios["usd_per_hour"],
    )
    including_encode = project(
        plateau["seconds_per_instance"],
        instances=instances,
        device_ratio=ratios["geometric_mean"],
        usd_per_hour=ratios["usd_per_hour"],
    )
    # The transfer's one independent check: §7 already carries a frozen ViT-B forward at
    # 0.179 ms/tile from p120, and this run re-measures that primitive on a different phase of the
    # same laptop. Carried across by the same ratio, the two should agree — and how well they do is
    # the only evidence in this record about whether the ratio itself travels.
    encode_ms_on_host = plateau["encode_seconds_per_instance"] / ratios["geometric_mean"] * 1e3
    encode_check = {
        "measured_encode_ms_per_tile_local": plateau["encode_seconds_per_instance"] * 1e3,
        "transferred_encode_ms_per_tile_host": encode_ms_on_host,
        "plan_section_7_ms_per_tile": PLAN_ENCODER_MS_PER_TILE,
        "ratio_to_plan": encode_ms_on_host / PLAN_ENCODER_MS_PER_TILE,
    }
    conservative = project(
        per_instance,
        instances=instances,
        device_ratio=ratios["conservative_floor"],
        usd_per_hour=ratios["usd_per_hour"],
    )

    import importlib.metadata as md

    record = {
        "issue": 163,
        "map": 145,
        "arm": ARM,
        "amends": "task1-void-fill-plan.md §7",
        "device": args.device,
        "timesteps": TIMESTEPS,
        "tile_px": TILE_PX,
        "instances": {
            "total": instances,
            "synthetic": cfg.synthetic_eval_instances,
            "real_void": REAL_VOID_INSTANCES,
            "plan_section_7_rounds_to": "40k",
        },
        "encoder_agreement": agreement,
        "encoder": {
            "name": entry.name,
            "checkpoint_md5": entry.checkpoint_md5,
            "pos_emb": entry.pos_emb,
            "normalization": entry.normalization,
            "parameters": {
                name: sum(p.numel() for p in child.parameters())
                for name, child in model.named_children()
            },
        },
        "extractor_spec_md5": spec.md5(),
        "measured": measured,
        "transfer": ratios,
        "projection": projection,
        "projection_priced": "decode only; §7's inference line already bills every scored pass's "
        "encoder forward, and A9 is one of the 38",
        "projection_including_encode": including_encode,
        "projection_conservative": conservative,
        "encoder_forward_cross_check": encode_check,
        "sensitivity": sensitivity(
            per_instance,
            instances=instances,
            device_ratio=ratios["geometric_mean"],
            usd_per_hour=ratios["usd_per_hour"],
        ),
        "breakeven_ratio_overstated_by": breakeven_factor(
            per_instance,
            instances=instances,
            device_ratio=ratios["geometric_mean"],
            usd_per_hour=ratios["usd_per_hour"],
        ),
        "budget_usd": BUDGET_USD,
        # §7's placeholder, kept beside the measurement so the amendment is legible as a
        # replacement of a specific claim rather than as a new number appearing.
        "plan_section_7_placeholder": {
            "seconds_per_instance": 1.0,
            "gpu_hours": 11.1,
            "status": "unpriced",
        },
        "tiles": [
            {
                "tile_id": row["tile_id"],
                "stratum_id": row["stratum_id"],
                "split": row["split"],
                "lon": row["lon"],
                "lat": row["lat"],
                "local_relief_metres": row["local_relief"],
            }
            for row in rows
        ],
        "selection": {
            "rule": "p134's: evenly-spaced, all-land, test-split, per stratum",
            "per_stratum": args.per_stratum,
        },
        "versions": {
            p: md.version(p) for p in ("terratorch", "diffusers", "torch", "timm", "numpy")
        },
    }
    text = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
