"""#191: Task 2's §8, priced as primitive x count from docs/research/p191-task2-primitives.json.

Every GPU primitive is an MPS timing carried to the rented 5080 at the ratio of one step measured
on both: Task 1's frozen-head step (R5 inline + the 1.18M head), 23.2 ms/tile on MPS against
1.15 ms/tile in the 5080 logs. B7 needs no transfer (A7 was timed on the 5080).

    python scripts/p191_price.py [--upernet-channels 256] [--tune-fraction 10%] [--steps-matched]

#196 froze the head recipes; its budget is

    python scripts/p191_price.py --floor 10% --val-reads 10 --tune-per-rung

i.e. the 1% rung trains as many steps as the 10% rung, val is read 10 times per run, and tuning is
Task 1's discipline (lr only, 5 points, the first fifth of the schedule, one seed) on B6 at every
rung, with B7 tuning itself the same way.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

P = json.loads((Path(__file__).parents[1] / "docs/research/p191-task2-primitives.json").read_text())

ap = argparse.ArgumentParser()
ap.add_argument("--upernet-channels", type=int, default=512, choices=(512, 256))
ap.add_argument("--tune-fraction", default="100%")
ap.add_argument("--tune-configs", type=int, default=12)
ap.add_argument("--steps-matched", action="store_true", help="1%%/10%% cells cost as much as 100%%")
ap.add_argument("--floor", default=None, help="#196: a smaller rung trains as many steps as this one")
ap.add_argument("--val-reads", type=int, default=None, help="#196: val passes per run (default: every epoch)")
ap.add_argument("--tune-per-rung", action="store_true",
                help="#196: 5-point lr passes at a fifth of the schedule, at every rung, linear + UperNet + B7")
a = ap.parse_args()

R = P["anchor"]["ratio_mps_to_5080"]
m = P["gpu_steps_mps_batch16"]
u = P["upernet_head_alone_mps"]
fwd = m["vitb_forward_ms"] / R
lin = m["linear_probe_ms"] / R
sweep = m["depth_sweep_7probes_one_forward_ms"] / R
upn_train = (m["vitb_forward_ms"] + u[f"ch{a.upernet_channels}_b16_train_ms"]) / R
upn_fwd = (m["vitb_forward_ms"] + u[f"ch{a.upernet_channels}_b16_fwd_ms"]) / R
b3 = (m["finetune_vitb_upernet_ms"] - u["ch512_b16_train_ms"] + u[f"ch{a.upernet_channels}_b16_train_ms"]) / R
LOADER = P["loader"]["task2_box_ms_per_tile_transferred"]
unet = P["b7_unet_5080"]

#: Assumed, not measured: DataLoader start-up for the train and val passes, per epoch (the
#: laptop's spawn fit is 2.0 s at 4 workers), and a per-cell fixed cost (encoder load, scoring
#: set-up; B9's checkpoint read is 1.5 GB).
OVER, CELL_FIXED = 2.0, 60.0
VAL_N, TEST_N = 2000, 4523
FR = {"1%": 200, "10%": 2000, "100%": 20000}

KINDS = {
    "linear": dict(ep=30, tr=lin, val=lin, test=lin),
    "upernet": dict(ep=40, tr=upn_train, val=upn_fwd, test=upn_fwd),
    "b3": dict(ep=50, tr=b3, val=upn_fwd, test=upn_fwd),
    "unet": dict(ep=40, tr=unet["train_ms_per_tile"], val=None, test=unet["train_ms_per_tile"] / 3),
    "concat": dict(ep=30, tr=2 * fwd + (lin - fwd), val=2 * fwd, test=2 * fwd),
    "sweep": dict(ep=30, tr=sweep, val=sweep, test=sweep),
}


def cell(kind: str, frac: str, schedule: float = 1.0, test: bool = True) -> float:
    """Minutes for one cell: epochs x (train + val + start-up) + one test scoring pass + fixed.

    With --val-reads, val is read that many times per run rather than every epoch, and half of the
    per-epoch start-up goes with it. `schedule` < 1 is a tuning run cut to that share of its steps.
    """
    k = KINDS[kind]
    n = FR["100%" if a.steps_matched else frac]
    if a.floor:
        n = max(n, FR[a.floor])
    val_s = unet["val_s_per_epoch"] if k["val"] is None else VAL_N * k["val"] / 1e3
    train_s = n * max(k["tr"], LOADER) / 1e3
    ep = k["ep"] * schedule
    if a.val_reads:
        reads = max(1, round(a.val_reads * schedule))
        run_s = ep * (train_s + OVER / 2) + reads * (val_s + OVER / 2)
    else:
        run_s = ep * (train_s + val_s + OVER)
    return (run_s + (TEST_N * k["test"] / 1e3 if test else 0) + CELL_FIXED) / 60


def tuning() -> tuple[str, float]:
    if not a.tune_per_rung:
        return (f"{a.tune_configs} linear + {a.tune_configs} UperNet configs at {a.tune_fraction}",
                a.tune_configs * (cell("linear", a.tune_fraction) + cell("upernet", a.tune_fraction)))
    return ("5 lr points x a fifth of the schedule x 3 rungs, linear + UperNet on B6, B7 on itself",
            sum(5 * cell(k, f, schedule=0.2, test=False) for k in ("linear", "upernet", "unet") for f in FR))


rows = [
    ("Head tuning on B6/val", *tuning()),
    ("Linear-probe grid", "6 encoders x 3 fractions x 3 seeds = 54",
     sum(18 * cell("linear", f) for f in FR)),
    ("UperNet grid", "6 encoders x 3 fractions x 3 seeds = 54",
     sum(18 * cell("upernet", f) for f in FR)),
    ("B7 UNet", "3 fractions x 3 seeds = 9", sum(3 * cell("unet", f) for f in FR)),
    ("B3 fine-tune", "100%, 2 seeds", 2 * cell("b3", "100%")),
    ("B8 + B4-ALT", "linear, 100%, 3 seeds each = 6", 6 * cell("linear", "100%")),
    ("B5-ALT", "UperNet, 100%, 3 seeds", 3 * cell("upernet", "100%")),
    ("Depth sweep", "6 encoders, 7 probes off one forward, 100%, 1 seed", 6 * cell("sweep", "100%")),
    ("Concatenation probe", "100%, 3 seeds", 3 * cell("concat", "100%")),
    ("§5 forward passes", "6 encoders x 4 analyses x 27k tiles", 6 * 4 * 27000 * fwd / 1e3 / 60),
    ("Sensitivity rescore", "4 label variants x B1/B4 x 3 seeds over test", 24 * TEST_N * lin / 1e3 / 60),
]
total = sum(r[2] for r in rows)
for name, count, mins in rows:
    print(f"| {name} | {count} | {mins:,.0f} |")
print(f"| **Total** | | **{total:,.0f} min = {total / 60:.1f} h, ${total / 60 * 0.4016:.2f}** |")
print(f"per-cell minutes: " + "; ".join(
    f"{k} " + "/".join(f"{cell(k, f):.1f}" for f in FR) for k in ("linear", "upernet", "unet")))
print(f"5080 ms/tile: fwd {fwd:.2f} linear {lin:.2f} upernet {upn_train:.2f} (fwd {upn_fwd:.2f}) "
      f"b3 {b3:.2f} unet {unet['train_ms_per_tile']} loader {LOADER}")
print(f"laptop at x{R}: ~{total * R / 60:,.0f} h")
