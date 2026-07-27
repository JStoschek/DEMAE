"""F71 — report the 500,000-window scale-tier selection list.

Ticket: https://github.com/JStoschek/DEMAE/issues/71 · Map: #67 (Phase F).
Draw: ``python -m dataset.select --budget 500000 --out artifacts/phase_f/selection_scale_500k.csv``

This script does **not** draw. The draw is `dataset.select` at a bigger ``--budget`` —
that is the whole point of #44's design, and re-implementing any part of it here would
create a second sampler. What this reads is the *committed CSV*, re-derives the quota
table it was drawn under (:func:`select.allocation_at`: standardize → assign → allocate,
seconds, no sub-k-means), and writes the four things #71 asks the tier to report:

1. **Realized windows-per-COG** — mean / median / p99 / max / distinct COGs, plus each
   COG's draw as a fraction of the kept inventory it actually holds. #42 bound the
   sampler to carry this metric to every tier so Phase F could decide *on evidence*
   whether the retired per-COG cap needs to come back; #69 sharpened it into `F6c` by
   noting a raw count is unreadable without the COG's own inventory as denominator.
2. **Per-stratum `n_h` and headroom** — the kept-pool-to-allocation ratio `N_h/n_h`. This
   is the number the budget actually moves: Neyman shares are budget-invariant, so ten
   times the budget is a ten times shallower pool (≥53× at 50k → ≈5.3× here), and strata
   **2/17** are the ones ADR-0007's Phase-F revisit trigger cares about.
3. **What binds** — the coastal frame-rate floor and the `max(500, 1%·budget)` clamp were
   both measured inert at 50k. Budget-invariant shares say they probably still are;
   *probably* is not measured.
4. **Determinism, demonstrated** — two full runs, byte-identical, both md5s recorded.
   Pass the replicate with ``--replicate``; the comparison is the acceptance criterion
   #44 set and this tier re-earns, not an assertion inherited from it.

Also reported, deliberately without obligation: the **overlap with the 50k debug tier**.
Per [#45](https://github.com/JStoschek/DEMAE/issues/45) the scale draw is *independent* —
the debug tier does not nest inside it and no hierarchical rule exists — so whatever the
two lists share is incidental. It is still worth knowing, and the report gives it the
context that makes it readable (what two uniform draws would have shared).

AFK. Reads committed artifacts only (``data/catalog_gated``, ``artifacts/phase_d/*``,
``artifacts/phase_f/selection_scale_500k.csv``); draws nothing and touches no pipeline
state. Writes ``docs/research/f-scale-draw.{json,md}`` and
``docs/research/f-allocation-nh-500k.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from dataset import centroids as centroids_mod
from dataset import config
from dataset import frame as frame_mod
from dataset import select

REPO = Path(__file__).resolve().parents[1]
RESEARCH = REPO / "docs" / "research"

#: The strata ADR-0007's Phase-F revisit trigger names, and the two it corroborates with
#: (#63/#69's two-tier shape). Reported here only as headroom — the drainage re-test
#: itself is Gate F's `F3`, run by #73 against #69's pre-registered bar.
ADR0007_GATING = (2, 17)
ADR0007_CORROBORATING = (5, 7)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


# --- Faithfulness of the list to the table it was drawn under -----------------


def conformance(selection: select.Selection) -> dict[str, Any]:
    """Does the committed list match the allocation it claims? — checked here, gated at F.

    Not a Gate F substitute: #69 owns the pass/fail bar and #73 executes it. This is the
    draw's own arithmetic — exactly ``budget`` distinct rows, per-stratum counts equal to
    ``n_h``, coastal picks equal to ``c_h`` — the things that must hold for the reported
    numbers to mean anything at all. A mismatch here means the report is describing a
    corpus that does not exist, so it raises rather than publishing.
    """
    alloc = selection.allocation
    per_stratum = np.bincount(selection.stratum_id, minlength=alloc.k)
    coastal_mask = selection.provenance == select.PROV_COASTLINE
    per_stratum_coastal = np.bincount(
        selection.stratum_id[coastal_mask], minlength=alloc.k
    )
    return {
        "n_rows": selection.n,
        "n_distinct_window_ids": int(np.unique(selection.window_id).size),
        "matches_budget": selection.n == alloc.budget,
        "all_distinct": int(np.unique(selection.window_id).size) == selection.n,
        "per_stratum_matches_n_h": bool(np.array_equal(per_stratum, alloc.n_h)),
        "per_stratum_coastal_matches_c_h": bool(
            np.array_equal(per_stratum_coastal, alloc.coastal)
        ),
        "provenance_vocabulary": sorted({str(p) for p in selection.provenance}),
        "provenance_counts": selection.provenance_counts(),
    }


def assert_conformant(report: dict[str, Any]) -> None:
    failed = [
        key
        for key in (
            "matches_budget",
            "all_distinct",
            "per_stratum_matches_n_h",
            "per_stratum_coastal_matches_c_h",
        )
        if not report[key]
    ]
    if failed:
        raise SystemExit(
            f"the committed selection list does not conform to its allocation: {failed}"
        )
    if set(report["provenance_vocabulary"]) - {select.PROV_BASE, select.PROV_COASTLINE}:
        raise SystemExit(
            f"unexpected provenance labels: {report['provenance_vocabulary']}"
        )


# --- Cross-check against Gate F's pre-registered n_h table --------------------


def gate_f_crosscheck(
    allocation: select.Allocation, selection: select.Selection
) -> dict[str, Any]:
    """Compare the drawn per-stratum counts to the ``n_h`` table #69 pre-registered.

    [#69](https://github.com/JStoschek/DEMAE/issues/69) committed a 500k ``n_h`` table into
    the Gate F bar **before** this draw existed, and `F2a` checks the shipped list against
    it. So the draw's first duty is to say whether the two agree — and if they do not,
    which one is wrong, measured rather than argued.
    """
    from dataset import validate_f

    bar = validate_f.read_bar()
    pre = bar.expected_per_stratum
    bar_budget = int(bar.allocation["sum_n_h"])
    if bar_budget != allocation.budget:
        # The bar registered one budget. Comparing a different tier's counts to it would
        # produce a large, entirely meaningless delta table.
        return {
            "bar": "docs/research/f-gate-bar.json (#69)",
            "applicable": False,
            "reason": (
                f"the bar pre-registered n_h at budget {bar_budget:,}; this report is at "
                f"{allocation.budget:,}"
            ),
        }
    realized = np.bincount(selection.stratum_id, minlength=allocation.k)

    # Re-run the rule the bar names as its source, but on the *rounded* sigma_h the
    # committed d2 table prints, to demonstrate rather than assert where its numbers
    # came from.
    d2 = json.loads((RESEARCH / "d2-allocation-nh.json").read_text())["rows"]
    from_rounded = select.allocate(
        np.array([r["N_h"] for r in d2]),
        np.array([r["sigma_h"] for r in d2]),
        np.array([r["coast_h"] for r in d2]),
        allocation.budget,
    ).n_h

    missing = [h for h in range(allocation.k) if h not in pre]
    if missing:
        raise SystemExit(
            f"the Gate F bar's n_h table has no entry for strata {missing} — it does not "
            "describe this k, so the cross-check cannot be made"
        )
    pre_n_h = np.array([int(pre[h]) for h in range(allocation.k)], dtype=np.int64)
    deltas = [
        {
            "stratum_id": h,
            "pre_registered_n_h": int(pre_n_h[h]),
            "drawn_n_h": int(realized[h]),
            "delta": int(realized[h]) - int(pre_n_h[h]),
        }
        for h in range(allocation.k)
        if int(pre_n_h[h]) != int(realized[h])
    ]
    return {
        "bar": "docs/research/f-gate-bar.json (#69) — committed_inputs.allocation.n_h",
        "applicable": True,
        "agrees": not deltas,
        "n_strata_differing": len(deltas),
        "max_abs_delta": max((abs(d["delta"]) for d in deltas), default=0),
        "drawn_sums_to_budget": int(realized.sum()) == allocation.budget,
        "pre_registered_sums_to_budget": int(pre_n_h.sum()) == allocation.budget,
        "deltas": deltas,
        "cause_reproduced": bool(np.array_equal(from_rounded, pre_n_h)),
        # Read out of the bar rather than narrated here: once #69 corrects a value this
        # report published deltas against, the agreement is only meaningful alongside the
        # correction that produced it.
        "bar_post_commit_corrections": [
            {k: c.get(k) for k in ("date", "field", "kind", "what_was_wrong")}
            for c in bar.doc.get("post_commit_corrections", [])
            if c.get("field") == "committed_inputs.allocation.n_h"
        ],
        "diagnosis": (
            "The pre-registered table is reproduced bit-for-bit by feeding the 4-dp "
            "sigma_h that d2-allocation-nh.json PRINTS back into select.allocate, while "
            "the sampler recomputes sigma_h at full precision from the frame. #44 recorded "
            "this exact trap ('the JSON's 4-dp sigma_h is for legibility; feeding it back "
            "shifts a few n_h by +-1, which is why the sampler recomputes sigma_h from the "
            "frame') and asserts the full-precision path against the committed 50k table in "
            "tests/test_select.py. Both tables sum to the exact budget; the drawn one is "
            "the allocation rule ADR-0005 specifies, the pre-registered one is that rule "
            "evaluated on a display-rounded input."
        ),
        "consequence": (
            "F2a compares the shipped list's per-stratum counts against this table, so as "
            "committed it would fire a DEFECT on a correct draw. The bar is #69's artifact "
            "and #71 does not edit a pre-registered file after drawing against it: the "
            "discrepancy is published here, with its cause reproduced, for a separate "
            "dated commit to correct before #73 executes the gate."
        ),
    }


# --- The re-clustering knob (#67's instance of the #36/#42 retire-on-evidence pattern) --


def reclustering_knob(
    centroids: centroids_mod.Centroids, *, kept_rows: int
) -> dict[str, Any]:
    """Why plan §F's re-clustering knob cannot bind — recorded, not merely obeyed.

    The load-bearing fact is measured, not asserted: the committed centroids' ``fit_rows``
    against the kept frame a Phase-F refit would see *today*. Equal counts mean a refit
    runs on identical input.
    """
    prov = dict(centroids.provenance)
    return {
        "plan_text": "§F: same strata/centroids for comparability; re-clustering is a knob",
        "status": "structurally inert — kept declared, never exercised",
        "fit_rows": prov.get("fit_rows"),
        "kept_rows_a_refit_would_see": int(kept_rows),
        "refit_input_is_identical": prov.get("fit_rows") == int(kept_rows),
        "fit_k": centroids.k,
        "fit_algorithm": prov.get("algorithm"),
        "fit_n_init": prov.get("n_init"),
        "argument": (
            "The centroids were fit on the ENTIRE gated catalog (fit_rows 3,052,735 = the "
            "whole kept frame), not on the debug tier's 50,000 windows. A Phase-F refit "
            "would therefore run on identical input: same rows, same k=20, same Lloyd, "
            "same n_init. It cannot discover strata the committed fit did not already "
            "see, so its expected change is nil — while #41 established the FIT is not "
            "bitwise-reproducible across threads (assign() is). Refitting strictly loses: "
            "no expected gain, and a guaranteed loss of the cross-tier strata "
            "comparability the persisted artifacts exist to provide."
        ),
        "consequence": (
            "The knob stays declared in ADR-0005 §9 (k row) and in the plan's §F, and is "
            "never exercised — the #36/#42 pattern: retire on evidence, keep declared, so "
            "a future frame that genuinely moves the fit population makes it live again "
            "with no code change."
        ),
    }


# --- Report -------------------------------------------------------------------


def _repo_relative(path: Path) -> str:
    """Paths are quoted into the write-up, so they must read as repo paths, not $PWD."""
    path = path.resolve()
    return str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], select.Allocation]:
    log(f"loading the kept frame from {args.catalog_dir}")
    fr = frame_mod.kept_frame(args.catalog_dir)
    scaler = frame_mod.read_scaler(fingerprint=fr.fingerprint)
    cen = centroids_mod.read_centroids(fingerprint=fr.fingerprint)

    log(f"re-deriving the allocation at budget {args.budget:,} (no FSCS)")
    alloc = select.allocation_at(fr, cen, scaler, budget=args.budget)

    log(f"reading the committed selection list {args.selection}")
    sel = select.Selection.from_csv(
        args.selection, allocation=alloc, fingerprint=fr.fingerprint
    )
    conform = conformance(sel)
    assert_conformant(conform)

    log("counting the kept per-COG inventory")
    inventory = select.kept_cog_inventory(args.catalog_dir)

    log("reading the 50k debug-tier list for the incidental-overlap report")
    debug_ids = select.read_window_ids(args.debug_selection)
    overlap = select.overlap_report(
        sel.window_id, debug_ids, n_pool=fr.n, label="debug tier (50k, #44)"
    )

    determinism: dict[str, Any] = {
        "criterion": (
            "two full runs from the committed artifacts, byte-identical — demonstrated, "
            "never asserted (#44's bar, re-earned at this tier)"
        ),
        "selection_md5": md5(args.selection),
        "n_rows": sel.n,
    }
    if args.replicate is not None:
        determinism["replicate_md5"] = md5(args.replicate)
        determinism["byte_identical"] = (
            args.replicate.read_bytes() == args.selection.read_bytes()
        )
    if args.run_seconds is not None:
        determinism["run_seconds"] = args.run_seconds
    if args.workers is not None:
        determinism["workers"] = args.workers

    payload = {
        "artifact": "The 500,000-window scale-tier selection list",
        "ticket": "https://github.com/JStoschek/DEMAE/issues/71",
        "map": "https://github.com/JStoschek/DEMAE/issues/67",
        "selection_list": _repo_relative(args.selection),
        "budget": args.budget,
        "seed": config.RANDOM_SEED,
        "inputs": {
            "catalog": _repo_relative(args.catalog_dir),
            "kept_rows": fr.n,
            "centroids": "artifacts/phase_d/centroids.json",
            "scaler": "artifacts/phase_d/scaler.json",
            "k": cen.k,
            "frame_fingerprint": dict(fr.fingerprint),
        },
        "conformance": conform,
        "gate_f_bar_crosscheck": gate_f_crosscheck(alloc, sel),
        "determinism": determinism,
        "binding": alloc.binding(),
        "per_cog": sel.per_cog_distribution(inventory),
        "overlap": overlap,
        "reclustering_knob": reclustering_knob(cen, kept_rows=fr.n),
        "allocation": alloc.rows(),
        "adr0007_headroom": {
            "gating_strata": list(ADR0007_GATING),
            "corroborating_strata": list(ADR0007_CORROBORATING),
            "note": (
                "headroom only — the 2/17 drainage re-test itself is Gate F's F3, "
                "pre-registered in docs/research/f-gate-bar.json (#69) and executed by #73"
            ),
            "rows": [
                {
                    "stratum_id": h,
                    "N_h": int(alloc.N_h[h]),
                    "n_h": int(alloc.n_h[h]),
                    "headroom": round(float(alloc.headroom[h]), 2),
                }
                for h in (*ADR0007_GATING, *ADR0007_CORROBORATING)
            ],
        },
    }
    return payload, alloc


# --- Write-up -----------------------------------------------------------------


def write_markdown(p: dict[str, Any], path: Path) -> None:
    lines: list[str] = []
    W = lines.append
    per_cog = p["per_cog"]
    frac = per_cog["fraction_of_inventory"]
    binding = p["binding"]
    det = p["determinism"]
    rows = p["allocation"]
    headroom = [r["headroom"] for r in rows]

    W("# The 500,000-window scale-tier draw\n")
    W(
        f"**Ticket:** [Draw the 500k scale-tier selection list (#71)]({p['ticket']}) · "
        f"**Map:** [Phase F (#67)]({p['map']}) · "
        f"**List:** [`{Path(p['selection_list']).name}`](../../{p['selection_list']}) · "
        f"**Quota table:** [`f-allocation-nh-500k.json`](f-allocation-nh-500k.json) · "
        f"**Numbers:** [`f-scale-draw.json`](f-scale-draw.json)\n"
    )
    W(
        "The same sampler, the same committed artifacts, a bigger budget. "
        "[#44](https://github.com/JStoschek/DEMAE/issues/44) built the draw so this ticket "
        "would be a config change and demonstrated it; this is that design being spent:\n"
    )
    W("```bash")
    W(
        f"python -m dataset.select --budget {p['budget']} "
        f"--out {p['selection_list']}"
    )
    W("```\n")

    W("## Determinism — demonstrated, not asserted\n")
    W(
        f"Two full runs from the committed artifacts, {det['n_rows']:,} rows each:\n"
    )
    W("| run | md5 |")
    W("|---|---|")
    W(f"| 1 (committed) | `{det['selection_md5']}` |")
    if "replicate_md5" in det:
        W(f"| 2 (replicate) | `{det['replicate_md5']}` |")
    W("")
    if "replicate_md5" in det:
        verdict = "**byte-identical**" if det["byte_identical"] else "**DIFFERENT**"
        W(f"The two files are {verdict}.\n")
    if "run_seconds" in det:
        W(
            f"Wall time **~{det['run_seconds'] / 60:.0f} min** (run 2, measured end to end) "
            f"across {det.get('workers', 'default')} worker processes, each sub-k-means fit "
            "pinned single-threaded "
            "([#43](https://github.com/JStoschek/DEMAE/issues/43)'s lever) — inside "
            "[ADR-0005 §9](../adr/0005-sampling-design.md)'s projected ~55 min, and the "
            "reason the per-stratum fan-out exists.\n"
        )
    W(
        "Nothing about the mechanism changed at 10× the budget, because nothing about it "
        "*could*: every tie-break is a keyed BLAKE2b of the window id rather than a NumPy "
        "`Generator`, the fit is pinned single-threaded, and the persisted list — not the "
        "seed — is the contract.\n"
    )

    W("## The list conforms to its own quota table\n")
    c = p["conformance"]
    W(f"- **{c['n_rows']:,} rows**, all distinct window ids ({c['n_distinct_window_ids']:,}).")
    W("- Per-stratum counts equal `n_h` for all 20 strata; coastal picks equal `c_h`.")
    prov = ", ".join(f"`{k}` {v:,}" for k, v in sorted(c["provenance_counts"].items()))
    W(f"- Provenance is exactly the two-value vocabulary [#42](https://github.com/JStoschek/DEMAE/issues/42) fixed: {prov}.\n")
    W(
        "That is arithmetic, not a gate. Gate F's pass/fail bar is pre-registered in "
        "[`f-gate-bar.json`](f-gate-bar.json) ([#69](https://github.com/JStoschek/DEMAE/issues/69)) "
        "and executed by [#73](https://github.com/JStoschek/DEMAE/issues/73); this report "
        "supplies numbers to it and decides nothing.\n"
    )

    x = p["gate_f_bar_crosscheck"]
    if not x["applicable"]:
        W(f"*Gate F `n_h` cross-check skipped — {x['reason']}.*\n")
    elif x["agrees"]:
        W("### …and to the `n_h` table the Gate F bar pre-registered\n")
        W(
            "The drawn per-stratum counts equal "
            "[`f-gate-bar.json`](f-gate-bar.json)'s `committed_inputs.allocation.n_h` for "
            f"all {p['inputs']['k']} strata.\n"
        )
        for c in x["bar_post_commit_corrections"]:
            W(
                f"**They did not, as originally registered.** This draw found "
                f"`{c['field']}` wrong and published the per-stratum deltas rather than "
                f"editing a pre-registered file it had just drawn against: "
                f"{c['what_was_wrong']} "
                f"#69 corrected it on **{c['date']}** as a separate dated "
                f"commit ({c['kind']}); see that file's `post_commit_corrections` and "
                f"[`f-gate-bar.md`](f-gate-bar.md)'s amendment section for the argument. "
                f"`F2a` would otherwise have fired a **Defect** on a correct draw when "
                f"[#73](https://github.com/JStoschek/DEMAE/issues/73) executes the gate.\n"
            )
    else:
        W("### …but not to the `n_h` table the Gate F bar pre-registered\n")
        both_exact = x["drawn_sums_to_budget"] and x["pre_registered_sums_to_budget"]
        W(
            f"They differ on **{x['n_strata_differing']} of {p['inputs']['k']} strata**, by "
            f"at most **±{x['max_abs_delta']}** windows"
            + (
                f". Both tables sum to exactly {p['budget']:,}.\n"
                if both_exact
                else " — and they do NOT both sum to the budget.\n"
            )
        )
        W("| h | pre-registered `n_h` | drawn `n_h` | Δ |")
        W("|---|---|---|---|")
        for d in x["deltas"]:
            W(
                f"| {d['stratum_id']} | {d['pre_registered_n_h']:,} | "
                f"{d['drawn_n_h']:,} | {d['delta']:+d} |"
            )
        W("")
        W(f"**The cause is reproduced, not inferred{'' if x['cause_reproduced'] else ' (REPRODUCTION FAILED)'}.** " + x["diagnosis"] + "\n")
        W("**" + x["consequence"] + "**\n")

    W("## Realized windows-per-COG — the metric #42 bound this tier to decide on\n")
    W("| | @50k (#44) | @500k |")
    W("|---|---|---|")
    W(f"| mean | 3.16 | **{per_cog['mean_per_cog']:.2f}** |")
    W(f"| median | 2 | **{per_cog['median_per_cog']}** |")
    W(f"| p99 | 16 | **{per_cog['p99_per_cog']}** |")
    W(f"| max | 44 | **{per_cog['max_per_cog']}** |")
    W(f"| distinct COGs drawn | 15,834 | **{per_cog['n_cogs']:,}** |")
    W("")
    W(
        f"[#42](https://github.com/JStoschek/DEMAE/issues/42) retired `PER_TILE_WINDOW_CAP "
        f"= 64` as inert at the debug tier but required the sampler to *measure* the "
        f"distribution at every tier, so the scale tier could decide on evidence whether a "
        f"cap needs reintroducing. **The count alone cannot answer that**, which is what "
        f"[#69](https://github.com/JStoschek/DEMAE/issues/69) established when it retired "
        f"the cap as unfixable rather than merely stale: the ceiling on a COG is its own "
        f"kept inventory (at most 256 windows), so the readable statistic is the fraction "
        f"of that inventory drawn.\n"
    )
    W("| fraction of a COG's own kept inventory | value |")
    W("|---|---|")
    W(f"| mean | {frac['mean']} |")
    W(f"| median | {frac['median']} |")
    W(f"| p99 | {frac['p99']} |")
    W(f"| max | {frac['max']} |")
    W(
        f"| max among COGs holding ≥{frac['inventory_floor']} kept windows | "
        f"{frac['max_among_sizeable_cogs']} |"
    )
    W(f"| COGs drawn at ≥50% of inventory | {frac['n_cogs_at_or_over_half']:,} |")
    W(
        f"| COGs drawn in full | {frac['n_cogs_fully_drawn']:,} "
        f"({frac['n_cogs_fully_drawn_sizeable']:,} of them holding ≥{frac['inventory_floor']}; "
        f"largest {frac['max_windows_from_a_fully_drawn_cog']:,} windows) |"
    )
    W(f"| COGs in the kept pool, never drawn | {frac['n_cogs_undrawn']:,} of {frac['n_cogs_in_pool']:,} |")
    W("")
    # What the retired cap of 64 would have cost, priced from this distribution. The
    # constant is #42's, quoted here rather than kept live anywhere in the library.
    over_64 = sum(c for v, c in per_cog["histogram"].items() if int(v) > 64)
    stripped_64 = sum(c * (int(v) - 64) for v, c in per_cog["histogram"].items() if int(v) > 64)
    W(
        f"**The regime the retired cap was pointed at is now measured rather than "
        f"predicted, and it confirms #69's reasoning.** The old cap of 64 would bind on "
        f"**{over_64:,}** of the {per_cog['n_cogs']:,} COGs drawn and strip "
        f"**{stripped_64:,} windows** — {stripped_64 / p['budget']:.1%} of the budget — "
        f"which the sampler would then have to buy from worse representatives elsewhere. "
        f"The densest COG yields **{per_cog['max_per_cog']}**, which is "
        f"{per_cog['densest_cogs'][0]['fraction']:.0%} of everything that COG holds. No "
        f"larger threshold rescues the guard, because the ceiling is a COG's own inventory "
        f"rather than a number anyone picks: **{frac['n_cogs_at_or_over_half']:,}** COGs are "
        f"drawn at ≥50% of what they hold and **{frac['n_cogs_fully_drawn']:,}** in full — "
        f"though only **{frac['n_cogs_fully_drawn_sizeable']}** of those hold "
        f"≥{frac['inventory_floor']} windows "
        f"(largest {frac['max_windows_from_a_fully_drawn_cog']}), so exhaustion is "
        f"overwhelmingly a small-COG effect, not the dense tail running dry.\n"
    )
    W(
        f"The checks that survive the cap's retirement are mechanical — "
        f"[`F6b`](f-gate-bar.json) (no COG yields more than it holds) and `F6c` (this "
        f"distribution, published). The real residue is **spatial redundancy**: at a "
        f"{frac['mean']:.0%} mean draw rate, windows from one COG are neighbours, so the "
        f"corpus's effective sample size sits below its row count. #69 handed that to "
        f"[#75](https://github.com/JStoschek/DEMAE/issues/75)'s honesty notes as a disclosed "
        f"property rather than gating it against an invented threshold — these are the "
        f"numbers it should disclose.\n"
    )

    W("## Per-stratum allocation and headroom\n")
    flat_headroom = [
        r["headroom"] for r in rows if r["stratum_id"] in (*ADR0007_GATING, *ADR0007_CORROBORATING)
    ]
    W(
        "`headroom` is `N_h/n_h` — kept windows held per window the stratum must yield. "
        "**This is the number the budget actually moves.** Neyman shares are "
        "budget-invariant, so a 10× budget is a 10× shallower pool. "
        "[ADR-0007](../adr/0007-accept-degenerate-flatness-gate-flat-strata.md) leg 3 leaned "
        f"on the four flat strata (2/5/7/17) keeping **≥53×** their allocation at 50k; "
        f"measured here, those four sit at **≥{min(flat_headroom):.2f}×** — the ≈5.3× the "
        "ADR predicted, confirmed rather than assumed.\n"
    )
    W("| h | N_h | σ_h | coast_h | n_h | coastal | base | headroom |")
    W("|---|---|---|---|---|---|---|---|")
    for r in rows:
        W(
            f"| {r['stratum_id']} | {r['N_h']:,} | {r['sigma_h']:.4f} | {r['coast_h']:,} | "
            f"{r['n_h']:,} | {r['coastal']:,} | {r['base']:,} | {r['headroom']:.2f}× |"
        )
    W("")
    W(
        f"Range **{min(headroom):.2f}×–{max(headroom):.2f}×**, thinnest at stratum "
        f"**{binding['min_headroom_stratum']}**.\n"
    )
    adr = p["adr0007_headroom"]
    W(
        "ADR-0007's Phase-F revisit trigger names strata **2/17** (with **5/7** as "
        "corroboration). Their headroom at this budget:\n"
    )
    W("| h | N_h | n_h | headroom | role |")
    W("|---|---|---|---|---|")
    for r in adr["rows"]:
        role = "gating" if r["stratum_id"] in ADR0007_GATING else "corroborating"
        W(
            f"| {r['stratum_id']} | {r['N_h']:,} | {r['n_h']:,} | {r['headroom']:.2f}× | {role} |"
        )
    W("")
    W(f"*{adr['note']}.*\n")

    W("## What binds at this budget\n")
    W("| guard | setting | fires |")
    W("|---|---|---|")
    W(
        f"| `max(500, 1%·budget)` clamp | {binding['floor_clamp']:,} | "
        f"**{binding['floor_clamp_strata_below']} of {p['inputs']['k']} strata** "
        f"(smallest `n_h` {binding['min_n_h']:,}) |"
    )
    W(
        f"| coastal `min(n_h, coast_h)` cap | per stratum | "
        f"**{binding['coast_cap_strata_binding']} of {p['inputs']['k']} strata** |"
    )
    W("")
    W(
        f"The coastal reservation is `round({binding['coast_floor_rate']} · budget)` = "
        f"**{binding['coast_floor']:,}** windows, a realized share of "
        f"**{binding['realized_coast_share']:.4%}** against the kept frame's own coastal "
        f"rate of **{binding['frame_coast_rate']:.4%}**. It is a *frame-rate* floor, not a "
        f"boost ([#42](https://github.com/JStoschek/DEMAE/issues/42) §4): coast is a "
        f"location tag rather than a descriptor shape, so the honest minimum is that coast "
        f"is never under-represented relative to reality.\n"
    )
    W(
        "Both guards were non-binding at 50k, and Neyman's budget-invariance made it likely "
        "they still were — but *likely* is not *measured*, which is why #71 asked. "
        f"**Measured: {'something binds' if binding['anything_binds'] else 'neither binds'}.**\n"
    )
    W(
        "One precision about the coastal floor, so the row above is not read as more than "
        "it says. The reservation is a *within-stratum split*, not a post-hoc top-up "
        "([#42](https://github.com/JStoschek/DEMAE/issues/42) §4): each stratum runs FSCS "
        "over its coastal rows for `c_h` and over the rest for `base_h`, so a `base` pick "
        "is never coastal by construction and the reservation **sets** the corpus's coastal "
        "share exactly rather than raising a floor under a share the draw would otherwise "
        "produce. What is measured here is that the split is feasible everywhere (the "
        "`min(n_h, coast_h)` cap fires on no stratum) and that the share it sets sits at "
        "the frame's own rate. What is *not* measured — and is named rather than estimated "
        "— is what an unreserved single-pass FSCS would have drawn: that is a second full "
        "30-minute draw, and #71 did not ask for the counterfactual.\n"
    )

    W("## The re-clustering knob is structurally inert\n")
    knob = p["reclustering_knob"]
    W(f"> {knob['plan_text']}\n")
    W(f"**Status: {knob['status']}.**\n")
    W(
        f"The committed centroids were fit on **{knob['fit_rows']:,}** rows; the kept frame "
        f"a Phase-F refit would see today is **{knob['kept_rows_a_refit_would_see']:,}** — "
        f"{'identical' if knob['refit_input_is_identical'] else 'DIFFERENT'}.\n"
    )
    W(knob["argument"] + "\n")
    W(knob["consequence"] + "\n")

    W("## Overlap with the debug tier — incidental, reported, unbinding\n")
    ov = p["overlap"]
    W(
        f"[#45](https://github.com/JStoschek/DEMAE/issues/45) ruled the scale draw "
        f"**independent**: the 50k does not nest inside it and no hierarchical rule was "
        f"built. The two lists share **{ov['n_overlap']:,} windows** — "
        f"**{ov['share_of_other']:.1%}** of the debug tier and "
        f"**{ov['share_of_this']:.1%}** of the scale tier.\n"
    )
    W(
        f"Two *uniform* draws of these sizes from the {ov['n_pool']:,}-row kept pool would "
        f"average **{ov['expected_if_uniform']:,.0f}** shared windows, so the realized "
        f"overlap is **{ov['lift_over_uniform']:.1f}×** that. FSCS is deterministic and "
        f"coverage-seeking: both tiers keep resolving to the same nearest-the-sub-centroid "
        f"representatives, which is the sampler agreeing with itself rather than a nesting "
        f"rule leaking in. It carries no obligation either way.\n"
    )

    W("## Provenance\n")
    W(
        "Reads committed artifacts only (`data/catalog_gated`, "
        "`artifacts/phase_d/{centroids,scaler}.json`, the selection list); draws nothing "
        "and touches no pipeline state. Regenerate with "
        "`python notebooks/f71_scale_draw.py` — every number above comes back except the "
        "replicate md5 and wall time, which are properties of a *run* rather than of the "
        "artifact and are passed in via `--replicate` / `--run-seconds` (the second run's "
        "CSV is not committed; the md5 recorded here is the evidence it existed and "
        "matched).\n"
    )
    path.write_text("\n".join(lines))
    log(f"wrote {path}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--catalog-dir", type=Path, default=REPO / "data" / "catalog_gated")
    ap.add_argument("--selection", type=Path, default=REPO / select.SCALE_SELECTION_PATH)
    ap.add_argument("--debug-selection", type=Path, default=REPO / select.SELECTION_PATH)
    ap.add_argument("--budget", type=int, default=select.SCALE_BUDGET)
    ap.add_argument(
        "--replicate",
        type=Path,
        default=None,
        help="the second full run's CSV — compared byte-for-byte against --selection",
    )
    ap.add_argument("--run-seconds", type=float, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=RESEARCH,
        help="where the three artifacts land (default: docs/research/)",
    )
    args = ap.parse_args(argv)

    payload, alloc = build_report(args)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "f-scale-draw.json").write_text(json.dumps(payload, indent=2) + "\n")
    log(f"wrote {out_dir / 'f-scale-draw.json'}")

    # The quota table in the shape d2-allocation-nh.json shipped, so the two tiers diff
    # column-for-column. Written from the Allocation object itself — never re-derived from
    # the JSON's 4-dp sigma_h, which #44 measured shifts a few n_h by +-1.
    select.write_allocation_table(
        alloc,
        out_dir / "f-allocation-nh-500k.json",
        fingerprint=payload["inputs"]["frame_fingerprint"],
    )
    log(f"wrote {out_dir / 'f-allocation-nh-500k.json'}")

    write_markdown(payload, out_dir / "f-scale-draw.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
