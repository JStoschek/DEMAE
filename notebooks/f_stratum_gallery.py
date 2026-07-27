"""F4 (#73) — the stratum gallery, re-rendered from the shipped 500k scale corpus.

Same 12 tiles × 20 strata as [#53](https://github.com/JStoschek/DEMAE/issues/53), drawn
from a corpus ten times deeper — and, unlike #53's, **gating nothing**.

## Why it still exists once it stopped gating

[#69](https://github.com/JStoschek/DEMAE/issues/69) retired E2.4's pass/fail and said why:
that criterion's registered question was whether ``k = 20`` is terrain-meaningful, its knob
was ``k``, and the *same persisted centroids* produce the same partition — so #53/#54's
**20/20 coherent, 0 mixed, 0 incoherent** already answered it, and ``k`` is out of scope
for map [#67](https://github.com/JStoschek/DEMAE/issues/67). Re-rendering draws different
tiles from identical strata.

Three things survive that, and they are the reasons this runs:

* **[#75](https://github.com/JStoschek/DEMAE/issues/75)'s honesty notes need a gallery of
  the corpus that actually shipped**, not of the debug tier.
* **The terrain-family labels are still recorded** — for continuity with #53/#54, and
  because a label written against 50,000 tiles is not evidence about 500,000.
* **It is the only look at the imagery that would catch a corruption preserving
  land-std.** ``F1d_pixel_integrity`` is now the whole-corpus pixel check, and the bar is
  explicit that a horizontal flip would pass it and be obvious by eye.

There is one genuinely new effect to look for: a 10×-deeper draw reaches further from each
centroid, so more peripheral cluster members appear. The bar notes twelve tiles per stratum
has almost no power to detect it — which is part of why the marks do not gate.

## What it reuses, and why that is the safe direction

The rendering is [`e2_stratum_gallery.py`](e2_stratum_gallery.py)'s, imported as a sibling
module (the driver's own directory is ``sys.path[0]``). That is deliberate: the sheets are
a *presentation* of tiles, #75 will read the two galleries side by side, and a second
hand-maintained copy of the layout arithmetic would drift into an unstated difference
between them. The tier-specific half — which corpus, which output paths, and a cover sheet
that does not claim a pass/fail it no longer has — is here.

The draw itself is [`dataset.gallery`](../src/dataset/gallery.py)'s and is unchanged:
:func:`~dataset.gallery.uniform_draw` and :func:`~dataset.gallery.spanning_draw` both order
by the house keyed BLAKE2b under :data:`~dataset.gallery.GALLERY_SEED`, so every sheet
reconstructs from the committed manifest and the seed after the gitignored shards are
gone. The sheets the labels are written against stay lossless PNG for the same reason.

Zero-network. Run:  uv run python notebooks/f_stratum_gallery.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import e2_stratum_gallery as sheets  # the layout arithmetic — see the module docstring
from dataset import gallery, provenance
from dataset.centroids import Centroids
from dataset.frame import METRICS, Scaler

REPO = Path(__file__).resolve().parents[1]
RESEARCH = REPO / "docs" / "research"

COVER = """# F4 stratum gallery — {n_strata} strata × {per_sheet} tiles from the 500k corpus

**Ticket:** [Run the Gate F validation report + gallery (#73)](https://github.com/JStoschek/DEMAE/issues/73) ·
**Map:** [Phase F (#67)](https://github.com/JStoschek/DEMAE/issues/67) ·
**Bar:** [`f-gate-bar.json`](f-gate-bar.json) → `F4_gallery_published` ·
**Labels recorded at:** [Gate F (#74)](https://github.com/JStoschek/DEMAE/issues/74)

> **Published, non-gating.** [#69](https://github.com/JStoschek/DEMAE/issues/69) retired
> E2.4's pass/fail: its registered question was whether `k={n_strata}` is terrain-meaningful,
> its knob was `k`, and the same persisted centroids produce the same partition — so
> [#53](https://github.com/JStoschek/DEMAE/issues/53)/[#54](https://github.com/JStoschek/DEMAE/issues/54)'s
> **20/20 coherent** already answered it, and `k` is out of scope for map #67. There is no
> coherence gate here and no fail condition to trip.

What this gallery is *for* at Gate F: [#75](https://github.com/JStoschek/DEMAE/issues/75)
needs a gallery of the corpus that actually shipped, the terrain-family labels are recorded
for continuity, and this is the only look at scale imagery that would catch a corruption
preserving land-std — which `F1d_pixel_integrity`, the whole-corpus pixel check that took
over F4's former gating role, cannot see.

Corpus `{dataset_version}` · {n_tiles:,} tiles · seed `{seed}` · rendered {today} at
`{code_version}`. The exact tile ids behind every sheet are in
[`gallery.json`](f-gallery/gallery.json), so this gallery is reconstructible after the
shards — which are gitignored, and at 47 GB will not be kept — are gone.

## How the tiles are drawn

Unchanged from [#53](https://github.com/JStoschek/DEMAE/issues/53), so the two galleries are
comparable: grey hillshade, north-west sun at {az:.0f}°/{alt:.0f}°, σ={sigma} px Gaussian
before shading, per-tile vertical exaggeration towards a {target:.0f} m land-std, true
anisotropic pixel spacing. **No colour carries terrain** — magnitude is printed in the
caption, and the coloured bar left of each tile carries absolute elevation on a shared ramp.
Exaggeration is per tile, so **apparent contrast means nothing across tiles**; the captioned
`relief` and `elev` are the magnitudes. The full argument for each choice is in
[`e2-gallery-protocol.md`](e2-gallery-protocol.md), which still describes the *rendering*
faithfully even though its pass/fail clause no longer applies.

## What is new at 500k

A ten-times-deeper draw reaches further from each centroid, so more **peripheral cluster
members** appear than at 50k. The bar records that twelve tiles per stratum has almost no
power to detect that, and that blending in a continuum is expected rather than a failure
([#41](https://github.com/JStoschek/DEMAE/issues/41)) — which is part of why these marks do
not gate. Read the sheets for *what shipped*, not for a verdict.

## The sheets

| stratum | n_h shipped | 12 uniform tiles | diagnostic (interior→edge) |
|---|---:|---|---|
{rows}

**[`coastal-floor.webp`](f-gallery/coastal-floor.webp)** — the {n_coastal:,}
`floor:coastline` tiles as a reporting axis (plan §133).
[#40](https://github.com/JStoschek/DEMAE/issues/40) measured coastline as *not* a stratum
(max purity 4.9%); there is no coastal criterion in the Gate F bar and this sheet does not
create one.

## Terrain-family labels — for #74 to complete

One short terrain-family label per stratum, plus any note worth carrying into #75. **No
mark column**: `coherent` / `mixed` / `incoherent` was E2.4's registered vocabulary and it
belonged to a gate that no longer exists, so recording one here would manufacture a verdict
the bar declines to ask for. Judge each stratum's tiles on their own; do not compare strata.

| stratum | terrain-family label | note |
|---|---|---|
{labels}

*Left blank deliberately — the labels are #74's, and #73 does not pre-empt them.*
"""


def write_cover(path: Path, *, per_stratum: list[dict], n_coastal: int, meta: dict) -> None:
    rows = "\n".join(
        f"| {d['stratum_id']} | {d['n_h']:,} | [`{d['sheet']}`](f-gallery/{d['sheet']}) | "
        f"[`{d['spanning']}`](f-gallery/{d['spanning']}) |"
        for d in per_stratum
    )
    labels = "\n".join(f"| {d['stratum_id']} |  |  |" for d in per_stratum)
    path.write_text(
        COVER.format(
            n_strata=len(per_stratum), per_sheet=gallery.TILES_PER_SHEET, rows=rows,
            labels=labels, n_coastal=n_coastal, az=gallery.AZIMUTH_DEG,
            alt=gallery.ALTITUDE_DEG, sigma=gallery.SMOOTH_SIGMA_PX,
            target=gallery.TARGET_LAND_STD_M, seed=gallery.GALLERY_SEED, **meta,
        )
    )
    print(f"[wrote {path.relative_to(REPO)}]")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--corpus", type=Path, default=REPO / "data" / "corpus_500k")
    p.add_argument("--centroids", type=Path,
                   default=REPO / "artifacts" / "phase_d" / "centroids.json")
    p.add_argument("--scaler", type=Path, default=REPO / "artifacts" / "phase_d" / "scaler.json")
    p.add_argument("--out-dir", type=Path, default=RESEARCH / "f-gallery")
    p.add_argument("--cover", type=Path, default=RESEARCH / "f-gallery.md")
    p.add_argument("--sigma", type=float, default=gallery.SMOOTH_SIGMA_PX,
                   help="Gaussian px before shading; 0 renders the stored field raw.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    m = sheets.load_manifest(args.corpus / "manifest.parquet")
    meta = json.loads((args.corpus / "corpus_meta.json").read_text())
    scaler = Scaler.from_dict(json.loads(args.scaler.read_text()))
    centroids = Centroids.from_dict(json.loads(args.centroids.read_text()))
    read = sheets.ShardReader(args.corpus / "shards")

    strata = m["stratum_id"]
    radii = gallery.centroid_radius(m, strata, scaler=scaler, centroids=centroids)
    pct = gallery.radius_percentile(radii, strata)

    print(f"F4 gallery: {len(m['tile_id']):,} tiles, k={len(np.unique(strata))}, "
          f"seed={gallery.GALLERY_SEED}, sigma={args.sigma} px "
          f"-> {args.out_dir.relative_to(REPO)}  (published, non-gating)")

    per_stratum: list[dict[str, Any]] = []
    drawn: dict[str, list[str]] = {}
    for h in sorted(np.unique(strata).tolist()):
        members = np.flatnonzero(strata == h)
        ids = m["tile_id"][members]
        centre = ", ".join(
            f"{name.split('_')[0][:5]}{centroids.centers[h][j]:+.2f}"
            for j, name in enumerate(METRICS)
        )
        shared = (
            f"n_h={len(members):,}  seed={gallery.GALLERY_SEED}  centroid(z): {centre}\n"
            f"grey hillshade az{gallery.AZIMUTH_DEG:.0f}/alt{gallery.ALTITUDE_DEG:.0f} · "
            f"σ={args.sigma:g} px · per-tile ve→{gallery.TARGET_LAND_STD_M:.0f} m land-std · "
            f"exaggeration differs per tile, so contrast is not comparable"
        )
        sheet, spanning = f"stratum-{h:02d}.png", f"stratum-{h:02d}-spanning.webp"

        drawn[sheet] = sheets.render_sheet(
            members[gallery.uniform_draw(ids)].tolist(), m, pct, read,
            title=f"Stratum {h} — {gallery.TILES_PER_SHEET} uniform tiles  (published, "
                  f"gates nothing)",
            subtitle=shared, path=args.out_dir / sheet, lossless=True, sigma=args.sigma,
        )
        drawn[spanning] = sheets.render_sheet(
            members[gallery.spanning_draw(ids, radii[members])].tolist(), m, pct, read,
            title=f"Stratum {h} — interior → edge  (published, gates nothing)",
            subtitle=shared, path=args.out_dir / spanning, lossless=False, sigma=args.sigma,
        )
        per_stratum.append(
            {"stratum_id": int(h), "n_h": int(len(members)), "sheet": sheet,
             "spanning": spanning}
        )

    coastal = np.flatnonzero(m["coastline"].astype(bool))
    drawn["coastal-floor.webp"] = sheets.render_sheet(
        coastal[gallery.uniform_draw(m["tile_id"][coastal])].tolist(), m, pct, read,
        title=f"Coastal floor — {gallery.TILES_PER_SHEET} uniform of {len(coastal):,}",
        subtitle=("the floor:coastline reservation (#42) as a reporting axis (plan §133)\n"
                  "#40 measured coastline as NOT a stratum (max purity 4.9%) — this sheet "
                  "gates nothing"),
        path=args.out_dir / "coastal-floor.webp", lossless=False, sigma=args.sigma,
    )

    manifest_out = args.out_dir / "gallery.json"
    manifest_out.write_text(json.dumps({
        "artifact": "F4 stratum gallery — the 500k scale corpus",
        "ticket": "https://github.com/JStoschek/DEMAE/issues/73",
        "map": "https://github.com/JStoschek/DEMAE/issues/67",
        "bar": "docs/research/f-gate-bar.json -> F4_gallery_published",
        "gates": False,
        "gating_status": "published, non-gating (#69 retired E2.4's pass/fail)",
        "labels_recorded_at": "https://github.com/JStoschek/DEMAE/issues/74",
        "rendering_protocol": "docs/research/e2-gallery-protocol.md (rendering only — its "
                              "pass/fail clause does not apply at Gate F)",
        "seed": gallery.GALLERY_SEED,
        "rendering": {
            "shading": "grey hillshade, surface normals · matplotlib LightSource formulation",
            "azimuth_deg": gallery.AZIMUTH_DEG, "altitude_deg": gallery.ALTITUDE_DEG,
            "smooth_sigma_px": args.sigma,
            "vert_exag": f"per tile, towards a {gallery.TARGET_LAND_STD_M:.0f} m land-std, "
                         f"clamped [{gallery.VE_MIN}, {gallery.VE_MAX}]",
            "pixel_spacing": "true anisotropic (dy, dx) from the latitude band",
            "colour": "none on terrain; edge bar = mean_elevation on a shared ramp",
            "reused_from": "notebooks/e2_stratum_gallery.py — identical layout, so the two "
                           "tiers' sheets are comparable",
        },
        "gating_sheets": [],
        "non_gating_sheets": (
            [d["sheet"] for d in per_stratum]
            + [d["spanning"] for d in per_stratum]
            + ["coastal-floor.webp"]
        ),
        "lossless_sheets": [d["sheet"] for d in per_stratum],
        "inputs": {
            "corpus": str(args.corpus.relative_to(REPO)),
            "dataset_version": meta["dataset_version"],
            "centroids": str(args.centroids.relative_to(REPO)),
            "scaler": str(args.scaler.relative_to(REPO)),
        },
        "code_version": provenance.git_sha(), "rendered": provenance.today(),
        "strata": per_stratum,
        "tiles": drawn,
    }, indent=2) + "\n")
    print(f"[wrote {manifest_out.relative_to(REPO)}]")

    write_cover(args.cover, per_stratum=per_stratum, n_coastal=len(coastal), meta={
        "dataset_version": meta["dataset_version"], "n_tiles": meta["n_tiles"],
        "today": provenance.today(), "code_version": provenance.git_sha(),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
