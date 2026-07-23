"""E2.4 (#53) — the stratum gallery: 12 tiles x 20 strata as hillshade contact sheets.

The fifth E2 analysis, and the only one a number cannot stand in for. #52 answered
"does the corpus implement its committed inputs faithfully" on the numbers; this asks
the question no metric can, the one plan §186 reserved for a human: **are the k=20
strata terrain-meaningful** — does each one read as a family of related landforms, or
as an arbitrary cut through a cloud?

This script renders the evidence. It does **not** review it and it does **not** render
a verdict: the marks, the terrain-family labels and the pass/fail roll-up all belong to
the Gate E review ([#54](https://github.com/JStoschek/DEMAE/issues/54)), against the
procedure pre-registered in [`e2-gallery-protocol.md`](../docs/research/e2-gallery-protocol.md)
before any of these pixels existed.

## What it writes

``docs/research/e2-gallery/``

* ``stratum-NN.png`` x20 — **the gating sheets**, lossless PNG. 12 tiles drawn keyed-at-
  random from the stratum, which is the "fixed random sample with a committed seed" the
  protocol registered. These are what a Gate E mark is written against.
* ``stratum-NN-spanning.webp`` x20 — **published, gating nothing**. The same stratum
  walked interior → edge. #41 found the descriptor space is a continuum, so boundary
  tiles are where "is this a real family" is really tested; but over-sampling the
  boundary makes the coherence gate stricter than the one #47 registered, so this
  informs the reviewer without moving the bar. Same standing #47 gave the pool-ECDF and
  the coverage map.
* ``coastal-floor.webp`` — the 669 ``floor:coastline`` tiles as a reporting axis (plan
  §133). #40 measured them as *not* a stratum (max purity 4.9%), the bar carries no
  coastal-coherence criterion, and this sheet must not become one.
* ``gallery.json`` — the seed and every rendered tile id, per protocol rule 2.
* ``e2-gallery.md`` — the cover sheet, with the empty 20-row mark/label table #54 fills.

Gating sheets are lossless because the shards are gitignored: once they are gone, these
PNGs plus ``gallery.json`` are the *only* surviving record of what a verdict was written
against — the same argument #50 made for committing the materialization ledger. The
non-gating sheets carry less weight and ship as WebP, which is where the ~22 MB a
fully-lossless gallery would have cost the repo actually goes.

## Re-runnable per tier

Phase F re-renders this at 200-500k: every path is an argument, the strata come from the
manifest rather than a constant, and a stratum with fewer than 12 tiles simply makes a
shorter sheet. The seed is the one thing deliberately *not* an argument — see
:data:`dataset.gallery.GALLERY_SEED`.

Zero-network; reads only the committed manifest, centroids and scaler plus the local
shards. numpy + pyarrow + matplotlib + Pillow (this repo's env has no pandas).

Run:   uv run python notebooks/e2_stratum_gallery.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from PIL import Image  # noqa: E402

from dataset import gallery, provenance  # noqa: E402
from dataset.centroids import Centroids  # noqa: E402
from dataset.frame import METRICS, Scaler  # noqa: E402
from dataset.shards import shard_path  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RESEARCH = REPO / "docs" / "research"

#: Manifest columns the sheets read. Everything here is denormalized into the manifest by
#: schema v2 (#48), which is what lets the gallery run without re-opening the catalog.
NEEDED = (
    "tile_id", "shard_id", "row_index", "lat", "lon", "lat_band", "stratum_id",
    "provenance", "land_frac", "wbm_water_frac", "edm_edited_frac", "flm_fill_frac",
    "water_affected", "coastline", *METRICS,
)

#: Shared ramp for the per-tile elevation edge bar. Spans the corpus's actual range
#: (-19 m to 6,275 m) rather than a round number, so the bar uses its full extent.
ELEV_NORM = Normalize(-500.0, 6500.0)
ELEV_CMAP = "YlOrBr"

#: WebP quality for the non-gating sheets — visually indistinguishable here, ~5x smaller.
WEBP_QUALITY = 88

#: Chosen with :data:`TILE_PX` so each tile lands at exactly 224 device pixels — **1:1,
#: never upsampled**. Upsampling a hillshade invents pixels the DEM never had, which on
#: a gallery whose whole subject is fine terrain texture is the wrong artifact to add,
#: and it costs a third more bytes in a set that lives in the repo forever.
SHEET_DPI = 82
TILE_PX = 224

#: Type sizes, in points. They are read by the layout arithmetic in :func:`render_sheet`
#: rather than being sprinkled through it, because at 1:1 the sheet has only just enough
#: room for its text and every one of these changes how much room that is.
TITLE_PT = 12.0
SUBTITLE_PT = 7.2
CAPTION_PT = 7.0

#: Left margin, inches — room for the elevation edge bar beside the first column.
LEFT_IN = 0.16


# --- Reading -------------------------------------------------------------------


def load_manifest(path: Path) -> dict[str, np.ndarray]:
    table = pq.read_table(path, columns=list(NEEDED))
    return {c: table.column(c).to_numpy(zero_copy_only=False) for c in NEEDED}


class ShardReader:
    """Random access into the corpus by ``(shard_id, row_index)`` — memmapped, cached.

    The gallery touches ~500 of 50,000 tiles scattered across every shard, so the whole
    point is *not* to stream: one memmap per shard, held open, and the OS pages in the
    ~100 KB each tile actually needs. That is the O(1)-random-access property ADR-0002 §1
    designed the shard layout for, used here exactly as the loader uses it.
    """

    def __init__(self, shard_dir: Path) -> None:
        self.shard_dir = shard_dir
        self._mm: dict[int, np.memmap] = {}

    def __call__(self, shard_id: int, row_index: int) -> np.ndarray:
        mm = self._mm.get(int(shard_id))
        if mm is None:
            mm = self._mm[int(shard_id)] = np.load(
                shard_path(self.shard_dir, int(shard_id)), mmap_mode="r"
            )
        return np.array(mm[int(row_index)])


# --- One sheet -----------------------------------------------------------------


def caption(m: dict[str, np.ndarray], i: int, ve: float, dy: float, dx: float, pct: float) -> str:
    """Three monospace lines under a tile: the six axes, the geometry, then the flags.

    The frozen six lead because they are the axes the stratum was *cut* on — a reviewer
    asked to justify a `mixed` mark has nothing to point at without them. The geometry
    line exists so a tile that looks strange can be recognised as a polar one rather than
    an incoherent one, and it states the exaggeration so the transform is never silent.
    Flags appear only when they fire, so an unremarkable tile stays visually quiet.
    """
    flags = []
    if m["coastline"][i]:
        flags.append("COAST")
    if m["water_affected"][i]:
        flags.append(f"wtr{m['wbm_water_frac'][i] * 100:.0f}%")
    if m["edm_edited_frac"][i] > 0.05:
        flags.append(f"edit{m['edm_edited_frac'][i] * 100:.0f}%")
    if m["flm_fill_frac"][i] > 0.05:
        flags.append(f"fill{m['flm_fill_frac'][i] * 100:.0f}%")
    if m["land_frac"][i] < 0.999:
        flags.append(f"land{m['land_frac'][i] * 100:.0f}%")
    return (
        f"relief {m['local_relief'][i]:>6.0f} m   elev {m['mean_elevation'][i]:>6.0f} m   "
        f"slope {m['slope_mean'][i]:>5.1f}°\n"
        f"tex {m['texture'][i]:.2f}  err {m['elevation_relief_ratio'][i]:.2f}  "
        f"H {m['geomorphon_entropy'][i]:.2f}   r→ctr p{pct:02.0f}\n"
        f"{m['lat'][i]:+06.2f},{m['lon'][i]:+07.2f}  {dy:.0f}×{dx:.0f} m  ve×{ve:.2g}"
        + (f"   {' '.join(flags)}" if flags else "")
    )


def render_sheet(
    rows: list[int], m: dict[str, np.ndarray], pct: np.ndarray, read: ShardReader,
    *, title: str, subtitle: str, path: Path, lossless: bool, sigma: float,
) -> list[str]:
    """Draw one contact sheet; return the tile ids in the grid order they were drawn.

    The figure geometry is *derived* from the tile size and the text it has to carry,
    not tuned by eye: a sheet is inches of tile plus inches of caption, and every margin
    below is that arithmetic. Hand-picked fractions looked fine until the dpi changed to
    put tiles at 1:1, and then the title landed on the subtitle and the bottom row's
    captions ran off the page — because points are absolute while those fractions are
    not. Deriving it means ``SHEET_DPI``, the font sizes and a short final row can all
    move without silently breaking the layout.
    """
    ncol = 4
    nrow = max(1, -(-len(rows) // ncol))

    tile_in = TILE_PX / SHEET_DPI  # a tile at exactly 1:1
    cap_in = 3 * CAPTION_PT * 1.5 / 72 + 0.08  # three caption lines + labelpad
    head_in = TITLE_PT / 72 + 2 * SUBTITLE_PT * 1.5 / 72 + 0.22  # title + 2-line subtitle
    wspace, hspace = 0.05, cap_in / tile_in

    fig_w = LEFT_IN + ncol * tile_in + (ncol - 1) * wspace * tile_in + 0.08
    fig_h = head_in + nrow * tile_in + (nrow - 1) * hspace * tile_in + cap_in + 0.06
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(fig_w, fig_h),
        gridspec_kw={"hspace": hspace, "wspace": wspace}, squeeze=False,
    )
    for ax in axes.ravel():
        ax.set_axis_off()

    drawn: list[str] = []
    for ax, i in zip(axes.ravel(), rows):
        rgb, ve, (dy, dx) = gallery.shade_tile(
            read(m["shard_id"][i], m["row_index"][i]), lat=float(m["lat"][i]), sigma=sigma
        )
        ax.set_axis_on()
        ax.imshow(rgb, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#999999")
            spine.set_linewidth(0.6)
        # Absolute elevation as an edge bar: mean_elevation is one of the six clustering
        # axes and grey shading cannot show it, but tinting the terrain to carry it makes
        # a 10 m-relief tile look like an alpine one. So it rides beside the tile instead.
        ax.add_patch(
            plt.Rectangle(
                (-7.5, 0), 5, rgb.shape[0] - 1, clip_on=False, linewidth=0,
                color=plt.get_cmap(ELEV_CMAP)(ELEV_NORM(m["mean_elevation"][i])),
            )
        )
        ax.set_xlabel(
            caption(m, i, ve, dy, dx, float(pct[i])),
            fontsize=CAPTION_PT, family="monospace", labelpad=4, linespacing=1.5,
            color="#222222",
        )
        drawn.append(str(m["tile_id"][i]))

    fig.suptitle(title, fontsize=TITLE_PT, y=1 - 0.16 / fig_h)
    fig.text(0.5, 1 - (0.20 + TITLE_PT / 72) / fig_h, subtitle, ha="center",
             fontsize=SUBTITLE_PT, color="#555555", family="monospace", va="top")
    fig.subplots_adjust(
        left=LEFT_IN / fig_w, right=1 - 0.08 / fig_w,
        top=1 - head_in / fig_h, bottom=(cap_in + 0.06) / fig_h,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    if lossless:
        fig.savefig(path, dpi=SHEET_DPI)
    else:
        tmp = path.with_suffix(".tmp.png")
        fig.savefig(tmp, dpi=SHEET_DPI)
        Image.open(tmp).convert("RGB").save(path, "WEBP", quality=WEBP_QUALITY, method=6)
        tmp.unlink()
    plt.close(fig)
    print(f"  [{path.stat().st_size // 1024:>4} KB] {path.relative_to(REPO)}")
    return drawn


# --- The cover sheet -----------------------------------------------------------

COVER = """# E2.4 stratum gallery — {n_strata} strata x {per_sheet} tiles

**Ticket:** [Render the stratum gallery (#53)](https://github.com/JStoschek/DEMAE/issues/53) ·
**Map:** [Phase E (#46)](https://github.com/JStoschek/DEMAE/issues/46) ·
**Reviewed at:** [Gate E (#54)](https://github.com/JStoschek/DEMAE/issues/54) ·
**Protocol:** [`e2-gallery-protocol.md`](e2-gallery-protocol.md)

The human backstop that k={n_strata} strata are terrain-meaningful. Read the protocol
first — it fixes what is being judged (**internal coherence**, one stratum at a time),
what is explicitly **not** (distinguishability from a neighbour: #41 found the descriptor
space is a continuum, so adjacent strata blending is expected), and the pre-registered
fail condition (**any stratum `incoherent`, or ≥ 3 `mixed`**).

Corpus `{dataset_version}` · {n_tiles:,} tiles · seed `{seed}` · rendered {today} at
`{code_version}`. The exact tile ids behind every sheet are in
[`gallery.json`](gallery.json), so this gallery is reconstructible after the shards —
which are gitignored — are gone.

## How the tiles are drawn

Grey hillshade, north-west sun at {az:.0f}°/{alt:.0f}°, σ={sigma} px Gaussian before
shading, per-tile vertical exaggeration towards a {target:.0f} m land-std, true
anisotropic pixel spacing. **No colour carries terrain**: a shared elevation ramp flattens
every low-relief stratum into one block, and a per-tile ramp gives a 10 m-relief tile the
full alpine treatment — so magnitude is *printed*, and the coloured bar left of each tile
carries absolute elevation without touching the terrain pixels. Exaggeration is per tile,
so **apparent contrast means nothing across tiles**; the captioned `relief` and `elev` are
the magnitudes. σ=1 px matters: hillshading is a derivative, and at the exaggeration a flat
tile needs, the stored int16's 1 m quantum becomes stipple that reads as texture.

## The sheets

| stratum | n_h | gating sheet (12 uniform) | diagnostic (interior→edge) |
|---|---:|---|---|
{rows}

**[`coastal-floor.webp`](e2-gallery/coastal-floor.webp)** — the {n_coastal} `floor:coastline`
tiles as a reporting axis (plan §133). #40 measured coastline as *not* a stratum (max
purity 4.9%); there is no coastal-coherence criterion in the bar and this sheet does not
create one.

The `-spanning` sheets are **published, not gating**. They walk each stratum from its
centroid outward, which is where a continuum's cluster boundaries are really tested — but
over-representing the boundary would make the gate stricter than the one pre-registered,
so they inform the reviewer without moving the bar.

## Marks — for #54 to complete

Per the protocol: one mark per stratum (`coherent` / `mixed` / `incoherent`) plus a short
terrain-family label. Record all {n_strata}, **including on a pass** — they are the evidence
behind the verdict and the seed for the E3 notes (#55) and ADR-0005 (#56). Judge each
stratum's tiles on their own; do not compare strata.

| stratum | mark | terrain-family label | note |
|---|---|---|---|
{marks}

*Left blank deliberately — the marks are #54's, and #53 does not pre-empt them.*
"""


def write_cover(path: Path, *, per_stratum: list[dict], n_coastal: int, meta: dict) -> None:
    rows = "\n".join(
        f"| {d['stratum_id']} | {d['n_h']:,} | [`{d['sheet']}`](e2-gallery/{d['sheet']}) | "
        f"[`{d['spanning']}`](e2-gallery/{d['spanning']}) |"
        for d in per_stratum
    )
    marks = "\n".join(f"| {d['stratum_id']} |  |  |  |" for d in per_stratum)
    path.write_text(
        COVER.format(
            n_strata=len(per_stratum), per_sheet=gallery.TILES_PER_SHEET, rows=rows, marks=marks,
            n_coastal=n_coastal, az=gallery.AZIMUTH_DEG, alt=gallery.ALTITUDE_DEG,
            sigma=gallery.SMOOTH_SIGMA_PX, target=gallery.TARGET_LAND_STD_M,
            seed=gallery.GALLERY_SEED, **meta,
        )
    )
    print(f"[wrote {path.relative_to(REPO)}]")


# --- Driver --------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--corpus", type=Path, default=REPO / "data" / "corpus_50k")
    p.add_argument("--centroids", type=Path,
                   default=REPO / "artifacts" / "phase_d" / "centroids.json")
    p.add_argument("--scaler", type=Path, default=REPO / "artifacts" / "phase_d" / "scaler.json")
    p.add_argument("--out-dir", type=Path, default=RESEARCH / "e2-gallery")
    p.add_argument("--cover", type=Path, default=RESEARCH / "e2-gallery.md")
    p.add_argument("--sigma", type=float, default=gallery.SMOOTH_SIGMA_PX,
                   help="Gaussian px before shading; 0 renders the stored field raw.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    m = load_manifest(args.corpus / "manifest.parquet")
    meta = json.loads((args.corpus / "corpus_meta.json").read_text())
    scaler = Scaler.from_dict(json.loads(args.scaler.read_text()))
    centroids = Centroids.from_dict(json.loads(args.centroids.read_text()))
    read = ShardReader(args.corpus / "shards")

    strata = m["stratum_id"]
    radii = gallery.centroid_radius(m, strata, scaler=scaler, centroids=centroids)
    pct = gallery.radius_percentile(radii, strata)

    print(f"gallery: {len(m['tile_id']):,} tiles, k={len(np.unique(strata))}, "
          f"seed={gallery.GALLERY_SEED}, sigma={args.sigma} px -> {args.out_dir.relative_to(REPO)}")

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

        drawn[sheet] = render_sheet(
            members[gallery.uniform_draw(ids)].tolist(), m, pct, read,
            title=f"Stratum {h} — {gallery.TILES_PER_SHEET} uniform tiles  (gating)",
            subtitle=shared, path=args.out_dir / sheet, lossless=True, sigma=args.sigma,
        )
        drawn[spanning] = render_sheet(
            members[gallery.spanning_draw(ids, radii[members])].tolist(), m, pct, read,
            title=f"Stratum {h} — interior → edge  (published, gates nothing)",
            subtitle=shared, path=args.out_dir / spanning, lossless=False, sigma=args.sigma,
        )
        per_stratum.append(
            {"stratum_id": int(h), "n_h": int(len(members)), "sheet": sheet, "spanning": spanning}
        )

    coastal = np.flatnonzero(m["coastline"].astype(bool))
    drawn["coastal-floor.webp"] = render_sheet(
        coastal[gallery.uniform_draw(m["tile_id"][coastal])].tolist(), m, pct, read,
        title=f"Coastal floor — {gallery.TILES_PER_SHEET} uniform of {len(coastal):,}",
        subtitle=("the floor:coastline reservation (#42) as a reporting axis (plan §133)\n"
                  "#40 measured coastline as NOT a stratum (max purity 4.9%) — this sheet "
                  "gates nothing"),
        path=args.out_dir / "coastal-floor.webp", lossless=False, sigma=args.sigma,
    )

    manifest_out = args.out_dir / "gallery.json"
    manifest_out.write_text(json.dumps({
        "artifact": "E2.4 stratum gallery",
        "ticket": "https://github.com/JStoschek/DEMAE/issues/53",
        "reviewed_at": "https://github.com/JStoschek/DEMAE/issues/54",
        "protocol": "docs/research/e2-gallery-protocol.md",
        "seed": gallery.GALLERY_SEED,
        "rendering": {
            "shading": "grey hillshade, surface normals · matplotlib LightSource formulation",
            "azimuth_deg": gallery.AZIMUTH_DEG, "altitude_deg": gallery.ALTITUDE_DEG,
            "smooth_sigma_px": args.sigma,
            "vert_exag": f"per tile, towards a {gallery.TARGET_LAND_STD_M:.0f} m land-std, "
                         f"clamped [{gallery.VE_MIN}, {gallery.VE_MAX}]",
            "pixel_spacing": "true anisotropic (dy, dx) from the latitude band",
            "colour": "none on terrain; edge bar = mean_elevation on a shared ramp",
        },
        "gating_sheets": [d["sheet"] for d in per_stratum],
        "non_gating_sheets": [d["spanning"] for d in per_stratum] + ["coastal-floor.webp"],
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
