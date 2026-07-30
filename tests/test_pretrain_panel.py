"""The committed fixed val panel — §10.2's 40 tiles (#98, plan §15's checklist item 3).

The panel is an *artifact*, in the same contract family as the selection list and the split
assignment: what later runs re-read is the committed file, never this code re-run. So the tests
split the same way those do.

**The draw is a function of the manifest and a committed seed.** Two draws over one manifest
agree, tile for tile; a draw is 2 tiles per stratum and every stratum is present; nothing
outside the val split can be drawn, because a train tile in the panel would put training data
in the one place a human looks to judge generalization.

**The reader refuses a panel that no longer describes this corpus.** A tile the manifest does
not hold, a tile whose split has moved, a stratum count that is not what was committed — each
is a way for a grid to keep rendering while quietly no longer being the panel it says it is,
and none of them shows up in a picture.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pretrain import panel as panel_mod
from pretrain.panel import PanelTile, draw_panel, read_panel, resolve_panel, write_panel


def frame(*, n_strata: int = 20, per_stratum: int = 5, val_from: int = 0) -> dict:
    """A manifest-shaped set of columns: ``per_stratum`` tiles in each of ``n_strata``.

    Every stratum gets the same number of tiles and the first ``val_from`` of each are labelled
    ``train``, which is how a stratum too thin to fill its quota is staged.
    """
    tile_id, stratum, split, lat = [], [], [], []
    for s in range(n_strata):
        for j in range(per_stratum):
            tile_id.append(f"COG_{s:02d}_{j:03d}_0")
            stratum.append(s)
            split.append("train" if j < val_from else "val")
            lat.append(float(s) - 30.0)
    return {
        "tile_id": np.array(tile_id),
        "stratum_id": np.array(stratum, dtype=np.int32),
        "split": np.array(split),
        "lat": np.array(lat),
    }


# --- The draw -----------------------------------------------------------------


def test_the_draw_takes_two_val_tiles_from_every_stratum():
    cols = frame()
    drawn = draw_panel(cols["tile_id"], cols["stratum_id"], cols["split"])

    assert len(drawn) == 40
    strata = [t.stratum_id for t in drawn]
    assert sorted(set(strata)) == list(range(20))
    assert all(strata.count(s) == 2 for s in range(20))


def test_the_same_manifest_and_seed_draw_the_same_forty_tiles():
    cols = frame()
    first = draw_panel(cols["tile_id"], cols["stratum_id"], cols["split"])
    second = draw_panel(cols["tile_id"], cols["stratum_id"], cols["split"])
    assert [t.tile_id for t in first] == [t.tile_id for t in second]


def test_the_draw_does_not_depend_on_the_order_the_manifest_arrives_in():
    """A keyed hash per tile id, not a shuffle of the rows: row order is not the contract."""
    cols = frame()
    order = np.arange(len(cols["tile_id"]))[::-1]
    straight = draw_panel(cols["tile_id"], cols["stratum_id"], cols["split"])
    reversed_ = draw_panel(
        cols["tile_id"][order], cols["stratum_id"][order], cols["split"][order]
    )
    assert [t.tile_id for t in straight] == [t.tile_id for t in reversed_]


def test_a_different_seed_draws_a_different_panel():
    cols = frame()
    other = draw_panel(cols["tile_id"], cols["stratum_id"], cols["split"], seed=1)
    default = draw_panel(cols["tile_id"], cols["stratum_id"], cols["split"])
    assert [t.tile_id for t in other] != [t.tile_id for t in default]


def test_nothing_outside_the_val_split_can_be_drawn():
    cols = frame(per_stratum=6, val_from=4)  # 4 train + 2 val per stratum
    drawn = {t.tile_id for t in draw_panel(cols["tile_id"], cols["stratum_id"], cols["split"])}
    val = set(cols["tile_id"][cols["split"] == "val"].tolist())
    assert drawn <= val


def test_a_stratum_too_thin_to_fill_its_quota_stops_the_draw():
    cols = frame(per_stratum=5, val_from=4)  # one val tile per stratum, two wanted
    with pytest.raises(ValueError, match="stratum"):
        draw_panel(cols["tile_id"], cols["stratum_id"], cols["split"])


# --- The artifact -------------------------------------------------------------


def test_the_artifact_round_trips_and_is_byte_stable(tmp_path):
    cols = frame()
    drawn = draw_panel(cols["tile_id"], cols["stratum_id"], cols["split"])
    first = write_panel(drawn, tmp_path / "panel.csv").read_bytes()
    again = write_panel(list(reversed(drawn)), tmp_path / "panel.csv").read_bytes()

    assert first == again  # the file is sorted, so the draw order cannot leak into the md5
    assert read_panel(tmp_path / "panel.csv") == list(drawn)


def test_a_file_that_is_not_a_panel_is_refused(tmp_path):
    path = tmp_path / "not-a-panel.csv"
    path.write_text("tile_id,split\nA,val\n")
    with pytest.raises(ValueError, match="not a val panel"):
        read_panel(path)


# --- Resolving it against a corpus --------------------------------------------


def test_resolving_finds_each_tiles_row_and_its_latitude():
    cols = frame()
    drawn = draw_panel(cols["tile_id"], cols["stratum_id"], cols["split"])
    resolved = resolve_panel(drawn, **cols)

    assert [t.tile_id for t in resolved.tiles] == [t.tile_id for t in drawn]
    assert list(resolved.rows) == [
        int(np.flatnonzero(cols["tile_id"] == t.tile_id)[0]) for t in drawn
    ]
    # The hillshade needs true ground spacing, which is a function of latitude alone.
    assert list(resolved.lat) == [float(cols["lat"][r]) for r in resolved.rows]


def test_the_panel_is_ordered_by_stratum_so_two_epochs_grids_line_up():
    cols = frame()
    resolved = resolve_panel(
        draw_panel(cols["tile_id"], cols["stratum_id"], cols["split"]), **cols
    )
    keys = [(t.stratum_id, t.tile_id) for t in resolved.tiles]
    assert keys == sorted(keys)


def test_a_panel_tile_this_corpus_does_not_hold_is_refused():
    cols = frame()
    stale = [PanelTile("COG_99_999_0", 0), *draw_panel(
        cols["tile_id"], cols["stratum_id"], cols["split"]
    )[1:]]
    with pytest.raises(ValueError, match="does not hold"):
        resolve_panel(stale, **cols)


def test_a_panel_tile_that_has_since_moved_out_of_val_is_refused():
    cols = frame()
    drawn = draw_panel(cols["tile_id"], cols["stratum_id"], cols["split"])
    moved = cols | {"split": cols["split"].copy()}
    moved["split"][np.flatnonzero(cols["tile_id"] == drawn[0].tile_id)[0]] = "train"
    with pytest.raises(ValueError, match="val"):
        resolve_panel(drawn, **moved)


def test_a_committed_stratum_that_disagrees_with_the_manifest_is_refused():
    """The label is what makes a grid readable as *this stratum reconstructs badly*."""
    cols = frame()
    drawn = draw_panel(cols["tile_id"], cols["stratum_id"], cols["split"])
    relabelled = [PanelTile(drawn[0].tile_id, 19), *drawn[1:]]
    with pytest.raises(ValueError, match="stratum"):
        resolve_panel(relabelled, **cols)


def test_a_panel_that_is_not_two_per_stratum_is_refused():
    cols = frame()
    drawn = draw_panel(cols["tile_id"], cols["stratum_id"], cols["split"])
    with pytest.raises(ValueError, match="2 tiles"):
        resolve_panel(drawn[:-1], **cols)


def test_a_panel_naming_a_stratum_the_corpus_does_not_hold_is_caught_by_the_label_check():
    """There is no separate "unknown stratum" refusal, and this is why there does not need to be.

    A stratum the val split does not hold can only reach the count check on a tile that
    exists, is val, and carries that stratum in the manifest — which is a contradiction. Any
    real attempt trips the stratum-label check first, with a message naming the disagreement.
    """
    cols = frame(n_strata=4)
    drawn = draw_panel(cols["tile_id"], cols["stratum_id"], cols["split"])
    invented = [*drawn, PanelTile("COG_00_000_0", 9), PanelTile("COG_00_001_0", 9)]
    with pytest.raises(ValueError, match=r"panel 9, manifest 0"):
        resolve_panel(invented, **cols)


def test_a_panel_missing_a_stratum_entirely_is_refused():
    cols = frame()
    drawn = [t for t in draw_panel(cols["tile_id"], cols["stratum_id"], cols["split"])
             if t.stratum_id != 7]
    with pytest.raises(ValueError, match="stratum"):
        resolve_panel(drawn, **cols)


# --- The shipped artifact ------------------------------------------------------

SHIPPED = Path(panel_mod.VAL_PANEL_PATH)
MANIFEST = Path("data/corpus_500k/manifest.parquet")


@pytest.mark.skipif(not SHIPPED.exists(), reason="the panel has not been committed yet")
def test_the_committed_panel_is_forty_tiles_two_from_each_of_the_twenty_strata():
    tiles = read_panel(SHIPPED)
    strata = [t.stratum_id for t in tiles]
    assert len(tiles) == 40
    assert sorted(set(strata)) == list(range(20))
    assert all(strata.count(s) == 2 for s in range(20))
    assert len(set(t.tile_id for t in tiles)) == 40


@pytest.mark.skipif(
    not (SHIPPED.exists() and MANIFEST.exists()),
    reason="needs the committed panel and the shipped manifest",
)
def test_the_committed_panel_still_describes_the_shipped_corpus():
    """The whole reason `resolve_panel` refuses things: this is the check, on the real files."""
    resolved = panel_mod.resolve_from_manifest(read_panel(SHIPPED), MANIFEST)
    assert len(resolved.tiles) == 40
    assert len(set(resolved.rows.tolist())) == 40
