"""The ``tile_id,stratum_id`` committed artifact — one format, one implementation.

Phase 1 commits two of these, and they are the same file in two roles: §10.2's **val panel**
(#98, 40 tiles, 2 per stratum) and §10.3's **kNN probe set** (#99, ~5k tiles, stratified). Both
sit in the contract family the selection list and the split assignment are in — *the committed
file is what later runs re-read, not this code re-run* (ADR-0005 §5) — and both carry the same
two columns for the same reason: the tile to read, and the stratum it was committed under, so a
manifest re-fit underneath the artifact is **caught** rather than silently changing what a
comparison is between.

Two columns and a header is barely code, which is exactly why it is worth having once. The
duplication that matters is not the four lines of parsing; it is the *decisions* — that the
header is a fixed string a mistyped file is refused for, that the file is LF-terminated and
written from a caller-chosen order so its md5 is a contract rather than a timestamp, and that
``rsplit(",", 1)`` is safe because a Copernicus tile id never contains a comma. Two copies of
those drift silently, and the symptom is one artifact's md5 moving on a rewrite while the
other's does not.

**Order is the caller's**, deliberately, because the two artifacts order for different reasons:
the panel writes ``(stratum_id, tile_id)`` because that is the order its grids render in, and
the probe set writes ``tile_id`` because it has no render order and sorting by name is the
plainest thing to diff. Both are stable and both are byte-identical on a redraw, which is the
property the family actually needs.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Final

#: The header both artifacts carry. A file whose first line is not exactly this is refused:
#: the two columns are positional, so a file with them the other way round parses cleanly and
#: means something else entirely.
HEADER: Final = "tile_id,stratum_id"

#: One committed row: the tile, and the stratum it was committed under.
TileStratum = tuple[str, int]


def write_tile_strata(rows: Iterable[TileStratum], path: Path | str) -> Path:
    """Write the artifact in the order given, LF-terminated, creating the directory.

    The caller has already chosen the order (see the module docstring) — this does not sort,
    because a writer that re-sorted would quietly discard the panel's render order.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [HEADER] + [f"{tile_id},{stratum}" for tile_id, stratum in rows]
    path.write_text("\n".join(lines) + "\n")
    return path


def read_tile_strata(path: Path | str, *, what: str) -> list[TileStratum]:
    """Read the artifact back. ``what`` names it in the refusal — "val panel", "probe set"."""
    lines = Path(path).read_text().splitlines()
    if not lines or lines[0] != HEADER:
        raise ValueError(f"{path} is not a {what} (header {lines[:1]}, expected {HEADER!r})")
    rows = []
    for line in lines[1:]:
        if not line:
            continue
        tile_id, stratum = line.rsplit(",", 1)
        rows.append((tile_id, int(stratum)))
    return rows
