"""Build-provenance stamps — the git SHA and snapshot date every artifact records.

Small, dependency-free helpers that answer "which code, on which day, produced this?".
They live here rather than in any one writer because every module that ships an artifact
needs the same two facts stamped identically: the catalog's ``code_version`` fingerprint,
the exemplar/pilot/prevalence/calibration report headers, and the corpus meta. Kept out
of a heavier module so importing them never drags in a reader, a pipeline, or numpy.
"""

from __future__ import annotations

import subprocess


def git_sha() -> str:
    """Short git SHA of the working tree (``-dirty`` if uncommitted), or ``"unknown"``.

    Never raises: a checkout without git, or outside a repo, degrades to ``"unknown"``
    rather than failing an artifact write. The ``-dirty`` suffix is the honest signal
    that an artifact was produced from uncommitted code.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True
    ).stdout.strip()
    return f"{sha}-dirty" if dirty else sha


def today() -> str:
    """Today's date as an ISO ``YYYY-MM-DD`` string — the artifact's snapshot date."""
    from datetime import date

    return date.today().isoformat()
