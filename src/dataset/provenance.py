"""Build-provenance stamps — the git SHA and snapshot date every artifact records.

Small, dependency-free helpers that answer "which code, on which day, produced this?".
They live here rather than in any one writer because every module that ships an artifact
needs the same two facts stamped identically: the catalog's ``code_version`` fingerprint,
the exemplar/pilot/prevalence/calibration report headers, and the corpus meta. Kept out
of a heavier module so importing them never drags in a reader, a pipeline, or numpy.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

#: Where :func:`git_sha` looks when git cannot answer — a one-line file beside the code,
#: written by whoever shipped that code to a machine without its history.
#:
#: Phase 1 rents a GPU box per session and rsyncs seven paths to it, ``.git`` deliberately
#: not among them: at 128 MB it is the largest thing that would move, and shipping it would
#: make the answer *wrong* rather than merely expensive — ``git status --porcelain`` on the
#: box sees every unshipped path (``docs/``, ``notebooks/``, ``CONTEXT.md``) as deleted, so
#: every run would stamp ``-dirty`` from a clean commit. A file carries the laptop's answer,
#: computed where the history actually is.
SHA_FILE = ".git_sha"


def git_sha() -> str:
    """Short git SHA of the working tree (``-dirty`` if uncommitted), or ``"unknown"``.

    Never raises: a checkout without git, or outside a repo, degrades to ``"unknown"``
    rather than failing an artifact write. The ``-dirty`` suffix is the honest signal
    that an artifact was produced from uncommitted code.

    Git is asked **first** and its answer is never overridden. Only when it cannot answer
    — no binary, or not a repo — does :data:`SHA_FILE` in the working directory get read.
    That ordering is the point: the fallback can only improve the degraded case, so a stale
    ``.git_sha`` left in a real checkout is inert rather than a lie. Both paths are relative
    to the process's working directory, which is what ``git rev-parse`` already keys on.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return _sha_from_file()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True
    ).stdout.strip()
    return f"{sha}-dirty" if dirty else sha


def _sha_from_file() -> str:
    """:data:`SHA_FILE`'s first line, or ``"unknown"`` — the no-git fallback.

    Deliberately incurious about the *shape* of what it reads: this is provenance, and a
    value someone chose to write is more informative than a rejection. It takes only the
    first line and strips it, so a trailing newline or an accidental second line cannot
    reach a W&B tag (:data:`pretrain.sink.MAX_TAG_CHARS`) or a checkpoint header.
    """
    try:
        first = Path(SHA_FILE).read_text().splitlines()
    except OSError:
        return "unknown"
    return first[0].strip() if first and first[0].strip() else "unknown"


def today() -> str:
    """Today's date as an ISO ``YYYY-MM-DD`` string — the artifact's snapshot date."""
    from datetime import date

    return date.today().isoformat()
