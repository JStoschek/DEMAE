"""``python -m dataset`` -> the corpus build CLI (:func:`dataset.build.main`)."""

from __future__ import annotations

from .build import main

if __name__ == "__main__":
    raise SystemExit(main())
