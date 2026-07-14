"""Stage 1 — corpus discovery.

Enumerate the released Copernicus GLO-30 COGs by listing the public bucket's
top-level prefixes (one per existing 1-degree cell). Ocean and withheld-country
cells are simply *absent* prefixes (recon §2-3), so "only sample existing
prefixes" is the entire — and free — first-pass land/withheld filter. Each prefix
is parsed into a :class:`~dataset.geometry.CogRef`; anything that is not a
Copernicus DEM prefix (e.g. ``readme.html``) is skipped.
"""

from __future__ import annotations

from . import config
from .geometry import CogRef, parse_cog_key


def list_cog_refs(bucket: str = config.COPERNICUS_GLO30_BUCKET) -> list[CogRef]:
    """List every released COG in ``bucket`` as a :class:`CogRef` (anonymous S3)."""
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    paginator = s3.get_paginator("list_objects_v2")
    refs: list[CogRef] = []
    for page in paginator.paginate(Bucket=bucket, Delimiter="/"):
        for common in page.get("CommonPrefixes", []):
            try:
                refs.append(parse_cog_key(common["Prefix"]))
            except ValueError:
                continue  # not a DEM prefix (INFO/, readme, ...)
    return refs
