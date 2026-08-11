"""Fetch the 24 SRTMGL1N `.num` granules the real-void family is built from, and prove they are real.

Issue #127, under map #117; the checklist this implements is §13 of
`docs/research/p125-srtm-void-geometry.md`, and the draw it downloads is the one #126 settled
(Atlas cut, Andes kept for geographic diversity rather than volume).

**The one step this script cannot do is the reason the ticket exists.** SRTMGL1N sits behind
Earthdata Login: the distribution URL 302s to a URS OAuth `authorize` endpoint, and until a human
has registered and clicked the LP DAAC consent screen *once*, every granule URL answers
`401 HTTP Basic: Access denied` no matter what credentials are presented. Registering an account
and accepting terms of service is not something an agent does on someone's behalf. Everything
either side of it is here:

    # before the login — anonymous, proves the draw exists and prices it
    python scripts/p127_fetch_srtm_voids.py check

    # after the login — 24 granules, 28.4 MB, ~1 minute
    python scripts/p127_fetch_srtm_voids.py fetch

`check` runs two probes. The **CMR** probe enumerates the collection's granule metadata, which is
anonymous — granule names, byte sizes and download URLs all come back without credentials — and
confirms the 24 cells exist at the URL this script builds. The **auth** probe requests one granule
and reports which of the three states the login is in: no credentials in `~/.netrc`, credentials
that URS rejects, or a working chain. It downloads nothing.

`fetch` writes `data/srtm_voids/num/<CELL>.SRTMGL1N.num.zip` and a `fetch-manifest.json` beside
them recording, per granule, the zip's size and SHA-256 and the member's name and size. That
manifest is the artifact #127 asks to be recorded; the directory is under the repo's gitignored
`data/*`, so nothing here is committed.

**Verification is not a formality.** The failure this guards is not an error — an unauthorized
request 302s into a login form, and `curl -o file.zip` writes the HTML page to disk under a `.zip`
name at a plausible size. Every granule is therefore opened as a zip and required to hold exactly
one member, named `<CELL>.num`, of exactly 12,967,201 bytes (3601 x 3601 uint8, P125 §13.2). A
granule that fails is deleted, not kept, so a re-run cannot mistake it for done.

Scope: the fetch and its verification only. Extraction — the `num > 10 AND num < 201` predicate,
8-connected labelling, and the offset-RLE mask list of P125 §13.4 — belongs to the module that
builds `benchmark/`, which map #117 puts out of scope; #135 checks the realised size and stratum
histogram once this lands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import netrc
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from http.cookiejar import CookieJar
from pathlib import Path
from typing import Callable, Iterable

#: The URS host that issues the Basic challenge. Credentials are looked up under this name.
EARTHDATA_HOST = "urs.earthdata.nasa.gov"
#: LP DAAC's protected distribution endpoint. Confirmed against CMR's own `GET DATA` URLs.
DAAC_ROOT = "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1N.003"
#: SRTMGL1N.003 in CMR. The metadata search under it needs no credentials.
CMR_COLLECTION = "C2763266368-LPCLOUD"
CMR_SEARCH = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"

#: A 1-arc-second cell is 3601 x 3601 posts, and `.num` is one uint8 per post. This exact size is
#: the whole verification: anything else is a login page, a truncated transfer, or the 3-arc-second
#: v2.1 product (1201 x 1201) arriving by mistake.
NUM_BYTES = 3601 * 3601  # 12,967,201

#: The draw, as #126 settled it. Grouped by massif because that is how the regions were chosen and
#: how a partial fetch is read; order within a group is P125 §9's.
DRAW = {
    # Himalaya — Everest and the Khumbu, the void-richest cells P125 measured.
    "himalaya-central": ("N28E086", "N28E085", "N29E085", "N28E087", "N27E088", "N28E088"),
    # Kumaon / western Himalaya.
    "himalaya-west": ("N30E080", "N29E080", "N30E081"),
    # Karakoram — the steepest relief in the draw, and the deepest radar shadow.
    "karakoram": ("N35E075", "N35E076", "N36E074", "N36E075"),
    # Eastern syntaxis, around Namcha Barwa.
    "himalaya-east": ("N29E095", "N29E094", "N28E095", "N30E095"),
    # Andes. Kept by #126 for geographic diversity — a second radar look-direction and a second
    # climate — and explicitly *not* for volume: ~148 small components per cell against the
    # Himalaya's ~700. Cost of keeping them: ~8 MB of the 28.4.
    "andes": ("S10W077", "S09W077", "S13W072", "S16W068", "S17W068", "S33W070", "S32W070"),
}
DRAW_CELLS: tuple[str, ...] = tuple(cell for group in DRAW.values() for cell in group)

DEFAULT_DEST = Path("data/srtm_voids/num")
MANIFEST_NAME = "fetch-manifest.json"

#: Printed whenever credentials are missing or rejected. This message *is* the human checklist —
#: an agent hitting this path can go no further, and the person reading it needs the whole recipe.
CHECKLIST = f"""\
Earthdata Login is required for SRTMGL1N, and the consent step needs a person:

  1. Register (free, no fee, no eligibility gate):
         https://{EARTHDATA_HOST}/users/new
  2. Authorize the LP DAAC application ONCE, by opening any granule URL in a browser while
     logged in and clicking through the consent screen, e.g.
         {DAAC_ROOT}/N28E086.SRTMGL1N.num/N28E086.SRTMGL1N.num.zip
     Until this click happens, every granule answers 401 whatever the credentials.
  3. Store the credentials (the password never belongs in this repo):
         echo "machine {EARTHDATA_HOST} login <username> password <password>" >> ~/.netrc
         chmod 600 ~/.netrc

Then re-run this script.\
"""


class MissingCredentials(RuntimeError):
    """No `~/.netrc` entry for URS — the human step has not been done."""


class VerificationError(RuntimeError):
    """A downloaded file is not the granule it claims to be."""


class FetchError(RuntimeError):
    """A granule could not be downloaded."""


# ------------------------------------------------------------------------------------------------
# Addressing
# ------------------------------------------------------------------------------------------------


def granule_ur(cell: str) -> str:
    """CMR's `GranuleUR` for a cell, which is also the directory name under the DAAC root."""
    return f"{cell}.SRTMGL1N.num"


def granule_url(cell: str) -> str:
    ur = granule_ur(cell)
    return f"{DAAC_ROOT}/{ur}/{ur}.zip"


# ------------------------------------------------------------------------------------------------
# Verification
# ------------------------------------------------------------------------------------------------


def verify_zip(path: Path, cell: str) -> dict:
    """Return the granule's record, or raise ``VerificationError`` saying precisely what is wrong.

    Checks, in the order they fail in practice: is it a zip at all (an unauthorized fetch is an
    HTML login page); does it hold exactly one member; is that member named `<CELL>.num`; is it
    exactly ``NUM_BYTES``. The member's *uncompressed* size is read from the central directory, so
    the check costs no decompression — but the SHA-256 is over the zip on disk, which is the thing
    a later run can re-verify without unzipping.
    """
    if not zipfile.is_zipfile(path):
        head = path.read_bytes()[:80].decode("utf-8", "replace").strip()
        raise VerificationError(
            f"{path.name} is not a zip ({path.stat().st_size} bytes). "
            f"An unauthorized fetch lands as an HTML login page. Starts: {head!r}"
        )

    with zipfile.ZipFile(path) as zf:
        infos = zf.infolist()
        if len(infos) != 1:
            names = ", ".join(i.filename for i in infos)
            raise VerificationError(f"{path.name} holds {len(infos)} members, expected 1: {names}")
        info = infos[0]
        if info.filename != f"{cell}.num":
            raise VerificationError(
                f"{path.name} holds {info.filename!r}, expected {cell + '.num'!r} — wrong granule"
            )
        if info.file_size != NUM_BYTES:
            raise VerificationError(
                f"{path.name}: {info.filename} is {info.file_size:,} bytes, "
                f"expected {NUM_BYTES:,} (3601 x 3601 uint8)"
            )

    return {
        "member": info.filename,
        "member_bytes": info.file_size,
        "zip_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "verified": True,
    }


# ------------------------------------------------------------------------------------------------
# Credentials and the redirect chain
# ------------------------------------------------------------------------------------------------


class _HostOnlyPasswordMgr:
    """Answer a Basic challenge from exactly one host, whatever the realm, path or port.

    urllib's stock manager matches on a URI *prefix*, which ties the credential to a path and a
    port. URS challenges come from `oauth/authorize` with a per-granule `state` query string, so
    the hostname is the only stable key. Keying on it alone is also the tighter property: nothing
    is offered to the DAAC or the storage host the chain then redirects to.
    """

    def __init__(self, host: str, login: str, password: str):
        self._host = host
        self._login = login
        self._password = password

    def add_password(self, *args, **kwargs):  # part of the manager interface; nothing to add
        raise NotImplementedError("this manager holds one host, fixed at construction")

    def find_user_password(self, realm, authuri):
        if urllib.parse.urlsplit(authuri).hostname == self._host:
            return self._login, self._password
        return None, None


def build_opener(netrc_path: Path | None = None, host: str = EARTHDATA_HOST):
    """An opener that can follow the DAAC's redirect chain: Basic auth from netrc, plus cookies.

    Both halves are load-bearing. The DAAC 302s the protected URL to URS `oauth/authorize`, which
    answers `401` with a Basic challenge; URS then 302s back to the DAAC's `login` callback, which
    sets a *session cookie* and 302s once more to the file. Drop the cookies and the chain loops
    until the redirect cap; drop the auth and it never starts.

    Credentials are read but never returned or logged. urllib sends Basic only in answer to a 401,
    so the `Authorization` header is not replayed onto whatever host the final redirect lands on.
    """
    path = Path(netrc_path) if netrc_path is not None else Path.home() / ".netrc"
    try:
        auth = netrc.netrc(str(path)).authenticators(host)
    except FileNotFoundError:
        raise MissingCredentials(f"no {path}\n\n{CHECKLIST}") from None
    except netrc.NetrcParseError as exc:
        raise MissingCredentials(f"{path} could not be parsed: {exc}\n\n{CHECKLIST}") from None
    if auth is None:
        raise MissingCredentials(f"no `machine {host}` entry in {path}\n\n{CHECKLIST}")

    login, _account, password = auth
    return urllib.request.build_opener(
        urllib.request.HTTPBasicAuthHandler(_HostOnlyPasswordMgr(host, login, password)),
        urllib.request.HTTPCookieProcessor(CookieJar()),
    )


def fetch_granule(cell: str, dest: Path, opener, url: str | None = None, timeout: float = 120.0) -> dict:
    """Download one granule into ``dest`` and verify it. Returns its manifest record.

    Written to a `.part` file and renamed only once verification passes, so a failed fetch leaves
    nothing behind that a re-run could mistake for done.
    """
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / f"{granule_ur(cell)}.zip"
    partial = target.with_suffix(".zip.part")

    try:
        with opener.open(url or granule_url(cell), timeout=timeout) as response:
            partial.write_bytes(response.read())
    except urllib.error.HTTPError as exc:
        partial.unlink(missing_ok=True)
        hint = f"\n\n{CHECKLIST}" if exc.code in (401, 403) else ""
        raise FetchError(f"{cell}: HTTP {exc.code} {exc.reason}{hint}") from None
    except urllib.error.URLError as exc:
        partial.unlink(missing_ok=True)
        raise FetchError(f"{cell}: {exc.reason}") from None

    try:
        record = verify_zip(partial, cell)
    except VerificationError:
        partial.unlink(missing_ok=True)
        raise

    partial.replace(target)
    record["path"] = str(target)
    record["url"] = url or granule_url(cell)
    return record


def fetch_all(
    cells: Iterable[str],
    dest: Path,
    opener,
    url_for: Callable[[str], str] | None = None,
) -> dict:
    """Fetch every cell, skipping ones already verified on disk, and write the manifest.

    One failure does not abandon the rest — 24 granules across two continents, and a single flaky
    transfer should leave 23 on disk and a manifest naming the one to retry.
    """
    url_for = url_for or granule_url
    cells = list(cells)  # iterated twice: once to fetch, once to record what was requested
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    granules: dict[str, dict] = {}
    fetched = skipped = 0
    failed: list[str] = []

    for cell in cells:
        target = dest / f"{granule_ur(cell)}.zip"
        if target.exists():
            try:
                record = verify_zip(target, cell)
            except VerificationError:
                target.unlink()  # a bad file on disk is worse than no file
            else:
                record["path"] = str(target)
                record["skipped"] = True
                granules[cell] = record
                skipped += 1
                continue
        try:
            granules[cell] = fetch_granule(cell, dest, opener, url=url_for(cell))
            fetched += 1
        except (FetchError, VerificationError) as exc:
            granules[cell] = {"verified": False, "error": str(exc).split("\n")[0]}
            failed.append(cell)

    manifest = {
        "collection": "SRTMGL1N.003",
        "cmr_collection_concept_id": CMR_COLLECTION,
        "expected_member_bytes": NUM_BYTES,
        "requested": list(cells),
        "fetched": fetched,
        "skipped": skipped,
        "failed": failed,
        "total_zip_bytes": sum(g.get("zip_bytes", 0) for g in granules.values()),
        "granules": granules,
    }
    (dest / MANIFEST_NAME).write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


# ------------------------------------------------------------------------------------------------
# `check` — everything that can be known without a login
# ------------------------------------------------------------------------------------------------


def cmr_granules(bounding_box: str, page_size: int = 2000) -> dict[str, dict]:
    """Enumerate SRTMGL1N granules in a box. Anonymous — this is the half #127 never needed a human for."""
    url = (
        f"{CMR_SEARCH}?collection_concept_id={CMR_COLLECTION}"
        f"&page_size={page_size}&bounding_box={bounding_box}"
    )
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = json.load(response)

    out: dict[str, dict] = {}
    for item in payload.get("items", []):
        umm = item["umm"]
        urls = [r["URL"] for r in umm.get("RelatedUrls", []) if r.get("Type") == "GET DATA"]
        sizes = umm.get("DataGranule", {}).get("ArchiveAndDistributionInformation", [])
        out[umm["GranuleUR"]] = {
            "url": urls[0] if urls else None,
            "size_mb": sizes[-1].get("Size") if sizes else None,
        }
    return out


#: The two boxes P125 §13.3 names. The Atlas box (-11,28,12,37) is deliberately absent — #126 cut it.
CMR_BOXES = {"himalaya": "72,26,98,37", "andes": "-79,-40,-64,-8"}


def check_draw_against_cmr() -> dict:
    """Confirm all 24 granules exist, that this script's URLs match CMR's, and price the download."""
    catalog: dict[str, dict] = {}
    for box in CMR_BOXES.values():
        catalog.update(cmr_granules(box))

    rows, missing, mismatched, total_mb = [], [], [], 0.0
    for cell in DRAW_CELLS:
        entry = catalog.get(granule_ur(cell))
        if entry is None:
            missing.append(cell)
            continue
        if entry["url"] != granule_url(cell):
            mismatched.append(cell)
        total_mb += entry["size_mb"] or 0.0
        rows.append({"cell": cell, "size_mb": entry["size_mb"], "url_matches": entry["url"] == granule_url(cell)})

    return {
        "catalog_size": len(catalog),
        "found": len(rows),
        "missing": missing,
        "url_mismatched": mismatched,
        "total_mb": round(total_mb, 2),
        "granules": rows,
    }


def check_auth(netrc_path: Path | None = None) -> dict:
    """Report which of the three login states we are in, without downloading a granule.

    A `Range: bytes=0-0` request is enough: the redirect chain and the challenge happen before any
    payload does, so this costs one byte on success and nothing on failure.
    """
    try:
        opener = build_opener(netrc_path=netrc_path)
    except MissingCredentials as exc:
        # `~`, not the absolute path: this record is committed, and a home directory in it is
        # noise that differs per machine while saying nothing the reader needs.
        detail = str(exc).split("\n")[0].replace(str(Path.home()), "~")
        return {"state": "no-credentials", "detail": detail, "ok": False}

    request = urllib.request.Request(granule_url(DRAW_CELLS[0]), headers={"Range": "bytes=0-0"})
    try:
        with opener.open(request, timeout=60) as response:
            return {"state": "ok", "detail": f"HTTP {response.status} from {response.url.split('?')[0]}", "ok": True}
    except urllib.error.HTTPError as exc:
        state = "rejected" if exc.code in (401, 403) else "error"
        return {"state": state, "detail": f"HTTP {exc.code} {exc.reason}", "ok": False}
    except urllib.error.URLError as exc:
        return {"state": "error", "detail": str(exc.reason), "ok": False}


# ------------------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------------------


def _cmd_check(args) -> int:
    print(f"Draw: {len(DRAW_CELLS)} cells across {len(DRAW)} groups (#126: Atlas cut, Andes kept)")
    report = check_draw_against_cmr()
    print(f"CMR:  {report['found']}/{len(DRAW_CELLS)} granules found in {report['catalog_size']} catalogued, "
          f"{report['total_mb']} MB total")
    if report["missing"]:
        print(f"      MISSING: {', '.join(report['missing'])}")
    if report["url_mismatched"]:
        print(f"      URL MISMATCH: {', '.join(report['url_mismatched'])}")

    auth = check_auth(netrc_path=args.netrc)
    print(f"Auth: {auth['state']} — {auth['detail']}")
    if not auth["ok"]:
        print(f"\n{CHECKLIST}")
    if args.out:
        record = {
            "checked_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "draw_cells": list(DRAW_CELLS),
            "cmr": report,
            "auth": auth,
        }
        Path(args.out).write_text(json.dumps(record, indent=1) + "\n")
        print(f"\nwrote {args.out}")
    return 0 if auth["ok"] and not report["missing"] else 1


def _cmd_fetch(args) -> int:
    try:
        opener = build_opener(netrc_path=args.netrc)
    except MissingCredentials as exc:
        print(exc, file=sys.stderr)
        return 2

    cells = args.cells or list(DRAW_CELLS)
    manifest = fetch_all(cells, Path(args.dest), opener)
    for cell in cells:
        record = manifest["granules"][cell]
        mark = "skip" if record.get("skipped") else ("ok" if record.get("verified") else "FAIL")
        detail = record.get("error") or f"{record.get('zip_bytes', 0):,} bytes"
        print(f"  {mark:>4}  {cell}  {detail}")
    print(
        f"\n{manifest['fetched']} fetched, {manifest['skipped']} already present, "
        f"{len(manifest['failed'])} failed — {manifest['total_zip_bytes']:,} bytes in {args.dest}"
    )
    return 1 if manifest["failed"] else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--netrc", type=Path, default=None, help="netrc file (default ~/.netrc)")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="verify the draw against CMR and probe the login (downloads nothing)")
    check.add_argument("--out", type=Path, default=None, help="write the report as JSON")
    check.set_defaults(func=_cmd_check)

    fetch = sub.add_parser("fetch", help="download and verify the granules")
    fetch.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    fetch.add_argument("--cells", nargs="*", default=None, help="subset of the draw (default: all 24)")
    fetch.set_defaults(func=_cmd_fetch)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
