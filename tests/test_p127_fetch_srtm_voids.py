"""Pin the fetch that #127 cannot run itself, so the half that *is* agent work is already proved.

#127's one irreducible step is a human clicking Earthdata Login's consent screen. Everything
either side of it — the draw list #126 settled, the URL template, the redirect chain, and the
12,967,201-byte verification that separates a real `.num` from a login page saved under a `.zip`
name — is ordinary code, and this file is what says it works before anyone has an account.

**Nothing here touches the network.** The redirect-chain test stands up a local HTTP server that
imitates the three hops the LP DAAC actually makes (protected URL -> URS `oauth/authorize` with a
Basic challenge -> `login` callback that sets the session cookie -> back to the file). That is the
part most likely to be wrong on the first real run, and the part hardest to debug against a server
that answers `401` to everything until a human has clicked through once.

The byte count is the load-bearing assertion. P125 §13.2's warning is that the failure mode is not
an error — it is a 302 into a login form, saved to disk, named `.zip`, of plausible size. A test
that only checks "the file exists" would pass on exactly the corpus that is worthless.
"""

from __future__ import annotations

import base64
import importlib.util
import io
import json
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO / "scripts" / "p127_fetch_srtm_voids.py"
PREFLIGHT_PATH = REPO / "docs" / "research" / "p127-fetch-preflight.json"


def _load_script():
    spec = importlib.util.spec_from_file_location("p127_fetch_srtm_voids", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return _load_script()


# --------------------------------------------------------------------------------------------
# The draw, and the URLs it resolves to
# --------------------------------------------------------------------------------------------


def test_draw_is_the_twenty_four_cells_126_settled(script):
    """#126 cut the Atlas and kept the Andes. The list is 24 unique cells across two regions."""
    cells = script.DRAW_CELLS
    assert len(cells) == 24
    assert len(set(cells)) == 24, "a repeated cell would download twice and inflate the count"

    himalaya = [c for c in cells if c.startswith("N")]
    andes = [c for c in cells if c.startswith("S")]
    assert len(himalaya) == 17
    assert len(andes) == 7, "#126: the Andes stay, for geographic diversity, not for volume"

    # The Atlas box is 28-37N, 11W-12E. Nothing in the draw may fall inside it.
    for cell in himalaya:
        lon = int(cell[4:7]) * (1 if cell[3] == "E" else -1)
        assert lon >= 74, f"{cell} looks like an Atlas cell; #126 cut the Atlas"


def test_granule_url_matches_the_lp_daac_template(script):
    """Checked against CMR's own `GET DATA` URL for all 24 granules on 2026-08-11."""
    assert script.granule_ur("N28E086") == "N28E086.SRTMGL1N.num"
    assert script.granule_url("N28E086") == (
        "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/SRTMGL1N.003/"
        "N28E086.SRTMGL1N.num/N28E086.SRTMGL1N.num.zip"
    )
    assert script.granule_url("S33W070").endswith("/S33W070.SRTMGL1N.num/S33W070.SRTMGL1N.num.zip")


def test_the_num_raster_is_one_byte_per_srtm_post(script):
    """3601 x 3601 uint8. The constant is the whole verification, so derive it, don't type it."""
    assert script.NUM_BYTES == 3601 * 3601 == 12_967_201


def test_the_preflight_record_still_describes_this_draw(script):
    """The committed pre-flight is what #127 rests on before anyone logs in. Keep it honest.

    It is a *record of a CMR query*, not a live check — this test does not hit the network. What
    it catches is the record drifting out of the draw: a cell added to `DRAW` and never priced, or
    a URL template edited after the day CMR agreed with it.
    """
    record = json.loads(PREFLIGHT_PATH.read_text())

    assert tuple(record["draw_cells"]) == script.DRAW_CELLS
    cmr = record["cmr"]
    assert cmr["found"] == len(script.DRAW_CELLS) == 24
    assert cmr["missing"] == [], "a granule CMR does not hold cannot be fetched"
    assert cmr["url_mismatched"] == [], "this script's URL template disagreed with CMR's own"
    assert all(row["url_matches"] for row in cmr["granules"])
    assert [row["cell"] for row in cmr["granules"]] == list(script.DRAW_CELLS)
    assert cmr["total_mb"] == pytest.approx(28.43, abs=0.01), "P125 §13.2's 28.4 MB, from CMR"


# --------------------------------------------------------------------------------------------
# Verification — the half of the job that decides whether the fetch was real
# --------------------------------------------------------------------------------------------


def _num_zip(path: Path, member: str, size: int) -> Path:
    """A zip shaped like the DAAC's: one member, uncompressed size ``size``."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(member, b"\x00" * size)
    return path


def test_verify_accepts_a_well_formed_num_zip(script, tmp_path):
    path = _num_zip(tmp_path / "N28E086.SRTMGL1N.num.zip", "N28E086.num", script.NUM_BYTES)

    record = script.verify_zip(path, "N28E086")

    assert record["member"] == "N28E086.num"
    assert record["member_bytes"] == script.NUM_BYTES
    assert record["zip_bytes"] == path.stat().st_size
    assert len(record["sha256"]) == 64


def test_verify_rejects_a_login_page_saved_as_a_zip(script, tmp_path):
    """P125 §13.2's actual failure mode: a 302 into a login form, on disk, named `.zip`."""
    path = tmp_path / "N28E086.SRTMGL1N.num.zip"
    path.write_bytes(b"<html><body>Earthdata Login</body></html>")

    with pytest.raises(script.VerificationError) as excinfo:
        script.verify_zip(path, "N28E086")

    assert "not a zip" in str(excinfo.value).lower()


def test_verify_rejects_a_short_member_and_names_both_numbers(script, tmp_path):
    path = _num_zip(tmp_path / "N28E086.SRTMGL1N.num.zip", "N28E086.num", 1201 * 1201)

    with pytest.raises(script.VerificationError) as excinfo:
        script.verify_zip(path, "N28E086")

    message = str(excinfo.value)
    assert "12967201" in message.replace(",", "")
    assert "1442401" in message.replace(",", ""), "say what arrived, not just what was wanted"


def test_verify_rejects_the_wrong_cell(script, tmp_path):
    """A mis-templated URL that 200s on the wrong granule is silent unless the name is checked."""
    path = _num_zip(tmp_path / "N28E086.SRTMGL1N.num.zip", "N28E087.num", script.NUM_BYTES)

    with pytest.raises(script.VerificationError) as excinfo:
        script.verify_zip(path, "N28E086")

    assert "N28E087.num" in str(excinfo.value)


def test_verify_rejects_a_multi_member_archive(script, tmp_path):
    path = tmp_path / "N28E086.SRTMGL1N.num.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("N28E086.num", b"\x00" * script.NUM_BYTES)
        zf.writestr("README.txt", b"surprise")

    with pytest.raises(script.VerificationError) as excinfo:
        script.verify_zip(path, "N28E086")

    assert "2 members" in str(excinfo.value)


# --------------------------------------------------------------------------------------------
# Credentials — the seam where the human step becomes an actionable message
# --------------------------------------------------------------------------------------------


def test_missing_earthdata_credentials_points_at_the_registration_form(script, tmp_path):
    netrc = tmp_path / "netrc"
    netrc.write_text("machine api.wandb.ai login user password secret\n")

    with pytest.raises(script.MissingCredentials) as excinfo:
        script.build_opener(netrc_path=netrc)

    message = str(excinfo.value)
    assert "urs.earthdata.nasa.gov" in message
    assert "users/new" in message, "the message is the human checklist, or it is not actionable"


def test_an_absent_netrc_is_the_same_missing_credential(script, tmp_path):
    with pytest.raises(script.MissingCredentials):
        script.build_opener(netrc_path=tmp_path / "does-not-exist")


# --------------------------------------------------------------------------------------------
# The redirect chain, against a local imitation of URS
# --------------------------------------------------------------------------------------------


class _URSHandler(BaseHTTPRequestHandler):
    """Three hops: protected file -> Basic-challenged authorize -> cookie-setting callback.

    Deliberately strict about the cookie: the file 302s back to `authorize` forever without one,
    so a client that drops cookies across the redirect chain hits the redirect cap rather than
    quietly succeeding. That is the bug this server exists to catch.
    """

    username = "demae-tester"
    password = "hunter2"
    payload = b""
    protected = "/lp-prod-protected/SRTMGL1N.003/N28E086.SRTMGL1N.num/N28E086.SRTMGL1N.num.zip"

    def log_message(self, *args):  # keep pytest output clean
        pass

    def do_GET(self):
        path = urlparse(self.path).path
        if path == self.protected:
            if "edl-session=granted" in self.headers.get("Cookie", ""):
                self.send_response(200)
                self.send_header("Content-Type", "application/zip")
                self.send_header("Content-Length", str(len(self.payload)))
                self.end_headers()
                self.wfile.write(self.payload)
            else:
                self._redirect(f"/oauth/authorize?state={path}")
            return
        if path == "/oauth/authorize":
            expected = base64.b64encode(f"{self.username}:{self.password}".encode()).decode()
            if self.headers.get("Authorization") != f"Basic {expected}":
                self.send_response(401)
                self.send_header("WWW-Authenticate", 'Basic realm="Earthdata Login"')
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self._redirect(self.protected, cookie="edl-session=granted; Path=/")
            return
        self.send_error(404)

    def _redirect(self, location: str, cookie: str | None = None):
        self.send_response(302)
        self.send_header("Location", location)
        if cookie:
            self.send_header("Set-Cookie", cookie)
        self.send_header("Content-Length", "0")
        self.end_headers()


@pytest.fixture
def urs_server(script):
    """A local stand-in for `urs.earthdata.nasa.gov` + the DAAC, yielding ``(base_url, handler)``.

    The handler is a fresh subclass per test rather than `_URSHandler` itself: the suite runs
    under `-n auto`, and its licence to do so (pyproject's `dev` group) is that no test writes
    state another test can see. A class attribute set on the shared base would be exactly that.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("N28E086.num", b"\x00" * script.NUM_BYTES)
    handler = type("_URSHandlerForTest", (_URSHandler,), {"payload": buf.getvalue()})

    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", handler
    finally:
        server.shutdown()
        server.server_close()


def _netrc_for(tmp_path: Path, host: str, user: str, password: str) -> Path:
    netrc = tmp_path / "netrc"
    netrc.write_text(f"machine {host} login {user} password {password}\n")
    netrc.chmod(0o600)
    return netrc


def test_fetch_follows_the_redirect_chain_and_keeps_the_session_cookie(script, tmp_path, urs_server):
    base, handler = urs_server
    netrc = _netrc_for(tmp_path, "127.0.0.1", handler.username, handler.password)
    opener = script.build_opener(netrc_path=netrc, host="127.0.0.1")
    dest = tmp_path / "num"

    record = script.fetch_granule("N28E086", dest, opener, url=base + handler.protected)

    assert record["verified"] is True
    assert record["member_bytes"] == script.NUM_BYTES
    assert (dest / "N28E086.SRTMGL1N.num.zip").exists()


def test_wrong_credentials_surface_as_an_unauthorized_fetch(script, tmp_path, urs_server):
    base, handler = urs_server
    netrc = _netrc_for(tmp_path, "127.0.0.1", handler.username, "wrong-password")
    opener = script.build_opener(netrc_path=netrc, host="127.0.0.1")

    with pytest.raises(script.FetchError) as excinfo:
        script.fetch_granule("N28E086", tmp_path / "num", opener, url=base + handler.protected)

    assert "401" in str(excinfo.value)


def test_a_failed_fetch_leaves_no_partial_file_behind(script, tmp_path, urs_server):
    """A half-written zip that survives the failure is what makes a re-run look already-done."""
    base, handler = urs_server
    netrc = _netrc_for(tmp_path, "127.0.0.1", handler.username, "wrong-password")
    opener = script.build_opener(netrc_path=netrc, host="127.0.0.1")
    dest = tmp_path / "num"

    with pytest.raises(script.FetchError):
        script.fetch_granule("N28E086", dest, opener, url=base + handler.protected)

    assert list(dest.glob("*")) == []


def test_fetch_all_writes_a_manifest_of_verified_sizes(script, tmp_path, urs_server):
    """The manifest is what #127 asks to be recorded: the granules fetched, and their sizes."""
    base, handler = urs_server
    netrc = _netrc_for(tmp_path, "127.0.0.1", handler.username, handler.password)
    opener = script.build_opener(netrc_path=netrc, host="127.0.0.1")
    dest = tmp_path / "num"

    manifest = script.fetch_all(
        ["N28E086"], dest, opener, url_for=lambda cell: base + handler.protected
    )

    assert manifest["expected_member_bytes"] == script.NUM_BYTES
    assert manifest["fetched"] == 1
    assert manifest["failed"] == []
    entry = manifest["granules"]["N28E086"]
    assert entry["verified"] is True
    assert entry["member"] == "N28E086.num"

    written = json.loads((dest / "fetch-manifest.json").read_text())
    assert written == manifest


def test_the_manifest_says_where_the_credentials_live_but_never_what_they_are(script, tmp_path, urs_server):
    """#127: record "where the credentials live (a `.netrc` path, not the secret itself)"."""
    base, handler = urs_server
    netrc = _netrc_for(tmp_path, "127.0.0.1", handler.username, handler.password)
    opener = script.build_opener(netrc_path=netrc, host="127.0.0.1")
    dest = tmp_path / "num"

    manifest = script.fetch_all(
        ["N28E086"],
        dest,
        opener,
        url_for=lambda cell: base + handler.protected,
        credentials_from=netrc,
    )

    assert manifest["credentials_from"] == str(netrc)
    assert manifest["earthdata_host"] == script.EARTHDATA_HOST
    assert manifest["fetched_utc"].endswith("Z")
    assert handler.password not in json.dumps(manifest), "the secret never enters the record"


def test_a_recorded_credential_path_hides_the_home_directory(script, tmp_path):
    """The manifest is read by whoever picks the work up next; `/Users/<someone>` is not for them."""
    assert script.redact_home(Path.home() / ".netrc") == "~/.netrc"
    assert script.redact_home(tmp_path / "netrc") == str(tmp_path / "netrc")


def test_fetch_all_skips_what_is_already_verified_on_disk(script, tmp_path, urs_server):
    """28 MB is cheap, but a re-run after a partial failure should not re-pull the good tiles."""
    base, handler = urs_server
    netrc = _netrc_for(tmp_path, "127.0.0.1", handler.username, handler.password)
    opener = script.build_opener(netrc_path=netrc, host="127.0.0.1")
    dest = tmp_path / "num"
    script.fetch_all(["N28E086"], dest, opener, url_for=lambda cell: base + handler.protected)

    def refuse(cell):
        raise AssertionError("already-verified granule was re-fetched")

    manifest = script.fetch_all(["N28E086"], dest, opener, url_for=refuse)

    assert manifest["fetched"] == 0
    assert manifest["skipped"] == 1
    assert manifest["granules"]["N28E086"]["verified"] is True


def test_fetch_all_accepts_an_iterator_of_cells(script, tmp_path, urs_server):
    """`requested` is read after the loop, so a generator argument must not arrive empty."""
    base, handler = urs_server
    netrc = _netrc_for(tmp_path, "127.0.0.1", handler.username, handler.password)
    opener = script.build_opener(netrc_path=netrc, host="127.0.0.1")

    manifest = script.fetch_all(
        iter(["N28E086"]), tmp_path / "num", opener, url_for=lambda cell: base + handler.protected
    )

    assert manifest["requested"] == ["N28E086"]
    assert manifest["fetched"] == 1


def test_fetch_all_records_a_failure_without_abandoning_the_rest(script, tmp_path, urs_server):
    base, handler = urs_server
    netrc = _netrc_for(tmp_path, "127.0.0.1", handler.username, handler.password)
    opener = script.build_opener(netrc_path=netrc, host="127.0.0.1")
    dest = tmp_path / "num"

    def url_for(cell):
        return base + (handler.protected if cell == "N28E086" else "/nope")

    manifest = script.fetch_all(["S33W070", "N28E086"], dest, opener, url_for=url_for)

    assert manifest["fetched"] == 1
    assert manifest["failed"] == ["S33W070"]
    assert "404" in manifest["granules"]["S33W070"]["error"]
    assert manifest["granules"]["N28E086"]["verified"] is True
