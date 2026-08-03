"""The build-provenance stamps — and specifically the no-git fallback (#80).

Phase 1 trains on a rented box that holds the code but not its history, so ``git_sha`` has a
second source there. What these tests hold is the *ordering*: git is authoritative wherever it
can answer, and ``.git_sha`` only speaks into the silence. Get that backwards and a stale file
in a real checkout would misattribute every artifact the repo writes — the catalog's
``code_version``, every report header, the corpus meta — not just a pretraining run.
"""

from __future__ import annotations

import subprocess

import pytest

from dataset.provenance import SHA_FILE, git_sha


def _git(path, *args):
    return subprocess.run(["git", *args], cwd=path, capture_output=True, text=True, check=True)


def _init_repo(path):
    """A real one-commit repo at ``path``, or skip if git is unavailable."""
    if not _has_git():
        pytest.skip("git not on PATH")
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "t@example.com")
    _git(path, "config", "user.name", "t")
    (path / "f.txt").write_text("x")
    _git(path, "add", "f.txt")
    _git(path, "commit", "-qm", "c")


def _has_git() -> bool:
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        return False
    return True


@pytest.fixture
def outside_a_repo(tmp_path, monkeypatch):
    """A working directory where ``git rev-parse`` genuinely fails.

    Asserted rather than assumed: if the tmp dir ever landed inside a checkout, every test
    below would still pass while testing nothing, because git would be answering instead.
    """
    monkeypatch.chdir(tmp_path)
    probe = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=tmp_path, capture_output=True
    )
    if probe.returncode == 0:
        pytest.skip(f"{tmp_path} is inside a checkout; the fallback cannot be exercised")
    return tmp_path


def test_without_git_and_without_the_file_it_is_unknown(outside_a_repo):
    assert git_sha() == "unknown"


def test_without_git_it_reads_the_shipped_sha(outside_a_repo):
    (outside_a_repo / SHA_FILE).write_text("2acf137\n")
    assert git_sha() == "2acf137"


def test_the_dirty_suffix_survives_the_trip(outside_a_repo):
    # the laptop knows the tree was dirty; the box cannot re-derive it, so it must carry
    (outside_a_repo / SHA_FILE).write_text("2acf137-dirty\n")
    assert git_sha() == "2acf137-dirty"


@pytest.mark.parametrize("junk", ["", "\n", "   \n", "\n2acf137\n"])
def test_an_empty_or_blank_first_line_is_unknown(outside_a_repo, junk):
    # a half-written file must degrade to "unknown", never to "" — sink.run_tags and the
    # checkpoint header both expect a non-empty string, and "" reads as a value
    (outside_a_repo / SHA_FILE).write_text(junk)
    assert git_sha() == "unknown"


def test_only_the_first_line_is_taken(outside_a_repo):
    # a stray second line must not reach a W&B tag, which would be dropped over MAX_TAG_CHARS
    (outside_a_repo / SHA_FILE).write_text("2acf137\nnote: written by provision_box.sh\n")
    assert git_sha() == "2acf137"


def test_surrounding_whitespace_is_stripped(outside_a_repo):
    (outside_a_repo / SHA_FILE).write_text("  2acf137  \n")
    assert git_sha() == "2acf137"


def test_git_wins_over_a_stale_file(tmp_path, monkeypatch):
    """The ordering that makes the fallback safe to ship repo-wide.

    A ``.git_sha`` left behind in a working checkout — rsynced back, committed by accident,
    stale by a hundred commits — must be inert. Only the silence of git lets it speak.
    """
    _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    (tmp_path / SHA_FILE).write_text("deadbee\n")

    resolved = git_sha()

    assert resolved != "deadbee"
    # the file itself is untracked, so the tree is dirty and git says so
    assert resolved.endswith("-dirty")


def test_a_committed_file_still_loses_to_git(tmp_path, monkeypatch):
    """The stale-file case at its most plausible: ``.git_sha`` rsynced back and committed.

    With the tree clean there is no ``-dirty`` suffix to distinguish the two answers, so this
    is where a precedence bug would read as correct — the returned value would simply be the
    wrong seven characters, on every artifact, silently.
    """
    _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    (tmp_path / SHA_FILE).write_text("deadbee\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "c2")

    expected = _git(tmp_path, "rev-parse", "--short", "HEAD").stdout.strip()

    assert git_sha() == expected != "deadbee"
