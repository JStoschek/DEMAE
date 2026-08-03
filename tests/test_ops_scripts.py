"""The parts of `scripts/*.sh` that are arithmetic rather than ssh (issue #107).

The ops scripts run on a rented GPU box against a real bucket, so most of what they do cannot
be tested anywhere else. Two things inside them *are* pure functions, and both were load-bearing
enough in R0 (#81) to be worth pinning here:

- **preflight's headroom sum** — the check that decides whether a box is worth provisioning. It
  compared a fixed 70 GB against *total free space*, which is right exactly once: on a box that
  has downloaded nothing. On an already-provisioned box the corpus it is reserving for is the
  thing occupying the disk, so provisioning refused to run against the box it had itself built.
- **the rclone version floor** — the vastai image ships a 2022 build that fails the first upload
  attempt of every checkpoint, and the comparison that spots it must be numeric, not
  lexicographic.

Each script defines these before it does anything, and returns early when ``DEMAE_LIB`` is set,
so a test can source it and call them with no box, no GPU and no network. That guard is the only
thing in those scripts that exists for this file's benefit; everything below drives the same code
the box runs.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
PROVISION = SCRIPTS / "provision_box.sh"
SYNC = SCRIPTS / "sync_checkpoints.sh"
PUSH = SCRIPTS / "push_code.sh"


def call(script: Path, snippet: str) -> str:
    """Source ``script`` in library mode and run ``snippet``, returning its stdout."""
    done = subprocess.run(
        ["bash", "-c", f'export DEMAE_LIB=1; . "{script}"; {snippet}'],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, f"{snippet!r} failed:\n{done.stdout}\n{done.stderr}"
    return done.stdout.strip()


def truth(script: Path, snippet: str) -> bool:
    """``snippet`` as a shell *condition* — its exit status, not its output."""
    done = subprocess.run(
        ["bash", "-c", f'export DEMAE_LIB=1; . "{script}"; {snippet}'],
        capture_output=True,
        text=True,
    )
    assert done.returncode in (0, 1), f"{snippet!r} errored:\n{done.stderr}"
    return done.returncode == 0


# --- Preflight headroom (#107 item 2) -----------------------------------------


@pytest.mark.parametrize(
    "present_gb, have_venv, skip_corpus, skip_env, expected, why",
    [
        # The fresh box, and the number this replaces. A machine with nothing on it still has
        # to land all three things, so the new sum reduces to the old constant exactly — which
        # is what says this loosened the check for provisioned boxes without loosening it for
        # the case it was written for.
        (0, 0, 0, 0, 70, "a fresh box needs corpus + venv + checkpoint room, i.e. the old 70"),
        # R0's box (#81): 102 GB disk, 53 GB of it already corpus and venv, 50 GB free. Under
        # the old check this failed at step 1/8 and provisioning could not be re-run against
        # the box it had itself provisioned.
        (47, 1, 0, 0, 17, "everything is already down; only a run's own writes are still coming"),
        # A copy killed half way. What is left to land is the *remainder*, which is the whole
        # idea: 27 GB of corpus, plus a venv that is not there yet, plus the reserve.
        (20, 0, 0, 0, 50, "a resumed download needs only the shards it is missing"),
        # The flags the runbook documents for verifying an existing box.
        (47, 1, 1, 1, 15, "--skip-corpus --skip-env still reserves what a run will write"),
        # Rounding, and the case where the disk holds more than the manifest describes.
        (60, 1, 0, 0, 17, "a corpus larger than the estimate does not earn negative headroom"),
    ],
)
def test_preflight_reserves_what_still_has_to_land(
    present_gb, have_venv, skip_corpus, skip_env, expected, why
):
    got = call(PROVISION, f"required_free_gb {present_gb} {have_venv} {skip_corpus} {skip_env}")
    assert int(got) == expected, why


def test_the_checkpoint_reserve_survives_every_flag():
    """No combination of flags reserves nothing.

    The corpus and the venv are things that *have* landed or have not; the checkpoints a run
    writes are always ahead of it. A box that passes preflight with 2 GB free would provision
    cleanly and then die 40 hours in with the disk full, which is the one failure mode this
    check exists to make cheap.
    """
    for skip_corpus in (0, 1):
        for skip_env in (0, 1):
            got = int(call(PROVISION, f"required_free_gb 47 1 {skip_corpus} {skip_env}"))
            assert got >= int(call(PROVISION, "printf %s $CKPT_GB"))


def test_r0s_box_would_now_provision_and_a_genuinely_small_one_still_would_not():
    """The two ends of the decision, stated as the disk sizes that produced them."""
    provisioned = int(call(PROVISION, "required_free_gb 47 1 0 0"))
    fresh = int(call(PROVISION, "required_free_gb 0 0 0 0"))
    assert provisioned <= 50 < fresh  # R0's box had 50 GB free with the corpus already down


# --- The rclone floor (#107 item 4) -------------------------------------------


@pytest.mark.parametrize("script", [PROVISION, SYNC], ids=["provision", "sync"])
class TestRcloneFloor:
    """Both scripts carry the comparison, so both are tested.

    The duplication is deliberate and matches how the R2 signing-region swap is already carried
    by every bucket-touching path here: `sync_checkpoints.sh` runs on the box in its own tmux
    window and cannot assume anything about what provisioning left behind.
    """

    @pytest.mark.parametrize(
        "line, expected",
        [
            # What the vastai image actually reports. The suffix is a build stamp, not a
            # release, so a comparison that keeps it is comparing strings.
            ("rclone v1.60.1-DEV", "1.60.1"),
            ("rclone v1.69.1", "1.69.1"),
            ("rclone v1.71.0-beta.8123.abcdef", "1.71.0"),
            ("v1.65", "1.65.0"),
            # Unparseable reads as ancient, so the floor rejects it and provisioning installs a
            # known one. The other direction would let an unreadable version silently pass.
            ("rclone: command not found", "0.0.0"),
            ("", "0.0.0"),
        ],
    )
    def test_the_version_is_read_as_three_numbers(self, script, line, expected):
        assert call(script, f'rclone_semver "{line}"') == expected

    @pytest.mark.parametrize(
        "have, floor, below",
        [
            ("1.60.1", "1.65.0", True),
            ("1.65.0", "1.65.0", False),  # the floor itself is not below the floor
            ("1.69.1", "1.65.0", False),
            ("2.0.0", "1.65.0", False),
            # The regression that a `<` on strings would pass and a `sort` on text would get
            # wrong: 9 > 65 as text, and 1.9 < 1.65 as a version.
            ("1.9.0", "1.65.0", True),
            ("1.100.0", "1.99.0", False),
            ("1.65.0", "1.65.1", True),  # patch level counts
            ("0.0.0", "1.65.0", True),  # the unreadable case, from the parametrize above
        ],
    )
    def test_versions_compare_as_numbers_not_text(self, script, have, floor, below):
        assert truth(script, f'rclone_version_lt "{have}" "{floor}"') is below

    def test_the_2022_build_r0_hit_is_below_the_floor(self, script):
        old = call(script, 'rclone_semver "rclone v1.60.1-DEV"')
        assert truth(script, f'rclone_version_lt "{old}" "$RCLONE_MIN"')


def test_both_scripts_agree_on_the_floor():
    """A floor that drifted between the two would let provisioning bless what sync rejects."""
    assert call(PROVISION, "printf %s $RCLONE_MIN") == call(SYNC, "printf %s $RCLONE_MIN")


def test_the_pinned_version_satisfies_the_floor_it_ships():
    """The pin is what provisioning installs; the floor is what it accepts.

    Nothing else checks that they point the same way, and a pin below its own floor would
    install rclone, reject it, and die — after the download.
    """
    assert not truth(PROVISION, 'rclone_version_lt "$RCLONE_PIN" "$RCLONE_MIN"')


# --- Shipping .git_sha (#107 item 1) ------------------------------------------


def test_the_push_ships_the_sha_file_with_the_code():
    """R0's run header read `git sha unknown` because this one file was not in the rsync.

    The mechanism #80 built works — `git_sha()` falls back to `.git_sha`, and provisioning
    prints what it finds. What failed was that generating and shipping it was a line in a
    document, and the code went up without it. So the assertion is not about the file's
    contents but about the *rsync list*: the sha cannot be pushed separately from the code it
    describes, because it is never a separate command.
    """
    plan = subprocess.run(
        [str(PUSH), "--host", "demae-box", "--dry-run"],
        capture_output=True,
        text=True,
    )
    assert plan.returncode == 0, plan.stderr
    rsync = [ln for ln in plan.stdout.splitlines() if "rsync" in ln]
    assert rsync, f"no rsync in the plan:\n{plan.stdout}"
    assert all(".git_sha" in ln for ln in rsync)
    # And it is regenerated in the same breath rather than trusted from last time: a `.git_sha`
    # from three pushes ago is a confident wrong answer, which is worse than the `unknown` it
    # replaced.
    assert "git_sha" in plan.stdout.split("rsync")[0]


def test_the_push_refuses_a_provision_it_cannot_reach():
    """`--host` is not optional — there is no sensible default for whose box this is."""
    done = subprocess.run([str(PUSH), "--dry-run"], capture_output=True, text=True)
    assert done.returncode == 2
    assert "--host" in done.stderr
