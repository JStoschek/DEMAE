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
WATCH = SCRIPTS / "watch_run.sh"


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


# --- Watching a run to its end (#82's idle-cost bug) --------------------------


def matches(pattern: str, command_line: str) -> bool:
    """Would ``pgrep -f pattern`` match a process whose command line is ``command_line``?

    ``pgrep -f`` applies an extended regular expression to each process's *full command line*,
    so that is the operation modelled here — ``grep -E`` over the literal string, which is the
    same match on the same two inputs.

    Not spawning a real process and calling ``pgrep``, for two reasons that both matter: the
    box is Linux while this suite runs on the laptop, and macOS's ``pgrep`` has no ``-c`` at
    all — so the "real" version would exercise a usage error here and pass for the wrong
    reason. Racing a ``Popen`` against a process-table scan would make it flaky on top.
    """
    done = subprocess.run(
        ["grep", "-qE", pattern],
        input=command_line,
        capture_output=True,
        text=True,
    )
    return done.returncode == 0


def test_the_liveness_pattern_cannot_match_its_own_command_line():
    """The #82 bug, pinned: `pgrep -f "pretrain.launch"` matches the ssh that carries it.

    R1 finished at ~07:17 UTC and billed until ~14:53 because of this one property. The
    watcher asked the box `pgrep -fc "pretrain.launch"`, and `pgrep -f` matches full command
    lines — so the shell ssh spawned to run the question was itself an answer. The count never
    reached zero, the completion branch never ran, and 7.6 idle hours cost more than the run.

    So the assertion is not "the pattern finds the trainer" — the broken one did that too. It
    is that the pattern **fails to find itself**, which is the property that lets zero mean
    zero.
    """
    raw = "pretrain.launch"
    probed = call(WATCH, f'probe_pattern "{raw}"')

    # It still describes the same processes...
    assert matches(probed, "uv run python -m pretrain.launch --set foo=1")
    # ...but a command line quoting the probe itself is not one of them. This is the whole fix.
    assert not matches(probed, f"ssh demae-box pgrep -fc '{probed}'")
    # And the naive form is exactly what fails that, which is why the bracket has to stay.
    assert matches(raw, f"ssh demae-box pgrep -fc '{raw}'")


def test_an_unreachable_box_is_never_reported_as_a_finished_run():
    """The three non-running states are distinct, and conflating them is the expensive bug.

    A vanished vast instance, a crashed trainer and a clean finish all present as "no process".
    Reporting the first two as `finished` sends someone to tear down a box whose checkpoints
    still needed rescuing; reporting a finish as anything else keeps paying for an idle GPU.
    """
    # ssh failed: the state is unknown, and unknown is never success — regardless of what the
    # last poll happened to see.
    assert call(WATCH, 'classify 0 0 1') == "unreachable"
    assert call(WATCH, 'classify 0 0 0') == "unreachable"
    # Reachable and alive is the quiet case.
    assert call(WATCH, 'classify 1 3 0') == "running"
    # Gone, with launch.py:339's summary line in the log: the run ended on its own terms.
    assert call(WATCH, 'classify 1 0 1') == "finished"
    # Gone without it: killed, OOMed, or crashed. Not a finish, and not a teardown signal.
    assert call(WATCH, 'classify 1 0 0') == "died"


def test_finishing_is_a_positive_signal_rather_than_an_absence():
    """`finished` requires the completion line, so a killed run cannot be mistaken for a done one.

    Process absence is symmetric — it is what a finish, a crash and a `kill -9` all look like
    from outside. The only asymmetric evidence is the line `pretrain.launch` prints when it
    completes (launch.py:339), so that is what the state is keyed on.
    """
    assert call(WATCH, 'classify 1 0 1') == "finished"
    assert call(WATCH, 'classify 1 0 0') != "finished"


def test_the_deadline_fires_without_observing_the_run():
    """The backstop for the case where I am wrong about the rest of the watcher.

    Every other check here measures something — a process table, a log, whether ssh answered —
    and each can be wrong in the direction that costs money, because the expensive failure is
    always "looks like it is still running". #82 was one way to get that. A deadline is the
    only check that cannot be fooled by a bad measurement, because it does not take one.
    """
    hour = 3600
    assert not truth(WATCH, f"overdue $(( 23 * {hour} )) 24")
    assert truth(WATCH, f"overdue $(( 25 * {hour} )) 24")
    # Exactly at the limit is not yet past it.
    assert not truth(WATCH, f"overdue $(( 24 * {hour} )) 24")
    # 0 disables — R5 may legitimately run 86 hours, and a backstop that fires mid-run would
    # train the operator to ignore it, which is worse than not having one.
    assert not truth(WATCH, f"overdue $(( 500 * {hour} )) 0")


def test_the_watcher_refuses_to_arm_without_a_host():
    done = subprocess.run([str(WATCH)], capture_output=True, text=True)
    assert done.returncode == 2
    assert "--host" in done.stderr


def test_an_unreachable_box_fails_loudly_rather_than_with_ssh_s_bare_255():
    """A watcher that dies in silence is the failure it exists to prevent, one level up.

    `set -e` plus a failing ssh inside a command substitution kills the script at the
    assignment, so the diagnosis never prints and the operator gets a bare 255 — which reads
    like a broken invocation rather than an unreachable box. Found while testing the fix for
    #82, which is the point: the same silence, one layer out.
    """
    done = subprocess.run(
        [str(WATCH), "--host", "demae-nonexistent.invalid"],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 2, f"expected a diagnosed exit, got {done.returncode}"
    assert "cannot reach" in done.stderr
    assert "255" not in done.stderr
