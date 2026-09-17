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

import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
PROVISION = SCRIPTS / "provision_box.sh"
SYNC = SCRIPTS / "sync_checkpoints.sh"
PUSH = SCRIPTS / "push_code.sh"
WATCH = SCRIPTS / "watch_run.sh"
RUN = SCRIPTS / "run.sh"
RUNBOOK = REPO / "docs" / "phase1-ops-runbook.md"


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


# --- Preflight's download gate ------------------------------------------------
#
# vast's advertised inet_down is the host's claim about itself and it is regularly fiction. The
# probe that checks it has to move real bytes off the real bucket, so none of it can be tested
# here — but the conversion from "this many bytes in this many milliseconds" to "reject this
# box" is arithmetic, and it is the part that decides whether money gets spent.

MIB = 1048576


@pytest.mark.parametrize(
    "mib, ms, gib, expected, why",
    [
        # 384 MiB (6 streams x 64) in 3s is ~128 MiB/s — an ordinary good host.
        (384, 3_000, 47, 7, "a fast host lands the corpus inside the coffee break"),
        # The same probe taking 25s is ~15 MiB/s, which is the case this gate exists for: it
        # provisions fine, it just costs half an hour of rent before anything trains.
        (384, 25_000, 47, 53, "a slow host is rejected on time, not on speed"),
        # A resumed box with 5 GB left is on exactly that slow host and is *fine*, which is why
        # the budget is minutes and not MiB/s.
        (384, 25_000, 5, 6, "the same speed passes when there is less left to land"),
        # Rounding is upward, at the boundary that matters: 47 GiB at 40 MiB/s is 20.05 minutes,
        # and a gate that reads that as 20 is not a 20-minute gate.
        (1000, 25_000, 47, 21, "20.05 minutes is over the 20-minute budget"),
        # Nothing left to download cannot take any time, however slow the box.
        (384, 25_000, 0, 0, "a complete corpus needs no bandwidth"),
    ],
)
def test_the_download_eta_is_what_the_gate_compares(mib, ms, gib, expected, why):
    got = call(PROVISION, f"corpus_eta_minutes {mib * MIB} {ms} {gib}")
    assert int(got) == expected, why


def test_no_bytes_at_all_is_not_a_measurement_of_zero_speed():
    """A dead network and a slow one must not land in the same branch.

    Zero bytes means the probe learned nothing — dead link, or credentials that never reached
    the shell. It comes back as an ETA nothing can pass, so the caller's own check for it runs
    before any comparison, and a divide-by-zero never gets the chance to read as "fast".
    """
    unpassable = int(call(PROVISION, "corpus_eta_minutes 0 25000 47"))
    assert unpassable > 24 * 60


def test_the_gate_implies_a_bandwidth_floor_a_human_can_check_against_a_listing():
    """What the 20-minute budget is, expressed as the number vast actually advertises.

    Not a second threshold — it is derived from the same function the box runs. It is here so
    that the machine filter in the vast console can be set to something, and so that a change to
    MAX_CORPUS_MIN or CORPUS_GB shows up as a change to this number rather than silently.
    """
    budget = int(call(PROVISION, "printf %s $MAX_CORPUS_MIN"))
    corpus = int(call(PROVISION, "printf %s $CORPUS_GB"))
    passing = [
        mibps
        for mibps in range(1, 400)
        if int(call(PROVISION, f"corpus_eta_minutes {mibps * MIB} 1000 {corpus}")) <= budget
    ]
    assert min(passing) == 41  # MiB/s sustained, i.e. ~340 Mbit/s on a 47 GB corpus


@pytest.mark.parametrize(
    "mib, ms, expected",
    [
        (384, 3_000, "128.0"),
        (384, 25_000, "15.3"),
        (10, 25_000, "0.4"),  # a genuinely terrible host still gets a legible number
        (0, 25_000, "0.0"),
    ],
)
def test_the_measured_rate_is_printed_for_the_human_not_the_gate(mib, ms, expected):
    assert call(PROVISION, f"probe_mib_per_s {mib * MIB} {ms}") == expected


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
    # **The transfer that carries the code** carries the sha, which is the invariant — stated
    # against `src` rather than against "every rsync" since #162 added a second transfer for the
    # plan. That one ships a document Task 1 scoring reads and no code at all, so requiring the
    # sha in it would assert something this test does not mean; requiring it wherever `src` goes
    # is exactly what R0 needed and is stronger than counting lines.
    code = [ln for ln in rsync if " src" in ln or ln.rstrip().endswith("src")]
    assert code, f"no rsync carries src:\n{plan.stdout}"
    assert all(".git_sha" in ln for ln in code)
    assert not any(" src" in ln for ln in rsync if ".git_sha" not in ln)
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


def test_a_count_of_zero_comes_back_as_two_lines_and_still_reads_as_zero():
    """Found on a live box while testing #108: `watch_run.sh --once` could never arm.

    Both counting commands the watcher runs — `pgrep -fc` and `grep -c` — print their count
    **and exit non-zero when that count is zero**. The `|| echo 0` guard exists for the other
    failure (the command missing entirely), and it fires on the ordinary empty case too, so the
    answer arrives as `0\\n0`. Every comparison against the literal `"0"` then reads false, and
    the two of them fail in opposite directions:

    - the self-test decides its sentinel *is* running, calls it the #82 bug, and refuses to arm
    - `completed_marker` decides the completion line *is* in the log — so a crashed run would
      be reported as `finished`, which is precisely the conflation that file exists to prevent

    So the parse is one function, and an unparseable answer still means "unknown" rather than
    zero, because ssh failing and nothing running must not be the same observation.
    """
    assert call(WATCH, "read_count \"$(printf '0\\n0')\"") == "0"
    assert call(WATCH, 'read_count "3"') == "3"
    assert call(WATCH, "read_count \"$(printf '2\\n0')\"") == "0", "the fallback's line wins"
    # ssh failed: nothing at all, and nothing is not zero.
    assert call(WATCH, 'read_count ""') == ""
    assert call(WATCH, 'read_count "ssh: connect to host demae-box port 22: No route"') == ""


def test_the_watcher_notices_the_sync_loop_going_quiet():
    """#108 item 2's other half: the loop can also stop *after* it has started.

    `run.sh` gates on the heartbeat once, at startup, and that closes the failure the issue
    describes — the loop that never came up. It cannot see the loop killed at hour 3 by a closed
    pane or a full disk, and that failure has the same shape: training stays healthy and the
    checkpoints stop leaving the box. The watcher is already the thing polling this box every
    five minutes from somewhere the box cannot take with it, so it is where the observation
    belongs.

    Not a terminal state. The run is still worth finishing and the checkpoints are still on the
    box; what is wanted is a line in front of the operator while there is still time to restart
    the window — so it warns and keeps watching.
    """
    minute = 60
    assert not truth(WATCH, f"sync_stalled $(( 20 * {minute} )) 90")
    assert truth(WATCH, f"sync_stalled $(( 95 * {minute} )) 90")
    # The default tolerates two missed passes at the loop's own 30-minute interval; one slow
    # upload is not a stall, and a warning that cries wolf is one the operator learns to skip.
    assert int(call(WATCH, "printf %s $SYNC_STALE_MINUTES")) >= 2 * 30
    # 0 disables, like every other backstop here.
    assert not truth(WATCH, f"sync_stalled $(( 500 * {minute} )) 0")


def test_the_watcher_and_the_loop_agree_on_where_the_heartbeat_is():
    """A watcher looking in the wrong place reports a stall that is not happening, forever.

    Same shape as `RCLONE_MIN` and `train_log_path`: the path is carried by the script that
    writes it and the script that reads it, on two different machines, and this is what keeps
    the copies equal.
    """
    for run_id in ("R1", "R5"):
        assert call(RUN, f'heartbeat_for /workspace/DEMAE {run_id}') == call(
            WATCH, f'heartbeat_path /workspace/DEMAE {run_id}'
        )


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


# --- Starting a run from wherever you are standing (#108) ---------------------
#
# The bug: §7's start command was `cd /workspace/DEMAE && tmux new -s r1`, and the vastai image's
# shell rc `cd`s every new pane back to `/workspace`. Typed by hand into an attached pane it
# happened to work; driven any other way the pane started one directory up, `uv run` found no
# `pyproject.toml`, and — the half that matters — `scripts/sync_checkpoints.sh` was not there to
# run. Training died loudly; the sync loop printed one "No such file or directory" into a window
# nobody was watching and never started, with no retry and no rising failure count to notice.
#
# So the fix is a committed script rather than a corrected paragraph: this whole class of bug is
# "documented correctly, executed differently", and a paragraph cannot be tested.


def plan(*args: str, cwd: Path | str | None = None) -> subprocess.CompletedProcess:
    """``run.sh --dry-run`` from ``cwd``. Prints what it would do and touches nothing.

    ``--dry-run`` goes first because everything after ``--`` belongs to the launcher.
    """
    return subprocess.run(
        [str(RUN), "--dry-run", *args],
        capture_output=True,
        text=True,
        cwd=str(cwd) if cwd is not None else None,
    )


def test_the_plan_does_not_depend_on_the_directory_it_is_invoked_from(tmp_path):
    """The #108 bug, pinned at its root: the pane's cwd must not reach the commands.

    R1's pane was created with `tmux new-session -c /workspace/DEMAE` and started in
    `/workspace` anyway, because the image's rc `cd`s. Every path this script emits is therefore
    resolved from the script's *own* location, which is a fact about the checkout rather than
    about whoever's shell started it.
    """
    from_repo = plan("--run-id", "R1", cwd=REPO)
    from_elsewhere = plan("--run-id", "R1", cwd=tmp_path)
    assert from_repo.returncode == 0, from_repo.stderr
    assert from_elsewhere.returncode == 0, from_elsewhere.stderr
    assert from_repo.stdout == from_elsewhere.stdout


def test_nothing_in_the_plan_is_a_relative_path(tmp_path):
    """`scripts/sync_checkpoints.sh` is the exact string that failed, so no line may hold it.

    A relative invocation is right only for a cwd nobody controls. Absolute is the property —
    not "the documented cwd is correct", which it already was.
    """
    done = plan("--run-id", "R1", cwd=tmp_path)
    offenders = [
        line
        for line in done.stdout.splitlines()
        if "scripts/" in line and f"{REPO}/scripts/" not in line
    ]
    assert not offenders, f"relative script invocation(s) in the plan:\n" + "\n".join(offenders)


def test_the_run_id_reaches_every_place_that_has_to_agree_on_it():
    """One argument, threaded — the config, the checkpoint dir, the bucket prefix, the log.

    §7 used to name the run four times in two commands, and #108 item 3 is what that costs: its
    example passed `--vit-base`, whose preset sets `run_id="R5"`, in the R1/R2/R3 position — so
    following it literally for R1 trained a ViT-B and wrote it to `runs/R5` while the sync
    window shipped `runs/R1`.
    """
    done = plan("--run-id", "R3")
    assert done.returncode == 0, done.stderr
    out = done.stdout
    assert f"{REPO}/runs/R3" in out
    assert "run_id=R3" in out                       # the config's own run id, set explicitly
    assert "demae-checkpoints/r3/" in out           # the bucket prefix
    assert f"{REPO}/r3_train.log" in out            # where watch_run.sh looks by default
    assert "tmux session     r3" in out             # and the session you attach to


def test_the_trainer_writes_where_the_watcher_reads():
    """`watch_run.sh` defaults to `<root>/<run-id lowercased>_train.log` and nothing wrote it.

    Its own default is only useful if something produces that file; §7's command printed to a
    tmux pane and no further. The two expressions are carried in both scripts, like `RCLONE_MIN`
    is, and this is what holds them in step.
    """
    for run_id in ("R1", "R5", "R10"):
        assert call(RUN, f'train_log_path /workspace/DEMAE {run_id}') == call(
            WATCH, f'train_log_path /workspace/DEMAE {run_id}'
        )


def test_the_release_preset_cannot_be_silently_attached_to_another_arm():
    """`--vit-base` is R5's, because `RunConfig.vit_base()` sets `run_id="R5"` itself.

    Not fatal — an operator may legitimately want a ViT-B smoke test under another id, and
    `--set run_id=` now makes the config agree with the directory either way. But it is said,
    because the silent version of this is a run whose recipe and whose label disagree.
    """
    mismatched = plan("--run-id", "R1", "--", "--vit-base")
    assert mismatched.returncode == 0, mismatched.stderr
    assert "R5" in mismatched.stderr, "the preset's own run id is not mentioned"

    matched = plan("--run-id", "R5", "--", "--vit-base")
    assert matched.returncode == 0, matched.stderr
    assert "R5" not in matched.stderr


def test_the_continual_preset_cannot_be_silently_attached_to_another_arm():
    """R4's preset carries a warm start as well as a run id, so the same warning is louder.

    Also pins the exactness the bash guard needs: `--vit-base-continual` must not trip the
    `--vit-base` branch, or R4's own launch line would warn about R5 every time.
    """
    mismatched = plan("--run-id", "R1", "--", "--vit-base-continual", "/w/ckpt.pth")
    assert mismatched.returncode == 0, mismatched.stderr
    assert "R4" in mismatched.stderr, "the preset's own run id is not mentioned"

    matched = plan("--run-id", "R4", "--", "--vit-base-continual", "/w/ckpt.pth")
    assert matched.returncode == 0, matched.stderr
    assert "R4" not in matched.stderr
    assert "R5" not in matched.stderr, "--vit-base-continual tripped the --vit-base guard"


@pytest.mark.parametrize(
    "passthrough, why",
    [
        (["--set", "run_id=R9"], "moves the config's run id away from the directories"),
        (["--set=run_id=R9"], "argparse's equals form is the same flag"),
        (["--checkpoint-dir", "/workspace/ckpt"], "moves where the trainer writes"),
        (["--checkpoint-dir=/workspace/ckpt"], "same, spelled with an equals sign"),
        (["--no-checkpoints"], "leaves the loop shipping a directory nothing writes to"),
    ],
)
def test_a_passthrough_that_moves_the_checkpoints_out_from_under_the_loop_is_refused(
    passthrough, why
):
    """Each of these reaches #108's outcome *through* the flag meant to prevent it.

    The sync loop is pointed at one directory, chosen from `--run-id` before anything starts.
    A launcher flag that sends the checkpoints somewhere else — or stops them being written —
    leaves a run that trains healthily, writes a heartbeat, passes every check here, and never
    gets a checkpoint off the box. That is the issue's own sentence: *"a 400-epoch run whose
    checkpoints never leave the box, with the training side looking perfectly healthy"*.
    """
    done = plan("--run-id", "R2", "--", *passthrough)
    assert done.returncode == 2, f"accepted a passthrough that {why}"
    assert "--run-id" in done.stderr


@pytest.mark.parametrize("flag", ["--heartbeat-wait", "--interval"])
def test_a_gate_that_would_wait_no_time_is_refused(flag):
    """`(( waited < abc ))` reads `abc` as zero, and a gate that waits no time passes anything.

    The one check this whole file exists to hold cannot be turned off by a typo — which is the
    same argument that makes `--run-id` validated rather than trusted.
    """
    done = plan("--run-id", "R1", flag, "abc")
    assert done.returncode == 2
    assert flag in done.stderr


def test_the_run_id_is_a_run_id():
    """It names a directory and a bucket prefix, so it is validated before either is built."""
    for bad in ("", "r1 ", "../etc", "R1;rm"):
        done = plan("--run-id", bad)
        assert done.returncode == 2, f"{bad!r} was accepted"


def test_starting_a_run_needs_a_run_id():
    done = subprocess.run([str(RUN)], capture_output=True, text=True)
    assert done.returncode == 2
    assert "--run-id" in done.stderr


def test_the_sync_loop_is_proven_up_before_any_gpu_time_is_spent():
    """The ordering is the fix for the silent half, and it is the reverse of §7's.

    §7 started training first and the sync loop second, so the loop's failure arrived after the
    GPU was already spending — into a window nobody had a reason to look at yet. Starting the
    cheap half first and refusing to start the expensive one until it has proven itself alive
    turns a silent 7.2-hour loss into a failure at the keyboard.
    """
    out = plan("--run-id", "R1").stdout
    positions = {
        key: out.index(key)
        for key in ("sync_checkpoints.sh", ".sync_heartbeat", "pretrain.launch")
    }
    assert positions["sync_checkpoints.sh"] < positions["pretrain.launch"]
    assert positions[".sync_heartbeat"] < positions["pretrain.launch"]


def test_the_plan_prints_the_commands_it_would_actually_run():
    """A dry-run that prints a hand-written copy of the command is this bug, again.

    #108 is "documented correctly, executed differently". A plan assembled separately from the
    execution path would reproduce exactly that, one layer in — so both come from the same
    arrays, and this asserts the plan carries the whole invocation rather than a summary.
    """
    out = plan("--run-id", "R2", "--", "--set", 'masking.ratio=0.9').stdout
    assert "python -m pretrain.launch" in out
    assert "masking.ratio=0.9" in out               # the passthrough reaches the trainer
    assert "--checkpoint-dir" in out and "--remote" in out   # and the sync half is spelled out


def test_the_run_reports_online_without_being_asked_to():
    """`monitoring.wandb_mode` defaults to `offline`, and §7's old command overrode it by hand.

    An override typed once per run is one more thing to forget, and forgetting it is quiet: the
    run trains fine and writes an offline directory that only exists on a box which is going to
    be destroyed. This runs on a rented box by definition, so it is what the flag was always
    set to — and an operator who wants otherwise says so after `--`, which is applied later and
    therefore wins.
    """
    out = plan("--run-id", "R1").stdout
    assert "monitoring.wandb_mode=online" in out

    overridden = plan("--run-id", "R1", "--", "--set", "monitoring.wandb_mode=offline").stdout
    launch = [ln for ln in overridden.splitlines() if "pretrain.launch" in ln][0]
    assert launch.index("wandb_mode=online") < launch.index("wandb_mode=offline"), (
        "the operator's override has to come last, or it is the one that loses"
    )


def test_the_heartbeat_wait_answers_both_ways(tmp_path):
    """The gate itself: absent means the loop never came up, present means it did."""
    beat = tmp_path / ".sync_heartbeat"
    assert not truth(RUN, f'wait_for_heartbeat "{beat}" 1')
    beat.write_text("2026-08-04T00:00:00Z pass=0\n")
    assert truth(RUN, f'wait_for_heartbeat "{beat}" 1')


def test_the_gate_gives_up_the_moment_the_window_is_gone(tmp_path):
    """#108's own failure kills the loop in milliseconds; waiting 120s to say so trains
    the operator to ignore the wait. So the gate takes a liveness check and believes it."""
    beat = tmp_path / ".sync_heartbeat"
    started = time.monotonic()
    assert not truth(RUN, f'dead() {{ return 0; }}; wait_for_heartbeat "{beat}" 120 dead')
    assert time.monotonic() - started < 20, "it waited out the deadline it could have skipped"
    # And a live window still gets its full wait rather than being cut short.
    assert not truth(RUN, f'alive() {{ return 1; }}; wait_for_heartbeat "{beat}" 1 alive')


def test_both_scripts_agree_on_the_sync_interval():
    """The plan prints a cadence; the loop runs one. A drifted copy documents the wrong run."""
    assert call(RUN, "printf %s $INTERVAL") == call(SYNC, "printf %s $INTERVAL")


# --- "Started and found nothing" is not "never started" (#108 item 2) ---------


@pytest.fixture
def fake_rclone(tmp_path):
    """An rclone that is new enough and always succeeds.

    The sync loop's own logic is what is under test here, not rclone's — and a real one would
    need a remote. It answers the version probe with something above the floor so the script
    takes its normal path rather than the below-floor one, which has its own tests above.
    """
    binary = tmp_path / "bin" / "rclone"
    binary.parent.mkdir(parents=True)
    binary.write_text('#!/bin/sh\ncase "$1" in version) echo "rclone v1.69.1";; esac\nexit 0\n')
    binary.chmod(0o755)
    env = dict(os.environ, PATH=f"{binary.parent}:{os.environ['PATH']}")
    return env


def run_the_loop_until_it_beats(env, ckpt_dir: Path, seconds: float = 20) -> Path:
    """Start the sync loop, wait for its heartbeat, stop it. Returns the heartbeat path.

    Stopped by signalling the whole process group, not just bash: the loop spends most of its
    life inside `sleep`, and bash defers a trap until the foreground command it is running
    returns — so a bare SIGTERM to the script alone would be held for a whole interval.
    """
    beat = ckpt_dir / ".sync_heartbeat"
    loop = subprocess.Popen(
        [
            str(SYNC),
            "--checkpoint-dir", str(ckpt_dir),
            "--remote", "fake:demae-checkpoints/r1/",
            "--interval", "1",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline and not beat.exists():
            time.sleep(0.1)
    finally:
        os.killpg(os.getpgid(loop.pid), signal.SIGTERM)
        loop.wait(timeout=20)
    return beat


def test_the_loop_says_it_is_alive_even_with_nothing_to_ship(fake_rclone, tmp_path):
    """The distinction the whole item turns on.

    `sync_checkpoints.sh` could not tell "started and found nothing" from "never started": both
    are an empty bucket and no local evidence. An empty run directory early in a run is the
    *normal* state — the first checkpoint is an epoch away — so silence there cannot be read as
    health. The loop now writes a heartbeat the moment it is up, before its first pass, and
    again after every one.
    """
    ckpt = tmp_path / "runs" / "R1"          # deliberately not created: this is minute zero
    beat = run_the_loop_until_it_beats(fake_rclone, ckpt)
    assert beat.exists(), "the loop ran and left nothing an assert could see"
    line = beat.read_text().strip()
    assert "pass=" in line
    assert "fake:demae-checkpoints/r1/" in line, "the heartbeat names what it is shipping to"
    assert line.split()[0].endswith("Z"), f"no UTC stamp to age: {line!r}"


def test_a_window_started_from_the_wrong_directory_still_finds_everything(
    fake_rclone, tmp_path
):
    """The ticket itself, driven rather than planned — no tmux, but the same command.

    `--exec sync` is what a tmux window runs, and the whole failure was that the window's cwd
    was one directory up from the repo. So this starts it from `/`, through a checkout laid out
    the way `push_code.sh` leaves one, and asserts the loop came up anyway. The dry-run tests
    above check what would be *printed*; this checks what a window actually does with it.

    A separate root rather than this checkout because the sync loop writes its heartbeat into
    `<root>/runs/<run-id>`, and a test has no business writing into the working tree.
    """
    root = tmp_path / "workspace" / "DEMAE"
    (root / "scripts").mkdir(parents=True)
    for script in (RUN, SYNC):
        (root / "scripts" / script.name).symlink_to(script)

    window = subprocess.Popen(
        [
            str(root / "scripts" / "run.sh"),
            "--exec", "sync",
            "--run-id", "R1",
            "--remote", "fake:demae-checkpoints/r1/",
            "--interval", "1",
        ],
        cwd="/",                          # the pane the image's rc actually hands you
        env=fake_rclone,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    beat = root / "runs" / "R1" / ".sync_heartbeat"
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and not beat.exists():
            time.sleep(0.1)
    finally:
        os.killpg(os.getpgid(window.pid), signal.SIGTERM)
        output = window.communicate(timeout=20)[0]

    assert beat.exists(), f"the window found nothing to run from `/`:\n{output}"
    assert str(root / "runs" / "R1") in output, "it shipped from somewhere other than the root"


def test_the_credential_probe_leaves_no_heartbeat_behind(fake_rclone, tmp_path):
    """`--once` must not be able to satisfy the assert that the *loop* started.

    This is the exact confusion #108 describes: R1 ran `--once` successfully, and the loop that
    then failed to start produced no signal at all. If the probe wrote the same file, a launcher
    checking for it would be reassured by the run before the one it cares about.
    """
    ckpt = tmp_path / "runs" / "R1"
    ckpt.mkdir(parents=True)
    done = subprocess.run(
        [
            str(SYNC),
            "--checkpoint-dir", str(ckpt),
            "--remote", "fake:demae-checkpoints/r1/",
            "--once",
        ],
        env=fake_rclone,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert not (ckpt / ".sync_heartbeat").exists()


def test_the_heartbeat_is_never_shipped_to_the_bucket():
    """It is local evidence about this box, not an artifact. The include pattern is what says so."""
    assert call(SYNC, "printf %s \"$PATTERN\"") == "checkpoint_*.pt"
    assert call(SYNC, 'heartbeat_line 2026-08-04T00:00:00Z 3 2 0 r:/').startswith("2026-")


# --- The document, held to the same standard as the scripts -------------------


def section(name: str) -> str:
    """One `## ` section of the runbook, by its heading prefix."""
    text = RUNBOOK.read_text()
    start = text.index(f"\n## {name}")
    end = text.find("\n## ", start + 1)
    return text[start : end if end != -1 else len(text)]


def fenced_commands(markdown: str) -> list[str]:
    """Every line inside a ```bash fence."""
    lines, inside, out = markdown.splitlines(), False, []
    for line in lines:
        if line.startswith("```"):
            inside = line.startswith("```bash")
            continue
        if inside and line.strip():
            out.append(line)
    return out


def test_section_seven_never_invokes_a_script_by_a_relative_path():
    """The documented command is the artifact that failed, so it is asserted on.

    Not a substitute for the script — the point of `run.sh` is that the paragraph stops being
    load-bearing. But §7 still shows a fallback for a box where something has gone wrong, and a
    fallback that reintroduces the bug is worse than no fallback.
    """
    offenders = [
        line for line in fenced_commands(section("7."))
        if "scripts/" in line and "/workspace/DEMAE/scripts/" not in line
    ]
    assert not offenders, "relative script path(s) in §7:\n" + "\n".join(offenders)


# Each preset flag sets its own `run_id`, so a documented example carrying one has to name that
# arm. Matched as a whole token rather than a substring: `--vit-base-continual` *contains*
# `--vit-base`, so the substring version of this test read R4's line as an R5 example and
# demanded the wrong id — a check that fires on the correct command is worse than no check.
PRESET_ARMS = {"--vit-base": "R5", "--vit-base-continual": "R4"}


def test_the_runbook_only_shows_a_preset_under_its_own_run_id():
    for line in fenced_commands(RUNBOOK.read_text()):
        tokens = line.split()
        for flag, arm in PRESET_ARMS.items():
            if flag in tokens:
                assert arm in tokens, f"{flag} outside its arm ({arm}): {line!r}"


# --- Destroying the box, and proving it (#111) --------------------------------
#
# The bug this section exists for did not fail loudly. `teardown_box.sh --destroy --yes` printed
#
#     Aborted.
#     instance 46808051 destroyed. ./teardown-46808051 holds everything that was unique to it.
#
# on a box that was still running and still billing. `--yes` skips *this script's* confirmation;
# the vastai CLI has a second one of its own, which with no tty reads EOF, aborts, and exits
# **0**. Nothing checked the status, so the success line printed unconditionally.
#
# Every other failure in that script refuses to proceed and says nothing was destroyed. This one
# claimed the irreversible action had succeeded when it had not — so the natural next move, close
# the terminal and stop watching the spend, was exactly wrong.

TEARDOWN = SCRIPTS / "teardown_box.sh"

VASTAI_STUB = """#!/usr/bin/env bash
# Stands in for vastai 1.5.1. STUB_MODE picks the behaviour; STUB_STATE is the instance list.
sub="$1 $2"; shift 2
case "$sub" in
  "destroy instance")
    id=""; for a in "$@"; do [[ "$a" == -* ]] || id="$a"; done
    case "${STUB_MODE:-abort}" in
      abort) echo "Are you sure you want to destroy instance $id? [y/N]"; echo "Aborted."; exit 0 ;;
      real)  echo "destroying instance $id"; rm -f "$STUB_STATE"; exit 0 ;;
      rc1)   echo "boom" >&2; exit 1 ;;
    esac ;;
  "show instances")
    [[ "${STUB_LIST_BROKEN:-0}" == 1 ]] && exit 1
    [[ -f "$STUB_STATE" ]] && cat "$STUB_STATE"; exit 0 ;;
esac
exit 9
"""


@pytest.fixture
def vastai(tmp_path):
    """A stubbed `vastai` on PATH, plus the instance list it answers from.

    Returns a callable: ``vastai(mode, listed=[...], list_broken=False)`` sources
    ``teardown_box.sh`` in library mode, runs ``destroy_instance``, and hands back
    ``(return_code, output)``.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "vastai"
    stub.write_text(VASTAI_STUB)
    stub.chmod(0o755)
    state = tmp_path / "instances.txt"

    def drive(mode: str, listed=("46808051",), list_broken: bool = False, instance="46808051"):
        state.write_text("".join(f"{i}\n" for i in listed))
        env = {
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "STUB_MODE": mode,
            "STUB_STATE": str(state),
            "STUB_LIST_BROKEN": "1" if list_broken else "0",
            "DEMAE_LIB": "1",
        }
        done = subprocess.run(
            ["bash", "-c", f'. "{TEARDOWN}"; rc=0; destroy_instance "$(vastai_bin)" {instance} || rc=$?; exit $rc'],
            capture_output=True, text=True, env=env,
        )
        return done.returncode, done.stdout + done.stderr

    return drive


def test_a_destroy_that_aborts_and_still_exits_zero_is_not_reported_as_destroyed(vastai):
    """The exact R2/R3 failure, reproduced: the CLI aborts, exits 0, the instance survives.

    This is the regression. `-y` now covers the CLI's own prompt so this path should not arise
    again — but a flag is a promise about one version of one tool, and what is at stake is an
    instance billing by the hour. So the verdict does not come from the exit status at all.
    """
    rc, out = vastai("abort")
    assert rc == 1, f"an aborted destroy reported success:\n{out}"
    assert "Aborted." in out  # the stub really did take the failing path


def test_a_verified_destroy_is_the_only_thing_that_returns_clean(vastai):
    rc, out = vastai("real")
    assert rc == 0, out


def test_a_nonzero_destroy_is_not_reported_as_destroyed(vastai):
    rc, out = vastai("rc1")
    assert rc == 1
    assert "exited 1" in out


def test_an_unreadable_instance_list_refuses_to_claim_either_way(vastai):
    """Return 3 — attempted, unverifiable — which is neither success nor a claim of failure.

    The instance list is the evidence, so when it cannot be read there is no evidence. Reporting
    "destroyed" here would be the original bug with a different cause, and reporting "still
    running" would send the operator hunting for an instance that may well be gone.
    """
    rc, out = vastai("real", list_broken=True)
    assert rc == 3, out
    assert "cannot be proven" in out


def test_absence_from_the_list_only_counts_if_the_list_could_see_it_first(vastai):
    """A listing that cannot see the instance *before* the destroy proves nothing after it.

    Without this, any reason the id is missing — wrong API key, a changed output format, an
    account holding the instance under another key — reads as "gone" and prints the success line
    this whole section exists to withhold.
    """
    rc, out = vastai("abort", listed=("99999999",))
    assert rc == 0, out
    assert "already gone" in out


def test_the_instance_id_is_matched_whole_line_not_as_a_substring(vastai):
    """Instance 4680 must never be read as 46808051 — that would report a live box as destroyed."""
    rc, _ = vastai("abort", listed=("4680",), instance="46808051")
    assert rc == 0  # 46808051 genuinely is not in that list

    rc, out = vastai("abort", listed=("46808051",), instance="4680")
    assert rc == 0, out  # ...and 4680 is not in *that* one


def test_the_success_line_is_unreachable_unless_the_destroy_verified(vastai):
    """The property that actually failed, asserted on the script's text.

    The behavioural tests above drive `destroy_instance`; this one pins the *caller*, because the
    bug was never inside the destroy — it was that the success line sat outside it, on a path
    reached whatever happened. It must now live under the branch for return code 0 alone.
    """
    text = TEARDOWN.read_text()
    body = text[text.index("destroy_instance \"$VASTAI\""):]
    said_destroyed = body.index("destroyed. $HAUL_DIR")
    zero_branch = body.index("    0)")
    next_branch = body.index("    1)")
    assert zero_branch < said_destroyed < next_branch, (
        "the 'destroyed.' line is not inside the verified-success branch"
    )


def test_the_cli_is_found_where_the_runbook_installs_it(tmp_path):
    """`uv tool install vastai` lands it in ~/.local/bin, which a non-interactive shell lacks.

    R2/R3's teardown died with "--destroy needs the vastai CLI on PATH" on a laptop that had it
    installed — after the deep verify had already re-downloaded ~900 MB.
    """
    home = tmp_path / "home"
    (home / ".local" / "bin").mkdir(parents=True)
    hidden = home / ".local" / "bin" / "vastai"
    hidden.write_text("#!/usr/bin/env bash\nexit 0\n")
    hidden.chmod(0o755)

    done = subprocess.run(
        ["bash", "-c", f'. "{TEARDOWN}"; vastai_bin'],
        capture_output=True, text=True,
        env={**os.environ, "HOME": str(home), "PATH": "/usr/bin:/bin", "DEMAE_LIB": "1"},
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == str(hidden)


def test_the_cli_precondition_is_checked_before_the_expensive_verification(tmp_path):
    """One line, knowable at startup, used to be checked after a ~900 MB re-download."""
    text = TEARDOWN.read_text()
    precondition = text.index("--destroy needs the vastai CLI")
    deep_verify = text.index("Deep verify — re-read the bytes")
    assert precondition < deep_verify, "the CLI check still runs after the deep verify"


# --- §11's RAM floor is pretraining's, and Task 1 says so ----------------------


def test_the_ram_floor_defaults_to_section_11s_pretraining_number():
    """The default is unchanged by #162 — a benchmark flag must not relax the normal path."""
    assert int(call(PROVISION, "printf %s $MIN_RAM_GB")) == 60


def test_the_ram_floor_is_a_flag_because_the_workload_sets_it():
    """`--min-ram-gb`, on `--max-corpus-min`'s own rule (#162).

    §11 wants >=64 GB so the **whole 47 GB corpus** page-caches, because pretraining streams all
    500k tiles every epoch. A Task 1 benchmark run reads §3.1's draws — 20,000 head-train + 2,000
    val + 5,000 test — which is 27,000 tiles at 98 KiB, a 2.71 GB working set out of a memmap the
    kernel pages lazily. A 30 GB box is refused for pretraining and ample for this, and the floor
    is therefore a property of the workload rather than of the plan.

    Asserted as *parseable and honoured*, both ways, because a flag that parsed and did not move
    the check would read exactly like this test passing.
    """
    assert "--min-ram-gb" in PROVISION.read_text()
    # Parsed off the command line at all.
    assert (
        call(PROVISION, 'set -- --min-ram-gb 24; while [[ $# -gt 0 ]]; do case "$1" in '
                        '--min-ram-gb) MIN_RAM_GB="$2"; shift 2 ;; *) shift ;; esac; done; '
                        'printf %s "$MIN_RAM_GB"')
        == "24"
    )


@pytest.mark.parametrize(
    "ram,floor,passes",
    [
        (30, 60, False),  # this benchmark's box under §11's pretraining floor
        (30, 24, True),   # ...and over a floor set for the 2.71 GB Task 1 working set
        (30, 0, True),    # 0 waives it entirely
        (64, 60, True),   # §11's own box, unaffected
    ],
)
def test_the_floor_admits_and_refuses_the_boxes_it_should(ram, floor, passes):
    assert truth(PROVISION, f'[[ {floor} -eq 0 || {ram} -ge {floor} ]]') is passes


# --- The plan is an input on a Task 1 box, not documentation -------------------


def test_the_push_ships_the_plan_because_scoring_reads_it():
    """#162. `prereg.read_bar` re-checks §6 against the plan text every time it opens the bar.

    "The box trains, it does not get read on" is Phase 1's rule and stays true of `docs/` as a
    whole — pretraining never opens the plan. Task 1's scoring path does, so this one file is a
    runtime input, and a box without it fails in `run_pass` *after* the training it just paid
    for. Asserted on the script rather than trusted, because the failure costs a whole seed.
    """
    text = PUSH.read_text()
    assert "docs/research/task1-void-fill-plan.md" in text
    # With --relative, or it lands at the destination root where nothing looks for it.
    assert "rsync -azR" in text
    # And the wider docs/ tree is still not shipped.
    assert "PAYLOAD=(src tests artifacts scripts" in text


def test_the_plan_the_push_ships_is_the_one_the_config_reads():
    """One path, asserted against the config default rather than typed twice."""
    from benchmark.config import Paths

    assert Paths().plan_path in PUSH.read_text()


def test_the_dry_run_names_both_transfers():
    """A dry run that hid half of what a real push does would be the wrong kind of preview."""
    done = subprocess.run(
        ["bash", str(PUSH), "--host", "nowhere.invalid", "--dry-run"],
        capture_output=True, text=True, cwd=REPO,
    )
    assert done.returncode == 0, done.stderr
    assert "task1-void-fill-plan.md" in done.stdout


# --- The #165 sweep runner's cell list, held equal to the code's ----------------------------
# Not arithmetic like the two above, but the same class of defect: a constant copied into a
# second file, where the copy can go stale without anything failing. `RCLONE_MIN` is the
# precedent — carried in two scripts and pinned equal here because neither can source the other
# at the moment it needs the answer. This one is a shell array against a Python tuple.


SWEEP = SCRIPTS / "p165_sweep.sh"


def test_the_sweep_runner_runs_exactly_the_arms_the_code_registers():
    """Otherwise the script is a fourth authority on §7's shape, beside §7, §4.5 and the code.

    The script cannot import `benchmark.arms`, so its loop is a literal — which is how §4.5
    drifted from §7 in the first place (#129). Pinned rather than refactored: a bash script that
    shelled out to Python to learn which arms to run would fail in a way this test cannot see.
    """
    from benchmark import arms

    body = SWEEP.read_text()
    loop = next(line for line in body.splitlines() if line.strip().startswith("for arm in "))
    named = loop.strip().removeprefix("for arm in ").removesuffix("; do").split()
    assert tuple(named) == arms.LABEL_SWEEP_ARMS


def test_the_sweep_runner_runs_exactly_the_rungs_and_seeds_the_code_registers():
    """The other two axes of §7's `3 cells x 2 seeds x 2 sizes, plus A7`."""
    from benchmark.config import LABEL_EFFICIENCY_SIZES, Head, SIZE_TILES

    body = SWEEP.read_text()
    sizes = next(line for line in body.splitlines() if line.strip().startswith("for size in "))
    named = sizes.strip().removeprefix("for size in ").removesuffix("; do").split()
    # Every rung but the headline — the sweep is the sub-headline half of the ladder.
    assert tuple(named) == tuple(s for s in LABEL_EFFICIENCY_SIZES if s != "20k")
    assert SIZE_TILES["20k"] == max(SIZE_TILES.values())  # which is what makes 20k the headline

    seeds = next(line for line in body.splitlines() if line.strip().startswith("for seed in "))
    named = seeds.strip().removeprefix("for seed in ").removesuffix("; do").split()
    assert [int(s) for s in named] == list(range(Head().label_sweep_seeds))
