#!/usr/bin/env bash
# Start a run on the box — both halves of it, from wherever you happen to be standing:
#
#     /workspace/DEMAE/scripts/run.sh --run-id R1
#     /workspace/DEMAE/scripts/run.sh --run-id R2 -- --set masking.ratio=0.9
#     /workspace/DEMAE/scripts/run.sh --run-id R5 -- --vit-base
#
# Why this is a script and not the two commands §7 used to print. Both of them were correct, and
# neither ran (#108). The runbook said:
#
#     cd /workspace/DEMAE && tmux new -s r1
#     uv run python -m pretrain.launch ...
#     scripts/sync_checkpoints.sh --checkpoint-dir /workspace/DEMAE/runs/R1 ...
#
# and the vastai image's shell rc `cd`s every new pane to `/workspace`, overriding even
# `tmux new-session -c /workspace/DEMAE` — confirmed on R1's box, where
# `tmux display-message -p '#{pane_current_path}'` answered `/workspace` for a pane created with
# `-c /workspace/DEMAE`. Typed by hand into a pane that was already attached it happened to work.
# Driven any other way — `send-keys`, a detached session, a script, an agent — the two halves
# broke differently, and that asymmetry is the whole point:
#
#   training  failed loudly. `uv run` found no pyproject.toml, fell back to the image's
#             interpreter, and died with `ModuleNotFoundError: No module named 'pretrain'`.
#   the sync  failed **silently**. `scripts/sync_checkpoints.sh: No such file or directory` — one
#             line, in a window nobody is watching, with no retry and no rising failure count,
#             because the loop that prints those never started.
#
# R1 escaped it only because the loud half failed at the same instant and forced a look. Had the
# cwd been right for `uv run` and wrong for the relative path — or had the two been started
# separately, which is what §7 told you to do — R1 would have trained 7.2 hours with no
# checkpoint ever leaving the box.
#
# So the fix is a committed script rather than a corrected paragraph. This class of bug is
# "documented correctly, executed differently", and a paragraph cannot be tested; every property
# that failed here is asserted in tests/test_ops_scripts.py against this file. What it fixes:
#
# 1. **Every path is absolute**, resolved from this file's own location rather than from the
#    caller's cwd. That is a fact about the checkout; a cwd is a fact about somebody's shell.
# 2. **The sync loop starts first, and proves it is up before the GPU spends anything.** §7's
#    ordering was the other way round, so the cheap half's failure arrived after the expensive
#    half was already running. The loop writes a heartbeat as soon as it is alive (see
#    `sync_checkpoints.sh --heartbeat`); this refuses to start training until it appears.
# 3. **One run id, threaded everywhere** — the config's `run_id`, the checkpoint directory, the
#    bucket prefix, the training log, the tmux session. §7 named the run four times across two
#    commands and got one of them wrong: it passed `--vit-base` in the R1/R2/R3 position, and
#    `RunConfig.vit_base()` sets `run_id="R5"`, so following it literally for R1 would have
#    trained a ViT-B into `runs/R5` while the sync window shipped `runs/R1`.
# 4. **The trainer's output lands where `watch_run.sh` looks for it** —
#    `<root>/<run-id lowercased>_train.log`. That default existed; nothing wrote the file.
#
# What this deliberately does NOT do: rent, provision, or destroy. Push and provision first
# (§5), and read the provision footer (§6), before any of this is worth running.

set -euo pipefail

# This file's checkout, not the caller's cwd. The one line the whole ticket is about.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SELF="$REPO/scripts/run.sh"

RUN_ID=""
REMOTE=""
SESSION=""
# sync_checkpoints.sh's own default, restated so the plan can print it — and held equal to it by
# tests/test_ops_scripts.py, like `RCLONE_MIN` is across provisioning and the sync loop. A copy
# that drifted would make the plan describe a cadence the loop is not running.
INTERVAL=1800
HEARTBEAT_WAIT=120       # seconds to wait for the sync loop to say it is alive
SKIP_PROBE=0
DRY_RUN=0
EXEC_MODE=""
EXTRA=()

# --- The pure parts, and the ones two scripts have to agree on --------------------------------

lower() { printf '%s' "$1" | tr '[:upper:]' '[:lower:]'; }

# A run id names a directory and a bucket prefix, so it is checked before either is built. R0-R5
# are §12's schedule and any other `R<n>` is a smoke run; the pattern is deliberately narrow
# rather than "anything without a slash".
valid_run_id() { [[ "$1" =~ ^R[0-9]+$ ]]; }

# Whole numbers, because both of these are counted against. `--heartbeat-wait abc` would read as
# zero inside `(( ))` and wait no time at all, which passes the one gate this whole file exists
# to hold — a check that silently becomes vacuous is worse than no check.
whole_number() { [[ "$1" =~ ^[0-9]+$ ]]; }

# Launcher flags that would move where checkpoints are written without moving where the sync loop
# is looking. Refused rather than accommodated: `--checkpoint-dir` in the passthrough gives a run
# whose trainer is healthy, whose heartbeat is present, and whose checkpoints never leave the box
# — which is, word for word, the outcome #108 exists to prevent. `--no-checkpoints` is the same
# thing with nothing to ship at all, and `run_id=` in any spelling moves the config's own idea of
# the run away from the directories named after it.
conflicts_with_run_id() {
    case "$1" in
        --checkpoint-dir|--checkpoint-dir=*) return 0 ;;
        --no-checkpoints)                    return 0 ;;
        *run_id=*)                           return 0 ;;   # `run_id=R2`, `--set=run_id=R2`
        *) return 1 ;;
    esac
}

# Where the trainer's output goes, and where watch_run.sh reads it from. Carried in both files
# rather than shared, exactly as `RCLONE_MIN` is between provisioning and the sync loop — this
# one runs on the box, that one on the laptop, and neither can source the other at the moment it
# needs the answer. tests/test_ops_scripts.py holds the two copies in step.
train_log_path() { printf '%s/%s_train.log\n' "$1" "$(lower "$2")"; }

checkpoint_dir_for() { printf '%s/runs/%s\n' "$1" "$2"; }
heartbeat_for()      { printf '%s/.sync_heartbeat\n' "$(checkpoint_dir_for "$1" "$2")"; }
remote_for()         { printf 'bucket:demae-checkpoints/%s/\n' "$(lower "$1")"; }

# The gate that turns #108's silent half into a failure at the keyboard. Absent after the wait
# means the loop never came up; present means it is running and has said so once, whether or not
# there was anything to ship. Polled rather than slept through, so the normal case costs a
# second.
#
# The optional third argument is a command that answers "the window it should be running in is
# dead". The commonest failure here — the one #108 actually produced — kills the loop in
# milliseconds, and waiting out a two-minute deadline to report something already knowable is how
# an operator learns to start ignoring the wait.
wait_for_heartbeat() {
    local path="$1" seconds="$2" dead="${3:-}" waited=0
    while (( waited < seconds )); do
        [[ -f "$path" ]] && return 0
        if [[ -n "$dead" ]] && $dead; then
            return 1
        fi
        sleep 1
        waited=$(( waited + 1 ))
    done
    [[ -f "$path" ]]
}

# Sourced by tests/test_ops_scripts.py, which drives the functions above with no box and no tmux.
[[ -n "${DEMAE_LIB:-}" ]] && return 0

usage() {
    cat <<'EOF'
run.sh — start a run on the box: the sync loop, then training, in one tmux session.

  --run-id ID          which run this is (R1…R5). Required — it is the config's run_id, the
                       checkpoint directory, the bucket prefix, the log name and the session
  --remote REMOTE      rclone destination (default: bucket:demae-checkpoints/<run-id lowered>/)
  --session NAME       tmux session name (default: the run id, lowercased)
  --interval SECONDS   sync interval (default: 1800)
  --heartbeat-wait N   seconds to wait for the sync loop to prove it started (default: 120)
  --skip-probe         skip the `--once` credential probe. Only when it has just been run
  --dry-run            print the plan — every command, in order — and touch nothing
  -- ARGS...           everything after -- goes to `python -m pretrain.launch`
  -h, --help           this

Run it by absolute path. That is not fussiness: the image's shell rc puts every new pane in
/workspace, which is not this repo, and #108 is what that cost.
EOF
}

# Defined before argv is read, so a bad flag is refused the same way everything else here is
# rather than through a bare `echo` — matching push_code.sh and provision_box.sh.
step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
ok()   { printf '   ok    %s\n' "$*"; }
warn() { printf '   \033[33mwarn\033[0m  %s\n' "$*" >&2; }
die()  { printf '\n\033[31mSTOPPED: %s\033[0m\n' "$*" >&2; exit 2; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-id) RUN_ID="${2:-}"; shift 2 ;;
        --remote) REMOTE="$2"; shift 2 ;;
        --session) SESSION="$2"; shift 2 ;;
        --interval) INTERVAL="$2"; shift 2 ;;
        --heartbeat-wait) HEARTBEAT_WAIT="$2"; shift 2 ;;
        --skip-probe) SKIP_PROBE=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        # Internal: what the two tmux windows run. Spelled as a flag rather than a second script
        # so there is one file to ship, one place the commands are written, and no way for a
        # window to be started with a different idea of the run than the plan printed.
        --exec) EXEC_MODE="$2"; shift 2 ;;
        --) shift; EXTRA=("$@"); break ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -n "$RUN_ID" ]] || { echo "--run-id is required (R1, R2, … — see #108)" >&2; usage >&2; exit 2; }
valid_run_id "$RUN_ID" || die "--run-id '$RUN_ID' is not a run id (R1…R5 are the schedule's;
        any other R<n> is a smoke run). It names a directory and a bucket prefix, so it is
        checked before either is built."
whole_number "$INTERVAL"       || die "--interval '$INTERVAL' is not a whole number of seconds"
whole_number "$HEARTBEAT_WAIT" || die "--heartbeat-wait '$HEARTBEAT_WAIT' is not a whole number
        of seconds. It would read as zero, and a gate that waits no time passes everything."

# One pass over the passthrough args: what must be refused, and what is merely worth saying.
for arg in ${EXTRA+"${EXTRA[@]}"}; do
    if conflicts_with_run_id "$arg"; then
        die "'$arg' after -- would point the trainer somewhere this session is not watching.
        The sync loop ships $(checkpoint_dir_for "$REPO" "$RUN_ID") and nothing else, so a run
        started this way trains healthily, writes a heartbeat, and never gets a checkpoint off
        the box — #108's outcome, reached through the flag meant to prevent it. Use --run-id,
        which moves the config, the directory, the bucket prefix and the log together."
    fi
    # `--vit-base` is R5's preset and sets `run_id="R5"` itself (config.py:753). Ours is applied
    # last so the directories and the config still agree — but it is said, because the silent
    # version of this is a run whose recipe and whose label disagree, which is #108 item 3.
    if [[ "$arg" == "--vit-base" && "$RUN_ID" != "R5" ]]; then
        warn "--vit-base is the R5 release preset and sets run_id=R5 on its own; this run is
        $RUN_ID, so its config will carry $RUN_ID with R5's shape. Intended?"
    fi
    # Same shape, one arm along: `--vit-base-continual` sets `run_id="R4"` (config.py:709). It
    # also carries §8's warm start and `pos_encoding=sincos2d_rowfirst`, so wearing another run's
    # label is worse here than for --vit-base: this is the only arm whose weights do not start
    # from scratch, and the label is what the readout compares by.
    if [[ "$arg" == "--vit-base-continual" && "$RUN_ID" != "R4" ]]; then
        warn "--vit-base-continual is R4's preset and sets run_id=R4 on its own; this run is
        $RUN_ID, so its config will carry $RUN_ID with a warm-started ViT-B. Intended?"
    fi
done

CKPT_DIR="$(checkpoint_dir_for "$REPO" "$RUN_ID")"
HEARTBEAT="$(heartbeat_for "$REPO" "$RUN_ID")"
LOG="$(train_log_path "$REPO" "$RUN_ID")"
[[ -n "$REMOTE" ]]  || REMOTE="$(remote_for "$RUN_ID")"
[[ -n "$SESSION" ]] || SESSION="$(lower "$RUN_ID")"

# --- The two commands, written once ------------------------------------------------------------
# Both windows and the dry-run plan read these same arrays. A plan assembled separately from the
# execution path would be this ticket again, one layer in: correct on the page, different in the
# pane.

# `monitoring.wandb_mode` defaults to `offline` (config.py:600), which is right for a laptop and
# for the test suite and wrong for every run this script starts: an offline run directory lives
# only on a box that is going to be destroyed, and has to be hauled and `wandb sync`ed at
# teardown (§10). §7 used to carry the override by hand in the documented command, which is one
# more thing to forget per run and quiet when forgotten. Ours goes first, so anything the
# operator passes after `--` is applied later and wins.
TRAIN_CMD=(
    uv run python -m pretrain.launch
    --set "run_id=$RUN_ID"
    --set "monitoring.wandb_mode=online"
    ${EXTRA+"${EXTRA[@]}"}
)
SYNC_CMD=(
    "$REPO/scripts/sync_checkpoints.sh"
    --checkpoint-dir "$CKPT_DIR"
    --remote "$REMOTE"
    --interval "$INTERVAL"
)
# Spelled out rather than sliced off SYNC_CMD's front: a positional slice reads correctly today
# and silently ships the wrong flags the first time that array is reordered.
PROBE_CMD=(
    "$REPO/scripts/sync_checkpoints.sh"
    --checkpoint-dir "$CKPT_DIR"
    --remote "$REMOTE"
    --once
)

# What a tmux window runs: this file, by absolute path, which `cd`s before anything else. Belt
# and braces against the rc — tmux's own `-c` was the thing that did not hold.
window_command() {
    local mode="$1"
    printf '%q ' "$SELF" --exec "$mode" --run-id "$RUN_ID" --remote "$REMOTE" \
        --interval "$INTERVAL" ${EXTRA+"--"} ${EXTRA+"${EXTRA[@]}"}
}

# --- What a window does ------------------------------------------------------------------------

if [[ -n "$EXEC_MODE" ]]; then
    cd "$REPO"                       # the entire content of interest, as R1's hand-written pair
    case "$EXEC_MODE" in
        sync)  exec "${SYNC_CMD[@]}" ;;
        train)
            # Tee'd rather than redirected: the pane still shows the run, and watch_run.sh gets
            # the file it has always defaulted to. `pipefail` (set at the top of this file) is
            # what keeps a dead trainer from being laundered into tee's exit status — with
            # `set -e` it is also what ends this process, so the pipeline's status is the
            # window's without a line here to pass it on.
            "${TRAIN_CMD[@]}" 2>&1 | tee -a "$LOG"
            ;;
        *) die "--exec takes 'train' or 'sync', not '$EXEC_MODE'" ;;
    esac
fi

# --- The plan ----------------------------------------------------------------------------------

print_plan() {
    printf '   repo             %s\n' "$REPO"
    printf '   checkpoints      %s\n' "$CKPT_DIR"
    printf '   remote           %s\n' "$REMOTE"
    printf '   train log        %s\n' "$LOG"
    printf '   tmux session     %s\n' "$SESSION"
    printf '   heartbeat        %s\n' "$HEARTBEAT"
    printf '\n'
    printf '   1. probe   %s\n' "${PROBE_CMD[*]}"
    printf '   2. sync    tmux new-session -d -s %s -n sync\n' "$SESSION"
    printf '              %s\n' "${SYNC_CMD[*]}"
    printf '   3. wait    up to %ss for %s\n' "$HEARTBEAT_WAIT" "$HEARTBEAT"
    printf '   4. train   tmux new-window -t %s: -n train\n' "$SESSION"
    printf '              %s\n' "${TRAIN_CMD[*]}"
    printf '              2>&1 | tee -a %s\n' "$LOG"
}

if [[ "$DRY_RUN" -eq 1 ]]; then
    step "Plan for $RUN_ID — nothing below has been run"
    print_plan
    exit 0
fi

# --- Starting it -------------------------------------------------------------------------------

command -v tmux >/dev/null || die "no tmux on PATH — the vastai image ships one; this is not it"

step "Starting $RUN_ID"
print_plan

# `remain-on-exit` keeps a window's last words on screen after its command exits, instead of
# closing over them — a traceback, a CUDA OOM, or the `No such file or directory` this whole
# ticket is about.
#
# Set in the *same tmux invocation* that creates the window, not a call afterwards. tmux runs a
# command sequence to completion before it processes the child's exit, so this is the only
# ordering that holds for the failure that matters: the sync command that dies in milliseconds
# would otherwise take its window, its session, and the diagnosis with it before a second
# invocation could arrive — leaving the operator a refusal with nothing attached to it.
window_dead() { [[ "$(tmux list-panes -t "$1" -F '#{pane_dead}' 2>/dev/null)" == *1* ]]; }
sync_window_dead() { window_dead "$SESSION":sync; }

if tmux has-session -t "$SESSION" 2>/dev/null; then
    die "tmux session '$SESSION' already exists. Two trainers writing $CKPT_DIR would resume
        from each other's checkpoints. Attach with \`tmux attach -t $SESSION\`, or pick another
        --session if that one is something else."
fi

if [[ "$SKIP_PROBE" -eq 0 ]]; then
    step "Probing the remote before any GPU time (§7)"
    "${PROBE_CMD[@]}" || die "the credential probe failed — fix it before starting a run. It
        costs seconds now and 43 hours of unshipped checkpoints later."
fi

# Cleared first: a heartbeat from a previous run's loop would make the gate below vacuous, which
# is the same mistake as letting `--once` write one. The `.tmp` goes too — a loop killed mid-write
# leaves one behind, and it is the file the next write renames over.
rm -f "$HEARTBEAT" "$HEARTBEAT.tmp"

step "The sync loop, first — it is the half that fails quietly"
tmux new-session -d -s "$SESSION" -n sync -c "$REPO" "$(window_command sync)" \
    \; set-option -t "$SESSION":sync -w remain-on-exit on

# Until training is up, this script owns the session it just made. A Ctrl-C here would otherwise
# leave a detached sync loop shipping for a run that never started — and, since the loop only
# exits once it has *seen* training, one that would keep looping until the box is destroyed.
trap 'printf "\ninterrupted — killing session %s\n" "$SESSION" >&2;
      tmux kill-session -t "$SESSION" 2>/dev/null || true; exit 130' INT TERM

if ! wait_for_heartbeat "$HEARTBEAT" "$HEARTBEAT_WAIT" sync_window_dead; then
    printf '\n--- the sync window said: ------------------------------------------\n' >&2
    tmux capture-pane -p -t "$SESSION":sync >&2 2>/dev/null || true
    printf -- '--------------------------------------------------------------------\n' >&2
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    die "the sync loop wrote no heartbeat, so it is not running. Training has NOT been started —
        a 400-epoch run whose checkpoints never leave the box is #108's expensive outcome, and
        this is the check that exists to prevent it."
fi
ok "the sync loop is up — $(cat "$HEARTBEAT")"

step "Training"
tmux new-window -t "$SESSION": -n train -c "$REPO" "$(window_command train)" \
    \; set-option -t "$SESSION":train -w remain-on-exit on
trap - INT TERM     # the session is the run's now, not this script's
ok "started in session '$SESSION', window 'train'"

cat <<EOF

Watch it:            tmux attach -t $SESSION
From the laptop:     scripts/watch_run.sh --host demae-box --run-id $RUN_ID --once
                     scripts/watch_run.sh --host demae-box --run-id $RUN_ID

Read the launcher's first ten lines in the train window: they name the checkpoint directory,
both triggers and the retention rule. It has to be $CKPT_DIR — the sync loop is pointed there.
EOF
