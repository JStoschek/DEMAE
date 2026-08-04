#!/usr/bin/env bash
# Watch a run on a rented box from the **laptop**, and say — once, unambiguously — when it stops:
#
#     scripts/watch_run.sh --host demae-box --run-id R1
#
# One line per event on stdout, so it drives a notifier or a `| while read` loop. Silence means
# the run is healthy; every terminal state emits.
#
# Why this exists, and what it is a fix for. R1 (#82) finished at ~07:17 UTC and the box was not
# destroyed until ~14:53 — 7.6 idle hours, ~$3.05, more than R1's own $2.90 of GPU time. Nothing
# was lost, but the money was, and the cause was the watcher rather than the runbook: its
# liveness check was
#
#     ssh box 'pgrep -fc "pretrain.launch"'
#
# and `pgrep -f` matches against *full command lines*, including the command line of the very
# shell ssh spawned to run it — which contains the pattern. The check therefore returned at least
# 1 forever and its completion branch could never fire. It finally reported "training gone" only
# when the box stopped answering, long after a human had torn it down by hand.
#
# The bug is one character wide. The lesson is not: **a check whose failure mode is silence needs
# a positive test that it can fire at all.** A watcher that cannot observe completion looks
# exactly like a run that has not completed, for as long as you are willing to keep paying. So
# this script refuses to arm until it has proven, on the live box, that its own liveness check
# answers *both* ways — see `self_test` and §7.5 of the header below.
#
# The four states, kept distinct. The old watcher collapsed three of them into "process gone",
# which is why a destroyed box and a finished run produced the same event:
#
#   running      the training process is there. No event; this is the quiet case.
#   finished     process gone AND the log carries launch.py's completion line. Go and tear down.
#   died         process gone and no completion line — a crash, an OOM, or a kill. The
#                checkpoints are worth rescuing, and resume is a real option (§11).
#   unreachable  ssh itself failed, repeatedly. On vast this is the one that means *hurry*:
#                instances vanish permanently rather than pause, and an unreachable box may be
#                taking its local checkpoints with it. Never reported as "finished".
#
# `finished` is a positive signal, not an absence. `pretrain.launch` prints one summary line on
# clean completion (`<N> steps: loss <first> -> <last>`, launch.py:339); that string is what
# distinguishes a run that ended from a run that was killed, and no amount of process-table
# inspection can tell those apart.
#
# This runs on the laptop, like teardown_box.sh and for the same reason: a watcher living on the
# box cannot report the one event that matters most, which is the box going away.

set -euo pipefail

HOST=""
RUN_ID="R1"
ROOT="/workspace/DEMAE"
LOG=""
INTERVAL=300
MILESTONE_EVERY=50
UNREACHABLE_STRIKES=3     # consecutive ssh failures before calling it; one blip is not a death
PROCESS_MATCH="pretrain.launch"
ONCE=0
# Generous for R1-R4 (R1 took 7.2 h); R5 at 400-800 epochs on ViT-B needs an explicit larger
# value. Deliberately not derived from the config: a backstop computed from the same numbers the
# run trusts would fail in the same direction the run does.
MAX_HOURS=24

# --- The two pure functions, and the whole point of the file ----------------------------------

# The bracket trick, in the one place it is allowed to live. `pgrep -f 'pretrain.launch'` matches
# any command line *containing* that text — including ssh's own. Rewriting the first character as
# a one-character class leaves the regex matching exactly the same processes while making the
# pattern's own literal text un-matchable by itself: the shell's command line then holds
# `[p]retrain.launch`, in which no `p` is ever followed by `retrain`.
#
# Do not "simplify" this back to a plain pattern. That is the bug this file exists for.
probe_pattern() {
    local raw="$1"
    printf '[%s]%s' "${raw:0:1}" "${raw:1}"
}

# The backstop, and the only check here that observes nothing.
#
# Everything else in this file is a *measurement* — of a process table, of a log, of whether ssh
# answered — and every measurement can be wrong in the direction that costs money, because the
# expensive failure is always "looks like it is still running". #82's pattern bug was one way to
# get that; a lingering worker, a zombie, an `exec`-less wrapper whose command line outlives the
# trainer, or a box whose clock and log agree on something false are others I have not thought of.
#
# A deadline cannot be fooled by any of them, because it is not looking at the run. If the watcher
# has been up longer than a run could plausibly take, it says so and gets out of the way — even
# when every other signal insists all is well. That is the point: it is the check that still fires
# when I am wrong about the rest of this file.
overdue() {
    local elapsed_s="$1" max_h="$2"
    [[ "${max_h%.*}" -gt 0 ]] || return 1          # 0 disables
    (( elapsed_s > max_h * 3600 ))
}

# Turn what was observed into one of the four states. Split out so the decision is testable
# without a box, and so `unreachable` cannot be silently folded into `finished` by a later edit.
classify() {
    local reachable="$1" alive="$2" completed="$3"
    if [[ "$reachable" != "1" ]]; then printf 'unreachable\n'; return; fi
    if [[ "${alive:-0}" -gt 0 ]]; then printf 'running\n'; return; fi
    if [[ "$completed" == "1" ]]; then printf 'finished\n'; else printf 'died\n'; fi
}

# Sourced by tests/test_ops_scripts.py, which drives the two functions above with no box, no GPU
# and no network. Everything past here needs a host.
[[ -n "${DEMAE_LIB:-}" ]] && return 0

usage() {
    cat <<'EOF'
watch_run.sh — watch a run on a rented box and report, once, when it stops.

  --host ALIAS         ssh alias for the box (required)
  --run-id ID          run id; sets the default log path (default: R1)
  --root DIR           repo root on the box (default: /workspace/DEMAE)
  --log PATH           training log to read (default: <root>/<run-id lowercased>_train.log)
  --interval SECONDS   poll interval (default: 300)
  --milestone-every N  emit a progress line every N epochs (default: 50; 0 disables)
  --once               run the self-test and one poll, print the state, exit
  --max-hours N        say something regardless after N hours awake (default: 24; 0 disables)
  -h, --help           this

Exit status: 0 finished cleanly, 1 died, 2 unreachable, 3 overdue.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host) HOST="$2"; shift 2 ;;
        --run-id) RUN_ID="$2"; shift 2 ;;
        --root) ROOT="$2"; shift 2 ;;
        --log) LOG="$2"; shift 2 ;;
        --interval) INTERVAL="$2"; shift 2 ;;
        --milestone-every) MILESTONE_EVERY="$2"; shift 2 ;;
        --once) ONCE=1; shift ;;
        --max-hours) MAX_HOURS="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -n "$HOST" ]] || { echo "--host is required" >&2; usage >&2; exit 2; }
[[ -n "$LOG" ]] || LOG="$ROOT/$(printf '%s' "$RUN_ID" | tr '[:upper:]' '[:lower:]')_train.log"

die() { printf 'watch_run: %s\n' "$*" >&2; exit 2; }

# --- Talking to the box -------------------------------------------------------------------------

# `|| true` is load-bearing twice over. Under `set -e` a failing ssh inside a command
# substitution takes the whole script down — with ssh's bare 255 and none of the diagnosis
# below, which is the same die-in-silence failure this file exists to remove. It also makes
# unreachability *representable*: every remote command here prints something on success, so an
# empty answer means the box did not respond, and the caller can say so rather than guess.
remote() { ssh -o ConnectTimeout=20 -o BatchMode=yes "$HOST" "$1" 2>/dev/null || true; }

# How many processes match. Prints an integer, or nothing when ssh itself failed — which the
# caller must treat as "unknown", never as zero. Conflating those two is how a vanished box
# reports as a finished run.
count_alive() {
    local pattern; pattern="$(probe_pattern "$PROCESS_MATCH")"
    remote "pgrep -fc '$pattern' 2>/dev/null || echo 0"
}

completed_marker() {
    # launch.py:339's one-line summary. Anchored, so a mention in a traceback cannot pass for it.
    remote "grep -cE '^[0-9]+ steps: loss ' '$LOG' 2>/dev/null || echo 0"
}

last_epoch() { remote "grep -E '^epoch ' '$LOG' 2>/dev/null | tail -1 | awk '{print \$2}'"; }

# --- The self-test: prove the check can fire, before trusting its silence ------------------------
#
# Both directions, because the two ways to break this are mirror images and each is invisible in
# ordinary operation:
#
#   false-positive  the pattern matches something that is not the run — ssh's own command line,
#                   as in #82 — so the watcher never reports completion and the box bills until
#                   somebody notices. Caught by asking about a process that cannot exist.
#   false-negative  the check reports nothing alive when the run is right there, so the watcher
#                   fires "finished" on its first poll and a healthy run gets torn down. Caught
#                   by asking about the run we were pointed at.
#
# A watcher that fails either test is worse than no watcher, so this is fatal rather than a warn.
self_test() {
    local sentinel absent present
    sentinel="demae-watch-sentinel-no-such-process-$$"

    absent="$(remote "pgrep -fc '$(probe_pattern "$sentinel")' 2>/dev/null || echo 0")"
    [[ -n "$absent" ]] || die "self-test: cannot reach $HOST"
    [[ "$absent" == "0" ]] || die \
        "self-test: the liveness check reports $absent process(es) matching a sentinel that
   cannot exist. It is matching its own command line — the #82 bug — so it can never
   report completion. Refusing to arm."

    present="$(count_alive)"
    [[ -n "$present" ]] || die "self-test: cannot reach $HOST"
    [[ "${present:-0}" -gt 0 ]] || die \
        "self-test: no process matching '$PROCESS_MATCH' on $HOST. Either the run has not
   started — start it first, then arm this — or the match string is wrong. Refusing to
   arm, because a watcher that starts here would report 'finished' on its first poll."

    printf 'watch_run: armed on %s — %s alive, sentinel clean, log %s\n' \
        "$HOST" "$present" "$LOG"
}

# --- The loop -------------------------------------------------------------------------------

self_test

strikes=0
last_milestone=0

while true; do
    # Asked first, and independently of everything below, so a run that measures as healthy
    # forever still ends this watch. `SECONDS` is bash's own counter since the script started.
    if overdue "$SECONDS" "$MAX_HOURS"; then
        printf '%s OVERDUE — this watcher has been up %sh, past the %sh limit.\n' \
            "$RUN_ID" "$(( SECONDS / 3600 ))" "$MAX_HOURS"
        printf 'Every other signal says the run is still going, and one of us is wrong.\n'
        printf 'Check the box by hand — it is billing either way. This is not a completion.\n'
        remote "tail -5 '$LOG'"
        exit 3
    fi

    alive="$(count_alive)"
    reachable=1
    [[ -n "$alive" ]] || reachable=0

    # An ssh blip is not a dead box. Only a run of failures is, and until then the state is
    # unknown rather than terminal — so nothing is emitted and nothing is concluded.
    if [[ "$reachable" == "0" ]]; then
        strikes=$(( strikes + 1 ))
        if (( strikes < UNREACHABLE_STRIKES )); then
            sleep "$INTERVAL"; continue
        fi
    else
        strikes=0
    fi

    completed=0
    if [[ "$reachable" == "1" && "${alive:-0}" -eq 0 ]]; then
        [[ "$(completed_marker)" != "0" ]] && completed=1
    fi

    state="$(classify "$reachable" "${alive:-0}" "$completed")"

    case "$state" in
        running)
            if (( MILESTONE_EVERY > 0 )); then
                ep="$(last_epoch)"
                if [[ "$ep" =~ ^[0-9]+$ ]] && (( ep >= last_milestone + MILESTONE_EVERY )); then
                    last_milestone=$(( (ep / MILESTONE_EVERY) * MILESTONE_EVERY ))
                    printf '%s epoch %s\n' "$RUN_ID" "$ep"
                    remote "grep -E '^epoch ' '$LOG' | tail -1"
                    remote "grep -E '^probe epoch' '$LOG' | tail -1"
                fi
            fi
            [[ "$ONCE" == "1" ]] && { printf 'running\n'; exit 0; }
            ;;
        finished)
            printf '%s FINISHED cleanly — tear down the box now (it is billing).\n' "$RUN_ID"
            remote "tail -6 '$LOG'"
            remote "grep -E '^probe epoch' '$LOG' | tail -1"
            exit 0
            ;;
        died)
            printf '%s STOPPED WITHOUT COMPLETING — no summary line in the log.\n' "$RUN_ID"
            printf 'Checkpoints are worth rescuing before anything else; resume is the default (§11).\n'
            remote "tail -25 '$LOG'"
            exit 1
            ;;
        unreachable)
            printf '%s BOX UNREACHABLE after %s tries — on vast this may be permanent.\n' \
                "$RUN_ID" "$UNREACHABLE_STRIKES"
            printf 'Off-instance checkpoints are in the bucket; anything local to the box is gone.\n'
            printf 'This is NOT a completion — do not read it as one.\n'
            exit 2
            ;;
    esac

    sleep "$INTERVAL"
done
