#!/usr/bin/env bash
# Ship each checkpoint off the box *while the run is still going* — run this on the **box**, in
# its own tmux window beside the training one:
#
#     tmux new-window -t r1 -n sync
#     scripts/sync_checkpoints.sh --checkpoint-dir /workspace/DEMAE/runs/R1 \
#         --remote bucket:demae-checkpoints/r1/
#
# Why this exists. Plan §11 asks for every checkpoint to land off-instance "every epoch *and*
# every 30-60 minutes of wall-clock", because a vast.ai instance can vanish permanently rather
# than merely pause. Nothing in src/pretrain implements that — checkpoint.py writes to local disk
# and only mentions syncing in a comment — and teardown_box.sh calls its own pass "the safety
# net, not the mechanism". Without this loop there is no mechanism: a box that disappears at hour
# 40 of a 43-hour run takes every checkpoint with it, not merely the last interval's worth.
#
# `copy`, never `sync`. This is the one decision here that must not be "tidied up" later.
# checkpoint.py prunes local checkpoints as it goes (§11 retention: last 3 plus the 50/100/200/400
# milestones), so the box deliberately stops holding most of what it has written. `rclone sync`
# mirrors deletions, and would walk that prune straight into the bucket — destroying the archive
# this script exists to build, at the exact moment the archive is the only copy. `rclone copy`
# has no delete path at all. That asymmetry is the entire safety property.
#
# Overwrites. Re-copying a checkpoint is a destructive PUT: on B2 an overwrite makes a new
# version and the old survives, but **R2 has no object versioning at all** and its deletes are
# irreversible — so the caveat provision_box.sh and teardown_box.sh state about the checkpoint
# key ("only holds with versioning enabled") has no R2 implementation. On R2 the equivalent is a
# *bucket lock*: a retention rule, optionally scoped to a key prefix, that refuses deletes and
# overwrites outright rather than keeping a copy behind them. It is the stronger property, and
# an Object Read & Write token cannot lift it — editing bucket configuration is Admin-tier.
#
# That makes this script's copy-only behaviour a precondition rather than a nicety: under a lock
# an overwrite is refused, so a pass that tried to re-ship an existing checkpoint *filename* with
# different bytes would fail loudly. Normal operation never does — names carry the step number,
# so passes only ever add keys, and --checksum skips anything already hash-matching.
#
# Failures do not stop the loop. A transient DNS blip or a 500 from the store must not end
# checkpoint shipping for the next 40 hours — exiting would guarantee the outcome this script
# exists to prevent, while retrying merely risks it. So a failed pass warns, counts, and comes
# back. The count is printed every time precisely so a persistent failure cannot masquerade as
# a healthy loop scrolling past in tmux.
#
# Credentials: --remote names an rclone remote configured from the environment, exactly like the
# corpus one (RCLONE_CONFIG_<NAME>_TYPE, _PROVIDER, _ENDPOINT, _ACCESS_KEY_ID, …). It needs write
# on the checkpoint prefix and nothing else. Nothing is written to this box's disk.

set -euo pipefail

CKPT_DIR=""
REMOTE=""
INTERVAL=1800          # §11 asks for 30-60 min; the shorter end, since the cost is bandwidth only
ONCE=0
VERIFY=0
FOREVER=0
PATTERN='checkpoint_*.pt'

# The rclone floor, kept in step with provision_box.sh's — tests/test_ops_scripts.py asserts the
# two agree, because a floor that drifted would let provisioning bless what this script rejects.
# The 2022 build the vastai image ships fails the first upload attempt of every checkpoint with
# `NotImplemented (501)` and succeeds on the retry (R0, #81: three for three against R2). This
# script is where that matters most, because the failure arrives as ERROR lines in a tmux window
# somebody is meant to be able to read for 43 hours — see the check just below the helpers.
RCLONE_MIN="1.65.0"

usage() {
    cat <<'EOF'
sync_checkpoints.sh — copy checkpoints off the box on an interval, while the run is still going.

  --checkpoint-dir DIR  the run's checkpoint directory on this box (required)
  --remote REMOTE       rclone destination, e.g. bucket:demae-checkpoints/r1/ (required)
  --interval N          seconds between passes (default: 1800)
  --pattern GLOB        which files to ship (default: checkpoint_*.pt)
  --verify              re-hash against the remote after each pass (moves no bytes)
  --once                run a single pass and exit — use this to test credentials
  --forever             keep looping after training exits, instead of a final pass and exit
  -h, --help            this

Exits on its own when training has started and then stopped, after one final pass — so the
window closes itself at the end of a run rather than leaving a loop nobody remembers to stop.
EOF
}

step()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
ok()    { printf '   ok    %s\n' "$*"; }
warn()  { printf '   \033[33mwarn\033[0m  %s\n' "$*"; }
die()   { printf '\n\033[31mSTOPPED: %s\033[0m\n' "$*" >&2; exit 1; }
stamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }

# --- The rclone version, read as three numbers ------------------------------------------------
# Carried here as well as in provision_box.sh rather than shared through a file both source: this
# script runs on the box in its own tmux window, started by hand, and can assume nothing about
# what provisioning left behind — the same reason the R2 signing-region swap is repeated by every
# bucket-touching path in this repo. tests/test_ops_scripts.py drives both copies.

# "rclone v1.60.1-DEV" -> "1.60.1". Suffixes are dropped; unreadable input reads as 0.0.0, which
# fails the floor rather than passing it.
rclone_semver() {
    if [[ "$1" =~ ([0-9]+)\.([0-9]+)\.([0-9]+) ]]; then
        printf '%s.%s.%s\n' "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}" "${BASH_REMATCH[3]}"
    elif [[ "$1" =~ ([0-9]+)\.([0-9]+) ]]; then
        printf '%s.%s.0\n' "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"
    else
        printf '0.0.0\n'
    fi
}

# rclone_version_lt <have> <floor> — numeric per component, because 1.9 is older than 1.65 and
# every comparison that treats these as text says the opposite.
rclone_version_lt() {
    local have floor h f i have_n floor_n
    have="$(rclone_semver "$1")"; floor="$(rclone_semver "$2")"
    IFS=. read -r -a h <<< "$have"
    IFS=. read -r -a f <<< "$floor"
    for i in 0 1 2; do
        have_n="${h[$i]:-0}"; floor_n="${f[$i]:-0}"
        if (( 10#$have_n < 10#$floor_n )); then return 0; fi
        if (( 10#$have_n > 10#$floor_n )); then return 1; fi
    done
    return 1
}

# Sourcing with DEMAE_LIB set stops here, before argv is read — see tests/test_ops_scripts.py.
if [[ -n "${DEMAE_LIB:-}" ]]; then
    return 0
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint-dir) CKPT_DIR="$2"; shift 2 ;;
        --remote) REMOTE="$2"; shift 2 ;;
        --interval) INTERVAL="$2"; shift 2 ;;
        --pattern) PATTERN="$2"; shift 2 ;;
        --verify) VERIFY=1; shift ;;
        --once) ONCE=1; shift ;;
        --forever) FOREVER=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -n "$CKPT_DIR" ]] || { echo "--checkpoint-dir is required" >&2; exit 2; }
[[ -n "$REMOTE" ]]   || { echo "--remote is required" >&2; exit 2; }
[[ "$REMOTE" == *:* ]] || { echo "--remote must be an rclone remote (name:path)" >&2; exit 2; }

command -v rclone >/dev/null || die "no rclone on PATH — provision_box.sh installs it"

RCLONE_HAVE="$(rclone_semver "$(rclone version 2>/dev/null | head -1 || true)")"
if rclone_version_lt "$RCLONE_HAVE" "$RCLONE_MIN"; then
    # `--once` is the pre-run credential probe the runbook has you run before committing GPU
    # hours, so it is the one place a hard stop is cheap: nothing is training yet, and the fix
    # is to re-run provisioning. The loop must never die of this — it is started *during* a run,
    # and exiting would end checkpoint shipping for the next 40 hours to avoid some ERROR lines,
    # which is precisely the trade this script's header refuses to make.
    BELOW_FLOOR="rclone $RCLONE_HAVE is below the $RCLONE_MIN floor. A client this old failed the
        first upload attempt of every checkpoint against R2 with NotImplemented (501) and
        succeeded on the retry, three for three in R0 — so a *healthy* pass logs ERROR lines, and
        a real failure then has somewhere to hide (#107). Replace it from the repo root on this
        box:
            scripts/provision_box.sh --root \"\$PWD\" --skip-corpus --skip-env --probe-steps 0"
    if [[ "$ONCE" -eq 1 ]]; then
        die "$BELOW_FLOOR"
    fi
    warn "$BELOW_FLOOR"
fi

PASS=0
FAILS=0
SHIPPED=0
SAW_TRAINING=0

# One pass. Deliberately not fatal on failure — see the header. Returns 0/1 so the caller can
# count, and never trips `set -e` on the rclone call itself.
sync_once() {
    local n
    if [[ ! -d "$CKPT_DIR" ]]; then
        warn "$(stamp)  $CKPT_DIR does not exist yet — nothing written this early in a run"
        return 0
    fi
    n="$(find "$CKPT_DIR" -maxdepth 1 -name "$PATTERN" -type f 2>/dev/null | wc -l | tr -d ' ')"
    if [[ "$n" -eq 0 ]]; then
        warn "$(stamp)  no $PATTERN in $CKPT_DIR yet"
        return 0
    fi

    # --checksum, matching provision_box.sh's corpus fetch and for the same reason: compare
    # content hashes rather than size+mtime, so a pass moves only what genuinely differs and a
    # re-run after a killed transfer resumes instead of restarting.
    # --s3-no-check-bucket is #107's second lever, and it is the one with a *mechanism* behind
    # it rather than a version number. Before its first upload, rclone's S3 backend probes the
    # bucket and will try to create it; it then caches the answer for the life of the process.
    # That shape matches R0's evidence exactly — one failure per pass, on attempt 1, then a
    # success, three for three — because this loop runs a fresh `rclone copy` per pass, so the
    # probe happens once per pass and not once per run. The box's token is Object Read & Write,
    # which cannot create a bucket, and R2 is entitled to answer that with NotImplemented (501).
    # The flag skips the probe outright.
    #
    # Stated honestly: this was never confirmed on the box, so it is a hypothesis and the pinned
    # rclone in provisioning is the other half. It is safe whether or not it is the cause — the
    # buckets are created by a human long before this runs, the token could not create one
    # anyway, and skipping the probe only turns "rclone silently creates it" into a clear error.
    # Harmless on a non-S3 remote too, which is what the local-remote tests use.
    #
    # Timeouts are not optional here. rclone's defaults retry a dead endpoint for a very long
    # time, and a pass that hangs does not merely fail — it stops the loop from ever reaching the
    # next interval, so checkpoint shipping stops with no warning printed. Bounded retries turn a
    # network partition into a visible failed pass that the loop comes back from.
    if rclone copy "$CKPT_DIR" "$REMOTE" \
        --include "$PATTERN" --checksum --stats-one-line --stats 60s \
        --s3-no-check-bucket \
        --contimeout 30s --timeout 5m --retries 3 --low-level-retries 5; then
        SHIPPED="$n"
        FAILS=0
        ok "$(stamp)  pass $PASS — $n checkpoint(s) in sync with $REMOTE"
    else
        FAILS=$(( FAILS + 1 ))
        warn "$(stamp)  pass $PASS FAILED (${FAILS} consecutive) — retrying in ${INTERVAL}s"
        return 1
    fi

    if [[ "$VERIFY" -eq 1 ]]; then
        # One-way: every local checkpoint must exist and hash-match remotely. Extra objects in
        # the bucket are expected and fine — they are the ones retention has already pruned here,
        # which is the whole point of copy-not-sync.
        if rclone check "$CKPT_DIR" "$REMOTE" --include "$PATTERN" --checksum --one-way \
            --s3-no-check-bucket --contimeout 30s --timeout 5m --retries 3; then
            ok "$(stamp)  verified — every local checkpoint hash-matches the bucket"
        else
            FAILS=$(( FAILS + 1 ))
            warn "$(stamp)  VERIFY FAILED — the bucket disagrees with this box"
            return 1
        fi
    fi
    return 0
}

# A pass on the way out, so an interrupted loop still ships whatever landed since the last one.
final_pass() {
    step "Final pass"
    sync_once || warn "the final pass failed — do NOT destroy this box; run teardown_box.sh"
    printf '\n%s passes, %s checkpoint(s) last confirmed in %s\n' "$PASS" "$SHIPPED" "$REMOTE"
    exit 0
}
trap final_pass INT TERM

step "Checkpoint sync — $CKPT_DIR -> $REMOTE"
ok "interval ${INTERVAL}s, pattern '$PATTERN', verify=$VERIFY, rclone v$RCLONE_HAVE"
echo "   note  this is copy-only; retention pruning on this box never reaches the bucket"

if [[ "$ONCE" -eq 1 ]]; then
    PASS=1
    # Probe the remote first. Without this, --once on a run that has not written its first
    # checkpoint yet returns 0 having never touched the network — which is exactly backwards for
    # the flag's advertised job of proving the credentials work *before* 43 hours of GPU time.
    PROBE=(--contimeout 10s --timeout 30s --retries 1 --low-level-retries 1)
    if rclone lsd "$REMOTE" "${PROBE[@]}" >/dev/null 2>&1 \
        || rclone lsd "${REMOTE%%/*}" "${PROBE[@]}" >/dev/null; then
        ok "remote $REMOTE is reachable and the credentials authenticate"
    else
        die "cannot reach $REMOTE — check RCLONE_CONFIG_* for that remote, and the onstart line"
    fi
    sync_once || die "the single pass failed — fix this before starting a run"
    exit 0
fi

while true; do
    PASS=$(( PASS + 1 ))
    sync_once || true

    # Exit condition, checked *after* a pass so the last checkpoint of a finished run is always
    # shipped. Only fires once training has actually been seen: starting this window before the
    # trainer would otherwise make the loop exit immediately on an empty pgrep.
    if [[ "$FOREVER" -eq 0 ]]; then
        if pgrep -f 'pretrain\.(launch|train)' >/dev/null 2>&1; then
            SAW_TRAINING=1
        elif [[ "$SAW_TRAINING" -eq 1 ]]; then
            ok "$(stamp)  training has exited"
            final_pass
        fi
    fi

    sleep "$INTERVAL"
done
