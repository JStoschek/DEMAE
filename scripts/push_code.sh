#!/usr/bin/env bash
# Push this checkout to a rented box — run it on the **laptop**, where the history is:
#
#     scripts/push_code.sh --host demae-box
#     scripts/push_code.sh --host demae-box --provision -- --skip-corpus --skip-env
#
# Why this is a script and not a line in the runbook. Everything here was already documented,
# and R0 (#81) still recorded `git sha unknown` for a whole run, because the code reached the box
# by a plain rsync and the one file that carries the code version did not (#107 item 1). The
# mechanism #80 built is sound — `dataset.provenance.git_sha` falls back to `.git_sha`, and
# provisioning prints what it finds — but nothing made *generating and shipping it* inseparable
# from the push. That is this file's entire job: there is no way to run it and send the code
# without the sha, because it is one command rather than two.
#
# The failure it prevents is silent in both directions. A run with no `.git_sha` trains perfectly
# and records `unknown`; a run with a *stale* one records a confident wrong answer, which is
# worse. So the sha is recomputed on every push and never reused.
#
# `.git` is deliberately not in the rsync list. At ~128 MB it is the largest thing that would
# move, and shipping it would make the answer wrong rather than merely expensive: `git status
# --porcelain` on the box sees every unshipped path (`docs/`, `notebooks/`, `CONTEXT.md`) as
# deleted, so every run would stamp `-dirty` from a clean commit. The laptop computes the SHA
# where the history actually is and ships the one-line answer.
#
# What this script deliberately does NOT do: rent, destroy, or start a run. See
# scripts/provision_box.sh for the box-side half and scripts/teardown_box.sh for the other end.

set -euo pipefail

HOST=""
ROOT="/workspace/DEMAE"
DRY_RUN=0
PROVISION=0
PROVISION_ARGS=()

# The list the runbook §5 documents, plus the sha. `docs/`, `notebooks/` and the plan itself are
# absent on purpose — the box trains, it does not get read on.
PAYLOAD=(src tests artifacts scripts pyproject.toml uv.lock README.md)

# The one document the box does not merely get read on — it *reads*. `benchmark.prereg.read_bar`
# re-checks the committed bar against plan §6's text every time it is opened, so a Task 1 scoring
# run without the plan on disk fails at `run_pass`, after training (#162 hit this at the end of a
# 56-minute seed). "The box trains, it does not get read on" was written about Phase 1, which
# never opens the plan; Task 1 does, which makes this file an input rather than documentation.
#
# Shipped with `--relative` so it lands at docs/research/ rather than the destination root, which
# is where `paths.plan_path` looks for it. Its own rsync, so PAYLOAD's `--delete` keeps meaning
# what the comment below says.
PLAN_PAYLOAD=(docs/research/task1-void-fill-plan.md)

usage() {
    cat <<'EOF'
push_code.sh — send this checkout, and the code version with it, to a rented box.

  --host ALIAS       ssh alias for the box (required; vast never gives you port 22, so this
                     wants an alias with the port in it rather than root@ip)
  --root DIR         where the repo lives on the box (default: /workspace/DEMAE)
  --provision        stream scripts/provision_box.sh over the same connection afterwards
  --dry-run          print the plan and touch nothing — not the box, not .git_sha
  -- ARGS...         everything after -- is passed to provision_box.sh
  -h, --help         this

Run it again on every push. A .git_sha from three pushes ago is a confident wrong answer, which
is worse than the `unknown` it replaced.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host) HOST="$2"; shift 2 ;;
        --root) ROOT="$2"; shift 2 ;;
        --provision) PROVISION=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --) shift; PROVISION_ARGS=("$@"); break ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -n "$HOST" ]] || { echo "--host is required" >&2; usage >&2; exit 2; }

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
ok()   { printf '   ok   %s\n' "$*"; }
warn() { printf '\033[33m   warn %s\033[0m\n' "$*" >&2; }
die()  { printf '\n\033[31mFAILED: %s\033[0m\n' "$*" >&2; exit 1; }

# --- The code version -------------------------------------------------------------------------
# Through the project's own git_sha() rather than `git rev-parse` directly: that is what carries
# the `-dirty` suffix, and a run from uncommitted code should say so on its own run record.
step ".git_sha — the code version every run on that box will record"
# `|| true` because `set -o pipefail` would otherwise abort the script *at this assignment* when
# uv fails — taking with it the two checks below, which exist for exactly that case.
SHA="$(uv run python -c 'from dataset.provenance import git_sha; print(git_sha())' 2>/dev/null \
    | tail -1 | tr -d '[:space:]' || true)"

# An empty answer and the literal `unknown` are both refused, and `unknown` is the one worth
# arguing about: writing it would leave a *non-empty* .git_sha on the box, which silences
# provisioning's step-6 warning while recording nothing. Absent is honest; "unknown" written
# confidently to a file is not.
[[ -n "$SHA" ]] || die "git_sha() produced nothing — is this a checkout with its dependencies?"
[[ "$SHA" != "unknown" ]] \
    || die "git_sha() says 'unknown' here, so there is no version to ship. Run this from the
        laptop checkout that has the .git directory."
if [[ "$SHA" == *-dirty ]]; then
    warn "$SHA — this box will run uncommitted code, and its run record will say so"
else
    ok "$SHA"
fi

# --- The push ---------------------------------------------------------------------------------
# --delete keeps the box's copy of each *transferred directory* honest, and reaches no further:
# rsync scopes deletion to the directories it is given, so data/, runs/, .venv/ and .box_identity
# at the destination root are untouched. That is what makes this safe to re-run against a box
# mid-corpus — and it is the property to re-check before ever adding a source to PAYLOAD.
RSYNC=(rsync -az --delete "${PAYLOAD[@]}" .git_sha "$HOST:$ROOT/")
# No --delete: this one carries a single file into a directory the box may hold other things in,
# and scoping deletion to `docs/` would be a rule about documents this script has no business
# enforcing.
RSYNC_PLAN=(rsync -azR "${PLAN_PAYLOAD[@]}" "$HOST:$ROOT/")

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo
    echo "   would write .git_sha  ($SHA)"
    printf '   would run  %s\n' "${RSYNC[*]}"
    printf '   would run  %s\n' "${RSYNC_PLAN[*]}"
    if [[ "$PROVISION" -eq 1 ]]; then
        printf '   would run  ssh %s bash -s -- --root %s %s < scripts/provision_box.sh\n' \
            "$HOST" "$ROOT" "${PROVISION_ARGS[*]:-}"
    fi
    exit 0
fi

printf '%s\n' "$SHA" > .git_sha
"${RSYNC[@]}" || die "rsync failed — nothing on that box has changed, so just re-run"
"${RSYNC_PLAN[@]}" || die "the plan did not ship — Task 1 scoring reads it; just re-run"
ok "code and .git_sha are on $HOST:$ROOT/"
ok "${PLAN_PAYLOAD[*]} — §6's bar is re-checked against it at scoring time"

if [[ "$PROVISION" -eq 1 ]]; then
    step "provision_box.sh, streamed over the same connection"
    # Streamed over stdin so it never lands on that disk as a file, exactly as its own header
    # documents. The && chain is the point of doing it here: a failed rsync never reaches it.
    ssh "$HOST" "bash -s -- --root '$ROOT' ${PROVISION_ARGS[*]:-}" < scripts/provision_box.sh
else
    cat <<EOF

Next, on the box:

    ssh $HOST 'bash -s -- --root $ROOT' < scripts/provision_box.sh

or re-run this with --provision to do both over one connection.
EOF
fi
