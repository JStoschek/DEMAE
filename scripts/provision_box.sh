#!/usr/bin/env bash
# Bring a freshly rented vast.ai box to the line plan §15 gates R1 at — run it *on the box*,
# streamed over the SSH connection so it never lands on that disk as a file. scripts/push_code.sh
# does both halves, and is the documented route because it is what also ships `.git_sha`:
#
#     scripts/push_code.sh --host demae-box --provision
#     scripts/push_code.sh --host demae-box --provision -- --skip-corpus --skip-env
#
# Idempotent throughout: `rclone copy` re-transfers only what disagrees, and every other step
# checks whether it is already done, so a run that dies on a flaky download is resumed by
# re-running the same command rather than from zero — the same property ADR-0001's live-read
# stages have, and for the same reason.
#
# Credentials. The box holds two scoped keys, supplied by vast `--env` and read here out of the
# environment; neither is an account master key:
#
#   corpus prefix       read + list only. Worst case if the host owner takes it: they read a
#                       derived copy of a public elevation dataset.
#   checkpoint prefix   read + write, **no delete capability**. A run invents checkpoint
#                       filenames over 43 hours, so this one cannot be presigned or expired —
#                       what limits it instead is that it cannot remove anything. On B2 that
#                       is airtight (an overwrite makes a new version and the old survives);
#                       on R2/S3 it only holds with **versioning enabled** on the prefix,
#                       because a plain PUT to an existing key overwrites destructively.
#
# What this script deliberately does NOT do: rent or destroy. Both spend money or delete data
# and stay a human's to run. See scripts/teardown_box.sh for the other end.
#
# Image: vastai/pytorch (any recent py311-py313 tag). Its bundled torch is not used — uv.lock
# pins torch 2.13.0 and its whole CUDA userspace as wheels, and that lock is the reproducibility
# anchor. The image supplies CUDA drivers, sshd and a Python to bootstrap uv from; ~2.7 GB of
# wheels land in .venv on top. Worth knowing rather than fixing: the image's own pytorch is
# dead weight, but it is *cached on vast hosts*, so it mostly costs nothing to pull.

set -euo pipefail

ROOT="/workspace/DEMAE"
CORPUS_REMOTE="${DEMAE_CORPUS_REMOTE:-}"
JOBS="${JOBS:-6}"
EXPECT_VERSION="bcbd6484d8116199c5c1160e21ea0284"  # the shipped 500k tier (#72)
MIN_RAM_GB=60      # §11 asks for >=64; hosts report a little under what they advertise
                   # — and this is a *pretraining* floor. §11 wants it because pretraining
                   # streams all 500k tiles every epoch, so the whole 47 GB corpus has to sit in
                   # page cache. A Task 1 benchmark run does not: §3.1's draws are 20,000
                   # head-train + 2,000 val + 5,000 test = 27,000 tiles, and at 98 KiB a tile
                   # that is a **2.71 GB** working set out of a memmap the kernel pages lazily.
                   # So `--min-ram-gb` exists, on `--max-corpus-min`'s own rule: the number is a
                   # property of the *workload*, and a box this benchmark can use is not the same
                   # box §11 was written about. 0 skips the check.

# The disk budget, as its three parts rather than one number (#107). The sum on a box holding
# nothing is 70 GB, which is the constant this replaces — the change is not a looser threshold,
# it is the same threshold asked about what still has to *land*.
CORPUS_GB=47       # the 500k tier on disk (ADR-0002's manifest; 34 shards)
VENV_GB=8          # ~2.7 GB of wheels in .venv, and uv's cache holds its own copy of them
VENV_RESYNC_GB=2   # a .venv that already exists: `uv sync --frozen` mostly re-checks hashes
CKPT_GB=15         # §11 retention keeps last 3 + the 4 milestones = 7 files, ~1.2 GB each at
                   # ViT-B (§12's largest), plus the one being written and W&B's own directory

# rclone. The vastai image ships v1.60.1-DEV — a 2022 build — and against R2 every checkpoint
# upload failed `NotImplemented (501)` on attempt 1 and succeeded on attempt 2, three for three
# in R0 (#81). The bytes were fine (a re-downloaded checkpoint hashed identically), so the cost
# was three ERROR lines per sync pass for 43 hours, which is how a real failure gets ignored.
#
# What is *not* claimed here: that some particular release fixed it. The mechanism was never
# isolated on the box, so RCLONE_MIN is a floor chosen to get off the 2022 series and onto a
# client that knows R2, not a bisected commit. What the pin buys regardless is that R1-R5 all
# ship their checkpoints through the same client, and that .box_identity records which one.
RCLONE_MIN="1.65.0"
RCLONE_PIN="1.69.1"

# The download gate. vast's advertised inet_down is a host-reported number and it is regularly
# fiction — a box billed at 2 Gbit/s that pulls the corpus at 25 MiB/s costs 30 minutes of rent
# before anything trains, and the only way to know is to move bytes off the real bucket. So
# preflight does exactly that, and converts what it measures into the only number that decides
# anything: how long *this* box would take to land the corpus it is still missing.
#
# 20 minutes is the budget, not a speed. Speeds do not compare across boxes with different
# amounts already on disk, and a re-provision that only needs 5 GB should not be held to the
# same MiB/s as a fresh one.
MAX_CORPUS_MIN=20   # 0 disables the gate
NET_PROBE_MIB=64    # per stream, ranged out of a real shard
NET_PROBE_SECONDS=25 # ceiling on the probe itself; a host too slow to finish it has failed

RECHECK=0
SKIP_ENV=0
SKIP_CORPUS=0
PROBE_STEPS=200

usage() {
    cat <<'EOF'
provision_box.sh — bring a rented box to plan §15's pre-flight line.

  --root DIR         where the repo lives on the box (default: /workspace/DEMAE)
  --corpus-remote U  rclone URI of the corpus, e.g. demae:demae-bucket/corpus_500k
                     (default: $DEMAE_CORPUS_REMOTE)
  --jobs N           rclone parallel transfers (default: 6)
  --expect-version H dataset_version the corpus must report (default: the 500k tier)
  --recheck          re-run `rclone check` after the copy (hash compare, no re-download)
  --skip-corpus      leave the corpus alone; verify whatever is already on disk
  --skip-env         skip the uv sync step
  --probe-steps N    steps for the loader throughput probe, 0 to skip (default: 200)
  --max-corpus-min N reject the box if preflight measures the corpus taking longer than this
                     many minutes to download, 0 to skip the measurement (default: 20)
  --min-ram-gb N     reject the box below this much RAM, 0 to skip (default: 60). The default is
                     §11's pretraining floor, which exists so the 47 GB corpus page-caches; a
                     Task 1 benchmark run touches 27,000 tiles (2.71 GB) and does not need it
  --rclone-version V rclone to install when this box's is older than the floor (default: 1.69.1)
  -h, --help         this

The rclone remote is configured entirely from the environment, so nothing needs writing to
this box's disk. Pass these through vast `--env` (with the documented
`env | grep _ >> /etc/environment` onstart line, or they are invisible to your shell):

  RCLONE_CONFIG_DEMAE_TYPE=s3
  RCLONE_CONFIG_DEMAE_PROVIDER=Other
  RCLONE_CONFIG_DEMAE_ENDPOINT=https://s3.us-west-004.backblazeb2.com
  RCLONE_CONFIG_DEMAE_ACCESS_KEY_ID=...        # the read-only corpus key
  RCLONE_CONFIG_DEMAE_SECRET_ACCESS_KEY=...
  DEMAE_CORPUS_REMOTE=demae:demae-bucket/corpus_500k
EOF
}

step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
ok()   { printf '   ok   %s\n' "$*"; }
warn() { printf '\033[33m   warn %s\033[0m\n' "$*" >&2; }
die()  { printf '\n\033[31mFAILED: %s\033[0m\n' "$*" >&2; exit 1; }

# --- The two decisions that are arithmetic ----------------------------------------------------
# Both are pure functions of their arguments, which is what lets tests/test_ops_scripts.py source
# this file and call them with no box, no GPU and no network — see the DEMAE_LIB guard below.

# required_free_gb <corpus_gb_on_disk> <have_venv 0|1> <skip_corpus 0|1> <skip_env 0|1>
#
# How much free space this box still needs, as opposed to how much the finished article occupies.
# The distinction is the whole fix (#107): the old check compared a flat 70 GB against total free
# space, and 70 GB is what corpus + venv + a run's writes *add up to* — so the moment the corpus
# was down, the disk it was sitting on failed the check reserving room for it. R0's box (102 GB
# disk, 53 GB used by exactly the two things provisioning had put there) could not be
# re-provisioned at all, which took `--skip-corpus` and the runbook's "re-run to resume" with it.
#
# Corpus and venv are subtracted only because they are *checkable* — step 4 re-verifies the
# corpus against the manifest and step 2 re-runs `uv sync`, so trusting the disk here costs
# nothing that is not re-earned later in the same run. The checkpoint reserve is never
# subtracted: those writes are always still ahead.
required_free_gb() {
    local present="$1" have_venv="$2" skip_corpus="$3" skip_env="$4" need=0 remaining
    if [[ "$skip_corpus" -eq 0 ]]; then
        remaining=$(( CORPUS_GB - present ))
        if (( remaining > 0 )); then need=$(( need + remaining )); fi
    fi
    if [[ "$skip_env" -eq 0 ]]; then
        if [[ "$have_venv" -eq 1 ]]; then
            need=$(( need + VENV_RESYNC_GB ))
        else
            need=$(( need + VENV_GB ))
        fi
    fi
    printf '%s\n' $(( need + CKPT_GB ))
}

# corpus_eta_minutes <bytes_measured> <elapsed_ms> <gib_still_to_land>
#
# The probe's whole output, reduced to the question preflight is asking. Rounds *up*, because a
# gate that reads 20.9 minutes as 20 is not a 20-minute gate.
#
# Two things it deliberately does not do. It does not extrapolate a per-stream speed — the
# measurement is already the aggregate of the same number of parallel transfers step 3 will use,
# so the rate goes in as one number. And it does not pad for slow start or TLS setup: those make
# a short probe read *slower* than the sustained copy, which is the safe direction for a gate
# that spends money when it is wrong in the other one.
#
# Zero bytes is not zero speed, it is no measurement, and the caller has to be able to tell those
# apart — so it comes back as an ETA nothing can pass rather than a division by zero.
corpus_eta_minutes() {
    local bytes="$1" ms="$2" gib="$3" num denom
    if (( bytes <= 0 )); then printf '999999\n'; return 0; fi
    if (( ms <= 0 )); then ms=1; fi
    if (( gib <= 0 )); then printf '0\n'; return 0; fi
    # gib * 2^30 bytes, at (bytes / ms) bytes per millisecond, in minutes. Grouped so the
    # multiplications happen before the divisions — integer division early would quantise a
    # 47 GB estimate to the nearest minute-per-GB and throw away the point of measuring.
    num=$(( gib * 1073741824 * ms ))
    denom=$(( bytes * 60000 ))
    printf '%s\n' $(( (num + denom - 1) / denom ))
}

# probe_mib_per_s <bytes_measured> <elapsed_ms> -> "112.4"
#
# For the human reading the log, not for the decision — the decision is corpus_eta_minutes. One
# decimal because the interesting range spans two orders of magnitude and "0 MiB/s" is a worse
# thing to print than "0.4 MiB/s" when a host is genuinely that bad.
probe_mib_per_s() {
    local bytes="$1" ms="$2" tenths
    if (( bytes <= 0 || ms <= 0 )); then printf '0.0\n'; return 0; fi
    tenths=$(( bytes * 10000 / ms / 1048576 ))
    printf '%s.%s\n' $(( tenths / 10 )) $(( tenths % 10 ))
}

# rclone_semver "<rclone version's first line>" -> "1.60.1"
#
# Deliberately incurious about everything but the three numbers. The image's build calls itself
# `v1.60.1-DEV` and a beta calls itself `v1.71.0-beta.8123.abcdef`; neither suffix says anything
# about which release the code is. Anything unreadable reads as 0.0.0, so it fails the floor and
# gets replaced — the other direction would let an unparseable version quietly pass.
rclone_semver() {
    if [[ "$1" =~ ([0-9]+)\.([0-9]+)\.([0-9]+) ]]; then
        printf '%s.%s.%s\n' "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}" "${BASH_REMATCH[3]}"
    elif [[ "$1" =~ ([0-9]+)\.([0-9]+) ]]; then
        printf '%s.%s.0\n' "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"
    else
        printf '0.0.0\n'
    fi
}

# rclone_version_lt <have> <floor> — true when `have` is older than `floor`.
#
# Numeric per component, because the interesting comparison is not: 1.9 is older than 1.65, and
# every string or `sort` answer gets that backwards.
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

# Sourcing this file with DEMAE_LIB set defines the above and stops here, before argv is read or
# anything is measured. It is the only concession in this script to being tested off-box.
if [[ -n "${DEMAE_LIB:-}" ]]; then
    return 0
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --root) ROOT="$2"; shift 2 ;;
        --corpus-remote) CORPUS_REMOTE="$2"; shift 2 ;;
        --jobs) JOBS="$2"; shift 2 ;;
        --expect-version) EXPECT_VERSION="$2"; shift 2 ;;
        --recheck) RECHECK=1; shift ;;
        --skip-corpus) SKIP_CORPUS=1; shift ;;
        --skip-env) SKIP_ENV=1; shift ;;
        --probe-steps) PROBE_STEPS="$2"; shift 2 ;;
        --max-corpus-min) MAX_CORPUS_MIN="$2"; shift 2 ;;
        --min-ram-gb) MIN_RAM_GB="$2"; shift 2 ;;
        --rclone-version) RCLONE_PIN="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

CORPUS_DIR="$ROOT/data/corpus_500k"

# ---------------------------------------------------------------------------------------------
step "1/8  Preflight — is this box worth provisioning?"
# Deliberately first and deliberately fatal. A wrong host should cost 30 seconds, not the 20
# minutes it takes to discover the corpus does not fit after it has half-downloaded. Every check
# here is free except the last, which spends ~25s and ~384 MB measuring what this host's link
# actually does — and buys back the half hour a host that lies about its bandwidth would cost.

command -v nvidia-smi >/dev/null || die "no nvidia-smi — this is not a GPU box"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
GPU_MEM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)"
GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
ok "GPU     ${GPU_COUNT}x ${GPU_NAME} (${GPU_MEM_MB} MiB)"
[[ "$GPU_COUNT" -eq 1 ]] || echo "   note   plan §11 assumes a single GPU; $GPU_COUNT present"

RAM_GB="$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo)"
if [[ "$MIN_RAM_GB" -eq 0 ]]; then
    ok "RAM     ${RAM_GB} GB (floor waived; see --min-ram-gb)"
else
    [[ "$RAM_GB" -ge "$MIN_RAM_GB" ]] \
        || die "RAM ${RAM_GB} GB < ${MIN_RAM_GB} GB — the 47 GB corpus will not page-cache (§11). \
That floor is pretraining's: a Task 1 benchmark run reads 27,000 tiles (2.71 GB) and can pass \
--min-ram-gb with a lower number, or 0"
    ok "RAM     ${RAM_GB} GB"
fi

CPUS="$(nproc)"
ok "CPUs    ${CPUS} (the loader wants workers; §7.5 names IO the #1 risk)"

mkdir -p "$ROOT"
FREE_GB="$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')"
# What is already down, so the requirement is the remainder rather than the total. `du` over the
# corpus is 34 shard files, not 500k tiles — it is a stat call each, and returns instantly.
CORPUS_ON_DISK_GB=0
if [[ -d "$CORPUS_DIR" ]]; then
    # `|| true` so one unreadable shard cannot abort step 1/8 through `set -o pipefail` — it
    # would exit with nothing printed, and the `:-0` below is what is supposed to handle it.
    CORPUS_ON_DISK_GB="$(du -s --block-size=1G "$CORPUS_DIR" 2>/dev/null | cut -f1 || true)"
    CORPUS_ON_DISK_GB="${CORPUS_ON_DISK_GB:-0}"
fi
HAVE_VENV=0
if [[ -d "$ROOT/.venv" ]]; then HAVE_VENV=1; fi
NEED_GB="$(required_free_gb "$CORPUS_ON_DISK_GB" "$HAVE_VENV" "$SKIP_CORPUS" "$SKIP_ENV")"
[[ "$FREE_GB" -ge "$NEED_GB" ]] \
    || die "only ${FREE_GB} GB free at $ROOT; this run still has to land ${NEED_GB} GB
        (corpus on disk ${CORPUS_ON_DISK_GB}/${CORPUS_GB} GB, venv present=${HAVE_VENV},
         and ${CKPT_GB} GB reserved for what the run itself writes)"
ok "Disk    ${FREE_GB} GB free at $ROOT, ${NEED_GB} GB still to land"
if [[ "$CORPUS_ON_DISK_GB" -gt 0 ]]; then
    ok "        ${CORPUS_ON_DISK_GB} GB of corpus already here — step 4 is what says it is whole"
fi

# --- How long the corpus would take to land ---------------------------------------------------
# Last in preflight because it is the only part that moves bytes, and everything above it is free.
#
# Measured against the actual corpus bucket, with the actual rclone that is on this box right
# now, over the same number of parallel streams step 3 will use. Nothing else answers the
# question: a generic speed test measures the path to a speed-test server, and vast's own
# inet_down is the host's claim about itself, which is the thing being checked.
#
# Ranged reads out of real shards, discarded to `wc -c`. It writes nothing to disk, and the
# bytes are not wasted twice: they are the same bytes step 3 fetches, from the same objects, so
# the worst case is that a box that passes has re-downloaded 384 MB of a 47 GB copy.
#
# Whatever rclone is here is good enough for this, including the image's 2022 build — step 3's
# floor is about a 501 on *upload*, not about how fast a GET arrives. So the probe runs before
# the install, where a bad host is still 30 seconds old.
CORPUS_TO_LAND_GB=$(( CORPUS_GB - CORPUS_ON_DISK_GB ))
(( CORPUS_TO_LAND_GB > 0 )) || CORPUS_TO_LAND_GB=0

now_ms() {
    local t
    t="$(date +%s%3N 2>/dev/null || true)"
    # GNU date gives milliseconds; anything else leaves the %3N in place and gets whole seconds.
    if [[ "$t" =~ ^[0-9]+$ ]]; then printf '%s\n' "$t"; else printf '%s000\n' "$(date +%s)"; fi
}

if [[ "$MAX_CORPUS_MIN" -eq 0 ]]; then
    ok "Network probe skipped (--max-corpus-min 0)"
elif [[ "$SKIP_CORPUS" -eq 1 || "$CORPUS_TO_LAND_GB" -eq 0 ]]; then
    ok "Network probe skipped — no corpus left to download"
elif [[ -z "$CORPUS_REMOTE" ]]; then
    # Not fatal here only because step 3 already dies on this, with the better message.
    warn "no corpus remote set, so the download cannot be timed — step 3 will refuse anyway"
elif ! command -v rclone >/dev/null; then
    warn "no rclone on this box yet, so the download cannot be timed here. Step 3 installs one,"
    warn "and a slow host will then be discovered the expensive way, mid-copy."
else
    NET_CHUNK=$(( NET_PROBE_MIB * 1024 * 1024 ))
    NET_TMP="$(mktemp -d)"
    trap 'rm -rf "$NET_TMP"' EXIT
    # Objects at least one chunk long, so every stream pulls the same amount and a manifest
    # sidecar cannot end a stream after 4 KB and drag the aggregate down with it. Listing is
    # outside the timed section: it is one round trip and it is not what is being measured.
    NET_OBJECTS=()
    while IFS= read -r line; do NET_OBJECTS+=("$line"); done < <(
        rclone lsf --files-only --format "sp" --separator "|" "$CORPUS_REMOTE" 2>"$NET_TMP/lserr" \
            | awk -F'|' -v min="$NET_CHUNK" '$1 >= min {print $2}' \
            | head -n "$JOBS" || true
    )
    if [[ "${#NET_OBJECTS[@]}" -eq 0 ]]; then
        warn "could not list $CORPUS_REMOTE, so the download cannot be timed:"
        sed 's/^/        /' "$NET_TMP/lserr" >&2 || true
        warn "step 3 is where this becomes fatal."
        rm -rf "$NET_TMP"; trap - EXIT
    else
        NET_TIMEOUT=()
        if command -v timeout >/dev/null; then NET_TIMEOUT=(timeout "$NET_PROBE_SECONDS"); fi
        echo "   probing ${#NET_OBJECTS[@]} parallel streams x ${NET_PROBE_MIB} MiB from $CORPUS_REMOTE…"
        NET_N=0
        NET_START="$(now_ms)"
        for NET_OBJ in "${NET_OBJECTS[@]}"; do
            NET_N=$(( NET_N + 1 ))
            {
                ${NET_TIMEOUT[@]+"${NET_TIMEOUT[@]}"} \
                    rclone cat --head "$NET_CHUNK" "$CORPUS_REMOTE/$NET_OBJ" \
                    2>>"$NET_TMP/err" | wc -c > "$NET_TMP/n$NET_N"
            } &
        done
        # A stream cut off by `timeout` is a data point, not an error: what it transferred in
        # NET_PROBE_SECONDS is exactly the throughput being asked about. So nothing here reads
        # an exit status — only the byte counts, and the clock they arrived against.
        wait
        NET_MS=$(( $(now_ms) - NET_START ))
        (( NET_MS > 0 )) || NET_MS=1
        NET_BYTES="$(cat "$NET_TMP"/n* 2>/dev/null | awk '{s+=$1} END {printf "%d", s+0}')"

        if [[ "${NET_BYTES:-0}" -eq 0 ]]; then
            warn "not one byte arrived from $CORPUS_REMOTE in ${NET_PROBE_SECONDS}s:"
            tail -3 "$NET_TMP/err" 2>/dev/null | sed 's/^/        /' >&2 || true
            rm -rf "$NET_TMP"; trap - EXIT
            die "this box cannot read the corpus bucket — either the network is dead or the
        RCLONE_CONFIG_DEMAE_* keys did not reach this shell (see --help's onstart note).
        Nothing has been downloaded, so destroy it and rent another, or fix the env."
        fi
        rm -rf "$NET_TMP"; trap - EXIT

        NET_MIBPS="$(probe_mib_per_s "$NET_BYTES" "$NET_MS")"
        CORPUS_ETA_MIN="$(corpus_eta_minutes "$NET_BYTES" "$NET_MS" "$CORPUS_TO_LAND_GB")"
        [[ "$CORPUS_ETA_MIN" -le "$MAX_CORPUS_MIN" ]] \
            || die "corpus download would take ~${CORPUS_ETA_MIN} min, over the ${MAX_CORPUS_MIN} min limit.
        measured ${NET_MIBPS} MiB/s across ${#NET_OBJECTS[@]} streams ($(( NET_BYTES / 1048576 )) MiB in $(( NET_MS / 1000 ))s),
        with ${CORPUS_TO_LAND_GB} GB still to land. vast's advertised bandwidth is the host's own
        claim; this is what the bucket actually delivers here. Nothing has been downloaded
        and no GPU time is committed — destroy this instance and rent another. To take it
        anyway: re-run with --max-corpus-min ${CORPUS_ETA_MIN}."
        ok "Network ${NET_MIBPS} MiB/s over ${#NET_OBJECTS[@]} streams — ${CORPUS_TO_LAND_GB} GB in ~${CORPUS_ETA_MIN} min (limit ${MAX_CORPUS_MIN})"
    fi
fi

# ---------------------------------------------------------------------------------------------
step "2/8  Python environment"
if [[ "$SKIP_ENV" -eq 1 ]]; then
    ok "skipped (--skip-env)"
elif [[ ! -f "$ROOT/uv.lock" ]]; then
    die "no uv.lock at $ROOT — rsync the code across before this step"
else
    command -v uv >/dev/null || python3 -m pip install --quiet --upgrade uv
    ok "uv      $(uv --version)"
    # --frozen: resolve nothing, install exactly what uv.lock pins. The CUDA userspace
    # (cudnn, cublas, nccl, triton) rides in as wheels, so this pins far more of the
    # environment than the base image does.
    (cd "$ROOT" && uv sync --frozen --extra pretrain)
    TORCH_LINE="$(cd "$ROOT" && uv run python -c \
        'import torch;print(torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())')"
    ok "torch   $TORCH_LINE"
    [[ "$TORCH_LINE" == *"avail True"* ]] || die "torch cannot see the GPU"
fi

# ---------------------------------------------------------------------------------------------
step "3/8  rclone, then the corpus"
# The client is settled *before* --skip-corpus is consulted, and that ordering is the point:
# --skip-corpus is the re-provision path, and the thing an already-corpused box still needs
# rclone for is shipping its checkpoints for the next 43 hours (#107 item 4).
RCLONE_HAVE=""
if command -v rclone >/dev/null; then
    RCLONE_HAVE="$(rclone_semver "$(rclone version 2>/dev/null | head -1 || true)")"
fi
if [[ -z "$RCLONE_HAVE" ]] || rclone_version_lt "$RCLONE_HAVE" "$RCLONE_MIN"; then
    if [[ -n "$RCLONE_HAVE" ]]; then
        echo "   rclone $RCLONE_HAVE is below the $RCLONE_MIN floor — installing v$RCLONE_PIN"
        echo "   (a client this old is what 501s on attempt 1 of every upload; see the header)"
    else
        echo "   installing rclone v$RCLONE_PIN…"
    fi
    case "$(uname -m)" in
        x86_64|amd64) RCLONE_ARCH=amd64 ;;
        aarch64|arm64) RCLONE_ARCH=arm64 ;;
        *) die "no rclone build for $(uname -m); install one by hand and re-run" ;;
    esac
    RCLONE_TMP="$(mktemp -d)"
    trap 'rm -rf "$RCLONE_TMP"' EXIT
    RCLONE_PKG="rclone-v${RCLONE_PIN}-linux-${RCLONE_ARCH}"
    # Extracted with python3 rather than unzip: python3 is already a hard dependency (step 4
    # runs check_corpus.py with it) and unzip is not always in the image.
    if curl -fsSL "https://downloads.rclone.org/v${RCLONE_PIN}/${RCLONE_PKG}.zip" \
            -o "$RCLONE_TMP/rclone.zip" \
        && python3 -m zipfile -e "$RCLONE_TMP/rclone.zip" "$RCLONE_TMP/x" \
        && mkdir -p /usr/local/bin \
        && cp "$RCLONE_TMP/x/$RCLONE_PKG/rclone" /usr/local/bin/rclone \
        && chmod 0755 /usr/local/bin/rclone
    then
        ok "installed v$RCLONE_PIN to /usr/local/bin/rclone"
    else
        # The pin is reproducibility; the floor is the requirement. A withdrawn or renamed
        # release must not strand a box that only needs *some* recent rclone, so fall back to
        # upstream's own installer and let the re-check below decide whether that was enough.
        # That fallback wants `unzip`, which the direct path above deliberately avoids — so it
        # is genuinely a second chance and not the same attempt twice.
        warn "could not install the pinned v$RCLONE_PIN — falling back to rclone's installer"
        curl -fsSL https://rclone.org/install.sh | bash >/dev/null \
            || warn "that failed too"
    fi
    rm -rf "$RCLONE_TMP"
    trap - EXIT
    hash -r
    RCLONE_HAVE=""
    if command -v rclone >/dev/null; then
        RCLONE_HAVE="$(rclone_semver "$(rclone version 2>/dev/null | head -1 || true)")"
    fi
    # Absence is fatal; age is not. Nothing works without an rclone at all — the corpus arrives
    # through it and the checkpoints leave through it. But an *old* one demonstrably ships a run:
    # R0's did, noisily, for 43 hours. Dying here would hand back exactly what step 1/8 just
    # stopped doing — refusing to re-provision a box that has already paid for its corpus, over
    # something cosmetic. It is the same call the .git_sha warning in step 6 makes, and the hard
    # gate is elsewhere: sync_checkpoints.sh --once refuses to run at all below this floor, and
    # that is what a human passes *before* committing GPU hours.
    [[ -n "$RCLONE_HAVE" ]] || die "no rclone on this box and none could be installed — the
        corpus arrives through it and the checkpoints leave through it. Fix the network or
        install one by hand, then re-run."
fi
if rclone_version_lt "$RCLONE_HAVE" "$RCLONE_MIN"; then
    # Also reached when an old rclone earlier in PATH than /usr/local/bin won silently, which
    # would otherwise leave the 501s in place on a box whose log says it installed a new one.
    warn "rclone is still v$RCLONE_HAVE, below the $RCLONE_MIN floor — $(command -v rclone)."
    warn "expect a NotImplemented (501) on attempt 1 of every checkpoint upload (#107). Runs"
    warn "work; their sync logs just cannot be read for real failures. If /usr/local/bin/rclone"
    warn "is newer, something earlier in PATH is shadowing it."
else
    ok "rclone  v$RCLONE_HAVE (floor $RCLONE_MIN)"
fi

if [[ "$SKIP_CORPUS" -eq 1 ]]; then
    ok "corpus  skipped (--skip-corpus) — step 4 still checks whatever is on disk"
else
    [[ -n "$CORPUS_REMOTE" ]] \
        || die "no corpus remote: pass --corpus-remote or set DEMAE_CORPUS_REMOTE (see --help)"

    mkdir -p "$CORPUS_DIR"
    START=$(date +%s)
    # --checksum rather than the default size+mtime: compare *content hashes* and re-transfer
    # anything that disagrees. This is the whole reason the fetch is one command instead of a
    # hand-rolled download loop — the store keeps a hash per object, rclone will use it, and
    # that is a stronger check than any size comparison for no extra bytes on the wire.
    # Resumption falls out of the same property: a copy killed at 30 GB re-runs and moves only
    # what is missing or wrong, which is what makes this step idempotent.
    rclone copy "$CORPUS_REMOTE" "$CORPUS_DIR" \
        --checksum --transfers "$JOBS" --progress --stats-one-line \
        || die "rclone copy failed — nothing here has been trained on, so just re-run"
    ELAPSED=$(( $(date +%s) - START ))
    ok "corpus  copied in $((ELAPSED / 60))m $((ELAPSED % 60))s"

    if [[ "$RECHECK" -eq 1 ]]; then
        # Belt and braces, and honest about what it adds: `copy --checksum` already compared
        # every hash on the way in, so this only catches a disk that changed underneath us
        # since. Rare enough to be opt-in, cheap enough to offer — it moves no bytes.
        rclone check "$CORPUS_REMOTE" "$CORPUS_DIR" --checksum --one-way \
            || die "rclone check found differences after a clean copy"
        ok "recheck every object still hash-matches the bucket"
    fi
fi

# ---------------------------------------------------------------------------------------------
step "4/8  Corpus completeness — the question rclone does not answer"
# `rclone copy --checksum` proves the transfer was faithful. It cannot prove the *source* was
# whole: a bucket missing shard_00007 copies cleanly, reports success, and leaves this box
# training on less corpus than the ADR-0002 manifest describes — with `dataset_version` still
# reading correctly, because that hash is over the manifest, not over what is on disk. Shard
# gaps, the sidecars and the identity check live in one script that can equally be pointed at
# the bucket from the laptop *before* a box is rented, which is where an incomplete upload
# should be discovered.
CHECKER="$ROOT/scripts/check_corpus.py"
[[ -f "$CHECKER" ]] || die "no $CHECKER — rsync scripts/ across with the rest of the code"
python3 "$CHECKER" --local "$CORPUS_DIR" --expect-version "$EXPECT_VERSION" \
    || die "the corpus on this box is not the complete 500k tier"

# ---------------------------------------------------------------------------------------------
step "5/8  Committed artifacts (#77, #98, #99)"
# Paths mirror src/dataset/config.py, and the box mirrors the repo layout so that every
# `python -m pretrain.*` default resolves with no --corpus/--assignment/--blocks flags at all.
MISSING=0
for rel in \
    artifacts/phase1/split_scale_500k.csv \
    artifacts/phase1/split_blocks_500k.json \
    artifacts/phase1/val_panel_40.csv \
    artifacts/phase1/probe_set_500k.csv
do
    if [[ -f "$ROOT/$rel" ]]; then ok "$rel"; else echo "   MISSING $rel" >&2; MISSING=1; fi
done
[[ "$MISSING" -eq 0 ]] || die "rsync artifacts/ across — these are git-tracked, not in the bucket"

# ---------------------------------------------------------------------------------------------
step "6/8  Config and fingerprints resolve"
if [[ "$SKIP_ENV" -eq 1 ]]; then
    ok "skipped (--skip-env)"
else
    # #89's containment assertion (plan §15.2) fires when the loader is built, so a --dry-run
    # proves the split, the manifest and the corpus agree — and spends nothing doing it.
    (cd "$ROOT" && uv run python -m pretrain.launch --dry-run) || die "config/fingerprints failed"
fi

# §14 and §15.7 want the code version on every run, and this box has no `.git` — deliberately,
# since shipping it would stamp every run `-dirty` from a clean commit (see
# `dataset.provenance.SHA_FILE`). The laptop's answer rides in on the rsync instead. Checked
# here rather than left to the header because "git sha unknown" is legible only in an artifact
# nobody reads until the run it describes is over.
#
# Warned, not fatal: this costs a field on the run record, not the run, and a box that has
# already paid for its corpus should not be re-provisioned over a string.
if [[ -s "$ROOT/.git_sha" ]]; then
    ok "git sha $(head -1 "$ROOT/.git_sha" | tr -d '[:space:]')  (no .git here; from .git_sha)"
else
    warn "no $ROOT/.git_sha — every run on this box records 'git sha unknown' (§15.7)."
    warn "the code reached this box by some route that is not scripts/push_code.sh, which"
    warn "generates and ships it in the same breath. From the laptop, before starting a run:"
    warn "  scripts/push_code.sh --host demae-box --root $ROOT"
fi

# ---------------------------------------------------------------------------------------------
step "7/8  Throughput — §15 item 1, and it does NOT transfer between hosts"
# The loader >=2x gate is a fact about *this box's* NVMe, cores and free RAM, not about the
# recipe. Plan §15 reads as though there is one machine; there is a different one every session,
# so the gate is re-measured here rather than inherited from R0. #95's counters make it cheap.
#
# Deliberately the ViT-S default rather than --vit-base: §12 plans ViT-S at ~3,000 img/s against
# ViT-B's ~1,300, so the small model is the *stricter* test of the loader. A box that feeds
# ViT-S feeds ViT-B; the converse does not follow.
if [[ "$SKIP_ENV" -eq 1 || "$PROBE_STEPS" -eq 0 ]]; then
    ok "skipped"
else
    # `--no-checkpoints` and a disabled sink: since #106 the launcher is the *whole* run, so
    # left alone the probe would write §11 checkpoints into runs/R1 and open a W&B run for
    # something that is a measurement of this box rather than an experiment. What the probe
    # needs is the loop and the counters, which is what is left.
    (cd "$ROOT" && uv run python -m pretrain.launch --steps "$PROBE_STEPS" --log-every 50 \
        --no-checkpoints --set 'monitoring.wandb_mode="disabled"') \
        || die "the throughput probe did not complete"
    echo
    echo "   Read the footer above before starting a real run:"
    echo "     - loader wait  >~10% means the ADR-0003 headroom argument does not hold here"
    echo "     - recompiles   §7.3 expects zero; any is a moving shape costing the whole run"
    echo "     - attention    must be the sdpa path the config asked for, not a silent fallback"
fi

# ---------------------------------------------------------------------------------------------
step "8/8  Record what this box is"
# Every run carries git SHA, config hash, selection md5 and split md5 already (§15 item 7).
# The host is the axis that Phase 1's session-per-host reality adds: when a throughput number
# looks wrong six weeks from now, it needs to be attributable rather than reconstructed.
INSTANCE_ID="${CONTAINER_ID:-${VAST_CONTAINERLABEL:-unknown}}"
cat > "$ROOT/.box_identity" <<EOF
gpu=${GPU_NAME// /_}
gpu_count=${GPU_COUNT}
gpu_mem_mb=${GPU_MEM_MB}
cpus=${CPUS}
ram_gb=${RAM_GB}
vast_instance=${INSTANCE_ID}
image=${VAST_IMAGE:-vastai/pytorch}
dataset_version=${EXPECT_VERSION}
rclone=${RCLONE_HAVE}
net_mib_s=${NET_MIBPS:-unmeasured}
corpus_eta_min=${CORPUS_ETA_MIN:-unmeasured}
EOF
chmod 644 "$ROOT/.box_identity"
ok "wrote $ROOT/.box_identity — paste these into the W&B run tags"
cat "$ROOT/.box_identity" | sed 's/^/        /'

cat <<EOF

$(printf '\033[1m')Provisioned.$(printf '\033[0m')  Start a run from $ROOT, under tmux so it outlives the SSH session:

    cd $ROOT && tmux new -s r1
    uv run python -m pretrain.launch --vit-base --set 'monitoring.wandb_mode="online"'

That writes §11 checkpoints to $ROOT/runs/R1 (the run id), every epoch and every 45 minutes,
and it resumes from there if the box you are on is a replacement for one that vanished — the
same command either way. Start sync_checkpoints.sh against the same directory in a second
window; the trainer writes locally and nothing in src/pretrain pushes off-instance.

Before you destroy this box: confirm the last checkpoint is in the bucket AND re-read it from
there. Sync-then-verify-then-destroy — a self-destruct after an unverified sync loses the run.
EOF
