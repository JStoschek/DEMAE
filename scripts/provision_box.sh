#!/usr/bin/env bash
# Bring a freshly rented vast.ai box to the line plan §15 gates R1 at — run it *on the box*,
# streamed over the SSH connection so it never lands on that disk as a file:
#
#     rsync -az --delete src tests artifacts scripts pyproject.toml uv.lock README.md \
#         demae-box:/workspace/DEMAE/
#     ssh demae-box 'bash -s -- --root /workspace/DEMAE' < scripts/provision_box.sh
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
MIN_FREE_GB=70     # 47 GB corpus + venv + the checkpoints a run will write
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
  --probe-steps N    steps for the throughput probe, 0 to skip (default: 200)
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
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

CORPUS_DIR="$ROOT/data/corpus_500k"

step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
ok()   { printf '   ok   %s\n' "$*"; }
warn() { printf '\033[33m   warn %s\033[0m\n' "$*" >&2; }
die()  { printf '\n\033[31mFAILED: %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------------------------
step "1/8  Preflight — is this box worth provisioning?"
# Deliberately first and deliberately fatal. A wrong host should cost 30 seconds, not the 20
# minutes it takes to discover the corpus does not fit after it has half-downloaded.

command -v nvidia-smi >/dev/null || die "no nvidia-smi — this is not a GPU box"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
GPU_MEM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)"
GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
ok "GPU     ${GPU_COUNT}x ${GPU_NAME} (${GPU_MEM_MB} MiB)"
[[ "$GPU_COUNT" -eq 1 ]] || echo "   note   plan §11 assumes a single GPU; $GPU_COUNT present"

RAM_GB="$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo)"
[[ "$RAM_GB" -ge "$MIN_RAM_GB" ]] \
    || die "RAM ${RAM_GB} GB < ${MIN_RAM_GB} GB — the 47 GB corpus will not page-cache (§11)"
ok "RAM     ${RAM_GB} GB"

CPUS="$(nproc)"
ok "CPUs    ${CPUS} (the loader wants workers; §7.5 names IO the #1 risk)"

mkdir -p "$ROOT"
FREE_GB="$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')"
[[ "$FREE_GB" -ge "$MIN_FREE_GB" ]] \
    || die "only ${FREE_GB} GB free at $ROOT; need >= ${MIN_FREE_GB} GB"
ok "Disk    ${FREE_GB} GB free at $ROOT"

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
step "3/8  Corpus — rclone copy"
if [[ "$SKIP_CORPUS" -eq 1 ]]; then
    ok "skipped (--skip-corpus) — step 4 still checks whatever is on disk"
else
    [[ -n "$CORPUS_REMOTE" ]] \
        || die "no corpus remote: pass --corpus-remote or set DEMAE_CORPUS_REMOTE (see --help)"
    if ! command -v rclone >/dev/null; then
        echo "   installing rclone…"
        curl -fsSL https://rclone.org/install.sh | bash >/dev/null || die "rclone install failed"
    fi
    ok "rclone  $(rclone version | head -1)"

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
    warn "from the laptop, before starting a run:"
    warn "  uv run python -c 'from dataset.provenance import git_sha; print(git_sha())' > .git_sha"
    warn "  rsync -az .git_sha demae-box:$ROOT/"
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
    (cd "$ROOT" && uv run python -m pretrain.launch --steps "$PROBE_STEPS" --log-every 50) \
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
EOF
chmod 644 "$ROOT/.box_identity"
ok "wrote $ROOT/.box_identity — paste these into the W&B run tags"
cat "$ROOT/.box_identity" | sed 's/^/        /'

cat <<EOF

$(printf '\033[1m')Provisioned.$(printf '\033[0m')  Start a run from $ROOT, under tmux so it outlives the SSH session:

    cd $ROOT && tmux new -s r1
    uv run python -m pretrain.launch --vit-base --set 'monitoring.wandb_mode="online"'

Before you destroy this box: confirm the last checkpoint is in the bucket AND re-read it from
there. Sync-then-verify-then-destroy — a self-destruct after an unverified sync loses the run.
EOF
