#!/usr/bin/env bash
#
# Put one commit of this tree on a rented box, and make the box say which commit it is (#159).
#
# The failure this closes is a run record that names the wrong commit. #159's first three A6
# seeds recorded git 67f02de while training at a learning rate that commit did not hold: the box
# had a `.git_sha` written once at deploy, and code was rsynced over it afterwards without the
# file moving. The record was wrong in exactly the way a record exists to prevent, and nothing on
# the box could have noticed.
#
# Two rules follow, and this script is only useful because it enforces both:
#
#   1. A dirty tree is refused. `dataset.provenance.git_sha` would report `<sha>-dirty` if git
#      were present, but a worktree's `.git` is a *file* pointing into the parent repository, so
#      it cannot simply be shipped — the box reads the `.git_sha` fallback instead, and a
#      fallback cannot say "dirty". So the check moves here, where it can refuse.
#   2. `.git_sha` is written by the same invocation that ships the code, from the tree actually
#      being shipped. Re-running this script is how code is updated; there is no path that moves
#      one without the other.
#
# Everything it copies is verified by md5 on arrival, because a truncated 2.7 GB corpus and a
# complete one look identical to `run` until the numbers come out wrong.
#
# Usage:  scripts/p159_deploy.sh <ssh-host> <ssh-port> [remote-dir]
#
set -euo pipefail

HOST="${1:?usage: p159_deploy.sh <ssh-host> <ssh-port> [remote-dir]}"
PORT="${2:?usage: p159_deploy.sh <ssh-host> <ssh-port> [remote-dir]}"
REMOTE="${3:-/root/DEMAE}"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=30 -p ${PORT}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CORPUS="${CORPUS:-${HERE}/data/corpus_subset}"
BANK="${BANK:-${HERE}/data/task1/train_mask_bank.parquet}"

cd "${HERE}"

# --- Rule 1: the tree must be exactly one commit ------------------------------
if [ -n "$(git status --porcelain)" ]; then
  echo "refusing to deploy: the working tree has uncommitted changes." >&2
  echo "The box records its commit from a file, and a file cannot say 'dirty' —" >&2
  echo "so an uncommitted deploy produces a run record that names a commit whose" >&2
  echo "code is not what ran. Commit first (#159)." >&2
  git status --short >&2
  exit 1
fi
SHA="$(git rev-parse --short HEAD)"
echo "deploying ${SHA} to ${HOST}:${REMOTE}"

for p in "${CORPUS}" "${BANK}"; do
  [ -e "$p" ] || { echo "missing input: $p" >&2; exit 1; }
done

# --- The code, and the sha in the same breath ---------------------------------
$SSH "${HOST}" "mkdir -p ${REMOTE}/artifacts/task1 ${REMOTE}/data/task1 ${REMOTE}/data/corpus_subset"
rsync -az --delete \
  --exclude '.venv' --exclude 'data' --exclude 'artifacts' --exclude '.git' \
  --exclude '__pycache__' --exclude '.pytest_cache' --exclude '.ruff_cache' \
  --exclude '.DS_Store' --exclude '.release-work' \
  `# .claude holds this machine's agent worktrees — full sibling checkouts, each with its` \
  `# own data/ and artifacts/, 1.8 GB of them when #161 found this. Shipping it is slow, it` \
  `# fills the box's 63 GB disk with nothing that runs, and it puts other tickets' in-flight` \
  `# state on a rented host somebody else owns the hardware of. The list above was written` \
  `# when the directory did not exist.` \
  --exclude '.claude' \
  -e "${SSH}" ./ "${HOST}:${REMOTE}/"
# Written here, from the tree just shipped, and never anywhere else.
echo "${SHA}" | $SSH "${HOST}" "cat > ${REMOTE}/.git_sha"

# --- The committed inputs -----------------------------------------------------
rsync -az -e "${SSH}" artifacts/task1/ "${HOST}:${REMOTE}/artifacts/task1/"
rsync -az -e "${SSH}" "${BANK}" "${HOST}:${REMOTE}/data/task1/train_mask_bank.parquet"
rsync -a  -e "${SSH}" "${CORPUS}/" "${HOST}:${REMOTE}/data/corpus_subset/"

# --- The encoder, fetched on the box's own bandwidth --------------------------
MAE_URL="https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-1600e_in1k/mae_vit-base-p16_8xb512-fp16-coslr-1600e_in1k_20220825-f7569ca2.pth"
MAE_MD5="91ecc0345a6d6ecefe9c59041841f180"
$SSH "${HOST}" "mkdir -p /root/ckpt && [ -f /root/ckpt/mae.pth ] || wget -q -O /root/ckpt/mae.pth '${MAE_URL}'"

# --- Verify, because a truncated copy scores rather than failing --------------
echo "verifying on the box..."
$SSH "${HOST}" "cd ${REMOTE} && \
  echo \"  sha        \$(cat .git_sha)\" && \
  echo \"  bank       \$(md5sum data/task1/train_mask_bank.parquet | cut -d' ' -f1)\" && \
  echo \"  eval masks \$(md5sum artifacts/task1/eval_masks.parquet | cut -d' ' -f1)\" && \
  echo \"  encoder    \$(md5sum /root/ckpt/mae.pth | cut -d' ' -f1)\" && \
  echo \"  corpus     \$(du -sh data/corpus_subset | cut -f1)\""

cat <<EOF

expected:
  sha        ${SHA}
  bank       $(md5 -q "${BANK}")
  eval masks $(md5 -q artifacts/task1/eval_masks.parquet)
  encoder    ${MAE_MD5}

then, on the box:
  cd ${REMOTE} && source /venv/main/bin/activate && export PYTHONPATH=src
  python -m benchmark.launch run --arm A6 --dry-run \\
    --set 'paths.corpus_dir="${REMOTE}/data/corpus_subset"' \\
    --set 'paths.imagenet_mae_checkpoint="/root/ckpt/mae.pth"' \\
    --set 'runtime.device="cuda"' --set 'runtime.loader_workers=12'
EOF
