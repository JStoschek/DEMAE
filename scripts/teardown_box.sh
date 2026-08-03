#!/usr/bin/env bash
# Prove the box holds nothing unique, then let it go — run this on the **laptop**, not the box.
#
#     ./scripts/teardown_box.sh --host demae-box --checkpoint-dir /workspace/DEMAE/runs/R1 \
#         --bucket demae-corpus --prefix checkpoints/r1/ \
#         --endpoint-url https://s3.us-west-004.backblazeb2.com
#
# Running here rather than on the box is the whole design. A teardown script that destroys the
# machine it is running on cannot report what it found, and — worse — would verify the bucket
# copy by asking the same box that wrote it. The check that means something is an *independent*
# one: this laptop, its own credentials, its own network path, re-reading the bytes the box
# claims it uploaded. §11's "the box holds nothing unique" is a claim, and this is the test.
#
# Report-only by default. `--destroy` is opt-in and still asks; a failed verification refuses to
# destroy no matter what flags are passed, because the one irreversible action here is the one
# with no undo. Sync, verify, *then* destroy — a self-destruct after an unverified sync is how
# a 43-hour R5 run disappears.
#
# On credentials, matching provision_box.sh's header: the box holds two scoped keys, one
# read-only on the corpus prefix and one write+read on the checkpoint prefix with **no delete
# capability**. The checkpoint key cannot be expiring or presigned — a run invents filenames
# over 43 hours — so what limits it is that it cannot remove anything. On B2 that is airtight
# (an overwrite makes a new version, the old survives); on R2/S3 it only holds with versioning
# enabled, because a plain PUT to an existing key overwrites destructively. Worth knowing here
# in particular: this script's whole job is to trust the bucket copy, and how much that copy
# can be tampered with by the host that wrote it is exactly the scoping question.

set -euo pipefail

HOST=""
ROOT="/workspace/DEMAE"
CKPT_DIR=""
BUCKET=""
PREFIX=""
ENDPOINT="${AWS_ENDPOINT_URL:-}"
REGION="${AWS_REGION:-us-east-1}"
PROFILE=""
HAUL_DIR=""
INSTANCE=""
DEEP_ALL=0
DO_SYNC=1
SYNC_CMD=""
DO_DESTROY=0
ASSUME_YES=0
FORCE=0

usage() {
    cat <<'EOF'
teardown_box.sh — verify the box holds nothing unique, then optionally destroy it.

  --host ALIAS          ssh alias for the box (required)
  --root DIR            repo root on the box (default: /workspace/DEMAE)
  --checkpoint-dir DIR  checkpoint directory *on the box* (required)
  --bucket NAME         bucket the checkpoints were synced to (required)
  --prefix KEY/         key prefix for this run's checkpoints (required)
  --endpoint-url URL    S3-compatible endpoint (B2/R2); omit for AWS S3
  --region / --profile  boto3 region / named profile
  --haul-dir DIR        where to land the retrieved artifacts (default: ./teardown-<instance>)
  --instance ID         vast instance id (default: read from the box's .box_identity)
  --deep-all            re-download and re-hash EVERY checkpoint, not just newest + milestones
  --no-sync             skip the catch-up sync; verify only
  --sync-cmd 'CMD'      the sync command to run on the box (default: rclone copy …)
  --destroy             destroy the instance if every check passes
  --yes                 skip the interactive confirmation (still requires every check to pass)
  --force               proceed even if training still looks alive (dangerous)
  -h, --help            this
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host) HOST="$2"; shift 2 ;;
        --root) ROOT="$2"; shift 2 ;;
        --checkpoint-dir) CKPT_DIR="$2"; shift 2 ;;
        --bucket) BUCKET="$2"; shift 2 ;;
        --prefix) PREFIX="$2"; shift 2 ;;
        --endpoint-url) ENDPOINT="$2"; shift 2 ;;
        --region) REGION="$2"; shift 2 ;;
        --profile) PROFILE="$2"; shift 2 ;;
        --haul-dir) HAUL_DIR="$2"; shift 2 ;;
        --instance) INSTANCE="$2"; shift 2 ;;
        --deep-all) DEEP_ALL=1; shift ;;
        --no-sync) DO_SYNC=0; shift ;;
        --sync-cmd) SYNC_CMD="$2"; shift 2 ;;
        --destroy) DO_DESTROY=1; shift ;;
        --yes) ASSUME_YES=1; shift ;;
        --force) FORCE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -n "$HOST" ]]      || { echo "--host is required" >&2; exit 2; }
[[ -n "$CKPT_DIR" ]]  || { echo "--checkpoint-dir is required" >&2; exit 2; }
[[ -n "$BUCKET" ]]    || { echo "--bucket is required" >&2; exit 2; }
[[ -n "$PREFIX" ]]    || { echo "--prefix is required" >&2; exit 2; }
[[ "$PREFIX" == */ ]] || PREFIX="$PREFIX/"

step()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
ok()    { printf '   ok    %s\n' "$*"; }
warn()  { printf '   \033[33mwarn\033[0m  %s\n' "$*"; }
die()   { printf '\n\033[31mSTOPPED: %s\033[0m\n' "$*" >&2; exit 1; }

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# ---------------------------------------------------------------------------------------------
step "1/7  Is anything still running?"
# First because it is the only check whose failure means "come back later" rather than "fix
# something". A checkpoint written 20 seconds after the inventory is a checkpoint this script
# never verified and would then have destroyed.
ALIVE="$(ssh "$HOST" "pgrep -af 'pretrain\.(launch|train)' || true" 2>/dev/null)"
if [[ -n "$ALIVE" ]]; then
    echo "$ALIVE" | sed 's/^/        /'
    if [[ "$FORCE" -eq 0 ]]; then
        die "training is still alive on $HOST. Let it reach a checkpoint and exit, or --force."
    fi
    warn "training is alive and --force was given — the inventory may be stale before it is used"
else
    ok "no pretrain process on $HOST"
fi

# ---------------------------------------------------------------------------------------------
step "2/7  Inventory what the box holds"
# sha256 computed *on the box*, over the bytes the box has. Every later comparison is against
# this list, so that a mismatch is attributable to the transfer rather than to a re-read.
ssh "$HOST" "cd '$CKPT_DIR' 2>/dev/null && ls checkpoint_*.pt >/dev/null 2>&1 && sha256sum checkpoint_*.pt || true" \
    > "$WORK/box_sums" 2>/dev/null
CKPT_N="$(wc -l < "$WORK/box_sums" | tr -d ' ')"
[[ "$CKPT_N" -gt 0 ]] || die "no checkpoint_*.pt in $CKPT_DIR on $HOST — wrong --checkpoint-dir?"
ok "$CKPT_N checkpoint(s)"
awk '{print "        " $2}' "$WORK/box_sums"

# Offline W&B runs are the quiet way to violate "the box holds nothing unique": the transaction
# log lives only here until `wandb sync` ships it, and wandb_mode defaults to offline (#97).
WANDB_DIRS="$(ssh "$HOST" "find '$ROOT' -maxdepth 4 -type d -name 'offline-run-*' 2>/dev/null || true")"
if [[ -n "$WANDB_DIRS" ]]; then
    warn "$(echo "$WANDB_DIRS" | wc -l | tr -d ' ') offline W&B run(s) — unique until synced"
    echo "$WANDB_DIRS" | sed 's/^/        /'
else
    ok "no unsynced offline W&B runs"
fi

[[ -n "$INSTANCE" ]] || INSTANCE="$(ssh "$HOST" "sed -n 's/^vast_instance=//p' '$ROOT/.box_identity' 2>/dev/null || true")"
[[ -n "$INSTANCE" ]] || INSTANCE="unknown"
[[ -n "$HAUL_DIR" ]] || HAUL_DIR="./teardown-${INSTANCE}"
mkdir -p "$HAUL_DIR"
ok "instance $INSTANCE, hauling to $HAUL_DIR"

# ---------------------------------------------------------------------------------------------
step "3/7  Catch-up sync (§11 syncs continuously; this is the safety net, not the mechanism)"
if [[ "$DO_SYNC" -eq 0 ]]; then
    ok "skipped (--no-sync)"
else
    # --s3-no-check-bucket for the same reason sync_checkpoints.sh passes it (#107): rclone's
    # bucket-existence probe is a create the box's Object Read & Write token cannot perform, and
    # R2 may answer it with NotImplemented (501). This is the *last* sync before a destroy, so a
    # spurious failure here is the most expensive one in the workflow.
    [[ -n "$SYNC_CMD" ]] || SYNC_CMD="rclone copy '$CKPT_DIR' 'bucket:$BUCKET/$PREFIX' --checksum --progress --s3-no-check-bucket"
    echo "   on $HOST: $SYNC_CMD"
    ssh "$HOST" "$SYNC_CMD" || die "the catch-up sync failed — nothing has been destroyed"
    ok "sync returned clean"
fi

# ---------------------------------------------------------------------------------------------
step "4/7  Haul the small unique things here"
# Small, so it is unconditional. These are the artifacts that exist nowhere else and cost
# seconds: the box identity the W&B run should be tagged with, and any offline run directory.
scp -q "$HOST:$ROOT/.box_identity" "$HAUL_DIR/box_identity" 2>/dev/null \
    && ok "box_identity" || warn "no .box_identity on the box (provision_box.sh writes it)"
if [[ -n "$WANDB_DIRS" ]]; then
    while IFS= read -r dir; do
        [[ -n "$dir" ]] || continue
        rsync -az "$HOST:$dir" "$HAUL_DIR/wandb/" && ok "$(basename "$dir")"
    done <<< "$WANDB_DIRS"
    echo "        sync them with: wandb sync $HAUL_DIR/wandb/offline-run-*"
fi

# ---------------------------------------------------------------------------------------------
step "5/7  Independent check — does the bucket really hold them?"
# From this laptop, with this laptop's credentials. The box is not consulted.
BUCKET="$BUCKET" PREFIX="$PREFIX" ENDPOINT="$ENDPOINT" REGION="$REGION" PROFILE="$PROFILE" \
python3 - "$WORK/box_sums" "$WORK/remote_meta.json" <<'PY' || die "bucket verification failed — nothing has been destroyed"
import json, os, sys
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

sums_path, out_path = sys.argv[1], sys.argv[2]
want = {}
for line in open(sums_path):
    digest, name = line.split(maxsplit=1)
    # sha256sum marks binary-mode entries with a leading '*'; strip it so a name never
    # reaches the bucket lookup as "*checkpoint_…" and reports a confusing miss.
    want[name.strip().lstrip("*")] = digest

endpoint = os.environ["ENDPOINT"] or None
# R2 has no AWS regions and wants the literal "auto" in the SigV4 credential scope. Getting it
# wrong raises SignatureDoesNotMatch, which reads like a wrong secret key.
region = os.environ["REGION"]
if endpoint and "r2.cloudflarestorage.com" in endpoint and region == "us-east-1":
    region = "auto"

session = boto3.Session(profile_name=os.environ["PROFILE"] or None)
s3 = session.client("s3", endpoint_url=endpoint, region_name=region)
bucket, prefix = os.environ["BUCKET"], os.environ["PREFIX"]

missing, meta = [], {}
for name in sorted(want):
    try:
        head = s3.head_object(Bucket=bucket, Key=prefix + name)
        meta[name] = head["ContentLength"]
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code in ("404", "NoSuchKey", "NotFound"):
            missing.append(name)
        else:
            print(f"  {name}: {exc}", file=sys.stderr); missing.append(name)
    except NoCredentialsError:
        print("no credentials on this laptop — set AWS_ACCESS_KEY_ID / "
              "AWS_SECRET_ACCESS_KEY or pass --profile", file=sys.stderr)
        sys.exit(2)

for name in missing:
    print(f"  NOT IN BUCKET: {name}", file=sys.stderr)
json.dump(meta, open(out_path, "w"))
print(f"   ok    {len(meta)}/{len(want)} checkpoint(s) present in s3://{bucket}/{prefix}")
sys.exit(1 if missing else 0)
PY

# ---------------------------------------------------------------------------------------------
step "6/7  Deep verify — re-read the bytes and hash them here"
# A HEAD proves a key exists at some length. It does not prove the content is the checkpoint
# the box wrote: a half-finished multipart upload, a retry that raced, or a corrupted PUT all
# leave an object of plausible size. Only re-reading and hashing settles it. The newest
# checkpoint is what a resume would load and every milestone is what §14 ships, so those are
# always deep-checked; --deep-all covers the rest at the cost of the bandwidth.
python3 - "$WORK/box_sums" <<'PY' > "$WORK/deep_list"
import re, sys
names = [l.split(maxsplit=1)[1].strip() for l in open(sys.argv[1])]
key = lambda n: tuple(int(x) for x in re.findall(r"_e(\d+)_s(\d+)", n)[0])
newest = max(names, key=key)
for n in sorted({newest, *[m for m in names if m.endswith("_milestone.pt")]}, key=key):
    print(n)
PY
if [[ "$DEEP_ALL" -eq 1 ]]; then
    awk '{print $2}' "$WORK/box_sums" > "$WORK/deep_list"
fi
DEEP_N="$(wc -l < "$WORK/deep_list" | tr -d ' ')"
echo "   deep-checking $DEEP_N of $CKPT_N checkpoint(s)"

BUCKET="$BUCKET" PREFIX="$PREFIX" ENDPOINT="$ENDPOINT" REGION="$REGION" PROFILE="$PROFILE" \
python3 - "$WORK/box_sums" "$WORK/deep_list" "$HAUL_DIR" <<'PY' || die "deep verification failed — nothing has been destroyed"
import hashlib, os, sys
import boto3

sums_path, deep_path, haul = sys.argv[1], sys.argv[2], sys.argv[3]
want = {}
for line in open(sums_path):
    digest, name = line.split(maxsplit=1)
    # sha256sum marks binary-mode entries with a leading '*'; strip it so a name never
    # reaches the bucket lookup as "*checkpoint_…" and reports a confusing miss.
    want[name.strip().lstrip("*")] = digest
deep = [l.strip() for l in open(deep_path) if l.strip()]

endpoint = os.environ["ENDPOINT"] or None
region = os.environ["REGION"]
if endpoint and "r2.cloudflarestorage.com" in endpoint and region == "us-east-1":
    region = "auto"  # see step 5 — R2's SigV4 credential scope

session = boto3.Session(profile_name=os.environ["PROFILE"] or None)
s3 = session.client("s3", endpoint_url=endpoint, region_name=region)
bucket, prefix = os.environ["BUCKET"], os.environ["PREFIX"]
os.makedirs(os.path.join(haul, "checkpoints"), exist_ok=True)

bad = []
for name in deep:
    dest = os.path.join(haul, "checkpoints", name)
    # Streamed to disk and hashed as it lands: these are ~1 GB each (ViT-B weights plus AdamW's
    # two moments), and a teardown should not need them all in memory at once.
    h = hashlib.sha256()
    with open(dest, "wb") as fh:
        body = s3.get_object(Bucket=bucket, Key=prefix + name)["Body"]
        for chunk in iter(lambda: body.read(8 << 20), b""):
            h.update(chunk); fh.write(chunk)
    got, expected = h.hexdigest(), want[name]
    mark = "ok   " if got == expected else "BAD  "
    print(f"   {mark} {name}  {got[:16]}…")
    if got != expected:
        bad.append(name)
        print(f"         box said {expected[:16]}… — the bucket copy is NOT what was written",
              file=sys.stderr)
print(f"   kept the deep-checked copies in {os.path.join(haul, 'checkpoints')}")
sys.exit(1 if bad else 0)
PY

# ---------------------------------------------------------------------------------------------
step "7/7  Verdict"
cat <<EOF
   checkpoints on box        $CKPT_N
   present in bucket         $CKPT_N
   re-read and hash-matched  $DEEP_N
   hauled to                 $HAUL_DIR
   instance                  $INSTANCE
EOF
if [[ -n "$WANDB_DIRS" ]]; then
    warn "offline W&B run(s) are in $HAUL_DIR/wandb — run 'wandb sync' before you rely on them"
fi

if [[ "$DO_DESTROY" -eq 0 ]]; then
    cat <<EOF

Every check passed. Nothing was destroyed — this run was report-only.
To destroy, re-run with --destroy, or do it yourself:

    vastai destroy instance $INSTANCE

Destroying is permanent: vast.ai container storage is deleted with the instance, not archived.
EOF
    exit 0
fi

[[ "$INSTANCE" != "unknown" ]] || die "--destroy needs an instance id; pass --instance"
command -v vastai >/dev/null || die "--destroy needs the vastai CLI on PATH (pip install vastai)"

if [[ "$ASSUME_YES" -eq 0 ]]; then
    printf '\nDestroy vast.ai instance %s permanently? Type the instance id to confirm: ' "$INSTANCE"
    read -r reply
    [[ "$reply" == "$INSTANCE" ]] || die "confirmation did not match — nothing was destroyed"
fi

vastai destroy instance "$INSTANCE"
echo "instance $INSTANCE destroyed. $HAUL_DIR holds everything that was unique to it."
