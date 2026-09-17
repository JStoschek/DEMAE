#!/usr/bin/env bash
# #165 — C3's label-efficiency sweep at 1k and 5k. Sixteen passes, on the box, one after another.
#
# §7's pass breakdown is the authority on which cells these are: "16 for the label-efficiency
# sweep (3 cells x 2 seeds x 2 sizes, plus A7)". The three frozen cells are A4-R5, A5 and A6 —
# R4 and ViT-S are both out, cut by §2.3 — and A7 carries its own four. `launch.resolve` now
# refuses anything else at a rung, so this list and the code agree by construction rather than
# by my having typed it correctly.
#
# Sequential, on #161's reasoning: one RTX 5080 holds one frozen ViT-B plus a head and not two.
#
# 1k before 5k, and every arm at a rung before the next rung starts. The cheap half of the sweep
# is therefore complete first, so a systematic failure — a missing checkpoint, an OOM — shows up
# in the first hour against four different arms rather than in the sixth hour against one.
set -uo pipefail

ROOT=/root/DEMAE
cd "$ROOT"
source /venv/main/bin/activate
export PYTHONPATH=src

COMMON=(
  --set "paths.corpus_dir=\"$ROOT/data/corpus_subset\""
  --set 'paths.demae_r5_checkpoint="/root/ckpt/demae_r5_encoder.pt"'
  --set 'paths.imagenet_mae_checkpoint="/root/ckpt/mae.pth"'
  --set 'paths.terramind_checkpoint="/root/ckpt/TerraMind_Tokenizer_DEM.pt"'
  --set 'runtime.device="cuda"'
  --set 'runtime.loader_workers=12'
)

# A7's §4.4 batch of 128 needs ~30.7 GiB of activations at a measured 0.24 GiB/sample (§7), and
# this is a 16 GB 5080. `runtime.micro_batch` splits the step and leaves the effective batch
# alone, so it moves memory and not the recipe — and `runtime` is outside `config_hash`, which is
# what makes that claim checkable rather than asserted: an A7 rung here carries the same recipe
# fingerprint it would carry on a 5090. The frozen arms are a ViT-B forward plus a 1.18M head and
# need none of this.
#
# 32 measured right: the A7 5k cells peaked at 11.5 GiB of the card's 16. (64 is what OOMs here.)
A7_EXTRA=(--set 'runtime.micro_batch=32')

STATUS="$ROOT/p165_status.log"
: > "$STATUS"

for size in 1k 5k; do
  for arm in A4-R5 A5 A6 A7; do
    for seed in 0 1; do
      extra=(); [ "$arm" = "A7" ] && extra=("${A7_EXTRA[@]}")
      log="/root/p165_${arm}_${size}_seed${seed}.log"
      echo "$(date -u +%FT%TZ)  START $arm $size seed $seed -> $log" | tee -a "$STATUS"
      if python -m benchmark.launch run --arm "$arm" --size "$size" --seed "$seed" \
           "${COMMON[@]}" ${extra+"${extra[@]}"} > "$log" 2>&1; then
        echo "$(date -u +%FT%TZ)  OK    $arm $size seed $seed" | tee -a "$STATUS"
      else
        # Recorded and not fatal, on #161's rule: a cell that dies takes its own rows and nothing
        # else, and stopping here would strand the rest on a box being paid for by the hour.
        echo "$(date -u +%FT%TZ)  FAIL  $arm $size seed $seed (see $log)" | tee -a "$STATUS"
      fi
    done
  done
done
echo "$(date -u +%FT%TZ)  SWEEP DONE" | tee -a "$STATUS"

# --- What this measured, 2026-09-16/17 on the rented 5080 -------------------------------------
# All 16 cells OK, no failures. 21:47:35Z -> 02:14Z, 4h27m wall-clock. Per cell, and the shape is
# the point: every cell scores the same 36,411 eval instances, so the ~5 min scoring leg is
# constant and only the training leg moves with data volume.
#
#   frozen 1k   9m46s     epoch 5.7s  = 2.1s train-work + 3.5s val
#   frozen 5k  17m00s     epoch 14.9s = 10.5s          + 3.5s
#   A7 1k      14m59s     epoch 11.1s = 5.6s           + 5.4s
#   A7 5k      33m52s     epoch 33.7s = 28.3s          + 5.4s
#
# The train-work column scales x5 between the rungs and the val column does not, which is the
# "varies only data volume" property showing up in the clock rather than only in the config hash.
#
# §7 prices this line at 5.4 5090-min for the three frozen cells and bundles A7's rungs into the
# A7 row. The frozen half really cost ~82 min of a 5080 and A7's rungs ~49 min — so the head
# primitive is still the transferred line §7 says it is, and it is still low. §7's own worst case
# (scaling the exposed 27% by A7's measured 4.5x) put the sweep at 10.5 min; the measurement is
# larger than that, and a re-derivation of the head primitive from these epoch times is the thing
# to do next rather than another transfer.
