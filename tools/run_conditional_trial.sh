#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
trial_out=${1:-runs/event_policy_conditional_trial}
if [[ -e "$trial_out" ]] && [[ -n "$(ls -A -- "$trial_out")" ]]; then
  echo "输出目录非空，请指定新目录：bash tools/run_conditional_trial.sh runs/NEW_NAME" >&2
  exit 1
fi
mkdir -p -- "$trial_out"
echo "开始 20 轮条件生产计划实验，日志：$trial_out/train.log"
native/target/release/event-train \
  --out "$trial_out" \
  --init-from runs/event_policy_complete_sets_trial/best.json \
  --followup-scope single \
  --iterations 25 \
  --games-per-update 8 \
  --workers 7 \
  --device cuda \
  --branch-points 2 \
  --branch-steps-per-game 4320 \
  --max-branches-per-game 8 \
  --epochs 8 \
  --batch-size 64 \
  --learning-rate 0.0003 \
  --eval-every 5 \
  --eval-games 8 \
  --confirm-games 16 \
  --seed 480000 \
  --eval-seed 1980000000 \
  > "$trial_out/train.log" 2>&1
