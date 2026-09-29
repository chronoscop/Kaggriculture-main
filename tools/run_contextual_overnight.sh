#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
trial_out=${1:-runs/event_policy_contextual_overnight}
trial_hours=${2:-10}
if [[ ! "$trial_hours" =~ ^[1-9][0-9]*$ ]] || (( trial_hours > 72 )); then
  echo '用法：bash tools/run_contextual_overnight.sh NEW_DIR HOURS（1–72小时）' >&2
  exit 1
fi
if [[ -e "$trial_out" ]] && [[ ! -d "$trial_out" || -n "$(ls -A -- "$trial_out")" ]]; then
  echo "输出目录必须为空，请换新目录：$trial_out" >&2
  exit 1
fi
trial_init=${INIT_FROM:-runs/event_policy_conditional_trial/best.json}
[[ -f "$trial_init" ]] || { echo "找不到已验收检查点：$trial_init" >&2; exit 1; }
if ! command -v cargo >/dev/null 2>&1; then
  export PATH="/tmp/route-rl-rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin:$PATH"
  export CARGO_HOME="${CARGO_HOME:-/tmp/route-rl-cargo}"
fi
cargo build --manifest-path native/Cargo.toml --release --features train --offline --bin event-train --bin event-agent -j 4
mkdir -p -- "$trial_out"
echo "开始约 $trial_hours 小时实验，完成当前轮后保存退出；日志：$trial_out/train.log"
echo "初始化来源为已验收策略：$trial_init；旧提议网络、经验和 Adam 不带入。"
exec native/target/release/event-train \
  --out "$trial_out" \
  --init-from "$trial_init" \
  --iterations 10000 \
  --max-seconds "$((trial_hours * 3600))" \
  --games-per-update 16 \
  --workers "${WORKERS:-7}" \
  --device cuda \
  --branch-points 2 \
  --branch-steps-per-game 8640 \
  --max-branches-per-game 12 \
  --epochs 8 \
  --batch-size 64 \
  --learning-rate 0.0003 \
  --eval-every 5 \
  --eval-games 8 \
  --confirm-games 16 \
  --seed 560000 \
  --eval-seed 1995000000 \
  > "$trial_out/train.log" 2>&1
