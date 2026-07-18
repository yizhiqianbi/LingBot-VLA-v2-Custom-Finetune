#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
require_dir "$UPSTREAM_ROOT"
require_env LINGBOT_TRAIN_DATASET_ROOT
require_env CUDA_VISIBLE_DEVICES

"$PYTHON_BIN" -m lingbot_vla_finetune.render \
  --output-root "$RUNTIME_ROOT" \
  --upstream-root "$UPSTREAM_ROOT" \
  --norm-stats-path "$NORM_STATS_PATH" \
  --require-norm-stats

CONFIG="$RUNTIME_ROOT/take_wrong_item_right_arm.yaml"
require_file "$CONFIG"
require_file "$NORM_STATS_PATH"

"$PYTHON_BIN" -m lingbot_vla_finetune.preflight \
  --upstream-root "$UPSTREAM_ROOT" \
  --require-cuda

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export DISABLE_TELEMETRY=1
export PATH="$(dirname -- "$PYTHON_BIN"):$PATH"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

cd "$UPSTREAM_ROOT"
exec bash -o pipefail train.sh tasks/vla/train_lingbotvla.py "$CONFIG" "$@"
