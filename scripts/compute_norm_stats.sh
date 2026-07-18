#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
require_dir "$UPSTREAM_ROOT"
require_env LINGBOT_TRAIN_DATASET_ROOT

RENDER_ARGS=()
if [[ "${LINGBOT_ALLOW_UNCONFIRMED:-0}" == "1" ]]; then
  RENDER_ARGS+=(--allow-unconfirmed)
fi

"$PYTHON_BIN" -m lingbot_vla_finetune.render \
  --output-root "$RUNTIME_ROOT" \
  --upstream-root "$UPSTREAM_ROOT" \
  --norm-stats-path "$NORM_STATS_PATH" \
  "${RENDER_ARGS[@]}"

CONFIG="$RUNTIME_ROOT/take_wrong_item_right_arm.yaml"
MANIFEST="$RUNTIME_ROOT/runtime_manifest.json"
require_file "$CONFIG"
require_file "$MANIFEST"
mkdir -p -- "$(dirname -- "$NORM_STATS_PATH")"

NORM_ARGS=()
if [[ "${LINGBOT_ALLOW_UNCONFIRMED:-0}" == "1" ]]; then
  NORM_ARGS+=(--allow-unconfirmed)
fi

"$PYTHON_BIN" -m lingbot_vla_finetune.norm_stats \
  --upstream-root "$UPSTREAM_ROOT" \
  --training-config "$CONFIG" \
  --runtime-manifest "$MANIFEST" \
  --output "$NORM_STATS_PATH" \
  --num-workers "${NORM_WORKERS:-4}" \
  --batch-size "${NORM_BATCH_SIZE:-128}" \
  "${NORM_ARGS[@]}"

printf '%s\n' "$NORM_STATS_PATH"
