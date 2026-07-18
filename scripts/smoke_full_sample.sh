#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
require_dir "$UPSTREAM_ROOT"

CONFIG="$RUNTIME_ROOT/take_wrong_item_right_arm.yaml"
require_file "$CONFIG"
require_file "$NORM_STATS_PATH"

SMOKE_ARGS=()
if [[ "${LINGBOT_ALLOW_UNCONFIRMED:-0}" == "1" ]]; then
  SMOKE_ARGS+=(--allow-unconfirmed)
fi

exec "$PYTHON_BIN" -m lingbot_vla_finetune.full_sample \
  --upstream-root "$UPSTREAM_ROOT" \
  --training-config "$CONFIG" \
  --norm-stats "$NORM_STATS_PATH" \
  "${SMOKE_ARGS[@]}" \
  "$@"
