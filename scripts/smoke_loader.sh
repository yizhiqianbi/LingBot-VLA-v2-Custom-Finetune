#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
require_dir "$UPSTREAM_ROOT"

CONFIG="$RUNTIME_ROOT/take_wrong_item_right_arm.yaml"
require_file "$CONFIG"

exec "$PYTHON_BIN" -m lingbot_vla_finetune.smoke_loader \
  --upstream-root "$UPSTREAM_ROOT" \
  --training-config "$CONFIG" \
  "$@"
