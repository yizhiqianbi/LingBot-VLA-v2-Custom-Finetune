#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

exec "$PYTHON_BIN" -m lingbot_vla_finetune.render \
  --output-root "$RUNTIME_ROOT" \
  --upstream-root "$UPSTREAM_ROOT" \
  --norm-stats-path "$NORM_STATS_PATH" \
  "$@"
