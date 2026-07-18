#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
require_dir "$UPSTREAM_ROOT"

exec "$PYTHON_BIN" -m lingbot_vla_finetune.preflight \
  --upstream-root "$UPSTREAM_ROOT" \
  "$@"
