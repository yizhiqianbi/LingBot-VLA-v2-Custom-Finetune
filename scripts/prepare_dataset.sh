#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
require_env LINGBOT_SOURCE_DATASET_ROOT
require_env LINGBOT_TRAIN_DATASET_ROOT

exec "$PYTHON_BIN" -m lingbot_vla_finetune.prepare "$@"
