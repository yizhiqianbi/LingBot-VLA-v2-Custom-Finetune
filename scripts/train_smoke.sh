#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export LINGBOT_RUN_OUTPUT="${LINGBOT_SMOKE_OUTPUT:-$PROJECT_ROOT/work/training_smoke}"

exec "$PROJECT_ROOT/scripts/train.sh" \
  --train.max_steps 2 \
  --train.save_steps 2 \
  --train.num_train_epochs 1 \
  --train.use_compile false \
  "$@"
