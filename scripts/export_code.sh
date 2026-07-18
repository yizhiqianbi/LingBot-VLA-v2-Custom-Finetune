#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  printf 'Usage: %s OUTPUT.tar.gz\n' "$0" >&2
  exit 2
fi

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_PARENT="$(dirname -- "$PROJECT_ROOT")"
PROJECT_NAME="$(basename -- "$PROJECT_ROOT")"
OUTPUT="$(realpath -m -- "$1")"

if [[ "$OUTPUT" == "$PROJECT_ROOT" || "$OUTPUT" == "$PROJECT_ROOT/"* ]]; then
  printf 'Output archive must be outside the project directory: %s\n' "$OUTPUT" >&2
  exit 2
fi

mkdir -p -- "$(dirname -- "$OUTPUT")"
tar \
  --exclude="$PROJECT_NAME/.git" \
  --exclude="$PROJECT_NAME/.upstream" \
  --exclude="$PROJECT_NAME/work" \
  --exclude="$PROJECT_NAME/build" \
  --exclude="$PROJECT_NAME/dist" \
  --exclude="$PROJECT_NAME/data" \
  --exclude="$PROJECT_NAME/datasets" \
  --exclude="$PROJECT_NAME/models" \
  --exclude="$PROJECT_NAME/weights" \
  --exclude="$PROJECT_NAME/checkpoints" \
  --exclude="$PROJECT_NAME/output" \
  --exclude="$PROJECT_NAME/runs" \
  --exclude="$PROJECT_NAME/logs" \
  --exclude="$PROJECT_NAME/artifacts" \
  --exclude='*.egg-info' \
  --exclude='*/.pytest_cache' \
  --exclude='*/__pycache__' \
  --exclude='*.pyc' \
  --exclude='*.token' \
  --exclude='*token*.txt' \
  --exclude='*.safetensors' \
  --exclude='*.ckpt' \
  --exclude='*.pth' \
  --exclude='*.pt' \
  --exclude='*.bin' \
  --exclude='*.mp4' \
  --exclude='*.parquet' \
  --exclude='*.hdf5' \
  --exclude='*.h5' \
  -czf "$OUTPUT" \
  -C "$PROJECT_PARENT" \
  "$PROJECT_NAME"

printf '%s\n' "$OUTPUT"
