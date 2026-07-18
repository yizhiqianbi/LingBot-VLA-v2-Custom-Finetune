#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${LINGBOT_PYTHON:-python3}"
UPSTREAM_ROOT="${LINGBOT_UPSTREAM_ROOT:-$PROJECT_ROOT/.upstream/lingbot-vla-v2}"
RUNTIME_ROOT="${LINGBOT_RUNTIME_ROOT:-$PROJECT_ROOT/work/runtime}"
NORM_STATS_PATH="${LINGBOT_NORM_STATS:-$PROJECT_ROOT/work/norm_stats/take_wrong_item_right_arm.json}"

export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PROJECT_ROOT/src:$UPSTREAM_ROOT${PYTHONPATH:+:$PYTHONPATH}"

require_file() {
  if [[ ! -f "$1" ]]; then
    printf 'Required file is missing: %s\n' "$1" >&2
    exit 1
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    printf 'Required directory is missing: %s\n' "$1" >&2
    exit 1
  fi
}

require_env() {
  if [[ -z "${!1:-}" ]]; then
    printf 'Required environment variable is unset: %s\n' "$1" >&2
    exit 1
  fi
}
