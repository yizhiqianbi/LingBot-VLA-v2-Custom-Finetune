#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

require_dir "$UPSTREAM_ROOT"
require_dir "$RUNTIME_ROOT"
require_env LINGBOT_TRAIN_DATASET_ROOT
require_env CUDA_VISIBLE_DEVICES
require_dir "$LINGBOT_TRAIN_DATASET_ROOT"

UPSTREAM_ROOT="$(cd -- "$UPSTREAM_ROOT" && pwd)"
RUNTIME_ROOT="$(cd -- "$RUNTIME_ROOT" && pwd)"
DATASET_ROOT="$(cd -- "$LINGBOT_TRAIN_DATASET_ROOT" && pwd)"

EVAL_STEP="${LINGBOT_EVAL_STEP:-2000}"
if [[ -n "${LINGBOT_EVAL_MODEL_PATH:-}" ]]; then
  MODEL_PATH="$LINGBOT_EVAL_MODEL_PATH"
else
  require_env LINGBOT_RUN_OUTPUT
  MODEL_PATH="$LINGBOT_RUN_OUTPUT/checkpoints/global_step_${EVAL_STEP}/hf_ckpt"
fi

require_dir "$MODEL_PATH"
MODEL_PATH="$(cd -- "$MODEL_PATH" && pwd)"
require_file "$MODEL_PATH/model.safetensors.index.json"
require_file "$NORM_STATS_PATH"
require_file "$RUNTIME_ROOT/configs/robot_configs/take_wrong_item_right_arm.yaml"
NORM_STATS_PATH="$(cd -- "$(dirname -- "$NORM_STATS_PATH")" && pwd)/$(basename -- "$NORM_STATS_PATH")"

shopt -s nullglob
MODEL_SHARDS=("$MODEL_PATH"/model-*.safetensors)
shopt -u nullglob
if (( ${#MODEL_SHARDS[@]} == 0 )); then
  printf 'No model safetensor shards found in: %s\n' "$MODEL_PATH" >&2
  exit 1
fi

RUN_ROOT="$(cd -- "$MODEL_PATH/../../.." && pwd)"
require_file "$RUN_ROOT/lingbotvla_cli.yaml"

export QWEN3VL_PATH="${QWEN3VL_PATH:-${LINGBOT_TOKENIZER_PATH:-}}"
require_env QWEN3VL_PATH
require_dir "$QWEN3VL_PATH"
QWEN3VL_PATH="$(cd -- "$QWEN3VL_PATH" && pwd)"
export QWEN3VL_PATH

read -r -a TRAJ_IDS <<< "${LINGBOT_EVAL_TRAJ_IDS:-0 10 20 30 43}"
if (( ${#TRAJ_IDS[@]} == 0 )); then
  printf 'LINGBOT_EVAL_TRAJ_IDS must contain at least one episode id.\n' >&2
  exit 1
fi

USE_LENGTH="${LINGBOT_EVAL_USE_LENGTH:-50}"
MAX_INFER_TIME="${LINGBOT_EVAL_MAX_INFER_TIME:-3}"
if ! [[ "$USE_LENGTH" =~ ^[1-9][0-9]*$ ]]; then
  printf 'LINGBOT_EVAL_USE_LENGTH must be a positive integer: %s\n' "$USE_LENGTH" >&2
  exit 1
fi
if ! [[ "$MAX_INFER_TIME" =~ ^[1-9][0-9]*$ ]]; then
  printf 'LINGBOT_EVAL_MAX_INFER_TIME must be a positive integer: %s\n' "$MAX_INFER_TIME" >&2
  exit 1
fi

CHECKPOINT_NAME="$(basename -- "$(dirname -- "$MODEL_PATH")")"
EVAL_OUTPUT="${LINGBOT_EVAL_OUTPUT:-$RUN_ROOT/eval/open_loop_${CHECKPOINT_NAME}}"
mkdir -p "$EVAL_OUTPUT"
EVAL_OUTPUT="$(cd -- "$EVAL_OUTPUT" && pwd)"

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export DISABLE_TELEMETRY=1
export MPLBACKEND=Agg

printf 'Model: %s\n' "$MODEL_PATH"
printf 'Dataset: %s\n' "$DATASET_ROOT"
printf 'Episodes: %s\n' "${TRAJ_IDS[*]}"
printf 'Output: %s\n' "$EVAL_OUTPUT"

# The upstream policy resolves configs/robot_configs relative to the current
# directory, so evaluation must run from the rendered runtime root.
cd "$RUNTIME_ROOT"
"$PYTHON_BIN" "$UPSTREAM_ROOT/scripts/open_loop_eval.py" \
  --model_path "$MODEL_PATH" \
  --policy auto \
  --robo_name take_wrong_item_right_arm \
  --norm_path "$NORM_STATS_PATH" \
  --data_path "$DATASET_ROOT" \
  --traj_ids "${TRAJ_IDS[@]}" \
  --use_length "$USE_LENGTH" \
  --chunk_ret true \
  --max_infer_time "$MAX_INFER_TIME" \
  --use_bf16 \
  --save_plot_path "$EVAL_OUTPUT" \
  "$@" 2>&1 | tee "$EVAL_OUTPUT/eval.log"
