#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_env.sh"

# GPUMode TriMul launcher for K-Search (world-model mode).
#
# Common:
# - KSEARCH_ROOT: repo root (default: auto-detected)
# - MODEL_NAME: LLM model name
# - LLM_API_KEY or API_KEY: OpenAI-compatible API key
# - BASE_URL: OpenAI-compatible base URL
#
# Task/generation:
# - LANGUAGE: triton|python|cuda (default: triton)
# - TARGET_GPU: e.g. H100 (default: H100)
# - MAX_OPT_ROUNDS: default 300
# - ARTIFACTS_DIR: default ${KSEARCH_ROOT}/.ksearch-output/gpumode
# - CHECKPOINT_DIR: default ${KSEARCH_ROOT}/checkpoints
# - RUN_ID: optional explicit checkpoint run id
# - RUN_LABEL: optional human-readable checkpoint label
# - CONTINUE_FROM_SOLUTION: optional solution name or path to persisted solution JSON
# - CONTINUE_FROM_RUN_ID: optional checkpoint run id to resume from
# - CONTINUE_FROM_ROUND: optional round number inside CONTINUE_FROM_RUN_ID (default: latest)
#
# World model:
# - WM: 1 to enable world model (default: 1)
# - WM_STAGNATION_WINDOW: default 5
# - WORLD_MODEL_JSON: optional path or "auto"
#
# GPUMode:
# - GPUMODE_MODE: benchmark|test|profile|leaderboard (default: leaderboard)
# - GPUMODE_KEEP_TMP: 1 to keep temp dirs (default: 0)
# - GPUMODE_TASK_DIR: override task dir (default: vendored trimul task)
#
# Optional W&B:
# - WANDB: 1 to enable (default: 0)
# - WANDB_PROJECT, RUN_NAME, WANDB_API_KEY

MODEL_NAME="${MODEL_NAME:-gpt-5.2}"
API_KEY="${API_KEY:-${LLM_API_KEY:-}}"
BASE_URL="${BASE_URL:-https://us.api.openai.com/v1}"

LANGUAGE="${LANGUAGE:-triton}"
TARGET_GPU="${TARGET_GPU:-H100}"
MAX_OPT_ROUNDS="${MAX_OPT_ROUNDS:-300}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-${KSEARCH_ROOT}/.ksearch-output/gpumode}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${KSEARCH_ROOT}/checkpoints}"
RUN_ID="${RUN_ID:-}"
RUN_LABEL="${RUN_LABEL:-}"
CONTINUE_FROM_SOLUTION="${CONTINUE_FROM_SOLUTION:-}"
CONTINUE_FROM_RUN_ID="${CONTINUE_FROM_RUN_ID:-}"
CONTINUE_FROM_ROUND="${CONTINUE_FROM_ROUND:-}"

WM="${WM:-1}"
WM_STAGNATION_WINDOW="${WM_STAGNATION_WINDOW:-5}"
WORLD_MODEL_JSON="${WORLD_MODEL_JSON:-}"

GPUMODE_MODE="${GPUMODE_MODE:-leaderboard}"
GPUMODE_KEEP_TMP="${GPUMODE_KEEP_TMP:-0}"
GPUMODE_TASK_DIR="${GPUMODE_TASK_DIR:-${KSEARCH_ROOT}/k_search/tasks/gpu_mode/trimul}"

WANDB="${WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-k-search}"
RUN_NAME="${RUN_NAME:-${MODEL_NAME}-${LANGUAGE}-gpumode-trimul-wm-opt${MAX_OPT_ROUNDS}}"
if [[ -z "${RUN_LABEL}" ]]; then
  RUN_LABEL="${RUN_NAME}"
fi

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${VENV_PATH}/bin/python}"
  if [[ ! -x "${PYTHON_BIN}" ]]; then
    PYTHON_BIN="$(command -v python3)"
  fi
  exec "${PYTHON_BIN}" -u "${KSEARCH_ROOT}/generate_kernels_and_eval.py" --help
fi

if [[ -z "${MODEL_NAME}" ]]; then
  echo "ERROR: MODEL_NAME is required" >&2
  exit 2
fi
if [[ -z "${API_KEY}" ]]; then
  echo "ERROR: API key is required (set LLM_API_KEY or API_KEY)" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-${VENV_PATH}/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

export WANDB_API_KEY="${WANDB_API_KEY:-}"

ARGS=(
  --task-source gpumode
  --model-name "${MODEL_NAME}"
  --api-key "${API_KEY}"
  --base-url "${BASE_URL}"
  --language "${LANGUAGE}"
  --target-gpu "${TARGET_GPU}"
  --max-opt-rounds "${MAX_OPT_ROUNDS}"
  --save-solutions
  --artifacts-dir "${ARTIFACTS_DIR}"
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --gpumode-mode "${GPUMODE_MODE}"
  --gpumode-task-dir "${GPUMODE_TASK_DIR}"
)

if [[ -n "${RUN_ID}" ]]; then
  ARGS+=(--run-id "${RUN_ID}")
fi
if [[ -n "${RUN_LABEL}" ]]; then
  ARGS+=(--run-label "${RUN_LABEL}")
fi
if [[ -n "${CONTINUE_FROM_SOLUTION}" ]]; then
  ARGS+=(--continue-from-solution "${CONTINUE_FROM_SOLUTION}")
fi
if [[ -n "${CONTINUE_FROM_RUN_ID}" ]]; then
  ARGS+=(--continue-from-run-id "${CONTINUE_FROM_RUN_ID}")
fi
if [[ -n "${CONTINUE_FROM_ROUND}" ]]; then
  ARGS+=(--continue-from-round "${CONTINUE_FROM_ROUND}")
fi
if [[ "${WM}" == "1" ]]; then
  ARGS+=(--world-model --wm-stagnation-window "${WM_STAGNATION_WINDOW}")
fi
if [[ -n "${WORLD_MODEL_JSON}" ]]; then
  ARGS+=(--continue-from-world-model "${WORLD_MODEL_JSON}")
fi
if [[ "${GPUMODE_KEEP_TMP}" == "1" ]]; then
  ARGS+=(--gpumode-keep-tmp)
fi
if [[ "${WANDB}" == "1" ]]; then
  ARGS+=(--wandb --run-name "${RUN_NAME}" --wandb-project "${WANDB_PROJECT}")
fi

exec "${PYTHON_BIN}" -u "${KSEARCH_ROOT}/generate_kernels_and_eval.py" "${ARGS[@]}" "$@"
