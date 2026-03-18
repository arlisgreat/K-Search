#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_env.sh"

# FlashInfer MLA decode launcher for K-Search (world-model mode).
#
# Useful env vars:
# - DATASET_ROOT: flashinfer-trace root inside this repo
# - CHECKPOINT_DIR: default ${KSEARCH_ROOT}/checkpoints
# - RUN_ID: optional explicit checkpoint run id
# - RUN_LABEL: optional human-readable checkpoint label
# - CONTINUE_FROM_SOLUTION: optional solution name or path
# - CONTINUE_FROM_RUN_ID: optional checkpoint run id to resume from
# - CONTINUE_FROM_ROUND: optional round number inside CONTINUE_FROM_RUN_ID (default: latest)
# - WORLD_MODEL_JSON: optional world-model snapshot path or "auto"
# - WANDB / WANDB_PROJECT / RUN_NAME: optional Weights & Biases logging

DATASET_ROOT="${DATASET_ROOT:-${KSEARCH_ROOT}/data/flashinfer-trace}"

MODEL_NAME="${MODEL_NAME:-gemini-3-pro-preview}"
API_KEY="${API_KEY:-${LLM_API_KEY:-}}"
BASE_URL="${BASE_URL:-https://generativelanguage.googleapis.com/v1beta/}"

DEFINITION="${DEFINITION:-mla_paged_decode_h16_ckv512_kpe64_ps1}"
LANGUAGE="${LANGUAGE:-cuda}"
TARGET_GPU="${TARGET_GPU:-H100}"

BASELINE_SOLUTION="${BASELINE_SOLUTION:-flashinfer_wrapper_03f7b0}"
CONTINUE_FROM_SOLUTION="${CONTINUE_FROM_SOLUTION:-}"
WORLD_MODEL_JSON="${WORLD_MODEL_JSON:-}"

MAX_OPT_ROUNDS="${MAX_OPT_ROUNDS:-20}"
WM="${WM:-1}"
WM_STAGNATION_WINDOW="${WM_STAGNATION_WINDOW:-7}"

ARTIFACTS_DIR="${ARTIFACTS_DIR:-${KSEARCH_ROOT}/.ksearch-output/flashinfer}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${KSEARCH_ROOT}/checkpoints}"
RUN_ID="${RUN_ID:-}"
RUN_LABEL="${RUN_LABEL:-}"
CONTINUE_FROM_RUN_ID="${CONTINUE_FROM_RUN_ID:-}"
CONTINUE_FROM_ROUND="${CONTINUE_FROM_ROUND:-}"

WANDB="${WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-k-search}"
RUN_NAME="${RUN_NAME:-${MODEL_NAME}-${LANGUAGE}-wm-${DEFINITION}-seed-opt${MAX_OPT_ROUNDS}}"
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

if [[ -z "${API_KEY}" ]]; then
  echo "ERROR: API key is required (set LLM_API_KEY or API_KEY)" >&2
  exit 2
fi
if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "ERROR: DATASET_ROOT does not exist: ${DATASET_ROOT}" >&2
  echo "Run scripts/fetch_flashinfer_trace.sh to download the dataset into this repo." >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-${VENV_PATH}/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

export WANDB_API_KEY="${WANDB_API_KEY:-}"

ARGS=(
  --local "${DATASET_ROOT}"
  --task-source flashinfer
  --task-path "${DATASET_ROOT}"
  --definition "${DEFINITION}"
  --model-name "${MODEL_NAME}"
  --api-key "${API_KEY}"
  --base-url "${BASE_URL}"
  --language "${LANGUAGE}"
  --target-gpu "${TARGET_GPU}"
  --wm-stagnation-window "${WM_STAGNATION_WINDOW}"
  --max-opt-rounds "${MAX_OPT_ROUNDS}"
  --parallel-workloads
  --save-solutions
  --use-isolated-runner
  --artifacts-dir "${ARTIFACTS_DIR}"
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --feedback-workloads
  bd2dae14-7bae-4edb-964f-2163accf506e
  84221f45-78f8-4d44-84f6-998153d2c1fa
  d0da33e2-2d94-42b5-be8a-09111f9f2649
  e417264f-195d-4204-89fa-3ebdb539f1cf
  939f995a-1ab2-4d19-8d94-50f07e73542d
)

if [[ -n "${RUN_ID}" ]]; then
  ARGS+=(--run-id "${RUN_ID}")
fi
if [[ -n "${RUN_LABEL}" ]]; then
  ARGS+=(--run-label "${RUN_LABEL}")
fi
if [[ "${WM}" == "1" ]]; then
  ARGS+=(--world-model)
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
if [[ -n "${WORLD_MODEL_JSON}" ]]; then
  ARGS+=(--continue-from-world-model "${WORLD_MODEL_JSON}")
fi
if [[ -n "${BASELINE_SOLUTION}" ]]; then
  ARGS+=(--baseline-solution "${BASELINE_SOLUTION}")
fi
if [[ "${WANDB}" == "1" ]]; then
  ARGS+=(--wandb --wandb-project "${WANDB_PROJECT}" --run-name "${RUN_NAME}")
fi

exec "${PYTHON_BIN}" -u "${KSEARCH_ROOT}/generate_kernels_and_eval.py" "${ARGS[@]}" "$@"
