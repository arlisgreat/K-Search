#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KSEARCH_ROOT="${KSEARCH_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
VENV_PATH="${VENV_PATH:-${KSEARCH_ROOT}/.venv}"

export KSEARCH_ROOT
export VENV_PATH

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${KSEARCH_ROOT}/.cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${XDG_CACHE_HOME}/uv}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${XDG_CACHE_HOME}/pip}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME}/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export WANDB_DIR="${WANDB_DIR:-${KSEARCH_ROOT}/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${XDG_CACHE_HOME}/wandb}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${XDG_CACHE_HOME}/wandb-config}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${XDG_CACHE_HOME}/triton}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${XDG_CACHE_HOME}/torch_extensions}"
export TMPDIR="${TMPDIR:-${KSEARCH_ROOT}/.tmp}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

mkdir -p \
  "${KSEARCH_ROOT}/data" \
  "${KSEARCH_ROOT}/vendor" \
  "${XDG_CACHE_HOME}" \
  "${UV_CACHE_DIR}" \
  "${PIP_CACHE_DIR}" \
  "${HF_HOME}" \
  "${WANDB_DIR}" \
  "${WANDB_CACHE_DIR}" \
  "${WANDB_CONFIG_DIR}" \
  "${TRITON_CACHE_DIR}" \
  "${TORCH_EXTENSIONS_DIR}" \
  "${TMPDIR}"

if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${VENV_PATH}/bin/activate"
fi

PYTHONPATH_PREFIX="${KSEARCH_ROOT}"
if [[ -d "${KSEARCH_ROOT}/vendor/flashinfer-bench-ksearch" ]]; then
  PYTHONPATH_PREFIX="${KSEARCH_ROOT}/vendor/flashinfer-bench-ksearch:${PYTHONPATH_PREFIX}"
fi

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${PYTHONPATH_PREFIX}:${PYTHONPATH}"
else
  export PYTHONPATH="${PYTHONPATH_PREFIX}"
fi
