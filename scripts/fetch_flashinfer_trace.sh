#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_env.sh"

TARGET_DIR="${TARGET_DIR:-${KSEARCH_ROOT}/data/flashinfer-trace}"
TRACE_URL="${FLASHINFER_TRACE_REPO_URL:-https://hf-mirror.com/datasets/flashinfer-ai/flashinfer-trace}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_PATH}/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

mkdir -p "${KSEARCH_ROOT}/data"

if command -v git-lfs >/dev/null 2>&1; then
  if [[ -d "${TARGET_DIR}/.git" ]]; then
    echo "[fetch] updating ${TARGET_DIR}"
    git -C "${TARGET_DIR}" pull --ff-only
  else
    echo "[fetch] cloning ${TRACE_URL} -> ${TARGET_DIR}"
    git clone "${TRACE_URL}" "${TARGET_DIR}"
  fi
else
  echo "[fetch] git-lfs not found; falling back to huggingface_hub snapshot_download via ${HF_ENDPOINT}"
  "${PYTHON_BIN}" - "${TARGET_DIR}" <<'PY'
import os
import sys
from huggingface_hub import snapshot_download

target_dir = sys.argv[1]
endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
snapshot_download(
    repo_id="flashinfer-ai/flashinfer-trace",
    repo_type="dataset",
    local_dir=target_dir,
    endpoint=endpoint,
)
PY
fi

echo "[fetch] dataset available at ${TARGET_DIR}"
