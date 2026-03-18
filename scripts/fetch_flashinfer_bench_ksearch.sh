#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_env.sh"

REPO_URL="${FLASHINFER_BENCH_KSEARCH_REPO_URL:-https://ghfast.top/https://github.com/caoshiyi/flashinfer-bench-ksearch.git}"
TARGET_DIR="${TARGET_DIR:-${KSEARCH_ROOT}/vendor/flashinfer-bench-ksearch}"

if [[ -d "${TARGET_DIR}/.git" ]]; then
  echo "[fetch] updating ${TARGET_DIR}"
  if ! git -C "${TARGET_DIR}" pull --ff-only; then
    echo "[fetch] git pull failed; using the existing local checkout"
  fi
else
  echo "[fetch] cloning ${REPO_URL} -> ${TARGET_DIR}"
  git clone --depth 1 "${REPO_URL}" "${TARGET_DIR}"
fi

if [[ -x "${VENV_PATH}/bin/python" ]]; then
  echo "[fetch] installing vendor package into ${VENV_PATH}"
  uv pip install --python "${VENV_PATH}/bin/python" --no-deps -e "${TARGET_DIR}"
fi
