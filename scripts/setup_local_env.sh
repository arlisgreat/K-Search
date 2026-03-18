#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_env.sh"

if [[ ! -d "${VENV_PATH}" ]]; then
  echo "[setup] creating repo-local virtualenv: ${VENV_PATH}"
  uv venv --python 3.13 --system-site-packages "${VENV_PATH}"
fi

PYTHON_BIN="${VENV_PATH}/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[setup] failed to create Python interpreter at ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ -d "${KSEARCH_ROOT}/vendor/flashinfer-bench-ksearch" ]]; then
  echo "[setup] installing vendor/flashinfer-bench-ksearch into the repo-local venv"
  uv pip install --python "${PYTHON_BIN}" --no-deps -e "${KSEARCH_ROOT}/vendor/flashinfer-bench-ksearch"
else
  echo "[setup] vendor/flashinfer-bench-ksearch not found; run scripts/fetch_flashinfer_bench_ksearch.sh if you want the official fork inside this repo"
fi

"${PYTHON_BIN}" - <<'PY'
import importlib
mods = ["openai", "wandb", "yaml", "numpy", "torch", "triton", "flashinfer_bench", "huggingface_hub"]
missing = []
for name in mods:
    try:
        mod = importlib.import_module(name)
        origin = getattr(mod, "__file__", "built-in")
        print(f"[setup] {name}: OK {getattr(mod, '__version__', 'unknown')} ({origin})")
    except Exception as exc:
        missing.append((name, exc))
        print(f"[setup] {name}: FAIL {exc}")

if missing:
    raise SystemExit(1)
PY

echo "[setup] repo-local environment is ready"
echo "[setup] dataset target: ${KSEARCH_ROOT}/data/flashinfer-trace"
echo "[setup] artifacts root: ${KSEARCH_ROOT}/.ksearch-output"
