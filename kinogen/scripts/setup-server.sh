#!/usr/bin/env bash
set -euo pipefail
KINOGEN_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export UV_CACHE_DIR="${KINOGEN_PROJECT_ROOT}/.cache/uv"
mkdir -p "${KINOGEN_PROJECT_ROOT}/.venvs" "${KINOGEN_PROJECT_ROOT}/.vscode" "${KINOGEN_PROJECT_ROOT}/outputs/previews"
if [[ ! -x "${KINOGEN_PROJECT_ROOT}/.venvs/tools/bin/python" ]]; then
  /root/.local/bin/uv venv --python /usr/bin/python3.12 "${KINOGEN_PROJECT_ROOT}/.venvs/tools"
fi
/root/.local/bin/uv pip sync --python "${KINOGEN_PROJECT_ROOT}/.venvs/tools/bin/python" "${KINOGEN_PROJECT_ROOT}/requirements-tools.lock.txt"
if [[ ! -x "${KINOGEN_PROJECT_ROOT}/.venvs/gpu/bin/python" ]]; then
  /root/.local/bin/uv venv --python /data/grail-envs/grail/bin/python --system-site-packages "${KINOGEN_PROJECT_ROOT}/.venvs/gpu"
fi
cp "${KINOGEN_PROJECT_ROOT}/configs/vscode-server-settings.json" "${KINOGEN_PROJECT_ROOT}/.vscode/settings.json"
# The GPU venv reuses the existing Torch binary without modifying GRAIL.
# This is a baseline for component checks, not an official KinoGen lockfile.
"${KINOGEN_PROJECT_ROOT}/.venvs/tools/bin/python" -c 'import requests, gdown; print("Server download tools: OK")'
"${KINOGEN_PROJECT_ROOT}/.venvs/gpu/bin/python" -c 'import torch; print("GPU baseline:", torch.__version__, "CUDA:", torch.version.cuda, "available:", torch.cuda.is_available())'
