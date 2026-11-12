#!/usr/bin/env bash
set -euo pipefail
KINOGEN_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export UV_CACHE_DIR="${KINOGEN_PROJECT_ROOT}/.cache/uv"
export PYTHONDONTWRITEBYTECODE=1
export PIP_CACHE_DIR="${KINOGEN_PROJECT_ROOT}/.cache/pip"
export PIP_REQUIRE_VIRTUALENV=true
KINOGEN_PIP_SOURCE=(--no-index --find-links "${KINOGEN_PROJECT_ROOT}/downloads/runtime-wheels")
KINOGEN_PROFILES=(kimodo vision sam3)
if (( $# )); then KINOGEN_PROFILES=("$@"); fi
for KINOGEN_PROFILE in "${KINOGEN_PROFILES[@]}"; do
  case "${KINOGEN_PROFILE}" in kimodo|vision|sam3) ;; *) exit 2 ;; esac
  KINOGEN_ENV="${KINOGEN_PROJECT_ROOT}/.venvs/${KINOGEN_PROFILE}"
  if [[ ! -x "${KINOGEN_ENV}/bin/python" ]]; then
    /root/.local/bin/uv venv --python /data/grail-envs/grail/bin/python --system-site-packages "${KINOGEN_ENV}"
  fi
  # pip accounts for inherited site-packages; uv pip's resolver does not.
  # All new/overridden packages are written into this venv, never the base.
  KINOGEN_CONSTRAINTS="${KINOGEN_PROJECT_ROOT}/configs/runtime-constraints.txt"
  if [[ "${KINOGEN_PROFILE}" == sam3 ]]; then
    KINOGEN_CONSTRAINTS="${KINOGEN_PROJECT_ROOT}/configs/sam3-runtime-constraints.txt"
  fi
  "${KINOGEN_ENV}/bin/python" -m pip install -q "${KINOGEN_PIP_SOURCE[@]}" -c "${KINOGEN_CONSTRAINTS}" -r "${KINOGEN_PROJECT_ROOT}/requirements-${KINOGEN_PROFILE}-runtime.txt"
  "${KINOGEN_ENV}/bin/python" -c 'import torch; assert torch.__version__ == "2.5.1+cu121"; print("Component environment Torch:", torch.__version__)'
done
mkdir -p "${KINOGEN_PROJECT_ROOT}/outputs/previews"
