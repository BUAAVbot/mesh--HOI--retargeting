#!/usr/bin/env bash
set -euo pipefail
KINOGEN_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "${KINOGEN_PROJECT_ROOT}/scripts/activate-server.sh" gpu
KINOGEN_PYBIND_DIR="$(python -m pybind11 --cmakedir)"
cmake -S "${KINOGEN_PROJECT_ROOT}/dependencies/foundationpose/mycpp" \
  -B "${KINOGEN_PROJECT_ROOT}/.build/foundationpose-mycpp" \
  -DCMAKE_BUILD_TYPE=Release \
  -Dpybind11_DIR="${KINOGEN_PYBIND_DIR}" \
  -DPYTHON_EXECUTABLE="${KINOGEN_PROJECT_ROOT}/.venvs/gpu/bin/python" \
  -DPython3_EXECUTABLE="${KINOGEN_PROJECT_ROOT}/.venvs/gpu/bin/python"
cmake --build "${KINOGEN_PROJECT_ROOT}/.build/foundationpose-mycpp" --parallel 2
