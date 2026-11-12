#!/usr/bin/env bash
# Source this file on yz-ai-4090-01.bce; KINOGEN_CUDA_DEVICES selects GPUs.
KINOGEN_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
KINOGEN_ENV_PROFILE="${1:-gpu}"
case "${KINOGEN_ENV_PROFILE}" in
  gpu|tools|kimodo|kimodo-text|vision|sam3) ;;
  *) printf 'Use: source scripts/activate-server.sh [gpu|tools|kimodo|kimodo-text|vision|sam3]\n' >&2; return 1 ;;
esac
if [[ ! -f "${KINOGEN_PROJECT_ROOT}/.venvs/${KINOGEN_ENV_PROFILE}/bin/activate" ]]; then
  printf 'Environment has not been created: %s\n' "${KINOGEN_ENV_PROFILE}" >&2
  return 1
fi
source "${KINOGEN_PROJECT_ROOT}/.venvs/${KINOGEN_ENV_PROFILE}/bin/activate" || return 1
export KINOGEN_PROJECT_ROOT
export HF_HOME="${KINOGEN_PROJECT_ROOT}/.cache/huggingface"
export TORCH_HOME="${KINOGEN_PROJECT_ROOT}/.cache/torch"
export PIP_CACHE_DIR="${KINOGEN_PROJECT_ROOT}/.cache/pip"
export PIP_REQUIRE_VIRTUALENV=true
export PYTHONDONTWRITEBYTECODE=1
if [[ "${KINOGEN_ENV_PROFILE}" != tools ]]; then
  export CUDA_HOME=/usr/local/cuda-12.4
  export CUDA_VISIBLE_DEVICES="${KINOGEN_CUDA_DEVICES:-${CUDA_VISIBLE_DEVICES:-6}}"
  export PYOPENGL_PLATFORM=egl
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:/data/grail-envs/grail/lib/python3.10/site-packages/torch/lib:/data/grail-envs/grail/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  # The shared baseline contains a cu13 cuDNN overlay. Reuse the already
  # installed cu12 cuDNN per process; never replace files in the shared env.
  KINOGEN_CUDNN_PRELOAD="${CUDA_HOME}/lib64/libcudnn.so.9:${CUDA_HOME}/lib64/libcudnn_graph.so.9"
  if [[ -f "${CUDA_HOME}/lib64/libcudnn.so.9" && -f "${CUDA_HOME}/lib64/libcudnn_graph.so.9" ]]; then
    if [[ ":${LD_PRELOAD:-}:" != *":${KINOGEN_CUDNN_PRELOAD}:"* ]]; then
      export LD_PRELOAD="${KINOGEN_CUDNN_PRELOAD}${LD_PRELOAD:+:${LD_PRELOAD}}"
    fi
  fi
  export CUDA_CACHE_PATH="${KINOGEN_PROJECT_ROOT}/.cache/cuda"
  export TRITON_CACHE_DIR="${KINOGEN_PROJECT_ROOT}/.cache/triton"
  export TORCH_EXTENSIONS_DIR="${KINOGEN_PROJECT_ROOT}/.cache/torch_extensions"
fi
cd "${KINOGEN_PROJECT_ROOT}" || return 1
