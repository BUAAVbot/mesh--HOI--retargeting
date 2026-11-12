#!/usr/bin/env bash
set -euo pipefail
source /data/grail-envs/tracking45/bin/activate
export CUDA_VISIBLE_DEVICES="${KINOGEN_CUDA_DEVICES:-7}"
case "$CUDA_VISIBLE_DEVICES" in 6|7) ;; *) echo 'Choose reserved physical GPU6 or7' >&2; exit 2;; esac
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export OMNI_KIT_ACCEPT_EULA=Yes OMNI_KIT_ALLOW_ROOT=1
export WANDB_MODE=disabled WANDB_DISABLED=true HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PXR_WORK_THREAD_LIMIT=4
export CUDA_CACHE_PATH=/data/KinoGen/.cache/cuda
export TORCH_EXTENSIONS_DIR=/data/KinoGen/.cache/torch_extensions
cd /data/KinoGen
exec python scripts/train_g1_box.py "$@"
