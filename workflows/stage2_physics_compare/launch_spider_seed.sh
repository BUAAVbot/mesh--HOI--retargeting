#!/usr/bin/env bash
set -u

if [ "$#" -ne 4 ]; then
  echo "usage: $0 GPU SEED OUTPUT_DIR LOG_FILE" >&2
  exit 2
fi

gpu="$1"
seed="$2"
output="$3"
log="$4"
root="/data/yihang.bai/experiments/stage2_same_track_20261102"
mkdir -p "$output" "$(dirname "$log")"
started="$(date -Is)"

CUDA_VISIBLE_DEVICES="$gpu" \
PYTHONPATH="$root/envs/spider_overlay_py310" \
/root/vita02a-mjlab/.venv310/bin/python "$root/workflow/spider_adapter.py" \
  --repo "$root/repos/spider" \
  --model "$root/contracts/spider_proxy_stride3.mjb" \
  --reference "$root/contracts/canonical_reference.npz" \
  --output "$output" \
  --device cuda:0 \
  --seed "$seed" \
  --samples 64 \
  --iterations 2 \
  --horizon 0.2 >"$log" 2>&1
code="$?"

{
  echo "started=$started"
  echo "finished=$(date -Is)"
  echo "exit_code=$code"
  echo "physical_gpu=$gpu"
  echo "logical_device=cuda:0"
  echo "seed=$seed"
  echo "samples=64"
  echo "iterations=2"
  echo "horizon_s=0.2"
  echo "optimization_model=spider_proxy_stride3.mjb"
  echo "acceptance_model=original_frozen_compiled_model.mjb"
} >"$output/status.txt"
exit "$code"
