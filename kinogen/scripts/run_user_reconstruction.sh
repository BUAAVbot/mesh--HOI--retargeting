#!/usr/bin/env bash
# This entry point is specific to the inspected SHA-locked 121-frame user clip.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
stage="${1:-verify}"
run_python() (
  export KINOGEN_CUDA_DEVICES=6
  source scripts/activate-server.sh "$1"
  shift
  python "$@"
)
case "$stage" in
  all)
    run_python vision scripts/prepare_user_reconstruction.py
    run_python sam3 scripts/process_user_vision.py --stage segmentation
    run_python vision scripts/prepare_reconstruction_geometry.py
    run_python vision scripts/verify_reconstruction_camera_drift.py
    run_python vision scripts/prepare_shared_reconstruction_points.py
    run_python vision scripts/process_user_vision.py --stage depth
    run_python vision scripts/track_user_video.py
    run_python vision scripts/reconstruct_box.py
    run_python vision scripts/calibrate_user_depth.py
    run_python vision scripts/verify_user_reconstruction.py
    ;;
  verify) run_python vision scripts/verify_user_reconstruction.py ;;
  *) printf 'Usage: bash scripts/run_user_reconstruction.sh [verify|all]\n' >&2; exit 2 ;;
esac
