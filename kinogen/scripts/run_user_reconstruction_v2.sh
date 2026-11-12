#!/usr/bin/env bash
# Run on yz-ai-4090-01.bce:/data/KinoGen; default is a CPU-only final audit.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
stage="${1:-verify}"
run_cpu() (
  source scripts/activate-server.sh vision
  export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS="${KINOGEN_CPU_THREADS:-6}" MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS="${KINOGEN_BLAS_THREADS:-4}"
  python -u "$@"
)
check_inputs() {
  for item in outputs/user_reconstruction/input.npz outputs/user_reconstruction/segmentation.npz outputs/user_reconstruction/tracking.npz outputs/user_reconstruction/geometry.npz outputs/user_reconstruction_v2/camera_refinement.npz outputs/user_reconstruction_v2/box_trajectory.npz data/video_generation/candidate_scene/scene_rohand.json data/robots/g1_rohand_five_finger/g1_rohand.urdf; do
    [[ -f "$item" ]] || { printf 'Required prepared input is missing: %s\n' "$item" >&2; exit 2; }
  done
  run_cpu scripts/verify_user_reconstruction_v2.py --preflight
}
body() {
  check_inputs
  run_cpu scripts/optimize_user_body.py
  run_cpu scripts/optimize_user_body.py --resume-existing --refine-frames 59:120 --temporal-strength 10 --arm-prior 4
}
hand() {
  check_inputs
  run_cpu scripts/optimize_user_hands.py --refine-wrist-yaw --max-nfev 24
}
render() (
  # Exporter has an explicit physical-GPU7 contract. This stage is opt-in.
  export KINOGEN_CUDA_DEVICES=7
  source scripts/activate-server.sh vision
  export CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
  python -u scripts/export_user_full_scene.py --hand-file outputs/user_reconstruction_v2/hand_trajectory_wrist_refined.npz
)
render_contact() (
  export KINOGEN_CUDA_DEVICES=7
  source scripts/activate-server.sh vision
  export CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
  python -u scripts/export_user_full_scene.py --variant contact --body-file outputs/user_reconstruction_v2/body_trajectory_contact.npz --hand-file outputs/user_reconstruction_v2/hand_trajectory_contact.npz
)
velocity_hand() {
  check_inputs
  KINOGEN_CPU_THREADS=4 KINOGEN_BLAS_THREADS=4 run_cpu scripts/optimize_user_hands.py --refine-wrist-yaw --wrist-velocity-limit 22 --output-suffix velocity_refined --max-nfev 24
}
contact_body() {
  check_inputs
  KINOGEN_BLAS_THREADS=6 run_cpu scripts/optimize_user_arm_contact.py
  # Preserve the exact first pass and retry only its measured collision failures.
  # SHA-derived archive names also work for a legitimate fresh rerun.
  local repair_spec
  repair_spec="$(.venvs/vision/bin/python - <<'PY'
from pathlib import Path
import hashlib,json,shutil
p=Path('outputs/user_reconstruction_v2/body_trajectory_contact.npz')
rp=Path('reports/user-arm-contact.json');r=json.loads(rp.read_text())
h=hashlib.sha256(p.read_bytes()).hexdigest()
if h!=r['output']['sha256']:raise RuntimeError('First-pass report/body binding mismatch')
archive=p.with_name('body_trajectory_contact_firstpass_'+h[:12]+'.npz')
report=rp.with_name('user-arm-contact_firstpass_'+h[:12]+'.json')
for src,dst in ((p,archive),(rp,report)):
    if dst.exists() and dst.read_bytes()!=src.read_bytes():raise RuntimeError('Immutable archive conflict: '+str(dst))
    if not dst.exists():shutil.copyfile(src,dst)
print(archive);print(report)
print(','.join(map(str,r['unresolved_collision_frame_indices'])))
PY
)"
  local -a repair
  mapfile -t repair <<< "$repair_spec"
  if [[ -n "${repair[2]:-}" ]]; then
    KINOGEN_BLAS_THREADS=6 run_cpu scripts/optimize_user_arm_contact.py --body-file "${repair[0]}" --source-contact-report "${repair[1]}" --repair-frames "${repair[2]}" --hand-image-weight .65
  fi
  KINOGEN_BLAS_THREADS=6 run_cpu scripts/optimize_user_arm_contact.py --finalize-metadata
}
contact_hand() {
  check_inputs
  KINOGEN_CPU_THREADS=4 KINOGEN_BLAS_THREADS=4 run_cpu scripts/optimize_user_hands.py --body-file outputs/user_reconstruction_v2/body_trajectory_contact.npz --initial-hand-file outputs/user_reconstruction_v2/hand_trajectory_velocity_refined.npz --refine-wrist-yaw --wrist-velocity-limit 22 --full-mesh-contact --output-suffix contact --max-nfev 24
}
case "$stage" in
  verify) run_cpu scripts/verify_user_reconstruction_v2.py ;;
  preflight) run_cpu scripts/verify_user_reconstruction_v2.py --preflight ;;
  body) body ;;
  hand) hand ;;
  export) render ;;
  contact-export) render_contact ;;
  contact-verify) run_cpu scripts/verify_user_reconstruction_v2.py --run-variant contact ;;
  contact-preflight) run_cpu scripts/verify_user_reconstruction_v2.py --run-variant contact --preflight ;;
  velocity-hand) velocity_hand ;;
  contact-body) contact_body ;;
  contact-hand) contact_hand ;;
  contact-all) velocity_hand; contact_body; contact_hand; render_contact; run_cpu scripts/verify_user_reconstruction_v2.py --run-variant contact ;;
  all) body; hand; render; run_cpu scripts/verify_user_reconstruction_v2.py ;;
  *) printf 'Usage: bash scripts/run_user_reconstruction_v2.sh [verify|preflight|body|hand|export|all|velocity-hand|contact-body|contact-hand|contact-export|contact-verify|contact-preflight|contact-all]\n' >&2; exit 2 ;;
esac
