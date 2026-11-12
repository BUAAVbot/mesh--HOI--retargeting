#!/usr/bin/env bash
# Server only. Sequential public-component checks; never a full KinoGen run.
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# Do not inherit CUDA_VISIBLE_DEVICES=0 from another task or shell.
export KINOGEN_CUDA_DEVICES="${KINOGEN_CUDA_DEVICES:-6}"
if [[ ! "$KINOGEN_CUDA_DEVICES" =~ ^[67](,[67])?$ ]]; then
  printf 'Use physical GPU 6 or 7 only: KINOGEN_CUDA_DEVICES=6 (or 7).\n' >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="$KINOGEN_CUDA_DEVICES"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MPLBACKEND=Agg PYOPENGL_PLATFORM=egl
KINOGEN_DEPTH_ENCODER="${KINOGEN_DEPTH_ENCODER:-vitl}"
case "$KINOGEN_DEPTH_ENCODER" in vitl|vits) ;; *) printf 'KINOGEN_DEPTH_ENCODER must be vitl or vits.\n' >&2; exit 2 ;; esac

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
RUN_DIR="$ROOT/outputs/component_flow/$RUN_ID"
mkdir -p "$RUN_DIR" "$ROOT/reports" "$ROOT/outputs/previews"
export KINOGEN_FLOW_RUN_DIR="$RUN_DIR" KINOGEN_FLOW_DEPTH_ENCODER="$KINOGEN_DEPTH_ENCODER"
trap 'code=$?; printf "Component flow stopped (exit %s). Logs: %s\n" "$code" "$RUN_DIR" >&2; exit "$code"' ERR

for profile in kimodo vision sam3; do
  if [[ ! -x "$ROOT/.venvs/$profile/bin/python" ]]; then
    printf 'Required server environment missing: .venvs/%s\n' "$profile" >&2
    exit 2
  fi
done
if [[ ! -f "$ROOT/data/smoke_inputs/test1.jpg" ]]; then
  printf 'Missing official WiLoR demo input: data/smoke_inputs/test1.jpg\n' >&2
  exit 2
fi

printf '[1/4] Kimodo: empty text, official root constraints, +Z travel 1 m\n'
(
  source "$ROOT/scripts/activate-server.sh" kimodo
  python scripts/smoke_kimodo.py --root-distance-m 1 --frames 90 --steps 50 \
    --output-dir "$RUN_DIR/kimodo"
) 2>&1 | tee "$RUN_DIR/01-kimodo.log"

printf '[2/4] FoundationPose: strict network weights, synthetic crop forward\n'
(
  source "$ROOT/scripts/activate-server.sh" vision
  python scripts/smoke_foundationpose.py --report "$RUN_DIR/foundationpose.json"
) 2>&1 | tee "$RUN_DIR/02-foundationpose.log"

printf '[3/4] SAM3: first-frame hand/person segmentation (compatibility runtime)\n'
(
  source "$ROOT/scripts/activate-server.sh" sam3
  python scripts/smoke_sam3.py --allow-unsupported-runtime --prompts hand person
) 2>&1 | tee "$RUN_DIR/03-sam3.log"

printf '[4/4] WiLoR + MANO, VDA, and TAPNext++ using the actual SAM3 hand mask\n'
(
  source "$ROOT/scripts/activate-server.sh" vision
  # SAM may find a person but no hand. Fail rather than using a generic grid.
  python - <<'PY'
import json
from pathlib import Path
import numpy as np
from PIL import Image
root = Path.cwd()
report = json.loads((root / "reports/sam3-smoke.json").read_text())
if report.get("status") != "passed":
    raise RuntimeError("SAM3 inference has not passed")
hand = next((item for item in report["prompts"] if item["prompt"] == "hand"), None)
if not hand or hand.get("union_area_pixels", 0) < 9:
    raise RuntimeError("SAM3 produced no usable hand mask; no tracking fallback is allowed")
mask_path = root / "outputs/sam3_smoke/hand_mask.png"
if (root / hand["mask"]).resolve() != mask_path.resolve():
    raise RuntimeError("Unexpected SAM3 hand-mask provenance")
if int(np.count_nonzero(np.asarray(Image.open(mask_path)))) < 9:
    raise RuntimeError("Saved SAM3 hand mask is empty or too small")
print("SAM3 hand mask verified before tracking")
PY
  python scripts/smoke_vision.py --component all --encoder "$KINOGEN_DEPTH_ENCODER" \
    --depth-input-size 280 --frames 12 --fps 6 \
    --query-mask outputs/sam3_smoke/hand_mask.png
  python - <<'PY'
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
root = Path.cwd()
run_dir = Path(os.environ["KINOGEN_FLOW_RUN_DIR"])
def read(path):
    return json.loads(path.read_text())
kimodo_paths = list((run_dir / "kimodo").glob("kimodo_g1_kinematic_*.json"))
kimodo_paths = [path for path in kimodo_paths if not path.name.endswith(".constraints.json")]
if len(kimodo_paths) != 1:
    raise RuntimeError("Expected exactly one fresh Kimodo generation report")
kimodo = read(kimodo_paths[0])
if not kimodo.get("constraint_checks", {}).get("constraint_following_passed"):
    raise RuntimeError("Kimodo root constraint check did not pass")
fp = read(run_dir / "foundationpose.json")
if fp.get("status") != "passed":
    raise RuntimeError("FoundationPose network smoke did not pass")
sam_path = root / "reports/sam3-smoke.json"
sam = read(sam_path)
vision_path = root / "reports/vision-smoke-all.json"
vision = read(vision_path)
if not all(vision["components"].get(name, {}).get("status") == "passed" for name in ("wilor", "depth", "tracking")):
    raise RuntimeError("A vision component did not pass")
mask = root / "outputs/sam3_smoke/hand_mask.png"
mask_hash = hashlib.sha256(mask.read_bytes()).hexdigest()
tracking_mask = vision["components"]["tracking"].get("query_mask")
if not tracking_mask or tracking_mask.get("sha256") != mask_hash:
    raise RuntimeError("Tracking did not consume this SAM3 hand mask")
if vision.get("input", {}).get("sha256") != sam.get("input", {}).get("sha256"):
    raise RuntimeError("SAM3 and vision components used different source images")
for file in (run_dir / "kimodo").glob("*.mp4"):
    shutil.copy2(file, root / "outputs/previews" / file.name)
shutil.copy2(sam_path, run_dir / "sam3.json")
shutil.copy2(vision_path, run_dir / "vision.json")
result = {
    "checked_at": datetime.now(timezone.utc).isoformat(),
    "status": "component_flow_passed", "run_directory": str(run_dir),
    "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
    "depth_encoder": os.environ["KINOGEN_FLOW_DEPTH_ENCODER"],
    "full_kinogen_pipeline": False, "training_executed": False,
    "video_generation_api_executed": False, "sam3_unsupported_runtime_allowed": True,
    "input_kind": "synthetic_camera_translation_of_official_wilor_demo_image",
    "sam3_to_tapnextpp_mask_sha256": mask_hash,
    "reports": {"kimodo": str(kimodo_paths[0]), "foundationpose": str(run_dir / "foundationpose.json"), "sam3": str(run_dir / "sam3.json"), "vision": str(run_dir / "vision.json")},
    "limitations": ["Independent Kimodo motion is not fed to the vision input", "Static image translations, not an actual action video", "SAM3 and hand reconstruction use only the first frame", "FoundationPose checks network inference, not object registration", "No Qwen replacement or text conditioning", "KinoGen official integration is unreleased"],
}
text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
(run_dir / "flow.json").write_text(text)
(root / "reports/component-flow-latest.json").write_text(text)
print(text)
PY
) 2>&1 | tee "$RUN_DIR/04-vision.log"
printf 'Component checks passed. Review %s/flow.json and outputs/previews/.\n' "$RUN_DIR"
