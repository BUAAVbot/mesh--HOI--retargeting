"""Independent, read-only kinematic QA for exo Omni/HOI HDMI28 tracks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np


def main(root: Path, model_path: Path) -> None:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    feet = [i for i in range(model.ngeom) if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("collision_foot_")]
    limited = [(j, model.jnt_qposadr[j], model.jnt_range[j]) for j in range(model.njnt) if model.jnt_limited[j]]
    rows = []
    for path in sorted(root.glob("*/person_*/[oh]*/reference.npz")):
        if path.parent.name not in ("omni", "hoi"):
            continue
        meta_path = path if path.parent.name == "omni" else Path(json.loads(path.with_suffix(".json").read_text())["stage1_reference_path"])
        with np.load(path) as a:
            q = np.asarray(a["qpos"], dtype=float)
            fps = float(a["fps"])
        with np.load(meta_path) as a:
            rotation = np.asarray(a["world_from_source_rotation"], dtype=float)
            floor = float(a["floor_offset"])
            valid = np.asarray(a["source_valid"], dtype=bool)
        coordinate_estimate = json.loads(meta_path.with_suffix(".json").read_text()).get("coordinate_estimate", {})
        human = Path("/data/yihang.bai/incoming/aliyun_hoi_result/exo") / path.parents[2].name / "humans" / path.parents[1].name / "smplx_human_motion.npz"
        with np.load(human) as a:
            sj = np.asarray(a["joints"][:len(q), [7, 8]], dtype=float)
            source_total_frames = len(a["joints"])
            source_valid_total = int(np.count_nonzero(a["valid"]))
        ankle_heights = (sj @ rotation.T)[..., 2] - floor
        human_min_ankle = np.min(ankle_heights, axis=1)
        foot_heights = np.empty(len(q))
        for t, pose in enumerate(q):
            data.qpos[:] = pose
            mujoco.mj_forward(model, data)
            lower = []
            for i in feet:
                # Exact lowest point for MuJoCo box collision geoms.
                R = data.geom_xmat[i].reshape(3, 3)
                lower.append(data.geom_xpos[i, 2] - np.abs(R[2]) @ model.geom_size[i])
            foot_heights[t] = min(lower)
        speeds = np.abs(np.diff(q[:, 7:], axis=0)) * fps
        root_step = np.linalg.norm(np.diff(q[:, :3], axis=0), axis=1)
        quat_norm = np.linalg.norm(q[:, 3:7], axis=1)
        limits = np.stack([np.maximum(np.maximum(r[0] - q[:, idx], q[:, idx] - r[1]), 0) for _, idx, r in limited], axis=1)
        foot_ankle_gap = foot_heights - human_min_ankle
        row = {
            "task": path.parents[2].name,
            "person": path.parents[1].name,
            "method": path.parent.name,
            "frames": len(q),
            "fps": fps,
            "qpos_finite": bool(np.isfinite(q).all()),
            "quat_norm_max_error": float(np.max(np.abs(quat_norm - 1))),
            "joint_limit_violation_max_rad": float(np.max(limits)),
            "joint_limit_violation_frames": int(np.count_nonzero(np.any(limits > 1e-3, axis=1))),
            "joint_speed_max_rad_s": float(np.max(speeds)),
            "joint_speed_p95_rad_s": float(np.quantile(speeds, .95)),
            "joint_speed_over_12_transitions": int(np.count_nonzero(np.any(speeds > 12, axis=1))),
            "root_step_max_m": float(np.max(root_step)),
            "root_step_max_transition": int(np.argmax(root_step)),
            "root_step_over_0_2m_transitions": int(np.count_nonzero(root_step > .2)),
            "foot_lowest_min_m": float(np.min(foot_heights)),
            "foot_lowest_median_m": float(np.median(foot_heights)),
            "foot_lowest_p95_m": float(np.quantile(foot_heights, .95)),
            "foot_above_0_1m_frames": int(np.count_nonzero(foot_heights > .1)),
            "foot_above_0_3m_frames": int(np.count_nonzero(foot_heights > .3)),
            "foot_above_0_6m_frames": int(np.count_nonzero(foot_heights > .6)),
            "foot_below_minus_0_03m_frames": int(np.count_nonzero(foot_heights < -.03)),
            "source_min_ankle_median_m": float(np.median(human_min_ankle)),
            "foot_minus_source_ankle_median_m": float(np.median(foot_ankle_gap)),
            "foot_minus_source_ankle_abs_p95_m": float(np.quantile(np.abs(foot_ankle_gap), .95)),
            "source_valid_frames": int(np.count_nonzero(valid)),
            "source_total_frames": source_total_frames,
            "source_valid_total": source_valid_total,
            "floor_source": coordinate_estimate.get("floor_source", "unknown"),
            "source_pelvis_to_ankle_median_m": coordinate_estimate.get("human_pelvis_to_ankle_median_m"),
            "worst_foot_height_frame": int(np.argmax(foot_heights)),
            "worst_foot_height_m": float(np.max(foot_heights)),
            "path": str(path),
        }
        rows.append(row)
    summary = {
        "scope": "Independent kinematic FK QA of HDMI28 Omni and HOI trajectories; no dynamics, contact-force, or SONIC evaluation",
        "model": str(model_path),
        "foot_geoms": [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in feet],
        "source_floor_method": "per-track coordinate estimate; for rejected scene plane uses ankle proxy",
        "track_count": len(rows),
        "methods": {k: sum(r["method"] == k for r in rows) for k in ("omni", "hoi")},
        "interpretation": {
            "kinematic_fk_verified": True,
            "grounded_claim_supported": False,
            "reason": "Many tracks have feet hundreds of millimeters above the estimated floor for most frames; source ankles share much of this offset, so source floor/scale alignment also needs review. Neither collision force nor dynamic tracking was validated.",
            "notable": [
                "gallery03: both methods foot lowest median 0.535m; 234/350 frames over 0.3m",
                "gallery08 person02: only 157/174 source frames retained; low median foot gap does not establish full-clip validity",
                "train_vid2 Omni: 0.368m root step on first transition",
                "gallery04 Omni: one transition exceeds 12rad/s joint speed",
            ],
        },
        "fair_ranking_recommendation": "Normalize pose/bone and relative end-effector errors by source/robot leg or body scale and compare in root-local coordinates. Gate separately on source validity, velocity spikes, joint limits, foot-to-floor and foot-to-source-ankle clearance, and observed support/contact. Raw world-body RMSE alone rewards floating poses under human/robot morphology mismatch. Require scene/object collision and SONIC dynamics before physical success claims.",
        "rows": rows,
    }
    out = root / "qa_exo_20261030.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"output": str(out), "track_count": len(rows), "rows": [{k: r[k] for k in ("task", "person", "method", "foot_lowest_median_m", "foot_above_0_3m_frames", "joint_limit_violation_max_rad", "joint_speed_max_rad_s", "root_step_max_m")} for r in rows]}))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
