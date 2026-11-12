"""Project only the opening frames onto double-foot support in the compiled scene."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from common import classify_robot_static_bodies, sha256, strip_namespace, write_json


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", required=True)
    parser.add_argument("--motion-json", required=True)
    parser.add_argument("--model-mjb", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hold-frames", type=int, default=5)
    parser.add_argument("--taper-frames", type=int, default=25)
    parser.add_argument("--target-gap-m", type=float, default=0.001)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    with np.load(args.motion, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    sidecar = json.loads(Path(args.motion_json).read_text(encoding="utf-8"))
    model = mujoco.MjModel.from_binary_path(args.model_mjb)
    data = mujoco.MjData(model)
    robot_bodies, static_bodies, robot_order = classify_robot_static_bodies(model, mujoco)
    joint_names = sidecar["joint_names"]
    joint_ids = [
        next(j for j in range(model.njnt)
             if strip_namespace(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or "") == name)
        for name in joint_names
    ]
    qpos_ids = np.asarray([model.jnt_qposadr[joint_id] for joint_id in joint_ids], dtype=int)
    qpos_to_joint = dict(zip(qpos_ids.tolist(), joint_ids))
    leg_qpos = {
        side: np.asarray([
            model.jnt_qposadr[joint_ids[joint_names.index(f"{side}_{token}_joint")]]
            for token in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
        ], dtype=int)
        for side in ("left", "right")
    }
    support_geoms = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in static_bodies]
    foot_geoms = {
        side: [
            g for g in range(model.ngeom)
            if int(model.geom_bodyid[g]) in robot_bodies
            and ("foot" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "")
                 or "toe" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""))
            and (f"_{side[0]}_" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""))
        ]
        for side in ("left", "right")
    }
    if not support_geoms or any(not geoms for geoms in foot_geoms.values()):
        raise ValueError(f"Missing support or foot geometry: support={support_geoms}, feet={foot_geoms}")

    frames = len(arrays["joint_pos"])
    if not 0 < args.hold_frames < args.taper_frames < frames:
        raise ValueError("Expected 0 < hold-frames < taper-frames < motion frames")
    qpos = np.empty((frames, model.nq), dtype=np.float64)
    qpos[:, :3] = arrays["body_pos_w"][:, 0]
    qpos[:, 3:7] = arrays["body_quat_w"][:, 0]
    qpos[:, qpos_ids] = arrays["joint_pos"]
    original_qpos = qpos.copy()

    def set_pose(pose: np.ndarray) -> None:
        data.qpos[:] = pose
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

    def foot_gap(pose: np.ndarray, side: str) -> float:
        set_pose(pose)
        return min(float(mujoco.mj_geomDistance(model, data, foot, support, 0.20, None))
                   for foot in foot_geoms[side] for support in support_geoms)

    original_gap = np.asarray([
        [foot_gap(original_qpos[frame], side) for side in ("left", "right")]
        for frame in range(args.taper_frames)
    ])
    weights = np.ones(args.taper_frames, dtype=np.float64)
    phase = np.linspace(0.0, 1.0, args.taper_frames - args.hold_frames)
    weights[args.hold_frames:] = 0.5 * (1.0 + np.cos(np.pi * phase))
    eps = 1e-4
    for frame, weight in enumerate(weights):
        common_drop = max(0.0, float(original_gap[frame].min() - args.target_gap_m)) * weight
        qpos[frame, 2] -= common_drop
        for side_index, side in enumerate(("left", "right")):
            desired = (1.0 - weight) * original_gap[frame, side_index] + weight * args.target_gap_m
            indices = leg_qpos[side]
            for _ in range(12):
                current = foot_gap(qpos[frame], side)
                if current <= desired + 2e-4:
                    break
                gradient = np.empty(len(indices), dtype=np.float64)
                for index, qpos_id in enumerate(indices):
                    center = qpos[frame, qpos_id]
                    qpos[frame, qpos_id] = center + eps
                    high = foot_gap(qpos[frame], side)
                    qpos[frame, qpos_id] = center - eps
                    low = foot_gap(qpos[frame], side)
                    qpos[frame, qpos_id] = center
                    gradient[index] = (high - low) / (2.0 * eps)
                delta = gradient * (desired - current) / (float(gradient @ gradient) + 1e-6)
                qpos[frame, indices] += np.clip(delta, -0.03, 0.03)
                for qpos_id in indices:
                    joint_id = qpos_to_joint[int(qpos_id)]
                    qpos[frame, qpos_id] = np.clip(
                        qpos[frame, qpos_id], model.jnt_range[joint_id, 0] + 1e-5,
                        model.jnt_range[joint_id, 1] - 1e-5)

    projected_gap = np.asarray([
        [foot_gap(qpos[frame], side) for side in ("left", "right")]
        for frame in range(args.taper_frames)
    ])
    min_scene_distance = np.inf
    for frame in range(args.taper_frames):
        set_pose(qpos[frame])
        for robot_geom in range(model.ngeom):
            if int(model.geom_bodyid[robot_geom]) not in robot_bodies:
                continue
            for support_geom in support_geoms:
                min_scene_distance = min(
                    min_scene_distance,
                    float(mujoco.mj_geomDistance(model, data, robot_geom, support_geom, 0.20, None)),
                )

    body_pos = np.empty_like(arrays["body_pos_w"], dtype=np.float64)
    body_quat = np.empty_like(arrays["body_quat_w"], dtype=np.float64)
    for frame in range(frames):
        set_pose(qpos[frame])
        body_pos[frame] = data.xpos[robot_order]
        body_quat[frame] = data.xquat[robot_order]
    fps = float(np.asarray(arrays["fps"]).reshape(-1)[0])
    arrays["joint_pos"] = qpos[:, qpos_ids].astype(np.float32)
    arrays["joint_vel"] = np.gradient(arrays["joint_pos"], 1.0 / fps, axis=0).astype(np.float32)
    arrays["body_pos_w"] = body_pos.astype(np.float32)
    arrays["body_quat_w"] = body_quat.astype(np.float32)
    arrays["body_lin_vel_w"] = np.gradient(body_pos, 1.0 / fps, axis=0).astype(np.float32)
    angular = np.zeros_like(body_pos)
    for frame in range(frames - 1):
        for body_index in range(len(robot_order)):
            inverse = np.empty(4, dtype=np.float64)
            mujoco.mju_negQuat(inverse, body_quat[frame, body_index])
            delta = np.empty(4, dtype=np.float64)
            mujoco.mju_mulQuat(delta, body_quat[frame + 1, body_index], inverse)
            mujoco.mju_quat2Vel(angular[frame, body_index], delta, 1.0 / fps)
    angular[-1] = angular[-2]
    arrays["body_ang_vel_w"] = angular.astype(np.float32)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    motion_out = output / "mjlab_motion.npz"
    np.savez_compressed(motion_out, **arrays)
    body_names = [
        strip_namespace(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or "")
        for body_id in robot_order
    ]
    sidecar.update({"source": str(Path(args.motion)), "output": str(motion_out),
                    "frames": frames, "fps": fps, "body_names": body_names})
    write_json(output / "mjlab_motion.json", sidecar)
    report = {
        "schema": "stage2-initial-double-support-projection-v1",
        "source_motion_sha256": sha256(args.motion),
        "model_mjb_sha256": sha256(args.model_mjb),
        "initial_gap_before_mm": (original_gap[0] * 1000.0).tolist(),
        "initial_gap_after_mm": (projected_gap[0] * 1000.0).tolist(),
        "max_root_drop_mm": float(np.max(original_qpos[:, 2] - qpos[:, 2]) * 1000.0),
        "max_leg_joint_change_rad": float(np.max(np.abs(qpos[:, 7:19] - original_qpos[:, 7:19]))),
        "minimum_robot_support_distance_mm": float(min_scene_distance * 1000.0),
        "hold_frames": args.hold_frames,
        "taper_frames": args.taper_frames,
    }
    report["geometry_gate_passed"] = bool(
        np.all(np.abs(projected_gap[0]) <= 0.002 + 1e-6) and min_scene_distance >= -0.003
    )
    write_json(output / "projection_report.json", report)
    if not report["geometry_gate_passed"]:
        raise RuntimeError(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
