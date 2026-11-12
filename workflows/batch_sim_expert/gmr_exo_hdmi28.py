"""Run upstream GMR on one exo SMPL-X person and HDMI Vita28.

Output is a kinematic robot reference. Physics, contact and task success must
be checked by a separate evaluator.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from exo_world_frame import estimate


PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
MATCHES = [
    ("base_link", "pelvis", 0, 100., 10.),
    ("waist_pitch_link", "spine3", 9, 10., 10.),
    ("left_hip_roll_link", "left_hip", 1, 10., 5.),
    ("right_hip_roll_link", "right_hip", 2, 10., 5.),
    ("left_knee_link", "left_knee", 4, 10., 5.),
    ("right_knee_link", "right_knee", 5, 10., 5.),
    ("left_ankle_roll_link", "left_ankle", 7, 100., 50.),
    ("right_ankle_roll_link", "right_ankle", 8, 100., 50.),
    ("left_shoulder_yaw_link", "left_shoulder", 16, 10., 5.),
    ("right_shoulder_yaw_link", "right_shoulder", 17, 10., 5.),
    ("left_elbow_link", "left_elbow", 18, 10., 5.),
    ("right_elbow_link", "right_elbow", 19, 10., 5.),
    ("left_wrist_pitch_link", "left_wrist", 20, 20., 5.),
    ("right_wrist_pitch_link", "right_wrist", 21, 20., 5.),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--gmr-source", type=Path, default=Path("/data/grail-tools/gmr-pinned-source"))
    ap.add_argument("--frames", type=int, default=0, help="0 means full clip")
    a = ap.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    import mujoco as mj
    sys.path.insert(0, str(a.gmr_source))
    import general_motion_retargeting.motion_retarget as gmr

    model = mj.MjModel.from_xml_path(str(a.model.resolve()))
    if (model.nq, model.nv) != (35, 34):
        raise ValueError(f"expected HDMI Vita28 nq/nv=35/34, got {model.nq}/{model.nv}")
    joint_ids = np.array([i for i in range(model.njnt) if model.jnt_type[i] == mj.mjtJoint.mjJNT_HINGE])
    joint_ids = joint_ids[np.argsort(model.jnt_qposadr[joint_ids])]
    names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, int(i)) for i in joint_ids]
    if len(names) != 28:
        raise ValueError("HDMI model has incorrect joint count")
    for body, *_ in MATCHES:
        if mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body) < 0:
            raise ValueError(f"HDMI model missing mapped body {body}")

    scene_mesh = a.input.parents[2] / "scene" / "mesh.npz"
    world_rotation, floor_offset, frame_audit = estimate(a.input, scene_mesh)
    with np.load(a.input, allow_pickle=False) as data:
        count = len(data["joints"]) if a.frames <= 0 else min(a.frames, len(data["joints"]))
        joints = np.asarray(data["joints"][:count, :22], dtype=np.float64)
        orient = np.asarray(data["global_orient"][:count], dtype=np.float64)
        body_pose = np.asarray(data["body_pose"][:count], dtype=np.float64).reshape(count, 21, 3)
        times = np.asarray(data["pts_seconds"][:count], dtype=np.float64)
        valid = np.asarray(data["valid"][:count], dtype=bool)
    if count < 2 or not np.isfinite(joints).all() or not np.isfinite(orient).all() or not np.isfinite(body_pose).all():
        raise ValueError("invalid or nonfinite SMPL-X trajectory")
    times -= times[0]
    dt = np.diff(times)
    if np.any(dt <= 0):
        raise ValueError("non-increasing timestamps")
    fps = float(1 / np.median(dt))
    joints = joints @ world_rotation.T
    joints[..., 2] -= floor_offset
    local = Rotation.from_rotvec(np.concatenate([orient[:, None], body_pose], axis=1).reshape(-1, 3)).as_matrix()
    local = local.reshape(count, 22, 3, 3)
    rotations = np.zeros_like(local)
    for j, parent in enumerate(PARENTS):
        rotations[:, j] = world_rotation @ local[:, j] if parent < 0 else rotations[:, parent] @ local[:, j]

    # The upstream solver needs a robot-specific table. Calibrate neutral
    # offsets from the actual HDMI MJCF, following the audited Vita adapter.
    neutral_map = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
    neutral = mj.MjData(model)
    neutral.qpos[:] = model.qpos0
    neutral.qpos[2] = 1.
    for side, sign in (("left", 1), ("right", -1)):
        for suffix, value in (("shoulder_roll_joint", sign * np.pi / 2), ("elbow_joint", np.pi / 2)):
            jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, side + "_" + suffix)
            neutral.qpos[model.jnt_qposadr[jid]] = value
    mj.mj_forward(model, neutral)
    table1, table2 = {}, {}
    for robot, human, _, pos_weight, rot_weight in MATCHES:
        bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, robot)
        offset = Rotation.from_matrix(neutral_map @ neutral.xmat[bid].reshape(3, 3)).as_quat(scalar_first=True).tolist()
        table2[robot] = [human, pos_weight, rot_weight, [0., 0., 0.], offset]
        first_weight = pos_weight if human in ("pelvis", "left_ankle", "right_ankle") else 0.
        table1[robot] = [human, first_weight, min(rot_weight, 10.), [0., 0., 0.], offset]
    config = {
        "robot_root_name": "base_link", "human_root_name": "pelvis", "ground_height": 0.,
        "human_height_assumption": 1.8, "use_ik_match_table1": True, "use_ik_match_table2": True,
        "human_scale_table": {human: 1. for _, human, *_ in MATCHES},
        "ik_match_table1": table1, "ik_match_table2": table2,
    }
    config_path = a.output.with_suffix(".gmr_config.json")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    key = "hdmi_vita28_exo"
    gmr.ROBOT_XML_DICT[key] = a.model.resolve()
    gmr.IK_CONFIG_DICT.setdefault("smplx", {})[key] = config_path.resolve()
    solver = gmr.GeneralMotionRetargeting("smplx", key, actual_human_height=None,
                                          solver="daqp", damping=.5, verbose=False,
                                          use_velocity_limit=False)
    initial = model.qpos0.copy()
    initial[:3] = joints[0, 0]
    initial[3:7] = Rotation.from_matrix(rotations[0, 0] @ neutral_map).as_quat(scalar_first=True)
    solver.configuration.update(initial)
    rows, errors = [], []
    for frame in range(count):
        targets = {human: [joints[frame, idx], Rotation.from_matrix(rotations[frame, idx]).as_quat(scalar_first=True)]
                   for _, human, idx, _, _ in MATCHES}
        for _ in range(3 if frame == 0 else 1):
            q = solver.retarget(targets, offset_to_ground=False)
        solver.configuration.update(q)
        rows.append(q.copy())
        errors.append([solver.error1(), solver.error2()])
        if frame % 30 == 0 or frame == count - 1:
            print(json.dumps({"frame": frame, "frames": count, "error2": errors[-1][1]}), flush=True)
    qpos = np.asarray(rows)
    if qpos.shape != (count, 35) or not np.isfinite(qpos).all():
        raise ValueError("GMR produced an invalid HDMI reference")
    np.savez_compressed(a.output, qpos=qpos, robot_joint_names=np.asarray(names), fps=fps,
                        time_s=times, source_valid=valid, task_error=np.asarray(errors),
                        world_from_source_rotation=world_rotation, floor_offset=floor_offset)
    report = {"method": "upstream GMR Mink/DAQP", "model": str(a.model), "input": str(a.input),
              "output": str(a.output), "frames": count, "qpos_shape": list(qpos.shape),
              "fps": fps, "valid_source_fraction": float(valid.mean()),
              "coordinate_estimate": frame_audit, "elapsed_seconds": time.perf_counter() - started,
              "physical_success": False}
    a.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
