"""Server gate: compile Vita28 MJCF and verify mjlab motion tensor contracts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


REQUIRED = {
    "joint_pos": (2, 28), "joint_vel": (2, 28),
    "body_pos_w": (3, 3), "body_quat_w": (3, 4),
    "body_lin_vel_w": (3, 3), "body_ang_vel_w": (3, 3),
}


def validate(model_file: Path, motion_file: Path) -> dict:
    model = mujoco.MjModel.from_xml_path(str(model_file))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
                   for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(1, model.nbody)]
    if len(joint_names) != 28 or model.nq != 35 or model.nv != 34:
        raise ValueError(f"Vita28 model dimension mismatch: joints={len(joint_names)}, nq={model.nq}, nv={model.nv}")
    if model.nsensor < 2:
        raise ValueError("missing robot IMU sensors")
    with np.load(motion_file, allow_pickle=False) as m:
        n = len(m["joint_pos"])
        for key, (rank, width) in REQUIRED.items():
            a = m[key]
            if a.ndim != rank or a.shape[0] != n or a.shape[-1] != width or not np.isfinite(a).all():
                raise ValueError(f"invalid {key}: {a.shape}")
            if rank == 3 and a.shape[1] != len(body_names):
                raise ValueError(f"body count mismatch in {key}: {a.shape[1]} vs {len(body_names)}")
        fps = float(m["fps"].reshape(-1)[0])
        if abs(fps - 50.0) > 1e-4:
            raise ValueError(f"mjlab task requires 50 Hz, found {fps}")
        norms = np.linalg.norm(m["body_quat_w"], axis=-1)
        if np.max(np.abs(norms - 1.0)) > 1e-3:
            raise ValueError("body quaternion normalization failed")
    return {"frames": n, "fps": fps, "nq": model.nq, "nv": model.nv,
            "bodies": len(body_names), "hinge_joints": len(joint_names),
            "sensors": model.nsensor, "total_mass_kg": float(model.body_mass.sum()),
            "joint_names": joint_names, "body_names": body_names}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mjcf", type=Path, required=True)
    p.add_argument("--motion", type=Path, required=True)
    a = p.parse_args()
    receipt = validate(a.mjcf, a.motion)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
