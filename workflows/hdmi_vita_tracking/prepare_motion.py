"""Rebuild an HDMI motion for mjlab using the *target* MuJoCo model.

Run in the server's mjlab Python environment.  The HDMI NPZ body order is
Isaac-derived; mjlab indexes bodies in MuJoCo order, so body arrays must be
recomputed by FK instead of copied or merely renamed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


KEYS = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")


def resample(jp, root_pos, root_quat, fps_in: float, fps_out: float):
    """Resample positions/quaternions and recompute velocities on the target clock."""
    if fps_in <= 0 or fps_out <= 0 or len(jp) < 2:
        raise ValueError("frame rates must be positive")
    old_t = np.arange(len(jp), dtype=np.float64) / fps_in
    new_t = np.arange(int(round(old_t[-1] * fps_out)) + 1, dtype=np.float64) / fps_out
    new_t = np.minimum(new_t, old_t[-1])
    out_jp = np.column_stack([np.interp(new_t, old_t, jp[:, k]) for k in range(jp.shape[1])])
    out_pos = np.column_stack([np.interp(new_t, old_t, root_pos[:, k]) for k in range(3)])
    q = root_quat / np.linalg.norm(root_quat, axis=1, keepdims=True)
    q[1:] *= np.where(np.sum(q[1:] * q[:-1], axis=1) < 0, -1.0, 1.0)[:, None]
    idx = np.clip(np.searchsorted(old_t, new_t, side="right") - 1, 0, len(q) - 2)
    alpha = ((new_t - old_t[idx]) / (old_t[idx + 1] - old_t[idx]))[:, None]
    qa, qb = q[idx], q[idx + 1]
    dot = np.clip(np.sum(qa * qb, axis=1, keepdims=True), -1.0, 1.0)
    angle = np.arccos(dot)
    small = angle < 1e-6
    denom = np.where(small, 1.0, np.sin(angle))
    w0 = np.where(small, 1.0 - alpha, np.sin((1.0 - alpha) * angle) / denom)
    w1 = np.where(small, alpha, np.sin(alpha * angle) / denom)
    out_quat = w0 * qa + w1 * qb
    out_quat /= np.linalg.norm(out_quat, axis=1, keepdims=True)
    dt = 1.0 / fps_out
    out_jv = np.gradient(out_jp, dt, axis=0)
    out_linvel = np.gradient(out_pos, dt, axis=0)
    out_angvel = np.zeros((len(out_quat), 3), np.float64)
    for t in range(len(out_quat)):
        a, b = max(0, t - 1), min(len(out_quat) - 1, t + 1)
        if a == b:
            continue
        left, right = out_quat[a], out_quat[b]
        # q_delta = q_right * inverse(q_left), wxyz convention.
        w1, x1, y1, z1 = right
        w2, x2, y2, z2 = left * np.array([1, -1, -1, -1])
        delta = np.array([w1*w2-x1*x2-y1*y2-z1*z2,
                          w1*x2+x1*w2+y1*z2-z1*y2,
                          w1*y2-x1*z2+y1*w2+z1*x2,
                          w1*z2+x1*y2-y1*x2+z1*w2])
        if delta[0] < 0:
            delta = -delta
        theta = 2 * np.arctan2(np.linalg.norm(delta[1:]), delta[0])
        if theta > 1e-12:
            out_angvel[t] = delta[1:] / np.linalg.norm(delta[1:]) * theta / ((b-a)*dt)
    return out_jp, out_jv, out_pos, out_quat, out_linvel, out_angvel


def names_from_model(model, mujoco):
    joints = []
    for jid in range(model.njnt):
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE:
            joints.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid))
    bodies = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) for bid in range(1, model.nbody)]
    return joints, bodies


def convert(source: Path, model_file: Path, output: Path, target_fps: float = 50.0) -> dict:
    import mujoco

    meta = json.loads((source / "meta.json").read_text(encoding="utf-8"))
    source_joint_names = [n for n in meta["joint_names"] if n != "root"]
    if len(set(source_joint_names)) != len(source_joint_names):
        raise ValueError("duplicate HDMI joint name")
    with np.load(source / "motion.npz", allow_pickle=False) as src:
        jp = np.asarray(src["joint_pos"], dtype=np.float64)
        jv = np.asarray(src["joint_vel"], dtype=np.float64)
        root_pos = np.asarray(src["body_pos_w"][:, 0], dtype=np.float64)
        root_quat = np.asarray(src["body_quat_w"][:, 0], dtype=np.float64)
        root_linvel = np.asarray(src["body_lin_vel_w"][:, 0], dtype=np.float64)
        root_angvel = np.asarray(src["body_ang_vel_w"][:, 0], dtype=np.float64)
    if jp.ndim != 2 or jv.shape != jp.shape or jp.shape[1] != len(source_joint_names):
        raise ValueError("HDMI joint shapes do not match meta.json")
    if any(x.shape != (len(jp), d) for x, d in ((root_pos, 3), (root_quat, 4), (root_linvel, 3), (root_angvel, 3))):
        raise ValueError("HDMI root trajectory shape mismatch")
    if not all(np.isfinite(x).all() for x in (jp, jv, root_pos, root_quat, root_linvel, root_angvel)):
        raise ValueError("motion contains NaN or Inf")

    source_fps = float(meta["fps"])
    source_frames = len(jp)
    jp, jv, root_pos, root_quat, root_linvel, root_angvel = resample(
        jp, root_pos, root_quat, source_fps, target_fps)
    model = mujoco.MjModel.from_xml_path(str(model_file))
    data = mujoco.MjData(model)
    free = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    if len(free) != 1 or model.jnt_qposadr[free[0]] != 0 or model.jnt_dofadr[free[0]] != 0:
        raise ValueError("target MJCF needs one root freejoint at qpos/qvel offset 0")
    target_joints, body_names = names_from_model(model, mujoco)
    if set(target_joints) != set(source_joint_names) or len(target_joints) != 28:
        raise ValueError(f"target must have HDMI Vita28 hinge joints; missing={set(source_joint_names)-set(target_joints)}, extra={set(target_joints)-set(source_joint_names)}")
    if body_names[0] != "base_link" or "base_link" not in meta["body_names"]:
        raise ValueError("source and target root body must be base_link")
    target_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in target_joints]
    source_ids = [source_joint_names.index(n) for n in target_joints]
    n, b = len(jp), len(body_names)
    out = {"joint_pos": jp[:, source_ids].astype(np.float32),
           "joint_vel": jv[:, source_ids].astype(np.float32),
           "body_pos_w": np.empty((n, b, 3), np.float32),
           "body_quat_w": np.empty((n, b, 4), np.float32),
           "body_lin_vel_w": np.empty((n, b, 3), np.float32),
           "body_ang_vel_w": np.empty((n, b, 3), np.float32)}
    for t in range(n):
        data.qpos[:] = model.qpos0
        data.qvel[:] = 0
        data.qpos[:3] = root_pos[t]
        data.qpos[3:7] = root_quat[t] / np.linalg.norm(root_quat[t])
        # MuJoCo freejoint qvel angular velocity is world-frame.
        data.qvel[:3] = root_linvel[t]
        data.qvel[3:6] = root_angvel[t]
        for k, jid in enumerate(target_ids):
            data.qpos[model.jnt_qposadr[jid]] = jp[t, source_ids[k]]
            data.qvel[model.jnt_dofadr[jid]] = jv[t, source_ids[k]]
        mujoco.mj_forward(model, data)
        out["body_pos_w"][t] = data.xpos[1:]
        out["body_quat_w"][t] = data.xquat[1:]
        # cvel is [angular, linear] in world-aligned COM frame; mj_objectVelocity
        # returns velocity at the body origin in world coordinates.
        for bid in range(1, model.nbody):
            vel = np.empty(6, dtype=np.float64)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, bid, vel, 0)
            out["body_ang_vel_w"][t, bid - 1] = vel[:3]
            out["body_lin_vel_w"][t, bid - 1] = vel[3:]
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **out, fps=np.array([target_fps], np.float32))
    receipt = {"source": str(source.resolve()), "model": str(model_file.resolve()),
               "output": str(output.resolve()), "source_frames": source_frames,
               "source_fps": source_fps, "frames": n, "fps": target_fps,
               "joint_names": target_joints, "body_names": body_names,
               "root_position_max_error_m": float(np.max(np.abs(out["body_pos_w"][:, 0] - root_pos)))}
    output.with_suffix(".json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    return receipt


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True, help="HDMI directory with motion.npz and meta.json")
    p.add_argument("--mjcf", type=Path, required=True, help="validated HDMI Vita28 MJCF")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--target-fps", type=float, default=50.0)
    args = p.parse_args()
    print(json.dumps(convert(args.source, args.mjcf, args.output, args.target_fps), ensure_ascii=False, indent=2))
