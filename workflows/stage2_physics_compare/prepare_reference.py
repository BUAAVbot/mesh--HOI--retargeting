"""Build one canonical input used unchanged by SPIDER, SBTO and PRIME."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from common import (classify_robot_static_bodies, configure_position_pd, sha256,
                    strip_namespace, write_json)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", required=True)
    parser.add_argument("--motion-json", required=True)
    parser.add_argument("--model-mjb", required=True)
    parser.add_argument("--scene-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--contact-distance", type=float, default=0.015)
    return parser.parse_args()


def _distance(model, data, geom_a: int, geom_b: int) -> tuple[float, np.ndarray]:
    fromto = np.zeros(6, dtype=np.float64)
    distance = mujoco.mj_geomDistance(model, data, geom_a, geom_b, 0.20, fromto)
    return float(distance), fromto


def main() -> None:
    args = _arguments()
    motion_path, model_path = Path(args.motion), Path(args.model_mjb)
    sidecar = json.loads(Path(args.motion_json).read_text(encoding="utf-8"))
    with np.load(motion_path, allow_pickle=False) as source:
        joint_pos = np.asarray(source["joint_pos"], dtype=np.float64)
        source_joint_vel = np.asarray(source["joint_vel"], dtype=np.float64)
        body_pos = np.asarray(source["body_pos_w"], dtype=np.float64)
        body_quat = np.asarray(source["body_quat_w"], dtype=np.float64)
        body_lin_vel = np.asarray(source["body_lin_vel_w"], dtype=np.float64)
        body_ang_vel = np.asarray(source["body_ang_vel_w"], dtype=np.float64)
        fps = float(np.asarray(source["fps"]).reshape(-1)[0])
    if joint_pos.shape[1] != 28 or body_pos.shape[0] != joint_pos.shape[0]:
        raise ValueError(f"Unexpected Vita28 motion shapes: {joint_pos.shape}, {body_pos.shape}")

    model = mujoco.MjModel.from_binary_path(str(model_path))
    if (model.nq, model.nv, model.nu) != (35, 34, 28):
        raise ValueError(f"Unexpected model dimensions: nq={model.nq}, nv={model.nv}, nu={model.nu}")
    actuator_contract = configure_position_pd(model, mujoco)
    data = mujoco.MjData(model)
    model_joint_names = [
        strip_namespace(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id))
        for joint_id in range(1, model.njnt)
    ]
    if model_joint_names != sidecar["joint_names"]:
        raise ValueError("Motion/model joint order differs; refusing an implicit reorder")

    frames, dt = joint_pos.shape[0], 1.0 / fps
    qpos = np.empty((frames, model.nq), dtype=np.float64)
    qpos[:, :3] = body_pos[:, 0]
    qpos[:, 3:7] = body_quat[:, 0]
    qpos[:, 7:] = joint_pos
    qvel = np.zeros((frames, model.nv), dtype=np.float64)
    for frame in range(frames - 1):
        mujoco.mj_differentiatePos(model, qvel[frame], dt, qpos[frame], qpos[frame + 1])
    qvel[-1] = qvel[-2]
    qacc = np.gradient(qvel, dt, axis=0, edge_order=2)

    # PRIME requires an input torque trace.  The source is synthetic rather
    # than a force-sensor log, so seed PRIME with MuJoCo inverse dynamics at
    # the same q/v/a reference and record the provenance explicitly.
    tau_unclipped = np.zeros((frames, model.nv - 6), dtype=np.float64)
    inverse_data = mujoco.MjData(model)
    for frame in range(frames):
        inverse_data.qpos[:] = qpos[frame]
        inverse_data.qvel[:] = qvel[frame]
        inverse_data.qacc[:] = qacc[frame]
        mujoco.mj_inverse(model, inverse_data)
        tau_unclipped[frame] = inverse_data.qfrc_inverse[6:]
    effort_by_dof = np.zeros(model.nv - 6, dtype=np.float64)
    for entry in actuator_contract["actuators"]:
        effort_by_dof[entry["dof_address"] - 6] = entry["effort_limit"]
    if np.any(effort_by_dof <= 0.0):
        raise ValueError("Missing an effort limit for one or more actuated joints")
    tau = np.clip(tau_unclipped, -effort_by_dof, effort_by_dof)

    actuator_qpos = np.array([entry["qpos_address"] for entry in actuator_contract["actuators"]])
    ctrl = qpos[:, actuator_qpos]
    robot_body_ids, static_body_ids, robot_body_order = classify_robot_static_bodies(model, mujoco)
    body_names = sidecar.get("body_names") or [
        strip_namespace(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id))
        for body_id in robot_body_order
    ]
    if len(body_names) != body_pos.shape[1]:
        raise ValueError(f"Body name/pose count differs: {len(body_names)} != {body_pos.shape[1]}")
    support_geoms = [
        geom_id for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) in static_body_ids
    ]
    foot_geoms = [
        [geom_id for geom_id in range(model.ngeom)
         if ("foot" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "")
             or "toe" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""))
         and int(model.geom_bodyid[geom_id]) in robot_body_ids
         and "_l_" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "")],
        [geom_id for geom_id in range(model.ngeom)
         if ("foot" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "")
             or "toe" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""))
         and int(model.geom_bodyid[geom_id]) in robot_body_ids
         and "_r_" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "")],
    ]
    if any(not side for side in foot_geoms) or not support_geoms:
        raise ValueError(f"Missing support geoms: support={support_geoms}, feet={foot_geoms}")
    contact = np.zeros((frames, 2), dtype=np.float64)
    contact_pos = np.zeros((frames, 2, 3), dtype=np.float64)
    min_distance = np.full((frames, 2), np.inf, dtype=np.float64)
    for frame in range(frames):
        data.qpos[:] = qpos[frame]
        data.qvel[:] = qvel[frame]
        mujoco.mj_forward(model, data)
        for side in range(2):
            best_fromto = None
            for foot_geom in foot_geoms[side]:
                for support_geom in support_geoms:
                    distance, fromto = _distance(model, data, foot_geom, support_geom)
                    if distance < min_distance[frame, side]:
                        min_distance[frame, side], best_fromto = distance, fromto.copy()
            contact[frame, side] = min_distance[frame, side] <= args.contact_distance
            if best_fromto is not None:
                contact_pos[frame, side] = 0.5 * (best_fromto[:3] + best_fromto[3:])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        # Canonical names used by our adapters.
        qpos_ref=qpos, qvel_ref=qvel, ctrl_ref=ctrl,
        contact_ref=contact, contact_pos_ref=contact_pos,
        # Exact aliases expected by the unmodified SPIDER data loader.
        qpos=qpos, qvel=qvel, ctrl=ctrl,
        contact=contact, contact_pos=contact_pos,
        tau_ref=tau,
        joint_pos=joint_pos, joint_vel=source_joint_vel,
        body_pos_w=body_pos, body_quat_w=body_quat,
        body_lin_vel_w=body_lin_vel, body_ang_vel_w=body_ang_vel,
        joint_names=np.asarray(sidecar["joint_names"], dtype="U"),
        body_names=np.asarray(body_names, dtype="U"),
        contact_min_distance_m=min_distance,
        fps=np.array([fps]), dt=np.array([dt]),
    )
    manifest = {
        "schema": "stage2-canonical-reference-v1",
        "frames": frames, "fps": fps, "duration_s": (frames - 1) / fps,
        "motion": str(motion_path), "motion_sha256": sha256(motion_path),
        "model_mjb": str(model_path), "model_mjb_sha256": sha256(model_path),
        "scene_json": args.scene_json, "scene_json_sha256": sha256(args.scene_json),
        "motion_sidecar": args.motion_json, "motion_sidecar_sha256": sha256(args.motion_json),
        "contact_schedule": {
            "method": "reference-FK minimum geom distance",
            "threshold_m": args.contact_distance,
            "left_fraction": float(contact[:, 0].mean()),
            "right_fraction": float(contact[:, 1].mean()),
        },
        "tau_reference": {
            "method": "MuJoCo inverse dynamics evaluated at canonical q/v/finite-difference acceleration",
            "measured": False,
            "effort_limited_seed": True,
            "width": int(tau.shape[1]),
            "max_abs_Nm": float(np.max(np.abs(tau))),
            "unclipped_max_abs_Nm": float(np.max(np.abs(tau_unclipped))),
            "clipped_fraction": float(np.mean(np.abs(tau_unclipped) > effort_by_dof[None, :])),
        },
        "actuation": actuator_contract,
        "canonical_reference_sha256": sha256(output),
    }
    write_json(output.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

