"""Validate that the first reference pose establishes and holds physical support."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from common import classify_robot_static_bodies, configure_position_pd, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--model-mjb", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--duration-s", type=float, default=0.2)
    args = parser.parse_args()
    with np.load(args.reference, allow_pickle=False) as archive:
        qpos = np.asarray(archive["qpos_ref"][0], dtype=np.float64)
    model = mujoco.MjModel.from_binary_path(args.model_mjb)
    actuator = configure_position_pd(model, mujoco)
    robot_bodies, static_bodies, _ = classify_robot_static_bodies(model, mujoco)
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    data.ctrl[:] = qpos[[entry["qpos_address"] for entry in actuator["actuators"]]]
    mujoco.mj_forward(model, data)
    foot_geoms = {
        side: {
            geom_id for geom_id in range(model.ngeom)
            if int(model.geom_bodyid[geom_id]) in robot_bodies
            and ("foot" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "")
                 or "toe" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""))
            and f"_{side}_" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "")
        }
        for side in ("l", "r")
    }
    if any(not geoms for geoms in foot_geoms.values()):
        raise ValueError(f"Missing foot geoms: {foot_geoms}")
    steps = max(1, int(round(args.duration_s / model.opt.timestep)))
    foot_force = np.zeros((steps, 2), dtype=np.float64)
    total_force = np.zeros(steps, dtype=np.float64)
    penetration = np.zeros(steps, dtype=np.float64)
    root_z = np.zeros(steps, dtype=np.float64)
    contact_force = np.zeros(6, dtype=np.float64)
    for step in range(steps):
        mujoco.mj_step(model, data)
        root_z[step] = data.qpos[2]
        for contact_id, contact in enumerate(data.contact[:data.ncon]):
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            body1, body2 = int(model.geom_bodyid[geom1]), int(model.geom_bodyid[geom2])
            if not ((body1 in robot_bodies and body2 in static_bodies)
                    or (body2 in robot_bodies and body1 in static_bodies)):
                continue
            mujoco.mj_contactForce(model, data, contact_id, contact_force)
            normal = max(0.0, float(contact_force[0]))
            total_force[step] += normal
            penetration[step] = max(penetration[step], max(0.0, -float(contact.dist)))
            robot_geom = geom1 if body1 in robot_bodies else geom2
            for side_index, side in enumerate(("l", "r")):
                if robot_geom in foot_geoms[side]:
                    foot_force[step, side_index] += normal

    mass = sum(float(model.body_mass[body_id]) for body_id in robot_bodies)
    weight = mass * abs(float(model.opt.gravity[2]))
    tail = max(1, int(round(0.05 / model.opt.timestep)))
    report = {
        "schema": "stage2-initial-support-hold-v1",
        "duration_s": args.duration_s,
        "physics_timestep_s": float(model.opt.timestep),
        "robot_weight_N": weight,
        "first_step_foot_force_N": foot_force[0].tolist(),
        "first_step_total_support_N": float(total_force[0]),
        "foot_contact_fraction": (foot_force > 1.0).mean(axis=0).tolist(),
        "tail_total_support_N_mean": float(total_force[-tail:].mean()),
        "tail_support_weight_ratio": float(total_force[-tail:].mean() / weight),
        "max_penetration_mm": float(penetration.max() * 1000.0),
        "root_z_drop_cm": float((qpos[2] - root_z.min()) * 100.0),
    }
    report["passed"] = bool(
        np.all(foot_force[0] > 1.0)
        and np.all((foot_force > 1.0).mean(axis=0) >= 0.5)
        and report["tail_support_weight_ratio"] >= 0.5
        and report["max_penetration_mm"] <= 5.0
        and report["root_z_drop_cm"] <= 2.0
    )
    write_json(args.output, report)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
