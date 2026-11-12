"""Audit whether each reference frame admits bounded torque and frictional foot forces."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from scipy.optimize import lsq_linear

from common import classify_robot_static_bodies, configure_position_pd, load_canonical, sha256, write_json


def required_force(model, data, acceleration):
    """Unconstrained demand: contacts are supplied exactly once by the force fit."""
    mass_acceleration = np.empty(model.nv)
    mujoco.mj_mulM(model, data, mass_acceleration, acceleration)
    return mass_acceleration + data.qfrc_bias - data.qfrc_passive


def contact_rays(frame, friction):
    angles = np.arange(8) * (np.pi / 4)
    return (frame[0, :, None] + friction[0] * frame[1, :, None] * np.cos(angles)
            + friction[1] * frame[2, :, None] * np.sin(angles))


def fit_forces(required, actuator_matrix, effort, contact_columns, scales):
    matrix = np.hstack([actuator_matrix, *contact_columns])
    if not matrix.shape[1]:
        return np.empty(0), required.copy()
    rays = matrix.shape[1] - len(effort)
    fit = lsq_linear(matrix / scales[:, None], required / scales,
                     bounds=(np.r_[-effort, np.zeros(rays)],
                             np.r_[effort, np.full(rays, np.inf)]),
                     method="bvls", tol=1e-10, max_iter=1000)
    if not fit.success or not np.isfinite(fit.x).all():
        raise RuntimeError(f"Contact force fit failed: {fit.message}")
    return fit.x, required - matrix @ fit.x


def ranges(mask: np.ndarray) -> list[list[int]]:
    ids = np.flatnonzero(mask)
    if not len(ids):
        return []
    cuts = np.flatnonzero(np.diff(ids) > 1) + 1
    return [[int(chunk[0]), int(chunk[-1])] for chunk in np.split(ids, cuts)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--model-mjb", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mu", type=float, default=0.8)
    args = parser.parse_args()
    if not np.isfinite(args.mu) or args.mu <= 0:
        raise ValueError("mu must be finite and positive")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    reference = load_canonical(args.reference)
    model = mujoco.MjModel.from_binary_path(args.model_mjb)
    actuator = configure_position_pd(model, mujoco)["actuators"]
    robot_bodies, _, _ = classify_robot_static_bodies(model, mujoco)
    data = mujoco.MjData(model)
    frames = len(reference["qpos_ref"])
    dt = float(np.asarray(reference["dt"]).reshape(-1)[0])
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    qacc = np.gradient(reference["qvel_ref"], dt, axis=0, edge_order=min(2, frames - 1))
    effort = np.asarray([entry["effort_limit"] for entry in actuator], dtype=np.float64)
    foot_body = {}
    for side in (0, 1):
        token = "_l_" if side == 0 else "_r_"
        bodies = {
            int(model.geom_bodyid[geom]) for geom in range(model.ngeom)
            if int(model.geom_bodyid[geom]) in robot_bodies
            and ("foot" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or "")
                 or "toe" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""))
            and token in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or "")
        }
        if len(bodies) != 1:
            raise ValueError(f"Expected one foot body for side {side}, got {bodies}")
        foot_body[side] = bodies.pop()

    weight = float(sum(model.body_mass[list(robot_bodies)]) * abs(model.opt.gravity[2]))
    scales = np.r_[np.full(3, weight), np.full(3, weight * 0.5), effort]
    residual = np.empty((frames, model.nv), dtype=np.float64)
    normalized_rms = np.empty(frames)
    normalized_max = np.empty(frames)
    tau_ratio = np.empty(frames)
    contact_force = np.zeros((frames, 2, 3), dtype=np.float64)
    contact_moment = np.zeros_like(contact_force)
    contact_points = np.zeros((frames, 2), dtype=int)
    declared_without_manifold = np.zeros_like(contact_points, dtype=bool)
    required = np.empty((frames, model.nv), dtype=np.float64)
    tau = np.empty((frames, model.nu), dtype=np.float64)
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros_like(jacp)
    actuator_matrix = np.zeros((model.nv, model.nu), dtype=np.float64)
    for entry in actuator:
        aid = entry["actuator_id"]
        if (model.actuator_trntype[aid] != mujoco.mjtTrn.mjTRN_JOINT
                or model.jnt_type[entry["joint_id"]] != mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError("Audit requires scalar joint transmissions")
        actuator_matrix[entry["dof_address"], aid] = model.actuator_gear[aid, 0]
    for frame in range(frames):
        data.qpos[:] = reference["qpos_ref"][frame]
        data.qvel[:] = reference["qvel_ref"][frame]
        mujoco.mj_forward(model, data)
        required[frame] = required_force(model, data, qacc[frame])
        columns = []
        patches = []
        active = reference["contact_ref"][frame] > 0.5
        for contact in data.contact:
            b1, b2 = model.geom_bodyid[[contact.geom1, contact.geom2]]
            side = next((s for s, body in foot_body.items()
                         if active[s] and ((b1 == body and b2 not in robot_bodies)
                                           or (b2 == body and b1 not in robot_bodies))), None)
            if side is None or contact.efc_address < 0:
                continue
            body = foot_body[side]
            normal_sign = 1.0 if b2 == body else -1.0
            generators = normal_sign * contact_rays(
                contact.frame.reshape(3, 3), np.minimum(contact.friction[:2], args.mu))
            mujoco.mj_jac(
                model, data, jacp, jacr, contact.pos, body,
            )
            columns.append(jacp.T @ generators)
            patches.append((side, contact.pos.copy(), generators))
            contact_points[frame, side] += 1
        declared_without_manifold[frame] = active & (contact_points[frame] == 0)
        solution, residual[frame] = fit_forces(required[frame], actuator_matrix, effort, columns, scales)
        tau[frame] = solution[:model.nu]
        tau_ratio[frame] = float(np.max(np.abs(tau[frame]) / effort))
        cursor = model.nu
        for side, point, generators in patches:
            force = generators @ solution[cursor:cursor + 8]
            contact_force[frame, side] += force
            contact_moment[frame, side] += np.cross(point - data.xpos[foot_body[side]], force)
            cursor += 8
        normalized = residual[frame] / scales
        normalized_rms[frame] = float(np.sqrt(np.mean(normalized**2)))
        normalized_max[frame] = float(np.max(np.abs(normalized)))

    feasible = (normalized_rms <= 0.10) & (normalized_max <= 0.25)
    np.savez_compressed(
        output / "dynamics_feasibility.npz", required_generalized_force=required,
        residual_generalized_force=residual, normalized_residual_rms=normalized_rms,
        normalized_residual_max=normalized_max, actuator_torque_Nm=tau,
        actuator_effort_ratio=tau_ratio, contact_force_world_N=contact_force,
        contact_moment_world_Nm=contact_moment, actual_contact_point_count=contact_points,
        declared_without_manifold=declared_without_manifold,
        feasible=feasible, qacc=qacc,
    )
    report = {
        "schema": "stage2-framewise-contact-wrench-feasibility-v2",
        "reference_sha256": sha256(args.reference),
        "model_sha256": sha256(args.model_mjb),
        "audit_code_sha256": sha256(__file__),
        "force_demand": "M(q) a + bias - passive; no inverse-solver contact subtraction",
        "contact_model": "Native declared-foot contact manifold, eight-ray inscribed friction cones; no contact at separated feet",
        "declared_without_manifold_samples": int(declared_without_manifold.sum()),
        "frames": frames,
        "feasible_frames": int(feasible.sum()),
        "feasible_fraction": float(feasible.mean()),
        "infeasible_ranges": ranges(~feasible),
        "first_infeasible_frame": int(np.flatnonzero(~feasible)[0]) if np.any(~feasible) else None,
        "normalized_residual_rms_max": float(normalized_rms.max()),
        "normalized_residual_max": float(normalized_max.max()),
        "actuator_effort_ratio_max": float(tau_ratio.max()),
        "contact_frames": int(np.any(reference["contact_ref"] > 0.5, axis=1).sum()),
        "friction_mu": args.mu,
        "criterion": "RMS normalized dynamics residual <=0.10 and max <=0.25",
        "limitations": "Restricted-contact-model framewise screen, not a rollout certificate or a general infeasibility proof. Fixed q/v/a, declared feet only; no seat/hand contacts, torsional contact couples or impact impulses. Conservative friction discretization; no trajectory adjustment.",
    }
    write_json(output / "report.json", report)
    time = np.arange(frames) * dt
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(time, normalized_rms, label="normalized RMS")
    axes[0].axhline(0.10, color="red", linestyle="--", label="gate")
    axes[0].legend(); axes[0].set_ylabel("dynamics residual")
    axes[1].plot(time, normalized_max, label="normalized max")
    axes[1].axhline(0.25, color="red", linestyle="--", label="gate")
    axes[1].legend(); axes[1].set_ylabel("worst component")
    axes[2].plot(time, tau_ratio, label="max torque / limit")
    axes[2].axhline(1.0, color="red", linestyle="--", label="limit")
    axes[2].fill_between(time, 0, 1, where=~feasible, color="red", alpha=0.15,
                         transform=axes[2].get_xaxis_transform(), label="infeasible")
    axes[2].legend(); axes[2].set_ylabel("effort ratio"); axes[2].set_xlabel("time (s)")
    fig.suptitle(f"{Path(args.reference).stem}: framewise dynamics/contact feasibility")
    fig.tight_layout(); fig.savefig(output / "feasibility_heatmap.png", dpi=180)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
