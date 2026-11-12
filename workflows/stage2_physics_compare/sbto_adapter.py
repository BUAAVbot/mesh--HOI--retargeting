#!/usr/bin/env python3
"""Run the official SBTO CEM core on the frozen Vita28 Stage-2 contract.

The adapter deliberately keeps robot/scenario handling outside the upstream
repository.  It loads the calibrated binary MuJoCo scene, changes the existing
torque actuators into the same position-PD contract used by HDMI/MJLab, and
then calls SBTO's ``CEM`` and ``optimize_incremental_opt`` implementations.

Expected canonical input (``np.savez``):

* ``qpos_ref``: [F, model.nq], root quaternion is MuJoCo wxyz
* ``qvel_ref``: [F, model.nv]
* ``fps`` or ``source_fps``: scalar (defaults to 50)
* optional ``ctrl_ref`` [F, model.nu] and contact arrays

This file does not claim a dynamics certificate: SBTO is a sampling based
forward-simulation optimizer.  The normalized manifest records that fact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from common import (classify_robot_static_bodies,
                    configure_position_pd as _configure_shared_position_pd)


NATURAL_FREQ = 20.0 * np.pi
ARM_RP90S = 21.913**2 * 9.33e-5
ARM_LZ5735 = 35.0**2 * 1.23e-5
ARM_RP70H = 20.0**2 * 3.975e-5
ANKLE_ROLL_RATIO = 0.02175 / 0.05


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scalar(data: Any, keys: tuple[str, ...], default: float) -> float:
    for key in keys:
        if key in data:
            return float(np.asarray(data[key]).reshape(-1)[0])
    return default


def load_reference(path: Path, max_frames: int | None) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as source:
        qpos = np.asarray(source["qpos_ref"], dtype=np.float64)
        qvel = np.asarray(source["qvel_ref"], dtype=np.float64)
        fps = _scalar(source, ("fps", "source_fps"), 50.0)
        ctrl = (
            np.asarray(source["ctrl_ref"], dtype=np.float64)
            if "ctrl_ref" in source
            else None
        )
        contacts = {
            key: np.asarray(source[key])
            for key in ("contact_ref", "contact_pos_ref")
            if key in source
        }

    if qpos.ndim != 2 or qvel.ndim != 2 or len(qpos) != len(qvel):
        raise ValueError("qpos_ref/qvel_ref must be same-length rank-2 arrays")
    if len(qpos) < 2 or fps <= 0:
        raise ValueError("reference needs at least two frames and positive fps")
    if max_frames is not None:
        if max_frames < 2:
            raise ValueError("--max-frames must be at least 2")
        nframe = min(max_frames, len(qpos))
        qpos, qvel = qpos[:nframe], qvel[:nframe]
        if ctrl is not None:
            ctrl = ctrl[:nframe]
        contacts = {key: value[:nframe] for key, value in contacts.items()}
    return {"qpos": qpos, "qvel": qvel, "ctrl": ctrl, "fps": fps, **contacts}


def _interp_rows(values: np.ndarray, old_t: np.ndarray, new_t: np.ndarray) -> np.ndarray:
    return np.stack(
        [np.interp(new_t, old_t, values[:, col]) for col in range(values.shape[1])],
        axis=1,
    )


def geom_bounds(model, data, ids):
    ids=np.asarray(ids,dtype=int)
    rotation=data.geom_xmat[ids].reshape(-1,3,3)
    center=data.geom_xpos[ids]+np.einsum('nij,nj->ni',rotation,model.geom_aabb[ids,:3])
    extent=np.einsum('nij,nj->ni',np.abs(rotation),model.geom_aabb[ids,3:])
    return center-extent,center+extent


def nearby_supports(model,data,foot_geoms,support_ids,support_bounds,cutoff=.2):
    lower,upper=geom_bounds(model,data,foot_geoms)
    lo,hi=support_bounds
    mask=np.all(hi>=lower.min(axis=0)-cutoff-1e-9,axis=1)
    mask&=np.all(lo<=upper.max(axis=0)+cutoff+1e-9,axis=1)
    return np.asarray(support_ids)[mask]


def resample_reference(
    qpos: np.ndarray,
    qvel: np.ndarray,
    ctrl: np.ndarray | None,
    fps: float,
    dt: float,
    act_qposadr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    duration = (len(qpos) - 1) / fps
    nstep = max(1, int(round(duration / dt)))
    sim_t = np.arange(nstep + 1, dtype=np.float64) * dt
    sim_t[-1] = duration
    source_t = np.arange(len(qpos), dtype=np.float64) / fps

    qpos_sim = _interp_rows(qpos, source_t, sim_t)
    # Keep quaternion sign continuous before interpolation, then renormalize.
    quat = qpos[:, 3:7].copy()
    for frame in range(1, len(quat)):
        if np.dot(quat[frame - 1], quat[frame]) < 0:
            quat[frame] *= -1
    qpos_sim[:, 3:7] = _interp_rows(quat, source_t, sim_t)
    quat_norm = np.linalg.norm(qpos_sim[:, 3:7], axis=1, keepdims=True)
    qpos_sim[:, 3:7] /= np.maximum(quat_norm, 1e-12)
    qvel_sim = _interp_rows(qvel, source_t, sim_t)

    if ctrl is None:
        ctrl = qpos[:, act_qposadr]
    if ctrl.shape[1] != len(act_qposadr):
        raise ValueError(
            f"ctrl_ref width {ctrl.shape[1]} does not match model.nu {len(act_qposadr)}"
        )
    ctrl_sim = _interp_rows(ctrl, source_t, sim_t[:-1])
    return sim_t, qpos_sim, qvel_sim, ctrl_sim


def _joint_pd(joint_name: str) -> tuple[float, float, float, float]:
    if "hip_" in joint_name or "knee_joint" in joint_name:
        return (
            ARM_RP90S * NATURAL_FREQ**2,
            2.0 * ARM_RP90S * NATURAL_FREQ,
            ARM_RP90S,
            120.0,
        )
    if "ankle_pitch_joint" in joint_name:
        return 40.0, 2.0, 2.0 * ARM_LZ5735, 141.0
    if "ankle_roll_joint" in joint_name:
        ratio2 = ANKLE_ROLL_RATIO**2
        return 40.0 * ratio2, 2.0 * ratio2, 2.0 * ARM_LZ5735, 141.0
    if joint_name in ("waist_yaw_joint", "waist_pitch_joint"):
        return (
            ARM_LZ5735 * NATURAL_FREQ**2,
            2.0 * ARM_LZ5735 * NATURAL_FREQ,
            ARM_LZ5735,
            70.5,
        )
    if "shoulder_" in joint_name or "elbow_joint" in joint_name:
        return (
            ARM_RP70H * NATURAL_FREQ**2,
            2.0 * ARM_RP70H * NATURAL_FREQ,
            ARM_RP70H,
            60.0,
        )
    if "wrist_" in joint_name:
        return 20.0, 1.0, 0.00316508, 15.0
    raise ValueError(f"no HDMI PD gain contract for joint {joint_name!r}")


def configure_position_pd(model: Any, mujoco: Any) -> dict[str, list[float]]:
    if model.nu == 0:
        raise ValueError("compiled model has no actuators")
    gains: dict[str, list[float]] = {}
    model.actuator_gaintype[:] = mujoco.mjtGain.mjGAIN_FIXED
    model.actuator_biastype[:] = mujoco.mjtBias.mjBIAS_AFFINE
    model.actuator_dyntype[:] = mujoco.mjtDyn.mjDYN_NONE
    model.actuator_gainprm[:] = 0.0
    model.actuator_biasprm[:] = 0.0
    model.actuator_ctrllimited[:] = 1
    model.actuator_forcelimited[:] = 1

    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        joint_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
        )
        kp, kd, armature, effort = _joint_pd(joint_name)
        model.actuator_gainprm[actuator_id, 0] = kp
        model.actuator_biasprm[actuator_id, 1] = -kp
        model.actuator_biasprm[actuator_id, 2] = -kd
        model.actuator_forcerange[actuator_id] = (-effort, effort)
        model.actuator_ctrlrange[actuator_id] = model.jnt_range[joint_id]
        model.dof_armature[int(model.jnt_dofadr[joint_id])] = armature
        gains[joint_name] = [float(kp), float(kd), float(armature), float(effort)]
    return gains


class BinaryScene:
    """Small MJB scene shim implementing the interface used by SimMjRollout."""

    def __init__(self, model_path: Path, mujoco: Any):
        self.mujoco = mujoco
        self.mj_model = mujoco.MjModel.from_binary_path(str(model_path))
        self.mj_data = mujoco.MjData(self.mj_model)

    @property
    def Nq(self) -> int:
        return self.mj_model.nq

    @property
    def Nv(self) -> int:
        return self.mj_model.nv

    @property
    def Nu(self) -> int:
        return self.mj_model.nu

    @property
    def Nx(self) -> int:
        return self.Nq + self.Nv

    @property
    def Nobs(self) -> int:
        return self.mj_model.nsensordata

    @property
    def dt(self) -> float:
        return float(self.mj_model.opt.timestep)

    @property
    def act_joint_ids(self) -> np.ndarray:
        return self.mj_model.actuator_trnid[:, 0].astype(np.int64)

    @property
    def act_qposadr(self) -> np.ndarray:
        return self.mj_model.jnt_qposadr[self.act_joint_ids]

    @property
    def act_dofadr(self) -> np.ndarray:
        return self.mj_model.jnt_dofadr[self.act_joint_ids]

    @property
    def q_min(self) -> np.ndarray:
        return np.asarray(self.mj_model.jnt_range)[self.act_joint_ids, 0]

    @property
    def q_max(self) -> np.ndarray:
        return np.asarray(self.mj_model.jnt_range)[self.act_joint_ids, 1]

    @property
    def is_floating_base(self) -> bool:
        return bool(self.act_qposadr[0] > 0)

    def update_data(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        self.mj_data.qpos[:] = qpos
        self.mj_data.qvel[:] = qvel
        self.mujoco.mj_forward(self.mj_model, self.mj_data)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    sys.path.insert(0, str(args.sbto_root.resolve()))
    import mujoco
    from sbto.run.optimize import optimize_incremental_opt
    from sbto.sim.sim_mj_rollout import ConfigMjRollout, SimMjRollout
    from sbto.solvers.cem import CEM, ConfigCEM
    from sbto.tasks.cost import quadratic_cost_nb, quaternion_dist_logmap_nb
    from sbto.tasks.task_mj import TaskMj

    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    reference = load_reference(args.reference, args.max_frames)
    scene = BinaryScene(args.model, mujoco)
    scene.mj_model.opt.timestep = args.sim_dt
    gains = _configure_shared_position_pd(scene.mj_model, mujoco)["actuators"]
    if reference["qpos"].shape[1] != scene.Nq:
        raise ValueError("qpos_ref width does not match compiled model")
    if reference["qvel"].shape[1] != scene.Nv:
        raise ValueError("qvel_ref width does not match compiled model")

    sim_t, qpos_ref, qvel_ref, ctrl_ref = resample_reference(
        reference["qpos"],
        reference["qvel"],
        reference["ctrl"],
        reference["fps"],
        scene.dt,
        scene.act_qposadr,
    )
    nstep = len(sim_t) - 1
    if args.zero_initial_velocity:
        qvel_ref[0] = 0.0
    sim = SimMjRollout(
        scene,
        ConfigMjRollout(
            T=nstep,
            step_knots=min(args.step_knots, max(1, nstep // 2)),
            interp_kind="linear",
            Nthread=args.threads,
            _chunk_size=args.chunk_size,
        ),
    )
    x_ref = np.concatenate((qpos_ref, qvel_ref), axis=1)
    sim.set_initial_state(x_ref[0])
    scene.update_data(qpos_ref[0], qvel_ref[0])

    contact_plan = None
    if args.contact_aware:
        if "contact_ref" not in reference:
            raise ValueError("--contact-aware requires contact_ref in the canonical input")
        source_frame = np.clip(np.rint(sim_t[1:] * reference["fps"]).astype(int),
                               0, len(reference["contact_ref"]) - 1)
        contact_plan = np.asarray(reference["contact_ref"][source_frame] > 0.5, dtype=bool)

        class ContactAwareTask(TaskMj):
            def __init__(self, rollout):
                super().__init__(rollout)
                self.data = mujoco.MjData(scene.mj_model)
                robot, static, _ = classify_robot_static_bodies(scene.mj_model, mujoco)
                self.support = [g for g in range(scene.mj_model.ngeom)
                                if int(scene.mj_model.geom_bodyid[g]) in static]
                self.robot = robot
                self.feet = [
                    [g for g in range(scene.mj_model.ngeom)
                     if int(scene.mj_model.geom_bodyid[g]) in robot
                     and ("foot" in (mujoco.mj_id2name(scene.mj_model, mujoco.mjtObj.mjOBJ_GEOM, g) or "")
                          or "toe" in (mujoco.mj_id2name(scene.mj_model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""))
                     and token in (mujoco.mj_id2name(scene.mj_model, mujoco.mjtObj.mjOBJ_GEOM, g) or "")]
                    for token in ("_l_", "_r_")
                ]
                if not self.support or any(not foot for foot in self.feet):
                    raise ValueError("contact-aware SBTO could not classify support/foot geoms")
                mujoco.mj_forward(scene.mj_model,self.data)
                self.support_bounds=geom_bounds(scene.mj_model,self.data,self.support)

            def cost(self, x_traj, u_traj, obs_traj):
                total = super().cost(x_traj, u_traj, obs_traj)
                horizon = x_traj.shape[1]
                for candidate in range(x_traj.shape[0]):
                    contact_cost = 0.0
                    penetration_cost = 0.0
                    for step in range(horizon):
                        self.data.qpos[:] = x_traj[candidate, step, :scene.Nq]
                        self.data.qvel[:] = x_traj[candidate, step, scene.Nq:]
                        mujoco.mj_forward(scene.mj_model, self.data)
                        for side in range(2):
                            near=nearby_supports(scene.mj_model,self.data,self.feet[side],
                                                 self.support,self.support_bounds)
                            distance = min((float(mujoco.mj_geomDistance(
                                scene.mj_model, self.data, foot, support, 0.20, None))
                                for foot in self.feet[side] for support in near),default=.20)
                            if contact_plan[step, side]:
                                contact_cost += max(0.0, abs(distance) - args.contact_tolerance) ** 2
                            else:
                                penetration_cost += max(0.0, -distance - args.contact_tolerance) ** 2
                        for contact in self.data.contact[:self.data.ncon]:
                            body1 = int(scene.mj_model.geom_bodyid[int(contact.geom1)])
                            body2 = int(scene.mj_model.geom_bodyid[int(contact.geom2)])
                            if body1 in self.robot or body2 in self.robot:
                                penetration_cost += max(0.0, -float(contact.dist) - args.contact_tolerance) ** 2
                    total[candidate] += (args.contact_weight * contact_cost
                                         + args.penetration_weight * penetration_cost)
                return total

        task = ContactAwareTask(sim)
    else:
        task = TaskMj(sim)
    task.add_state_cost(
        "root_position",
        quadratic_cost_nb,
        np.arange(3),
        ref_values=qpos_ref[1:, :3],
        weights=np.asarray((args.root_xy_weight, args.root_xy_weight, args.root_z_weight)),
        weights_terminal=(np.asarray((200.0, 200.0, 300.0)) if args.terminal_root_scale is None
                          else args.terminal_root_scale * np.asarray((args.root_xy_weight, args.root_xy_weight, args.root_z_weight))),
    )
    task.add_state_cost(
        "root_orientation",
        quaternion_dist_logmap_nb,
        np.arange(3, 7),
        ref_values=qpos_ref[1:, 3:7],
        weights=args.root_orientation_weight,
        weights_terminal=args.root_orientation_weight * 2.0,
    )
    joint_qpos = np.arange(7, scene.Nq)
    task.add_state_cost(
        "joint_position",
        quadratic_cost_nb,
        joint_qpos,
        ref_values=qpos_ref[1:, 7:],
        weights=args.joint_weight,
        weights_terminal=args.joint_weight * 2.0,
    )
    task.add_state_cost(
        "velocity",
        quadratic_cost_nb,
        np.arange(scene.Nq, scene.Nx),
        ref_values=qvel_ref[1:],
        weights=args.velocity_weight,
        weights_terminal=args.velocity_weight,
    )
    task.add_control_cost(
        "pd_target",
        quadratic_cost_nb,
        np.arange(scene.Nu),
        ref_values=ctrl_ref,
        weights=args.control_weight,
        weights_terminal=args.control_weight,
    )

    solver = CEM(
        sim.Nvars_u,
        ConfigCEM(
            N_samples=args.samples,
            seed=args.seed,
            quasi_random=args.quasi_random,
            N_it=1,
            sigma0=args.sigma,
            elite_frac=max(2.0 / args.samples, min(0.1, (args.samples - 1) / args.samples)),
            alpha_mean=0.8,
            alpha_cov=0.2,
            std_incr=1e-8,
            keep_frac=0.5,
            min_std_collapsed=1e-10,
        ),
    )
    knot_indices = np.clip(sim.t_knots, 0, len(ctrl_ref) - 1)
    initial_knots = ctrl_ref[knot_indices]
    if args.warmstart_controls:
        with np.load(args.warmstart_controls, allow_pickle=False) as archive:
            warm_controls = np.asarray(archive['pd_target'], dtype=float)
        if warm_controls.shape != (nstep, scene.Nu) or not np.isfinite(warm_controls).all():
            raise ValueError('Warm-start controls must match the exact physics grid')
        initial_knots = warm_controls[knot_indices]
    solver.state.mean[:] = initial_knots.reshape(-1)
    solver.state.best[:] = solver.state.mean
    solver.state.best_all[:] = solver.state.mean

    state, _, _, costs, stats = optimize_incremental_opt(
        sim,
        task,
        solver,
        N_max_it_per_knots=args.max_iterations_per_knot,
        min_std_next=args.min_std_next,
        min_std_final=args.min_std_final,
    )
    best_knots = state.best_all.reshape(sim.Nknots, scene.Nu)
    time_rollout, state_rollout, control_rollout, _ = sim.rollout(
        best_knots[None, ...], with_x0=True
    )
    states = state_rollout[0]
    controls = control_rollout[0]
    elapsed = time.perf_counter() - started

    np.savez_compressed(
        args.output / "optimized_reference.npz",
        qpos=states[:, : scene.Nq],
        qvel=states[:, scene.Nq :],
        sim_time=np.asarray(time_rollout[0]).reshape(-1),
        source_fps=np.float64(reference["fps"]),
        backend=np.asarray("sbto_official_cem"),
    )
    np.savez_compressed(
        args.output / "controls.npz",
        pd_target=controls,
        control_knots=best_knots,
        knot_steps=sim.t_knots,
        sim_dt=np.float64(scene.dt),
    )
    np.savez_compressed(
        args.output / "physical_record.npz",
        qpos=states[:, : scene.Nq],
        qvel=states[:, scene.Nq :],
        ctrl=np.concatenate((controls[:1], controls), axis=0),
        sim_time=np.arange(len(states), dtype=np.float64) * scene.dt,
    )
    manifest = {
        "schema_version": "stage2-normalized-result.v1",
        "status": "completed_unvalidated",
        "method": "DynaRetarget/SBTO",
        "backend": "official sbto.solvers.cem.CEM + optimize_incremental_opt",
        "official_commit": args.sbto_commit,
        "reference": {"path": str(args.reference), "sha256": _sha256(args.reference)},
        "model": {"path": str(args.model), "sha256": _sha256(args.model)},
        "frames": int(len(reference["qpos"])),
        "physics_steps": int(nstep),
        "duration_s": float((len(reference["qpos"]) - 1) / reference["fps"]),
        "wall_time_s": elapsed,
        "optimizer": {
            "seed": args.seed,
            "samples": args.samples,
            "step_knots": args.step_knots,
            "max_iterations_per_knot": args.max_iterations_per_knot,
            "sigma": args.sigma,
            "minimum_cost": float(state.min_cost_all),
            "cost_evaluations": int(sum(len(row) for row in costs)),
            "stats_type": type(stats).__name__,
        },
        "pd_gains": gains,
        "certificate": {
            "available": False,
            "reason": "SBTO forward simulation does not emit a formal dynamics certificate",
        },
        "contact_schedule": {
            "present_in_canonical_input": "contact_ref" in reference,
            "used_by_optimizer": bool(args.contact_aware),
            "reason": ("exact MuJoCo foot/support distance and penetration cost"
                       if args.contact_aware else
                       "contact is evaluated only by uniform replay"),
            "weight": args.contact_weight if args.contact_aware else None,
            "penetration_weight": args.penetration_weight if args.contact_aware else None,
            "tolerance_m": args.contact_tolerance if args.contact_aware else None,
        },
        "zero_initial_velocity": bool(args.zero_initial_velocity),
        "validation": {"status": "pending_uniform_strict_replay"},
    }
    _write_json(args.output / "manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sbto-root", type=Path, required=True)
    parser.add_argument("--sbto-commit", default="cbed4ede80d16657585f93b2158fffbb3de9c33e")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--sim-dt", type=float, default=0.005,
                        help="Frozen HDMI MJLab physics timestep")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--step-knots", type=int, default=50)
    parser.add_argument("--max-iterations-per-knot", type=int, default=3)
    parser.add_argument("--threads", type=int, default=-1)
    parser.add_argument("--chunk-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sigma", type=float, default=0.08)
    parser.add_argument("--min-std-next", type=float, default=1e-3)
    parser.add_argument("--min-std-final", type=float, default=5e-4)
    parser.add_argument("--quasi-random", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--root-xy-weight", type=float, default=80.0)
    parser.add_argument("--root-z-weight", type=float, default=150.0)
    parser.add_argument("--root-orientation-weight", type=float, default=80.0)
    parser.add_argument("--joint-weight", type=float, default=15.0)
    parser.add_argument("--velocity-weight", type=float, default=1.0)
    parser.add_argument("--control-weight", type=float, default=0.2)
    parser.add_argument("--terminal-root-scale", type=float)
    parser.add_argument("--warmstart-controls", type=Path)
    parser.add_argument("--contact-aware", action="store_true")
    parser.add_argument("--contact-weight", type=float, default=2e6)
    parser.add_argument("--penetration-weight", type=float, default=2e7)
    parser.add_argument("--contact-tolerance", type=float, default=0.003)
    parser.add_argument("--zero-initial-velocity", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.samples < 3:
        raise ValueError("--samples must be >= 3")
    if args.terminal_root_scale is not None and (not np.isfinite(args.terminal_root_scale) or args.terminal_root_scale <= 0):
        raise ValueError('--terminal-root-scale must be finite and positive')
    manifest = run(args)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
