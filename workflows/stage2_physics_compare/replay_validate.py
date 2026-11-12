"""Fresh no-reset MuJoCo replay, frozen metrics, and optional video rendering."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from audit_dynamics_feasibility import required_force

from common import (classify_robot_static_bodies, configure_position_pd,
                    load_canonical, physics_substeps, sha256, write_json)

TRACK_BODIES = (
    "base_link", "waist_pitch_link", "left_hip_pitch_link", "right_hip_pitch_link",
    "left_knee_link", "right_knee_link", "left_ankle_roll_link", "right_ankle_roll_link",
    "left_shoulder_pitch_link", "right_shoulder_pitch_link", "left_elbow_link",
    "right_elbow_link", "left_wrist_pitch_link", "right_wrist_pitch_link",
)
EE_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link",
             "left_wrist_pitch_link", "right_wrist_pitch_link")


def _angular_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return 2.0 * np.arccos(np.clip(np.abs(np.sum(a * b, axis=-1)), 0.0, 1.0))


def _load_controls(path: Path, method: str, model) -> tuple[np.ndarray, str]:
    with np.load(path, allow_pickle=False) as archive:
        if method in ("sbto", "dsms"):
            value = np.asarray(archive["pd_target"], dtype=np.float64)
            semantics = "pd_target"
        elif method == "spider":
            value = np.asarray(archive["ctrl"], dtype=np.float64)
            semantics = "pd_target"
        else:
            value = np.asarray(archive["generalized_force"], dtype=np.float64)
            semantics = "torque"
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError("Controls must be one finite [time, width] array")
    if method == "prime":
        if value.shape[1] == model.nv:
            value = value[:, 6:]
        if value.shape[1] != model.nv - 6:
            raise ValueError("PRIME generalized force must have nv or nv-6 columns")
        dof_order = model.jnt_dofadr[model.actuator_trnid[:, 0]] - 6
        value = value[:, dof_order]
    if value.shape[1] != model.nu:
        raise ValueError(f"Control width {value.shape[1]} != model.nu {model.nu}")
    return value, semantics


def _resample_controls(controls: np.ndarray, count: int) -> np.ndarray:
    if len(controls) == count:
        return controls
    old = np.linspace(0.0, 1.0, len(controls), dtype=np.float64)
    new = np.linspace(0.0, 1.0, count, dtype=np.float64)
    return np.stack([np.interp(new, old, controls[:, i]) for i in range(controls.shape[1])], axis=1)


def _physics_sample(model, data, robot_body_ids: set[int], static_body_ids: set[int], weight: float):
    penetration = 0.0
    support = 0.0
    wrench = np.zeros(6, dtype=np.float64)
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        body_a = int(model.geom_bodyid[contact.geom1])
        body_b = int(model.geom_bodyid[contact.geom2])
        robot_static = ((body_a in robot_body_ids and body_b in static_body_ids) or
                        (body_b in robot_body_ids and body_a in static_body_ids))
        if not robot_static:
            continue
        penetration = max(penetration, max(0.0, -float(contact.dist)))
        mujoco.mj_contactForce(model, data, contact_id, wrench)
        support += abs(float(wrench[0]))
    joint = data.qpos[7:]
    ranges = model.jnt_range[1:]
    excess = np.maximum(ranges[:, 0] - joint, 0.0)
    excess = np.maximum(excess, np.maximum(joint - ranges[:, 1], 0.0))
    limits = np.maximum(np.abs(model.actuator_forcerange), 1e-9).max(axis=1)
    effort_ratio = float(np.max(np.abs(data.actuator_force) / limits))
    return penetration, float(np.max(excess)), effort_ratio, support, support >= 0.1 * weight


def contact_loads(model, data, robot_bodies, foot_bodies):
    forces=np.zeros((len(foot_bodies),3));self_depth=0.;local=np.zeros(6)
    for i,c in enumerate(data.contact):
        a,b=map(int,model.geom_bodyid[[c.geom1,c.geom2]])
        if a in robot_bodies and b in robot_bodies:
            self_depth=max(self_depth,-float(c.dist))
        if (a in robot_bodies)==(b in robot_bodies):continue
        body=b if b in robot_bodies else a
        if body not in foot_bodies:continue
        mujoco.mj_contactForce(model,data,i,local)
        sign=1. if b==body else -1.
        forces[foot_bodies.index(body)]+=sign*c.frame.reshape(3,3).T@local[:3]
    return self_depth,forces


def _tracking_metrics(reference: dict[str, np.ndarray], record: dict[str, np.ndarray]) -> dict:
    source_bodies = tuple(map(str, reference["body_names"]))
    record_bodies = tuple(map(str, record["body_names"]))
    source_joints = tuple(map(str, reference["joint_names"]))
    record_joints = tuple(map(str, record["joint_names"]))
    body_ref_ids = [source_bodies.index(name) for name in TRACK_BODIES]
    body_out_ids = [record_bodies.index(name) for name in TRACK_BODIES]
    joint_out_ids = [record_joints.index(name) for name in source_joints]
    sp = reference["body_pos_w"][:, body_ref_ids]
    rp = record["body_pos_w"][:, body_out_ids]
    sq = reference["body_quat_w"][:, body_ref_ids]
    rq = record["body_quat_w"][:, body_out_ids]
    root = np.linalg.norm(rp[:, 0] - sp[:, 0], axis=-1)
    relative = np.linalg.norm((rp - rp[:, :1]) - (sp - sp[:, :1]), axis=-1)
    ee_ids = [TRACK_BODIES.index(name) for name in EE_BODIES]
    ee_z = np.abs(rp[:, ee_ids, 2] - sp[:, ee_ids, 2])
    root_angle = _angular_distance(rq[:, 0], sq[:, 0])
    body_angle = _angular_distance(rq, sq)
    joint_error = record["joint_pos"][:, joint_out_ids] - reference["joint_pos"]
    return {
        "root_max_m": float(root.max()),
        "root_rmse_m": float(np.sqrt(np.mean(root**2))),
        "final_root_error_m": float(root[-1]),
        "ee_z_max_m": float(ee_z.max()),
        "body_relative_max_m": float(relative.max()),
        "body_relative_rmse_m": float(np.sqrt(np.mean(relative**2))),
        "root_orientation_rmse_rad": float(np.sqrt(np.mean(root_angle**2))),
        "final_root_orientation_rad": float(root_angle[-1]),
        "body_orientation_rmse_rad": float(np.sqrt(np.mean(body_angle**2))),
        "joint_rmse_rad": float(np.sqrt(np.mean(joint_error**2))),
        "joint_error_max_rad": float(np.abs(joint_error).max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("spider", "sbto", "prime", "dsms"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--gates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--sim-dt", type=float, default=0.005)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reference = load_canonical(args.reference)
    model = mujoco.MjModel.from_binary_path(str(args.model))
    model.opt.timestep = args.sim_dt
    controls, semantics = _load_controls(args.controls, args.method, model)
    if semantics == "pd_target":
        configure_position_pd(model, mujoco)
        model.opt.timestep = args.sim_dt
    fps=float(np.asarray(reference['fps']).ravel()[0])
    substeps=physics_substeps(1./fps,args.sim_dt)
    total_steps = (len(reference["qpos_ref"]) - 1) * substeps
    controls = _resample_controls(controls, total_steps)
    data = mujoco.MjData(model)
    data.qpos[:] = reference["qpos_ref"][0]
    data.qvel[:] = reference["qvel_ref"][0]
    mujoco.mj_forward(model, data)
    robot_body_ids, static_body_ids, robot_body_order = classify_robot_static_bodies(model, mujoco)
    feet=[next(i for i in robot_body_order if (model.body(i).name or '').split('/')[-1]==side+'_ankle_roll_link')
          for side in ('left','right')]
    fk=mujoco.MjData(model)
    weight = float(np.sum(model.body_mass[list(robot_body_ids)]) * np.linalg.norm(model.opt.gravity))
    body_names = np.asarray([
        (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or "").rsplit("/", 1)[-1]
        for body_id in robot_body_order
    ], dtype="U")
    joint_names = np.asarray([
        (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or "").rsplit("/", 1)[-1]
        for joint_id in range(1, model.njnt)
    ], dtype="U")
    frames = len(reference["qpos_ref"])
    qpos = np.empty((frames, model.nq)); qvel = np.empty((frames, model.nv))
    body_pos = np.empty((frames, len(body_names), 3)); body_quat = np.empty((frames, len(body_names), 4))
    body_lin = np.empty_like(body_pos); body_ang = np.empty_like(body_pos)
    qpos[0], qvel[0] = data.qpos, data.qvel
    body_pos[0], body_quat[0] = data.xpos[robot_body_order], data.xquat[robot_body_order]
    body_ang[0], body_lin[0] = data.cvel[robot_body_order, :3], data.cvel[robot_body_order, 3:]
    penetration = np.empty((frames - 1, substeps)); joint_excess = np.empty_like(penetration)
    effort_ratio = np.empty_like(penetration); support_force = np.empty_like(penetration)
    supported = np.empty_like(penetration, dtype=bool)
    executed = np.empty((frames - 1, substeps, model.nu))
    dynamics_residual = np.empty_like(penetration)
    self_penetration=np.empty_like(penetration)
    foot_force=np.zeros((*penetration.shape,2,3))
    foot_slip=np.zeros((*penetration.shape,2))
    step = 0
    for frame in range(1, frames):
        for substep in range(substeps):
            velocity_before=data.qvel.copy()
            data.ctrl[:] = controls[step]
            executed[frame - 1, substep] = data.ctrl
            mujoco.mj_step(model, data)
            if not np.isclose(data.time,(step+1)*args.sim_dt,rtol=0,atol=1e-9):
                raise ValueError('Physics clock changed during replay; possible simulator reset')
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise ValueError('Non-finite physical state during replay')
            if np.any(data.xfrc_applied) or np.any(data.qfrc_applied):
                raise ValueError("Unexpected external force during strict replay")
            dynamics_residual[frame - 1, substep] = np.max(np.abs(
                required_force(model, data, data.qacc) - data.qfrc_actuator - data.qfrc_constraint))
            self_penetration[frame-1,substep],foot_force[frame-1,substep]=contact_loads(model,data,robot_body_ids,feet)
            for side,body in enumerate(feet):
                jacp,jacr=np.zeros((3,model.nv)),np.zeros((3,model.nv))
                point=data.xpos[body]+data.xmat[body].reshape(3,3)@np.array([0.,0.,-.072])
                mujoco.mj_jac(model,data,jacp,jacr,point,body)
                foot_slip[frame-1,substep,side]=np.linalg.norm((jacp@velocity_before)[:2])
            values = _physics_sample(model, data, robot_body_ids, static_body_ids, weight)
            penetration[frame - 1, substep], joint_excess[frame - 1, substep], effort_ratio[frame - 1, substep], support_force[frame - 1, substep], supported[frame - 1, substep] = values
            step += 1
        qpos[frame], qvel[frame] = data.qpos, data.qvel
        fk.qpos[:]=data.qpos;fk.qvel[:]=data.qvel;mujoco.mj_forward(model,fk)
        body_pos[frame], body_quat[frame] = fk.xpos[robot_body_order], fk.xquat[robot_body_order]
        body_ang[frame], body_lin[frame] = fk.cvel[robot_body_order, :3], fk.cvel[robot_body_order, 3:]
    record = {
        "qpos": qpos, "qvel": qvel, "joint_pos": qpos[:, 7:], "joint_vel": qvel[:, 6:],
        "body_pos_w": body_pos, "body_quat_w": body_quat,
        "body_lin_vel_w": body_lin, "body_ang_vel_w": body_ang,
        "joint_names": joint_names, "body_names": body_names,
        "fps": np.asarray([fps]), "times": np.arange(frames) / fps,
        "executed_controls": executed,
        "physics_penetration_m": penetration,
        "physics_joint_excess_rad": joint_excess,
        "physics_effort_ratio": effort_ratio,
        "physics_support_force_N": support_force,
        "physics_dynamics_residual": dynamics_residual,
        "physics_self_penetration_m":self_penetration,
        "physics_foot_force_world_N":foot_force,
        "physics_foot_xy_speed_m_s":foot_slip,
        "reset_count": np.asarray([0]), "state_writes_after_initialization": np.asarray([0]),
        "continuous_physics": np.asarray([True]),
    }
    record_path = args.output / "physical_record.npz"
    np.savez_compressed(record_path, **record)
    metrics = _tracking_metrics(reference, record)
    metrics.update({
        "penetration_max_m": float(penetration.max()),
        "joint_limit_excess_max_rad": float(joint_excess.max()),
        "effort_ratio_max": float(effort_ratio.max()),
        "support_fraction": float(supported.mean()),
        "self_penetration_max_m":float(self_penetration.max()),
        "foot_vertical_support_fraction":float(np.mean(foot_force[:,:,:,2].sum(axis=-1)>=.1*weight)),
        "dynamics_residual_max_abs":float(dynamics_residual.max()),
    })
    if 'stationary_mask' in reference:
        mask=np.broadcast_to(reference['stationary_mask'][:-1,None,:],foot_slip.shape)
        metrics['stationary_foot_slip_p95_m_s']=float(np.quantile(foot_slip[mask],.95)) if mask.any() else 0.
    gates = json.loads(args.gates.read_text(encoding="utf-8"))
    checks = {key: metrics[key] <= float(limit) for key, limit in gates["maximum"].items()}
    checks.update({key: metrics[key] >= float(limit) for key, limit in gates["minimum"].items()})
    report = {
        "schema": "stage2-fresh-replay-v1", "method": args.method,
        "passed": bool(all(checks.values())), "metrics": metrics, "checks": checks,
        "fresh_replay": {"performed": True, "continuous": True,
                         "no_midrollout_reset": True, "no_root_state_write": True,
                         "no_external_root_force": True},
        "robot_weight_N": weight, "control_semantics": semantics,
        "reference_sha256": sha256(args.reference), "controls_sha256": sha256(args.controls),
        "model_sha256": sha256(args.model), "validator_sha256": sha256(__file__),
        "mujoco_version": mujoco.__version__,
        "dynamics_residual_max_abs": float(dynamics_residual.max()),
        "physical_record_sha256": sha256(record_path),
    }
    write_json(args.output / "report.json", report)
    if args.video:
        args.video.parent.mkdir(parents=True, exist_ok=True)
        renderer = mujoco.Renderer(model, height=480, width=640)
        camera = mujoco.MjvCamera(); camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        camera.trackbodyid = 2; camera.distance = 3.5; camera.azimuth = 135; camera.elevation = -18
        images = []
        render_data = mujoco.MjData(model)
        for frame in range(frames):
            render_data.qpos[:] = qpos[frame]; render_data.qvel[:] = qvel[frame]
            mujoco.mj_forward(model, render_data)
            renderer.update_scene(render_data, camera=camera)
            images.append(renderer.render().copy())
        renderer.close()
        imageio.mimsave(args.video, images, fps=fps, macro_block_size=1)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
