"""Deterministic rollout of a trained tracking policy with physics-rate export of robot state and
contact-point-level contact forces against the ground and scene boxes.

Example:
  python scripts/vita/rollout_export.py algo=ppo_roa_train task=Vita/stair_rgb_145317 \
      checkpoint_path=/path/checkpoint_final.pt +export_dir=/path/out

Outputs (export_dir):
  rollout.npz   physics-rate (1/physics_dt) arrays, see keys below
  meta.json     names, rates, filter (surface) names, success flag, tracking errors
"""
import itertools
import json
import os
import sys

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from isaaclab.app import AppLauncher
from torchrl.envs.utils import set_exploration_type, ExplorationType

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import active_adaptation.learning  # noqa: F401  registers algo configs with hydra


@hydra.main(config_path="../../cfg", config_name="play", version_base=None)
def main(cfg):
    OmegaConf.set_struct(cfg, False)
    cfg.headless = True
    cfg.app.headless = True
    cfg.task.num_envs = 1
    cfg.task.scene_boxes.track_contact_points = True
    cfg.task.randomization = {}
    cfg.task.command.init_joint_pos_noise = 0.0
    cfg.task.command.init_joint_vel_noise = 0.0
    fixed_delay = int(cfg.get("export_action_delay", 2))
    cfg.task.action.min_delay = fixed_delay
    cfg.task.action.max_delay = fixed_delay
    cfg.task.action.alpha = [1.0, 1.0]
    OmegaConf.resolve(cfg)
    export_dir = cfg.export_dir
    os.makedirs(export_dir, exist_ok=True)

    app = AppLauncher(OmegaConf.to_container(cfg.app)).app
    from scripts.helpers import make_env_policy

    env, policy, vecnorm = make_env_policy(cfg)
    base_env = env.base_env
    robot = base_env.scene["robot"]
    command = base_env.command_manager
    dt = base_env.physics_dt
    origin = base_env.scene.env_origins[0]

    body_names = list(base_env.scene_contact_body_names)
    sensors = [base_env.scene[f"scene_contact_{b}"] for b in body_names]
    filter_names = ["ground"] + list(base_env.scene_box_names)
    for s in sensors:
        assert s.contact_physx_view.filter_count == len(filter_names)

    rec = {k: [] for k in [
        "time", "ref_frame", "root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w",
        "joint_pos", "joint_vel", "joint_pos_target", "applied_torque",
        "body_pos_w", "body_quat_w", "com_pos_w", "contact_force_matrix_w", "contact_friction_w"]}
    body_mass = robot.root_physx_view.get_masses()[0].to(base_env.device)
    total_mass = float(body_mass.sum())
    pts = {k: [] for k in ["frame", "body", "filter", "pos_w", "normal_w", "normal_force", "separation"]}
    fric = {k: [] for k in ["frame", "body", "filter", "pos_w", "force_w"]}
    frame_counter = itertools.count()

    def record(substep):
        f = next(frame_counter)
        rec["time"].append(f * dt)
        rec["ref_frame"].append(int(command.t[0].item()))
        rec["root_pos_w"].append((robot.data.root_link_pos_w[0] - origin).cpu().numpy())
        rec["root_quat_w"].append(robot.data.root_link_quat_w[0].cpu().numpy())
        rec["root_lin_vel_w"].append(robot.data.root_com_lin_vel_w[0].cpu().numpy())
        rec["root_ang_vel_w"].append(robot.data.root_com_ang_vel_w[0].cpu().numpy())
        rec["joint_pos"].append(robot.data.joint_pos[0].cpu().numpy())
        rec["joint_vel"].append(robot.data.joint_vel[0].cpu().numpy())
        rec["joint_pos_target"].append(robot.data.joint_pos_target[0].cpu().numpy())
        rec["applied_torque"].append(robot.data.applied_torque[0].cpu().numpy())
        rec["body_pos_w"].append((robot.data.body_link_pos_w[0] - origin).cpu().numpy())
        rec["body_quat_w"].append(robot.data.body_link_quat_w[0].cpu().numpy())
        com = (robot.data.body_com_pos_w[0] * body_mass[:, None]).sum(0) / total_mass
        rec["com_pos_w"].append((com - origin).cpu().numpy())
        # PhysX reports normal and friction parts separately; both are forces acting on the robot body
        fm = np.stack([s.data.force_matrix_w[0, 0].cpu().numpy() for s in sensors])  # [B, F, 3]
        rec["contact_force_matrix_w"].append(fm)
        rec["contact_friction_w"].append(np.stack([torch.nan_to_num(s.data.friction_forces_w[0, 0]).cpu().numpy() for s in sensors]))
        for bi, s in enumerate(sensors):
            view = s.contact_physx_view
            forces, points, normals, seps, cnt, start = view.get_contact_data(dt=dt)
            cnt, start = cnt[0].cpu().numpy(), start[0].cpu().numpy()
            for fi in range(len(filter_names)):
                n = int(cnt[fi])
                if n == 0:
                    continue
                sl = slice(int(start[fi]), int(start[fi]) + n)
                fn = forces[sl, 0].cpu().numpy()
                nrm = normals[sl].cpu().numpy()
                # orient normals so that normal_force * normal_w is the force acting on the robot body
                if np.dot((fn[:, None] * nrm).sum(0), fm[bi, fi]) < 0:
                    nrm = -nrm
                pts["frame"].append(np.full(n, f)); pts["body"].append(np.full(n, bi)); pts["filter"].append(np.full(n, fi))
                pts["pos_w"].append(points[sl].cpu().numpy() - origin.cpu().numpy())
                pts["normal_w"].append(nrm); pts["normal_force"].append(fn); pts["separation"].append(seps[sl, 0].cpu().numpy())
            ff, fp, fcnt, fstart = view.get_friction_data(dt=dt)
            fcnt, fstart = fcnt[0].cpu().numpy(), fstart[0].cpu().numpy()
            for fi in range(len(filter_names)):
                n = int(fcnt[fi])
                if n == 0:
                    continue
                sl = slice(int(fstart[fi]), int(fstart[fi]) + n)
                fric["frame"].append(np.full(n, f)); fric["body"].append(np.full(n, bi)); fric["filter"].append(np.full(n, fi))
                fric["pos_w"].append(fp[sl].cpu().numpy() - origin.cpu().numpy())
                fric["force_w"].append(ff[sl].cpu().numpy())

    rollout_policy = policy.get_rollout_policy("eval")
    base_env.eval()
    env.eval()
    env.set_seed(cfg.seed)
    td_ = env.reset()
    base_env._post_step_callbacks.append(record)
    terminated = False
    steps = 0
    with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
        for steps in range(base_env.max_episode_length):
            td_ = rollout_policy(td_)
            td, td_ = env.step_and_maybe_reset(td_)
            if td["next", "done"].any():
                terminated = bool(td["next", "terminated"].any())
                break
    base_env._post_step_callbacks.remove(record)

    out = {k: np.asarray(v) for k, v in rec.items()}
    for k, v in pts.items():
        out["contact_point_" + k] = np.concatenate(v) if v else np.zeros((0,) if k in ("frame", "body", "filter", "normal_force", "separation") else (0, 3))
    for k, v in fric.items():
        out["friction_point_" + k] = np.concatenate(v) if v else np.zeros((0,) if k in ("frame", "body", "filter") else (0, 3))

    # closure check: per-point normal forces must sum to the filtered (normal-only) force matrix
    T, B, F = out["contact_force_matrix_w"].shape[:3]
    recon = np.zeros((T, B, F, 3))
    np.add.at(recon, (out["contact_point_frame"].astype(int), out["contact_point_body"].astype(int), out["contact_point_filter"].astype(int)),
              out["contact_point_normal_force"][:, None] * out["contact_point_normal_w"])
    fm = out["contact_force_matrix_w"]
    mag = np.linalg.norm(fm, axis=-1)
    err = np.linalg.norm(recon - fm, axis=-1)
    closure_rel = float(err[mag > 10].sum() / max(mag[mag > 10].sum(), 1e-6))
    out["contact_total_force_w"] = fm + out["contact_friction_w"]

    # Newton check on the whole body: M * a_com = sum(contact forces) + M g (only sensed bodies are counted)
    com = out["com_pos_w"]
    a_com = (com[2:] - 2 * com[1:-1] + com[:-2]) / dt**2
    f_ext = out["contact_total_force_w"].sum(axis=(1, 2))[1:-1] + np.array([0.0, 0.0, -9.81 * total_mass])
    newton_res = np.linalg.norm(total_mass * a_com - f_ext, axis=-1)
    # 200 Hz double differencing is noisy; compare 50 ms moving averages
    k = max(1, int(round(0.05 / dt)))
    kern = np.ones(k) / k
    smooth = lambda x: np.stack([np.convolve(x[:, i], kern, mode="valid") for i in range(3)], -1)
    newton_res_smooth = np.linalg.norm(smooth(total_mass * a_com) - smooth(f_ext), axis=-1)

    # tracking error against the reference at the policy rate
    ds = command.dataset
    ref = ds.data[ds.starts[0]: ds.ends[0]]
    ref_frames = np.clip(out["ref_frame"], 0, len(ref) - 1)
    body_idx_motion = [ds.body_names.index(n) for n in robot.body_names if n in ds.body_names]
    body_idx_robot = [robot.body_names.index(ds.body_names[i]) for i in body_idx_motion]
    ref_body_pos = ref.body_pos_w[:, body_idx_motion].cpu().numpy()[ref_frames]
    pos_err = np.linalg.norm(out["body_pos_w"][:, body_idx_robot] - ref_body_pos, axis=-1)
    root_err = np.linalg.norm(out["root_pos_w"] - ref.body_pos_w[:, ds.body_names.index(command.root_body_name)].cpu().numpy()[ref_frames], axis=-1)

    np.savez_compressed(os.path.join(export_dir, "rollout.npz"), **out)
    motion_len = int(ds.lengths[0].item())
    meta = {
        "task": cfg.task.name,
        "checkpoint_path": str(cfg.checkpoint_path),
        "physics_dt": dt,
        "policy_dt": base_env.step_dt,
        "num_frames": int(T),
        "policy_steps": steps + 1,
        "motion_len_policy_steps": motion_len,
        "terminated_early": terminated,
        "success": (not terminated) and int(out["ref_frame"].max()) >= motion_len - 1,
        "joint_names": robot.joint_names,
        "body_names": robot.body_names,
        "contact_body_names": body_names,
        "contact_filter_names": filter_names,
        "xy_offset_note": "positions are env-local; add scene.json xy_offset to return to the UMR world frame",
        "scene_json": str(cfg.task.scene_boxes.json_path),
        "action_delay_physics_steps": fixed_delay,
        "mean_body_pos_err_m": float(pos_err.mean()),
        "max_body_pos_err_m": float(pos_err.max()),
        "mean_root_pos_err_m": float(root_err.mean()),
        "max_root_pos_err_m": float(root_err.max()),
        "contact_closure_rel_err": closure_rel,
        "total_mass_kg": total_mass,
        # impacts on bodies without a contact sensor show up as outliers, hence median / p95
        "newton_residual_median_over_mg": float(np.median(newton_res) / (9.81 * total_mass)) if len(newton_res) else None,
        "newton_residual_p95_over_mg": float(np.percentile(newton_res, 95) / (9.81 * total_mass)) if len(newton_res) else None,
        "newton_residual_50ms_median_over_mg": float(np.median(newton_res_smooth) / (9.81 * total_mass)) if len(newton_res_smooth) else None,
        "num_contact_points": int(len(out["contact_point_frame"])),
    }
    with open(os.path.join(export_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    summary = {k: meta[k] for k in ["success", "terminated_early", "policy_steps", "motion_len_policy_steps",
                                    "mean_body_pos_err_m", "max_body_pos_err_m", "mean_root_pos_err_m",
                                    "max_root_pos_err_m", "contact_closure_rel_err", "newton_residual_median_over_mg",
                                    "newton_residual_p95_over_mg", "newton_residual_50ms_median_over_mg", "num_contact_points"]}
    print("[EXPORT]", json.dumps(summary))
    # per-surface contact impulse summary
    for fi, fn in enumerate(filter_names):
        for bi, bn in enumerate(body_names):
            fz = out["contact_total_force_w"][:, bi, fi]
            active = np.linalg.norm(fz, axis=-1) > 1.0
            if active.any():
                print(f"[EXPORT] contact {bn:>24s} <- {fn:<12s} active {active.mean()*100:5.1f}% of frames, "
                      f"mean |F| when active {np.linalg.norm(fz[active], axis=-1).mean():7.1f} N")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
