"""Record a finite physical rollout of a saved mjlab PPO smoke checkpoint."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

from rl_cfg import make_runner_cfg
from task import make_env_cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qpos-output", type=Path,
                        help="Save actual simulated Vita root and joints for scene-mesh rendering")
    parser.add_argument("--dynamics-output", type=Path,
                        help="Save actuator force and PD joint target for startup diagnosis")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames", type=int, default=360)
    parser.add_argument("--strict-no-reset", action="store_true",
                        help="Stop at the first termination; never render an auto-reset state")
    parser.add_argument("--ccd-iterations", type=int,
                        help="MuJoCo CCD iterations; use the training value for matched replay")
    parser.add_argument("--profile", choices=("baseline", "root_adaptive", "root_tight", "root_start",
                                                "root_start_limbs", "root_adaptive_limbs",
                                                "precision_start", "precision_start_limbs",
                                                "getup_teacher_long"),
                        default="baseline")
    args = parser.parse_args()
    if args.ccd_iterations is not None and args.ccd_iterations < 50:
        parser.error("--ccd-iterations must be at least 50")
    cfg = make_env_cfg(str(args.motion), str(args.scene), play=True, profile=args.profile)
    if args.ccd_iterations is not None:
        cfg.sim.mujoco.ccd_iterations = args.ccd_iterations
    cfg.viewer.width, cfg.viewer.height = 960, 540
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device, render_mode="rgb_array")
    wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
    rl_cfg = make_runner_cfg()
    runner = MjlabOnPolicyRunner(wrapped, asdict(rl_cfg), device=args.device)
    runner.load(str(args.checkpoint), load_cfg={"actor": True}, strict=True,
                map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    terminations = 0
    rewards = []
    failures = []
    max_anchor_error_m = 0.0
    max_body_error_m = 0.0
    max_ee_z_error_m = 0.0
    executed_qpos = []
    executed_motion_frames = []
    executed_done = []
    dynamics_force = []
    dynamics_target = []
    dynamics_joint_pos = []
    dynamics_joint_vel = []
    dynamics_motion_frame = []
    robot = env.scene["robot"]
    try:
        obs, _ = wrapped.reset()
        # mjlab 1.2 initializes MotionCommand's cached relative body targets
        # only at the end of the first step.  Without this explicit frame-0
        # refresh, every rollout reports a spurious ee_body_pos termination at
        # frame 0 even though reset wrote the exact reference state.
        command = env.command_manager.get_term("motion")
        command.time_steps.fill_(-1)
        command._update_command()
        with imageio.get_writer(str(args.output), fps=50, codec="libx264", quality=7) as writer:
            for _ in range(args.frames):
                frame = env.render()
                writer.append_data(frame)
                command = env.command_manager.get_term("motion")
                motion_frame = int(command.time_steps[0].item())
                if args.dynamics_output is not None:
                    dynamics_force.append(robot.data.actuator_force[0].detach().cpu().numpy().copy())
                    dynamics_target.append(robot.data.joint_pos_target[0].detach().cpu().numpy().copy())
                    dynamics_joint_pos.append(robot.data.joint_pos[0].detach().cpu().numpy().copy())
                    dynamics_joint_vel.append(robot.data.joint_vel[0].detach().cpu().numpy().copy())
                    dynamics_motion_frame.append(motion_frame)
                if args.qpos_output is not None:
                    root_pose = robot.data.root_link_pose_w[0].detach().cpu().numpy()
                    joint_pos = robot.data.joint_pos[0].detach().cpu().numpy()
                    executed_qpos.append(np.concatenate((root_pose, joint_pos)).astype(np.float32))
                    executed_motion_frames.append(motion_frame)
                anchor_error_m = float(torch.linalg.vector_norm(
                    command.anchor_pos_w[0] - command.robot_anchor_pos_w[0]).item())
                anchor_z_error_m = float(torch.abs(
                    command.anchor_pos_w[0, 2] - command.robot_anchor_pos_w[0, 2]).item())
                body_error_m = float(torch.linalg.vector_norm(
                    command.body_pos_relative_w[0] - command.robot_body_pos_w[0], dim=-1
                ).max().item())
                ee_names = env.termination_manager.get_term_cfg("ee_body_pos").params["body_names"]
                ee_z_errors = {
                    name: float(torch.abs(
                        command.body_pos_relative_w[0, command.cfg.body_names.index(name), 2]
                        - command.robot_body_pos_w[0, command.cfg.body_names.index(name), 2]
                    ).item())
                    for name in ee_names
                }
                max_anchor_error_m = max(max_anchor_error_m, anchor_error_m)
                max_body_error_m = max(max_body_error_m, body_error_m)
                max_ee_z_error_m = max(max_ee_z_error_m, *ee_z_errors.values())
                with torch.inference_mode():
                    action = policy(obs)
                obs, reward, done, _ = wrapped.step(action)
                rewards.append(float(reward[0].item()))
                terminations += int(done.sum().item())
                if args.qpos_output is not None:
                    executed_done.append(bool(done[0].item()))
                if bool(done[0].item()):
                    reasons = [name for name in env.termination_manager.active_terms
                               if bool(env.termination_manager.get_term(name)[0].item())]
                    failures.append({"rollout_frame": len(rewards) - 1,
                                     "motion_frame": motion_frame,
                                     "reasons": reasons,
                                     "anchor_error_m": anchor_error_m,
                                     "anchor_z_error_m": anchor_z_error_m,
                                     "max_body_error_m": body_error_m,
                                     "ee_z_errors_m": ee_z_errors})
                    if args.strict_no_reset:
                        break
    finally:
        env.close()
    if args.qpos_output is not None:
        qpos = np.asarray(executed_qpos, dtype=np.float32)
        if qpos.ndim != 2 or qpos.shape[1] != 35 or not np.isfinite(qpos).all():
            raise ValueError(f"invalid executed Vita28 states: {qpos.shape}")
        args.qpos_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.qpos_output, qpos=qpos,
                            robot_joint_names=np.asarray(robot.joint_names),
                            fps=np.array([50.0], np.float32),
                            time_s=np.arange(len(qpos), dtype=np.float32) / 50.0,
                            motion_frame=np.asarray(executed_motion_frames, np.int32),
                            done=np.asarray(executed_done, bool))
    if args.dynamics_output is not None:
        args.dynamics_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.dynamics_output,
                            actuator_force=np.asarray(dynamics_force, np.float32),
                            joint_pos_target=np.asarray(dynamics_target, np.float32),
                            joint_pos=np.asarray(dynamics_joint_pos, np.float32),
                            joint_vel=np.asarray(dynamics_joint_vel, np.float32),
                            joint_names=np.asarray(robot.joint_names),
                            actuator_names=np.asarray(robot.actuator_names),
                            motion_frame=np.asarray(dynamics_motion_frame, np.int32))
    executed_frames = len(rewards)
    receipt = {"frames": executed_frames, "requested_frames": args.frames,
               "completed_full_clip": executed_frames == args.frames and terminations == 0,
               "strict_no_reset": args.strict_no_reset,
               "fps": 50, "terminations": terminations,
               "mean_reward": sum(rewards) / len(rewards), "checkpoint": str(args.checkpoint),
               "video": str(args.output), "failures": failures,
               "max_anchor_error_m": max_anchor_error_m,
               "max_body_error_m": max_body_error_m,
               "max_ee_z_error_m": max_ee_z_error_m,
               "executed_qpos": str(args.qpos_output) if args.qpos_output else None}
    if args.qpos_output is not None:
        receipt["evaluation"] = {"terminations": terminations,
                                 "max_anchor_error_m": max_anchor_error_m,
                                 "frames": executed_frames,
                                 "requested_frames": args.frames,
                                 "completed_full_clip": executed_frames == args.frames and terminations == 0}
    args.output.with_suffix(".json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
