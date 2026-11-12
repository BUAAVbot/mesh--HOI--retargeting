"""Frozen HDMI tracking policy with bounded residuals and continuous physics.

No state writes after initialization, root forces, automatic resets or hidden
termination relaxation: strict failures are recorded while physics continues.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import torch
import warp as wp
from tensordict import TensorDict


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class HdmiDynamics:
    def __init__(self, root, motion, scene, checkpoint, count=1, device="cuda:0", seed=42,
                 ccd_iterations=50, instrument=False, clip_targets=False):
        self.root = Path(root)
        sys.path.insert(0, str(self.root))
        from task import make_env_cfg, TRACK_BODIES
        from rl_cfg import make_runner_cfg
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
        from mjlab.sensor.contact_sensor import ContactMatch, ContactSensorCfg
        from ground_scene import configure_ground_scene
        self.count, self.device = count, device
        self.instrument = instrument
        self.clip_targets = clip_targets
        self.motion, self.scene_file, self.checkpoint = map(Path, (motion, scene, checkpoint))
        cfg = make_env_cfg(str(motion), str(scene), play=True, profile="precision_start_limbs")
        cfg.scene.num_envs = count
        # MuJoCo-Warp worlds are independent; identical origins avoid any
        # large-world rounding or scene-translation differences in search.
        cfg.scene.env_spacing = 0.0
        cfg.seed = seed
        cfg.sim.mujoco.ccd_iterations = ccd_iterations
        cfg.terminations = {}
        cfg.rewards = {}
        cfg.metrics = {}
        cfg.events = {}
        self.ground_metadata = configure_ground_scene(cfg,scene)
        if self.ground_metadata['calibrated_heightfield']:
            cfg.sim.nconmax=max(cfg.sim.nconmax,1024)
        self.support_sensor_names = []
        if instrument:
            for index,body in enumerate(self.ground_metadata['support_bodies']):
                name=f'trajectory_foot_support_{index}'
                self.support_sensor_names.append(name)
                cfg.scene.sensors += (ContactSensorCfg(
                    name=name,
                    primary=ContactMatch(mode="body", pattern=("left_ankle_roll_link", "right_ankle_roll_link"), entity="robot"),
                    secondary=ContactMatch(mode="body", pattern=body),
                    fields=("found", "force", "dist"), reduce="netforce",
                    secondary_policy="error", history_length=4),)
        for group in cfg.observations.values():
            group.enable_corruption = False
            group.nan_policy = "disabled"
        self.env = ManagerBasedRlEnv(cfg, device=device)
        self.wrapped = RslRlVecEnvWrapper(self.env, clip_actions=None)
        runner = MjlabOnPolicyRunner(self.wrapped, asdict(make_runner_cfg()), device=device)
        runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
        self.policy = runner.get_inference_policy(device=device)
        self.robot = self.env.scene["robot"]
        self.names = tuple(self.robot.joint_names)
        self.body_names = tuple(self.robot.body_names)
        self.track_names = tuple(TRACK_BODIES)
        self.track_ids = [self.body_names.index(n) for n in self.track_names]
        self.ee_ids = [self.body_names.index(n) for n in (
            "left_ankle_roll_link", "right_ankle_roll_link",
            "left_wrist_pitch_link", "right_wrist_pitch_link")]
        self.command = self.env.command_manager.get_term("motion")
        self.cmd_ee = [self.track_names.index(n) for n in (
            "left_ankle_roll_link", "right_ankle_roll_link",
            "left_wrist_pitch_link", "right_wrist_pitch_link")]
        self.effort_limits = np.asarray([141. if "ankle" in name else
            120. if "hip" in name or "knee" in name else
            70.5 if "waist" in name else 15. if "wrist" in name else 60. for name in self.names])
        m = self.env.sim.mj_model
        root_id = int(m.jnt_bodyid[0])
        self.robot_weight_N = float(m.body_mass[m.body_rootid == root_id].sum() * 9.81)
        self.static_geom = torch.as_tensor(m.body_rootid[m.geom_bodyid] != root_id,device=device)
        geom_names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i) for i in range(m.ngeom)]
        self.foot_geom = torch.as_tensor([[n.startswith('robot/') and ('foot_' in n or 'toe_' in n)
                 and ('_l_' in n if side==0 else '_r_' in n) for n in geom_names] for side in range(2)],device=device)
        self.contact_force_buffer=None
        action_term = self.env.action_manager.get_term("joint_pos")
        if tuple(action_term.target_names) != self.names:
            raise ValueError("Action target order differs from robot joint order; explicit remapping required")
        self.action_scales = action_term.scale[0].clone()
        self.action_offsets = action_term.offset.clone()
        self.blend_groups = torch.as_tensor([2 if "wrist" in name else
                       1 if "shoulder" in name or "elbow" in name else 0 for name in self.names], device=device)
        self.velocity_feedforward = self._tensor([.05 if "ankle" in name or "wrist" in name else 2/(20*np.pi) for name in self.names])
        with np.load(motion, allow_pickle=False) as z:
            self.frames = len(z["joint_pos"])
        self.source_sha = sha(motion)
        self.scene_sha = sha(scene)
        self.checkpoint_sha = sha(checkpoint)
        self.step_orig = self.env.sim.step
        self.env.sim.step = self._physics_step
        self.capture = False

    def _tensor(self, value):
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def _physics_step(self):
        self.step_orig()
        d = self.env.sim.wp_data
        contact = d.contact
        dist = wp.to_torch(contact.dist)
        world = wp.to_torch(contact.worldid).long()
        nacon = wp.to_torch(d.nacon).reshape(-1)[0]
        valid = (torch.arange(len(dist), device=self.device) < nacon) & (world >= 0) & (world < self.count)
        depths = torch.zeros(self.count, device=self.device)
        depths.scatter_reduce_(0, world[valid], (-dist[valid]).clamp_min(0), reduce="amax", include_self=True)
        # Read native arrays after the substep; EntityData caches are refreshed
        # by scene.update only after sim.step returns.
        q = wp.to_torch(d.qpos)[:, 7:]
        qd = wp.to_torch(d.qvel)[:, 6:]
        limits = self.robot.data.joint_pos_limits
        excess = torch.maximum(limits[..., 0] - q, q - limits[..., 1]).clamp_min(0).amax(-1)
        force = wp.to_torch(d.qfrc_actuator)[:, 6:]
        self.max_depth = torch.maximum(self.max_depth, depths)
        self.max_excess = torch.maximum(self.max_excess, excess)
        if self.capture:
            self.sub_depth.append(depths[0].item())
            self.sub_excess.append(excess[0].item())
            self.sub_effort.append(force[0].detach().cpu().numpy().copy())
            # Official Warp contact-force decoder handles pyramidal and
            # elliptic cones; retain only load-bearing foot-to-static pairs.
            from mujoco_warp._src.support import contact_force
            if self.contact_force_buffer is None:
                self.contact_ids=wp.array(np.arange(len(dist),dtype=np.int32),dtype=wp.int32,device=self.env.sim.wp_device)
                self.contact_force_buffer=wp.zeros(len(dist),dtype=wp.spatial_vector,device=self.env.sim.wp_device)
            contact_force(self.env.sim.wp_model,d,self.contact_ids,False,self.contact_force_buffer)
            normal_force=wp.to_torch(self.contact_force_buffer)[:,0]
            geom=wp.to_torch(contact.geom).long()
            pos=wp.to_torch(contact.pos)
            stored=np.zeros((2,128,3),np.float32);stored_force=np.zeros((2,128),np.float32)
            stored_mask=np.zeros((2,128),bool);overflow=np.zeros(2,bool)
            for side in range(2):
                select=valid & (world==0) & (normal_force>1.) & (
                    (self.foot_geom[side,geom[:,0]] & self.static_geom[geom[:,1]]) |
                    (self.foot_geom[side,geom[:,1]] & self.static_geom[geom[:,0]]))
                positions=pos[select].cpu().numpy();forces=normal_force[select].cpu().numpy()
                count=min(len(positions),128);overflow[side]=len(positions)>128
                stored[side,:count]=positions[:count];stored_force[side,:count]=forces[:count];stored_mask[side,:count]=True
            self.sub_contact_points.append(stored);self.sub_contact_force.append(stored_force)
            self.sub_contact_mask.append(stored_mask);self.sub_contact_overflow.append(overflow)
            self.sub_qpos.append(wp.to_torch(d.qpos)[0].cpu().numpy().copy())
        self.finite &= torch.isfinite(q).all(-1) & torch.isfinite(qd).all(-1)

    @torch.no_grad()
    def reset(self):
        self.wrapped.reset()
        self.command.time_steps.fill_(-1)
        self.command._update_command()
        self.env.sim.forward()
        self.env.scene.update(dt=self.env.physics_dt)
        self.env.sim.sense()
        obs = self.env.observation_manager.compute(update_history=False)
        self.max_depth = torch.zeros(self.count, device=self.device)
        self.max_excess = torch.zeros_like(self.max_depth)
        self.finite = torch.ones(self.count, device=self.device, dtype=torch.bool)
        self.sub_depth, self.sub_excess, self.sub_effort = [], [], []
        self.sub_contact_points,self.sub_contact_force,self.sub_contact_mask,self.sub_contact_overflow=[],[],[],[]
        self.sub_qpos=[]
        return TensorDict(obs, batch_size=[self.count])

    def _state(self):
        d = self.robot.data
        origin = self.env.scene.env_origins[0]
        root = torch.cat((d.root_link_pos_w[0] - origin, d.root_link_quat_w[0],
                          d.root_link_lin_vel_w[0], d.root_link_ang_vel_w[0]))
        return {"qpos": torch.cat((root[:7], d.joint_pos[0])).cpu().numpy().copy(),
                "qvel": wp.to_torch(self.env.sim.wp_data.qvel)[0].cpu().numpy().copy(),
                "root_state": root.cpu().numpy().copy(),
                "joint_pos": d.joint_pos[0].cpu().numpy().copy(),
                "joint_vel": d.joint_vel[0].cpu().numpy().copy(),
                "body_pos_w": (d.body_link_pos_w[0] - origin).cpu().numpy().copy(),
                "body_quat_w": d.body_link_quat_w[0].cpu().numpy().copy(),
                "body_lin_vel_w": d.body_link_lin_vel_w[0].cpu().numpy().copy(),
                "body_ang_vel_w": d.body_link_ang_vel_w[0].cpu().numpy().copy()}

    @torch.no_grad()
    def rollout(self, residuals, record=False, blends=None):
        residuals = self._tensor(residuals)
        if residuals.shape != (self.count, self.frames - 1, 28):
            raise ValueError(f"residual shape mismatch: {residuals.shape}")
        self.capture = record
        blend_coefficients = self._tensor(np.zeros((self.count, 3)) if blends is None else blends)
        if blend_coefficients.shape != (self.count, 3):
            raise ValueError("Blend coefficients must be [count,3] for legs+waist/arms/wrists")
        blend = blend_coefficients[:, self.blend_groups]
        obs = self.reset()
        states = [self._state()] if record else []
        actions, targets, support_forces,foot_support_forces = [], [], [], []
        max_root = torch.zeros(self.count, device=self.device)
        max_ee, max_body, sum_joint, sum_root, sum_body, sum_ori = [max_root.clone() for _ in range(6)]
        first_fail = torch.full((self.count,), self.frames, device=self.device, dtype=torch.long)
        failed_frames = torch.zeros_like(first_fail)
        for t in range(self.frames - 1):
            actor_action = self.policy(obs)
            target = self.command.joint_pos + self.velocity_feedforward * self.command.joint_vel
            limits = self.robot.data.joint_pos_limits
            target = torch.clamp(target, limits[...,0] + .002, limits[...,1] - .002)
            ref_action = (target - self.action_offsets) / self.action_scales
            action = (1-blend) * actor_action + blend * ref_action + residuals[:, t]
            if self.clip_targets:
                target = self.action_offsets + self.action_scales * action
                target = torch.clamp(target, limits[...,0] + .002, limits[...,1] - .002)
                action = (target-self.action_offsets) / self.action_scales
            obs, _, done, _ = self.wrapped.step(action)
            if torch.any(done):
                raise RuntimeError("Unexpected automatic reset")
            cmd, robot = self.command, self.robot
            root_error = torch.linalg.vector_norm(cmd.anchor_pos_w - cmd.robot_anchor_pos_w, dim=-1)
            ee_error = torch.abs(cmd.body_pos_relative_w[:, self.cmd_ee, 2]
                                 - robot.data.body_link_pos_w[:, self.ee_ids, 2]).amax(-1)
            body_error = torch.linalg.vector_norm(cmd.body_pos_relative_w
                             - robot.data.body_link_pos_w[:, self.track_ids], dim=-1)
            joint_error = (cmd.joint_pos - robot.data.joint_pos).square().mean(-1)
            root_dot = (cmd.anchor_quat_w * cmd.robot_anchor_quat_w).sum(-1).abs().clamp(0, 1)
            ori_error = 2 * torch.acos(root_dot)
            max_root = torch.maximum(max_root, root_error)
            max_ee = torch.maximum(max_ee, ee_error)
            max_body = torch.maximum(max_body, body_error.amax(-1))
            sum_joint += joint_error
            sum_root += root_error.square()
            sum_body += body_error.square().mean(-1)
            sum_ori += ori_error.square()
            bad = (root_error > .25) | (ee_error > .25) | (~self.finite)
            first_fail = torch.where(bad & (first_fail == self.frames), t + 1, first_fail)
            failed_frames += bad.long()
            if record:
                states.append(self._state())
                actions.append(action[0].cpu().numpy().copy())
                targets.append(robot.data.joint_pos_target[0].cpu().numpy().copy())
                if self.instrument:
                    # [feet, substeps, xyz], history newest-first. Use actual
                    # static-scene contacts, excluding all self-collisions.
                    forces = [self.env.scene[name].data.force_history[0] for name in self.support_sensor_names]
                    vertical = sum(history[:, :, 2].sum(0) for history in forces)
                    support_forces.append(vertical.abs().flip(0).cpu().numpy().copy())
                    perfoot=sum(history[:,:,2] for history in forces)
                    foot_support_forces.append(perfoot.abs().flip(1).T.cpu().numpy().copy())
        metrics = {"max_root_error_m": max_root, "max_ee_z_error_m": max_ee,
                   "max_body_error_m": max_body, "joint_rmse_rad": (sum_joint / (self.frames - 1)).sqrt(),
                   "root_rmse_m": (sum_root / (self.frames - 1)).sqrt(),
                   "body_rmse_m": (sum_body / (self.frames - 1)).sqrt(),
                   "root_orientation_rmse_rad": (sum_ori / (self.frames - 1)).sqrt(),
                   "terminal_root_error_m": root_error,
                   "terminal_orientation_error_rad": ori_error,
                   "first_failure_frame": first_fail, "failed_frames": failed_frames,
                   "max_penetration_m": self.max_depth, "max_joint_excess_rad": self.max_excess,
                   "finite": self.finite}
        # Rank feasibility before tracking quality; continuous physics is never reset.
        constraints = ((max_root, .25), (max_ee, .25), (max_body, .35),
            (metrics["joint_rmse_rad"], .3), (metrics["root_rmse_m"], .1),
            (metrics["body_rmse_m"], .15), (metrics["root_orientation_rmse_rad"], .25),
            (self.max_depth, .012), (self.max_excess, .02),
            (root_error, .1), (ori_error, .3))
        n_failed = sum((value > limit).float() for value, limit in constraints)
        violation = sum((value / limit - 1).clamp_min(0) for value, limit in constraints)
        cost = 10000 * n_failed + 1000 * violation + 50 * max_root + 30 * max_ee
        cost += 20 * metrics["body_rmse_m"] + 10 * metrics["joint_rmse_rad"]
        metrics["failed_gate_count"] = n_failed
        metrics["normalized_gate_violation"] = violation
        cost = torch.where(self.finite, cost, torch.full_like(cost, 1e9))
        metrics = {k: v.detach().cpu().numpy() for k, v in metrics.items()}
        recording = None
        if record:
            recording = {k: np.stack([s[k] for s in states]) for k in states[0]}
            recording.update(actions=np.asarray(actions), processed_targets=np.asarray(targets),
                             residual_actions=residuals[0].cpu().numpy(),
                             physics_penetration_m=np.asarray(self.sub_depth).reshape(-1, 4),
                             physics_joint_excess_rad=np.asarray(self.sub_excess).reshape(-1, 4),
                             physics_actuator_force_Nm=np.asarray(self.sub_effort).reshape(-1, 4, 28),
                             joint_names=np.asarray(self.names), robot_joint_names=np.asarray(self.names),
                             body_names=np.asarray(self.body_names), fps=np.array(50.),
                             times=np.arange(self.frames) * .02,
                             source_reference_sha256=np.array(self.source_sha),
                             scene_sha256=np.array(self.scene_sha), checkpoint_sha256=np.array(self.checkpoint_sha),
                             reset_count=np.array(0), state_writes_after_initialization=np.array(0),
                             continuous_physics=np.array(True))
            recording["reference_blend"] = blend_coefficients[0].cpu().numpy()
            recording["clip_control_targets_to_limits"] = np.array(self.clip_targets)
            recording["physics_effort_ratio"] = (np.abs(recording["physics_actuator_force_Nm"])
                                                   / self.effort_limits[None, None]).max(-1)
            recording["joint_effort_limits_Nm"] = self.effort_limits
            recording["robot_weight_N"] = np.array(self.robot_weight_N)
            if self.instrument:
                recording["physics_support_force_N"] = np.asarray(support_forces)
                recording["physics_foot_support_force_N"] = np.asarray(foot_support_forces)
                recording["support_filter_verified"] = np.array(True)
                recording["support_filter_semantics"] = np.array('exact_per_static_body_excludes_self')
            recording['physics_foot_contact_pos_w']=np.asarray(self.sub_contact_points).reshape(-1,4,2,128,3)
            recording['physics_foot_contact_normal_force_N']=np.asarray(self.sub_contact_force).reshape(-1,4,2,128)
            recording['physics_foot_contact_mask']=np.asarray(self.sub_contact_mask).reshape(-1,4,2,128)
            recording['physics_foot_contact_overflow']=np.asarray(self.sub_contact_overflow).reshape(-1,4,2)
            recording['physics_qpos']=np.asarray(self.sub_qpos).reshape(-1,4,35)
            recording['support_contact_points_verified']=np.array(True)
            recording['support_contact_points_semantics']=np.array('warp_contact_force_static_pairs_force_gt_1N')
        self.capture = False
        return cost.cpu().numpy(), metrics, recording

    def export_model(self, output):
        mujoco.mj_saveModel(self.env.sim.mj_model, str(output))

    def close(self):
        self.env.close()
