"""mjlab Vita28 motion-tracking task scaffold derived from HDMI Vita config.

This is the physics/MDP migration, not a port of HDMI's TensorDict PPO-ROA
teacher/student optimizer. Do not label a checkpoint trained with the stock
RSL-RL runner as an HDMI student checkpoint.
"""
from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, fields
from pathlib import Path

import mujoco
import torch
from mjlab.actuator import IdealPdActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.tasks.tracking import mdp
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.mdp.commands import MotionCommand
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, sample_uniform


ROOT = Path(__file__).resolve().parent
ROBOT_XML = ROOT / "assets" / "vita28_shoes.xml"
NATURAL_FREQ = 20.0 * 3.1415926535
ARM_RP90S = 21.913**2 * 9.33e-5
ARM_LZ5735 = 35.0**2 * 1.23e-5
ARM_RP70H = 20.0**2 * 3.975e-5
ARM_WRIST = 0.00316508
ANKLE_ROLL_RATIO = 0.02175 / 0.05
TRACK_BODIES = (
    "base_link", "waist_pitch_link",
    "left_hip_pitch_link", "right_hip_pitch_link",
    "left_knee_link", "right_knee_link",
    "left_ankle_roll_link", "right_ankle_roll_link",
    "left_shoulder_pitch_link", "right_shoulder_pitch_link",
    "left_elbow_link", "right_elbow_link",
    "left_wrist_pitch_link", "right_wrist_pitch_link",
)


def numerical_failure(env, asset_name: str = "robot",
                      max_root_position_m: float = 50.0,
                      max_root_linear_velocity_m_s: float = 100.0,
                      max_root_angular_velocity_rad_s: float = 200.0,
                      max_joint_position_rad: float = 20.0,
                      max_joint_velocity_rad_s: float = 200.0):
    """Reset a diverged simulation before non-finite actor observations escape."""
    data = env.scene[asset_name].data
    tensors = (
        data.root_link_pose_w, data.root_link_vel_w,
        data.joint_pos, data.joint_vel,
        data.body_link_pose_w, data.body_link_vel_w,
    )
    valid = torch.ones(data.joint_pos.shape[0], dtype=torch.bool, device=data.joint_pos.device)
    for value in tensors:
        valid &= torch.isfinite(value).flatten(start_dim=1).all(dim=1)
    # Parallel environments are laid out on a large world-space grid.  The
    # safety bound must use each environment's local origin, otherwise distant
    # but perfectly valid replicas are reset every step.
    root_pos_local = data.root_link_pos_w - env.scene.env_origins
    valid &= root_pos_local.abs().amax(dim=1) < max_root_position_m
    valid &= data.root_link_lin_vel_w.abs().amax(dim=1) < max_root_linear_velocity_m_s
    valid &= data.root_link_ang_vel_w.abs().amax(dim=1) < max_root_angular_velocity_rad_s
    valid &= data.joint_pos.abs().amax(dim=1) < max_joint_position_rad
    valid &= data.joint_vel.abs().amax(dim=1) < max_joint_velocity_rad_s
    return ~valid


@dataclass(kw_only=True)
class GetupResetMotionCommandCfg(MotionCommandCfg):
    """GetUp-style reset: uniform reference frames with a small perturbed pool."""

    perturb_ratio: float = 0.20
    hold_at_end: bool = True
    focus_frame_start: int = 0
    focus_frame_end: int = -1
    focus_ratio: float = 0.0

    def build(self, env):
        return GetupResetMotionCommand(self, env)


class GetupResetMotionCommand(MotionCommand):
    """Restore exact reference state and perturb only a configured subset.

    The stock tracking command perturbs every reset and samples another frame
    when the reference reaches its end.  GetUp instead uses audited full-state
    resets and a fixed episode.  This term keeps 80% of resets exact, perturbs
    20%, and holds the final reference pose until the eight-second timeout.
    """

    cfg: GetupResetMotionCommandCfg

    def _resample_command(self, env_ids: torch.Tensor):
        if self.cfg.sampling_mode == "start":
            self.time_steps[env_ids] = 0
        elif self.cfg.sampling_mode == "uniform":
            self._uniform_sampling(env_ids)
        else:
            assert self.cfg.sampling_mode == "adaptive"
            self._adaptive_sampling(env_ids)

        if self.cfg.focus_ratio > 0.0:
            if not 0.0 <= self.cfg.focus_ratio <= 1.0:
                raise ValueError(f"invalid focus_ratio: {self.cfg.focus_ratio}")
            low = max(0, int(self.cfg.focus_frame_start))
            high = min(int(self.cfg.focus_frame_end), self.motion.time_step_total - 1)
            if high < low:
                raise ValueError(f"invalid focus frame range: {low}..{high}")
            focus_mask = torch.rand(len(env_ids), device=self.device) < self.cfg.focus_ratio
            focus_ids = env_ids[focus_mask]
            if len(focus_ids):
                self.time_steps[focus_ids] = torch.randint(
                    low, high + 1, (len(focus_ids),), device=self.device
                )

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()
        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        perturb_mask = torch.rand(len(env_ids), device=self.device) < self.cfg.perturb_ratio
        perturb_ids = env_ids[perturb_mask]
        if len(perturb_ids):
            range_list = [
                self.cfg.pose_range.get(key, (0.0, 0.0))
                for key in ["x", "y", "z", "roll", "pitch", "yaw"]
            ]
            ranges = torch.tensor(range_list, device=self.device)
            rand = sample_uniform(
                ranges[:, 0], ranges[:, 1], (len(perturb_ids), 6), device=self.device
            )
            root_pos[perturb_ids] += rand[:, :3]
            root_ori[perturb_ids] = quat_mul(
                quat_from_euler_xyz(rand[:, 3], rand[:, 4], rand[:, 5]),
                root_ori[perturb_ids],
            )

            range_list = [
                self.cfg.velocity_range.get(key, (0.0, 0.0))
                for key in ["x", "y", "z", "roll", "pitch", "yaw"]
            ]
            ranges = torch.tensor(range_list, device=self.device)
            rand = sample_uniform(
                ranges[:, 0], ranges[:, 1], (len(perturb_ids), 6), device=self.device
            )
            root_lin_vel[perturb_ids] += rand[:, :3]
            root_ang_vel[perturb_ids] += rand[:, 3:]
            joint_pos[perturb_ids] += sample_uniform(
                lower=self.cfg.joint_position_range[0],
                upper=self.cfg.joint_position_range[1],
                size=(len(perturb_ids), joint_pos.shape[1]),
                device=joint_pos.device,
            )

        limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], limits[:, :, 0], limits[:, :, 1]
        )
        self.robot.write_joint_state_to_sim(
            joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids
        )
        root_state = torch.cat(
            [root_pos[env_ids], root_ori[env_ids],
             root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1
        )
        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
        self.robot.clear_state(env_ids=env_ids)

    def _update_command(self):
        if not self.cfg.hold_at_end:
            return super()._update_command()
        self.time_steps.add_(1).clamp_(max=self.motion.time_step_total - 1)

        anchor_pos = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_pos = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_quat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        delta_pos = robot_pos
        delta_pos[..., 2] = anchor_pos[..., 2]
        from mjlab.utils.lab_api.math import quat_apply, quat_inv, yaw_quat
        delta_ori = yaw_quat(quat_mul(robot_quat, quat_inv(anchor_quat)))
        self.body_quat_relative_w = quat_mul(delta_ori, self.body_quat_w)
        self.body_pos_relative_w = delta_pos + quat_apply(
            delta_ori, self.body_pos_w - anchor_pos
        )


def motion_global_anchor_xy_error_exp(env, command_name: str, std: float):
    """Dense horizontal tracking reward for the TrainVid2 takeoff segment."""
    command = env.command_manager.get_term(command_name)
    delta_xy = command.anchor_pos_w[:, :2] - command.robot_anchor_pos_w[:, :2]
    return torch.exp(-torch.sum(torch.square(delta_xy), dim=-1) / std**2)


def motion_start_joint_position_error_exp(env, command_name: str, std: float,
                                          joint_names: tuple[str, ...],
                                          fade_begin: int = 20, fade_end: int = 40):
    """Track reference leg posture during the startup contact transition."""
    command = env.command_manager.get_term(command_name)
    robot = env.scene["robot"]
    indexes = [robot.joint_names.index(name) for name in joint_names]
    delta = command.joint_pos[:, indexes] - command.robot_joint_pos[:, indexes]
    error = torch.square(delta).mean(dim=-1)
    phase_weight = torch.clamp((fade_end - command.time_steps).float()
                               / (fade_end - fade_begin), min=0.0, max=1.0)
    return torch.exp(-error / std**2) * phase_weight


def robot_cfg(xml: Path = ROBOT_XML) -> EntityCfg:
    def pd(patterns, armature, effort):
        return IdealPdActuatorCfg(
            target_names_expr=patterns,
            stiffness=armature * NATURAL_FREQ**2,
            damping=2.0 * armature * NATURAL_FREQ,
            armature=armature, effort_limit=effort,
        )

    return EntityCfg(
        spec_fn=lambda: mujoco.MjSpec.from_file(str(xml)),
        init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 0.891088), joint_pos={
            ".*": 0.0, ".*_hip_pitch_joint": -0.2, ".*_knee_joint": 0.4,
            ".*_ankle_pitch_joint": -0.2,
            "left_hip_roll_joint": 0.088, "right_hip_roll_joint": -0.088,
            "left_ankle_roll_joint": -0.088, "right_ankle_roll_joint": 0.088,
            "left_shoulder_roll_joint": 0.4226, "right_shoulder_roll_joint": -0.4226,
        }),
        articulation=EntityArticulationInfoCfg(
            actuators=(
                pd((".*_hip_.*_joint", ".*_knee_joint"), ARM_RP90S, 120.0),
                IdealPdActuatorCfg(
                    target_names_expr=(".*_ankle_pitch_joint",),
                    stiffness=40.0, damping=2.0,
                    armature=2 * ARM_LZ5735, effort_limit=141.0,
                ),
                IdealPdActuatorCfg(
                    target_names_expr=(".*_ankle_roll_joint",),
                    stiffness=40.0 * ANKLE_ROLL_RATIO**2,
                    damping=2.0 * ANKLE_ROLL_RATIO**2,
                    armature=2 * ARM_LZ5735, effort_limit=141.0,
                ),
                pd(("waist_yaw_joint", "waist_pitch_joint"), ARM_LZ5735, 70.5),
                pd((".*_shoulder_.*_joint", ".*_elbow_joint"), ARM_RP70H, 60.0),
                IdealPdActuatorCfg(
                    target_names_expr=(".*_wrist_.*_joint",),
                    stiffness=20.0, damping=1.0,
                    armature=ARM_WRIST, effort_limit=15.0,
                ),
            ),
        ),
    )


def scene_entities(scene_file: str) -> dict[str, EntityCfg]:
    """Add HDMI's static seat boxes as fixed mjlab entities."""
    scene = json.loads(Path(scene_file).read_text(encoding="utf-8"))
    boxes = scene["boxes"]
    xy_offset = scene.get("xy_offset", [0.0, 0.0])
    if (not isinstance(xy_offset, list) or len(xy_offset) != 2 or
            any(not isinstance(value, (int, float)) for value in xy_offset)):
        raise ValueError(f"invalid scene xy_offset: {xy_offset}")
    result = {}
    for i, box in enumerate(boxes):
        size = tuple(float(x) / 2 for x in box["size"])
        raw_pos = tuple(float(x) for x in box["pos"])
        # Mesh-derived calibration scenes keep support boxes in the source
        # grid frame and store the retarget-world translation separately.
        # Training must apply the same contract as collision QA and rendering;
        # otherwise the robot tracks metres away from the collision proxies.
        pos = (raw_pos[0] + float(xy_offset[0]),
               raw_pos[1] + float(xy_offset[1]), raw_pos[2])
        quat = tuple(float(x) for x in box["quat_wxyz"])
        if min(size) <= 0 or len(pos) != 3 or len(quat) != 4:
            raise ValueError(f"invalid scene box: {box}")
        xyz = " ".join(map(str, pos))
        q = " ".join(map(str, quat))
        half = " ".join(map(str, size))
        xml = (f'<mujoco><worldbody><body name="seat_{i}" pos="{xyz}" quat="{q}">'
               f'<geom name="seat_{i}_collision" type="box" size="{half}" '
               'friction="0.8 0.01 0.0005"/></body></worldbody></mujoco>')
        result[f"seat_{i}"] = EntityCfg(spec_fn=lambda text=xml: mujoco.MjSpec.from_string(text))
    return result


def make_env_cfg(motion_file: str, scene_file: str, *, play: bool = False,
                 profile: str = "baseline"):
    cfg = make_tracking_env_cfg()
    cfg.scene.entities = {"robot": robot_cfg(), **scene_entities(scene_file)}
    cfg.scene.num_envs = 1 if play else 4096
    cfg.sim.mujoco.timestep = 0.005
    # Vita's 47 collision geoms plus both seats produce 98 contacts at the
    # initial pose; mjwarp's stock contact buffer is too small for put_data.
    cfg.sim.nconmax = max(cfg.sim.nconmax, 256)
    cfg.sim.njmax = max(cfg.sim.njmax, 4096)
    cfg.decimation = 4  # HDMI Vita command and motion are 50 Hz.
    cfg.viewer.body_name = "base_link"
    action = cfg.actions["joint_pos"]
    assert isinstance(action, JointPositionActionCfg)
    action.scale = {
        ".*_hip_.*_joint": 0.17,
        ".*_knee_joint": 0.17,
        ".*_ankle_.*_joint": 0.44,
        "waist_.*_joint": 0.30,
        ".*_shoulder_.*_joint": 0.24,
        ".*_elbow_joint": 0.24,
        ".*_wrist_.*_joint": 0.19,
    }
    command = cfg.commands["motion"]
    assert isinstance(command, MotionCommandCfg)
    if profile == "getup_teacher_long":
        command = GetupResetMotionCommandCfg(**{
            field.name: deepcopy(getattr(command, field.name))
            for field in fields(MotionCommandCfg)
        })
        cfg.commands["motion"] = command
    command.motion_file = str(Path(motion_file).resolve())
    command.anchor_body_name = "base_link"
    command.body_names = TRACK_BODIES
    command.pose_range = {"x": (-0.03, 0.03), "y": (-0.03, 0.03),
                          "z": (0.0, 0.01), "roll": (-0.05, 0.05),
                          "pitch": (-0.05, 0.05), "yaw": (-0.1, 0.1)}
    command.velocity_range = {"x": (-0.1, 0.1), "y": (-0.1, 0.1),
                              "z": (-0.1, 0.1), "roll": (-0.2, 0.2),
                              "pitch": (-0.2, 0.2), "yaw": (-0.2, 0.2)}
    command.joint_position_range = (-0.05, 0.05)
    if profile != "baseline":
        if profile not in {"root_adaptive", "root_tight", "root_start",
                           "root_start_limbs", "root_adaptive_limbs",
                           "precision_start", "precision_start_limbs",
                           "precision_start_feet", "precision_adaptive_feet",
                           "precision_adaptive_anchor",
                           "takeoff_xy", "precision_start_legs",
                           "getup_teacher_long"}:
            raise ValueError(f"unknown training profile: {profile}")
        # The source motion changes seats by roughly two metres.  The stock
        # tracking task mostly scores anchor-relative poses, which lets a
        # policy stand in place while mimicking the limbs.  These profiles
        # explicitly make world-frame root tracking the primary objective.
        action.scale = {
            ".*_hip_.*_joint": 0.50,
            ".*_knee_joint": 0.50,
            ".*_ankle_.*_joint": 0.50,
            "waist_.*_joint": 0.40,
            ".*_shoulder_.*_joint": 0.40,
            ".*_elbow_joint": 0.40,
            ".*_wrist_.*_joint": 0.30,
        }
        root_reward = cfg.rewards["motion_global_root_pos"]
        precision = profile in {"precision_start", "precision_start_limbs",
                                "precision_start_feet", "precision_adaptive_feet",
                                "precision_adaptive_anchor",
                                "precision_start_legs"}
        root_reward.weight = 12.0 if precision else (15.0 if profile == "root_tight" else 8.0)
        root_reward.params["std"] = 0.12 if precision else (0.15 if profile == "root_tight" else 0.20)
        if profile == "takeoff_xy":
            # The reference is airborne from frame 20 onward. A broad XY
            # reward retains gradient when the policy lags horizontally by
            # 20–30 cm, while the original 3D reward preserves height.
            root_reward.weight = 12.0
            root_reward.params["std"] = 0.16
            root_xy_reward = deepcopy(root_reward)
            root_xy_reward.func = motion_global_anchor_xy_error_exp
            root_xy_reward.weight = 12.0
            root_xy_reward.params["std"] = 0.30
            cfg.rewards["motion_global_root_xy_pos"] = root_xy_reward
        body_reward = cfg.rewards["motion_body_pos"]
        body_reward.weight = 6.0 if precision else 2.0
        body_reward.params["std"] = 0.12 if precision else 0.20
        wrist_reward = deepcopy(body_reward)
        wrist_reward.weight = 6.0 if precision else 4.0
        wrist_reward.params = {
            "command_name": "motion", "std": 0.08 if precision else 0.12,
            "body_names": ("left_wrist_pitch_link", "right_wrist_pitch_link"),
        }
        cfg.rewards["motion_wrist_pos"] = wrist_reward
        ankle_reward = deepcopy(body_reward)
        ankle_reward.weight = 4.0 if precision else 2.0
        ankle_reward.params = {
            "command_name": "motion", "std": 0.07 if precision else 0.10,
            "body_names": ("left_ankle_roll_link", "right_ankle_roll_link"),
        }
        cfg.rewards["motion_ankle_pos"] = ankle_reward
        if profile == "precision_start_legs":
            for label, names, weight, std in (
                ("knees", ("left_knee_joint", "right_knee_joint"), 8.0, 0.7),
                ("hips", ("left_hip_pitch_joint", "left_hip_roll_joint",
                          "left_hip_yaw_joint", "right_hip_pitch_joint",
                          "right_hip_roll_joint", "right_hip_yaw_joint"), 4.0, 0.7),
            ):
                reward = deepcopy(root_reward)
                reward.func = motion_start_joint_position_error_exp
                reward.weight = weight
                reward.params = {"command_name": "motion", "std": std,
                                 "joint_names": names, "fade_begin": 20,
                                 "fade_end": 40}
                cfg.rewards[f"motion_start_{label}_pos"] = reward
        if profile in {"precision_start_feet", "precision_adaptive_feet",
                       "precision_adaptive_anchor"}:
            # Contact-repair experiment: keep start-frame sampling while
            # focusing the loss on the support foot during step transitions.
            ankle_reward.weight = 8.0
            ankle_reward.params["std"] = 0.05
            root_reward.weight = 14.0
            root_reward.params["std"] = 0.10
        if profile == "precision_adaptive_anchor":
            # Final strict-gate repair after the feet have converged: retain
            # adaptive full-motion sampling and sharpen world-root tracking.
            root_reward.weight = 25.0
            root_reward.params["std"] = 0.07
            body_reward.weight = 8.0
            body_reward.params["std"] = 0.10
        cfg.rewards["motion_global_root_ori"].weight = 4.0 if precision else 1.0
        if precision:
            cfg.rewards["motion_global_root_ori"].params["std"] = 0.20
            cfg.rewards["motion_body_ori"].weight = 4.0
            cfg.rewards["motion_body_ori"].params["std"] = 0.25
        # Learn nominal execution before adding disturbance robustness.
        cfg.events.pop("push_robot", None)
        anchor_term = cfg.terminations["anchor_pos"]
        anchor_term.func = mdp.bad_anchor_pos
        anchor_term.params["threshold"] = 0.30 if profile in {"root_tight", "precision_start", "precision_start_limbs", "precision_start_feet", "precision_adaptive_feet", "precision_adaptive_anchor", "precision_start_legs"} else 0.45
        if profile in {"root_start", "root_start_limbs", "precision_start", "precision_start_limbs", "precision_start_feet", "takeoff_xy", "precision_start_legs"}:
            command.sampling_mode = "start"
        if profile == "getup_teacher_long":
            # Match the GetUp experiment: full-state uniform keyframe resets,
            # fixed horizon, and no tracking-error early termination. Tracking
            # errors remain rewards/metrics and are restored as strict gates
            # by play/evaluation configurations.
            command.sampling_mode = "uniform"
            command.perturb_ratio = 0.20
            command.hold_at_end = True
            cfg.episode_length_s = 8.0
            cfg.rewards["motion_global_root_pos"].weight = 12.0
            cfg.rewards["motion_global_root_pos"].params["std"] = 0.15
            cfg.rewards["motion_body_pos"].weight = 4.0
            cfg.rewards["motion_body_pos"].params["std"] = 0.15
            cfg.rewards["motion_ankle_pos"].weight = 6.0
            cfg.rewards["motion_ankle_pos"].params["std"] = 0.08
            cfg.rewards["motion_global_root_ori"].weight = 2.0
            if not play:
                for name in tuple(cfg.terminations):
                    if name != "time_out":
                        cfg.terminations.pop(name)
                cfg.terminations["numerical_failure"] = TerminationTermCfg(
                    func=numerical_failure,
                    params={"asset_name": "robot"},
                )
    if play:
        command.pose_range = {}
        command.velocity_range = {}
        command.joint_position_range = (0.0, 0.0)
        command.sampling_mode = "start"
        if isinstance(command, GetupResetMotionCommandCfg):
            command.perturb_ratio = 0.0
        cfg.events.pop("push_robot", None)
        cfg.events.pop("encoder_bias", None)
        cfg.episode_length_s = int(1e9)
        if profile != "baseline":
            # Success means following the world trajectory, not merely
            # holding a locally similar pose at the wrong location.
            if "anchor_pos" in cfg.terminations:
                cfg.terminations["anchor_pos"].func = mdp.bad_anchor_pos
                cfg.terminations["anchor_pos"].params["threshold"] = 0.25
    # Body names and geom names differ from the stock G1 task. Do not apply
    # G1-specific foot/self-collision/event selectors here.
    cfg.events.pop("foot_friction", None)
    cfg.events.pop("base_com", None)
    cfg.rewards.pop("self_collisions", None)
    if "ee_body_pos" in cfg.terminations:
        if profile in {"root_start_limbs", "root_adaptive_limbs", "precision_start_limbs",
                       "precision_start_feet", "precision_adaptive_feet",
                       "precision_adaptive_anchor", "precision_start_legs"} and not play:
            # Stage 1: do not let an early wrist miss prevent the agent from ever
            # seeing the later walking and sitting phases.  Wrist errors remain
            # explicitly rewarded and are restored as hard constraints in play.
            cfg.terminations["ee_body_pos"].params["body_names"] = (
                "left_ankle_roll_link", "right_ankle_roll_link",
            )
        else:
            cfg.terminations["ee_body_pos"].params["body_names"] = (
                "left_ankle_roll_link", "right_ankle_roll_link",
                "left_wrist_pitch_link", "right_wrist_pitch_link",
            )
    return cfg
