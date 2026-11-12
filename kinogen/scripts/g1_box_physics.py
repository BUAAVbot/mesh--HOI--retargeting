"""Floating G1 + bilateral ROHand contact task, run only after Isaac AppLauncher.

The 12->50 joint target coupling is a declared servo approximation. No box/root
poses, joints or external wrenches are written during active control.
"""
from __future__ import annotations

import inspect
import json
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
import time

import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.sim.converters import UrdfConverterCfg
from isaaclab.utils.math import quat_apply, quat_rotate_inverse

ROOT = Path(__file__).resolve().parents[1]


class G1BoxPhysics:
    dt = 0.02
    physics_dt = 0.005
    decimation = 4
    action_dim = 41

    def __init__(self, num_envs=256, episode_seconds=6.0, seed=1, mass=1.0, noise=0.01, physical_gpu=7,
                 asset_config=Path('data/robots/g1_box_training/training_asset.json'),table_x_size=.36,
                 stage_box_before_reset=False):
        self.device = 'cuda:0'
        self.physical_gpu = physical_gpu
        self.n = num_envs
        self.max_steps = round(episode_seconds / self.dt)
        self.seed = seed
        self.mass = mass
        self.noise = noise
        self.table_x_size=table_x_size
        self.stage_box_before_reset=stage_box_before_reset
        torch.manual_seed(seed)
        self.asset_path = (ROOT / asset_config).resolve()
        if not self.asset_path.is_relative_to(ROOT):raise ValueError('Asset config outside project')
        asset_dir = self.asset_path.parent
        self.asset = json.loads(self.asset_path.read_text())
        if 'source_training_asset_path' in self.asset:
            patch=self.asset
            base_path=(ROOT/patch['source_training_asset_path']).resolve()
            if not base_path.is_relative_to(ROOT)or hashlib.sha256(base_path.read_bytes()).hexdigest()!=patch['source_training_asset_sha256']:
                raise RuntimeError('Frozen base training asset changed')
            self.asset={**json.loads(base_path.read_text()),**patch}
        a = self.asset
        self.names = list(a['body_joint_names']) + list(a['hand_joint_names'])
        self.body_names = list(a['body_joint_names'])
        self.hand_names = list(a['hand_joint_names'])
        lut = np.load(ROOT / a['coupling_lut'], allow_pickle=False)
        self.lut = torch.as_tensor(lut['joint_values'], dtype=torch.float32, device=self.device)
        self.joint_controls = torch.as_tensor(lut['joint_control_indices'], dtype=torch.long, device=self.device)
        self.lower_act = self.tensor(a['actuator_lower'])
        self.upper_act = self.tensor(a['actuator_upper'])
        self.open_act = self.tensor(a['open_actuators'])
        self.closed_act = self.tensor(a['closed_actuators'])
        self.q_initial = self.tensor(a['reset_body_q'])
        self.q_reset = self.tensor(a.get('pregrasp_body_q',a['reset_body_q']))
        self.q_observation_anchor = self.tensor(a.get('observation_body_anchor_q',a['reset_body_q']))
        self.q_lift = self.tensor(a['lift_body_q'])
        self.base_reset = self.tensor(a['reset_base_xyz'])
        self.box_reset = self.tensor(a['box_center_xyz_m'])
        self.box_size = self.tensor(a['box_size_xyz_m'])
        self.table_z = float(a['table_top_z_m'])
        self.palm_offsets = self.tensor(a['palm_local_offsets'])
        self.steps = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.actions = torch.zeros((self.n, self.action_dim), device=self.device)
        self.previous_actions = self.actions.clone()
        self.hold_steps = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.max_hold_steps = self.hold_steps.clone()
        self.reset_context = True
        self.guard_counts = {}
        self.episode_initial_box_z = torch.zeros(self.n, device=self.device)
        self.torque_limits = a['joint_effort_limits']
        self.velocity_limits = a['joint_velocity_limits']
        stiffness, damping, armature = {}, {}, {}
        for name in self.names:
            hand = '_hand_ro_' in name
            slider = hand and 'slider' in name and 'abpart' not in name
            if hand:
                stiffness[name] = 5000.0 if slider else 3.0
                damping[name] = 30.0 if slider else 0.12
                armature[name] = 0.001 if slider else 0.00002
            else:
                stiffness[name] = 60.0 if any(x in name for x in ('shoulder', 'elbow')) else 25.0 if 'wrist' in name else 70.0 if 'ankle' in name else 180.0
                damping[name] = 3.0 if 'wrist' not in name else 1.0
                armature[name] = 0.01
        self.pd_stiffness,self.pd_damping=stiffness,damping
        init_fingers = self.expand(self.open_act[None])[0].cpu().tolist()
        init_q = dict(zip(self.names, a['reset_body_q'] + init_fingers))
        robot_cfg = ArticulationCfg(
            prim_path='{ENV_REGEX_NS}/Robot',
            spawn=sim_utils.UrdfFileCfg(
                asset_path=str(ROOT / a['urdf']),
                usd_dir=str(asset_dir / 'usd'), usd_file_name='g1_box.usd',
                fix_base=False, root_link_name='pelvis', merge_fixed_joints=False,
                self_collision=False, collision_from_visuals=False,
                collider_type='convex_hull',
                joint_drive=UrdfConverterCfg.JointDriveCfg(
                    gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)),
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False, linear_damping=0.0, angular_damping=0.0,
                    max_depenetration_velocity=1.0, max_linear_velocity=20.0, max_angular_velocity=40.0),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=2),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            ),
            init_state=ArticulationCfg.InitialStateCfg(pos=tuple(a['reset_base_xyz']), joint_pos=init_q, joint_vel={'.*': 0.0}),
            soft_joint_pos_limit_factor=1.0,
            actuators={'servos': ImplicitActuatorCfg(
                joint_names_expr=self.names, stiffness=stiffness, damping=damping,
                effort_limit_sim=self.torque_limits, velocity_limit_sim=self.velocity_limits, armature=armature)},
        )
        material = sim_utils.RigidBodyMaterialCfg(static_friction=1.2, dynamic_friction=1.0, restitution=0.0)
        cfg = InteractiveSceneCfg(num_envs=self.n, env_spacing=3.0, replicate_physics=True)
        cfg.robot = robot_cfg
        cfg.floor = AssetBaseCfg(prim_path='/World/Ground',
            spawn=sim_utils.CuboidCfg(size=(200., 200., .10), collision_props=sim_utils.CollisionPropertiesCfg(), physics_material=material),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0., 0., -.05)))
        cfg.table = RigidObjectCfg(prim_path='{ENV_REGEX_NS}/Table',
            spawn=sim_utils.CuboidCfg(size=(self.table_x_size, .80, .04),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                collision_props=sim_utils.CollisionPropertiesCfg(), mass_props=sim_utils.MassPropertiesCfg(mass=1000.),
                physics_material=material),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(float(self.box_reset[0])+.03, 0., self.table_z-.02)))
        cfg.box = RigidObjectCfg(prim_path='{ENV_REGEX_NS}/Box',
            spawn=sim_utils.CuboidCfg(size=tuple(a['box_size_xyz_m']),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, kinematic_enabled=False,
                    max_depenetration_velocity=1.0, solver_position_iteration_count=8, solver_velocity_iteration_count=2),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.002, rest_offset=0.),
                mass_props=sim_utils.MassPropertiesCfg(mass=mass), physics_material=material,
                activate_contact_sensors=True,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.52, .27, .08))),
            # Keep the box away from the imported URDF bind pose during PhysX
            # startup. The guarded episode reset below supplies the real pose.
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(
                float(v)+(2.0 if stage_box_before_reset and i==0 else 0.0)
                for i,v in enumerate(a['box_center_xyz_m']))))
        for side in ('left', 'right'):
            setattr(cfg, side+'_box_contact', ContactSensorCfg(
                prim_path='{ENV_REGEX_NS}/Box', update_period=0.0, history_length=1,
                filter_prim_paths_expr=['{ENV_REGEX_NS}/Robot/'+name for name in
                    [side+'_hand_ro_base_link']+[n for n in self.hand_names if n.startswith(side+'_')]+[side+'_wrist_yaw_link']]))
        cfg.table_box_contact = ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Box', update_period=0.0,
            filter_prim_paths_expr=['{ENV_REGEX_NS}/Table'])
        self.contact_body_names=[x.attrib['name']for x in ET.parse(ROOT/a['urdf']).getroot().findall('link')]
        cfg.all_robot_box_contact=ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Box',update_period=0.,
            filter_prim_paths_expr=['{ENV_REGEX_NS}/Robot/'+name for name in self.contact_body_names])
        cfg.feet_contact = ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/.*_ankle_roll_link', update_period=0.0)
        self.sim = sim_utils.SimulationContext(SimulationCfg(dt=self.physics_dt, device=self.device,
            render_interval=4, gravity=(0., 0., -9.81),
            physx=PhysxCfg(gpu_max_rigid_contact_count=2**22, gpu_max_rigid_patch_count=2**19)))
        self.scene = InteractiveScene(cfg)
        self.robot, self.box = self.scene['robot'], self.scene['box']
        self.sim.reset()
        self.scene.update(self.physics_dt)
        self.body_ids = self.robot.find_joints(self.body_names, preserve_order=True)[0]
        self.hand_ids = self.robot.find_joints(self.hand_names, preserve_order=True)[0]
        self.ids = self.body_ids + self.hand_ids
        if self.robot.num_joints != 79 or len(self.ids) != 79:
            raise RuntimeError(f'Unexpected physical joints: {self.robot.num_joints}')
        self.palm_ids = self.robot.find_bodies(a['palm_link_names'], preserve_order=True)[0]
        self.foot_ids = self.robot.find_bodies(['left_ankle_roll_link','right_ankle_roll_link'], preserve_order=True)[0]
        self.body_limits = self.robot.data.soft_joint_pos_limits[0, self.body_ids]
        self.body_action_scale = self.tensor([.30 if any(x in n for x in ('shoulder','elbow','wrist')) else .08 for n in self.body_names])
        self.up = torch.zeros((self.n, 3), device=self.device); self.up[:, 2] = 1.
        for label, obj in [('robot', self.robot), ('box', self.box)]:
            self._guard(obj, label)
        self.reset(torch.arange(self.n, device=self.device))

    def tensor(self, x):
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    def expand(self, actuators):
        u = ((actuators-self.lower_act)/(self.upper_act-self.lower_act)).clamp(0., 1.)
        at = u[:, self.joint_controls] * (self.lut.shape[1]-1)
        low = at.long().clamp(max=self.lut.shape[1]-2)
        frac = at-low
        rows = torch.arange(50, device=self.device)[None]
        return self.lut[rows, low]*(1.-frac) + self.lut[rows, low+1]*frac

    def _guard(self, asset, label):
        for name in ('write_root_state_to_sim','write_root_pose_to_sim','write_root_velocity_to_sim',
            'write_root_link_state_to_sim','write_root_link_pose_to_sim','write_root_link_velocity_to_sim',
            'write_root_com_state_to_sim','write_root_com_pose_to_sim','write_root_com_velocity_to_sim',
            'write_joint_state_to_sim','set_external_force_and_torque','write_external_wrench_to_sim'):
            if not hasattr(asset, name):
                continue
            original = getattr(asset, name)
            key = label+'.'+name
            self.guard_counts[key] = {'reset_calls': 0, 'active_calls': 0}
            def wrapped(*args, _original=original, _key=key, **kwargs):
                active = not self.reset_context
                self.guard_counts[_key]['active_calls' if active else 'reset_calls'] += 1
                if active:
                    raise RuntimeError('Forbidden state/wrench write during active control: '+_key)
                return _original(*args, **kwargs)
            setattr(asset, name, wrapped)

    def reset(self, env_ids, seeds=None):
        if not len(env_ids):
            return
        self.reset_context = True
        count = len(env_ids)
        q = self.robot.data.default_joint_pos[env_ids].clone()
        q[:, self.body_ids] = self.q_initial
        q[:, self.hand_ids] = self.expand(self.open_act[None].expand(count,-1))
        root = self.robot.data.default_root_state[env_ids].clone()
        root[:, :3] = self.base_reset + self.scene.env_origins[env_ids]
        root[:, 3:7] = self.tensor([1.,0.,0.,0.]); root[:,7:] = 0.
        box = self.box.data.default_root_state[env_ids].clone()
        box[:, :3] = self.box_reset + self.scene.env_origins[env_ids]
        if seeds is None:
            random_xy = torch.rand((count,2),device=self.device)
        else:
            if len(seeds) != count:raise ValueError('One actual RNG seed per evaluation episode required')
            random_xy = torch.stack([torch.rand(2, generator=torch.Generator(device=self.device).manual_seed(int(seed)), device=self.device)for seed in seeds])
        box[:, :2] += (random_xy-.5)*2*self.noise
        box[:, 3:7] = self.tensor([1.,0.,0.,0.]); box[:,7:] = 0.
        self.robot.write_root_state_to_sim(root, env_ids)
        self.robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)
        self.box.write_root_state_to_sim(box, env_ids)
        self.scene.reset(env_ids)
        self.robot.set_joint_position_target(q, env_ids=env_ids)
        self.scene.write_data_to_sim()
        self.sim.forward()  # Official DirectRLEnv reset semantics: update FK, no integration.
        self.scene.update(0.0)
        self.steps[env_ids] = 0
        self.actions[env_ids] = self.previous_actions[env_ids] = 0.
        self.hold_steps[env_ids] = self.max_hold_steps[env_ids] = 0
        self.episode_initial_box_z[env_ids] = self.box_reset[2]
        self.reset_context = False

    def contact(self, key):
        value = self.scene[key].data.force_matrix_w
        if value is None:
            raise RuntimeError('Missing real filtered contact matrix: '+key)
        return torch.linalg.vector_norm(value,dim=-1).flatten(1).sum(1)

    def state(self):
        root = self.robot.data.root_pos_w-self.scene.env_origins
        bpos = self.box.data.root_pos_w-self.scene.env_origins
        palms = self.robot.data.body_pos_w[:,self.palm_ids] - self.scene.env_origins[:,None]
        palms = palms + quat_apply(self.robot.data.body_quat_w[:,self.palm_ids].reshape(-1,4),
            self.palm_offsets[None].expand(self.n,-1,-1).reshape(-1,3)).reshape(self.n,2,3)
        axes = torch.eye(3,device=self.device)[None].expand(self.n,-1,-1)
        world_axes = quat_apply(self.box.data.root_quat_w[:,None].expand(-1,3,-1).reshape(-1,4),axes.reshape(-1,3)).reshape(self.n,3,3)
        box_bottom = bpos[:,2]-(world_axes[:,:,2].abs()*self.box_size/2).sum(1)
        return {'root_pos':root, 'root_quat':self.robot.data.root_quat_w,
            'box_pos':bpos, 'box_quat':self.box.data.root_quat_w,
            'box_vel':self.box.data.root_lin_vel_w, 'box_bottom_z':box_bottom,
            'body_pos':self.robot.data.body_pos_w-self.scene.env_origins[:,None],
            'body_quat':self.robot.data.body_quat_w,
            'palm_pos':palms, 'upright':quat_apply(self.robot.data.root_quat_w,self.up)[:,2],
            'hand_contact_force_left_N':self.contact('left_box_contact'),
            'hand_contact_force_right_N':self.contact('right_box_contact'),
            'table_contact_force_N':self.contact('table_box_contact'),
            'feet_force':self.scene['feet_contact'].data.net_forces_w,
            'joint_pos':self.robot.data.joint_pos[:,self.ids],
            'joint_vel':self.robot.data.joint_vel[:,self.ids],
            'joint_targets':self.robot._joint_pos_target_sim[:,self.ids]}

    def observations(self):
        s = self.state()
        relative = s['box_pos']-s['root_pos']
        gravity = quat_rotate_inverse(self.robot.data.root_quat_w, -self.up)
        phase = (self.steps.float()*self.dt/6.).clamp(0.,1.)[:,None]
        obs = torch.cat((self.robot.data.joint_pos[:,self.body_ids]-self.q_observation_anchor,
            self.robot.data.joint_pos[:,self.hand_ids], self.robot.data.joint_vel[:,self.ids]*.05,
            self.robot.data.root_lin_vel_b, self.robot.data.root_ang_vel_b*.25, gravity,
            s['root_pos'], relative, self.box.data.root_lin_vel_w*.25,
            (s['palm_pos']-s['box_pos'][:,None]).flatten(1),
            torch.stack((s['hand_contact_force_left_N'],s['hand_contact_force_right_N'],s['table_contact_force_N']),1).clamp(max=100.)*.01,
            self.actions, phase),1).clamp(-10.,10.)
        return obs

    def step(self, actions):
        self.previous_actions = self.actions.clone()
        self.actions = actions.clamp(-2.,2.)
        seconds = self.steps.float()*self.dt
        # Reset is open and separated. A joint-space reference closes then raises;
        # PPO learns bounded feedback residuals; only motor position targets act.
        lift = ((seconds-1.0)/2.).clamp(0.,1.)[:,None]
        close = ((seconds-.25)/.70).clamp(0.,1.)[:,None]
        approach = (seconds/.50).clamp(0.,1.)[:,None]
        body_target = self.q_initial+(self.q_reset-self.q_initial)*approach+(self.q_lift-self.q_reset)*lift+self.body_action_scale*self.actions[:,:29]
        body_target = body_target.maximum(self.body_limits[:,0]).minimum(self.body_limits[:,1])
        ref_act = self.open_act+(self.closed_act-self.open_act)*close
        act = (ref_act + .20*(self.upper_act-self.lower_act)*self.actions[:,29:]).maximum(self.lower_act).minimum(self.upper_act)
        target = self.robot.data.default_joint_pos.clone()
        target[:,self.body_ids] = body_target
        target[:,self.hand_ids] = self.expand(act)
        self.robot.set_joint_position_target(target)
        for _ in range(self.decimation):
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(self.physics_dt)
        self.steps += 1
        s = self.state()
        finite = torch.isfinite(s['joint_pos']).all(1) & torch.isfinite(s['box_pos']).all(1) & torch.isfinite(s['root_pos']).all(1)
        lift_height = s['box_pos'][:,2]-self.episode_initial_box_z
        double_contact = (s['hand_contact_force_left_N']>.5)&(s['hand_contact_force_right_N']>.5)
        upright = (s['upright']>.9)&(s['root_pos'][:,2]>.55)
        held = (s['box_bottom_z']>=self.table_z+.10)&double_contact&(s['table_contact_force_N']<.2)&upright
        self.hold_steps = torch.where(held,self.hold_steps+1,0)
        self.max_hold_steps = torch.maximum(self.max_hold_steps,self.hold_steps)
        failed = ~finite|(s['upright']<.65)|(s['root_pos'][:,2]<.48)|(s['box_pos'][:,2]<self.table_z-.2)
        timeout = self.steps>=self.max_steps
        # Palm targets are box side centers; an ordinary feedback reward, not a
        # hidden pose write or hand-box weld. Bilateral contact gates lift reward.
        desired = s['box_pos'][:,None].expand(-1,2,-1).clone()
        desired[:,0,1] += self.box_size[1]/2+.015
        desired[:,1,1] -= self.box_size[1]/2+.015
        distance = torch.linalg.vector_norm(s['palm_pos']-desired,dim=-1)
        reach = torch.exp(-distance.square()/.012).mean(1)
        lift_rew = (lift_height/.12).clamp(0.,1.5)*double_contact.float()
        reward = 2.*reach + double_contact.float()*1.5 + lift_rew*6. + held.float()*4. + (s['upright'].clamp(0.,1.)**4)*.5
        reward -= .10*(self.actions-self.previous_actions).square().mean(1)
        reward -= .03*self.actions.square().mean(1)
        reward -= .001*self.robot.data.joint_vel[:,self.body_ids].square().mean(1)
        reward -= 2.*s['root_pos'][:,:2].square().sum(1)/.01
        reward -= 5.*failed.float()
        info = {'lift_m':lift_height,'held':held,'double_contact':double_contact,'upright':s['upright'],
            'timeout':timeout,'failed':failed,'success':self.hold_steps>=round(2./self.dt)}
        return self.observations(), torch.nan_to_num(reward,nan=-5.)*self.dt, failed|timeout, info

    def live_audit(self):
        from pxr import UsdPhysics
        object_joints, boxes = [], []
        for prim in self.sim.stage.Traverse():
            path = str(prim.GetPath())
            if '/env_0/' not in path:
                continue
            if prim.IsA(UsdPhysics.Joint):
                joint = UsdPhysics.Joint(prim)
                ends = list(joint.GetBody0Rel().GetTargets())+list(joint.GetBody1Rel().GetTargets())
                if any('/Box' in str(x) for x in ends):object_joints.append(path)
            if '/Box' in path and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rigid = UsdPhysics.RigidBodyAPI(prim)
                boxes.append({'path':path,'enabled':rigid.GetRigidBodyEnabledAttr().Get(),
                    'kinematic':rigid.GetKinematicEnabledAttr().Get(),
                    'disable_gravity':prim.GetAttribute('physxRigidBody:disableGravity').Get()})
        if self.robot.is_fixed_base or object_joints or not boxes or any(x['kinematic']or x['disable_gravity']for x in boxes):
            raise RuntimeError('Physics audit rejected fixed/welded/kinematic setup')
        return {'robot_fixed_base':bool(self.robot.is_fixed_base),'robot_joint_count':self.robot.num_joints,
            'robot_body_count':self.robot.num_bodies,'joint_names':self.names,'simulator_joint_names':self.robot.joint_names,
            'joint_units':['m'if '_hand_ro_'in n and 'slider'in n and 'abpart'not in n else'rad'for n in self.names],
            'gravity':[0.,0.,-9.81], 'box_rigid_bodies':boxes,'box_connected_joints':object_joints,
            'box_mass_kg':self.box.root_physx_view.get_masses().cpu().tolist(),
            'box_size_xyz_m':self.box_size.cpu().tolist(),'table_top_z_m':self.table_z,
            'control_dt':self.dt,'physics_dt':self.physics_dt,'state_write_guard':self.guard_counts,
            'coupling_model':'12 manufacturer controls -> 50 servo position targets; not closed-loop linkage dynamics',
            'self_collision_enabled':False,'physical_gpu':self.physical_gpu,'cameras_enabled':False,
            'reset_only_pose_writes':True,'training_asset_assumptions':self.asset['simulation_assumptions'],
            'actual_command_stiffness':self.pd_stiffness,'actual_command_damping':self.pd_damping,
            'actual_sim_effort_limits':self.torque_limits,'actual_sim_velocity_limits':self.velocity_limits,
            'table_size_xyz_m':[self.table_x_size,.80,.04],'table_center_xyz_m':[float(self.box_reset[0])+.03,0.,self.table_z-.02]}
