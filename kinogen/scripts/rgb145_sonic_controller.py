"""Metadata-driven Vita SONIC controller for the full six-object scene.

This transfers an existing 28-body-joint policy to a 52-joint articulation;
22 fingers and 2 neck joints retain explicit reference PD control. Physical
validation is required: matching an ONNX ABI does not establish tracking.
"""
from __future__ import annotations

from collections import deque
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT/'dependencies/fourth_sonic'


def freeze_effective_model(model, output_prefix):
    """Save authoritative binary plus review XML after constructor mutation."""
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    binary, xml = prefix.with_suffix('.mjb'), prefix.with_suffix('.xml')
    mujoco.mj_saveLastXML(str(xml), model)
    mujoco.mj_saveModel(model, str(binary), None)
    reloaded = mujoco.MjModel.from_binary_path(str(binary))
    for key in ('actuator_gainprm','actuator_biasprm','actuator_gaintype','actuator_biastype',
                'actuator_forcerange','actuator_forcelimited','actuator_gear','dof_armature','dof_frictionloss','dof_damping','jnt_range','jnt_limited','eq_data','eq_active0','eq_obj1id','eq_obj2id'):
        if not np.array_equal(getattr(model,key),getattr(reloaded,key)):
            raise ValueError('Binary plant freeze mismatch: '+key)
    return {'binary':str(binary),'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),
            'review_xml':str(xml),'review_xml_sha256':hashlib.sha256(xml.read_bytes()).hexdigest(),
            'authoritative':'binary; XML may round numeric parameters'}


def _ort():
    try:
        import onnxruntime
    except ImportError:
        # Reuse the installed server dependency; never modify that environment.
        sys.path.append('/root/vita02a-mjlab/.venv310/lib/python3.10/site-packages')
        import onnxruntime
    return onnxruntime


def quaternion_matrix(q):
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    out = np.empty(9)
    mujoco.mju_quat2Mat(out, q)
    return out.reshape(3, 3)


def sampled_heading_matrix(rotation,mode='training_z_twist'):
    """Match training get_heading_q at the supplied sensor sampling instant."""
    r=np.asarray(rotation,dtype=float).reshape(3,3)
    if mode=='training_z_twist':
        q=np.empty(4);mujoco.mju_mat2Quat(q,np.ascontiguousarray(r).reshape(9))
        length=max(float(np.hypot(q[0],q[3])),1e-12)
        # The training quaternion is normalize((w,0,0,z)); atan2 is invariant
        # to its scale. At the undefined upside-down singularity use zero yaw.
        yaw=2*np.arctan2(q[3]/length,q[0]/length)
    elif mode=='euler_yaw':yaw=np.arctan2(r[1,0],r[0,0])
    else:raise ValueError('Unknown heading_mode '+str(mode))
    cy,sy=np.cos(yaw),np.sin(yaw)
    return np.array([[cy,-sy,0],[sy,cy,0],[0,0,1.]])


def reference_joint_velocity(times,joint_pos,query_times,mode='training_forward_50hz',legacy_velocity=None):
    """Query velocity without changing reference-array length or pose samples.

Training uses a50Hz forward difference. At the clipped final timestamp this
adapter explicitly uses the preceding20ms backward difference. The source
motion loader instead pads its last entry from dof_vel[:, -2:-1].
    """
    times=np.asarray(times,float);joint_pos=np.asarray(joint_pos,float)
    query=np.clip(np.asarray(query_times,float),times[0],times[-1])
    def sample(values,t):
        return np.stack([np.interp(np.asarray(t).ravel(),times,values[:,j])for j in range(values.shape[1])],axis=-1).reshape(np.shape(t)+(values.shape[1],))
    if mode=='legacy_centered':
        values=np.gradient(joint_pos,times,axis=0)if legacy_velocity is None else legacy_velocity
        return sample(values,query)
    if mode!='training_forward_50hz':raise ValueError('Unknown reference_velocity_mode '+str(mode))
    dt=.02;now=sample(joint_pos,query)
    velocity=(sample(joint_pos,np.minimum(query+dt,times[-1]))-now)/dt
    backward=(now-sample(joint_pos,np.maximum(query-dt,times[0])))/dt
    return np.where((query>=times[-1])[...,None],backward,velocity)


class Controller:
    """Call reset(data), then action(data, reference_elapsed_s) every physics step.

    Returns a complete model.nu control packet. Constructor configures ONLY
    named body actuators/armatures; save the resulting model before rollout.
    Config: mode=tracking|velocity, actuation=position|torque,
    arm_reference_override=False, source_root/path optional.
    """
    def __new__(cls, model, reference, config=None):
        if (config or {}).get('mode') in ('newft15225','token15225'):
            return Token15225Controller(model, reference, config)
        return super().__new__(cls)

    def __init__(self, model, reference, config=None):
        self.model = model
        self.config = dict(config or {})
        self.mode = self.config.get('mode', 'tracking')
        self.actuation = self.config.get('actuation', 'position')
        if self.mode not in ('tracking', 'velocity') or self.actuation not in ('position', 'torque'):
            raise ValueError('Unsupported SONIC mode or actuation')
        ort = _ort()
        default_policy = FROZEN/('tracking.onnx' if self.mode == 'tracking' else 'standing.onnx')
        self.policy_path = Path(self.config.get('policy', default_policy))
        opt = ort.SessionOptions()
        opt.intra_op_num_threads = 1
        opt.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(self.policy_path), sess_options=opt, providers=['CPUExecutionProvider'])
        self.metadata = dict(self.session.get_modelmeta().custom_metadata_map)
        md = self.metadata
        self.names = md['joint_names'].split(',')
        self.vector = lambda k: np.asarray([float(v) for v in md[k].split(',')])
        self.default = self.vector('default_joint_pos')
        self.scale = self.vector('action_scale')
        self.lower = self.vector('joint_pos_limit_lower')
        self.upper = self.vector('joint_pos_limit_upper')
        self.kp = self.vector('joint_stiffness')
        self.kd = self.vector('joint_damping')
        self.effort = self.vector('joint_effort_limit')
        self.velocity = self.vector('joint_velocity_limit')
        self.physics_dt = float(md['physics_dt'])
        self.policy_dt = float(md['policy_dt'])
        if abs(model.opt.timestep-self.physics_dt) > 1e-9:
            raise ValueError('SONIC requires physics_dt=' + str(self.physics_dt))
        self.decimation = int(md['decimation'])
        if self.mode == 'tracking':
            if md.get('action_contract') != 'reference_residual_policy_v1' or int(md['obs_dim']) != 149:
                raise ValueError('Expected the exact 149D SONIC reference residual contract')
            self.delay_steps = int(md['shared_actuator_delay_nominal_physics_steps'])
        else:
            self.delay_steps = int(self.config.get('delay_steps', 3))
        self.reference = dict(np.load(reference, allow_pickle=False)) if isinstance(reference, (str, Path)) else dict(reference)
        self.reference_names = list(map(str, self.reference.get('dof_names', self.reference.get('joint_names'))))
        self.times = np.asarray(self.reference['time_s'], dtype=float)
        self.qref = np.asarray(self.reference['qpos'], dtype=float)
        if self.qref.shape != (len(self.times), 7+len(self.reference_names)) or np.any(np.diff(self.times) <= 0):
            raise ValueError('Expected full-root qpos, named joints and increasing time_s')
        self.jref = np.asarray([self.reference_names.index(n) for n in self.names])
        self.body_q = self.qref[:, 7+self.jref]
        self.body_dq = np.gradient(self.body_q, self.times, axis=0)
        self.root_dv = np.gradient(self.qref[:, :3], self.times, axis=0)
        self.jids = np.asarray([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.names])
        self.aids = np.asarray([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'motor_'+n) for n in self.names])
        if min(self.jids) < 0 or min(self.aids) < 0 or len(self.names) != 28:
            raise ValueError('Model must have the 28 canonical named SONIC joints and motor_ actuators')
        self.qa = model.jnt_qposadr[self.jids]
        self.va = model.jnt_dofadr[self.jids]
        self.base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'base_link')
        self.ref_aids = np.asarray([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'motor_'+n)
                                   for n in self.reference_names])
        if min(self.ref_aids) < 0:
            raise ValueError('Every reference joint requires a named actuator')
        self.arm = np.asarray([n.startswith(('left_shoulder', 'right_shoulder', 'left_elbow',
                                           'right_elbow', 'left_wrist', 'right_wrist')) for n in self.names])
        model.dof_armature[self.va] = self.vector('joint_armature')
        friction = self.vector('joint_dynamic_friction') if 'joint_dynamic_friction' in md else np.zeros(28)
        viscous = self.vector('joint_viscous_friction') if 'joint_viscous_friction' in md else np.zeros(28)
        model.dof_frictionloss[self.va] = friction
        model.dof_damping[self.va] = viscous
        for i, aid in enumerate(self.aids):
            model.actuator_gainprm[aid] = 0
            model.actuator_biasprm[aid] = 0
            model.actuator_gaintype[aid] = mujoco.mjtGain.mjGAIN_FIXED
            model.actuator_dyntype[aid] = mujoco.mjtDyn.mjDYN_NONE
            model.actuator_forcelimited[aid] = np.isfinite(self.effort[i])
            if np.isfinite(self.effort[i]):
                model.actuator_forcerange[aid] = [-self.effort[i], self.effort[i]]
            model.actuator_ctrllimited[aid] = False
            if self.actuation == 'torque':
                model.actuator_gainprm[aid, 0] = 1
                model.actuator_biastype[aid] = mujoco.mjtBias.mjBIAS_NONE
            else:
                model.actuator_gainprm[aid, 0] = self.kp[i]
                model.actuator_biasprm[aid, 1:3] = [-self.kp[i], -self.kd[i]]
                model.actuator_biastype[aid] = mujoco.mjtBias.mjBIAS_AFFINE
        self.ankle = None
        if self.actuation == 'torque':
            source = Path(self.config.get('source_root', FROZEN/'source'))
            sys.path.insert(0, str(source))
            from legged_lab.scripts.sim2sim_vita02a import ParallelAnkle
            self.ankle = ParallelAnkle(lx=float(md['parallel_ankle_lx']), ly=float(md['parallel_ankle_ly']),
                z_top=float(md['parallel_ankle_z_top']), z_bot=float(md['parallel_ankle_z_bot']),
                tau_max=float(md['parallel_ankle_tau_motor_max']), qd_motor_max=float(md['parallel_ankle_qd_motor_max']))
            self.ankle_pairs = tuple((self.names.index(s+'_ankle_pitch_joint'), self.names.index(s+'_ankle_roll_joint'))
                                      for s in ('left', 'right'))
        self.history = deque(maxlen=int(md.get('history_length', 1)))
        self.target_history = deque(maxlen=self.delay_steps+1)
        self.previous_action = np.zeros(28)
        self.previous_imu = None
        self.next_inference_time = 0.
        self.inference_count = 0
        self.last_observation = None
        self.target = self.default.copy()

    def reset(self, data):
        self.history.clear()
        self.target_history.clear()
        self.previous_action[:] = 0
        self.previous_imu = None
        self.next_inference_time = 0.
        self.inference_count = 0

    def interpolate(self, array, t):
        return np.asarray([np.interp(t, self.times, array[:, j]) for j in range(array.shape[1])])

    def action(self, data, t):
        t = float(t)+float(self.times[0])
        all_q = self.interpolate(self.qref[:, 7:], t)
        ctrl = np.zeros(self.model.nu)
        ctrl[self.ref_aids] = all_q
        if t-float(self.times[0])+1e-8 >= self.next_inference_time:
            qdes = self.interpolate(self.body_q, t)
            dqdes = self.interpolate(self.body_dq, t)
            rotation = data.xmat[self.base].reshape(3, 3)
            velocity = np.empty(6)
            mujoco.mj_objectVelocity(self.model, data, mujoco.mjtObj.mjOBJ_BODY, self.base, velocity, 0)
            imu = rotation.T @ velocity[:3]
            if self.mode == 'tracking':
                fraction = float(self.metadata['imu_latency_s'])/self.policy_dt
                observed = imu if self.previous_imu is None else (1-fraction)*imu+fraction*self.previous_imu
                self.previous_imu = imu.copy()
                target_q = self.interpolate(self.qref[:, 3:7], t)
                target_q /= np.linalg.norm(target_q)
                relative = rotation.T @ quaternion_matrix(target_q)
                obs = np.r_[qdes, dqdes, relative[:, :2].reshape(-1), observed,
                            data.qpos[self.qa]-self.default, data.qvel[self.va], self.previous_action]
                residual = self.session.run(None, {self.session.get_inputs()[0].name:obs.astype(np.float32)[None]})[0][0]
                self.previous_action = (qdes-self.default)/self.scale+residual
                self.target = np.clip(self.default+self.scale*self.previous_action, self.lower, self.upper)
            else:
                xyz = self.interpolate(self.qref[:, :3], t)
                desired_velocity = self.interpolate(self.root_dv, t) + float(self.config.get('root_position_gain', .8))*(xyz-data.xpos[self.base])
                command = rotation.T @ desired_velocity
                yaw_ref = np.arctan2(quaternion_matrix(self.interpolate(self.qref[:, 3:7], t))[1, 0],
                                     quaternion_matrix(self.interpolate(self.qref[:, 3:7], t))[0, 0])
                yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
                yaw_err = np.arctan2(np.sin(yaw_ref-yaw), np.cos(yaw_ref-yaw))
                command = np.r_[command[:2], np.clip(yaw_err, -.5, .5)]
                # Respect the original backward-only training command ranges.
                for i, key in enumerate(['command_range_lin_vel_x', 'command_range_lin_vel_y', 'command_range_ang_vel_z']):
                    limits = self.vector(key)
                    command[i] = np.clip(command[i], limits[0], limits[1])
                frame = np.r_[imu, rotation.T@np.asarray([0.,0.,-1.]), command,
                              data.qpos[self.qa]-self.default, data.qvel[self.va], self.previous_action]
                if not self.history:
                    self.history.extend(frame.copy() for _ in range(self.history.maxlen))
                else:
                    self.history.append(frame.copy())
                obs = np.clip(np.concatenate(self.history), -float(self.metadata['clip_observations']), float(self.metadata['clip_observations']))
                self.previous_action = np.clip(self.session.run(None, {self.session.get_inputs()[0].name:obs.astype(np.float32)[None]})[0][0],
                                               -float(self.metadata['clip_actions']), float(self.metadata['clip_actions']))
                self.target = np.clip(self.default+self.scale*self.previous_action, self.lower, self.upper)
            if self.config.get('arm_reference_override', False):
                self.target[self.arm] = qdes[self.arm]
            if not np.isfinite(self.target).all():
                raise ValueError('Nonfinite SONIC target')
            self.last_observation = obs.copy()
            self.next_inference_time += self.policy_dt
            self.inference_count += 1
        self.target_history.append(self.target.copy())
        delayed = self.target_history[0]
        if self.actuation == 'position':
            ctrl[self.aids] = delayed
        else:
            q, dq = data.qpos[self.qa], data.qvel[self.va]
            tau = self.kp*(delayed-q)-self.kd*dq
            tau = np.clip(tau, -self.effort, self.effort)
            tau[(dq >= self.velocity)&(tau > 0)] = 0
            tau[(dq <= -self.velocity)&(tau < 0)] = 0
            self.ankle.project(tau, q, self.ankle_pairs, dq)
            ctrl[self.aids] = tau
        return ctrl

    def report(self):
        return {'controller':'SONIC '+self.mode+' 28 body joints + reference PD auxiliary joints',
                'policy_path':str(self.policy_path), 'policy_sha256':hashlib.sha256(self.policy_path.read_bytes()).hexdigest(),
                'policy_dt':self.policy_dt, 'physics_dt':self.physics_dt, 'delay_physics_steps':self.delay_steps,
                'actuation':self.actuation, 'arm_reference_override':bool(self.config.get('arm_reference_override',False)),
                'inference_count':self.inference_count, 'body_joint_names':self.names,
                'strict_original_articulation_reproduction':False,
                'limitation':'52DOF articulation transfer needs physical validation; position mode uses implicit native PD and omits parallel-ankle torque projection.'}


class Token15225Controller:
    """Supplied FSQ/tokenizer checkpoint with source-derived 900D actor ABI.

    The saved config's future spacing is 0.1 s. PolicyCfg's declared field
    order, rather than YAML display order, controls the actor history layout.
    Matching the original Isaac implicit drives with MuJoCo remains a transfer.
    """
    def __init__(self, model, reference, config=None):
        import ast
        self.model, self.config = model, dict(config or {})
        self.mode, self.actuation = 'newft15225', 'position'
        try:
            from tensordict import TensorDict
        except ImportError:
            sys.path.append('/data/grail-envs/tracking45/lib/python3.10/site-packages')
            from tensordict import TensorDict
        import torch
        from omegaconf import OmegaConf
        from rgb145_policy_tracking import load_checkpoint, make_policy
        self.torch, self.TensorDict = torch, TensorDict
        torch.set_num_threads(int(self.config.get('cpu_threads', 2)))
        self.device = torch.device(self.config.get('device','cuda:1'))
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('Requested server CUDA device unavailable')
        self.source = Path(self.config.get('sonic_root','/data/yunhao.xia/GR00T-WholeBodyControl'))
        self.policy_path = Path(self.config.get('checkpoint',ROOT/'data/vita_demo_20261026/checkpoints/new_ft15225/model_step_015225.pt'))
        self.model_config = Path(self.config.get('model_config',ROOT/'data/vita_demo_20261026/checkpoints/new_ft15225/model_config.yaml'))
        self.training_config = Path(self.config.get('training_config',ROOT/'data/vita_demo_20261026/checkpoints/new_ft15225/config.yaml'))
        self.policy, self.cfg = make_policy(load_checkpoint(self.policy_path),self.model_config,self.source)
        self.policy = self.policy.to(self.device).eval()
        training = OmegaConf.load(self.training_config)
        self.physics_dt=float(training.manager_env.config.sim_dt)
        self.policy_dt=self.physics_dt*int(training.manager_env.config.decimation)
        self.future_dt=float(training.manager_env.commands.motion.dt_future_ref_frames)
        self.nfuture=int(training.manager_env.commands.motion.num_future_frames)
        if abs(model.opt.timestep-self.physics_dt)>1e-9 or self.nfuture!=10:
            raise ValueError('New checkpoint expects physics_dt=.005 and 10 future frames')
        robot_source=self.source/'gear_sonic/envs/manager_env/robots/vita02a.py'
        body_links=None
        for node in ast.parse(robot_source.read_text()).body:
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='VITA02A_ISAACLAB_JOINTS' for t in node.targets):
                body_links=ast.literal_eval(node.value)
        if body_links is None:raise ValueError('Source canonical joint ordering missing')
        self.names=[n.removesuffix('_link')+'_joint' for n in body_links[1:]]
        self.reference=dict(np.load(reference,allow_pickle=False)) if isinstance(reference,(str,Path)) else dict(reference)
        self.reference_names=list(map(str,self.reference['dof_names']))
        self.times=np.asarray(self.reference['time_s'],dtype=float)
        self.qref=np.asarray(self.reference['qpos'],dtype=float).copy()
        for i in range(1,len(self.qref)):
            if np.dot(self.qref[i,3:7],self.qref[i-1,3:7])<0:self.qref[i,3:7]*=-1
        self.jref=np.array([self.reference_names.index(n)for n in self.names])
        self.body_q=self.qref[:,7+self.jref]
        self.body_dq=np.gradient(self.body_q,self.times,axis=0)
        self.jids=np.array([mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n)for n in self.names])
        self.aids=np.array([mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_ACTUATOR,'motor_'+n)for n in self.names])
        self.ref_aids=np.array([mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_ACTUATOR,'motor_'+n)for n in self.reference_names])
        if len(self.names)!=28 or min(self.jids)<0 or min(self.aids)<0 or min(self.ref_aids)<0:raise ValueError('Missing named 28 body or auxiliary joints')
        self.qa,self.va=model.jnt_qposadr[self.jids],model.jnt_dofadr[self.jids]
        ankle_limit_event=training.manager_env.events.ankle_pitch_limits.params
        self.ankle_pitch_hard_limits=[float(ankle_limit_event.lower),float(ankle_limit_event.upper)]
        for jid,n in zip(self.jids,self.names):
            if 'ankle_pitch_joint' in n:
                model.jnt_limited[jid]=True;model.jnt_range[jid]=self.ankle_pitch_hard_limits
        self.base=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'base_link')
        self.default=np.zeros(28);self.kp=np.zeros(28);self.kd=np.zeros(28);self.scale=np.zeros(28);self.armature=np.zeros(28);self.effort=np.zeros(28)
        dynamic=np.zeros(28);viscous=np.zeros(28)
        wn=20*np.pi
        for i,n in enumerate(self.names):
            if 'ankle' in n:
                foot=self.cfg.env_config.ankle_deploy_contract.joints[n]
                self.kp[i],self.kd[i],self.scale[i],self.armature[i]=map(float,[foot.kp,foot.kd,foot.action_scale,foot.armature]);self.effort[i]=141.
                dynamic[i]=2.56 if 'pitch' in n else 1.12;viscous[i]=.1 if 'pitch'in n else .025
            else:
                if 'hip' in n or 'knee' in n: inertia,eff=21.913**2*10.58e-5,120.;dynamic[i],viscous[i]=1.34,.1
                elif 'waist' in n:inertia,eff=35**2*1.23e-5,70.5;dynamic[i],viscous[i]=1.28,.075
                elif 'wrist' in n:inertia,eff=19.5278**2*1.85e-5,30.;dynamic[i],viscous[i]=.16,.025
                else:inertia,eff=20**2*4.38e-5,60.;dynamic[i],viscous[i]=.428,.04
                self.armature[i],self.effort[i]=inertia,eff;self.kp[i],self.kd[i]=inertia*wn**2,4*inertia*wn;self.scale[i]=.25*eff/self.kp[i]
            if 'hip_pitch' in n or 'ankle_pitch' in n:self.default[i]=-.2
            if 'knee' in n:self.default[i]=.4
            if 'hip_roll' in n:self.default[i]=.088 if n.startswith('left') else -.088
            if 'ankle_roll' in n:self.default[i]=-.088 if n.startswith('left') else .088
            if 'shoulder_roll' in n:self.default[i]=.4226 if n.startswith('left') else -.4226
            aid=self.aids[i]
            model.actuator_gaintype[aid]=mujoco.mjtGain.mjGAIN_FIXED;model.actuator_biastype[aid]=mujoco.mjtBias.mjBIAS_AFFINE;model.actuator_dyntype[aid]=mujoco.mjtDyn.mjDYN_NONE
            model.actuator_gainprm[aid]=0;model.actuator_gainprm[aid,0]=self.kp[i];model.actuator_biasprm[aid]=0;model.actuator_biasprm[aid,1:3]=[-self.kp[i],-self.kd[i]]
            model.actuator_ctrllimited[aid]=False;model.actuator_forcelimited[aid]=True;model.actuator_forcerange[aid]=[-self.effort[i],self.effort[i]]
        model.dof_armature[self.va]=self.armature
        model.dof_frictionloss[self.va]=dynamic if self.config.get('median_training_friction',True) else 0
        model.dof_damping[self.va]=viscous if self.config.get('median_training_friction',True) else 0
        self.delay_steps=int(self.config.get('delay_steps',3));self.imu_latency=float(self.config.get('imu_latency_s',.0025))
        self.heading_mode=str(self.config.get('heading_mode','training_z_twist'))
        if self.heading_mode not in ('training_z_twist','euler_yaw'):raise ValueError('Unknown heading_mode '+self.heading_mode)
        self.reference_velocity_mode=str(self.config.get('reference_velocity_mode','training_forward_50hz'))
        if self.reference_velocity_mode not in ('training_forward_50hz','legacy_centered'):raise ValueError('Unknown reference_velocity_mode '+self.reference_velocity_mode)
        self.terms=('base_ang_vel','joint_pos','joint_vel','actions','gravity_dir')
        self.history={k:deque(maxlen=10)for k in self.terms};self.target_history=deque(maxlen=self.delay_steps+1)
        self.last=np.zeros(28);self.previous_imu=None;self.previous_gravity=None;self.next_inference_time=0.;self.inference_count=0;self.target=self.default.copy();self.last_observation=None
        self.tokenizer={k:torch.zeros((1,1,*list(shape)),device=self.device)for k,shape in self.cfg.env_config.obs.group_obs_dims.tokenizer.items()}
        self.tokenizer['encoder_index'][...,0]=1
        self.source_files=[robot_source,self.source/'gear_sonic/envs/manager_env/mdp/observations.py',self.source/'gear_sonic/envs/manager_env/mdp/commands.py',self.source/'gear_sonic/utils/motion_lib/torch_humanoid_batch.py']
        self.passive_aids=np.flatnonzero(np.all(model.actuator_gear==0,axis=1))
        passive_contract={k:getattr(model,k)[self.passive_aids].copy()for k in ['actuator_gear','actuator_gainprm','actuator_biasprm']}
        if len(self.passive_aids)and any(np.any(v!=0)for v in passive_contract.values()):raise ValueError('Passive compatibility motors must have zero gear/gain/bias')
        self.arm_feedback=None;self.finger_feedback=None;self.hybrid_policy_records=[]
        if self.config.get('hand_object_feedback',False):
            from fourth_arm_object_feedback import ArmObjectFeedback
            from fourth_contact_feedback import ReferenceContactFeedback
            self.arm_feedback=ArmObjectFeedback(model,self.reference,self.config.get('arm_feedback',{}))
            finger_config=dict(self.config.get('finger_feedback',{}))
            if finger_config.get('geometry_backend')=='convex_sat_o6':
                from fourth_contact_feedback_o6_coupled import O6CoupledReferenceContactFeedback
                self.finger_feedback=O6CoupledReferenceContactFeedback(model,self.reference,finger_config)
            elif finger_config.get('geometry_backend')=='convex_sat':
                from fourth_contact_feedback_segmented import SegmentedReferenceContactFeedback
                self.finger_feedback=SegmentedReferenceContactFeedback(model,self.reference,finger_config)
            elif finger_config.get('table_clearance_guard',{}).get('enabled',False):
                from fourth_contact_feedback_tableguard import ReferenceContactFeedbackWithTableGuard
                self.finger_feedback=ReferenceContactFeedbackWithTableGuard(model,self.reference,finger_config)
            else:self.finger_feedback=ReferenceContactFeedback(model,self.reference,finger_config)
        for key,value in passive_contract.items():
            if not np.array_equal(getattr(model,key)[self.passive_aids],value):raise ValueError('Feedback constructor reactivated passive mimic motors: '+key)

    interpolate=Controller.interpolate

    def reset(self,data):
        for h in self.history.values():h.clear()
        self.target_history.clear();self.last[:]=0;self.previous_imu=None;self.previous_gravity=None;self.next_inference_time=0.;self.inference_count=0
        if self.arm_feedback is not None:self.arm_feedback.reset();self.finger_feedback.reset()
        self.hybrid_policy_records=[]

    def action(self,data,t):
        elapsed=float(t);t=elapsed+self.times[0];allq=self.interpolate(self.qref[:,7:],t);ctrl=np.zeros(self.model.nu);ctrl[self.ref_aids]=allq;ctrl[self.passive_aids]=0;did_infer=False
        if elapsed+1e-8>=self.next_inference_time:
            r=data.xmat[self.base].reshape(3,3);vel=np.empty(6);mujoco.mj_objectVelocity(self.model,data,mujoco.mjtObj.mjOBJ_BODY,self.base,vel,0)
            imu=r.T@vel[:3];gravity=r.T@np.array([0.,0.,-1.]);frac=np.clip(self.imu_latency/self.policy_dt,0,1)
            observed=imu if self.previous_imu is None else (1-frac)*imu+frac*self.previous_imu
            gravobs=gravity if self.previous_gravity is None else (1-frac)*gravity+frac*self.previous_gravity
            self.previous_imu,self.previous_gravity=imu.copy(),gravity.copy()
            current={'base_ang_vel':observed,'joint_pos':data.qpos[self.qa]-self.default,'joint_vel':data.qvel[self.va].copy(),'actions':self.last.copy(),'gravity_dir':gravobs}
            for k in self.terms:
                if not self.history[k]:self.history[k].extend(current[k].copy()for _ in range(10))
                else:self.history[k].append(current[k].copy())
            prop=np.concatenate([np.asarray(self.history[k]).reshape(-1)for k in self.terms])
            ft=np.minimum(t+np.arange(10)*self.future_dt,self.times[-1])
            qf=np.array([self.interpolate(self.body_q,x)for x in ft]);dqf=reference_joint_velocity(self.times,self.body_q,ft,self.reference_velocity_mode,self.body_dq);future=np.r_[qf.ravel(),dqf.ravel()].reshape(10,56)
            heading=sampled_heading_matrix(r,self.heading_mode)
            orientation=np.array([(heading.T@quaternion_matrix(self.interpolate(self.qref[:,3:7],x)))[:,:2].reshape(6)for x in ft])
            torch=self.torch
            self.tokenizer['command_multi_future_nonflat']=torch.as_tensor(future,dtype=torch.float32,device=self.device).reshape(1,1,10,56)
            self.tokenizer['motion_anchor_ori_heading_mf_nonflat']=torch.as_tensor(orientation,dtype=torch.float32,device=self.device).reshape(1,1,10,6)
            flat=torch.cat([self.tokenizer[k].reshape(1,1,-1)for k in self.cfg.env_config.obs.group_obs_names.tokenizer],dim=-1)
            inputs=self.TensorDict({'actor_obs':torch.as_tensor(prop,dtype=torch.float32,device=self.device).reshape(1,1,900),'tokenizer':flat},batch_size=[1,1])
            with torch.inference_mode():action=self.policy(inputs).reshape(28).detach().cpu().numpy()
            if not np.isfinite(action).all():raise ValueError('Nonfinite new SONIC action')
            self.last=np.clip(action,-float(self.cfg.env_config.action_clip_value),float(self.cfg.env_config.action_clip_value));self.target=self.default+self.scale*self.last
            self.last_observation=prop.copy();self.inference_count+=1;self.next_inference_time+=self.policy_dt
            did_infer=True
        self.target_history.append(self.target.copy());ctrl[self.aids]=self.target_history[0]
        if self.arm_feedback is not None:
            ctrl=self.arm_feedback.apply(data,ctrl,elapsed)
            ctrl=self.arm_feedback.finger_preshape(data,ctrl)
            ctrl=self.finger_feedback.apply(data,ctrl,elapsed,contact_plan=self.arm_feedback.contact_plan)
            for side,chain in enumerate(self.arm_feedback.chains):
                if self.arm_feedback.state[side]is not None or self.arm_feedback.last_targets[side]is not None:
                    indices=np.asarray([self.names.index(n)for n in chain['names']])
                    self.last[indices]=(ctrl[chain['aids']]-self.default[indices])/self.scale[indices]
            if did_infer:
                self.hybrid_policy_records.append({'time_s':elapsed,'learned_action':action.copy(),'effective_last_action':self.last.copy(),'learned_body_target':self.target.copy(),'applied_body_target':ctrl[self.aids].copy()})
        ctrl[self.passive_aids]=0
        return ctrl

    def dump_history(self,output_dir):
        if self.arm_feedback is None:return {}
        out=Path(output_dir);out.mkdir(parents=True,exist_ok=True);path=out/'sonic_contact_feedback.npz'
        payload=self.arm_feedback.history_arrays();payload.update({'finger_feedback_'+k:v for k,v in self.finger_feedback.history_arrays().items()})
        if self.hybrid_policy_records:payload.update({'hybrid_policy_'+k:np.asarray([r[k]for r in self.hybrid_policy_records])for k in self.hybrid_policy_records[0]})
        np.savez_compressed(path,**payload)
        report={'arm':self.arm_feedback.report(),'finger':self.finger_feedback.report()}
        rp=out/'sonic_contact_feedback.json';rp.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
        return {'trace':{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()},'report':{'path':str(rp),'sha256':hashlib.sha256(rp.read_bytes()).hexdigest()}}

    def report(self):
        report={'controller':'new_ft15225 actual FSQ G1-only encoder + 900D actor, 28-body implicit PD + auxiliary reference PD',
            'policy_path':str(self.policy_path),'policy_sha256':hashlib.sha256(self.policy_path.read_bytes()).hexdigest(),
            'device':str(self.device),'policy_dt':self.policy_dt,'physics_dt':self.physics_dt,'future_spacing_s':self.future_dt,
            'delay_physics_steps':self.delay_steps,'imu_latency_s':self.imu_latency,'history_order':list(self.terms),'history_layout':'term-major oldest-first',
            'ankle_pitch_startup_hard_limits_rad':self.ankle_pitch_hard_limits,
            'passive_compatibility_actuator_names':[mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_ACTUATOR,int(i))for i in self.passive_aids],
            'passive_actuator_contract':'gear/gain/bias left exactly zero; ctrl set zero on every physics-step packet',
            'future_q_dq_layout':'all future q flat then all future dq flat, reshape(10,56)',
            'heading_mode':self.heading_mode,
            'reference_velocity_mode':self.reference_velocity_mode,
            'reference_velocity_query_dt_s':.02 if self.reference_velocity_mode=='training_forward_50hz'else None,
            'reference_velocity_rule':'At every future query use(q_interp(min(t+.02,end))-q_interp(t))/.02; t=end uses(q_end-q_interp(max(end-.02,start)))/.02'if self.reference_velocity_mode=='training_forward_50hz'else'Legacy np.gradient on original execution nodes, then interpolate cached velocity',
            'training_source_velocity_rule':'50Hz-resampled dof positions, forward difference divided bydt; loader terminal padding copies dof_vel[:, -2:-1]. Adapter terminal backward difference is an explicit endpoint convention.',
            'future_orientation':'inverse robot Z-twist heading normalize(w,0,0,z) from sampled rotation times future orientation; first2 columns row-major'if self.heading_mode=='training_z_twist'else'legacy inverse Euler/projected-X yaw times future orientation; first2 columns row-major',
            'body_joint_names':self.names,'inference_count':self.inference_count,'actuation':'position','strict_original_articulation_reproduction':False,
            'source_code_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in self.source_files},
            'limitations':['Actual saved checkpoint strictly loaded; current source fork lacks saved robot_type alias vita02a_28dof.',
                'MuJoCo native implicit PD approximates Isaac; parallel ankle coupled motor projection is not applied in this native position mode.',
                'Median training friction is an assumed evaluation setting; hand/neck DOFs extend the original articulation.']}
        if self.arm_feedback is not None:
            report['arm_object_feedback']=self.arm_feedback.report();report['finger_contact_feedback']=self.finger_feedback.report()
            report['controller']='new_ft15225 legs/waist + actual-object arm IK + force-aware finger PD (hybrid adaptation)'
            report['hybrid_previous_action_observation']='Effective IK arm targets normalized by original default/scale; original learned raw actions retained for legs and waist.'
            report['hybrid_arm_control_delay_physics_steps']=0
        return report
