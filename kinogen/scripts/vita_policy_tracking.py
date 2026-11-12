#!/usr/bin/env python3
"""CPU-only Vita28 checkpoint loading and transparent standalone MuJoCo trial.

No training, pose replay during control, package installation, or simulator
state overwrite in an active rollout. This is a cross-simulator feasibility
baseline, not a claim that the missing original Vita IsaacLab environment has
been reproduced. Reports distinguish strict network loading from physical
tracking. The source 52-joint clip is a reference, never measured contact.
"""
from __future__ import annotations
import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import types
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_NEW_SHA = '0a825e57cb9afe7cda15dd3103e1b2715aeec2f6d7e8fc0cedf529044bebc6c1'


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        while b := f.read(8*1024*1024): h.update(b)
    return h.hexdigest()


def receipt(p):
    return {'path': str(Path(p).resolve()), 'sha256': sha(p), 'bytes': Path(p).stat().st_size}


def json_write(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf-8')


class MetadataContainer:
    """Inert storage for training-only class metadata from the supplied archive."""
    def __new__(cls, *args, **kwargs): return object.__new__(cls)
    def __init__(self, *args, **kwargs): pass
    def __setstate__(self, state):
        if isinstance(state, dict): self.__dict__.update(state)
        else: self.metadata_state = state


class MetadataUnpickler(pickle.Unpickler):
    substitutions = []
    def find_class(self, module, name):
        inert = {('accelerate.state', 'PartialState'),
                 ('transformers.trainer_pt_utils', 'AcceleratorConfig'),
                 ('trl.experimental.ppo.ppo_trainer', 'OnlineTrainerState'),
                 ('trl.trainer.ppo_trainer', 'OnlineTrainerState'),
                 ('trl.trainer.ppo_config', 'PPOConfig')}
        enums = set()
        enums.add(('accelerate.utils.dataclasses', 'DistributedType'))
        enums.update(('transformers.trainer_utils', x) for x in ('HubStrategy', 'IntervalStrategy', 'SaveStrategy', 'SchedulerType'))
        enums.add(('transformers.training_args', 'OptimizerNames'))
        if (module, name) in inert or (module, name) in enums:
            self.substitutions.append(module+'.'+name)
            return str if (module, name) in enums else MetadataContainer
        return super().find_class(module, name)


def load_checkpoint(path):
    import torch
    if sha(path) != EXPECTED_NEW_SHA:
        raise ValueError('Expected the supplied new_ft15225 checkpoint SHA256')
    pm = types.ModuleType('vita_metadata_pickle')
    pm.Unpickler = MetadataUnpickler
    pm.load, pm.loads = pickle.load, pickle.loads
    # Tensor deserialization uses PyTorch's original storage loader. Only the
    # listed training metadata classes are replaced; tensor bytes are unchanged.
    return torch.load(path, map_location='cpu', pickle_module=pm, weights_only=False)


def make_policy(ckpt, model_config, sonic_root):
    import torch
    from omegaconf import OmegaConf
    sys.path.insert(0, str(sonic_root))
    from gear_sonic.trl.modules.actor_critic_modules import Actor
    cfg = load_config(model_config)
    actor_cfg = cfg.algo_config.actor.copy()
    actor_cfg.pop('_target_')
    policy = Actor(env_config=cfg.env_config, algo_config=cfg.algo_config, **actor_cfg).cpu().eval()
    state = ckpt['policy_state_dict']
    if not all(torch.isfinite(v).all() for v in state.values() if torch.is_tensor(v)):
        raise ValueError('Nonfinite checkpoint policy tensor')
    result = policy.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys: raise ValueError(str(result))
    return policy, cfg


def actor_input(policy, cfg, proprio, future_joint, future_orientation):
    import torch
    from tensordict import TensorDict
    tokenizer = {}
    for name, shape in cfg.env_config.obs.group_obs_dims.tokenizer.items():
        tokenizer[name] = torch.zeros((1, 1, *list(shape)), dtype=torch.float32)
    tokenizer['encoder_index'][..., 0] = 1
    tokenizer['command_multi_future_nonflat'] = torch.as_tensor(future_joint, dtype=torch.float32).reshape(1,1,10,56)
    key = 'motion_anchor_ori_heading_mf_nonflat'
    if key not in tokenizer: raise ValueError('The requested checkpoint must use heading orientation')
    tokenizer[key] = torch.as_tensor(future_orientation, dtype=torch.float32).reshape(1,1,10,6)
    # Other encoders are inactive. Their explicit placeholders are never treated
    # as SMPL/teleop measurements and cannot contribute to the G1-only branch.
    flat_tokenizer = torch.cat([tokenizer[k].reshape(1,1,-1) for k in cfg.env_config.obs.group_obs_names.tokenizer],dim=-1)
    data = TensorDict({'actor_obs': torch.as_tensor(proprio, dtype=torch.float32).reshape(1,1,900),
                       'tokenizer': flat_tokenizer}, batch_size=[1,1])
    with torch.inference_mode(): result = policy(data).reshape(28)
    if not torch.isfinite(result).all(): raise ValueError('Nonfinite real checkpoint forward')
    return result.numpy().copy()


def motion_read(path, names):
    import numpy as np
    with np.load(path, allow_pickle=False) as z: d = {k:z[k] for k in z.files}
    # Inputs are the root's inert-pickle-to-NPZ conversion, not executable pickle.
    candidates = ('joint_names', 'dof_names')
    source_names = next((d[k].tolist() for k in candidates if k in d), None)
    if source_names is None: raise ValueError('Motion NPZ requires explicit joint names')
    qkey = next((k for k in ('dof_pos', 'joint_positions', 'joint_pos', 'qpos') if k in d), None)
    if qkey is None: raise ValueError('Motion NPZ has no recognized joint positions: '+str(list(d)))
    q = np.asarray(d[qkey], dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != len(source_names): raise ValueError('Joint name/shape disagreement')
    root = next((d[k] for k in ('root_trans_offset', 'root_translation', 'root_pos') if k in d), None)
    quat = next((d[k] for k in ('root_rot_xyzw', 'root_rot', 'root_quaternion', 'root_quat') if k in d), None)
    if root is None or quat is None: raise ValueError('Explicit root translation/quaternion required')
    fps = float(d['fps']) if 'fps' in d else 30.
    # Source archive records root_rot as XYZW; a declared contrary layout is rejected.
    order = str(d.get('quaternion_order', 'xyzw'))
    if order != 'xyzw': raise ValueError('Expected audited source XYZW root quaternion')
    if any(n not in source_names for n in names): raise ValueError('Missing policy joint in 52-joint clip')
    return {'q_all':q, 'q':q[:,[source_names.index(n) for n in names]], 'names':source_names,
            'root':np.asarray(root), 'quat_xyzw':np.asarray(quat), 'fps':fps, 'source_keys':list(d)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    base = ROOT/'data/vita_demo_20261026'
    p.add_argument('--checkpoint', type=Path, default=base/'checkpoints/new_ft15225/model_step_015225.pt')
    p.add_argument('--model-config', type=Path, default=base/'checkpoints/new_ft15225/model_config.yaml')
    p.add_argument('--training-config', type=Path, default=base/'checkpoints/new_ft15225/config.yaml')
    p.add_argument('--contract', type=Path, default=ROOT/'reports/vita17-old-policy-contract.json')
    p.add_argument('--sonic-root', type=Path, default=Path('/data/grail/imports/SONIC'))
    p.add_argument('--motion', type=Path)
    p.add_argument('--mode', choices=('probe','rollout'), default='probe')
    p.add_argument('--output-dir', type=Path, default=ROOT/'outputs/vita_policy_smoke')
    p.add_argument('--seconds', type=float, default=2.)
    p.add_argument('--mujoco-site', type=Path, default=Path('/data/grail-envs/retarget/lib/python3.10/site-packages'))
    args = p.parse_args()
    if os.name != 'posix': p.error('All ML/physics runs are server-only; local machine is preview-only')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    import numpy as np
    torch.set_num_threads(4)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    # Keep the actual adapter source beside each trial, including failed ones.
    snapshot=out/'source_vita_policy_tracking.py'
    if snapshot.exists() and sha(snapshot)!=sha(__file__):
        raise ValueError('Use a new trial output directory when the adapter changes')
    if not snapshot.exists(): snapshot.write_bytes(Path(__file__).read_bytes())
    report = {'schema':'vita28.policy_feasibility.v1', 'status':'not_verified', 'gpu_used':False,
              'training_launched':False, 'physics_tracking_verified':False,
              'script':receipt(__file__), 'sources':{}, 'limitations':[
                  'The supplied full original Vita IsaacLab robot registry, 28DOF MJCF and deploy module are absent.',
                  'A strict policy load is not an independent proof of observation or simulator equivalence.',
                  'Reference clips do not supply measured object pose, contact force, torque or support surfaces.',
                  'Standalone MuJoCo is explicitly a cross-simulator baseline with the supplied 52DOF URDF, not the original trained dynamics.']}
    try:
        for k in ('checkpoint','model_config','training_config','contract'):
            report['sources'][k] = receipt(getattr(args,k))
        contract = json.loads(args.contract.read_text(encoding='utf-8'))['contract']
        ckpt = load_checkpoint(args.checkpoint)
        report['checkpoint_keys'] = list(ckpt)
        report['metadata_class_substitutions'] = sorted(set(MetadataUnpickler.substitutions))
        policy, cfg = make_policy(ckpt, args.model_config, args.sonic_root)
        report['strict_policy_loading_passed'] = True
        report['policy_tensor_shapes'] = {k:list(v.shape) for k,v in ckpt['policy_state_dict'].items()}
        report['checkpoint_env_state'] = summarize(ckpt.get('env_state_dict', {}))
        report['checkpoint_training_semantics'] = summarize(ckpt.get('training_semantics', {}))
        from omegaconf import OmegaConf
        train_cfg = OmegaConf.load(args.training_config)
        report['new_config_future_spacing_s'] = float(train_cfg.manager_env.commands.motion.dt_future_ref_frames)
        report['old_export_contract_future_spacing_s'] = float(contract['future_reference']['dt'])
        report['future_spacing_conflict'] = report['new_config_future_spacing_s'] != report['old_export_contract_future_spacing_s']
        report['new_ankle_vs_old_contract'] = {
            n:{k:{'new':float(cfg.env_config.ankle_deploy_contract.joints[n][k]),'old':float(contract['ankle']['joints'][n][k]),
                  'equal':abs(float(cfg.env_config.ankle_deploy_contract.joints[n][k])-float(contract['ankle']['joints'][n][k]))<1e-7}
               for k in ('kp','kd','armature','action_scale')}
            for n in cfg.env_config.ankle_deploy_contract.joints}
        report['nonankle_new_yaml_pd_present'] = False
        report['pd_source_note'] = 'New YAML supplies only ankle gains. Nonankle arrays come from old supplied audit unless checkpoint env_state independently confirms them.'
        if args.motion is None: args.motion = sorted((ROOT/'data/vita_demo_20261026/motions').glob('*.npz'))[0]
        report['sources']['motion'] = receipt(args.motion)
        motion = motion_read(args.motion, contract['joint_names'])
        report['motion_keys'] = motion['source_keys']
        # Real clip reference into the checkpoint; no synthetic action or force.
        proprio = np.zeros(900, dtype=np.float32)
        future = np.zeros((10,56), dtype=np.float32)
        ori = np.tile([1,0,0,1,0,0], (10,1)).astype(np.float32)
        a0 = actor_input(policy,cfg,proprio,future,ori)
        future[:, :28] = motion['q'][0]
        a1 = actor_input(policy,cfg,proprio,future,ori)
        np.savez_compressed(out/'policy_probe.npz', action_zero_diagnostic=a0, action_reference_diagnostic=a1)
        report['network_forward_diagnostic'] = {'action_shape':list(a1.shape), 'finite':bool(np.isfinite(a1).all()),
                                              'max_action_abs':float(np.max(abs(a1))), 'reference_changes_action_max_abs':float(np.max(abs(a1-a0))),
                                              'not_physics_or_exact_observation':True}
        if args.mode == 'rollout':
            report['rollout'] = rollout(args,policy,cfg,contract,motion)
            report['status'] = ('physical_rollout_completed_conditional' if report['rollout']['end_reason']=='requested_duration'
                                else 'physical_rollout_failed_tracking')
        else: report['status'] = 'strict_policy_loaded_and_forward_only'
    except Exception as exc:
        import traceback
        report['error'] = str(exc)
        report['traceback'] = traceback.format_exc()
    json_write(out/'report.json',report)
    print(json.dumps({'report':receipt(out/'report.json'),'status':report['status'],'error':report.get('error')},indent=2))
    if report['status']=='not_verified': return 2
    return 3 if report['status']=='physical_rollout_failed_tracking' else 0


def summarize(v, depth=0):
    import torch
    if torch.is_tensor(v):
        r={'shape':list(v.shape),'dtype':str(v.dtype)}
        if v.numel() <= 120: r['values']=v.cpu().tolist()
        return r
    if isinstance(v,dict): return {str(k):summarize(x,depth+1)for k,x in v.items()}
    if isinstance(v,(list,tuple)) and len(v)<150: return [summarize(x,depth+1)for x in v]
    if v is None or isinstance(v,(str,bool,int,float)): return v
    return {'metadata_type':str(type(v))}


def load_config(path):
    import yaml
    from omegaconf import OmegaConf
    class MetadataYamlLoader(yaml.SafeLoader): pass
    # The sole Python YAML tag in the supplied model_config is a serialized
    # pathname. Store it as text; never enable unsafe Python YAML constructors.
    MetadataYamlLoader.add_constructor('tag:yaml.org,2002:python/object/apply:pathlib.PosixPath',
                                      lambda loader,node: str(Path(*loader.construct_sequence(node))))
    return OmegaConf.create(yaml.load(Path(path).read_text(encoding='utf-8'),Loader=MetadataYamlLoader))


def rollout(args, policy, cfg, contract, motion):
    import numpy as np
    from scipy.spatial.transform import Rotation, Slerp
    sys.path.append(str(args.mujoco_site))
    import mujoco
    urdf = ROOT/'data/vita_demo_20261026/robot/vita02a/urdf/vita02a_linkerhand_meshcoll_relative.urdf'
    xml = ET.parse(urdf).getroot()
    for mesh in xml.findall('.//mesh'):
        name = mesh.get('filename')
        if name.startswith('package://'): raise ValueError('Use audited relative URDF')
        path = (urdf.parent/name).resolve()
        if not path.is_file(): raise FileNotFoundError(path)
        mesh.set('filename', str(path))
    extension = xml.find('mujoco')
    if extension is None: extension = ET.SubElement(xml,'mujoco')
    compiler = extension.find('compiler')
    if compiler is None: compiler = ET.SubElement(extension,'compiler')
    compiler.attrib.update(discardvisual='false',fusestatic='false',strippath='false')
    # Preserve all supplied masses/inertias. Inconsistent tensors must fail,
    # rather than reuse the renderer's explicitly nonphysical regularization.
    for key in ('boundmass','boundinertia','balanceinertia'): compiler.attrib.pop(key,None)
    spec = mujoco.MjSpec.from_string(ET.tostring(xml,encoding='unicode'))
    spec.body('base_link').add_freejoint(name='vita_policy_free_root')
    spec.option.timestep = .005
    spec.option.gravity = [0,0,-9.81]
    spec.option.disableflags = 0
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.worldbody.add_geom(name='declared_floor_z0',type=mujoco.mjtGeom.mjGEOM_PLANE,
                           pos=[0,0,0],size=[0,0,.02],friction=[1.,.005,.0001],
                           contype=1,conaffinity=1,group=2)
    spec.compile()
    mjxml = ET.fromstring(spec.to_xml())
    ankle_overrides=cfg.env_config.ankle_deploy_contract.joints
    for joint in mjxml.findall('.//joint'):
        if joint.get('name')in ankle_overrides:
            joint.set('armature',str(float(ankle_overrides[joint.get('name')].armature)))
    actxml = mjxml.find('actuator')
    if actxml is None: actxml = ET.SubElement(mjxml,'actuator')
    source_joint = {j.attrib['name']:j for j in xml.findall('joint')if j.attrib['type']!='fixed'}
    for n,j in source_joint.items():
        effort = float(j.find('limit').get('effort'))
        if effort<=0: raise ValueError('No positive source effort specification: '+n)
        if n in contract['joint_names']:
            i=contract['joint_names'].index(n)
            gain,damp=contract['joint_stiffness'][i],contract['joint_damping'][i]
        else: gain,damp=1.,.05
        # Native position+velocity feedback participates in implicitfast's
        # velocity derivatives, matching an implicit PD rather than an unstable
        # externally evaluated explicit damping torque for tiny finger inertia.
        ET.SubElement(actxml,'position',name='policy_motor_'+n,joint=n,gear='1',
                      kp=str(gain),kv=str(damp),forcelimited='true',forcerange=f'{-effort} {effort}')
    model = mujoco.MjModel.from_xml_string(ET.tostring(mjxml,encoding='unicode'))
    model_path=args.output_dir/'physical_trial_model.xml'
    model_path.write_text(ET.tostring(mjxml,encoding='unicode'),encoding='utf-8')
    if model.nq!=59 or model.nv!=58 or model.nu!=52: raise ValueError('Not full52+freebase model')
    ids={n:mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n)for n in source_joint}
    qadr={n:int(model.jnt_qposadr[j])for n,j in ids.items()}
    vadr={n:int(model.jnt_dofadr[j])for n,j in ids.items()}
    aids={n:mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_ACTUATOR,'policy_motor_'+n)for n in source_joint}
    body_names=list(contract['joint_names'])
    extra=[n for n in source_joint if n not in body_names]
    qa=np.asarray([qadr[n]for n in body_names]);va=np.asarray([vadr[n]for n in body_names])
    aa=np.asarray([aids[n]for n in body_names]);allnames=list(source_joint)
    allqa=np.asarray([qadr[n]for n in allnames]);allva=np.asarray([vadr[n]for n in allnames]);allaa=np.asarray([aids[n]for n in allnames])
    dq0=np.asarray(contract['default_joint_pos']);kp=np.asarray(contract['joint_stiffness']);kd=np.asarray(contract['joint_damping'])
    scale=np.asarray(contract['action_scale'])
    # The only armature values actually supplied in both model contracts.
    dat=mujoco.MjData(model)
    freeid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'vita_policy_free_root')
    fq=int(model.jnt_qposadr[freeid]);fv=int(model.jnt_dofadr[freeid])
    times=np.arange(len(motion['q']))/motion['fps']
    rotations=Rotation.from_quat(motion['quat_xyzw'])
    slerp=Slerp(times,rotations)
    def qref(t,all_joints=False):
        src=motion['q_all']if all_joints else motion['q']
        return np.asarray([np.interp(t,times,src[:,k])for k in range(src.shape[1])]).T
    def rref(t): return slerp(np.clip(np.atleast_1d(t),times[0],times[-1]))
    def qvelref(t):
        lo=np.clip(np.asarray(t)-.01,times[0],times[-1]);hi=np.clip(np.asarray(t)+.01,times[0],times[-1])
        return (qref(hi)-qref(lo))/np.maximum((hi-lo)[...,None],1e-9)
    # Exactly one initialization; active simulation below never assigns qpos,
    # qvel, root pose, mocap, qfrc_applied or xfrc_applied.
    dat.qpos[fq:fq+3]=motion['root'][0]
    dat.qpos[fq+3:fq+7]=motion['quat_xyzw'][0,[3,0,1,2]]
    dat.qpos[allqa]=motion['q_all'][0,[motion['names'].index(n)for n in allnames]]
    dat.qvel[va]=qvelref(0.)
    dat.qvel[fv:fv+3]=(motion['root'][1]-motion['root'][0])*motion['fps']
    dat.qvel[fv+3:fv+6]=(rotations[0].inv()*rotations[1]).as_rotvec()*motion['fps']
    mujoco.mj_forward(model,dat)
    baseid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'base_link')
    terms=('base_ang_vel','joint_pos','joint_vel','actions','gravity_dir')
    history={k:deque(maxlen=10)for k in terms}
    last=np.zeros(28);records=[];contact_rows=[];steps=0
    horizon=min(args.seconds,times[-1])
    if not np.isfinite(horizon)or horizon<=0: raise ValueError('Positive rollout duration required')
    end_reason='requested_duration'
    future_dt=float(cfg.env_config.get('future_spacing_override_for_trial',.1))
    # Saved new YAML defines .1s; source checkpoint semantics are disclosed in
    # the report alongside the old .02s exported contract. Never guess silently.
    from omegaconf import OmegaConf
    tc=OmegaConf.load(args.training_config)
    future_dt=float(tc.manager_env.commands.motion.dt_future_ref_frames)
    warn0=np.asarray([x.number for x in dat.warning]).copy()
    while dat.time < horizon-1e-9:
        t=float(dat.time);q=dat.qpos[qa].copy();vel=dat.qvel[va].copy()
        rot=Rotation.from_quat(dat.qpos[fq+3:fq+7][[1,2,3,0]])
        objvel=np.zeros(6);mujoco.mj_objectVelocity(model,dat,mujoco.mjtObj.mjOBJ_BODY,baseid,objvel,1)
        current={'base_ang_vel':objvel[:3].copy(),'joint_pos':q-dq0,'joint_vel':vel,
                 'actions':last.copy(),'gravity_dir':rot.inv().apply([0,0,-1])}
        for k in terms:
            if not history[k]: history[k].extend(current[k].copy()for _ in range(10))
            else: history[k].append(current[k].copy())
        prop=np.concatenate([np.asarray(history[k]).reshape(-1)for k in terms])
        ft=np.minimum(t+np.arange(10)*future_dt,times[-1])
        # Match commands.py: flatten all future q, then all future dq, then
        # reshape10x56. Do not silently interleave q/dq per source frame.
        fj=np.concatenate([qref(ft).reshape(-1),qvelref(ft).reshape(-1)]).reshape(10,56)
        yaw=np.arctan2(rot.as_matrix()[1,0],rot.as_matrix()[0,0])
        heading=Rotation.from_euler('z',yaw)
        ori=(heading.inv()*rref(ft)).as_matrix()[...,:2].reshape(10,6)
        action=actor_input(policy,cfg,prop,fj,ori)
        target=dq0+np.clip(action,-float(cfg.env_config.action_clip_value),float(cfg.env_config.action_clip_value))*scale
        alltarget=qref(t,all_joints=True)[[motion['names'].index(n)for n in allnames]]
        for n,value in zip(body_names,target):alltarget[allnames.index(n)]=value
        last=action.copy()
        tau_before=np.zeros(model.nu)
        for _ in range(4):
            # The control variable is a desired joint angle; recorded torque
            # is the real solver actuator_force, never mislabeled ctrl angles.
            dat.ctrl[allaa]=alltarget
            if np.any(dat.qfrc_applied)or np.any(dat.xfrc_applied): raise ValueError('External generalized/body force introduced')
            previous=float(dat.time);mujoco.mj_step(model,dat);steps+=1
            tau_before=dat.actuator_force.copy()
            if abs(dat.time-previous-.005)>1e-8: raise ValueError('Simulator reset/time discontinuity')
            if not np.isfinite(dat.qpos).all()or not np.isfinite(dat.qvel).all():raise ValueError('Nonfinite physical state')
        forces=np.zeros((2,3));pairrows=[]
        for ci in range(dat.ncon):
            c=dat.contact[ci];f=np.zeros(6);mujoco.mj_contactForce(model,dat,ci,f)
            world=c.frame.reshape(3,3).T@f[:3]
            bn=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[g]))or'world'for g in (c.geom1,c.geom2)]
            for side,prefix in enumerate(('left_ankle','right_ankle')):
                if any(n.startswith(prefix)for n in bn)and'declared_floor_z0'in(mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,g)for g in(c.geom1,c.geom2)):
                    forces[side]+=world if bn[1].startswith(prefix)else-world
            pairrows.append({'bodies':bn,'force_contact_frame_N':f[:3].tolist(),'world_force_on_geom2_N':world.tolist(),'signed_distance_m':float(c.dist)})
        records.append({'timestamps':float(dat.time),'qpos':dat.qpos.copy(),'qvel':dat.qvel.copy(),
                        'joint_positions':dat.qpos[allqa].copy(),'joint_velocities':dat.qvel[allva].copy(),
                        'joint_targets':alltarget.copy(),'policy_action':action,'motor_torque_Nm':tau_before[allaa].copy(),
                        'feet_force_world_N':forces,'reference_body_q':qref(float(dat.time)),
                        'root_position':dat.qpos[fq:fq+3].copy(),'root_quaternion_wxyz':dat.qpos[fq+3:fq+7].copy(),
                        'joint_tracking_rms_rad':float(np.sqrt(np.mean((dat.qpos[qa]-qref(float(dat.time)))**2)))})
        contact_rows.append({'time_s':float(dat.time),'contacts':pairrows})
        if dat.qpos[fq+2]<.35: end_reason='base_below_0p35m';break
        if np.any(np.asarray([x.number for x in dat.warning])>warn0):end_reason='mujoco_solver_warning';break
    data={k:np.asarray([r[k]for r in records])for k in records[0]}
    data.update(joint_names=np.asarray(allnames),body_joint_names=np.asarray(body_names),
                source_checkpoint_sha256=np.asarray(sha(args.checkpoint)),source_motion_sha256=np.asarray(sha(args.motion)),
                source_urdf_sha256=np.asarray(sha(urdf)))
    dest=args.output_dir/'physical_rollout.npz';np.savez_compressed(dest,**data)
    json_write(args.output_dir/'physical_contacts.json',contact_rows)
    return {'trajectory':receipt(dest),'model':receipt(model_path),'contact_records':receipt(args.output_dir/'physical_contacts.json'),
            'mujoco_version':mujoco.__version__,'frames':len(records),'physics_steps':steps,'policy_forward_calls':len(records),
            'duration_s':float(dat.time),'end_reason':end_reason,'gravity_m_s2':model.opt.gravity.tolist(),
            'contact_constraints_enabled':model.opt.disableflags==0,'nq':model.nq,'nv':model.nv,'actuators':model.nu,
            'active_state_writes':0,'active_external_wrenches':0,'root_is_free':True,
            'nonpolicy_reference_pd_joints':extra,'nonpolicy_pd_kp_Nm_rad':1.,'nonpolicy_pd_kd_Nm_s_rad':.05,
            'body_pd_source':'Supplied old checkpoint audit; new ankle values cross-checked; original full Vita dynamic registry absent',
            'actuator_implementation':'Native MuJoCo position kp/kv, implicitfast; torque from actual actuator_force after step',
            'contact_sample_phase':'Last real physics substep solver contact at t_recorded minus 0.005s; no synthetic force or object inference',
            'future_spacing_s':future_dt,'joint_command_layout':'q_all_frames_flat_then_dq_all_frames_flat_reshape10x56',
            'history_order':'base_ang_vel,joint_pos_relative_default,joint_vel,last_policy_action,gravity; term-major oldest-first',
            'foot_plane':'Declared z0 flat rigid plane; actual source support heights unknown',
            'object_or_hand_object_contacts':'not_available: no object asset/pose/support table supplied',
            'joint_tracking_rms_rad_median':float(np.median(data['joint_tracking_rms_rad'])),
            'joint_tracking_rms_rad_max':float(np.max(data['joint_tracking_rms_rad'])),
            'root_height_min_m':float(np.min(data['root_position'][:,2])),
            'tracking_success_claimed':False,'original_isaaclab_dynamics_reproduced':False}


if __name__ == '__main__': raise SystemExit(main())
