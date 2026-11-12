"""Deploy the newly trained residual PPO through real position actuators."""
from pathlib import Path
import hashlib
from collections import deque
import numpy as np


def checkpoint_sonic_protocol(checkpoint):
    """Resolve declared SONIC ABI, with explicit per-field legacy fallback."""
    if not checkpoint.get('sonic_base',False):return None
    nested=checkpoint.get('sonic_protocol')
    if nested is None:nested={}
    if not isinstance(nested,dict):raise ValueError('Checkpoint sonic_protocol must be a mapping')
    defaults={'heading_mode':'euler_yaw','reference_velocity_mode':'legacy_centered'}
    allowed={'heading_mode':{'euler_yaw','training_z_twist'},'reference_velocity_mode':{'legacy_centered','training_forward_50hz'}}
    result={};fallback=[]
    for name,default in defaults.items():
        if name in checkpoint and name in nested and checkpoint[name]!=nested[name]:raise ValueError('Conflicting checkpoint SONIC metadata: '+name)
        if name in checkpoint:value=checkpoint[name]
        elif name in nested:value=nested[name]
        else:value=default;fallback.append(name)
        if not isinstance(value,str)or value not in allowed[name]:raise ValueError('Unsupported checkpoint '+name+': '+str(value))
        result[name]=value
    result['metadata_source']='legacy_missing_metadata_fallback'if len(fallback)==2 else'partial_metadata_with_legacy_field_fallback'if fallback else'explicit_checkpoint_metadata'
    result['legacy_fallback_fields']=fallback
    return result


class TaskPPOController:
    def __init__(self,model,reference_path,checkpoint_path,model_path,device='cuda:0'):
        import torch
        from rgb145_ppo_env import Reference,ActorCritic,POLICY_SENSOR_CONTRACT
        self.torch=torch;self.model=model;self.device=device;self.path=Path(checkpoint_path)
        torch.set_num_threads(2)
        ckpt=torch.load(self.path,map_location=device,weights_only=False)
        if ckpt['reference_sha256']!=hashlib.sha256(Path(reference_path).read_bytes()).hexdigest():raise ValueError('Task policy/reference SHA mismatch')
        if ckpt['model_sha256']!=hashlib.sha256(Path(model_path).read_bytes()).hexdigest():raise ValueError('Task policy/model SHA mismatch')
        self.control_bound_mode=ckpt.get('control_bound_mode','legacy_joint')
        self.checkpoint_reset_rule=ckpt.get('episode_reset_rule')
        self.checkpoint_physical_quality_training=bool(ckpt.get('physical_quality_training',False));self.checkpoint_physical_quality_training_contract=ckpt.get('physical_quality_training_contract')
        self.ref=Reference(model,reference_path,device,control_bound_mode=self.control_bound_mode);self.actor=ActorCritic(ckpt['obs_dim']).to(device);self.actor.load_state_dict(ckpt['model'],strict=True);self.actor.eval()
        self.base=None
        self.checkpoint_protocol=checkpoint_sonic_protocol(ckpt)
        if ckpt.get('sonic_base',False):
            from rgb145_batch_sonic import BatchSonic
            self.base=BatchSonic(model,self.ref,1,device,heading_mode=self.checkpoint_protocol['heading_mode'],reference_velocity_mode=self.checkpoint_protocol['reference_velocity_mode'])
        self.scale=float(ckpt['action_scale']);self.iteration=int(ckpt['iteration'])
        self.sensor_contract=POLICY_SENSOR_CONTRACT
        self.aids=self.ref.body_aids;self.qa=self.ref.qa;self.jref=self.ref.body_cols;self.names=self.ref.body_names
        self.arm_feedback=None;self.policy_dt=.02;self.physics_dt=float(model.opt.timestep);self.target_history=deque(maxlen=1)
        self.last=np.zeros(28);self.target=np.zeros(28);self.inference_count=0;self.next_time=0.;self.ctrl=np.zeros(model.nu)
        self.last_base_target=np.zeros(28)

    def reset(self,data):
        self.last[:]=0.;self.inference_count=0;self.next_time=0.;self.ctrl[:]=0.;self.target_history.clear()
        if self.base is not None:self.base.reset(self.torch.tensor([0],device=self.device))

    def action(self,data,t):
        if t+1e-8>=self.next_time:
            torch=self.torch
            q=torch.as_tensor(data.qpos.copy(),dtype=torch.float32,device=self.device)[None]
            v=torch.as_tensor(data.qvel.copy(),dtype=torch.float32,device=self.device)[None]
            x=torch.as_tensor(data.xpos.copy(),dtype=torch.float32,device=self.device)[None]
            phase=torch.as_tensor([t],dtype=torch.float32,device=self.device)
            previous=torch.as_tensor(self.last,dtype=torch.float32,device=self.device)[None]
            with torch.inference_mode():
                obs=self.ref.observations(q,v,x,phase,previous)
                if self.base is not None:
                    b=self.base.c.base;r=torch.as_tensor(data.xmat[b].reshape(3,3).copy(),device=self.device,dtype=torch.float32)[None];omega=torch.as_tensor(data.cvel[b,:3].copy(),device=self.device,dtype=torch.float32)[None]
                    base=self.base.targets(q,v,phase,r,omega);nominal=self.ref.controls(phase,torch.zeros_like(previous),self.scale);nominal[:,self.base.aids]=base
                    self.last_base_target=nominal[0,self.ref.body_aids_t].cpu().numpy().copy()
                    target,_,_=self.ref.sample(phase);obs=torch.cat([obs,nominal[:,self.ref.body_aids_t]-target[:,self.ref.qa_t]],dim=-1)
                action=self.actor.actor(self.actor.normalize(obs))
                if self.base is None:packet=self.ref.controls(phase,action,self.scale)
                else:
                    packet=nominal;ids=self.ref.body_aids_t;packet[:,ids]=(packet[:,ids]+self.scale*torch.tanh(action)).clamp(self.ref.lower,self.ref.upper)
                    self.base.record_applied(packet)
            self.last=action[0].cpu().numpy();self.ctrl=packet[0].cpu().numpy();self.target=self.ctrl[self.aids].copy()
            self.target_history.append(self.target.copy());self.inference_count+=1;self.next_time+=self.policy_dt
        return self.ctrl.copy()

    def report(self):
        report={'controller':'Task-specific PPO residual motion imitation; new trained28-body actor, auxiliary reference PD',
                'policy_path':str(self.path),'policy_sha256':hashlib.sha256(self.path.read_bytes()).hexdigest(),
                'training_iteration':self.iteration,'policy_dt':self.policy_dt,'physics_dt':self.physics_dt,
                'inference_count':self.inference_count,'body_joint_names':self.names,'residual_scale_rad':self.scale,
                'is_original_sonic_checkpoint':False,'privileged_simulator_observations':True,
                'observation_contract':'Root pose/velocity tracking error, body position/velocity, future reference, feet world error and previous action.',
                'sensor_sampling_contract':self.sensor_contract,
                'checkpoint_sonic_protocol':self.checkpoint_protocol,
                'control_bound_mode':self.ref.control_bound_mode,'control_bound_contract':self.ref.control_bound_contract,
                'checkpoint_episode_reset_rule':self.checkpoint_reset_rule,
                'checkpoint_physical_quality_training':self.checkpoint_physical_quality_training,'checkpoint_physical_quality_training_contract':self.checkpoint_physical_quality_training_contract,
                'active_root_writes':False,'external_wrenches':False,'body_IK_override':False}
        if self.base is not None:
            report['controller']='Frozen SONIC all28-body outputs plus trained PPO residual, auxiliary reference PD'
            report['frozen_sonic']=self.base.c.report();report['sonic_batch_inference_rows']=self.base.inference_count
            report['previous_sonic_action']='Normalized effective all28 hybrid body targets; no IK overwrite'
        return report
