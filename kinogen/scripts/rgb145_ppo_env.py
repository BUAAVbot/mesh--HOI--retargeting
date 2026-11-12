"""GPU motion-imitation environment, with privileged simulation tracking state.

Actions are learned residuals on 28 real body position actuators. The free root
is never controlled directly. State writes occur only on declared episode reset.
This trains a new task policy; it is not the supplied SONIC checkpoint.
"""
from pathlib import Path
import numpy as np
import torch
from torch import nn

POLICY_SENSOR_CONTRACT='native_mj_step_kinematics_v1: qpos/qvel after integration; xpos/xmat/cvel before final integration; current kinematics only on episode reset'


def body_control_bounds(joint_ranges,actuator_ctrllimited,actuator_ctrlrange,names,wrist_targets=None,mode='legacy_joint'):
    """Bounds for PD commands, separately from physical qpos and effort limits."""
    if mode not in ('legacy_joint','actuator_ctrl'):raise ValueError('Unsupported control_bound_mode: '+str(mode))
    ranges=np.asarray(joint_ranges,dtype=float);limited=np.asarray(actuator_ctrllimited,dtype=bool);ctrlrange=np.asarray(actuator_ctrlrange,dtype=float)
    n=len(names)
    if ranges.shape!=(n,2) or limited.shape!=(n,) or ctrlrange.shape!=(n,2):raise ValueError('Invalid command-bound array dimensions')
    if mode=='legacy_joint':
        lower=np.asarray(ranges[:,0]+.01,dtype=np.float32);upper=np.asarray(ranges[:,1]-.01,dtype=np.float32)
    else:
        lower=np.where(limited,ctrlrange[:,0],-np.inf).astype(np.float32);upper=np.where(limited,ctrlrange[:,1],np.inf).astype(np.float32)
    wrist_windows={}
    for name,goal in (wrist_targets or {}).items():
        if name not in names:continue
        if not np.isfinite(goal):raise ValueError('Nonfinite wrist command goal')
        k=names.index(name);lo=max(float(lower[k]),float(goal)-.20);hi=min(float(upper[k]),float(goal)+.20)
        lower[k]=lo;upper[k]=hi;wrist_windows[name]=[lo,hi]
    if np.isnan(lower).any() or np.isnan(upper).any() or np.any(lower>upper):raise ValueError('Empty or invalid PD command bound interval')
    return lower,upper,wrist_windows


def reference_reset_rule(source_times,execution_times,source_window=None,start_probability=.30):
    """Map an optional source-clock reset window without changing task duration."""
    source=np.asarray(source_times,dtype=float);execution=np.asarray(execution_times,dtype=float)
    if source.ndim!=1 or len(source)<2 or execution.shape!=source.shape or not np.isfinite(source).all() or not np.isfinite(execution).all() or np.any(np.diff(source)<=0) or np.any(np.diff(execution)<=0):
        raise ValueError('Reset mapping requires finite increasing source/execution clocks of equal length')
    if source[0]!=0. or execution[0]!=0.:raise ValueError('From-start resets require both reference clocks to start at zero')
    if not np.isfinite(start_probability) or not 0.<=start_probability<=1.:raise ValueError('start-probability must be between zero and one')
    window=None
    if source_window is None:bounds=[0.,float(execution[-1])]
    else:
        window=np.asarray(source_window,dtype=float)
        if window.shape!=(2,) or not np.isfinite(window).all() or not 0.<=window[0]<window[1]<=source[-1]:
            raise ValueError('reset-source-window requires 0 <= START < END <= last source timestamp')
        bounds=np.interp(window,source,execution).tolist();window=window.tolist()
    return {'schema':'rgb145.episode_reset_rule.v1','random_reference_start_source_window_s':window,
            'random_reference_start_execution_window_s':bounds,'random_reference_start_distribution':'uniform execution time within the declared window',
            'source_to_execution_mapping':'piecewise linear interpolation of the unchanged reference clock arrays',
            'from_start_probability':float(start_probability),'from_start_execution_time_s':0.,
            'full_reference_last_execution_timestamp_s':float(execution[-1]),'episode_timeout':'unchanged full reference horizon including final display interval',
            'from_start_success_rule':'Only episodes reset at zero that reach the full reference horizon without failure count as from-start successes',
            'state_writes':'Episode reset only; no runtime root correction'}


def quat_mul(a,b):
    w=a[...,:1]*b[...,:1]-(a[...,1:]*b[...,1:]).sum(-1,keepdim=True)
    xyz=a[...,:1]*b[...,1:]+b[...,:1]*a[...,1:]+torch.cross(a[...,1:],b[...,1:],dim=-1)
    return torch.cat([w,xyz],-1)


def rotate_inverse(q,v):
    xyz=-q[...,1:];t=2*torch.cross(xyz,v,dim=-1)
    return v+q[...,:1]*t+torch.cross(xyz,t,dim=-1)


def physical_joint_violation(q,qa,lower,upper,limited):
    values=q[:,qa]
    violation=torch.maximum(lower-values,values-upper).clamp_min(0.)
    return torch.where(limited,violation,torch.zeros_like(violation)).amax(dim=1)


def build_contact_quality_kernel():
    # Lazy module import keeps ordinary Reference/ActorCritic users independent
    # of Warp. The kernel reads the actual installed Contact.dist/worldid ABI.
    global wp
    import warp as wp
    @wp.kernel
    def collect(dist:wp.array(dtype=wp.float32),worldid:wp.array(dtype=wp.int32),count:wp.array(dtype=wp.int32),peak:wp.array(dtype=wp.float32),invalid:wp.array(dtype=wp.int32),bad_mapping:wp.array(dtype=wp.int32),nworld:int):
        i=wp.tid()
        if i==0 and count[0]>dist.shape[0]:wp.atomic_max(bad_mapping,0,1)
        if i<count[0]:
            world=worldid[i]
            if world<0 or world>=nworld:wp.atomic_max(bad_mapping,0,1)
            else:
                if not wp.isfinite(dist[i]):wp.atomic_max(invalid,world,1)
                else:wp.atomic_max(peak,world,wp.max(-dist[i],0.0))
    return collect


class Reference:
    def __init__(self,model,path,device,control_bound_mode='legacy_joint'):
        import mujoco as mj
        self.z=np.load(path,allow_pickle=False);self.device=device
        def tensor(v,dtype=torch.float32):return torch.as_tensor(v,device=device,dtype=dtype)
        q=self.z['qpos'].copy()
        for k in range(1,len(q)):
            if np.dot(q[k,3:7],q[k-1,3:7])<0:q[k,3:7]*=-1
        times=self.z['time_s'];self.end=float(times[-1]);self.horizon=self.end+float(np.median(np.diff(times)))
        self.times=tensor(times);self.source_times=tensor(self.z['source_time_s']);self.q=tensor(q)
        names=list(map(str,self.z['dof_names']));self.names=names
        self.body_cols=np.array([k for k,n in enumerate(names)if not n.startswith(('lh_','rh_','neck_'))])
        self.body_names=[names[k]for k in self.body_cols]
        self.jids=np.array([mj.mj_name2id(model,mj.mjtObj.mjOBJ_JOINT,n)for n in self.body_names]);self.qa=model.jnt_qposadr[self.jids];self.va=model.jnt_dofadr[self.jids]
        self.body_aids=np.array([mj.mj_name2id(model,mj.mjtObj.mjOBJ_ACTUATOR,'motor_'+n)for n in self.body_names])
        self.all_aids=np.array([mj.mj_name2id(model,mj.mjtObj.mjOBJ_ACTUATOR,'motor_'+n)for n in names])
        self.all_jids=np.array([mj.mj_name2id(model,mj.mjtObj.mjOBJ_JOINT,n)for n in names])
        if len(names)!=52 or np.any(self.all_jids<0) or np.any(model.jnt_type[self.all_jids]!=int(mj.mjtJoint.mjJNT_HINGE)):raise ValueError('Expected all52 real hinge coordinates for physical quality checks')
        self.all_qa_t=tensor(model.jnt_qposadr[self.all_jids],torch.long);self.physical_joint_lower=tensor(model.jnt_range[self.all_jids,0]);self.physical_joint_upper=tensor(model.jnt_range[self.all_jids,1]);self.physical_joint_limited=tensor(model.jnt_limited[self.all_jids].astype(bool),torch.bool)
        self.passive=np.flatnonzero(np.all(model.actuator_gear==0,axis=1))
        if len(self.body_names)!=28 or len(self.passive)!=10:raise ValueError('Expected 28-body and10 passive ABI channels')
        self.qa_t=tensor(self.qa,torch.long);self.va_t=tensor(self.va,torch.long);self.body_aids_t=tensor(self.body_aids,torch.long);self.all_aids_t=tensor(self.all_aids,torch.long);self.passive_t=tensor(self.passive,torch.long)
        wrist_targets={}
        if 'wrist_reference_joint_names'in self.z:
            wrist_targets={str(name):float(goal)for name,goal in zip(self.z['wrist_reference_joint_names'],self.z['wrist_reference_joint_targets_rad'])}
        lo,hi,self.software_wrist_command_windows=body_control_bounds(model.jnt_range[self.jids],model.actuator_ctrllimited[self.body_aids],model.actuator_ctrlrange[self.body_aids],self.body_names,wrist_targets,control_bound_mode)
        self.lower=tensor(lo);self.upper=tensor(hi);self.control_bound_mode=control_bound_mode
        self.control_bound_contract={'mode':control_bound_mode,'body_joint_names':self.body_names,'lower_command_rad':[float(x)if np.isfinite(x)else None for x in lo],'upper_command_rad':[float(x)if np.isfinite(x)else None for x in hi],'null_bound_means':'unbounded PD command','bound_source':'physical actuator ctrllimited/ctrlrange plus relaxed wrist window'if control_bound_mode=='actuator_ctrl'else'legacy physical joint range plus/minus0.01rad plus relaxed wrist window','physical_joint_limits_modified':False,'actuator_force_limits_modified':False,'software_wrist_command_windows':self.software_wrist_command_windows}
        self.feet=np.array([mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,n+'_ankle_roll_link')for n in ('left','right')]);self.feet_t=tensor(self.feet,torch.long)
        data=mj.MjData(model);positions=[];vel=[]
        for k in range(len(q)):
            data.qpos[:]=q[k];mj.mj_fwdPosition(model,data);positions.append(data.xpos[self.feet].copy())
            vv=np.zeros(model.nv)
            if k<len(q)-1:mj.mj_differentiatePos(model,vv,times[k+1]-times[k],q[k],q[k+1])
            vel.append(vv)
        self.feet_ref=tensor(np.asarray(positions));self.dq=tensor(np.asarray(vel))

    def indices(self,t):
        t=t.clamp(float(self.times[0]),self.end);j=torch.searchsorted(self.times,t,right=True).clamp(1,len(self.times)-1);i=j-1
        w=((t-self.times[i])/(self.times[j]-self.times[i])).clamp(0,1)
        return i,j,w

    def sample(self,t):
        i,j,w=self.indices(t);q=self.q[i]+w[:,None]*(self.q[j]-self.q[i]);q[:,3:7]/=q[:,3:7].norm(dim=-1,keepdim=True).clamp_min(1e-8)
        dq=self.dq[i]+w[:,None]*(self.dq[j]-self.dq[i]);dq=torch.where((t>self.end)[:,None],torch.zeros_like(dq),dq)
        feet=self.feet_ref[i]+w[:,None,None]*(self.feet_ref[j]-self.feet_ref[i])
        return q,dq,feet

    def source_clock(self,t):
        i,j,w=self.indices(t);return self.source_times[i]+w*(self.source_times[j]-self.source_times[i])

    def observations(self,q,v,xpos,phase,previous):
        target,dq,feet=self.sample(phase);future,_,_=self.sample(phase+.2)
        actual=q[:,3:7]/q[:,3:7].norm(dim=-1,keepdim=True).clamp_min(1e-8)
        conjugate=torch.cat([actual[:,:1],-actual[:,1:]],-1);rotation=quat_mul(conjugate,target[:,3:7]);rotation=torch.where(rotation[:,:1]<0,-rotation,rotation)
        foot_error=(feet-xpos[:,self.feet_t]).reshape(-1,6)
        return torch.cat([rotate_inverse(actual,target[:,:3]-q[:,:3]),rotation,
            q[:,self.qa_t]-target[:,self.qa_t],v[:,self.va_t]*.1,
            rotate_inverse(actual,v[:,:3])*.25,v[:,3:6]*.25,
            rotate_inverse(actual,dq[:,:3])*.25,dq[:,self.va_t]*.1,
            future[:,self.qa_t]-q[:,self.qa_t],previous,foot_error,
            (phase/self.horizon)[:,None]],dim=-1).clamp(-10.,10.)

    def controls(self,phase,actions,scale=.45):
        target,_,_=self.sample(phase);u=torch.zeros((len(phase),len(self.names)),device=self.device)
        u[:,self.all_aids_t]=target[:,7:]
        u[:,self.body_aids_t]=(target[:,self.qa_t]+scale*torch.tanh(actions)).clamp(self.lower,self.upper)
        u[:,self.passive_t]=0.
        return u


class ActorCritic(nn.Module):
    def __init__(self,obs_dim,action_dim=28):
        super().__init__()
        def net(out):return nn.Sequential(nn.Linear(obs_dim,256),nn.ELU(),nn.Linear(256,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,out))
        self.actor=net(action_dim);self.critic=net(1);self.log_std=nn.Parameter(torch.full((action_dim,),-1.6))
        for layer in self.modules():
            if isinstance(layer,nn.Linear):nn.init.orthogonal_(layer.weight,2**.5);nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor[-1].weight,.01);nn.init.orthogonal_(self.critic[-1].weight,1.)
        self.register_buffer('obs_mean',torch.zeros(obs_dim));self.register_buffer('obs_var',torch.ones(obs_dim));self.register_buffer('obs_count',torch.tensor(1e-4))
    def normalize(self,x):return ((x-self.obs_mean)/(self.obs_var+1e-6).sqrt()).clamp(-10,10)
    @torch.no_grad()
    def update_stats(self,x):
        var,mean=torch.var_mean(x,dim=0,unbiased=False);count=x.shape[0];delta=mean-self.obs_mean;total=self.obs_count+count
        self.obs_var.copy_((self.obs_var*self.obs_count+var*count+delta.square()*self.obs_count*count/total)/total)
        self.obs_mean.add_(delta*count/total);self.obs_count.copy_(total)
    def distribution(self,x):return torch.distributions.Normal(self.actor(x),self.log_std.clamp(-3.5,-.3).exp())


class WarpTracking:
    dt=.02
    def __init__(self,model_path,reference_path,n=512,seed=1,action_scale=.45,sonic_base=False,reset_source_window=None,start_probability=.30,control_bound_mode='legacy_joint',physical_quality_training=False):
        import mujoco as mj,mujoco_warp as mw,warp as wp
        self.mj,self.mw,self.wp=mj,mw,wp;self.device='cuda:0';self.n=n;self.action_scale=action_scale
        torch.manual_seed(seed);wp.init();wp.set_device(self.device)
        self.model=mj.MjModel.from_xml_path(str(Path(model_path).resolve()));self.cpu_data=mj.MjData(self.model)
        self.ref=Reference(self.model,reference_path,self.device,control_bound_mode=control_bound_mode)
        self.reset_rule=reference_reset_rule(self.ref.z['source_time_s'],self.ref.z['time_s'],reset_source_window,start_probability)
        self.reset_execution_bounds=self.reset_rule['random_reference_start_execution_window_s']
        self.start_probability=float(start_probability)
        self.sonic=None
        if sonic_base:
            from rgb145_batch_sonic import BatchSonic
            self.sonic=BatchSonic(self.model,self.ref,n,self.device)
        self.cpu_data.qpos[:]=self.ref.q[0].cpu().numpy();mj.mj_forward(self.model,self.cpu_data)
        self.m=mw.put_model(self.model);self.d=mw.put_data(self.model,self.cpu_data,nworld=n,nconmax=128,njmax=512)
        self.q=wp.to_torch(self.d.qpos);self.v=wp.to_torch(self.d.qvel);self.u=wp.to_torch(self.d.ctrl)
        self.physical_quality_training=bool(physical_quality_training)
        self.physical_quality_training_contract={'enabled':self.physical_quality_training,'trained_quality_items':['actual_limited_hinge_joint_violation','actual_contact_penetration']if self.physical_quality_training else[],
            'joint_limit_violation_threshold_rad':.05,'joint_violation_reward_weight_per_rad':20.,'joint_sampling':'Actual post-integration qpos of all52 named hinges, with true model.jnt_limited mask, at50Hz policy steps',
            'penetration_threshold_m':.02,'penetration_reward_weight_per_m':50.,'contact_sampling':'Actual Warp Contact.dist/worldid over active nacon slots, per-world maximum across all4 real5ms physics substeps',
            'contact_buffer_overflow_or_invalid_mapping':'Terminates all worlds; invalid distances terminate the corresponding world','penalty_saturation':'Each violation penalty saturates at twice its failure threshold; nonfinite quality values are failures and receive the saturated penalty',
            'model_parameters_modified':False,'root_thresholds_modified':False,'deployment_QC_modified':False}
        self.quality_terminated_episodes=0;self.joint_limit_terminated_episodes=0;self.penetration_terminated_episodes=0;self.nonfinite_quality_episodes=0;self.invalid_contact_quality_episodes=0
        if self.physical_quality_training:
            self._contact_dist=wp.to_torch(self.d.contact.dist);self._contact_worldid=wp.to_torch(self.d.contact.worldid);self._contact_count=wp.to_torch(self.d.nacon)
            if self._contact_dist.ndim!=1 or self._contact_worldid.shape!=self._contact_dist.shape or tuple(self._contact_count.shape)!=(1,):raise ValueError('Unsupported actual Warp contact ABI for quality training')
            self._contact_quality_kernel=build_contact_quality_kernel();self._quality_peak_wp=wp.zeros(n,dtype=wp.float32,device=self.device);self._quality_invalid_wp=wp.zeros(n,dtype=wp.int32,device=self.device);self._quality_mapping_wp=wp.zeros(1,dtype=wp.int32,device=self.device)
            self._quality_peak=wp.to_torch(self._quality_peak_wp);self._quality_invalid=wp.to_torch(self._quality_invalid_wp);self._quality_mapping=wp.to_torch(self._quality_mapping_wp)
        self._physics_xpos=wp.to_torch(self.d.xpos);self._physics_rotation=wp.to_torch(self.d.xmat);self._physics_cvel=wp.to_torch(self.d.cvel)
        # Initialize reset observations in an isolated data batch. A forward
        # on the live batch would modify unrelated worlds' solver caches and
        # kinematic timing, even if qpos/qvel themselves stayed unchanged.
        free_before=torch.cuda.mem_get_info()[0]
        self.reset_d=mw.put_data(self.model,self.cpu_data,nworld=n,nconmax=128,njmax=512)
        self.reset_scratch_allocated_bytes=free_before-torch.cuda.mem_get_info()[0]
        self._reset_q=wp.to_torch(self.reset_d.qpos);self._reset_v=wp.to_torch(self.reset_d.qvel);self._reset_u=wp.to_torch(self.reset_d.ctrl)
        self._reset_xpos=wp.to_torch(self.reset_d.xpos);self._reset_rotation=wp.to_torch(self.reset_d.xmat);self._reset_cvel=wp.to_torch(self.reset_d.cvel)
        self.xpos=self._physics_xpos.clone();self.rotation=self._physics_rotation.clone();self.cvel=self._physics_cvel.clone()
        self.policy_sensor_contract=POLICY_SENSOR_CONTRACT
        self.phase=torch.zeros(n,device=self.device);self.previous=torch.zeros((n,28),device=self.device)
        self.from_start=torch.zeros(n,dtype=torch.bool,device=self.device);self.steps=torch.zeros(n,dtype=torch.long,device=self.device)
        self.start_successes=0;self.starts=0;self.max_from_start_source_s=0.;self.nan_episodes=0
        self.u.copy_(self.ref.controls(self.phase,self.previous,self.action_scale));mw.step(self.m,self.d);wp.synchronize()
        with wp.ScopedCapture()as capture:
            if self.physical_quality_training:self._quality_peak_wp.zero_();self._quality_invalid_wp.zero_();self._quality_mapping_wp.zero_()
            for _ in range(4):
                mw.step(self.m,self.d)
                if self.physical_quality_training:wp.launch(self._contact_quality_kernel,dim=self.d.contact.dist.shape[0],inputs=[self.d.contact.dist,self.d.contact.worldid,self.d.nacon,self._quality_peak_wp,self._quality_invalid_wp,self._quality_mapping_wp,n],device=self.device)
        self.graph=capture.graph
        self.reset(torch.arange(n,device=self.device))

    def refresh_policy_sensors(self,ids=None):
        """Snapshot native post-step sampling without writing old physics state."""
        for cached,physical in ((self.xpos,self._physics_xpos),(self.rotation,self._physics_rotation),(self.cvel,self._physics_cvel)):
            if ids is None:cached.copy_(physical)
            else:cached[ids]=physical[ids]

    def reset(self,ids,force_start=False):
        if not len(ids):return
        lower,upper=self.reset_execution_bounds
        phase=lower+torch.rand(len(ids),device=self.device)*(upper-lower)
        start=torch.rand(len(ids),device=self.device)<getattr(self,'start_probability',.30)
        if force_start:start[:]=True
        phase[start]=0.
        self.phase[ids]=phase;self.from_start[ids]=start;self.starts+=int(start.sum());self.steps[ids]=0
        mask=torch.zeros(self.n,dtype=torch.bool,device=self.device);mask[ids]=True
        self.mw.reset_data(self.m,self.d,self.wp.from_torch(mask,dtype=self.wp.bool))
        target,velocity,_=self.ref.sample(phase);self.q[ids]=target;self.v[ids]=velocity
        self.previous[ids]=0.;self.u[ids]=self.ref.controls(phase,self.previous[ids],self.action_scale)
        if self.sonic is not None:self.sonic.reset(ids)
        self.mw.reset_data(self.m,self.reset_d,self.wp.from_torch(mask,dtype=self.wp.bool))
        self._reset_q[ids]=target;self._reset_v[ids]=velocity;self._reset_u[ids]=self.u[ids]
        self.mw.forward(self.m,self.reset_d)
        # Only initialized policy observations are scattered back. Live
        # qacc_warmstart keeps the value established by masked reset_data;
        # contacts and other live derived arrays are rebuilt by the next step.
        self.xpos[ids]=self._reset_xpos[ids];self.rotation[ids]=self._reset_rotation[ids];self.cvel[ids]=self._reset_cvel[ids]
        if getattr(self,'record_reset_diagnostics',False):
            self.last_reset_solver_cache_diagnostics={'strategy':'isolated_reset_scratch_forward','live_batch_forward_calls':0,'live_warmstart_copied_from_scratch':False,'reset_scratch_allocated_bytes':self.reset_scratch_allocated_bytes}

    def base_controls(self):
        nominal=self.ref.controls(self.phase,torch.zeros_like(self.previous),self.action_scale)
        if self.sonic is not None:
            base=self.sonic.c.base;target=self.sonic.targets(self.q,self.v,self.phase,self.rotation[:,base],self.cvel[:,base,:3]);nominal[:,self.sonic.aids]=target
        return nominal

    def observations(self):
        obs=self.ref.observations(self.q,self.v,self.xpos,self.phase,self.previous)
        if self.sonic is not None:
            baseline=self.base_controls();target,_,_=self.ref.sample(self.phase)
            obs=torch.cat([obs,baseline[:,self.ref.body_aids_t]-target[:,self.ref.qa_t]],dim=-1)
        return torch.nan_to_num(obs)

    def step(self,actions):
        if self.sonic is None:self.u.copy_(self.ref.controls(self.phase,actions,self.action_scale))
        else:
            packet=self.base_controls();ids=self.ref.body_aids_t
            packet[:,ids]=(packet[:,ids]+self.action_scale*torch.tanh(actions)).clamp(self.ref.lower,self.ref.upper)
            self.u.copy_(packet);self.sonic.record_applied(self.u)
        self.wp.capture_launch(self.graph)
        self.refresh_policy_sensors()
        self.phase+=self.dt;self.steps+=1
        target,velocity,feet=self.ref.sample(self.phase)
        pos=self.q[:,:3]-target[:,:3];ori=2*torch.acos((self.q[:,3:7]*target[:,3:7]).sum(-1).abs().clamp(0,1))
        pose=(self.q[:,self.ref.qa_t]-target[:,self.ref.qa_t]).square().mean(-1)
        feeterr=(self.xpos[:,self.ref.feet_t]-feet).square().mean((1,2))
        vel=(self.v[:,:3]-velocity[:,:3]).square().mean(-1)
        change=(torch.tanh(actions)-torch.tanh(self.previous)).square().mean(-1)
        reward=3.*torch.exp(-12*pos[:,:2].square().sum(-1))+torch.exp(-40*pos[:,2].square())+torch.exp(-5*ori.square())+torch.exp(-5*pose)+torch.exp(-25*feeterr)+.5*torch.exp(-2*vel)-.03*change-.003*actions.square().mean(-1)
        finite=torch.isfinite(self.q).all(-1)&torch.isfinite(self.v).all(-1)
        xy_limit,z_limit,ori_limit=(.35,.20,.6)if getattr(self,'strict_training',False)else(.65,.30,1.2)
        failed=(pos[:,:2].norm(dim=-1)>xy_limit)|(pos[:,2].abs()>z_limit)|(ori>ori_limit)|~finite
        quality_info={}
        if self.physical_quality_training:
            violation=physical_joint_violation(self.q,self.ref.all_qa_t,self.ref.physical_joint_lower,self.ref.physical_joint_upper,self.ref.physical_joint_limited)
            penetration=self._quality_peak.clone();nonfinite_contact=self._quality_invalid>0;invalid=nonfinite_contact|(self._quality_mapping[0]>0)
            joint_failed=(violation>.05)|~torch.isfinite(violation);penetration_failed=(penetration>.02)|~torch.isfinite(penetration)|invalid
            quality_failed=joint_failed|penetration_failed;failed|=quality_failed
            penetration_penalty=torch.where(invalid,torch.full_like(penetration,.04),penetration)
            reward-=20.*torch.nan_to_num(violation,nan=.10,posinf=.10,neginf=.10).clamp(0.,.10)+50.*torch.nan_to_num(penetration_penalty,nan=.04,posinf=.04,neginf=.04).clamp(0.,.04)
            self.quality_terminated_episodes+=int(quality_failed.sum());self.joint_limit_terminated_episodes+=int(joint_failed.sum());self.penetration_terminated_episodes+=int(penetration_failed.sum());self.nonfinite_quality_episodes+=int((~torch.isfinite(violation)|~torch.isfinite(penetration)|nonfinite_contact).sum());self.invalid_contact_quality_episodes+=int(invalid.sum())
            quality_info={'physical_joint_violation_rad':violation,'physical_contact_penetration_m':penetration,'physical_quality_failed':quality_failed,'contact_quality_invalid':invalid}
        timeout=self.phase>=self.ref.horizon;done=failed|timeout
        reward=torch.nan_to_num(reward,nan=-2.)-2.*failed.float()
        self.previous.copy_(actions)
        if self.from_start.any():self.max_from_start_source_s=max(self.max_from_start_source_s,float(self.ref.source_clock(self.phase[self.from_start]).max()))
        self.start_successes+=int((timeout&~failed&self.from_start).sum());self.nan_episodes+=int((~finite).sum())
        info={'failed':failed,'timeout':timeout,'root_xy':pos[:,:2].norm(dim=-1),'root_angle':ori,'pose_mse':pose,'feet_mse':feeterr}
        info.update(quality_info)
        return self.observations(),reward*.1,done,info
