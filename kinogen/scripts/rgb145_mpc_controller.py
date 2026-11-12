"""Frozen SONIC + CEM leg/waist MPC using counterfactual MuJoCo-Warp rollouts.

Only actuator targets are returned to the live native simulator. All state
resets below belong to an isolated counterfactual planning batch. This is a
new hybrid controller, not an unmodified SONIC reproduction or a trained PPO.
"""
from pathlib import Path
from collections import deque
import argparse,hashlib,json,time
import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def residual_at_offset(knots,offset_s,duration_s):
    """Linearly sample [..., knot, joint] tensors, holding both end values.

    Knot times span the first through final control step of the horizon.
    This helper only reads tensors and never touches a simulator state.
    """
    count=knots.shape[-2]
    if count<1 or not np.isfinite(offset_s)or not np.isfinite(duration_s)or duration_s<=0:raise ValueError('Invalid residual interpolation dimensions/time')
    if count==1:return knots[...,0,:]
    position=float(np.clip(offset_s/duration_s,0.,1.))*(count-1)
    left=min(int(position),count-2);weight=position-left
    return knots[...,left,:]*(1-weight)+knots[...,left+1,:]*weight


class MPCController:
    def __init__(self,model,reference_path,model_path,device='cuda:0',config=None):
        import torch
        from rgb145_ppo_env import Reference,WarpTracking
        from rgb145_batch_sonic import BatchSonic
        if device!='cuda:0':raise ValueError('Set CUDA_VISIBLE_DEVICES to the planning GPU and use logical cuda:0')
        self.torch=torch;self.model=model;self.device=device;self.config=dict(config or {})
        self.reference_path=Path(reference_path);self.model_path=Path(model_path)
        torch.set_num_threads(2)
        self.control_bound_mode=self.config.get('control_bound_mode','legacy_joint')
        self.ref=Reference(model,reference_path,device,control_bound_mode=self.control_bound_mode);self.live=BatchSonic(model,self.ref,1,device);self.base=self.live
        self.n=int(self.config.get('candidates',64));self.elite=int(self.config.get('elites',8))
        self.horizon=int(self.config.get('horizon_steps',20));self.rounds=int(self.config.get('cem_rounds',2))
        self.residual_knots=int(self.config.get('residual_knots',1))
        self.policy_dt=.02;self.physics_dt=float(model.opt.timestep);self.replan_dt=float(self.config.get('replan_dt',.1))
        self.initial_std=float(self.config.get('initial_std_rad',.20));self.minimum_std=float(self.config.get('minimum_std_rad',.025));self.maximum_residual=float(self.config.get('maximum_residual_rad',1.6))
        if self.n<8 or self.n%2 or not(2<=self.elite<self.n)or self.horizon<1 or self.rounds<1:raise ValueError('Invalid CEM dimensions')
        if self.residual_knots<1 or(self.residual_knots>1 and self.horizon<2):raise ValueError('Sequence residuals need positive knots and at least two control steps')
        self.residual_span=max(self.horizon-1,1)*self.policy_dt
        if abs(self.physics_dt-.005)>1e-12:raise ValueError('MPC expects the native5ms/50Hz plant')
        self.env=WarpTracking(model_path,reference_path,self.n,145317,1.,True,control_bound_mode=self.control_bound_mode)
        self.wrist_window_override=self.config.get('wrist_ctrl_window_rad')
        if self.wrist_window_override is not None:
            window=float(self.wrist_window_override)
            if not np.isfinite(window)or window<0:raise ValueError('Invalid MPC wrist command window')
            # These are two private algorithm command-bound tensors. The real
            # model's actuator/joint/effort limits remain untouched.
            for ref in(self.ref,self.env.ref):
                if 'wrist_reference_joint_names'not in ref.z:raise ValueError('A wrist override requires named wrist targets')
                for name,goal in zip(ref.z['wrist_reference_joint_names'],ref.z['wrist_reference_joint_targets_rad']):
                    name=str(name);goal=float(goal);column=ref.body_names.index(name)
                    ref.lower[column]=torch.maximum(ref.lower[column],torch.tensor(goal-window,device=device))
                    ref.upper[column]=torch.minimum(ref.upper[column],torch.tensor(goal+window,device=device))
                    if not bool(ref.lower[column]<=ref.upper[column]):raise ValueError('Empty MPC wrist command window')
                    ref.software_wrist_command_windows[name]=[float(ref.lower[column]),float(ref.upper[column])]
                ref.control_bound_contract.update(software_wrist_window_override_rad=window,software_wrist_command_windows=ref.software_wrist_command_windows,
                    lower_command_rad=[float(x)if np.isfinite(x)else None for x in ref.lower.cpu().numpy()],upper_command_rad=[float(x)if np.isfinite(x)else None for x in ref.upper.cpu().numpy()])
        self.names=list(self.live.c.names);self.aids=self.live.c.aids.copy();self.qa=self.live.c.qa.copy();self.jref=self.live.c.jref.copy()
        self.leg_columns=np.asarray([k for k,n in enumerate(self.names)if any(s in n for s in ('hip_','knee_','ankle_'))])
        if len(self.names)!=28 or len(self.leg_columns)!=12:raise ValueError('Expected all28 SONIC body outputs and12 leg joints')
        self.include_waist=bool(self.config.get('include_waist',True))
        self.waist_columns=np.asarray([k for k,n in enumerate(self.names)if n in ('waist_yaw_joint','waist_pitch_joint')],dtype=int)
        if self.include_waist and len(self.waist_columns)!=2:raise ValueError('Expected waist yaw/pitch channels')
        self.residual_columns=np.r_[self.leg_columns,self.waist_columns]if self.include_waist else self.leg_columns.copy()
        self.residual_dim=len(self.residual_columns)
        self.residual_t=torch.as_tensor(self.residual_columns,device=device,dtype=torch.long)
        order=torch.as_tensor([self.ref.body_names.index(n)for n in self.names],device=device,dtype=torch.long)
        self.lower=self.ref.lower[order];self.upper=self.ref.upper[order]
        self.body_qa_t=torch.as_tensor(self.qa,device=device,dtype=torch.long)
        self.rng=torch.Generator(device=device);self.rng.manual_seed(int(self.config.get('seed',145317)))
        self.arm_feedback=None;self.target_history=deque(maxlen=1)
        self.last=np.zeros(28);self.target=np.zeros(28);self.last_base_target=np.zeros(28);self.ctrl=np.zeros(model.nu)
        self.residual=torch.zeros(self.residual_dim,device=device);self.inference_count=0;self.next_time=0.;self.next_plan_time=0.
        self.residual_sequence=torch.zeros((self.residual_knots,self.residual_dim),device=device)
        self.effective_residual=torch.zeros_like(self.residual);self.plan_start_time=0.
        self.teacher_std=float(self.config.get('teacher_std_rad',.25));self.global_std=float(self.config.get('global_std_rad',.50))
        self.recovery_threshold=float(self.config.get('recovery_cost_threshold',25.));self.recovery_mode=False
        self.progress_every=max(1,int(self.config.get('progress_every_plans',10)))
        self.plan_count=0;self.planning_control_steps=0;self.planning_inference_rows=0;self.counterfactual_initializations=0
        self.diagnostics=[];self.plant_checked=False
        self.weights={'root_xy':30.,'root_z':50.,'orientation':10.,'body_pose':2.,'feet':15.,'root_velocity':.5,'residual':.05,'residual_change':.10,'failed':100.,'terminal':3.,'com_xy':0.,'joint_limit':0.}
        self.weights.update(self.config.get('cost_weights',{}))
        self.reject_joint_limit=bool(self.config.get('reject_joint_limit_violation',False))
        self.joint_limit_margin=float(self.config.get('joint_limit_margin_rad',.01))
        self.limited_qadr=None
        if not np.isfinite(self.weights['joint_limit'])or self.weights['joint_limit']<0 or not np.isfinite(self.joint_limit_margin)or self.joint_limit_margin<0:raise ValueError('Invalid joint-limit objective weight/margin')
        if self.reject_joint_limit or self.weights['joint_limit']>0:
            import mujoco as mj
            limited=np.flatnonzero(model.jnt_limited&(model.jnt_type==int(mj.mjtJoint.mjJNT_HINGE)))
            if not len(limited):raise ValueError('Joint-limit cost requires actual limited hinge joints')
            self.limited_qadr=torch.as_tensor(model.jnt_qposadr[limited].copy(),device=device,dtype=torch.long)
            self.joint_lower=torch.as_tensor(model.jnt_range[limited,0].copy(),device=device,dtype=torch.float32)
            self.joint_upper=torch.as_tensor(model.jnt_range[limited,1].copy(),device=device,dtype=torch.float32)
            self.limited_joint_names=[mj.mj_id2name(model,mj.mjtObj.mjOBJ_JOINT,int(j))for j in limited]
        self.com_reference=None;self.com_sensor=None
        if not np.isfinite(self.weights['com_xy'])or self.weights['com_xy']<0:raise ValueError('COM cost must be finite and nonnegative')
        if self.weights['com_xy']>0:
            if 'reference_com_world'not in self.ref.z:raise ValueError('COM cost requires reference_com_world in the reference; no fallback COM is invented')
            raw=np.asarray(self.ref.z['reference_com_world'],float)
            if raw.shape!=(len(self.ref.times),3)or not np.isfinite(raw).all():raise ValueError('reference_com_world must be a finite T x 3 array matching the reference clock')
            self.com_reference=torch.as_tensor(raw,device=device,dtype=torch.float32)
            if not hasattr(self.env.d,'subtree_com'):raise ValueError('Planning MuJoCo-Warp data has no subtree_com field')
            self.com_sensor=self.env.wp.to_torch(self.env.d.subtree_com)
            if tuple(self.com_sensor.shape)!=(self.n,self.model.nbody,3):raise ValueError('Unexpected planning subtree_com shape: '+str(tuple(self.com_sensor.shape)))

    def _check_plant(self):
        if self.ref.control_bound_mode!=self.env.ref.control_bound_mode or not self.torch.equal(self.ref.lower,self.env.ref.lower) or not self.torch.equal(self.ref.upper,self.env.ref.upper):raise ValueError('Live/planning PD command bounds differ')
        fields=('body_mass','body_inertia','body_ipos','body_iquat','body_pos','body_quat','jnt_range','jnt_limited','dof_armature','dof_damping','dof_frictionloss','actuator_gainprm','actuator_biasprm','actuator_gear','actuator_forcerange','actuator_forcelimited','eq_type','eq_obj1id','eq_obj2id','eq_data','geom_type','geom_size','geom_pos','geom_quat','geom_friction','geom_contype','geom_conaffinity','geom_solref','geom_solimp')
        mismatched=[name for name in fields if not np.array_equal(getattr(self.model,name),getattr(self.env.model,name))]
        for name in ('timestep','gravity','integrator','solver','cone','iterations','tolerance'):
            if not np.array_equal(getattr(self.model.opt,name),getattr(self.env.model.opt,name)):mismatched.append('opt.'+name)
        if mismatched:raise ValueError('Live/planning plant differs: '+','.join(mismatched)+'. Use the effective frozen model and avoid adding finger rotor twice.')
        self.plant_checked=True

    def reset(self,data):
        self.last[:]=0.;self.target[:]=0.;self.ctrl[:]=0.;self.last_base_target[:]=0.
        self.residual.zero_();self.inference_count=0;self.next_time=0.;self.next_plan_time=0.;self.target_history.clear()
        self.residual_sequence.zero_();self.effective_residual.zero_();self.plan_start_time=0.
        self.live.reset(self.torch.tensor([0],device=self.device))
        self.plan_count=0;self.planning_control_steps=0;self.planning_inference_rows=0;self.counterfactual_initializations=0;self.diagnostics=[]
        self.recovery_mode=False

    def _packet(self,phase,base,residual):
        torch=self.torch
        packet=self.ref.controls(phase,torch.zeros((len(phase),28),device=self.device),0.)
        body=base.clone();body[:,self.residual_t]+=residual
        packet[:,self.live.aids]=torch.maximum(self.lower,torch.minimum(self.upper,body))
        packet[:,self.ref.passive_t]=0.
        return packet

    def _restore_counterfactuals(self,data,t):
        """Restore all planning worlds and every SONIC branch-history tensor."""
        e=self.env;torch=self.torch;wp=e.wp
        mask=torch.ones(self.n,device=self.device,dtype=torch.bool)
        e.mw.reset_data(e.m,e.d,wp.from_torch(mask,dtype=wp.bool))
        e.q.copy_(torch.as_tensor(data.qpos.copy(),device=self.device,dtype=e.q.dtype)[None].expand_as(e.q))
        e.v.copy_(torch.as_tensor(data.qvel.copy(),device=self.device,dtype=e.v.dtype)[None].expand_as(e.v))
        e.u.copy_(torch.as_tensor(data.ctrl.copy(),device=self.device,dtype=e.u.dtype)[None].expand_as(e.u))
        wp.to_torch(e.d.time).fill_(float(data.time))
        warm=wp.to_torch(e.d.qacc_warmstart)
        warm.copy_(torch.as_tensor(data.qacc_warmstart.copy(),device=self.device,dtype=warm.dtype)[None].expand_as(warm))
        active=wp.to_torch(e.d.eq_active)
        active.copy_(torch.as_tensor(data.eq_active.copy(),device=self.device,dtype=active.dtype)[None].expand_as(active))
        # At the live policy boundary, native kinematic sensors still describe
        # the preceding integration start. Do not replace them by new forward.
        for name,value in (('xpos',data.xpos),('rotation',data.xmat.reshape(self.model.nbody,3,3)),('cvel',data.cvel)):
            dest=getattr(e,name);dest.copy_(torch.as_tensor(value.copy(),device=self.device,dtype=dest.dtype)[None].expand_as(dest))
        e.phase.fill_(float(t));e.previous.zero_();e.steps.zero_();e.from_start.zero_()
        src=self.live;dst=e.sonic
        for name in ('valid','phase','last','previous_imu','previous_gravity','cached'):
            value=getattr(src,name);getattr(dst,name).copy_(value.expand_as(getattr(dst,name)))
        for name,value in src.history.items():dst.history[name].copy_(value.expand_as(dst.history[name]))
        for name,value in src.tokens.items():dst.tokens[name].copy_(value.expand_as(dst.tokens[name]))
        dst.inference_count=0
        self.counterfactual_initializations+=self.n

    def _rollout_cost(self,candidates,data,t):
        e=self.env;torch=self.torch;w=self.weights
        self._restore_counterfactuals(data,t)
        cost=torch.zeros(self.n,device=self.device);dead=torch.zeros(self.n,device=self.device,dtype=torch.bool)
        if self.residual_knots==1:
            # Keep the original constant-residual arithmetic and random stream
            # exactly, including its legacy per-horizon-step change penalty.
            action_cost=w['residual']*candidates.square().mean(1)+w['residual_change']*(candidates-self.residual).square().mean(1)
        else:previous_effective=self.effective_residual[None].expand(self.n,-1)
        last_cost=torch.zeros_like(cost)
        for h in range(self.horizon):
            b=e.sonic.c.base
            base=e.sonic.targets(e.q,e.v,e.phase,e.rotation[:,b],e.cvel[:,b,:3])
            step_residual=candidates if self.residual_knots==1 else residual_at_offset(candidates,h*self.policy_dt,self.residual_span)
            packet=self._packet(e.phase,base,step_residual);e.u.copy_(packet);e.sonic.record_applied(packet)
            if self.residual_knots>1:
                effective=(packet[:,self.live.aids]-base)[:,self.residual_t]
                action_cost=w['residual']*effective.square().mean(1)+w['residual_change']*(effective-previous_effective).square().mean(1)
                previous_effective=effective
            e.wp.capture_launch(e.graph);e.refresh_policy_sensors();e.phase+=self.policy_dt
            target,velocity,feet=self.ref.sample(e.phase)
            delta=e.q[:,:3]-target[:,:3]
            actual=e.q[:,3:7]/e.q[:,3:7].norm(dim=1,keepdim=True).clamp_min(1e-9)
            angle=2*torch.acos((actual*target[:,3:7]).sum(1).abs().clamp(0,1))
            pose=(e.q[:,self.body_qa_t]-target[:,self.body_qa_t]).square().mean(1)
            foot=(e.xpos[:,self.ref.feet_t]-feet).square().mean((1,2))
            stepcost=w['root_xy']*delta[:,:2].square().sum(1)+w['root_z']*delta[:,2].square()+w['orientation']*angle.square()+w['body_pose']*pose+w['feet']*foot+w['root_velocity']*(e.v[:,:3]-velocity[:,:3]).square().mean(1)+action_cost
            if self.com_reference is not None:
                i,j,blend=self.ref.indices(e.phase)
                com_goal=self.com_reference[i]+blend[:,None]*(self.com_reference[j]-self.com_reference[i])
                stepcost+=w['com_xy']*(self.com_sensor[:,self.live.c.base,:2]-com_goal[:,:2]).square().sum(1)
            joint_violation=None
            if self.limited_qadr is not None:
                actual_joint=e.q[:,self.limited_qadr]
                joint_violation=torch.maximum(self.joint_lower-actual_joint,actual_joint-self.joint_upper).clamp_min(0.)
                if w['joint_limit']>0:
                    near_limit=torch.maximum(self.joint_lower+self.joint_limit_margin-actual_joint,actual_joint-self.joint_upper+self.joint_limit_margin).clamp_min(0.)
                    stepcost+=w['joint_limit']*near_limit.square().sum(1)
            finite=torch.isfinite(e.q).all(1)&torch.isfinite(e.v).all(1)&torch.isfinite(stepcost)
            failed=(delta[:,:2].norm(dim=1)>.35)|(delta[:,2].abs()>.20)|(angle>.6)|~finite
            if self.reject_joint_limit:failed|=joint_violation.max(1).values>.05
            newly_failed=failed&~dead
            clean=torch.nan_to_num(stepcost,nan=1e6,posinf=1e6,neginf=1e6).clamp_max(1e6)
            cost+=torch.where(dead,torch.zeros_like(clean),clean)+newly_failed.float()*w['failed']*(self.horizon-h)
            dead|=failed;last_cost=clean
            # Failed candidates are not reset or repaired inside the horizon.
        cost+=w['terminal']*last_cost
        self.last_rollout_failed=dead.clone()
        self.planning_control_steps+=self.n*self.horizon;self.planning_inference_rows+=e.sonic.inference_count
        return torch.nan_to_num(cost,nan=1e9,posinf=1e9,neginf=1e9)

    def _candidates(self,mean,std,teacher,recovery):
        torch=self.torch
        teacher_fraction=float(self.config.get('recovery_teacher_fraction'if recovery else'teacher_fraction',.25 if recovery else .125))
        global_fraction=float(self.config.get('recovery_global_fraction'if recovery else'global_fraction',.25 if recovery else .125))
        if min(teacher_fraction,global_fraction)<0 or teacher_fraction+global_fraction>.8:raise ValueError('Candidate mixture fractions must be nonnegative and leave >=20% local samples')
        nt=int(self.n*teacher_fraction);ng=int(self.n*global_fraction)
        nt=min(nt,self.n-3);ng=min(ng,self.n-3-nt)
        sample_shape=tuple(mean.shape)
        half=torch.randn((self.n//2,*sample_shape),device=self.device,generator=self.rng)
        candidates=mean+std*torch.cat([half,-half],0)
        if nt:
            candidates[self.n-ng-nt:self.n-ng]=teacher+self.teacher_std*torch.randn((nt,*sample_shape),device=self.device,generator=self.rng)
        if ng:
            candidates[self.n-ng:]=self.global_std*torch.randn((ng,*sample_shape),device=self.device,generator=self.rng)
        candidates=candidates.clamp(-self.maximum_residual,self.maximum_residual)
        candidates[0]=mean;candidates[1]=0.;candidates[2]=teacher
        return candidates,{'fixed_seeds':3,'local_random':self.n-3-nt-ng,'around_teacher':nt,'global_zero_center':ng,'global_std_rad':self.global_std,'recovery':bool(recovery)}

    def _plan(self,data,t,base):
        torch=self.torch;started=time.monotonic()
        if self.residual_knots==1:mean=self.residual.clone()
        else:
            # Receding-horizon warm start: advance the previous temporal plan
            # by elapsed real execution time and hold its terminal value.
            elapsed=max(0.,float(t-self.plan_start_time))
            mean=torch.stack([residual_at_offset(self.residual_sequence,elapsed+k*self.residual_span/(self.residual_knots-1),self.residual_span)for k in range(self.residual_knots)])
        std=torch.full_like(mean,self.initial_std)
        target,_,_=self.ref.sample(torch.tensor([t],device=self.device))
        teacher=(target[0,self.body_qa_t]-base[0])[self.residual_t].clamp(-self.maximum_residual,self.maximum_residual)
        if self.residual_knots>1:teacher=teacher[None].expand(self.residual_knots,-1)
        best_cost=float('inf');best=mean.clone();best_failed=True;round_rows=[];recovery=self.recovery_mode
        for iteration in range(self.rounds):
            candidates,mixture=self._candidates(mean,std,teacher,recovery)
            costs=self._rollout_cost(candidates,data,t);elite=torch.topk(costs,self.elite,largest=False).indices
            index=int(elite[0]);value=float(costs[index])
            if value<best_cost:best_cost=value;best=candidates[index].clone();best_failed=bool(self.last_rollout_failed[index])
            chosen=candidates[elite];mean=chosen.mean(0);std=chosen.std(0,unbiased=False).clamp(self.minimum_std,self.initial_std)
            elite_all_failed=bool(self.last_rollout_failed[elite].all())
            round_rows.append({'round':iteration,'minimum_cost':value,'elite_cost_mean':float(costs[elite].mean()),'elite_all_failed':elite_all_failed,'failed_candidates':int(self.last_rollout_failed.sum()),'proposal_mixture':mixture})
            recovery=recovery or value>self.recovery_threshold or elite_all_failed
        if self.residual_knots==1:
            self.residual.copy_(best);self.residual_sequence[0].copy_(best)
        else:
            self.residual_sequence.copy_(best);self.residual.copy_(best[0])
        self.plan_start_time=float(t);self.plan_count+=1
        self.recovery_mode=best_cost>self.recovery_threshold or best_failed
        goal=target[0].cpu().numpy();delta=np.asarray(data.qpos[:3])-goal[:3]
        actual_q=np.asarray(data.qpos[3:7]);actual_q=actual_q/np.linalg.norm(actual_q)
        desired_q=goal[3:7]/np.linalg.norm(goal[3:7]);angle=float(2*np.arccos(np.clip(abs(actual_q@desired_q),0,1)))
        first=best if self.residual_knots==1 else best[0]
        row={'time_s':float(t),'source_time_s':float(self.ref.source_clock(torch.tensor([t],device=self.device))[0]),'best_rollout_cost':best_cost,'selected_residual_rad':first.cpu().tolist(),'selected_leg_residual_rad':first[:len(self.leg_columns)].cpu().tolist(),'selected_waist_residual_rad':first[len(self.leg_columns):].cpu().tolist(),'residual_knots':self.residual_knots,'selected_residual_knots_rad':self.residual_sequence.cpu().tolist(),'residual_knot_offsets_s':np.linspace(0.,self.residual_span,self.residual_knots).tolist(),'residual_joint_names':[self.names[k]for k in self.residual_columns],'best_candidate_failed':best_failed,'recovery_mode':self.recovery_mode,'root_xy_error_m':float(np.linalg.norm(delta[:2])),'root_z_error_m':float(abs(delta[2])),'root_orientation_error_rad':angle,'live_root_qpos_xyz_wxyz':data.qpos[:7].tolist(),'reference_root_xyz':goal[:3].tolist(),'wall_s':time.monotonic()-started,'rounds':round_rows}
        self.diagnostics.append(row)
        if self.plan_count==1 or self.plan_count%self.progress_every==0:
            print(json.dumps({'stage':'cem_mpc_progress','plan_count':self.plan_count,**{k:row[k]for k in ('time_s','source_time_s','best_rollout_cost','root_xy_error_m','root_z_error_m','root_orientation_error_rad','live_root_qpos_xyz_wxyz','recovery_mode','wall_s')}}),flush=True)

    def action(self,data,t):
        if t+1e-8<self.next_time:return self.ctrl.copy()
        if not self.plant_checked:self._check_plant()
        if np.any(data.qfrc_applied)or np.any(data.xfrc_applied):raise ValueError('MPC live plant must have zero external forces')
        torch=self.torch
        q=torch.as_tensor(data.qpos.copy(),device=self.device,dtype=torch.float32)[None]
        v=torch.as_tensor(data.qvel.copy(),device=self.device,dtype=torch.float32)[None]
        phase=torch.tensor([t],device=self.device,dtype=torch.float32);b=self.live.c.base
        r=torch.as_tensor(data.xmat[b].reshape(3,3).copy(),device=self.device,dtype=torch.float32)[None]
        omega=torch.as_tensor(data.cvel[b,:3].copy(),device=self.device,dtype=torch.float32)[None]
        with torch.inference_mode():
            base=self.live.targets(q,v,phase,r,omega)
            if t+1e-8>=self.next_plan_time:
                self._plan(data,t,base);self.next_plan_time+=self.replan_dt
            if self.residual_knots>1:self.residual.copy_(residual_at_offset(self.residual_sequence,float(t-self.plan_start_time),self.residual_span))
            packet=self._packet(phase,base,self.residual[None]);self.live.record_applied(packet)
            if self.residual_knots>1:self.effective_residual.copy_((packet[0,self.live.aids]-base[0])[self.residual_t])
        self.last_base_target=base[0].cpu().numpy().copy();self.ctrl=packet[0].cpu().numpy().copy()
        self.last=self.live.last[0].cpu().numpy().copy();self.target=self.ctrl[self.aids].copy();self.target_history.append(self.target.copy())
        self.inference_count+=1;self.next_time+=self.policy_dt
        return self.ctrl.copy()

    def _base_report(self):
        return {'controller':'Frozen all28 SONIC outputs +'+str(self.residual_dim)+' leg/waist CEM MPC residual, auxiliary reference PD','is_original_sonic_checkpoint':False,'method':'Counterfactual real MuJoCo-Warp rollouts; actual execution remains native MuJoCo actuator control','policy_dt':self.policy_dt,'physics_dt':self.physics_dt,'inference_count':self.inference_count,'body_joint_names':self.names,'leg_residual_joint_names':[self.names[k]for k in self.leg_columns],'include_waist':self.include_waist,'residual_dimensions':self.residual_dim,'residual_joint_names':[self.names[k]for k in self.residual_columns],'waist_residual_joint_names':[self.names[k]for k in self.waist_columns]if self.include_waist else[],'replan_dt_s':self.replan_dt,'candidates':self.n,'elites':self.elite,'cem_rounds':self.rounds,'horizon_s':self.horizon*self.policy_dt,'initial_std_rad':self.initial_std,'maximum_residual_rad':self.maximum_residual,'proposal_mixture':{'teacher_std_rad':self.teacher_std,'global_std_rad':self.global_std,'recovery_cost_threshold':self.recovery_threshold,'normal_teacher_fraction':float(self.config.get('teacher_fraction',.125)),'normal_global_fraction':float(self.config.get('global_fraction',.125)),'recovery_teacher_fraction':float(self.config.get('recovery_teacher_fraction',.25)),'recovery_global_fraction':float(self.config.get('recovery_global_fraction',.25))},'progress_every_plans':self.progress_every,'plan_count':self.plan_count,'planning_control_steps':self.planning_control_steps,'planning_inference_rows':self.planning_inference_rows,'counterfactual_world_initializations':self.counterfactual_initializations,'planning_wall_s_total':sum(r['wall_s']for r in self.diagnostics),'live_root_state_writes':0,'external_wrenches':False,'live_state_repairs':False,'counterfactual_state_writes':'Only isolated planning worlds, restored to current live observed integration state and SONIC history before each CEM round; never reset inside a horizon','live_planning_plant_equal':self.plant_checked,'last_record_units':'Normalized effective SONIC actuator commands including selected MPC residual; not a newly learned network action','cost_weights':self.weights,'software_wrist_command_windows':self.ref.software_wrist_command_windows,'control_bound_mode':self.ref.control_bound_mode,'control_bound_contract':self.ref.control_bound_contract,'frozen_sonic':self.live.c.report(),'reference_sha256':sha(self.reference_path),'model_input_sha256':sha(self.model_path),'source_sha256':sha(__file__),'limitations':['GPU FP32 predictions approximate native FP64 dynamics; independent native task validation remains required.','Only declared leg/waist residuals optimized over a finite horizon; no explicit contact-force reward or hypothetical force application.','Uses privileged actual simulator state and offline planning slower than real time.']}


    def report(self):
        result=self._base_report()
        result['residual_knots']=self.residual_knots
        result['wrist_ctrl_window_override_rad']=self.wrist_window_override
        result['optional_joint_limit_objective']={'enabled':self.limited_qadr is not None,'weight':self.weights['joint_limit'],'sum_squared_hinge_penalty_starts_inside_true_limit_rad':self.joint_limit_margin,'reject_rollout_violation_above_rad':.05 if self.reject_joint_limit else None,'joint_names':self.limited_joint_names if self.limited_qadr is not None else[],'actual_limited_hinge_count':len(self.limited_qadr)if self.limited_qadr is not None else 0,'physical_limits_or_target_bounds_modified':False}
        result['optional_com_cost']={'enabled':self.com_reference is not None,'world_xy_weight':self.weights['com_xy'],'reference_field':'reference_com_world interpolated with Reference.indices on the execution clock','actual_sensor':'MuJoCo-Warp data.subtree_com[:, base_link, :] from the existing per-step derived-state computation','sensor_shape':list(self.com_sensor.shape)if self.com_sensor is not None else None,'timing':'Like existing foot-position rollout costs, derived kinematics describe the final native integration start of each50Hz policy interval; no live or planning state is overwritten for this cost','forces_or_constraints_added':False}
        result['temporal_residual_contract']={
            'optimized_shape':[self.residual_knots,self.residual_dim]if self.residual_knots>1 else[self.residual_dim],
            'knot_offsets_s':np.linspace(0.,self.residual_span,self.residual_knots).tolist(),
            'interpolation':'Piecewise linear across the first through final control-step offsets; terminal value held beyond horizon',
            'live_execution':'Selected temporal plan evaluated at each50Hz policy boundary between replans; native5ms control remains held between policy boundaries',
            'warm_start':'Previous temporal plan shifted by elapsed execution time and resampled at the new knot offsets',
            'action_cost':'Original constant-residual per-step arithmetic preserved'if self.residual_knots==1 else'Per-step effective post-command-bound residual magnitude and difference from the preceding applied residual; first step compared with preceding live applied residual',
            'teacher_proposal':'Current reference-minus-SONIC residual repeated at each knot; temporal alternatives sampled by CEM',
            'live_state_or_force_writes':False}
        return result


def main():
    """A short native execution probe; the normal physics runner does full QC."""
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--config',type=Path);p.add_argument('--seconds',type=float,default=1.)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    import mujoco as mj
    model=mj.MjModel.from_xml_path(str(a.model.resolve()));c=MPCController(model,a.reference,a.model,config=json.loads(a.config.read_text())if a.config else None)
    data=mj.MjData(model);data.qpos[:]=c.ref.q[0].cpu().numpy();data.qvel[:]=c.ref.dq[0].cpu().numpy();mj.mj_forward(model,data);c.reset(data)
    a.output.mkdir(parents=True);mj.mj_saveModel(model,str(a.output/'model.mjb'))
    rows=[];q=[data.qpos.copy()];v=[data.qvel.copy()];ts=[data.time]
    warning_counts=np.asarray([w.number for w in data.warning]);end_reason='requested_duration'
    for k in range(int(np.ceil(a.seconds/model.opt.timestep))):
        packet=c.action(data,float(data.time));data.ctrl[:]=packet;mj.mj_step(model,data)
        rows.append(packet);q.append(data.qpos.copy());v.append(data.qvel.copy());ts.append(data.time)
        if not np.isfinite(data.qpos).all()or not np.isfinite(data.qvel).all()or np.any(np.asarray([w.number for w in data.warning])>warning_counts):end_reason='solver_failure';break
    np.savez_compressed(a.output/'probe_trace.npz',time=np.asarray(ts),qpos=np.asarray(q),qvel=np.asarray(v),ctrl=np.asarray(rows))
    result={'status':'mpc_native_execution_probe_only_not_full_task_acceptance','end_reason':end_reason,'controller':c.report(),'duration_s':float(data.time),'mujoco_version':mj.__version__,'diagnostics':c.diagnostics}
    (a.output/'probe_report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
