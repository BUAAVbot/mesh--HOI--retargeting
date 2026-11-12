"""Vita MuJoCo adaptation of privileged refinement and validated progressive resets.

This is an engineering adaptation inspired by SUGAR, not its IsaacLab trainer.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
import mujoco as mj
from two_sitting_native_ppo_env import NativeTracking
from two_sitting_ppo_env import rotate_inverse


class RefinerEnv(NativeTracking):
    def __init__(self,*args,**kwargs):
        self.pool={};self.pending=[];self.pool_added=0;self.pool_resets=0;self.reference_resets=0
        self.pool_rejected=0;self.control_count=0;self.iteration=0
        self.rng=np.random.default_rng(kwargs.get('seed',1))
        super().__init__(*args,**kwargs)
        self.tracked=[]
        names=['base_link','left_hip_roll_link','right_hip_roll_link',
               'left_knee_link','right_knee_link','left_ankle_roll_link','right_ankle_roll_link',
               'left_shoulder_roll_link','right_shoulder_roll_link','left_elbow_link','right_elbow_link',
               'left_wrist_yaw_link','right_wrist_yaw_link']
        for prefix in ('p1_','p2_'):
            ids=[mj.mj_name2id(self.model,mj.mjtObj.mjOBJ_BODY,prefix+name)for name in names]
            if min(ids)<0:raise ValueError('Missing tracked Vita body')
            self.tracked.append(ids)
        self.tracked_t=torch.as_tensor(self.tracked,device=self.device)
        target_positions=[];scratch=mj.MjData(self.model)
        for q in self.ref.q.detach().cpu().numpy():
            scratch.qpos[:]=q;mj.mj_kinematics(self.model,scratch)
            target_positions.append(scratch.xpos[np.asarray(self.tracked)].copy())
        self.target_positions=torch.as_tensor(np.asarray(target_positions),device=self.device,dtype=torch.float32)
        spec=json.loads(Path('/data/KinoGen/outputs/two_sitting_pipeline/assets_deploy_v3/support_geometry.json').read_text())
        seats=spec['seats'];self.seat_targets=[]
        qref=self.ref.q.cpu().numpy()
        for rq in self.ref.root_qadr:
            centers=np.asarray([s['center_xy_m']for s in seats]);points=qref[:,rq:rq+2]
            nearest=np.linalg.norm(points[:,None]-centers[None],axis=-1).argmin(-1)
            self.seat_targets.append(np.asarray([[*centers[k],seats[k]['top_z_m']+.175]for k in nearest]))
        self.seat_targets=torch.as_tensor(np.stack(self.seat_targets,axis=1),device=self.device,dtype=torch.float32)
        self.weight=torch.as_tensor([self.model.body_subtreemass[m['base']]*9.81 for m in self.ref.maps],device=self.device)
        self.contact_contract+=' Expert reward includes actual upward seat load and disclosed seat-relative pose shaping.'

    def reset(self,ids,force_start=False):
        if not len(ids):return
        if not isinstance(ids,torch.Tensor):ids=torch.as_tensor(ids,device=self.device,dtype=torch.long)
        count=len(ids);phase=torch.zeros(count,device=self.device)
        selected=[None]*count
        probability=min(.70,max(0.,(self.iteration-30)/170*.70))
        bins=list(self.pool)
        for j in range(count):
            if not force_start and bins and self.rng.random()<probability:
                entries=self.pool[bins[int(self.rng.integers(len(bins)))]]
                selected[j]=entries[int(self.rng.integers(len(entries)))];phase[j]=selected[j]['phase']
            elif not force_start and self.rng.random()>.40:
                # Conventional RSI is a disclosed warmup seed, never labeled a validated pool state.
                phase[j]=float(self.rng.uniform(0,self.ref.end))
        target,velocity,_=self.ref.sample(phase)
        previous=torch.zeros((count,56),device=self.device)
        controls=self.ref.controls(phase,previous,self.action_scale)
        body=controls[:,self.ref.body_aids_t].clone()
        cpu_q=target.cpu().numpy();cpu_v=velocity.cpu().numpy();cpu_u=controls.cpu().numpy()
        start=np.asarray([x is None and float(phase[j])==0 for j,x in enumerate(selected)])
        cpu_v[start]=0.
        for j,item in enumerate(selected):
            if item is not None:
                cpu_q[j]=item['q'];cpu_v[j]=item['v'];cpu_u[j]=item['ctrl']
                previous[j]=torch.as_tensor(item['previous'],device=self.device)
                body[j]=torch.as_tensor(item['body_target'],device=self.device)
                self.pool_resets+=1
            else:self.reference_resets+=1
        self.phase[ids]=phase;self.from_start[ids]=torch.as_tensor(start,device=self.device)
        self.starts+=int(start.sum());self.steps[ids]=0;self.previous[ids]=previous
        self.u[ids]=torch.as_tensor(cpu_u,device=self.device);self.previous_body_target[ids]=body
        self.cold_start[ids]=torch.as_tensor([x is None for x in selected],device=self.device)
        if self.sonic is not None:
            self.sonic.reset(ids)
        cpu_ids=ids.cpu().numpy()
        for i in cpu_ids:
            for pending in self.pending:
                if pending['env']==int(i):pending['valid']=False
        list(self.executor.map(self._reset_one,[(int(index),cpu_q[j],cpu_v[j],cpu_u[j])for j,index in enumerate(cpu_ids)]))
        self.state_write_reset_events+=count;self._generation+=1;self._host_generation=self._generation
        self.refresh_policy_sensors()

    def step(self,actions):
        obs,base_reward,done,info=super().step(actions)
        self.control_count+=1
        i,j,w=self.ref.indices(self.phase)
        goal=self.target_positions[i]+w[:,None,None,None]*(self.target_positions[j]-self.target_positions[i])
        actual=self.xpos[:,self.tracked_t]
        relative=[]
        for actor,rq in enumerate(self.ref.root_qadr):
            local=actual[:,actor]-self.q[:,None,rq:rq+3]
            target=self.ref.sample(self.phase)[0]
            target_local=goal[:,actor]-target[:,None,rq:rq+3]
            q=self.q[:,rq+3:rq+7];tq=target[:,rq+3:rq+7]
            relative.append((rotate_inverse(q[:,None].expand(-1,local.shape[1],-1),local)-
                             rotate_inverse(tq[:,None].expand(-1,local.shape[1],-1),target_local)).square().mean((1,2)))
        relative=torch.stack(relative,1).mean(-1)
        _,seat=self.ref.support_at(self.phase)
        seat=(seat>.5).float();seat_goal=self.seat_targets[i]+w[:,None,None]*(self.seat_targets[j]-self.seat_targets[i])
        roots=torch.stack([self.q[:,rq:rq+3]for rq in self.ref.root_qadr],1)
        err=roots-seat_goal
        geometry=torch.exp(-err[:,:,:2].square().sum(-1)/.15**2-err[:,:,2].square()/.10**2)
        load=(info['seat_support_force_N']/(.15*self.weight[None])).clamp(0,1)
        consistency=torch.where(seat.bool(),load,(info['seat_support_force_N']<1.).float()).mean(-1)
        torso=torch.stack([self.v[:,rv:rv+6].square().mean(-1)for rv in self.ref.root_vadr],1).mean(-1)
        reward=base_reward+.15*torch.exp(-relative/.12**2)+.2*(geometry*seat).mean(-1)+.2*consistency-.002*torso
        terminal_bonus=(self.phase>=self.ref.end)&(load.min(-1).values>.65)&(torso<.03)
        reward+=.3*terminal_bonus.float()
        # Validate candidate snapshots over the following ten control steps, not only at capture.
        qcpu=self.q.detach().cpu().numpy();tilt=np.stack([np.arccos(np.clip(1-2*(qcpu[:,rq+4]**2+qcpu[:,rq+5]**2),-1,1))for rq in self.ref.root_qadr],1)
        good=(~info['failed']&(info['root_xy']<.45)&(info['max_penetration_m']<.025)&
              (info['root_angle']<.65)&torch.as_tensor(tilt.max(-1)<.65,device=self.device))
        _,required_seat=self.ref.support_at(self.phase)
        actual_load_ok=((info['seat_support_force_N']>.10*self.weight[None])|(required_seat<=.5)).all(-1)
        good &= actual_load_ok
        live_good=good.cpu().numpy();failed=info['failed'].cpu().numpy()
        retained=[]
        for candidate in self.pending:
            candidate['valid'] &= bool(live_good[candidate['env']] and not failed[candidate['env']])
            if self.control_count>=candidate['due']:
                if candidate['valid']:
                    entry=candidate['state'];key=int(entry['phase']/.1)
                    bucket=self.pool.setdefault(key,[]);bucket.append(entry)
                    new_positions=[j for j,item in enumerate(bucket) if not bool(item.get('is_old_physical_prior_seed',False))]
                    if len(new_positions)>16:bucket.pop(new_positions[0])
                    self.pool_added+=1
                else:self.pool_rejected+=1
            else:retained.append(candidate)
        self.pending=retained
        if self.iteration>=30 and self.control_count%20==0:
            eligible=np.flatnonzero(live_good&~done.cpu().numpy())
            if len(eligible)>16:eligible=self.rng.choice(eligible,16,replace=False)
            for index in eligible:
                state={'phase':float(self.phase[index]),'q':self.datas[index].qpos.copy(),
                       'v':self.datas[index].qvel.copy(),'ctrl':self.datas[index].ctrl.copy(),
                       'previous':self.previous[index].cpu().numpy().copy(),
                       'body_target':self.previous_body_target[index].cpu().numpy().copy(),
                       'captured_iteration':self.iteration,'root_xy_m':float(info['root_xy'][index]),
                       'penetration_m':float(info['max_penetration_m'][index])}
                self.pending.append({'env':int(index),'due':self.control_count+10,'valid':True,'state':state})
        info.update(relative_body_mse=relative,seat_geometry=geometry.mean(-1),seat_load_fraction=load.min(-1).values)
        return obs,reward,done,info

    def save_pool(self,path):
        entries=[entry for key in sorted(self.pool)for entry in self.pool[key]]
        arrays={key:np.asarray([entry.get(key,False) if key=='is_old_physical_prior_seed' else entry[key] for entry in entries])for key in
                ('phase','q','v','ctrl','previous','body_target','captured_iteration','root_xy_m','penetration_m','is_old_physical_prior_seed')}
        np.savez_compressed(path,**arrays)
        return {'states':len(entries),'phase_bins':len(self.pool),'validated_candidates_added':self.pool_added,
                'candidates_rejected':self.pool_rejected,'pool_resets':self.pool_resets,
                'reference_RSI_resets':self.reference_resets,'validation_horizon_control_steps':10,
                'validation_seconds':.2,'max_pool_probability':.70,'immutable_old_positive_seeds':sum(bool(e.get('is_old_physical_prior_seed',False))for e in entries),'expected_seated_load_gate_fraction_of_body_weight':.10,
                'note':'Old validated positive seeds are immutable; new candidates must sustain actual seated load whenever expected by refined labels. Pool contains actual native integrated states that survived the subsequent validation window; RSI seeds are separately counted and are not called physically validated.'}


"""Explicit dense heading and terminal seat-load shaping for a frozen-SONIC expert.

This is a local engineering ablation, not SUGAR's original reward implementation.
"""
import torch


class ResidualHeadingEnv(RefinerEnv):
    def step(self, actions):
        obs, reward, done, info = super().step(actions)
        # The original exponential orientation reward is nearly flat for large
        # heading deviations; preserve a dense penalty before the failure gate.
        heading_cost = .08 * info['root_angle_per_actor'].square().mean(-1)
        root_cost = .10 * info['root_xy_per_actor'].mean(-1)
        i, j, w = self.ref.indices(self.phase)
        goals = self.seat_targets[i] + w[:, None, None] * (self.seat_targets[j]-self.seat_targets[i])
        roots = torch.stack([self.q[:, rq:rq+3] for rq in self.ref.root_qadr], 1)
        delta = roots-goals
        geometry = torch.exp(-delta[:, :, :2].square().sum(-1)/.18**2-delta[:, :, 2].square()/.12**2)
        _, seat = self.ref.support_at(self.phase)
        seat = (seat > .5).float()
        late = (self.phase > .55*self.ref.end).float()
        force, _, _ = self.contact_summary()
        load = (force/(.15*self.weight[None])).clamp(0, 1)
        root_speed = torch.stack([self.v[:, rv:rv+3].square().sum(-1) for rv in self.ref.root_vadr], 1)
        stable = torch.exp(-root_speed/.15**2)
        terminal_interaction = ((.40*load+.30*geometry+.15*stable*load)*seat).mean(-1)*late
        reward = reward-heading_cost-root_cost+terminal_interaction
        info['dense_heading_cost'] = heading_cost
        info['terminal_interaction_reward'] = terminal_interaction
        return obs, reward, done, info


"""Soft source-root tracking during final sitting; physical checks remain unchanged."""
import torch


class ResidualSoftSeatEnv(ResidualHeadingEnv):
    def step(self, actions):
        obs, reward, done, info = super().step(actions)
        _, seat = self.ref.support_at(self.phase)
        mask = (seat > .5).float()*(self.phase[:,None] > .55*self.ref.end).float()
        xy = info['root_xy_per_actor']
        # Remove only the declared global-source XY shaping for seated actors.
        # Root-relative body shape, heading, no-fall and independent final
        # source-error reporting remain present.
        original_xy = .30*torch.exp(-xy.square()/.15**2)-.20*xy.square()
        reward -= (original_xy*mask).mean(-1)
        reward += (.10*xy*mask).mean(-1)
        i,j,w = self.ref.indices(self.phase)
        goals = self.seat_targets[i]+w[:,None,None]*(self.seat_targets[j]-self.seat_targets[i])
        roots = torch.stack([self.q[:,rq:rq+3] for rq in self.ref.root_qadr],1)
        distance = (roots[:,:,:2]-goals[:,:,:2]).norm(dim=-1)
        force,_,_ = self.contact_summary()
        load = (force/(.15*self.weight[None])).clamp(0,1)
        reward += ((.30*load-.08*distance)*mask).mean(-1)
        info['soft_seat_actor_mask'] = mask
        return obs,reward,done,info


"""Use a declared previously executed physical start and validated reset-state pool."""
from pathlib import Path
import os
import numpy as np
import torch
import mujoco as mj


class ResidualSeededSeatEnv(ResidualSoftSeatEnv):
    def __init__(self,*args,**kwargs):
        self.fixed_initial_path=Path(os.environ.get('VITA_FIXED_INITIAL_STATE',
            '/data/KinoGen/outputs/sonic_expert_20261028/prior_physical_state_pool/candidate156_initial_state.npz'))
        with np.load(self.fixed_initial_path,allow_pickle=False) as z:
            self.fixed_initial={key:z[key].copy() for key in z.files}
        super().__init__(*args,**kwargs)

    def reset(self,ids,force_start=False):
        super().reset(ids,force_start=force_start)
        if not len(ids):return
        if not isinstance(ids,torch.Tensor):ids=torch.as_tensor(ids,device=self.device,dtype=torch.long)
        selected=ids[self.phase[ids].abs()<1e-8]
        if not len(selected):return
        state=self.fixed_initial['state'];spec=mj.mjtState(int(self.fixed_initial['state_spec']))
        if len(state)!=mj.mj_stateSize(self.model,spec):raise ValueError('Physical start state dimension mismatch')
        for index in selected.detach().cpu().numpy():
            d=self.datas[int(index)]
            mj.mj_setState(self.model,d,state,spec);mj.mj_forward(self.model,d)
            if np.any(d.qfrc_applied)or np.any(d.xfrc_applied):raise ValueError('Assisting wrench in seed')
            self._numeric_fault[int(index)]=False;self._contacts_one(int(index));self._cache_one(int(index))
        packet=torch.as_tensor(self.fixed_initial['ctrl'],device=self.device,dtype=self.u.dtype)
        self.u[selected]=packet
        self.previous[selected]=0.
        self.previous_body_target[selected]=packet[self.ref.body_aids_t]
        self.cold_start[selected]=True
        if self.sonic is not None:self.sonic.reset(selected)
        self._generation+=1;self._host_generation=self._generation;self.refresh_policy_sensors()
