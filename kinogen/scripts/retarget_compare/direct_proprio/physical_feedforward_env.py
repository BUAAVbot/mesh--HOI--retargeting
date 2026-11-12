"""Immutable historical200Hz physical control prior plus newly learned feedback.

The historical command already contains its original delay. Only the new
feedback is delayed by three5ms steps. Zero feedback must exactly replay it.
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import torch
import mujoco as mj
from direct_proprio import DirectProprioEnv
from two_sitting_ppo_env import rotate_inverse

PRIOR=Path('/data/KinoGen/outputs/sonic_expert_20261028/prior_physical_state_pool/candidate156_actual_states_controls.npz')
CONTRACT='direct_feedforward_proprio_seat_v3: SUGAR-style proprio/seat plus actual historical motor command and physical-plan body/root error; no SONIC inference'


class PhysicalFeedforwardEnv(DirectProprioEnv):
    def __init__(self,*args,**kwargs):
        self.prior=dict(np.load(PRIOR,allow_pickle=False))
        self.nominal_controls=self.prior['ctrl'];self.nominal_time=self.prior['time']
        self.last_controls=None;self.feedback_previous=None
        self.capture_full=False
        super().__init__(*args,**kwargs)
        self.ref.horizon=float(self.nominal_time[-1])
        self.prior_q=torch.as_tensor(self.prior['qpos'],device=self.device,dtype=torch.float32)
        self.prior_seat=torch.as_tensor(self.prior['seat_contact'],device=self.device,dtype=torch.float32)
        original=self.ref.support_at
        def support_at(phase):
            foot,_=original(phase)
            idx=torch.round(phase/.005).long().clamp(0,len(self.prior_seat)-1)
            return foot,self.prior_seat[idx]
        self.ref.support_at=support_at
        self.policy_sensor_contract+='; '+CONTRACT

    def enable_capture(self):
        if self.n!=1:raise ValueError('Full200Hz capture requires one live world')
        self.capture_full=True;self.capture_steps=0
        self.capture_geom_names=[mj.mj_id2name(self.model,mj.mjtObj.mjOBJ_GEOM,i)or str(i)for i in range(self.model.ngeom)]
        self.capture_body_names=[mj.mj_id2name(self.model,mj.mjtObj.mjOBJ_BODY,i)or str(i)for i in range(self.model.nbody)]

    def base_controls(self):
        self.refresh_policy_sensors()
        indices=np.rint(self.phase.detach().cpu().numpy()/.005).astype(int)+3
        indices=np.clip(indices,0,len(self.nominal_controls)-1)
        # Float32 nominal is used only to subtract the new learned feedback.
        # Actual immutable historical commands are applied in float64 below.
        self.nominal_policy_cpu=self.nominal_controls[indices].astype(np.float32)
        self.control_start_indices=np.rint(self.phase.detach().cpu().numpy()/.005).astype(int)
        source=self.prior['source_time_s'][np.clip(self.control_start_indices,0,len(self.prior['source_time_s'])-1)]
        self.feedback_gain=.15+.85*np.clip((source-4.2)/.7,0.,1.)
        return torch.as_tensor(self.nominal_policy_cpu,device=self.device).clone()

    def observations(self):
        original=super().observations()
        ids=torch.round(self.phase/.005).long().clamp(0,len(self.prior_q)-1)
        planned=self.prior_q[ids];values=[]
        for actor,rq in enumerate(self.ref.root_qadr):
            quat=self.q[:,rq+3:rq+7];quat=quat/quat.norm(dim=-1,keepdim=True).clamp_min(1e-8)
            qa=self.ref.qa_t[actor*28:(actor+1)*28]
            values.extend([rotate_inverse(quat,planned[:,rq:rq+3]-self.q[:,rq:rq+3]),
                           planned[:,qa]-self.q[:,qa]])
        nominal=torch.as_tensor(self.nominal_policy_cpu[:,self.ref.body_aids],device=self.device)
        return torch.cat([original,nominal,*values],-1).clamp(-10.,10.)

    def reset(self,ids,force_start=False):
        if not len(ids):return
        if not isinstance(ids,torch.Tensor):ids=torch.as_tensor(ids,device=self.device,dtype=torch.long)
        if self.feedback_previous is None:
            self.feedback_previous=np.zeros((self.n,56),dtype=np.float64)
            self.last_controls=np.zeros((self.n,4,104),dtype=np.float64)
            self.last_qpos_steps=np.zeros((self.n,4,118),dtype=np.float64)
            self.last_qvel_steps=np.zeros((self.n,4,116),dtype=np.float64)
        phase=torch.zeros(len(ids),device=self.device);entries=[None]*len(ids);bins=list(self.pool)
        for j in range(len(ids)):
            if not force_start and bins and self.rng.random()>=.5:
                bucket=self.pool[bins[int(self.rng.integers(len(bins)))]]
                entry=bucket[int(self.rng.integers(len(bucket)))];entries[j]=entry
                phase[j]=float(entry.get('old_execution_time_s',entry['phase']))
        self.phase[ids]=phase;starts=np.asarray([entry is None for entry in entries])
        self.from_start[ids]=torch.as_tensor(starts,device=self.device);self.starts+=int(starts.sum())
        self.steps[ids]=0;self.previous[ids]=0.;self.cold_start[ids]=False
        for j,index in enumerate(ids.cpu().numpy()):
            entry=entries[j];d=self.datas[int(index)]
            if entry is None:
                bag=self.physical_initial;state=np.asarray(bag['initial_state'])
                spec=mj.mjtState(int(np.asarray(bag['state_spec']).item()))
                mj.mj_resetData(self.model,d);mj.mj_setState(self.model,d,state,spec)
                mj.mj_forward(self.model,d);mj.mj_setState(self.model,d,state,spec)
                self.feedback_previous[index]=0.;self.reference_resets+=1
                noise=float(os.environ.get('FF_TRAIN_INITIAL_NOISE','0'))
                if noise and index:
                    d.qpos[self.ref.qa]+=self.rng.uniform(-noise,noise,len(self.ref.qa))
                    d.qpos[self.ref.qa]=np.clip(d.qpos[self.ref.qa],self.model.jnt_range[self.ref.jids,0],self.model.jnt_range[self.ref.jids,1])
                    for rq in self.ref.root_qadr:d.qpos[rq:rq+2]+=self.rng.uniform(-noise,noise,2)
                    mj.mj_forward(self.model,d)
            else:
                mj.mj_resetData(self.model,d);d.qpos[:]=entry['q'];d.qvel[:]=entry['v'];d.ctrl[:]=entry['ctrl']
                mj.mj_forward(self.model,d)
                self.previous[index]=torch.as_tensor(entry['previous'],device=self.device)
                self.feedback_previous[index]=entry.get('feedback_target',np.zeros(56))
                self.pool_resets+=1
            self._numeric_fault[index]=False;self._contacts_one(int(index));self._cache_one(int(index))
        for index in ids.cpu().numpy():
            for candidate in self.pending:
                if candidate['env']==int(index):candidate['valid']=False
        self.state_write_reset_events+=len(ids);self._generation+=1;self._host_generation=self._generation
        self.refresh_policy_sensors()

    def _advance_one(self,index):
        data=self.datas[index]
        if np.any(data.qfrc_applied)or np.any(data.xfrc_applied):raise RuntimeError('Assisting external force forbidden')
        new=(self._control_new_cpu[index,self.ref.body_aids]-self.nominal_policy_cpu[index,self.ref.body_aids])*self.feedback_gain[index]
        old=self.feedback_previous[index].copy()
        first=int(self.control_start_indices[index]);max_pen=0.
        if self.capture_full:self.current_capture=[]
        for sub in range(4):
            command=self.nominal_controls[min(first+sub,len(self.nominal_controls)-1)].copy()
            feedback=old if sub<3 else new
            if np.any(feedback):
                command[self.ref.body_aids]=np.clip(command[self.ref.body_aids]+feedback,
                    self.model.jnt_range[self.ref.jids,0],self.model.jnt_range[self.ref.jids,1])
            data.ctrl[:]=command;self.last_controls[index,sub]=command;mj.mj_step(self.model,data)
            self.last_qpos_steps[index,sub]=data.qpos;self.last_qvel_steps[index,sub]=data.qvel
            self._contacts_one(index);max_pen=max(max_pen,float(self._penetration_cpu[index]))
            if self.capture_full:
                from two_sitting_physics import contacts
                detail=contacts(self.model,data,self.capture_steps,self.capture_geom_names,self.capture_body_names)
                self.current_capture.append({'time':float(data.time),'qpos':data.qpos.copy(),'qvel':data.qvel.copy(),
                    'ctrl':command.copy(),'actuator_force':data.actuator_force.copy(),'qfrc_actuator':data.qfrc_actuator.copy(),
                    'contacts':detail})
                self.capture_steps+=1
                if any(data.warning[i].number>0 for i in self._warning_ids):break
        self._penetration_cpu[index]=max_pen;self.feedback_previous[index]=new
        self._numeric_fault[index]=any(data.warning[i].number>0 for i in self._warning_ids)
        self._cache_one(index)

    def step(self,actions):
        obs,reward,done,info=super().step(actions)
        idx=torch.round(self.phase/.005).long().clamp(0,len(self.prior_q)-1);goal=self.prior_q[idx]
        error=torch.stack([(self.q[:,rq:rq+2]-goal[:,rq:rq+2]).square().sum(-1)for rq in self.ref.root_qadr],1).mean(-1)
        gate=torch.as_tensor(self.feedback_gain,device=self.device,dtype=reward.dtype)
        reward=reward+.2*(1-gate)*torch.exp(-error/.1**2)
        for pending in self.pending:
            if 'feedback_target'not in pending['state']:
                pending['state']['feedback_target']=self.feedback_previous[pending['env']].copy()
                pending['state']['body_target']=self.last_controls[pending['env'],-1,self.ref.body_aids].copy()
        return obs,reward,done,info

    def save_pool(self,path):
        report=super().save_pool(path)
        entries=[entry for key in sorted(self.pool)for entry in self.pool[key]]
        with np.load(path,allow_pickle=False)as archive:values=dict(archive)
        values['feedback_target']=np.asarray([entry.get('feedback_target',np.zeros(56))for entry in entries])
        np.savez_compressed(path,**values)
        report.update(from_start_probability=.5,max_pool_probability=.5,reference_RSI_probability=0.,
                      physical_control_clock='actual old execution time, no double terminal hold',
                      historical_nominal_delay_added=False,new_feedback_delay_steps=3)
        return report
