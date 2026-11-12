"""Explicit proprioception for the independent privileged Vita expert."""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import torch
import mujoco as mj
from two_sitting_ppo_env import rotate_inverse
from sugar_expert_env import RefinerEnv

CONTRACT='direct_proprio_v1: tracking errors plus actual/command body q, actual/command projected gravity, actual body-frame root linear/angular velocities and actual root height'


def augment(model,ref,q,v,phase,cvel):
    target,_,_=ref.sample(phase);values=[]
    for actor,(rq,rv)in enumerate(zip(ref.root_qadr,ref.root_vadr)):
        actual=q[:,rq+3:rq+7];actual=actual/actual.norm(dim=-1,keepdim=True).clamp_min(1e-8)
        goal=target[:,rq+3:rq+7];goal=goal/goal.norm(dim=-1,keepdim=True).clamp_min(1e-8)
        gravity=torch.tensor([0.,0.,-1.],device=q.device).expand(len(q),3)
        ids=ref.qa_t[actor*28:(actor+1)*28]
        base=ref.maps[actor]['base']
        values.extend([q[:,ids],target[:,ids],rotate_inverse(actual,gravity),rotate_inverse(goal,gravity),
            .25*rotate_inverse(actual,v[:,rv:rv+3]),
            .25*rotate_inverse(actual,cvel[:,base,:3]),q[:,rq+2:rq+3]])
    return torch.cat(values,dim=-1).clamp(-10.,10.)


class DirectProprioEnv(RefinerEnv):
    def __init__(self,*args,**kwargs):
        path=os.environ.get('DIRECT_INITIAL_STATE_PATH')
        self.physical_initial=None if not path else dict(np.load(Path(path),allow_pickle=False))
        super().__init__(*args,**kwargs)
        if self.sonic is not None:raise ValueError('Direct proprio expert must not execute SONIC')
        self.policy_sensor_contract+='; '+CONTRACT

    def observations(self):
        original=super().observations()
        return torch.cat([original,augment(self.model,self.ref,self.q,self.v,self.phase,self.cvel)],-1)

    def reset(self,ids,force_start=False):
        super().reset(ids,force_start)
        if self.physical_initial is None or not len(ids):return
        if not isinstance(ids,torch.Tensor):ids=torch.as_tensor(ids,device=self.device,dtype=torch.long)
        selected=ids[self.from_start[ids]]
        if not len(selected):return
        bag=self.physical_initial;state=np.asarray(bag.get('initial_state',bag['state']))
        spec=mj.mjtState(int(np.asarray(bag['state_spec']).item()))
        for index in selected.cpu().numpy():
            data=self.datas[int(index)]
            mj.mj_setState(self.model,data,state,spec)
            # Only an explicit episode reset uses an archived physical state.
            mj.mj_forward(self.model,data)
            mj.mj_setState(self.model,data,state,spec)
            self._numeric_fault[int(index)]=False;self._contacts_one(int(index));self._cache_one(int(index))
        self._generation+=1;self._host_generation=self._generation;self.refresh_policy_sensors()
