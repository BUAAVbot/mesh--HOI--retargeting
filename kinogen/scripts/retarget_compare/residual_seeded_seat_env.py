"""Use a declared previously executed physical start and validated reset-state pool."""
from __future__ import annotations
from pathlib import Path
import os
import numpy as np
import torch
import mujoco as mj
from residual_soft_seat_env import ResidualSoftSeatEnv


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
