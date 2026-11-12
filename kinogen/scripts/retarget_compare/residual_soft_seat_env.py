"""Soft source-root tracking during final sitting; physical checks remain unchanged."""
from __future__ import annotations
import torch
from residual_heading_env import ResidualHeadingEnv


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
