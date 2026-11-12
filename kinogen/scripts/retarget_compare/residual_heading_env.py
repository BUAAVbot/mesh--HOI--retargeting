"""Explicit dense heading and terminal seat-load shaping for a frozen-SONIC expert.

This is a local engineering ablation, not SUGAR's original reward implementation.
"""
from __future__ import annotations
import torch
from sugar_expert_env import RefinerEnv


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
