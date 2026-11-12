"""Bounded swing-step augmentation through SONIC's reference input only.

The policy has no direct world-root XY error observation. This module turns a
small horizontal tracking error into a modest swing-foot reference extension.
It never writes the executing plant, its controls, or external forces.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import mujoco as mj
import numpy as np

SOURCE_BYTES = Path(__file__).read_bytes()
SOURCE_SHA256 = hashlib.sha256(SOURCE_BYTES).hexdigest()


def clip_norm(vector, maximum):
    vector = np.asarray(vector, float)
    return vector * min(1., maximum/max(float(np.linalg.norm(vector)), 1e-12))


def swing_envelope(times, support):
    """Exactly zero on declared stance, with zero-endpoint swing envelopes."""
    support = np.asarray(support, bool)
    output = np.zeros(len(times))
    edges = np.diff(np.r_[False, ~support, False].astype(int))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)-1
    for start, end in zip(starts, ends):
        if end <= start:
            continue
        phase = (times[start:end+1]-times[start])/(times[end]-times[start])
        output[start:end+1] = np.sin(np.pi*phase)**2
    return output


class ReferenceFootFeedback:
    """Call apply(part, actual_plant_data, elapsed) immediately before inference.

    ``part`` needs controller, proxy, pm, sm and p1_/p2_ prefix, as constructed
    by two_sitting_physics.Motors. Supports the existing newft and tracking
    controller's qref/body_q/body_dq interface. Original reference buffers are
    immutable private copies, so repeated calls cannot accumulate changes.
    """

    def __init__(self, reference, config=None):
        self.config = dict(config or {})
        c = self.config
        self.gain = float(c.get('reference_foot_gain', .7))
        self.max_delta = float(c.get('reference_foot_max_world_delta_m', .10))
        self.max_offset = float(c.get('reference_foot_max_joint_offset_rad', .12))
        self.max_speed = float(c.get('reference_foot_max_offset_speed_rad_s', .8))
        self.filter_tau = float(c.get('reference_foot_filter_tau_s', .15))
        self.offset_tau = float(c.get('reference_foot_offset_filter_tau_s', .08))
        self.damping = float(c.get('reference_foot_ik_damping', .002))
        self.orientation_weight = float(c.get('reference_foot_orientation_weight_m', .08))
        if not (0 <= self.gain <= 2 and 0 <= self.max_delta <= .10 and
                0 <= self.max_offset <= .12 and 0 < self.max_speed <= 2 and
                self.filter_tau >= 0 and self.offset_tau >= 0 and self.damping > 0):
            raise ValueError('Unsafe/unbounded reference-foot feedback configuration')
        self.times = np.asarray(reference['time_s'], float).copy()
        self.qpos = np.asarray(reference['qpos'], float).copy()
        self.support = np.asarray(reference['foot_support_expected'], bool).copy()
        if (len(self.times) < 3 or np.any(np.diff(self.times) <= 0) or
                self.qpos.shape != (len(self.times), 2, 59) or
                self.support.shape != (len(self.times), 2, 2)):
            raise ValueError('Expected aligned two-person reference and foot_support_expected[T,2,2]')
        seated = np.asarray(reference.get('seat_support_expected', np.zeros((len(self.times), 2))), float)
        if seated.shape != (len(self.times), 2):
            raise ValueError('seat_support_expected must be T x 2')
        # Smooth the seat gate, but never move a declared stance-foot reference.
        distance = abs(self.times[:, None]-self.times[None, :])
        weights = np.exp(-.5*(distance/.12)**2)
        self.seat_weight = np.clip(weights@seated/weights.sum(axis=1)[:, None], 0., 1.)
        self.states, self.records = {}, []

    def initialize(self, part):
        controller, proxy, pm = part['controller'], part['proxy'], part['pm']
        prefix = str(part['prefix'])
        if prefix not in ('p1_', 'p2_'):
            raise ValueError('Expected p1_/p2_ actor')
        actor = 0 if prefix == 'p1_' else 1
        if not np.allclose(controller.times, self.times, rtol=0, atol=1e-9):
            raise ValueError('Reference-feedback and policy timelines differ')
        if np.asarray(controller.body_q).shape != (len(self.times), 28):
            raise ValueError('Expected canonical 28-body-joint controller')
        original_q = np.asarray(controller.body_q, float).copy()
        original_dq = np.asarray(controller.body_dq, float).copy()
        kin = mj.MjData(proxy)
        chains = []
        for side_index, side in enumerate(('left', 'right')):
            indices = np.asarray([j for j, name in enumerate(controller.names)
                                  if name.startswith(side+'_') and any(
                                      token in name for token in ('_hip_', '_knee_', '_ankle_'))])
            if len(indices) != 6:
                raise ValueError('Expected six canonical '+side+' leg joints')
            ids = controller.jids[indices]
            foot = mj.mj_name2id(proxy, mj.mjtObj.mjOBJ_BODY, side+'_ankle_roll_link')
            if foot < 0:
                raise ValueError('Missing canonical ankle body')
            envelope = swing_envelope(self.times, self.support[:, actor, side_index])
            envelope *= (1-self.seat_weight[:, actor])**2
            chains.append({'side': side, 'indices': indices, 'jids': ids,
                           'va': proxy.jnt_dofadr[ids], 'body': foot,
                           'envelope': envelope, 'step_map': np.zeros((len(self.times), 6, 3)),
                           'foot_world': np.zeros((len(self.times), 3))})
        jac, jacr = np.zeros((3, proxy.nv)), np.zeros((3, proxy.nv))
        for frame in range(len(self.times)):
            original = self.qpos[frame, actor]
            kin.qpos[pm['rq']:pm['rq']+7] = original[:7]
            kin.qpos[pm['qa']] = original[7:]
            mj.mj_kinematics(proxy, kin)
            mj.mj_comPos(proxy, kin)
            for chain in chains:
                mj.mj_jacBody(proxy, kin, jac, jacr, chain['body'])
                matrix = np.vstack([jac[:, chain['va']],
                                    self.orientation_weight*jacr[:, chain['va']]])
                target = np.vstack([np.eye(3), np.zeros((3, 3))])
                chain['step_map'][frame] = np.linalg.solve(
                    matrix.T@matrix+self.damping*np.eye(6), matrix.T@target)
                chain['foot_world'][frame] = kin.xpos[chain['body']]
        state = {'actor': actor, 'original_body_q': original_q,
                 'original_body_dq': original_dq,
                 'original_reference_qpos': self.qpos[:, actor].copy(),
                 'offset': np.zeros_like(original_q), 'filtered_delta': np.zeros(2),
                 'last_time': None, 'last_update': None, 'chains': chains,
                 'original_body_q_sha256': hashlib.sha256(original_q.tobytes()).hexdigest(),
                 'original_body_dq_sha256': hashlib.sha256(original_dq.tobytes()).hexdigest(),
                 'reference_leg_columns': 7+np.asarray(controller.jref)}
        self.states[prefix] = state
        return state

    def apply(self, part, data, elapsed):
        controller = part.get('controller')
        if controller is None:
            raise ValueError('Reference-foot feedback requires a learned controller')
        # Match the policy's 20 ms update cadence. Repeated physical substeps
        # keep the previous plan; the next policy evaluation sees one update.
        if float(elapsed)+1e-8 < float(controller.next_inference_time):
            return None
        prefix = str(part['prefix'])
        state = self.states.get(prefix)
        if state is None:
            state = self.initialize(part)
        if state['last_update'] is not None and abs(float(elapsed)-state['last_update']) < 1e-10:
            return None
        actor, sm = state['actor'], part['sm']
        t = float(elapsed)+float(self.times[0])
        pref = np.asarray([np.interp(t, self.times, self.qpos[:, actor, j]) for j in range(3)])
        actual = np.asarray(data.qpos[sm['rq']:sm['rq']+3], float)
        error = pref[:2]-actual[:2]
        raw_delta = clip_norm(self.gain*error, self.max_delta)
        if t >= self.times[-1]:
            raw_delta[:] = 0.
        dt = float(controller.policy_dt) if state['last_time'] is None else max(0., float(elapsed)-state['last_time'])
        alpha = 1. if self.filter_tau == 0 else dt/(self.filter_tau+dt)
        state['filtered_delta'] += alpha*(raw_delta-state['filtered_delta'])
        delta = np.r_[clip_norm(state['filtered_delta'], self.max_delta), 0.]
        target_offset = np.zeros_like(state['original_body_q'])
        for chain in state['chains']:
            offset = np.einsum('tij,j->ti', chain['step_map'], delta)
            offset *= chain['envelope'][:, None]
            offset = np.clip(offset, -self.max_offset, self.max_offset)
            original = state['original_body_q'][:, chain['indices']]
            limits = part['proxy'].jnt_range[chain['jids']]
            lower = np.maximum(-self.max_offset, limits[:, 0]-original)
            upper = np.minimum(self.max_offset, limits[:, 1]-original)
            feasible = lower <= upper
            offset = np.where(feasible, np.minimum(np.maximum(offset, lower), upper), 0.)
            offset[chain['envelope'] <= 1e-10] = 0.
            target_offset[:, chain['indices']] = offset
        beta = 1. if self.offset_tau == 0 else dt/(self.offset_tau+dt)
        change = beta*(target_offset-state['offset'])
        state['offset'] += np.clip(change, -self.max_speed*dt, self.max_speed*dt)
        offset = state['offset'].copy()
        derivative = np.gradient(offset, self.times, axis=0, edge_order=2)
        peak_speed = float(np.max(np.abs(derivative)))
        if peak_speed > self.max_speed:
            scale = self.max_speed/peak_speed
            offset *= scale
            derivative *= scale
            state['offset'][:] = offset
        # Update only leg-reference columns. A concurrent reference-lean module
        # owns root orientations; all executing plant arrays remain read-only.
        leg_indices = np.concatenate([chain['indices'] for chain in state['chains']])
        controller.body_q[:, leg_indices] = state['original_body_q'][:, leg_indices]+offset[:, leg_indices]
        controller.body_dq[:, leg_indices] = state['original_body_dq'][:, leg_indices]+derivative[:, leg_indices]
        columns = state['reference_leg_columns'][leg_indices]
        controller.qref[:, columns] = state['original_reference_qpos'][:, columns]+offset[:, leg_indices]
        state['last_time'], state['last_update'] = float(elapsed), float(elapsed)
        now_offset = np.asarray([np.interp(t, self.times, offset[:, j]) for j in range(28)])
        current_swing = [float(np.interp(t, self.times, chain['envelope'])) for chain in state['chains']]
        self.records.append({'time_s': float(elapsed), 'actor': actor+1,
                             'root_error_world_xy_m': error.tolist(),
                             'requested_delta_world_xy_m': raw_delta.tolist(),
                             'filtered_delta_world_xy_m': delta[:2].tolist(),
                             'swing_envelope_left_right': current_swing,
                             'current_body_reference_offset_rad': now_offset.tolist(),
                             'max_future_joint_reference_offset_rad': float(np.max(np.abs(offset))),
                             'max_future_joint_reference_offset_velocity_rad_s': float(np.max(np.abs(derivative))),
                             'source_body_reference_sha256': state['original_body_q_sha256']})
        return now_offset

    def report(self):
        return {'controller': 'Bounded horizontal swing-foot extension in learned future joint-reference input',
                'source_file': str(Path(__file__).resolve()), 'source_sha256': SOURCE_SHA256,
                'gain': self.gain, 'world_delta_bound_m': self.max_delta,
                'joint_reference_offset_bound_rad': self.max_offset,
                'joint_reference_offset_speed_bound_rad_s': self.max_speed,
                'root_error_filter_tau_s': self.filter_tau,
                'joint_offset_filter_tau_s': self.offset_tau,
                'orientation_ik_weight_m': self.orientation_weight,
                'ik_damping': self.damping, 'records': len(self.records),
                'immutable_source_body_q_sha256': {p:s['original_body_q_sha256'] for p,s in self.states.items()},
                'immutable_source_body_dq_sha256': {p:s['original_body_dq_sha256'] for p,s in self.states.items()},
                'modified_fields': ['learned_controller.body_q[:, leg_indices]',
                                    'learned_controller.body_dq[:, leg_indices]',
                                    'learned_controller.qref[:, 7+leg_reference_indices]'],
                'stance_rule': 'Exact zero correction on declared support frames. Sin-squared envelope inside each swing interval; smoothed seated suppression.',
                'velocity_rule': 'Original body_dq plus exact reference-time finite difference of applied bounded offset.',
                'plant_state_writes': 0, 'motor_control_writes': 0, 'external_force_writes': 0,
                'limitations': ['Synthetic policy reference adaptation; checkpoint and learned motor outputs unchanged.',
                                'First-order original-reference FK Jacobian approximates foot displacement; no guaranteed task success.',
                                'Support labels are inferred; incorrect timing can still destabilize the learned tracker.',
                                'Simultaneous stance of both feet suppresses step correction; this does not create a gait from standing.']}
