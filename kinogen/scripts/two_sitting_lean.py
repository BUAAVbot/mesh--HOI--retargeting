"""Steer a learned tracker through bounded, synthetic reference root tilt.

Only ``part['controller'].qref[:, 3:7]`` is modified. The executing plant's
state, motor controls and applied forces are never modified by this module.
With +X forward, positive pitch leans toward +X; negative roll toward +Y.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np

SOURCE_BYTES = Path(__file__).read_bytes()


def yaw_of(q):
    q = np.asarray(q, float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def yaw_roll_pitch_quaternion(yaw, roll, pitch):
    """WXYZ for Rz(yaw) Ry(pitch) Rx(roll), using broadcastable arrays."""
    cy, sy = np.cos(yaw/2), np.sin(yaw/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    cr, sr = np.cos(roll/2), np.sin(roll/2)
    return np.stack((cy*cp*cr+sy*sp*sr,
                     cy*cp*sr-sy*sp*cr,
                     sy*cp*sr+cy*sp*cr,
                     sy*cp*cr-cy*sp*sr), axis=-1)


def norm_clip(value, limit):
    value = np.asarray(value, float)
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    return value*np.minimum(1., float(limit)/np.maximum(norm, 1e-12))


class ReferenceLeanController:
    """Call apply(part, shared_plant_data, elapsed) before learned inference.

    ``part`` is the dual-Vita motor adapter dictionary, including ``sm``
    (shared root qpos/qvel addresses), ``prefix`` and ``controller``.
    The checkpoint is unchanged. The bounded tilt is an explicitly synthetic
    goal supplied through the existing future-orientation observation.
    """

    def __init__(self, reference, config=None):
        self.config = dict(config or {})
        c = self.config
        self.max_lean = float(c.get('max_lean', c.get('lean_max', .1)))
        self.kp = float(c.get('kp', c.get('lean_kp', .8)))
        self.kd = float(c.get('kd', c.get('lean_kd', 1.0)))
        self.kff = float(c.get('kff', 1.))
        self.max_acc = float(c.get('max_acc', 1.5))
        self.tau = float(c.get('filter_tau_s', .15))
        self.seat_suppression_power = float(c.get('seat_suppression_power', 2.))
        self.gravity = float(c.get('gravity_m_s2', 9.81))
        if not (0 <= self.max_lean <= .3 and self.max_acc > 0 and self.tau >= 0
                and self.gravity > 0 and self.kp >= 0 and self.kd >= 0):
            raise ValueError('Invalid bounded reference-lean configuration')
        self.times = np.asarray(reference['time_s'], float).copy()
        self.qpos = np.asarray(reference['qpos'], float).copy()
        if self.qpos.shape != (len(self.times), 2, 59) or len(self.times) < 3 or np.any(np.diff(self.times) <= 0):
            raise ValueError('Need the two-actor full-root reference with increasing time')
        self.velocity = np.gradient(self.qpos[:, :, :3], self.times, axis=0)
        self.acceleration = np.gradient(self.velocity, self.times, axis=0)
        if 'seat_support_blend_weight' in reference:
            self.seat_weight = np.asarray(reference['seat_support_blend_weight'], float).copy()
        else:
            # Smooth a declared sitting phase in execution time, without
            # introducing a new dependency on the reconstruction source.
            seated = np.asarray(reference.get('seat_support_expected', np.zeros((len(self.times), 2))), float)
            radius = float(c.get('seat_smoothing_s', .25))
            distance = abs(self.times[:, None]-self.times[None, :])
            weights = np.exp(-.5*(distance/max(radius, 1e-6))**2)
            self.seat_weight = weights@seated/weights.sum(axis=1)[:, None]
        if self.seat_weight.shape != (len(self.times), 2):
            raise ValueError('Seat suppression requires T x 2 weights')
        self.seat_weight = np.clip(self.seat_weight, 0., 1.)
        self.states = {}
        self.records = []

    def at(self, values, t):
        values = np.asarray(values)
        if values.ndim == 1:
            return float(np.interp(t, self.times, values))
        return np.array([np.interp(t, self.times, values[:, k]) for k in range(values.shape[1])])

    def apply(self, part, data, elapsed):
        controller = part.get('controller')
        if controller is None:
            raise ValueError('Reference lean requires a learned controller')
        prefix = str(part.get('prefix', ''))
        if prefix not in ('p1_', 'p2_'):
            raise ValueError('Expected p1_/p2_ actor prefix')
        actor = 0 if prefix == 'p1_' else 1
        state = self.states.get(prefix)
        if state is None:
            original = np.asarray(controller.qref[:, 3:7], float).copy()
            if original.shape != (len(self.times), 4):
                raise ValueError('Policy and lean reference timeline dimensions differ')
            if not np.allclose(np.asarray(controller.times), self.times, rtol=0, atol=1e-9):
                raise ValueError('Policy and lean execution clocks differ')
            state = {'original_quaternion': original,
                     'yaw': np.unwrap(yaw_of(original)),
                     'filtered_acceleration': np.zeros(2),
                     'last_time': None,
                     'original_quaternion_sha256': hashlib.sha256(original.tobytes()).hexdigest()}
            self.states[prefix] = state
        sm = part['sm']
        rq, rv = int(sm['rq']), int(sm['rv'])
        t = float(elapsed)+float(self.times[0])
        pref = self.at(self.qpos[:, actor, :3], t)
        vref = self.at(self.velocity[:, actor], t)
        aref = self.at(self.acceleration[:, actor], t)
        # The terminal reference hold has zero desired translational velocity.
        if t >= self.times[-1]:
            vref[:] = 0.; aref[:] = 0.
        position = np.asarray(data.qpos[rq:rq+3], float)
        velocity = np.asarray(data.qvel[rv:rv+3], float)
        error = pref-position
        raw = self.kp*error[:2]+self.kd*(vref[:2]-velocity[:2])+self.kff*aref[:2]
        bounded = norm_clip(raw, self.max_acc)
        dt = 0. if state['last_time'] is None else max(0., float(elapsed)-state['last_time'])
        alpha = 1. if self.tau == 0 else dt/(self.tau+dt)
        state['filtered_acceleration'] += alpha*(bounded-state['filtered_acceleration'])
        state['last_time'] = float(elapsed)
        acceleration = state['filtered_acceleration']
        # Apply the same current world correction to each future frame, but
        # express it in that frame's original yaw and suppress sitting frames.
        yaw = state['yaw']; cy, sy = np.cos(yaw), np.sin(yaw)
        body_x = cy*acceleration[0]+sy*acceleration[1]
        body_y = -sy*acceleration[0]+cy*acceleration[1]
        tilt = np.c_[-np.arctan2(body_y, self.gravity), np.arctan2(body_x, self.gravity)]
        tilt = norm_clip(tilt, self.max_lean)
        gain = (1-self.seat_weight[:, actor])**self.seat_suppression_power
        tilt *= gain[:, None]
        synthetic = yaw_roll_pitch_quaternion(yaw, tilt[:, 0], tilt[:, 1])
        # Match the immutable original quaternion hemisphere for interpolation;
        # never multiply yesterday's synthetic quaternion by another tilt.
        sign = np.where(np.sum(synthetic*state['original_quaternion'], axis=1) < 0, -1., 1.)
        synthetic *= sign[:, None]
        controller.qref[:, 3:7] = synthetic
        nowtilt = self.at(tilt, t)
        self.records.append({'time_s': float(elapsed), 'actor': actor+1,
                             'reference_position_world_m': pref.tolist(),
                             'actual_position_world_m': position.copy().tolist(),
                             'position_error_world_m': error.tolist(),
                             'reference_velocity_world_m_s': vref.tolist(),
                             'actual_velocity_world_m_s': velocity.copy().tolist(),
                             'reference_acceleration_world_m_s2': aref.tolist(),
                             'raw_command_acceleration_world_xy_m_s2': raw.tolist(),
                             'bounded_command_acceleration_world_xy_m_s2': bounded.tolist(),
                             'filtered_command_acceleration_world_xy_m_s2': acceleration.copy().tolist(),
                             'synthetic_roll_pitch_rad': nowtilt.tolist(),
                             'synthetic_tilt_norm_rad': float(np.linalg.norm(nowtilt)),
                             'seat_suppression_gain': self.at(gain, t)})
        return nowtilt.copy()

    def report(self):
        return {'controller': 'bounded synthetic reference orientation lean through learned policy input',
                'source_sha256': hashlib.sha256(SOURCE_BYTES).hexdigest(),
                'max_lean_rad': self.max_lean, 'kp_s_minus2': self.kp,
                'kd_s_minus1': self.kd, 'acceleration_feedforward_gain': self.kff,
                'max_acceleration_m_s2': self.max_acc, 'filter_tau_s': self.tau,
                'seat_suppression_power': self.seat_suppression_power,
                'records': len(self.records),
                'peak_applied_reference_tilt_rad': max([r['synthetic_tilt_norm_rad'] for r in self.records] or [0.]),
                'original_quaternion_sha256': {k: v['original_quaternion_sha256'] for k, v in self.states.items()},
                'quaternion_composition': 'Rz(immutable original reference yaw) Ry(synthetic pitch) Rx(synthetic roll)',
                'sign_convention': 'positive pitch toward local +X; negative roll toward local +Y',
                'modified_fields': ['learned_controller.qref[:, 3:7]'],
                'plant_state_writes': 0, 'applied_external_force_writes': 0,
                'disclosure': 'Synthetic reference input adaptation based on actual root position/velocity. Checkpoint unchanged; success would validate this hybrid adaptation, not the original unmodified tracker.'}
