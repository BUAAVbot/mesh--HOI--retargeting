"""Map measured hand/object contact reactions to actuator feedforward only.

The returned vector is a controller estimate. This module never changes model
geometry, simulation state, actuator limits, qfrc_applied, or xfrc_applied.
"""
import time
import numpy as np
import mujoco as mj


class HandReactionFeedforward:
    def __init__(self, model, object_body_ids=None, hand_prefixes=('rh_', 'lh_'), filter_tau_s=.1):
        self.m = model
        if object_body_ids is None:
            object_body_ids = [int(model.jnt_bodyid[j]) for j in range(model.njnt)
                               if model.jnt_type[j] == mj.mjtJoint.mjJNT_FREE
                               and int(model.jnt_qposadr[j]) != 0]
        self.objects = set(map(int, object_body_ids))
        self.hands = {b for b in range(model.nbody)
                      if (model.body(b).name or '').startswith(tuple(hand_prefixes))}
        self.filter_tau = float(filter_tau_s)
        self.kin = mj.MjData(model)
        self.raw = np.zeros(model.nv)
        self.filtered = np.zeros(model.nv)
        self.last_time = None
        self.records = []
        self.wall_s = 0.
        self.update_count = 0

    def update(self, data):
        now = float(data.time)
        if self.last_time is not None and abs(now-self.last_time) < 1e-12:
            return self.filtered
        started = time.perf_counter()
        m = self.m
        kin = self.kin
        kin.qpos[:] = data.qpos
        mj.mj_kinematics(m, kin)
        mj.mj_comPos(m, kin)
        raw = np.zeros(m.nv)
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        force = np.zeros(6)
        patches = []
        for ci, c in enumerate(data.contact[:data.ncon]):
            b1, b2 = map(int, m.geom_bodyid[np.asarray(c.geom, dtype=int)])
            if b1 in self.hands and b2 in self.objects:
                hand, sign = b1, -1.
            elif b2 in self.hands and b1 in self.objects:
                hand, sign = b2, 1.
            else:
                continue
            mj.mj_contactForce(m, data, ci, force)
            frame = np.asarray(c.frame).reshape(3, 3)
            f_world = sign*frame.T@force[:3]
            tau_world = sign*frame.T@force[3:]
            mj.mj_jac(m, kin, jp, jr, c.pos, hand)
            raw += jp.T@f_world+jr.T@tau_world
            patches.append({'hand_body': m.body(hand).name, 'hand_geom_order': 1 if sign < 0 else 2,
                            'force_on_hand_world_N': f_world.tolist(),
                            'torque_on_hand_world_Nm': tau_world.tolist(),
                            'contact_position_world_m': c.pos.tolist()})
        dt = 0. if self.last_time is None else max(0., now-self.last_time)
        alpha = 1. if self.filter_tau <= 0 else -np.expm1(-dt/self.filter_tau)
        self.filtered += alpha*(raw-self.filtered)
        self.raw[:] = raw
        self.last_time = now
        self.wall_s += time.perf_counter()-started
        self.update_count += 1
        if not self.records or now-self.records[-1]['time_s'] >= .019999:
            self.records.append({'time_s': now, 'raw_generalized_reaction_N_Nm': raw.tolist(),
                                 'filtered_generalized_reaction_N_Nm': self.filtered.tolist(),
                                 'patches': patches})
        return self.filtered

    def report(self):
        return {'method': 'Actual contact force on hand mapped by point and rotational Jacobians; motor inverse statics subtracts this reaction.',
                'contact_sign': 'MuJoCo local wrench acts on geom2; invert if hand is geom1.',
                'filter_tau_s': self.filter_tau,
                'selected_object_body_ids': sorted(self.objects),
                'update_count': self.update_count,
                'wall_s': self.wall_s,
                'mean_update_wall_ms': 1000*self.wall_s/max(1, self.update_count),
                'applied_force_writes': 0,
                'records': self.records}
