"""Bounded late-seat joint-target feedback; no root/state/force injection.

The returned packet is an unbounded *position target*. The caller must preserve
it for action history, then apply its ordinary physical joint bounds/delay.
Only symmetric stance-basis and hip/knee/ankle targets change. Root translation
is never commanded by a root actuator or injected into the actual state.
"""
from __future__ import annotations

from pathlib import Path

import mujoco as mj
import numpy as np


class SeatingFeedback:
    def __init__(self, model, reference, config=None):
        self.model = model
        if isinstance(reference, (str, Path)):
            with np.load(reference, allow_pickle=False) as z:
                reference = {key: z[key] for key in z.files}
        self.reference = reference
        config = dict(config or {})
        # Configuration may make this mechanism more conservative, never expand
        # the requested 0.12-rad/0.04-rad-per-second envelope.
        self.max_offset = min(.12, float(config.get('max_offset_rad', .12)))
        self.rate = min(.04, float(config.get('offset_rate_rad_s', .04)))
        self.window = min(.3, float(config.get('activation_window_s', .3)))
        self.max_xy = min(.35, float(config.get('max_root_xy_error_m', .35)))
        self.max_tilt = min(.5, float(config.get('max_global_tilt_rad', .5)))
        self.height_margin = max(.005, float(config.get('root_height_margin_m', .005)))
        self.load_fraction = max(.10, float(config.get('seat_load_fraction', .10)))
        self.center_max = min(.20, float(config.get('centering_max_offset_rad', .20)))
        self.center_rate = min(.04, float(config.get('centering_rate_rad_s', .04)))
        self.center_xy_threshold = min(.15, float(config.get('centering_root_xy_threshold_m', .15)))
        self.center_gain = float(config.get('centering_integral_gain_s_inv', 1.))
        self.fk_epsilon = float(config.get('centering_fk_epsilon_rad', 1e-4))
        if min(self.max_offset, self.rate, self.window, self.max_xy, self.max_tilt) <= 0 or not all(np.isfinite(x) for x in (
            self.max_offset, self.rate, self.window, self.max_xy, self.max_tilt,
            self.height_margin, self.load_fraction)):
            raise ValueError('Finite positive seating feedback settings required')
        if min(self.center_max, self.center_rate, self.center_xy_threshold, self.center_gain, self.fk_epsilon) <= 0 or not all(np.isfinite(x) for x in (
            self.center_max, self.center_rate, self.center_xy_threshold, self.center_gain, self.fk_epsilon)):
            raise ValueError('Finite positive centering feedback settings required')
        times = np.asarray(reference['time_s'], dtype=float)
        qref = np.asarray(reference['qpos'], dtype=float)
        if qref.shape != (len(times), 2, 59) or len(times) < 2 or not np.all(np.diff(times) > 0):
            raise ValueError('Expected a timed T x 2 x 59 dual-Vita reference')
        self.times = times
        self.end_time = float(times[-1]-times[0])
        self.start_time = self.end_time-self.window
        self.final_root = qref[-1, :, :3].copy()
        names = list(map(str, reference['dof_names']))
        if len(names) != 52 or len(set(names)) != 52:
            raise ValueError('Expected 52 unique Vita dof_names')
        self.geom_names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, i) or '' for i in range(model.ngeom)]
        self.body_names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i) or '' for i in range(model.nbody)]
        self.seat_ids = {i for i, name in enumerate(self.geom_names)
                         if 'seat' in name.lower() and 'pedestal' not in name.lower()
                         and 'leg' not in name.lower() and model.geom_bodyid[i] == 0
                         and (model.geom_contype[i] or model.geom_conaffinity[i])}
        if not self.seat_ids:
            raise ValueError('No active fixed seat-surface colliders found')
        self.parts = []
        gravity = float(np.linalg.norm(model.opt.gravity))
        for actor, prefix in enumerate(('p1_', 'p2_')):
            jids = np.asarray([self.named(mj.mjtObj.mjOBJ_JOINT, prefix+name) for name in names])
            aids = np.asarray([self.named(mj.mjtObj.mjOBJ_ACTUATOR, prefix+'motor_'+name) for name in names])
            if not np.array_equal(model.actuator_trnid[aids, 0], jids):
                raise ValueError('Actuator/joint name mapping mismatch: '+prefix)
            root = next((mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, prefix+name)
                         for name in ('vita_demo_free_root', 'vita_ik_free_root')
                         if mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, prefix+name) >= 0), -1)
            if root < 0 or model.jnt_type[root] != mj.mjtJoint.mjJNT_FREE:
                raise ValueError('Expected an unactuated free root: '+prefix)
            change = []
            for side in ('left', 'right'):
                for suffix, factor in (('hip_pitch_joint', -1.), ('knee_joint', 2.), ('ankle_pitch_joint', -1.)):
                    name = side+'_'+suffix
                    change.append((int(aids[names.index(name)]), factor, prefix+name))
            basis = [[], []]
            foot_geom_ids = []
            foot_body_ids = []
            for side, short in (('left', 'l'), ('right', 'r')):
                for basis_index, spec in enumerate((
                    (('hip_pitch_joint', 1.), ('ankle_pitch_joint', -1.)),
                    (('hip_roll_joint', -1.), ('ankle_roll_joint', 1.)))):
                    for suffix, factor in spec:
                        col = names.index(side+'_'+suffix)
                        basis[basis_index].append((int(aids[col]), int(model.jnt_qposadr[jids[col]]), factor,
                                                   prefix+side+'_'+suffix))
                foot_body = self.named(mj.mjtObj.mjOBJ_BODY, prefix+side+'_ankle_roll_link')
                foot_body_ids.append(foot_body)
                geoms = [g for g, name in enumerate(self.geom_names)
                         if name.startswith(prefix+'collision_foot_'+short+'_')
                         and (model.geom_contype[g] or model.geom_conaffinity[g])]
                if not geoms:
                    geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == foot_body
                             and (model.geom_contype[g] or model.geom_conaffinity[g])]
                foot_geom_ids.append(geoms)
            bodies = [i for i, name in enumerate(self.body_names) if name.startswith(prefix)]
            mass = float(np.sum(model.body_mass[bodies]))
            if mass <= 0 or gravity <= 0:
                raise ValueError('Positive robot weight is required')
            self.parts.append({'actor': actor, 'prefix': prefix, 'jids': jids, 'aids': aids,
                               'rq': int(model.jnt_qposadr[root]), 'mass_kg': mass,
                               'weight_N': mass*gravity, 'load_threshold_N': self.load_fraction*mass*gravity,
                               'changes': change, 'basis': basis, 'foot_geom_ids': foot_geom_ids,
                               'foot_body_ids': foot_body_ids})
        self.offset = np.zeros(2)
        self.center_offset = np.zeros((2, 2))
        self.scratch = mj.MjData(model)
        self.fk_test_count = 0
        self.first_fk_test = None
        self.last_fk_test = None
        self.latched = np.zeros(2, dtype=bool)
        self.latch_time = [None, None]
        self.last_time = None
        self.records = []
        self.active_calls = 0

    def named(self, kind, name):
        value = mj.mj_name2id(self.model, kind, name)
        if value < 0:
            raise ValueError('Missing '+name)
        return value

    def seat_loads(self, data):
        """Use actual latest solver contact wrenches, not virtual spring forces."""
        loads = np.zeros(2)
        for i in range(data.ncon):
            contact = data.contact[i]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if (g1 in self.seat_ids) == (g2 in self.seat_ids):
                continue
            robot_g = g2 if g1 in self.seat_ids else g1
            robot_name = self.geom_names[robot_g]
            body_name = self.body_names[int(self.model.geom_bodyid[robot_g])]
            # Sitting support must act on hips/pelvis, not hands, toes or shin.
            if not any(token in robot_name.lower() or token in body_name.lower()
                       for token in ('hip', 'pelvis', 'base_link')):
                continue
            wrench = np.zeros(6)
            mj.mj_contactForce(self.model, data, i, wrench)
            world_force_on_g2 = np.asarray(contact.frame).reshape(3, 3).T@wrench[:3]
            upward = float(world_force_on_g2[2])*(1. if robot_g == g2 else -1.)
            for actor, part in enumerate(self.parts):
                if body_name.startswith(part['prefix']):
                    loads[actor] += max(0., upward)
                    break
        return loads

    def stance_root_sensitivity(self, actual_data, actor=0):
        """Numerical stance basis test in private FK data; actual data is read-only.

        For each foot use the mean active foot-collider centers (or its ankle
        body origin if that asset has no foot collider). A world-fixed stance
        implies a first-order root response opposite to the mean foot response.
        This is an approximate feedback basis, not a direct root displacement.
        """
        actor = int(actor)
        part = self.parts[actor]
        actual_qpos = np.asarray(actual_data.qpos).copy()

        def feet():
            return np.asarray([np.mean(self.scratch.geom_xpos[ids], axis=0) if ids
                               else self.scratch.xpos[body].copy()
                               for ids, body in zip(part['foot_geom_ids'], part['foot_body_ids'])])

        self.scratch.qpos[:] = actual_qpos
        mj.mj_kinematics(self.model, self.scratch)
        baseline = feet()
        deltas = []
        for basis in part['basis']:
            self.scratch.qpos[:] = actual_qpos
            for _, qadr, factor, _ in basis:
                self.scratch.qpos[qadr] += factor*self.fk_epsilon
            mj.mj_kinematics(self.model, self.scratch)
            deltas.append(np.mean(feet()-baseline, axis=0))
        sensitivity = -np.asarray(deltas).T[:2]/self.fk_epsilon
        live_unchanged = bool(np.array_equal(actual_qpos, actual_data.qpos))
        if not live_unchanged:
            raise AssertionError('Private FK must never change actual robot state')
        if not np.isfinite(sensitivity).all():
            raise ValueError('Nonfinite private FK centering sensitivity')
        test = {'actor': actor+1, 'epsilon_rad': self.fk_epsilon,
                'source': 'actual integrated qpos copied to private MjData; mj_kinematics only',
                'foot_point_definition': 'mean active foot collision geom centers, fallback ankle body origin',
                'baseline_foot_world_m': baseline.tolist(),
                'mean_foot_delta_for_basis_f_l_m': np.asarray(deltas).tolist(),
                'root_xy_sensitivity_m_per_rad_columns_f_l': sensitivity.tolist(),
                'actual_qpos_unchanged': live_unchanged,
                'basis_f': 'both hip_pitch +f and ankle_pitch -f',
                'basis_l': 'both hip_roll -l and ankle_roll +l'}
        self.fk_test_count += 1
        self.last_fk_test = test
        if self.first_fk_test is None:
            self.first_fk_test = test
        return sensitivity, test

    def apply(self, packet, actual_data, t):
        """Add bounded late-seat offsets to a copy of the raw nominal packet.

        NumPy arrays and Torch tensors of shape (nu,) or (1, nu) are supported.
        No clipping happens here. Already loaded actors keep their current
        offset, including exactly zero for an actor already sitting correctly.
        """
        t = float(t)
        if not np.isfinite(t) or (self.last_time is not None and t < self.last_time-1e-9):
            raise ValueError('Monotonically increasing finite feedback time required; create a new instance for a reset')
        dt = 0. if self.last_time is None else max(0., t-self.last_time)
        self.last_time = t
        tensor = hasattr(packet, 'detach') and hasattr(packet, 'clone')
        values = packet.detach().cpu().numpy() if tensor else np.asarray(packet)
        if values.shape not in ((self.model.nu,), (1, self.model.nu)) or not np.isfinite(values).all():
            raise ValueError('Expected one finite full position-target packet')
        if not np.issubdtype(values.dtype, np.floating):
            raise ValueError('Position target packet must use a floating dtype')
        result = packet.clone() if tensor else values.copy()
        active = t >= self.start_time
        u = float(np.clip((t-self.start_time)/self.window, 0., 1.))
        blend = u*u*(3.-2.*u)
        loads = self.seat_loads(actual_data)
        target_t = float(np.clip(t+self.times[0], self.times[0], self.times[-1]))
        # Use the present planned XY for safety while the last source segment
        # converges, and the fixed final Z to avoid lowering below its target.
        target_xy = np.asarray([[np.interp(target_t, self.times, self.reference['qpos'][:, actor, j])
                                 for j in range(2)] for actor in range(2)])
        if active:
            self.active_calls += 1
        for actor, part in enumerate(self.parts):
            rq = part['rq']
            root = np.asarray(actual_data.qpos[rq:rq+7])
            xy_error = float(np.linalg.norm(root[:2]-target_xy[actor]))
            quat = root[3:7]/max(float(np.linalg.norm(root[3:7])), 1e-12)
            rotation = np.empty(9)
            mj.mju_quat2Mat(rotation, quat)
            tilt = float(np.arccos(np.clip(rotation.reshape(3, 3)[2, 2], -1., 1.)))
            high = float(root[2]-self.final_root[actor, 2])
            old = float(self.offset[actor])
            old_center = self.center_offset[actor].copy()
            sensitivity = None
            centering_solution = None
            fk_test = None
            root_xy_correction = target_xy[actor]-root[:2]
            if not active:
                reason = 'before_final_seating_window'
            elif loads[actor] >= part['load_threshold_N']:
                self.latched[actor] = True
                if self.latch_time[actor] is None:
                    self.latch_time[actor] = t
                reason = 'actual_hip_seat_load_reached_hold_current_offset'
            elif self.latched[actor]:
                reason = 'previous_load_latched_hold_current_offset'
            elif xy_error > self.max_xy:
                reason = 'paused_root_xy_safety_gate'
            elif tilt > self.max_tilt:
                reason = 'paused_global_tilt_safety_gate'
            elif actor == 0 and xy_error >= self.center_xy_threshold:
                sensitivity, fk_test = self.stance_root_sensitivity(actual_data, actor)
                singular = np.linalg.svd(sensitivity, compute_uv=False)
                if singular[-1] < 1e-4 or singular[0]/singular[-1] > 1e4:
                    reason = 'paused_degenerate_private_stance_basis'
                else:
                    centering_solution = np.linalg.lstsq(sensitivity, root_xy_correction, rcond=1e-6)[0]
                    increment = np.clip(self.center_gain*centering_solution*dt*blend,
                                        -self.center_rate*dt*blend, self.center_rate*dt*blend)
                    self.center_offset[actor] = np.clip(self.center_offset[actor]+increment,
                                                       -self.center_max, self.center_max)
                    reason = 'private_fk_stance_basis_centering_before_squat'
            elif high <= self.height_margin:
                reason = 'root_not_above_final_height_margin'
            elif self.offset[actor] >= self.max_offset-1e-12:
                reason = 'maximum_offset_reached'
            else:
                self.offset[actor] = min(self.max_offset, self.offset[actor]+self.rate*dt*blend)
                reason = 'slow_symmetric_flexion_increment'
            for aid, factor, _ in part['changes']:
                delta = factor*float(self.offset[actor])
                if delta != 0.:
                    result[..., aid] = result[..., aid]+delta
            for basis_index, basis in enumerate(part['basis']):
                for aid, _, factor, _ in basis:
                    delta = factor*float(self.center_offset[actor, basis_index])
                    if delta != 0.:
                        result[..., aid] = result[..., aid]+delta
            self.records.append({'time_s': t, 'actor': actor+1, 'active': bool(active),
                                 'activation_blend': blend, 'dt_s': dt,
                                 'root_xy_error_m': xy_error, 'global_tilt_rad': tilt,
                                 'actual_root_z_m': float(root[2]), 'final_reference_root_z_m': float(self.final_root[actor, 2]),
                                 'root_above_final_z_m': high,
                                 'actual_hip_seat_upward_force_N': float(loads[actor]),
                                 'seat_load_threshold_N': part['load_threshold_N'],
                                 'load_latched': bool(self.latched[actor]),
                                 'offset_s_rad': float(self.offset[actor]),
                                 'offset_increment_s_rad': float(self.offset[actor]-old),
                                 'centering_offset_f_l_rad': self.center_offset[actor].tolist(),
                                 'centering_increment_f_l_rad': (self.center_offset[actor]-old_center).tolist(),
                                 'centering_root_xy_correction_m': root_xy_correction.tolist(),
                                 'centering_root_sensitivity_xy_m_per_rad': None if sensitivity is None else sensitivity.tolist(),
                                 'centering_unlimited_solution_f_l_rad': None if centering_solution is None else centering_solution.tolist(),
                                 'private_fk_actual_qpos_unchanged': None if fk_test is None else fk_test['actual_qpos_unchanged'],
                                 'knee_target_offset_rad': 2.*float(self.offset[actor]),
                                 'hip_pitch_target_offset_rad': -float(self.offset[actor]),
                                 'ankle_pitch_target_offset_rad': -float(self.offset[actor]),
                                 'reason': reason})
        return result

    def report(self):
        return {'method': 'late symmetric motor-position-target flexion with actual hip-seat load latch',
                'start_time_s': self.start_time, 'reference_end_time_s': self.end_time,
                'smooth_activation_window_s': self.window,
                'max_offset_s_rad': self.max_offset, 'max_offset_rate_rad_s': self.rate,
                'max_knee_offset_rad': 2.*self.max_offset, 'max_knee_offset_rate_rad_s': 2.*self.rate,
                'max_root_xy_error_m': self.max_xy, 'max_global_tilt_rad': self.max_tilt,
                'root_height_margin_m': self.height_margin, 'seat_load_fraction_of_weight': self.load_fraction,
                'final_offset_s_rad': self.offset.tolist(), 'load_latched': self.latched.tolist(),
                'centering': {'enabled_actor_indices_one_based': [1],
                              'prior_to_squat_until_root_xy_error_below_m': self.center_xy_threshold,
                              'max_each_basis_offset_rad': self.center_max,
                              'max_each_basis_rate_rad_s': self.center_rate,
                              'integral_gain_s_inv': self.center_gain,
                              'numeric_fk_epsilon_rad': self.fk_epsilon,
                              'final_basis_f_l_offsets_rad': self.center_offset.tolist(),
                              'direction': 'root sensitivity = negative mean world foot delta per numerical joint-basis perturbation; solve observed XY correction in that measured basis',
                              'pause_if': 'root XY > 0.35 m or global tilt > 0.5 rad; both all offsets hold after actual hip-seat load latch'},
                'load_latch_time_s': self.latch_time, 'active_calls': self.active_calls,
                'record_count': len(self.records),
                'actors': [{'prefix': p['prefix'], 'mass_kg': p['mass_kg'], 'weight_N': p['weight_N'],
                            'load_threshold_N': p['load_threshold_N'],
                            'modified_joint_names': [name for _, _, name in p['changes']]}
                           for p in self.parts],
                'seat_geom_names': [self.geom_names[i] for i in sorted(self.seat_ids)],
                'packet_contract': 'Input/output raw nominal position targets before physical clipping; caller records effective raw targets in previous-action history then applies normal bounds/delay.',
                'force_provenance': 'Actual latest MuJoCo solver contact forces on hip/pelvis seat contacts; no new applied force and no live-data recomputation.',
                'root_state_writes': 0, 'qpos_or_qvel_writes': 0, 'external_wrench_writes': 0,
                'equality_or_model_edits': 0, 'direct_lateral_root_target_correction': False,
                'stance_basis_motor_centering': True,
                'tests': {'sourcefork': {'private_data_object': True,
                                         'actual_qpos_written': False,
                                         'private_fk_test_count': self.fk_test_count,
                                         'first_direction_test': self.first_fk_test,
                                         'last_direction_test': self.last_fk_test}},
                'task_success_claimed': False,
                'limitations': ['A 10%-weight load latch only stops further flexion; it does not prove stable seated completion.',
                               'All full-task geometry, stability, force and trajectory checks remain the evaluator responsibility.',
                               'Offsets hold after load loss; feedback does not grow further after a load latch.',
                               'Numerical stance sensitivity is a local fixed-foot approximation; successful centering and seated stability require a physical rollout.']}
