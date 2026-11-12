"""Honest free-root dynamics for the two-person seat-exchange reference.

Only motor controls change after initialization. SONIC inference uses separate
non-integrated mirrors; the shared plant is never pose-written or force-assisted.
The frozen binary plant and complete integration state support control-only replay.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from rgb_near_physics import mapping, preserve_passive_hand_channels

ROOT = Path(__file__).resolve().parents[1]
SOURCE_BYTES = Path(__file__).read_bytes()
SOURCE_SHA256 = hashlib.sha256(SOURCE_BYTES).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    def finite(v):
        if isinstance(v, dict):
            return {k: finite(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [finite(x) for x in v]
        if isinstance(v, float) and not np.isfinite(v):
            return None
        return v
    Path(path).write_text(json.dumps(finite(value), ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')


def freeze_sources(output, args):
    """Preserve reviewable source bytes before controller construction/rollout."""
    import sys
    destination = output/'source_snapshot'
    destination.mkdir()
    files = []
    for name in ('two_sitting_physics.py', 'two_sitting_feedback.py',
                 'fourth_sonic_controller.py', 'rgb_near_task_feedback.py',
                 'rgb_near_physics.py', 'two_sitting_retarget.py', 'two_sitting_assets.py',
                 'two_sitting_lean.py', 'two_sitting_reference_foot_feedback.py'):
        path = ROOT/'scripts'/name
        if name == 'two_sitting_physics.py':
            payload = SOURCE_BYTES
        else:
            module = sys.modules.get(path.stem)
            payload = getattr(module, 'SOURCE_BYTES', None) if module is not None else None
            if payload is None:
                payload = path.read_bytes()
        saved = destination/name
        saved.write_bytes(payload)
        files.append({'source_path': str(path), 'snapshot_path': str(saved),
                      'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()})
    manifest = destination/'manifest.json'
    write_json(manifest, {'files': files, 'arguments': vars(args),
                          'timing': 'Frozen before controller construction and physical integration.'})
    return {'files': files, 'manifest_path': str(manifest), 'manifest_sha256': sha(manifest)}


def interp(reference, elapsed):
    t = float(reference['time_s'][0]) + float(elapsed)
    source = np.asarray(reference['qpos'], dtype=float)
    result = np.empty((2, 59))
    for actor in range(2):
        for j in range(59):
            result[actor, j] = np.interp(t, reference['time_s'], source[:, actor, j])
        # q and -q encode the same rotation. Evaluators can pass raw NPZ data,
        # so handle an antipodal segment here as well as in controller loaders.
        # Preserve the previous arithmetic on already continuous references.
        times = np.asarray(reference['time_s'])
        right = int(np.clip(np.searchsorted(times, t, side='right'), 1, len(times)-1))
        left = right-1
        if np.dot(source[left, actor, 3:7], source[right, actor, 3:7]) < 0:
            for j in range(3, 7):
                result[actor, j] = np.interp(t, times[left:right+1],
                                           [source[left, actor, j], -source[right, actor, j]])
        result[actor, 3:7] /= np.linalg.norm(result[actor, 3:7])
    return result


class Motors:
    def __init__(self, model, proxy_path, reference, mode, config, gravity_comp,
                 root_feedback=False, leg_feedback=False):
        self.model, self.reference = model, reference
        self.names = list(map(str, reference['dof_names']))
        self.parts, self.mode, self.gravity_comp = [], mode, gravity_comp
        self.clipped = 0
        self.policy_records = []
        for actor, prefix in enumerate(('p1_', 'p2_')):
            sm = mapping(model, self.names, prefix)
            part = {'sm': sm, 'controller': None, 'prefix': prefix,
                    'reference': reference['qpos'][:, actor]}
            if mode != 'pd':
                from fourth_sonic_controller import Controller
                proxy = mujoco.MjModel.from_xml_path(str(Path(proxy_path).resolve()))
                local_ref = {'qpos': reference['qpos'][:, actor],
                             'time_s': reference['time_s'],
                             'dof_names': reference['dof_names']}
                controller = Controller(proxy, local_ref, config)
                preserve_passive_hand_channels(proxy)
                pm = mapping(proxy, self.names)
                for key in ('actuator_gainprm', 'actuator_biasprm', 'actuator_dynprm',
                            'actuator_gaintype', 'actuator_biastype', 'actuator_dyntype',
                            'actuator_ctrllimited', 'actuator_ctrlrange',
                            'actuator_forcelimited', 'actuator_forcerange', 'actuator_gear'):
                    getattr(model, key)[sm['aids']] = getattr(proxy, key)[pm['aids']]
                for key in ('dof_armature', 'dof_frictionloss', 'dof_damping'):
                    getattr(model, key)[sm['va']] = getattr(proxy, key)[pm['va']]
                for key in ('jnt_range', 'jnt_limited'):
                    getattr(model, key)[sm['jids']] = getattr(proxy, key)[pm['jids']]
                part.update(proxy=proxy, data=mujoco.MjData(proxy), pm=pm,
                            controller=controller)
            self.parts.append(part)
        self.passive = preserve_passive_hand_channels(model)
        self.lean = None
        self.reference_foot = None
        self.root_feedback_scale = float(config.get('root_feedback_scale', 1.))
        if config.get('lean_feedback', False):
            from two_sitting_lean import ReferenceLeanController
            self.lean = ReferenceLeanController(reference, config)
        if config.get('reference_foot_feedback', False):
            from two_sitting_reference_foot_feedback import ReferenceFootFeedback
            self.reference_foot = ReferenceFootFeedback(reference, config)
        self.inactive = ((model.actuator_gainprm[:, 0] == 0) &
                         np.all(model.actuator_biasprm == 0, axis=1))
        self.root_feedback = self.leg_feedback = None
        if root_feedback or leg_feedback:
            from rgb_near_task_feedback import WorldLegFeedback
            from two_sitting_feedback import RootAccelerationFeedback
            if leg_feedback:
                self.leg_feedback = WorldLegFeedback(model, reference, self.parts, gravity_comp)
            if root_feedback:
                self.root_feedback = RootAccelerationFeedback(model, reference, self.parts,
                                                               objective='root_feet')

    def sync(self, part, data):
        pd, pm, sm = part['data'], part['pm'], part['sm']
        pd.qpos[pm['rq']:pm['rq']+7] = data.qpos[sm['rq']:sm['rq']+7]
        pd.qvel[pm['rv']:pm['rv']+6] = data.qvel[sm['rv']:sm['rv']+6]
        pd.qpos[pm['qa']] = data.qpos[sm['qa']]
        pd.qvel[pm['va']] = data.qvel[sm['va']]
        pd.time = data.time
        mujoco.mj_forward(part['proxy'], pd)

    def reset(self, data):
        for part in self.parts:
            if part['controller'] is not None:
                self.sync(part, data)
                part['controller'].reset(part['data'])

    def action(self, data, elapsed):
        desired = interp(self.reference, elapsed)
        result = np.zeros(self.model.nu)
        raw_action, raw_target, before_feedback = [], [], []
        for actor, part in enumerate(self.parts):
            sm = part['sm']
            if self.mode == 'pd':
                target = desired[actor, 7:].copy()
            else:
                self.sync(part, data)
                if self.lean is not None:
                    self.lean.apply(part, data, elapsed)
                if self.reference_foot is not None:
                    self.reference_foot.apply(part, data, elapsed)
                packet = part['controller'].action(part['data'], elapsed)
                target = packet[part['pm']['aids']].copy()
                controller = part['controller']
                raw_target.append(controller.target.copy())
                raw_action.append(((controller.target-controller.default)/controller.scale).copy())
            if self.gravity_comp:
                kp = self.model.actuator_gainprm[sm['aids'], 0]
                correction = self.gravity_comp * data.qfrc_bias[sm['va']] / np.maximum(kp, 1e-9)
                target += np.clip(correction, -.2, .2)
            limits = self.model.jnt_range[sm['jids']]
            limited = self.model.jnt_limited[sm['jids']].astype(bool)
            bounded = target.copy()
            bounded[limited] = np.clip(target[limited], limits[limited, 0], limits[limited, 1])
            self.clipped += int(np.sum(np.abs(target-bounded) > 1e-10))
            result[sm['aids']] = bounded
            before_feedback.append(bounded.copy())
        reference_t = elapsed + float(self.reference['time_s'][0])
        if self.leg_feedback is not None:
            result = self.leg_feedback.apply(data, result, reference_t)
        if self.root_feedback is not None:
            nominal = result.copy()
            result = nominal + self.root_feedback_scale*(self.root_feedback.apply(data, result, reference_t)-nominal)
        result[self.inactive] = 0.
        if self.mode != 'pd':
            effective = []
            hybrid = bool(self.gravity_comp or self.root_feedback is not None or self.leg_feedback is not None)
            for part in self.parts:
                controller, sm = part['controller'], part['sm']
                normalized = (result[sm['aids']][controller.jref]-controller.default)/controller.scale
                effective.append(normalized.copy())
                if hybrid:
                    if hasattr(controller, 'last'):
                        controller.last = normalized.copy()
                    else:
                        controller.previous_action = normalized.copy()
            self.policy_records.append({'time_s': float(elapsed),
                                        'raw_normalized_learned_target': np.asarray(raw_action),
                                        'raw_undelayed_learned_target_rad': np.asarray(raw_target),
                                        'motor_target_before_feedback_rad': np.asarray(before_feedback),
                                        'effective_normalized_motor_target': np.asarray(effective),
                                        'applied_motor_target_rad': np.asarray([result[p['sm']['aids']] for p in self.parts])})
        return result

    def report(self):
        return {'mode': self.mode,
                'actors': [p['controller'].report() if p['controller'] is not None
                           else {'controller': 'reference position PD; no learned policy'}
                           for p in self.parts],
                'gravity_compensation_fraction': self.gravity_comp,
                'gravity_compensation_method': 'bounded g(q)/kp motor-target offset; no applied external force',
                'joint_limit_clipped_target_values': self.clipped,
                'passive_follower_actuator_ids': self.passive,
                'root_feedback': self.root_feedback.report() if self.root_feedback is not None else None,
                'root_feedback_output_scale': self.root_feedback_scale,
                'lean_reference_feedback': self.lean.report() if self.lean is not None else None,
                'reference_foot_feedback': self.reference_foot.report() if self.reference_foot is not None else None,
                'leg_feedback': self.leg_feedback.report() if self.leg_feedback is not None else None,
                'feedback_disclosure': 'Optional actual-state motor-target corrections follow SONIC. Leg feedback replaces 24 leg targets; root feedback adds bounded hip/knee/ankle/waist offsets. Learned inference still runs, but those final controls are hybrid.',
                'hybrid_action_history': 'When feedback or gravity compensation is enabled, last-action observation uses normalized actual applied body targets. Raw undelayed learned targets and final effective targets are retained separately in trace.',
                'raw_action_definition': 'Normalized (raw undelayed target-default)/scale. For tracking this includes the reference plus learned residual, not the residual alone.',
                'hybrid_feedback_control_delay_physics_steps': 0,
                'proxy_method': 'Mirror actual root and joints in inference-only models; never integrate proxy or copy proxy state to plant.'}


def contacts(model, data, step, geom_names, body_names):
    rows, seat, feet = [], np.zeros(2, bool), np.zeros(2, bool)
    max_pen, interperson_pen = 0., 0.
    seat_force = np.zeros(2)
    for i in range(data.ncon):
        c = data.contact[i]
        gs = [int(c.geom1), int(c.geom2)]
        bs = [int(model.geom_bodyid[g]) for g in gs]
        gn, bn = [geom_names[g] for g in gs], [body_names[b] for b in bs]
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, wrench)
        frame = np.asarray(c.frame).reshape(3, 3)
        force, torque = frame.T @ wrench[:3], frame.T @ wrench[3:]
        normal, pen = max(0., float(wrench[0])), max(0., -float(c.dist))
        max_pen = max(max_pen, pen)
        if ((bn[0].startswith('p1_') and bn[1].startswith('p2_')) or
                (bn[0].startswith('p2_') and bn[1].startswith('p1_'))):
            interperson_pen = max(interperson_pen, pen)
        for actor, prefix in enumerate(('p1_', 'p2_')):
            for robot_side in range(2):
                other = 1-robot_side
                if not bn[robot_side].startswith(prefix):
                    continue
                on_robot = force if robot_side == 1 else -force
                support_geom = gn[other].lower()
                pelvis = any(x in bn[robot_side] for x in ('base_link', 'hip_', 'pelvis', 'thigh'))
                if (('seat' in support_geom or 'stool' in support_geom) and
                        pelvis and on_robot[2] > .5 and normal > .5):
                    seat[actor] = True
                    seat_force[actor] += max(0., float(on_robot[2]))
                if (('ground' in support_geom or 'floor' in support_geom) and
                        any(x in bn[robot_side] for x in ('ankle', 'foot')) and on_robot[2] > .5):
                    feet[actor] = True
        rows.append((step, float(data.time-model.opt.timestep), gs, bs,
                     np.asarray(c.pos).copy(), force, torque, normal, float(c.dist)))
    return rows, seat, feet, seat_force, max_pen, interperson_pen


def independent_replay(binary, initial_state, state_spec, controls, qpos, qvel):
    if not (np.isfinite(qpos).all() and np.isfinite(qvel).all() and np.isfinite(controls).all()):
        return {'passed': False, 'reason': 'nonfinite rollout state or control',
                'steps': len(controls)}
    model = mujoco.MjModel.from_binary_path(str(binary))
    data = mujoco.MjData(model)
    mujoco.mj_setState(model, data, initial_state, state_spec)
    mujoco.mj_forward(model, data)
    qerr, verr = float(np.max(np.abs(data.qpos-qpos[0]))), float(np.max(np.abs(data.qvel-qvel[0])))
    for i, ctrl in enumerate(controls):
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)
        qerr = max(qerr, float(np.max(np.abs(data.qpos-qpos[i+1]))))
        verr = max(verr, float(np.max(np.abs(data.qvel-qvel[i+1]))))
    return {'passed': bool(qerr <= 1e-9 and verr <= 1e-9),
            'max_qpos_absolute_error': qerr, 'max_qvel_absolute_error': verr,
            'steps': len(controls),
            'method': 'New MjModel from frozen MJB + exact mjSTATE_INTEGRATION; only recorded ctrl and mj_step.'}


def run(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any((output/n).exists() for n in ('trace.npz', 'report.json', 'model.mjb')):
        raise FileExistsError('Use a new output directory; refusing to overwrite trial evidence.')
    implementation_snapshot = freeze_sources(output, args)
    reference = dict(np.load(args.reference, allow_pickle=False))
    reference['qpos'] = np.asarray(reference['qpos'], dtype=float).copy()
    reference['time_s'] = np.asarray(reference['time_s'], dtype=float)
    times, qref = reference['time_s'], reference['qpos']
    if (qref.shape != (len(times), 2, 59) or len(times) < 2 or
            np.any(np.diff(times) <= 0) or not np.isfinite(qref).all() or
            not np.isfinite(times).all() or len(reference['dof_names']) != 52):
        raise ValueError('Expected finite qpos[T,2,59], increasing time_s[T], dof_names[52].')
    for actor in range(2):
        for i in range(len(times)):
            norm = np.linalg.norm(qref[i, actor, 3:7])
            if norm < .9 or norm > 1.1:
                raise ValueError('Invalid reference root quaternion.')
            qref[i, actor, 3:7] /= norm
            if i and np.dot(qref[i-1, actor, 3:7], qref[i, actor, 3:7]) < 0:
                qref[i, actor, 3:7] *= -1
    model = mujoco.MjModel.from_xml_path(str(Path(args.model).resolve()))
    if not (np.linalg.norm(model.opt.gravity[:2]) < 1e-8 and model.opt.gravity[2] < -1.):
        raise ValueError('Expected a Z-up plant with real downward gravity.')
    if args.hold_last_seconds < 0:
        raise ValueError('--hold-last-seconds cannot be negative.')
    config = json.loads(Path(args.config).read_text(encoding='utf-8')) if args.config else {}
    config.update(mode=args.mode, actuation='position')
    config.update(root_feedback_scale=args.root_feedback_scale, lean_feedback=args.lean_feedback,
                  max_lean=args.max_lean, reference_foot_feedback=args.reference_foot_feedback)
    if args.mode == 'newft15225':
        checkpoint = ROOT/'data/vita_demo_20261026/checkpoints/new_ft15225/model_step_015225.pt'
        expected = '0a825e57cb9afe7cda15dd3103e1b2715aeec2f6d7e8fc0cedf529044bebc6c1'
        if sha(checkpoint) != expected:
            raise ValueError('Expected original fixed new_ft15225 checkpoint hash.')
        config['checkpoint'] = str(checkpoint)
        config['device'] = args.device
        config['hand_object_feedback'] = False
    motor = Motors(model, args.proxy_model, reference, args.mode, config, args.gravity_comp,
                   args.root_feedback, args.leg_feedback)
    roots = [p['sm']['root'] for p in motor.parts]
    if model.nq != 118 or model.nv != 116 or model.nu != 104:
        raise ValueError('Expected exactly two free-root Vita52 actors: nq118/nv116/nu104.')
    if np.any(model.actuator_trntype != mujoco.mjtTrn.mjTRN_JOINT):
        raise ValueError('Only direct joint motor transmissions are allowed.')
    if any(int(j) in roots for j in model.actuator_trnid[:, 0]):
        raise ValueError('Root actuation is forbidden.')
    if any(model.eq_type != mujoco.mjtEq.mjEQ_JOINT):
        raise ValueError('Body weld/connect attachments are forbidden; only joint coupling allowed.')
    if model.nmocap:
        raise ValueError('Mocap bodies are forbidden in the physical plant.')
    binary = output/'model.mjb'
    mujoco.mj_saveModel(model, str(binary), None)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    for actor, part in enumerate(motor.parts):
        sm = part['sm']
        data.qpos[sm['rq']:sm['rq']+7] = qref[0, actor, :7]
        data.qpos[sm['qa']] = qref[0, actor, 7:]
    data.qvel[:] = 0.
    mujoco.mj_forward(model, data)
    motor.reset(data)
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    initial = np.empty(mujoco.mj_stateSize(model, spec))
    mujoco.mj_getState(model, data, initial, spec)
    np.savez_compressed(output/'initial_state.npz', state=initial, state_spec=int(spec),
                        qpos=data.qpos.copy(), qvel=data.qvel.copy())
    geom_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or str(i)
                  for i in range(model.ngeom)]
    body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or str(i)
                  for i in range(model.nbody)]
    duration = float(times[-1]-times[0])
    requested = duration+args.hold_last_seconds if args.seconds is None else float(args.seconds)
    if requested <= 0:
        raise ValueError('--seconds must be positive.')
    dt = float(model.opt.timestep)
    steps = int(np.ceil(requested/dt-1e-10))
    all_q, all_v, state_time = [data.qpos.copy()], [data.qvel.copy()], [float(data.time)]
    all_ctrl, all_force, all_torque, all_contact = [], [], [], []
    seats, feet, seat_forces, penetrations, interpen = [], [], [], [], []
    termination, wall_start = 'completed', time.time()
    relevant_warnings = [int(getattr(mujoco.mjtWarning, 'mjWARN_'+n))
                         for n in ('BADQPOS', 'BADQVEL', 'BADQACC', 'BADCTRL')]
    initial_warning = np.asarray([data.warning[i].number for i in relevant_warnings])
    for step in range(steps):
        ctrl = motor.action(data, float(data.time))
        if not np.isfinite(ctrl).all():
            termination = 'nonfinite_controller_output'
            break
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)
        rows, seat, foot, sf, pen, ipen = contacts(model, data, step, geom_names, body_names)
        all_contact.extend(rows)
        all_ctrl.append(ctrl.copy())
        all_force.append(data.actuator_force.copy())
        all_torque.append(data.qfrc_actuator.copy())
        all_q.append(data.qpos.copy())
        all_v.append(data.qvel.copy())
        state_time.append(float(data.time))
        seats.append(seat); feet.append(foot); seat_forces.append(sf)
        penetrations.append(pen); interpen.append(ipen)
        if (np.any(data.qfrc_applied) or np.any(data.xfrc_applied)):
            raise AssertionError('Unexpected external force in plant.')
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            termination = 'nonfinite_physics_state'; break
        if any(data.qpos[p['sm']['rq']+2] < .15 for p in motor.parts):
            termination = 'root_below_0p15m'; break
        if np.any(np.asarray([data.warning[i].number for i in relevant_warnings]) > initial_warning):
            termination = 'physics_numeric_warning'; break
        if (step+1) % 400 == 0:
            print(json.dumps({'step': step+1, 'time_s': float(data.time),
                              'requested_s': requested}), flush=True)
    qpos, qvel = np.asarray(all_q), np.asarray(all_v)
    ctrl = np.asarray(all_ctrl).reshape((-1, model.nu))
    force = np.asarray(all_force).reshape((-1, model.nu))
    torque = np.asarray(all_torque).reshape((-1, model.nv))
    st = np.asarray(state_time)
    desired = np.asarray([interp(reference, t) for t in st])
    source_track = reference.get('source_time_s', reference.get('source_pts_seconds', reference.get('pts_seconds', times)))
    source_track = np.asarray(source_track, dtype=float)
    if source_track.shape != times.shape:
        raise ValueError('source_time_s/pts_seconds must align with reference frames.')
    source_time = np.interp(st+times[0], times, source_track)
    xy, zerr, ori = np.zeros((len(st), 2)), np.zeros((len(st), 2)), np.zeros((len(st), 2))
    rpy = np.zeros((len(st), 2, 3))
    body_rms = np.zeros((len(st), 2))
    body_mask = np.asarray([not n.startswith(('lh_', 'rh_', 'neck_')) for n in motor.names])
    for a, part in enumerate(motor.parts):
        rq = part['sm']['rq']
        actual = qpos[:, rq:rq+7]
        xy[:, a] = np.linalg.norm(actual[:, :2]-desired[:, a, :2], axis=1)
        zerr[:, a] = np.abs(actual[:, 2]-desired[:, a, 2])
        dot = np.abs(np.sum(actual[:, 3:7]*desired[:, a, 3:7], axis=1))
        ori[:, a] = 2*np.arccos(np.clip(dot, 0., 1.))
        for i in range(len(st)):
            inverse = desired[i, a, 3:7].copy(); inverse[1:] *= -1
            relative = np.empty(4)
            mujoco.mju_mulQuat(relative, inverse, actual[i, 3:7])
            relative /= np.linalg.norm(relative)
            w, x, y, z = relative
            rpy[i, a] = [np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y)),
                         np.arcsin(np.clip(2*(w*y-z*x), -1., 1.)),
                         np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))]
        difference = qpos[:, part['sm']['qa'][body_mask]]-desired[:, a, 7:][:, body_mask]
        body_rms[:, a] = np.sqrt(np.mean(difference**2, axis=1))
    tail = np.flatnonzero(st[1:] >= max(duration, requested-.2)-1e-9)
    seated = np.asarray(seats, dtype=bool).reshape((-1, 2))
    final_fraction = np.mean(seated[tail], axis=0) if len(tail) else np.zeros(2)
    replay = independent_replay(binary, initial, spec, ctrl, qpos, qvel)
    checks = {'full_source_duration': bool(st[-1]+1e-8 >= duration),
              'completed_requested_duration': bool(termination == 'completed' and len(ctrl) == steps),
              'root_xy_error_below_0p25m': bool(np.max(xy) < .25),
              'root_z_error_below_0p18m': bool(np.max(zerr) < .18),
              'root_orientation_error_below_0p65rad': bool(np.max(ori) < .65),
              'penetration_below_0p025m': bool(max(penetrations, default=0.) < .025),
              'final_both_people_have_true_seat_contact': bool(len(seated) and np.all(seated[-1]) and np.all(final_fraction >= .5)),
              'independent_control_only_replay': replay['passed'],
              'no_state_overrides_or_external_forces': True}
    fields = {
        'time': st, 'qpos': qpos, 'qvel': qvel, 'ctrl': ctrl,
        'actuator_force': force, 'qfrc_actuator': torque,
        'source_time_s': source_time, 'reference_qpos': desired,
        'joint_names': np.asarray(motor.names),
        'joint_qadr': np.asarray([p['sm']['qa'] for p in motor.parts]),
        'joint_vadr': np.asarray([p['sm']['va'] for p in motor.parts]),
        'joint_actuator_ids': np.asarray([p['sm']['aids'] for p in motor.parts]),
        'robot_root_qadr': np.asarray([p['sm']['rq'] for p in motor.parts]),
        'robot_root_vadr': np.asarray([p['sm']['rv'] for p in motor.parts]),
        'geom_names': np.asarray(geom_names), 'body_names': np.asarray(body_names),
        'seat_contact': seated, 'foot_contact': np.asarray(feet, bool).reshape((-1, 2)),
        'seat_support_force_N': np.asarray(seat_forces).reshape((-1, 2)),
        'max_penetration_m': np.asarray(penetrations),
        'interperson_penetration_m': np.asarray(interpen),
        'root_xy_error_m': xy, 'root_z_error_m': zerr, 'root_orientation_error_rad': ori,
        'root_rpy_error_rad': rpy, 'body_joint_rms_error_rad': body_rms,
        'body_joint_rms_names': np.asarray(motor.names)[body_mask],
    }
    contact_fields = [
        ('interval_index', 0, (), np.int64), ('time_s', 1, (), float),
        ('geom_ids', 2, (2,), np.int64), ('body_ids', 3, (2,), np.int64),
        ('position_world_m', 4, (3,), float), ('force_world_on_geom2_N', 5, (3,), float),
        ('torque_world_on_geom2_Nm', 6, (3,), float),
        ('normal_force_N', 7, (), float), ('signed_distance_m', 8, (), float)]
    for name, i, shape, dtype in contact_fields:
        fields['contact_'+name] = np.asarray([r[i] for r in all_contact], dtype=dtype).reshape((-1,)+shape)
    for label, feedback in [('root_feedback', motor.root_feedback), ('leg_feedback', motor.leg_feedback)]:
        if feedback is not None and feedback.records:
            for key in feedback.records[0]:
                values = np.asarray([row[key] for row in feedback.records])
                if values.dtype.hasobject:
                    raise ValueError('Feedback log cannot contain object arrays: '+key)
                fields[label+'_'+key] = values
    if motor.policy_records:
        for key in motor.policy_records[0]:
            fields['policy_'+key] = np.asarray([row[key] for row in motor.policy_records])
    np.savez_compressed(output/'trace.npz', **fields)
    report = {'success': bool(all(checks.values())),
              'status': 'passed' if all(checks.values()) else 'failed_physical_quality',
              'termination': termination, 'checks': checks,
              'requested_seconds': requested, 'source_duration_seconds': duration,
              'requested_final_hold_seconds': args.hold_last_seconds,
              'actual_reference_time_range_s': [float(times[0]), float(times[-1])],
              'actual_source_time_range_s': [float(source_track[0]), float(source_track[-1])],
              'actual_final_reference_hold_seconds': max(0., float(st[-1])-duration),
              'simulated_seconds': float(st[-1]), 'physics_steps': len(ctrl),
              'physics_dt_s': dt, 'wall_seconds': time.time()-wall_start,
              'max_root_xy_error_m': np.max(xy, axis=0).tolist(),
              'max_root_z_error_m': np.max(zerr, axis=0).tolist(),
              'max_root_orientation_error_rad': np.max(ori, axis=0).tolist(),
              'max_root_roll_error_rad': np.max(np.abs(rpy[:, :, 0]), axis=0).tolist(),
              'max_root_pitch_error_rad': np.max(np.abs(rpy[:, :, 1]), axis=0).tolist(),
              'max_root_yaw_error_rad': np.max(np.abs(rpy[:, :, 2]), axis=0).tolist(),
              'max_body_joint_rms_error_rad': np.max(body_rms, axis=0).tolist(),
              'body_joint_rms_names': np.asarray(motor.names)[body_mask].tolist(),
              'root_rpy_convention': 'XYZ roll/pitch/yaw of inverse(reference rotation)*actual rotation; radians',
              'max_penetration_m': max(penetrations, default=0.),
              'max_interperson_penetration_m': max(interpen, default=0.),
              'final_0p2s_seat_contact_fraction': final_fraction.tolist(),
              'contact_records': len(all_contact),
              'controller': motor.report(), 'independent_replay': replay,
              'state_override_after_reset_count': 0, 'external_force_steps': 0,
              'root_actuators': 0, 'mocap_bodies': model.nmocap,
              'model_dimensions': {'nq': model.nq, 'nv': model.nv, 'nu': model.nu},
              'artifacts': {n: {'path': str(output/n), 'sha256': sha(output/n)}
                            for n in ('model.mjb', 'initial_state.npz', 'trace.npz')},
              'inputs': {'model': str(Path(args.model).resolve()), 'model_sha256': sha(args.model),
                         'reference': str(Path(args.reference).resolve()), 'reference_sha256': sha(args.reference)},
              'implementation': {'physics_script': str(Path(__file__).resolve()), 'physics_script_sha256': SOURCE_SHA256,
                                 'source_snapshot': implementation_snapshot},
              'contact_convention': 'Contact frame rows are axes in world coordinates. World force/torque acts on geom2; negate for geom1. Contact sample precedes integration endpoint by dt.',
              'limitations': ['Reconstructed source and engineered collision/material assumptions require independent review.',
                              'Passing these thresholds is one-trajectory simulation acceptance, not a hardware or generalization claim.']}
    write_json(output/'report.json', report)
    print(json.dumps({'report': str(output/'report.json'), 'success': report['success'],
                      'termination': termination, 'simulated_seconds': report['simulated_seconds']}, ensure_ascii=False), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--proxy-model', required=True)
    p.add_argument('--reference', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--mode', choices=('tracking', 'newft15225', 'pd'), default='tracking')
    p.add_argument('--config')
    p.add_argument('--device', default='cpu')
    p.add_argument('--seconds', type=float)
    p.add_argument('--hold-last-seconds', type=float, default=1.)
    p.add_argument('--gravity-comp', type=float, default=0.)
    p.add_argument('--root-feedback', action='store_true')
    p.add_argument('--root-feedback-scale', type=float, default=1.)
    p.add_argument('--lean-feedback', action='store_true')
    p.add_argument('--max-lean', type=float, default=.1)
    p.add_argument('--reference-foot-feedback', action='store_true')
    p.add_argument('--leg-feedback', action='store_true')
    run(p.parse_args())


if __name__ == '__main__':
    main()
