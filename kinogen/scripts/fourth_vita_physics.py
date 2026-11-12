#!/usr/bin/env python3
"""Physical Vita52 reference tracking with a free rigid object in MuJoCo.

The model and reference are separate audited inputs. qpos/qvel and the object
pose are assigned once, at reset. During rollout only position-actuator ctrl is
written; all following states arise from mj_step. This is an explicit PD
baseline, not the supplied SONIC policy or a claim of calibrated physics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def receipt(path: Path) -> dict:
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size,
            'sha256': sha(path)}


def atomic_json(path: Path, data: object) -> None:
    temp = path.with_name(path.name + '.partial')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def name_of(mujoco, model, kind, index: int) -> str:
    return mujoco.mj_id2name(model, kind, int(index)) or ''


def model_mapping(mujoco, model, names: list[str], object_id: int):
    jkind = mujoco.mjtObj.mjOBJ_JOINT
    akind = mujoco.mjtObj.mjOBJ_ACTUATOR
    bodykind = mujoco.mjtObj.mjOBJ_BODY
    robot_free_id = mujoco.mj_name2id(model, jkind, 'vita_demo_free_root')
    obj_name = f'target_object_{object_id:02d}'
    obj_body_id = mujoco.mj_name2id(model, bodykind, obj_name)
    obj_free_id = mujoco.mj_name2id(model, jkind, obj_name + '_free')
    if min(robot_free_id, obj_body_id, obj_free_id) < 0:
        raise ValueError(f'Model must contain Vita root and dynamic {obj_name} freejoint')
    robot_qadr = int(model.jnt_qposadr[robot_free_id])
    obj_qadr = int(model.jnt_qposadr[obj_free_id])
    if robot_qadr != 0 or obj_qadr < 59 or model.jnt_type[robot_free_id] != mujoco.mjtJoint.mjJNT_FREE or model.jnt_type[obj_free_id] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError('Unexpected freejoint address or type')
    qadr, vadr, aids = [], [], []
    for n in names:
        jid = mujoco.mj_name2id(model, jkind, n)
        aid = mujoco.mj_name2id(model, akind, 'motor_' + n)
        if jid < 0 or aid < 0:
            raise ValueError(f'Missing actuated source joint {n}')
        if int(model.actuator_trnid[aid, 0]) != jid:
            raise ValueError(f'Actuator transmission mismatch for {n}')
        qadr.append(int(model.jnt_qposadr[jid]))
        vadr.append(int(model.jnt_dofadr[jid]))
        aids.append(aid)
    if len(set(qadr)) != 52 or len(set(aids)) != 52 or model.nu != 52:
        raise ValueError('Expected a one-to-one 52-joint position actuator mapping')
    return {'robot_qadr': robot_qadr, 'object_qadr': obj_qadr,
            'object_vadr': int(model.jnt_dofadr[obj_free_id]),
            'object_body_id': obj_body_id, 'object_body_name': obj_name,
            'qadr': qadr, 'vadr': vadr, 'actuator_ids': aids}


def contact_snapshot(mujoco, model, data, object_body_id: int) -> dict:
    import numpy as np
    rows = []
    max_pen = 0.0
    obj_touch_hands = 0
    obj_touch_table = 0
    obj_hand_normal_N = 0.0
    for ci in range(data.ncon):
        c = data.contact[ci]
        geom_ids = [int(c.geom1), int(c.geom2)]
        bodies = [int(model.geom_bodyid[g]) for g in geom_ids]
        bn = [name_of(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, b) or 'world' for b in bodies]
        gn = [name_of(mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in geom_ids]
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, ci, wrench)
        frame = np.asarray(c.frame).reshape(3, 3)
        force_world = frame.T @ wrench[:3]
        torque_world = frame.T @ wrench[3:]
        normal_N = max(float(wrench[0]), 0.0)
        depth = max(-float(c.dist), 0.0)
        max_pen = max(max_pen, depth)
        is_obj = object_body_id in bodies
        is_hand = any(n.startswith(('lh_', 'rh_')) for n in bn)
        is_table = 'table_support' in gn
        if is_obj and is_hand:
            obj_touch_hands += 1
            obj_hand_normal_N += normal_N
        if is_obj and is_table:
            obj_touch_table += 1
        rows.append({'geom_ids': geom_ids, 'body_ids': bodies,
                     'geom_names': gn, 'body_names': bn,
                     'contact_pos_world_m': np.asarray(c.pos).tolist(),
                     'contact_frame_row_major': frame.tolist(),
                     'friction': np.asarray(c.friction).tolist(),
                     'force_world_on_geom2_N': force_world.tolist(),
                     'torque_world_on_geom2_Nm': torque_world.tolist(),
                     'solver_contact_time_s': max(float(data.time - model.opt.timestep), 0.0),
                     'normal_force_N': normal_N,
                     'contact_distance_m': float(c.dist),
                     'force_contact_frame_N_Nm': [float(x) for x in wrench]})
    return {'pairs': rows, 'max_penetration_m': max_pen,
            'object_hand_contact_pairs': obj_touch_hands,
            'object_table_contact_pairs': obj_touch_table,
            'object_hand_normal_force_N': obj_hand_normal_N}


def no_hand_contact_counterfactual(mujoco, model_path: Path, output_dir: Path,
                                   object_name: str, initial_state, state_spec: int,
                                   ctrl_steps, actual_qpos, object_qadr: int) -> dict:
    """Replay identical motor commands with only robot hand/object contacts excluded."""
    import numpy as np
    xml = ET.parse(model_path).getroot()
    contact = xml.find('contact')
    if contact is None:
        contact = ET.SubElement(xml, 'contact')
    model = mujoco.MjModel.from_xml_path(str(model_path))
    hands = [name_of(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, b)
             for b in range(1, model.nbody)]
    hands = [n for n in hands if n.startswith(('lh_', 'rh_'))]
    if not hands:
        raise ValueError('Cannot construct causal contact ablation without named hand bodies')
    for n in hands:
        ET.SubElement(contact, 'exclude', name='nohand_' + n,
                      body1=object_name, body2=n)
    cf_xml = output_dir/'counterfactual_no_hand_object_collision.xml'
    cf_xml.write_text(ET.tostring(xml, encoding='unicode'), encoding='utf-8')
    cf_model = mujoco.MjModel.from_xml_path(str(cf_xml))
    if cf_model.nq != model.nq or cf_model.nv != model.nv or cf_model.nu != model.nu:
        raise ValueError('Counterfactual changed state dimensions or actuator count')
    if mujoco.mj_stateSize(cf_model, state_spec) != len(initial_state):
        raise ValueError('Counterfactual integration state size differs')
    cf_data = mujoco.MjData(cf_model)
    mujoco.mj_setState(cf_model, cf_data, initial_state, state_spec)
    mujoco.mj_forward(cf_model, cf_data)
    positions = [cf_data.qpos[object_qadr:object_qadr+3].copy()]
    forbidden_pairs = 0
    obj_body = mujoco.mj_name2id(cf_model, mujoco.mjtObj.mjOBJ_BODY, object_name)
    for ctrl in ctrl_steps:
        cf_data.ctrl[:] = ctrl
        mujoco.mj_step(cf_model, cf_data)
        positions.append(cf_data.qpos[object_qadr:object_qadr+3].copy())
        for ci in range(cf_data.ncon):
            c = cf_data.contact[ci]
            b1,b2 = (int(cf_model.geom_bodyid[int(c.geom1)]), int(cf_model.geom_bodyid[int(c.geom2)]))
            if obj_body in (b1,b2):
                other = b2 if b1 == obj_body else b1
                if name_of(mujoco, cf_model, mujoco.mjtObj.mjOBJ_BODY, other).startswith(('lh_', 'rh_')):
                    forbidden_pairs += 1
    if forbidden_pairs:
        raise RuntimeError('Counterfactual hand/object collision exclusion did not apply')
    cf_xyz = np.asarray(positions)
    delta = np.linalg.norm(cf_xyz - actual_qpos[:,object_qadr:object_qadr+3], axis=1)
    trace = output_dir/'counterfactual_no_hand_object_collision.npz'
    np.savez_compressed(trace, object_position_m=cf_xyz, actual_minus_nohand_position_m=delta,
                        initial_state=initial_state, ctrl=np.asarray(ctrl_steps))
    return {'model': receipt(cf_xml), 'trace': receipt(trace),
            'excluded_hand_bodies': len(hands), 'forbidden_pair_count': forbidden_pairs,
            'same_ctrl_per_step': True, 'max_object_position_difference_m': float(np.max(delta)),
            'final_object_position_difference_m': float(delta[-1])}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=ROOT/'outputs/fourth_vita_demo/assets/vita_object_05_dynamic.xml')
    parser.add_argument('--reference', type=Path, default=ROOT/'outputs/fourth_vita_demo/retarget_reference.npz')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs/fourth_vita_demo/physics_pd')
    parser.add_argument('--seconds', type=float, default=2.0)
    parser.add_argument('--hold-last-seconds', type=float, default=0.0,
                        help='Extra physical settling after last reference frame; explicitly synthetic hold')
    parser.add_argument('--gravity-comp-scale', type=float, default=0.0,
                        help='Fractional qfrc_bias/kp compensation added only to actuator ctrl targets')
    parser.add_argument('--gravity-comp-max-offset-rad', type=float, default=.25)
    parser.add_argument('--stance-ankle-pitch-offset-rad', type=float, default=0.0,
                        help='Symmetric ankle pitch ctrl target offset for physical balance diagnosis')
    parser.add_argument('--stance-hip-pitch-offset-rad', type=float, default=0.0,
                        help='Symmetric hip pitch ctrl target offset for physical balance diagnosis')
    parser.add_argument('--left-hand-world-tracking', action='store_true',
                        help='Compensate free-root drift using bounded arm IK in temporary kinematic data; modifies only actuator targets')
    parser.add_argument('--hand-world-correction-speed-rad-s', type=float, default=1.5,
                        help='Maximum rate of the additional world-hand tracking joint correction')
    parser.add_argument('--joint-rms-limit-rad', type=float, default=.35)
    parser.add_argument('--root-horizontal-limit-m', type=float, default=.25)
    parser.add_argument('--root-drop-limit-m', type=float, default=.18)
    parser.add_argument('--max-penetration-m', type=float, default=.02)
    parser.add_argument('--min-object-displacement-m', type=float, default=.01)
    parser.add_argument('--min-hand-contact-duration-s', type=float, default=.10)
    parser.add_argument('--min-hand-normal-impulse-Ns', type=float, default=.5)
    parser.add_argument('--max-object-position-error-m', type=float, default=.12)
    parser.add_argument('--max-final-object-linear-speed-m-s', type=float, default=.15)
    parser.add_argument('--mujoco-site', type=Path, default=Path('/data/grail-envs/retarget/lib/python3.10/site-packages'))
    args = parser.parse_args()
    if os.name != 'posix':
        parser.error('Physics runs only on the SSH server; local machine is video preview only')
    sys.path.append(str(args.mujoco_site))
    import mujoco
    import numpy as np
    if args.seconds <= 0 or args.hold_last_seconds < 0 or not 0 <= args.gravity_comp_scale <= 1 or args.gravity_comp_max_offset_rad < 0 or args.hand_world_correction_speed_rad_s <= 0:
        raise ValueError('Invalid seconds/hold/gravity compensation arguments')
    model = mujoco.MjModel.from_xml_path(str(args.model))
    if model.opt.disableflags:
        raise ValueError('All collision/contact solver disableflags must be clear')
    if float(model.opt.gravity[2]) >= -9.0:
        raise ValueError('Expected real downward gravity')
    with np.load(args.reference, allow_pickle=False) as z:
        source = {k: z[k] for k in z.files}
    for key in ('qpos', 'dof_names', 'time_s', 'object_target_pose', 'object_id'):
        if key not in source:
            raise ValueError(f'Reference missing {key}')
    ref_qpos = np.asarray(source['qpos'], dtype=np.float64)
    names = [str(x) for x in source['dof_names'].tolist()]
    stance_offsets = np.zeros(52)
    for token, value in [('ankle_pitch', args.stance_ankle_pitch_offset_rad),
                         ('hip_pitch', args.stance_hip_pitch_offset_rad)]:
        matched = [j for j, name in enumerate(names) if token in name and ('left' in name or 'right' in name)]
        if value and len(matched) != 2:
            raise ValueError(f'Expected bilateral {token} joints, got {matched}')
        stance_offsets[matched] = value
    times = np.asarray(source['time_s'], dtype=np.float64)
    object_poses = np.asarray(source['object_target_pose'], dtype=np.float64)
    raw_source_poses = None
    if 'raw_source_object_pose_xyz_wxyz' in source:
        raw_source_poses = np.asarray(source['raw_source_object_pose_xyz_wxyz'], dtype=np.float64)
        if raw_source_poses.shape != object_poses.shape or not np.isfinite(raw_source_poses).all():
            raise ValueError('Raw source object poses must match the reference timeline and be finite')
    object_id = int(np.asarray(source['object_id']).reshape(-1)[0])
    if ref_qpos.ndim != 2 or ref_qpos.shape[1] != 59 or len(names) != 52 or times.ndim != 1 or times.size != len(ref_qpos):
        raise ValueError('Expected T x 59 Vita52 reference and matching times')
    if object_poses.shape != (len(times), 7) or not np.isfinite(ref_qpos).all() or not np.isfinite(object_poses).all():
        raise ValueError('Object or robot reference invalid/nonfinite')
    if np.any(np.diff(times) <= 0):
        raise ValueError('Reference times must strictly increase')
    m = model_mapping(mujoco, model, names, object_id)
    qadr = np.asarray(m['qadr'], dtype=int)
    aids = np.asarray(m['actuator_ids'], dtype=int)
    data = mujoco.MjData(model)
    # Exactly one reset. Never assign qpos/qvel again during the rollout.
    data.qpos[m['robot_qadr']:m['robot_qadr']+7] = ref_qpos[0, :7]
    data.qpos[qadr] = ref_qpos[0, 7:]
    data.qpos[m['object_qadr']:m['object_qadr']+7] = object_poses[0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    initial_qpos = data.qpos.copy()
    initial_qvel = data.qvel.copy()
    state_spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    initial_state = np.empty(mujoco.mj_stateSize(model, state_spec))
    mujoco.mj_getState(model, data, initial_state, state_spec)
    initial_contacts = contact_snapshot(mujoco, model, data, m['object_body_id'])
    hand_tracking = None
    if args.left_hand_world_tracking:
        selected = [i for i,n in enumerate(names) if n.startswith(('left_shoulder','left_elbow','left_wrist'))]
        hand_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'lh_hand_base_link')
        if len(selected) != 7 or hand_body < 0:
            raise ValueError('World hand tracking requires the seven-joint left arm')
        hand_tracking = (np.asarray(selected), hand_body, mujoco.MjData(model), mujoco.MjData(model))
    if not np.isfinite(initial_qpos).all() or not np.isfinite(initial_qvel).all():
        raise ValueError('Nonfinite reset state')
    if model.nq < 66 or model.nv < 64 or model.nu != 52:
        raise ValueError(f'Expected Vita52+free object, got nq={model.nq}, nv={model.nv}, nu={model.nu}')
    dt = float(model.opt.timestep)
    reference_duration = float(times[-1] - times[0])
    horizon = min(float(args.seconds), reference_duration + float(args.hold_last_seconds))
    steps_requested = int(np.ceil(horizon / dt))
    if steps_requested <= 0:
        raise ValueError('No physics steps requested')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_snapshot = args.output_dir / 'source_fourth_vita_physics.py'
    if source_snapshot.exists() and sha(source_snapshot) != sha(Path(__file__)):
        raise ValueError('Use a fresh output directory after modifying the physics adapter')
    if not source_snapshot.exists():
        source_snapshot.write_bytes(Path(__file__).read_bytes())
    t0 = time.monotonic()
    t_records = [float(data.time)]
    q_records = [initial_qpos]
    v_records = [initial_qvel]
    ctrl_records = []
    tau_records = []
    qfrc_records = []
    xfrc_records = []
    target_records = []
    rms_records = []
    root_error_records = []
    object_error_records = []
    raw_source_object_error_records = []
    hand_target_corrections = []
    previous_hand_correction = np.zeros(52)
    contact_records = []
    warn0 = np.asarray([x.number for x in data.warning], dtype=np.int64)
    end_reason = 'requested_duration'
    hand_contact_steps = 0
    hand_normal_impulse_Ns = 0.0
    table_contact_steps = 0
    max_pen = initial_contacts['max_penetration_m']
    for step in range(steps_requested):
        t_ref = min(times[0] + float(data.time), times[-1])
        # At 500 Hz, linearly interpolate only the controller target. Original
        # 30 FPS reference frames remain unchanged in the source artifact.
        targets = np.asarray([np.interp(t_ref, times, ref_qpos[:, 7+j]) for j in range(52)])
        if hand_tracking is not None:
            indices, hand_body, desired_data, solve_data = hand_tracking
            desired_data.qpos[:] = data.qpos
            desired_data.qpos[:59] = np.asarray([np.interp(t_ref, times, ref_qpos[:,j]) for j in range(59)])
            desired_data.qpos[3:7] /= np.linalg.norm(desired_data.qpos[3:7])
            mujoco.mj_forward(model, desired_data)
            desired_pos = desired_data.xpos[hand_body].copy()
            desired_rot = desired_data.xmat[hand_body].reshape(3,3).copy()
            solve_data.qpos[:] = desired_data.qpos
            solve_data.qpos[:7] = data.qpos[:7]
            original_targets = targets.copy()
            jacp, jacr = np.zeros((3,model.nv)), np.zeros((3,model.nv))
            for _ in range(4):
                mujoco.mj_forward(model, solve_data)
                actual_rot = solve_data.xmat[hand_body].reshape(3,3)
                orientation_error = .5*sum(np.cross(actual_rot[:,j],desired_rot[:,j]) for j in range(3))
                error = np.r_[desired_pos-solve_data.xpos[hand_body], .2*orientation_error]
                mujoco.mj_jacBody(model,solve_data,jacp,jacr,hand_body)
                jac = np.vstack([jacp[:,np.asarray(m['vadr'])[indices]], .2*jacr[:,np.asarray(m['vadr'])[indices]]])
                delta = jac.T@np.linalg.solve(jac@jac.T+np.eye(6)*1e-4,error)
                targets[indices] += np.clip(delta,-.06,.06)
                targets[indices] = np.clip(targets[indices],original_targets[indices]-.25,original_targets[indices]+.25)
                solve_data.qpos[qadr[indices]] = targets[indices]
            correction = targets-original_targets
            correction = np.clip(correction,previous_hand_correction-args.hand_world_correction_speed_rad_s*dt,
                                 previous_hand_correction+args.hand_world_correction_speed_rad_s*dt)
            targets = original_targets+correction
            joint_ids = model.actuator_trnid[aids[indices],0]
            targets[indices] = np.clip(targets[indices],model.jnt_range[joint_ids,0],model.jnt_range[joint_ids,1])
            previous_hand_correction = targets-original_targets
            hand_target_corrections.append(float(np.max(np.abs(targets-original_targets))))
        if args.gravity_comp_scale:
            gains = model.actuator_gainprm[aids, 0]
            if np.any(gains <= 0):
                raise ValueError('Gravity compensation requires positive position actuator gains')
            bias = data.qfrc_bias[np.asarray(m['vadr'], dtype=int)]
            offset = np.clip(args.gravity_comp_scale * bias / gains,
                             -args.gravity_comp_max_offset_rad,
                             args.gravity_comp_max_offset_rad)
        else:
            offset = np.zeros(52)
        data.ctrl[aids] = targets + offset + stance_offsets
        if np.any(data.qfrc_applied) or np.any(data.xfrc_applied):
            raise ValueError('External wrench detected before active step')
        ctrl_records.append(data.ctrl.copy())
        qfrc_records.append(data.qfrc_applied.copy())
        xfrc_records.append(data.xfrc_applied.copy())
        previous = float(data.time)
        mujoco.mj_step(model, data)
        if abs(float(data.time) - previous - dt) > 1e-9:
            end_reason = 'solver_reset_time_discontinuity'
        q_records.append(data.qpos.copy())
        v_records.append(data.qvel.copy())
        t_records.append(float(data.time))
        tau_records.append(data.actuator_force.copy())
        target_records.append(targets)
        rms = float(np.sqrt(np.mean((data.qpos[qadr] - targets)**2)))
        rms_records.append(rms)
        ref_root_xy = np.asarray([np.interp(t_ref, times, ref_qpos[:, j]) for j in (0, 1)])
        root_err = float(np.linalg.norm(data.qpos[m['robot_qadr']:m['robot_qadr']+2] - ref_root_xy))
        root_error_records.append(root_err)
        obj_ref_xyz = np.asarray([np.interp(t_ref, times, object_poses[:, j]) for j in range(3)])
        obj_err = float(np.linalg.norm(data.qpos[m['object_qadr']:m['object_qadr']+3] - obj_ref_xyz))
        object_error_records.append(obj_err)
        if raw_source_poses is not None:
            raw_xyz = np.asarray([np.interp(t_ref, times, raw_source_poses[:, j]) for j in range(3)])
            raw_source_object_error_records.append(float(np.linalg.norm(data.qpos[m['object_qadr']:m['object_qadr']+3] - raw_xyz)))
        c = contact_snapshot(mujoco, model, data, m['object_body_id'])
        c['time_s'] = float(data.time)
        contact_records.append(c)
        hand_contact_steps += int(c['object_hand_contact_pairs'] > 0)
        hand_normal_impulse_Ns += c['object_hand_normal_force_N'] * dt
        table_contact_steps += int(c['object_table_contact_pairs'] > 0)
        max_pen = max(max_pen, c['max_penetration_m'])
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            end_reason = 'nonfinite_state'
        if np.any(np.asarray([x.number for x in data.warning]) > warn0):
            end_reason = 'mujoco_solver_warning'
        if data.qpos[m['robot_qadr']+2] < initial_qpos[m['robot_qadr']+2] - args.root_drop_limit_m:
            end_reason = 'root_drop_limit'
        if end_reason != 'requested_duration':
            break
    q_np = np.asarray(q_records)
    v_np = np.asarray(v_records)
    ctrl_np = np.asarray(ctrl_records)
    tau_np = np.asarray(tau_records)
    counterfactual = no_hand_contact_counterfactual(
        mujoco, args.model, args.output_dir, m['object_body_name'],
        initial_state, state_spec, ctrl_np, q_np, m['object_qadr'])
    report = {
        'schema': 'fourth_vita_physical_reference_tracking.v1',
        'status': 'not_verified', 'tracking_success': False,
        'input': {'model': receipt(args.model), 'reference': receipt(args.reference),
                  'physics_script': receipt(source_snapshot)},
        'mujoco_version': mujoco.__version__, 'model_nq': model.nq, 'model_nv': model.nv,
        'actuators': model.nu, 'robot_free_root_qadr': m['robot_qadr'],
        'object_free_root_qadr': m['object_qadr'], 'object_body_name': m['object_body_name'],
        'object_id': object_id, 'physics_steps': len(ctrl_records),
        'physics_timestep_s': dt, 'duration_s': float(data.time),
        'reference_duration_s': reference_duration,
        'synthetic_post_hold_s_requested': float(args.hold_last_seconds),
        'synthetic_post_hold_s_executed': float(max(0.0, data.time-reference_duration)),
        'reference_is_synthetic_static_diagnostic': bool(np.asarray(source.get('synthetic_static_diagnostic', False)).reshape(-1)[0]),
        'initial_reference_frame': int(np.asarray(source['frame_ids'])[0]) if 'frame_ids' in source else None,
        'end_reason': end_reason,
        'initial_contacts': initial_contacts,
        'active_qpos_qvel_writes': 0, 'active_qfrc_applied_writes': 0,
        'active_xfrc_applied_writes': 0, 'object_mocap': False,
        'root_is_free': True,
        'controller': '52 native MuJoCo position actuators with explicit asset kp/kv, reference interpolation at physics dt; no SONIC policy inference',
        'gravity_compensation_fraction': float(args.gravity_comp_scale),
        'gravity_compensation_max_target_offset_rad': float(args.gravity_comp_max_offset_rad),
        'stance_ankle_pitch_offset_rad': float(args.stance_ankle_pitch_offset_rad),
        'stance_hip_pitch_offset_rad': float(args.stance_hip_pitch_offset_rad),
        'left_hand_world_tracking': bool(args.left_hand_world_tracking),
        'left_hand_world_tracking_max_joint_correction_rad': max(hand_target_corrections,default=0.0),
        'left_hand_world_tracking_correction_speed_limit_rad_s': float(args.hand_world_correction_speed_rad_s),
        'assumed_physical_parameters': True,
        'joint_rms_rad_median': float(np.median(rms_records)),
        'joint_rms_rad_max': float(np.max(rms_records)),
        'root_horizontal_error_m_max': float(np.max(root_error_records)),
        'root_height_drop_m': float(initial_qpos[m['robot_qadr']+2] - np.min(q_np[:, m['robot_qadr']+2])),
        'object_reference_position_error_m_max': float(np.max(object_error_records)),
        'raw_source_object_position_error_m_max': float(np.max(raw_source_object_error_records)) if raw_source_object_error_records else None,
        'reference_object_pose_equals_raw_source': bool(np.array_equal(object_poses, raw_source_poses)) if raw_source_poses is not None else None,
        'synthetic_task_adaptation': bool(np.asarray(source['synthetic_task_adaptation']).reshape(-1)[0]) if 'synthetic_task_adaptation' in source else None,
        'object_displacement_m': float(np.linalg.norm(q_np[-1, m['object_qadr']:m['object_qadr']+3] - initial_qpos[m['object_qadr']:m['object_qadr']+3])),
        'object_hand_contact_step_count': hand_contact_steps,
        'object_hand_contact_duration_s': float(hand_contact_steps*dt),
        'object_hand_normal_impulse_Ns': float(hand_normal_impulse_Ns),
        'object_hand_normal_force_peak_N': float(max(c['object_hand_normal_force_N'] for c in contact_records)),
        'object_table_contact_step_count': int(table_contact_steps),
        'object_table_contact_final_0p15s_fraction': float(np.mean([c['object_table_contact_pairs'] > 0 for c in contact_records[-max(1,int(.15/dt)):]])),
        'object_final_linear_speed_m_s': float(np.linalg.norm(v_np[-1,m['object_vadr']:m['object_vadr']+3])),
        'no_hand_contact_counterfactual': counterfactual,
        'max_contact_penetration_m': float(max_pen),
        'motor_torque_abs_peak_Nm': float(np.max(np.abs(tau_np))),
        'simulation_passed': False,
        'quality_thresholds': {'joint_rms_median_rad_max': args.joint_rms_limit_rad,
            'root_horizontal_error_m_max': args.root_horizontal_limit_m,
            'root_height_drop_m_max': args.root_drop_limit_m,
            'max_penetration_m': args.max_penetration_m,
            'min_object_displacement_m': args.min_object_displacement_m,
            'min_object_hand_contact_duration_s': args.min_hand_contact_duration_s,
            'min_object_hand_normal_impulse_Ns': args.min_hand_normal_impulse_Ns,
            'max_object_position_error_m': args.max_object_position_error_m,
            'max_final_object_linear_speed_m_s': args.max_final_object_linear_speed_m_s,
            'min_causal_object_position_difference_m': .005,
            'min_table_contact_final_0p15s_fraction': .5},
        'elapsed_wall_s': time.monotonic() - t0,
        'limitations': ['Model mass/friction and collision proxies are explicit assumptions, not measured.',
            'Controller is native PD reference tracking, not strict SONIC checkpoint reproduction.',
            'Reference contacts and object trajectories are reconstructed estimates, not force ground truth.']}
    checks = {
        'completed_requested_duration': end_reason == 'requested_duration',
        'joint_tracking': report['joint_rms_rad_median'] <= args.joint_rms_limit_rad,
        'root_horizontal_tracking': report['root_horizontal_error_m_max'] <= args.root_horizontal_limit_m,
        'no_fall': report['root_height_drop_m'] <= args.root_drop_limit_m,
        'penetration': max_pen <= args.max_penetration_m,
        'object_moved': report['object_displacement_m'] >= args.min_object_displacement_m,
        'hand_object_contact_duration': hand_contact_steps*dt >= args.min_hand_contact_duration_s,
        'hand_object_normal_impulse': hand_normal_impulse_Ns >= args.min_hand_normal_impulse_Ns,
        'object_trajectory_tracking': report['object_reference_position_error_m_max'] <= args.max_object_position_error_m,
        'object_final_speed': report['object_final_linear_speed_m_s'] <= args.max_final_object_linear_speed_m_s,
        'object_final_table_support': report['object_table_contact_final_0p15s_fraction'] >= .5,
        'hand_contact_causally_moves_object': counterfactual['max_object_position_difference_m'] >= .005,
    }
    report['checks'] = checks
    report['simulation_passed'] = bool(all(checks.values()))
    report['tracking_success'] = report['simulation_passed']
    report['status'] = 'passed_assumed_physics_demo' if report['simulation_passed'] else 'failed_physical_tracking_or_interaction'
    trace_path = args.output_dir / 'physical_trace.npz'
    np.savez_compressed(trace_path,
        time=np.asarray(t_records), qpos=q_np, qvel=v_np,
        ctrl=ctrl_np, actuator_force=tau_np,
        qfrc_applied=np.asarray(qfrc_records), xfrc_applied=np.asarray(xfrc_records),
        reference_joint_target=np.asarray(target_records),
        reference_qpos=ref_qpos, reference_time_s=times,
        initial_state=initial_state, state_spec=np.asarray(state_spec),
        initial_qpos=initial_qpos, initial_qvel=initial_qvel,
        object_qadr=np.asarray(m['object_qadr']), robot_qadr=np.asarray(m['robot_qadr']),
        joint_qadr=qadr, actuator_ids=aids, dof_names=np.asarray(names),
        joint_rms_rad=np.asarray(rms_records),
        root_horizontal_error_m=np.asarray(root_error_records),
        object_reference_position_error_m=np.asarray(object_error_records),
        raw_source_object_position_error_m=np.asarray(raw_source_object_error_records))
    contacts_path = args.output_dir / 'physical_contacts.json'
    atomic_json(contacts_path, contact_records)
    report['artifacts'] = {'trace': receipt(trace_path), 'contacts': receipt(contacts_path)}
    atomic_json(args.output_dir / 'report.json', report)
    print(json.dumps({'status': report['status'], 'steps': len(ctrl_records),
                      'duration_s': report['duration_s'], 'checks': checks,
                      'output_dir': str(args.output_dir)}, ensure_ascii=False), flush=True)
    return 0 if report['simulation_passed'] else 3


if __name__ == '__main__':
    raise SystemExit(main())
