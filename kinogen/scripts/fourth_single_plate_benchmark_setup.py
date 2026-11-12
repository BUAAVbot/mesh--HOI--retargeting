"""Explicit table-edge reset benchmark, with untouched hand geometry and free root."""
from pathlib import Path
import hashlib
import json
import numpy as np
import mujoco as mj
from fourth_sonic_controller import Controller
from fourth_arm_object_feedback import ArmObjectFeedback
from fourth_o6_load_hysteresis_c7 import O6CoupledReferenceContactFeedback

def prepare(setup_path, output, model_override=None, body_control='sonic'):
    root = Path(__file__).resolve().parents[1]
    setup = json.loads(Path(setup_path).read_text())
    if 'initial_qpos' not in setup:
        setup = {**setup, 'initial_qpos': setup['best']['initial_qpos'],
                 'declared_object_reset_change': setup.get('object1_reset_translation_world_m')}
    model_path = Path(model_override or setup['model'])
    m = (mj.MjModel.from_binary_path(str(model_path)) if model_path.suffix == '.mjb'
         else mj.MjModel.from_xml_path(str(model_path.resolve())))
    q0 = np.asarray(setup['initial_qpos'], float)
    if q0.shape != (m.nq,):
        raise ValueError('Complete declared reset state required')
    names = [m.joint(j).name for j in range(m.njnt) if m.jnt_type[j] == mj.mjtJoint.mjJNT_HINGE]
    times = np.linspace(0, 90., 923)
    ref = {'time_s': times, 'dof_names': np.asarray(names),
           'qpos': np.tile(q0[:59], (len(times), 1)), 'object_ids': np.arange(1, 7),
           'object_target_pose': np.tile(q0[59:].reshape(1, 6, 7), (len(times), 1, 1)),
           'contact_candidate_observed': np.zeros((len(times), 2, 6), bool)}
    ref['contact_candidate_observed'][:, 1, 0] = True
    ref['contact_candidate'] = ref['contact_candidate_observed'].copy()
    ref['contact_candidate_confidence'] = ref['contact_candidate'].astype(float)
    config_path = root/'outputs/fourth_full_sequence/physics_sonic15225_o6_support_absolute_hold_v1_5s/source_sonic_config.json'
    config = json.loads(config_path.read_text()); config['hand_object_feedback'] = False
    config.pop('arm_feedback', None); config.pop('finger_feedback', None)
    sonic = Controller(m, ref, config)
    arm = ArmObjectFeedback(m, ref, {'initial_geometric_seed': 'actual_state'})
    fcfg = {'geometry_backend': 'convex_sat_o6',
            'finger_urdf': str(root/'outputs/fourth_full_sequence/assets/o6_calibrated_linear_joint_contracts_v2.urdf'),
            'max_motor_torque_Nm': .4, 'force_to_gap_gain_m_per_N': .00005,
            'auto_gravity_normal_force': False, 'force_target_per_digit_N': 1.5,
            'max_force_per_digit_N': 5., 'max_hand_normal_force_N': 20.,
            'closure_rate_rad_s': .03, 'release_rate_rad_s': .03,
            'load_preload_normal_force_N': .2, 'load_force_ramp_s': .8,
            'table_clearance_guard': {'enabled': True, 'minimum_clearance_m': .002,
                                      'guard_phases': ['idle', 'release', 'fault', 'acquire', 'grasp']}}
    finger = O6CoupledReferenceContactFeedback(m, ref, fcfg)
    passive = np.flatnonzero(np.all(m.actuator_gear == 0, axis=1))
    held_joints = [j for j in range(m.njnt) if (m.joint(j).name or '').startswith(
        ('waist', 'left_shoulder', 'left_elbow', 'left_wrist', 'right_shoulder', 'right_elbow', 'right_wrist'))]
    qa, va = m.jnt_qposadr[held_joints], m.jnt_dofadr[held_joints]
    aids = np.asarray([m.actuator('motor_'+m.joint(j).name).id for j in held_joints])
    si = np.asarray([sonic.names.index(m.joint(j).name) for j in held_joints])
    d, kin = mj.MjData(m), mj.MjData(m)
    d.qpos[:] = q0; d.qvel[:] = 0.; mj.mj_forward(m, d); sonic.reset(d)
    if body_control != 'sonic':
        def pose_action(data, elapsed):
            packet = np.zeros(m.nu)
            packet[sonic.ref_aids] = q0[7:59]
            if body_control == 'pd_all_gff':
                kin.qpos[:] = data.qpos; kin.qvel[:] = 0.; mj.mj_forward(m, kin)
                packet[sonic.aids] = q0[sonic.qa]+kin.qfrc_bias[sonic.va]/m.actuator_gainprm[sonic.aids, 0]
            packet[passive] = 0.
            return packet
        sonic.action = pose_action
    spec = int(mj.mjtState.mjSTATE_INTEGRATION)
    state0 = np.empty(mj.mj_stateSize(m, spec)); mj.mj_getState(m, d, state0, spec)
    sequences = {key: [] for key in ['controls', 'torques', 'qforces', 'xforces', 'contacts']}
    qseq, vseq, tseq = [d.qpos.copy()], [d.qvel.copy()], [float(d.time)]
    objects = {m.body(f'target_object_{i:02d}').id for i in range(1, 7)}
    hand_bodies = {b for b in range(m.nbody) if (m.body(b).name or '').startswith(('rh_', 'lh_'))}
    impulse = 0.; maxmimic = maxpen = 0.; settle_stop = 'duration'
    for step in range(round(2./m.opt.timestep)):
        ctrl = sonic.action(d, float(d.time))
        kin.qpos[:] = d.qpos; kin.qvel[:] = 0.; mj.mj_forward(m, kin)
        ctrl[aids] = q0[qa]+kin.qfrc_bias[va]/m.actuator_gainprm[aids, 0]
        for hand in finger.hands:
            projection = np.zeros((m.nv, 6)); projection[hand['va']] = hand['coupling_matrix']
            ctrl[hand['root_aids']] = q0[hand['root_qa']]+projection.T@kin.qfrc_bias/m.actuator_gainprm[hand['root_aids'], 0]
        ctrl[passive] = 0.; sonic.last[si] = (ctrl[aids]-sonic.default[si])/sonic.scale[si]
        d.ctrl[:] = ctrl; mj.mj_step(m, d)
        for ci, c in enumerate(d.contact[:d.ncon]):
            gs = list(map(int, c.geom)); bs = list(map(int, m.geom_bodyid[gs])); force = np.zeros(6)
            mj.mj_contactForce(m, d, ci, force)
            sequences['contacts'].append((step, gs, bs, c.pos.copy(), c.frame.copy(), force.copy(), float(c.dist)))
            if objects.intersection(bs) and hand_bodies.intersection(bs):
                impulse += max(0., float(force[0]))*m.opt.timestep
            maxpen = max(maxpen, max(0., -float(c.dist)))
        for e in range(m.neq):
            if m.eq_type[e] == mj.mjtEq.mjEQ_JOINT:
                child, parent = int(m.eq_obj1id[e]), int(m.eq_obj2id[e])
                maxmimic = max(maxmimic, abs(float(d.qpos[m.jnt_qposadr[child]]
                                -m.eq_data[e, 1]*d.qpos[m.jnt_qposadr[parent]])))
        for key, value in [('controls', ctrl), ('torques', d.actuator_force),
                           ('qforces', d.qfrc_applied), ('xforces', d.xfrc_applied)]:
            sequences[key].append(value.copy())
        qseq.append(d.qpos.copy()); vseq.append(d.qvel.copy()); tseq.append(float(d.time))
        upright = float(d.xmat[sonic.base].reshape(3, 3)[2, 2])
        if maxmimic > .03 or upright < .8 or np.linalg.norm(d.qpos[:3]-q0[:3]) > .15:
            settle_stop = 'body_or_hand_protection'; break
    np.savez_compressed(output/'settle_trace.npz', time_s=np.asarray(tseq), qpos=np.asarray(qseq),
                        qvel=np.asarray(vseq), ctrl=np.asarray(sequences['controls']),
                        actuator_force=np.asarray(sequences['torques']), initial_state=state0,
                        state_spec=np.asarray(spec), qfrc_applied=np.asarray(sequences['qforces']),
                        xfrc_applied=np.asarray(sequences['xforces']))
    np.savez_compressed(output/'source_reference.npz', **ref)
    (output/'source_config.json').write_text(json.dumps(config, indent=2)+'\n')
    evidence = {'scope': 'Controlled reset table-edge benchmark, not source raw-scene reconstruction.',
                'setup': str(setup_path), 'setup_sha256': hashlib.sha256(Path(setup_path).read_bytes()).hexdigest(),
                'physical_model': str(model_path),
                'body_controller': body_control, 'learned_leg_policy_used': body_control == 'sonic',
                'settle_stop': settle_stop, 'settle_duration_s': float(d.time),
                'declared_object_reset_change': setup.get('declared_object_reset_change'),
                'settle_hand_object_impulse_Ns': impulse, 'max_mimic_error_rad': maxmimic,
                'robot_root_drift_m': float(np.linalg.norm(d.qpos[:3]-q0[:3])),
                'robot_up': float(d.xmat[sonic.base].reshape(3, 3)[2, 2]),
                'post_reset_object_state_writes': 0}
    (output/'settle_report.json').write_text(json.dumps(evidence, indent=2)+'\n')
    print(json.dumps(evidence), flush=True)
    if impulse > 1e-9 or maxmimic > .03 or evidence['robot_up'] < .9 or evidence['robot_root_drift_m'] > .1:
        raise ValueError('Declared non-contact stable initial settling did not pass')
    return {'m': m, 'd': d, 'sonic': sonic, 'ref': ref, 'fcfg': fcfg, 'passive': passive,
            'q0': q0, 'qseq': qseq, 'vseq': vseq, 'tseq': tseq, **sequences,
            'maxmimic': maxmimic, 'maxpen': maxpen, 'g': {'state0': state0, 'spec': spec}}
