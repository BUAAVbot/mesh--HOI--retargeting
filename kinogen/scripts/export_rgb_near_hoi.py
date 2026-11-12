"""Export an accepted full dual-Vita trial with collision and attachment forces distinct."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def receipt(path):
    p = Path(path)
    return {'path':str(p.resolve()), 'sha256':sha(p), 'bytes':p.stat().st_size}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trial', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data', type=Path, default=Path(__file__).resolve().parents[1]/'data/rgb_261026_near_handoff')
    a = p.parse_args()
    import numpy as np
    import mujoco
    r = json.loads((a.trial/'report.json').read_text())
    audit = json.loads((a.trial/'independent_replay_audit.json').read_text())
    source_audit_path = a.trial/'source_provenance_audit.json'
    source_audit = json.loads(source_audit_path.read_text())
    required = {'full_source_duration', 'finite_no_solver_warning', 'both_root_horizontal_tracking',
                'both_root_height_tracking', 'both_joint_tracking', 'penetration',
                'both_seated_support', 'shoe_source_tracking', 'both_root_orientation_tracking',
                'hand_shoe_manipulation_coupling', 'grasp_constraint_causal_motion',
                'grasp_constraint_force_plausible'}
    checks = r.get('checks', {})
    if r.get('simulation_passed') is not True or not required.issubset(checks) or any(v is not True for v in checks.values()):
        raise ValueError('Accepted dataset export requires complete physical quality checks')
    if audit.get('status') != 'deterministic_physics_trace_verified':
        raise ValueError('Independent replay audit required')
    if audit.get('force_trace_verified') is not True:
        raise ValueError('Independent actuator/contact/constraint force replay required')
    if source_audit.get('status') != 'raw_source_provenance_verified':
        raise ValueError('Original source preservation audit required')
    if r['source_frames'] != 398 or r['duration_s'] < r['source_duration_s']-.005:
        raise ValueError('Full 398-frame source must be covered')
    if r['mode'] != 'sonic' or not isinstance(r.get('controller'), dict):
        raise ValueError('The requested SONIC pipeline requires learned inference evidence')
    if r['controller'].get('sonic_target_weight', 0) <= 0:
        raise ValueError('Zero SONIC influence must remain a PD diagnostic')
    actors = r['controller'].get('actors', [])
    if len(actors) != 2 or any(v.get('inference_count', 0) <= 0 for v in actors):
        raise ValueError('Both actors require actual policy inference')
    for key in ('model', 'trace', 'contacts'):
        path = a.trial/Path(r[key]['path']).name
        if sha(path) != r[key]['sha256']:
            raise ValueError('Changed frozen artifact: '+key)
    if audit['model_sha256'] != r['model']['sha256'] or audit['trace_sha256'] != r['trace']['sha256']:
        raise ValueError('Replay audit corresponds to another trial')
    refpath = Path(r['input']['reference']['path'])
    if sha(refpath) != r['input']['reference']['sha256']:
        raise ValueError('Reference has changed')
    model = mujoco.MjModel.from_binary_path(str(a.trial/'model.mjb'))
    z = np.load(a.trial/'trace.npz', allow_pickle=False)
    c = np.load(a.trial/'contacts.npz', allow_pickle=False)
    ref = np.load(refpath, allow_pickle=False)
    legacy_grip = ref.get('O6_grasp_geometric_quality_passed')
    if legacy_grip is not None and bool(np.asarray(legacy_grip).any()):
        raise ValueError('Legacy native-GJK grasp-pass flag must be superseded by independent convex verification')
    nq = len(z['time'])
    if z['qpos'].shape != (nq, model.nq) or len(z['ctrl']) != nq-1:
        raise ValueError('Invalid trace shape')
    qa, va, aid = z['joint_qadr'].astype(int), z['joint_vadr'].astype(int), z['actuator_ids'].astype(int)
    if qa.shape != (2, 52) or va.shape != (2, 52) or aid.shape != (2, 52):
        raise ValueError('Both Vita52 name mappings required')
    rootadr = z['robot_root_qadr'].astype(int)
    oq = int(z['object_qadr'])
    data = mujoco.MjData(model)
    positions, orientations = [], []
    for q in z['qpos']:
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        positions.append(data.xpos.copy())
        orientations.append(data.xquat.copy())
    payload = {
        'state_time_s':z['time'], 'control_interval_start_time_s':z['time'][:-1],
        'actor_names':np.asarray(['person_01_to_vita', 'person_02_to_vita']),
        'joint_names_unprefixed':z['joint_names'],
        'root_pose_xyz_wxyz':np.stack([z['qpos'][:,q:q+7] for q in rootadr], axis=1),
        'joint_position_rad':np.stack([z['qpos'][:,v] for v in qa], axis=1),
        'joint_velocity_rad_s':np.stack([z['qvel'][:,v] for v in va], axis=1),
        'joint_control_target_rad':np.stack([z['ctrl'][:,v] for v in aid], axis=1),
        'joint_drive_torque_Nm':np.stack([z['actuator_force'][:,v]*model.actuator_gear[v,0] for v in aid], axis=1),
        'joint_reflected_armature_kg_m2':model.dof_armature[va],
        'joint_passive_damping_Nms_per_rad':model.dof_damping[va],
        'joint_friction_loss_Nm':model.dof_frictionloss[va],
        'actuator_gain_parameters':model.actuator_gainprm[aid],
        'actuator_bias_parameters':model.actuator_biasprm[aid],
        'shoe_pose_xyz_wxyz':z['qpos'][:,oq:oq+7],
        'body_position_world_m':np.asarray(positions), 'body_orientation_wxyz':np.asarray(orientations),
        'body_mass_kg':model.body_mass.copy(),
        'body_inertia_principal_kg_m2':model.body_inertia.copy(),
        'body_inertial_offset_local_m':model.body_ipos.copy(),
        'body_inertial_orientation_wxyz':model.body_iquat.copy(),
        'geom_friction':model.geom_friction.copy(), 'geom_body_id':model.geom_bodyid.copy(),
        'body_names':np.asarray([mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,b) or 'world' for b in range(model.nbody)])}
    for key in c.files:
        payload['collision_'+key] = c[key]
    for key in z.files:
        if key.startswith(('eq_', 'equality_', 'attachment_', 'shoe_raw', 'shoe_reference', 'raw_source_shoe', 'seat_', 'foot_', 'hand_', 'o6_', 'qfrc_constraint', 'qfrc_equality')):
            payload['physics_'+key] = z[key]
    if 'qfrc_constraint' in z:
        payload['joint_total_constraint_torque_Nm'] = np.stack([z['qfrc_constraint'][:,v] for v in va], axis=1)
    if 'qfrc_equality' in z:
        payload['joint_equality_constraint_torque_Nm'] = np.stack([z['qfrc_equality'][:,v] for v in va], axis=1)
    for key in ('root_horizontal_error_m','root_height_error_m','root_orientation_error_rad','joint_rms_rad','shoe_reference_error_m','raw_source_shoe_error_m'):
        if key in z:
            payload['quality_'+key] = z[key]
    depth = np.zeros(nq-1)
    np.maximum.at(depth, c['interval_index'], np.maximum(0.,-c['signed_distance_m']))
    payload['quality_contact_penetration_m'] = depth
    for key in ref.files:
        payload['reference_'+key] = ref[key]
    source_inputs = {}
    for actor in range(2):
        path = a.data/f'human/person_{actor+1:02}/smplx_params_world.npz'
        params = np.load(path, allow_pickle=False)
        for key in params.files:
            payload[f'source_person{actor+1:02}_smplx_'+key] = params[key]
        source_inputs[f'person{actor+1:02}_smplx_parameters'] = receipt(path)
    for name, prefix in [('cameras_world.npz','source_camera_'),('timeline.npz','source_timeline_')]:
        path = a.data/name
        values = np.load(path, allow_pickle=False)
        for key in values.files:
            payload[prefix+key] = values[key]
        source_inputs[name] = receipt(path)
    a.output.mkdir(parents=True, exist_ok=True)
    out = a.output/'hoi_motion_physics.npz'
    np.savez_compressed(out, **payload)
    meta = {
        'schema':'rgb_near_dual_vita_hoi.v1', 'status':'accepted_assumed_physics_full_sequence',
        'source_frames':398, 'source_duration_s':r['source_duration_s'], 'physics_steps':r['physics_steps'],
        'state_frames':nq, 'robot_actors':2, 'dofs_per_robot':52,
        'allocated_control_slots':int(model.nu),
        'nonzero_drive_gain_channels':int(np.sum(np.any(model.actuator_gainprm!=0,axis=1)|np.any(model.actuator_biasprm!=0,axis=1))),
        'equality_constraints':int(model.neq),
        'coordinate_system':'right-handed +Z up, approximate metres from monocular reconstruction',
        'quaternion_order':'wxyz', 'collision_edges':'one edge per contact point/interval; geom/body ids index included names',
        'collision_wrench_convention':'world-frame force and torque on geom2 at contact point; opposite sign on geom1',
        'attachment_wrench_convention':'world-frame equality force and torque on shoe about its freejoint/body origin; distinct from collision-point wrenches',
        'equality_graph_types':'Body weld constraints and joint mimic constraints have distinct object types/indices. Do not interpret joint IDs as body IDs.',
        'passive_joint_torques':'Motor torque and total/equality constraint torque are separate arrays; passive coupled joints can have zero motor torque and nonzero constraint torque.',
        'timing':'state N follows control interval N-1; solver contact/actuator outputs label interval start',
        'force_provenance':'simulated, not sensor ground truth',
        'physical_parameters':'robot inertias reused from near-source assets; shoe/bench/material parameters assumed',
        'source_fingers':'default source posture, not observed; hand grasp adaptations are inferred',
        'grasp_geometry_status':'Native-GJK zero-distance thumb-contact claim withdrawn. Compliant-assisted manipulation is reported separately from real finger contact; no pure-friction grasp claim.',
        'source_shoe_orientation':'unobserved rotation estimate; not complete rotation ground truth',
        'controller':r['controller'], 'checks':checks, 'quality_thresholds':r['quality_thresholds'],
        'limitations':r['limitations'], 'reference':r['input']['reference'],
        'trial_report':receipt(a.trial/'report.json'), 'replay_audit':receipt(a.trial/'independent_replay_audit.json'),
        'source_provenance_audit':receipt(source_audit_path), 'source_inputs':source_inputs,
        'contact_reference_changes':{key:value for key,value in source_audit.items() if key.startswith('contact_optimized_')},
        'model':r['model'], 'trace':r['trace'], 'contacts':r['contacts'], 'dataset':receipt(out)}
    for key in ('raw_source_shoe_position_error_m_max','shoe_reference_position_error_m_max',
                'attachments','counterfactual_no_grasp_constraint',
                'grasp_constraint_active_duration_s','grasp_constraint_force_impulse_Ns',
                'grasp_constraint_force_peak_N','root_orientation_error_rad_max',
                'physical_actuator_channels','zero_force_abi_channels',
                'assumed_additional_finger_rotor_kg_m2'):
        if key in r:
            meta[key] = r[key]
    (a.output/'metadata.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'status':meta['status'], 'dataset':meta['dataset'], 'arrays':len(payload)}), flush=True)


if __name__ == '__main__':
    main()
