"""Export an audited dynamic trial as timestamped HOI motion/physics data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REQUIRED_CHECKS = frozenset({
    'completed_requested_duration', 'joint_tracking', 'root_horizontal_tracking',
    'no_fall', 'penetration', 'object_moved', 'hand_object_contact_duration',
    'hand_object_normal_impulse', 'object_trajectory_tracking', 'object_final_speed',
    'object_final_table_support', 'hand_contact_causally_moves_object',
})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trial-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    trial = a.trial_dir.resolve()
    report = read_json(trial / 'report.json')
    audit = read_json(trial / 'independent_replay_audit.json')
    checks = report.get('checks', {})
    if (report.get('simulation_passed') is not True or
            not REQUIRED_CHECKS.issubset(checks) or
            any(value is not True for value in checks.values())):
        raise ValueError('Refusing accepted-dataset export: physical quality checks failed')
    if audit.get('status') != 'deterministic_physics_trace_verified':
        raise ValueError('Independent integration replay must pass before export')
    trace_path = trial / 'physical_trace.npz'
    contact_path = trial / 'physical_contacts.json'
    model_path = Path(report['input']['model']['path'])
    reference_path = Path(report['input']['reference']['path'])
    for path, expected in [(trace_path, report['artifacts']['trace']['sha256']),
                           (contact_path, report['artifacts']['contacts']['sha256']),
                           (model_path, report['input']['model']['sha256']),
                           (reference_path, report['input']['reference']['sha256'])]:
        if sha(path) != expected:
            raise ValueError('Frozen input hash mismatch: ' + str(path))
    if audit['trace_sha256'] != sha(trace_path) or audit['model_sha256'] != sha(model_path):
        raise ValueError('Replay audit does not correspond to this model and trace')

    import numpy as np
    import mujoco
    z = np.load(trace_path, allow_pickle=False)
    ref = np.load(reference_path, allow_pickle=False)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    contacts = read_json(contact_path)
    def reference_scalar(key, default=None):
        return ref[key].item() if key in ref.files else default

    sidecar_path = reference_path.with_suffix('.metadata.json')
    sidecar = read_json(sidecar_path) if sidecar_path.exists() else {}
    if sidecar and sidecar.get('frozen_reference_sha256') != sha(reference_path):
        raise ValueError('Reference metadata sidecar describes a different frozen reference')
    hand_adjusted = bool(reference_scalar('synthetic_reference_hand_adjustment', False))
    if sidecar.get('synthetic_hand_contact_adaptation') is True:
        hand_adjusted = True
    raw_pose_equal = None
    if 'raw_source_object_pose_xyz_wxyz' in ref.files:
        raw_pose_equal = bool(np.array_equal(ref['object_target_pose'], ref['raw_source_object_pose_xyz_wxyz']))
        if report.get('reference_object_pose_equals_raw_source') is not raw_pose_equal:
            raise ValueError('Reported source object identity disagrees with the frozen reference')
    adaptation = {
        'synthetic_reference_hand_adjustment': hand_adjusted,
        'synthetic_hand_contact_adaptation': hand_adjusted,
        'source_object_trajectory_modified': None if raw_pose_equal is None else not raw_pose_equal,
        'description': sidecar.get('adaptation_description', reference_scalar('adaptation_description')),
        'synthetic_initial_hand_away_m': reference_scalar('synthetic_initial_hand_away_m'),
        'synthetic_initial_hand_lift_m': reference_scalar('synthetic_initial_hand_lift_m'),
        'synthetic_hand_slide_m': reference_scalar('synthetic_hand_slide_m'),
        'synthetic_hand_slide_direction_world': ref['synthetic_hand_slide_direction_world'].tolist()
            if 'synthetic_hand_slide_direction_world' in ref.files else None,
        'precontact_robot_seed_qpos_semantics': sidecar.get('precontact_robot_seed_qpos_semantics'),
    }
    times = z['time']
    if len(contacts) != len(times)-1:
        raise ValueError('Need a contact record for each integration interval')
    names = z['dof_names']
    qadr = z['joint_qadr'].astype(int)
    aids = z['actuator_ids'].astype(int)
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(n)) for n in names]
    if (len(names) != 52 or len(set(names.tolist())) != 52 or min(jids) < 0 or
            qadr.shape != (52,) or aids.shape != (52,) or len(set(aids.tolist())) != 52 or
            np.any(aids < 0) or np.any(aids >= model.nu)):
        raise ValueError('Expected a complete one-to-one Vita52 joint/actuator mapping')
    if not np.array_equal(qadr, model.jnt_qposadr[jids]):
        raise ValueError('Trace joint addresses disagree with the frozen model')
    vadr = model.jnt_dofadr[jids]
    if any(model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE for j in jids):
        raise ValueError('Expected hinge joints when exporting N*m joint torques')
    if (not np.all(model.actuator_trnid[aids, 0] == jids) or
            not np.all(model.actuator_trntype[aids] == mujoco.mjtTrn.mjTRN_JOINT)):
        raise ValueError('Actuator to joint mapping changed')
    oq = int(z['object_qadr'])
    rq = int(z['robot_qadr'])
    data = mujoco.MjData(model)
    body_position, body_quaternion = [], []
    for q in z['qpos']:
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        body_position.append(data.xpos.copy())
        body_quaternion.append(data.xquat.copy())

    edge_step, edge_time, geom_ids, body_ids, pos, force, torque, normal, distance = ([] for _ in range(9))
    for step, record in enumerate(contacts):
        for pair in record['pairs']:
            if abs(float(pair['solver_contact_time_s']) - float(times[step])) > 1e-8:
                raise ValueError('Contact solver timestamp disagrees with its integration interval')
            pair_geoms = np.asarray(pair['geom_ids'], dtype=np.int64)
            pair_bodies = np.asarray(pair['body_ids'], dtype=np.int64)
            if (pair_geoms.shape != (2,) or pair_bodies.shape != (2,) or
                    np.any(pair_geoms < 0) or np.any(pair_geoms >= model.ngeom) or
                    not np.array_equal(model.geom_bodyid[pair_geoms], pair_bodies)):
                raise ValueError('Contact graph geometry/body mapping disagrees with the model')
            edge_step.append(step)
            edge_time.append(pair['solver_contact_time_s'])
            geom_ids.append(pair['geom_ids'])
            body_ids.append(pair['body_ids'])
            pos.append(pair['contact_pos_world_m'])
            force.append(pair['force_world_on_geom2_N'])
            torque.append(pair['torque_world_on_geom2_Nm'])
            normal.append(pair['normal_force_N'])
            distance.append(pair['contact_distance_m'])
    a.output_dir.mkdir(parents=True, exist_ok=True)
    out = a.output_dir / 'hoi_motion_physics.npz'
    payload = dict(
        state_time_s=times, control_interval_start_time_s=times[:-1],
        root_pose_xyz_wxyz=z['qpos'][:, rq:rq+7],
        joint_names=names, joint_position_rad=z['qpos'][:, qadr],
        joint_velocity_rad_s=z['qvel'][:, vadr],
        joint_control_target_rad=z['ctrl'][:, aids],
        joint_drive_torque_Nm=z['actuator_force'][:, aids] * model.actuator_gear[aids, 0],
        object_pose_xyz_wxyz=z['qpos'][:, oq:oq+7],
        body_names=np.asarray([mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or 'world'
                               for i in range(model.nbody)]),
        body_position_world_m=np.asarray(body_position), body_orientation_wxyz=np.asarray(body_quaternion),
        geom_names=np.asarray([mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or str(i)
                               for i in range(model.ngeom)]),
        contact_interval_index=np.asarray(edge_step, dtype=np.int64),
        contact_solver_time_s=np.asarray(edge_time),
        contact_geom_ids=np.asarray(geom_ids, dtype=np.int64).reshape(-1, 2),
        contact_body_ids=np.asarray(body_ids, dtype=np.int64).reshape(-1, 2),
        contact_position_world_m=np.asarray(pos).reshape(-1, 3),
        contact_force_world_on_geom2_N=np.asarray(force).reshape(-1, 3),
        contact_torque_world_on_geom2_Nm=np.asarray(torque).reshape(-1, 3),
        contact_normal_force_N=np.asarray(normal), contact_signed_distance_m=np.asarray(distance),
        reference_time_s=ref['time_s'], reference_robot_qpos=ref['qpos'],
        reference_object_pose_xyz_wxyz=ref['object_target_pose'])
    # Preserve observations and adaptation metadata verbatim, with a prefix.
    for key in ref.files:
        if key not in ('time_s', 'qpos', 'object_target_pose'):
            payload['source_' + key] = ref[key]
    np.savez_compressed(out, **payload)
    meta = {
        'schema': 'vita_hoi_motion_physics.v1', 'status': 'accepted_assumed_physics_demo',
        'coordinates': {'up': 'Z', 'length_unit': 'metre', 'quaternion_order': 'wxyz'},
        'state_frames': len(times), 'physics_steps': len(times)-1,
        'duration_s': float(times[-1]-times[0]), 'contact_points': len(edge_step),
        'robot_dof': len(names), 'object_id': report['object_id'],
        'timing': 'State N is after interval N-1; contacts and actuator forces are solver outputs for each interval start.',
        'contact_graph': 'One edge per contact point and interval; geom/body ids index the included name arrays.',
        'force_convention': 'World wrench on geom2 at contact point; negate for geom1. Not a wrench about a body COM.',
        'force_provenance': 'MuJoCo simulated contact forces and actuator torques, not sensor measurements.',
        'physical_parameters': 'Assumed collision proxies, mass and friction; not identified from measurements.',
        'controller': report['controller'], 'quality_checks': report['checks'],
        'reference_object_pose_equals_raw_source': raw_pose_equal,
        'raw_source_object_position_error_m_max': report.get('raw_source_object_position_error_m_max'),
        'synthetic_task_adaptation': hand_adjusted or bool(report.get('synthetic_task_adaptation', False)),
        'legacy_report_synthetic_task_adaptation': report.get('synthetic_task_adaptation'),
        'reference_adaptation': adaptation,
        'source_bundle_sha256': reference_scalar('source_bundle_sha256'),
        'reference_metadata_sidecar': {'path': str(sidecar_path.resolve()), 'sha256': sha(sidecar_path)}
            if sidecar else None,
        'quality_thresholds': report['quality_thresholds'], 'limitations': report['limitations'],
        'source': {k: report['input'][k] for k in ('model', 'reference')},
        'trace_sha256': sha(trace_path), 'contacts_sha256': sha(contact_path),
        'replay_audit_sha256': sha(trial/'independent_replay_audit.json'),
        'dataset': {'path': str(out.resolve()), 'sha256': sha(out), 'bytes': out.stat().st_size}}
    (a.output_dir/'metadata.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(meta, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
