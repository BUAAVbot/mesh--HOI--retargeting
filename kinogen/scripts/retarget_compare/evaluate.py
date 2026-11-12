"""Common kinematic-only evaluation on unchanged Vita52 and scene geometry."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time


METHODS = {'gmr': 'GMR / Vita adapter', 'custom_ik': 'Custom IK',
           'custom_ik_contact': 'Custom IK + contact', 'umr': 'UMR / Vita adapter',
           'omniretarget': 'OmniRetarget / Vita adapter', 'hoi_retarget': 'HOI-Retarget / Vita adapter',
           'phc': 'PHC fitting / Vita adapter', 'umr_refined': 'UMR + native refinement',
           'umr_contact_adapted': 'UMR + inferred contact',
           'umr_postprocessed': 'UMR + bundled ground/SQP'}
BODY_MAP = [('base_link', 0), ('left_hip_roll_link', 1), ('right_hip_roll_link', 2),
            ('left_knee_link', 4), ('right_knee_link', 5), ('left_ankle_roll_link', 7),
            ('right_ankle_roll_link', 8), ('left_shoulder_pitch_link', 16),
            ('right_shoulder_pitch_link', 17), ('left_elbow_link', 18),
            ('right_elbow_link', 19), ('left_wrist_pitch_link', 20),
            ('right_wrist_pitch_link', 21), ('neck_yaw_link', 12)]
BONES = [(1, 3), (2, 4), (3, 5), (4, 6), (7, 9), (8, 10), (9, 11), (10, 12), (1, 2), (7, 8)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def find_reference(root, method):
    alternate = {'umr_refined': 'umr/refined/umr_vita52.npz',
                 'umr_contact_adapted': 'umr/contact_adapted/umr_vita52.npz',
                 'umr_postprocessed': 'umr/postprocessed/umr_vita52.npz'}.get(method)
    if alternate:
        path = root / alternate
        return path if path.is_file() else None
    for name in ('reference.npz', 'umr_vita52.npz', 'phc_vita52.npz'):
        path = root / method / name
        if path.is_file():
            return path
    return None


def load_reference(path, names, np):
    with np.load(path, allow_pickle=False) as archive:
        qpos = np.asarray(archive['qpos'], float)
        times = np.asarray(archive[next(key for key in ('time_s', 'times', 'time', 'source_time_s') if key in archive)], float)
        given_names = [str(name) for name in archive['dof_names']]
    if qpos.shape == (2, len(times), 59):
        qpos = qpos.transpose(1, 0, 2)
    if qpos.shape != (172, 2, 59) or len(times) != 172:
        raise ValueError(f'Full172frames/2actors/Vita52 required, found {qpos.shape}')
    if len(given_names) != 52 or len(set(given_names)) != 52 or set(given_names) != set(names):
        raise ValueError('Joint-name contract mismatch')
    qpos[:, :, 7:] = qpos[:, :, [7 + given_names.index(name) for name in names]]
    if not np.isfinite(qpos).all() or not np.allclose(times, np.arange(172) / 30., atol=1e-6):
        raise ValueError('Nonfinite values or changed source timing')
    quaternion_norm_error = np.abs(np.linalg.norm(qpos[:, :, 3:7], axis=-1) - 1)
    if quaternion_norm_error.max() > 1e-4:
        raise ValueError('Invalid free-root quaternion')
    return qpos, times


def evaluate(root, method, path, common, manifest, model, mj, np):
    started = time.monotonic()
    names = manifest['dof_names']
    qpos, times = load_reference(path, names, np)
    n = len(times)
    source = common['repaired_source_joints_zup']
    original = common['raw_source_joints_zup']
    repaired = common['source_repaired_frame_mask']
    valid_observation = ~repaired
    robot = np.empty((n, 2, len(BODY_MAP), 3))
    ground_distance = np.empty((n, 2, 2))
    seat_distance = np.empty((n, 2))
    penetration = np.zeros((n, 4))
    support = json.loads(Path(manifest['support_geometry']).read_text())
    floor = np.asarray(support['ground']['z_eq_ax_by_c'])
    seat_centers = np.asarray([seat['center_xy_m'] for seat in support['seats']])
    data = mj.MjData(model)
    ids = [[mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, f'p{actor}_{name}') for name, _ in BODY_MAP] for actor in (1, 2)]
    if min(min(row) for row in ids) < 0:
        raise ValueError('Body mapping missing from the common model')
    root_q = [int(model.jnt_qposadr[next(j for j in range(model.njnt)
              if model.jnt_type[j] == mj.mjtJoint.mjJNT_FREE and model.jnt_bodyid[j] == ids[actor][0])]) for actor in (0, 1)]
    geom_names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, g) or '' for g in range(model.ngeom)]
    ground_id = geom_names.index('ground_proxy')
    seats = [geom_names.index(f'seat_{i}_proxy') for i in (1, 2)]
    feet = [[[g for g, name in enumerate(geom_names) if name.startswith(f'p{actor}_collision_foot_{side}_')]
             for side in ('l', 'r')] for actor in (1, 2)]
    hips = [[g for g, name in enumerate(geom_names) if name.startswith(f'p{actor}_collision_') and
             ('hip_roll_cylinder' in name or 'hip_front_roll_cylinder' in name)] for actor in (1, 2)]
    def pair_class(g1, g2):
        a, b = geom_names[g1], geom_names[g2]
        pa = 1 if a.startswith('p1_') else 2 if a.startswith('p2_') else 0
        pb = 1 if b.startswith('p1_') else 2 if b.startswith('p2_') else 0
        if pa and pb and pa != pb:
            return 3
        if g1 == ground_id or g2 == ground_id:
            return 1
        if ('seat_' in a and not pa) or ('seat_' in b and not pb):
            return 2
        return 0
    for frame in range(n):
        for actor, address in enumerate(root_q):
            data.qpos[address:address + 59] = qpos[frame, actor]
        mj.mj_kinematics(model, data)
        mj.mj_comPos(model, data)
        mj.mj_collision(model, data)
        for actor in (0, 1):
            robot[frame, actor] = data.xpos[ids[actor]]
            for side in (0, 1):
                ground_distance[frame, actor, side] = min(float(mj.mj_geomDistance(model, data, g, ground_id, 2., None)) for g in feet[actor][side])
            expected = int(np.argmin(np.linalg.norm(seat_centers - source[frame, actor, 0, :2], axis=1)))
            seat_distance[frame, actor] = min(float(mj.mj_geomDistance(model, data, g, seats[expected], 2., None)) for g in hips[actor])
        for contact in data.contact[:data.ncon]:
            depth = max(0., -float(contact.dist))
            penetration[frame, 0] = max(penetration[frame, 0], depth)
            kind = pair_class(int(contact.geom1), int(contact.geom2))
            if kind:
                penetration[frame, kind] = max(penetration[frame, kind], depth)
    selected = source[:, :, [joint for _, joint in BODY_MAP]]
    raw_selected = original[:, :, [joint for _, joint in BODY_MAP]]
    relative_error = np.linalg.norm((robot - robot[:, :, :1]) - (selected - selected[:, :, :1]), axis=-1)
    direction_errors = []
    for parent, child in BONES:
        rv = robot[:, :, child] - robot[:, :, parent]
        sv = selected[:, :, child] - selected[:, :, parent]
        cosine = np.sum(rv * sv, axis=-1) / np.maximum(np.linalg.norm(rv, axis=-1) * np.linalg.norm(sv, axis=-1), 1e-12)
        direction_errors.append(np.degrees(np.arccos(np.clip(cosine, -1., 1.))))
    direction_errors = np.stack(direction_errors, axis=-1)
    root_xy_error = np.linalg.norm(qpos[:, :, :2] - source[:, :, 0, :2], axis=-1)
    source_ankles = source[:, :, [7, 8]]
    source_floor = source_ankles[..., :2] @ floor[:2] + floor[2]
    height = source_ankles[..., 2] - source_floor
    source_speed = np.linalg.norm(np.gradient(source_ankles[..., :2], times, axis=0), axis=-1)
    source_height_floor = np.quantile(height, .1, axis=0)
    stance = (height < source_height_floor[None] + .055) & (source_speed < .18)
    ankle_speed = np.linalg.norm(np.gradient(robot[:, :, [5, 6], :2], times, axis=0), axis=-1)
    stance_values = ankle_speed[stance]
    joint_positions = qpos[:, :, 7:]
    velocity = np.gradient(joint_positions, times, axis=0)
    acceleration = np.gradient(velocity, times, axis=0)
    limits = []
    for actor in (1, 2):
        joints = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, f'p{actor}_{name}') for name in names]
        limits.append(model.jnt_range[joints])
    limits = np.asarray(limits)
    violation = np.maximum(limits[None, :, :, 0] - joint_positions, joint_positions - limits[None, :, :, 1])
    violation = np.maximum(violation, 0)
    main_joints = [i for i, name in enumerate(names) if not name.startswith(('lh_', 'rh_', 'neck_'))]
    ends = np.r_[np.arange(7), np.arange(n - 7, n)]
    end_gaps = seat_distance[ends]
    metrics = {
        'mean_bone_direction_error_deg': float(direction_errors.mean()),
        'p95_bone_direction_error_deg': float(np.quantile(direction_errors, .95)),
        'clean_observation_mean_bone_direction_error_deg': float(direction_errors[valid_observation].mean()),
        'unscaled_root_relative_mpjpe_m': float(relative_error.mean()),
        'root_xy_rmse_m': float(np.sqrt(np.mean(root_xy_error ** 2))),
        'root_xy_max_m': float(root_xy_error.max()),
        'root_xy_max_per_actor_m': root_xy_error.max(axis=0).tolist(),
        'max_penetration_m': float(penetration[:, 0].max()),
        'max_ground_penetration_m': float(penetration[:, 1].max()),
        'max_seat_penetration_m': float(penetration[:, 2].max()),
        'max_interperson_penetration_m': float(penetration[:, 3].max()),
        'seat_endpoint_absolute_distance_mean_m': float(np.abs(end_gaps).mean()),
        'seat_endpoint_within_2cm_fraction': float((np.abs(end_gaps) <= .02).mean()),
        'initial_seat_gap_per_actor_m': seat_distance[:7].mean(axis=0).tolist(),
        'final_seat_gap_per_actor_m': seat_distance[-7:].mean(axis=0).tolist(),
        'stance_foot_speed_mean_m_s': float(stance_values.mean()) if stance_values.size else None,
        'stance_foot_speed_p95_m_s': float(np.quantile(stance_values, .95)) if stance_values.size else None,
        'max_joint_limit_violation_rad': float(violation.max()),
        'max_body_joint_speed_rad_s': float(np.abs(velocity[:, :, main_joints]).max()),
        'body_joint_acceleration_rms_rad_s2': float(np.sqrt(np.mean(acceleration[:, :, main_joints] ** 2))),
        'frames_with_joint_limit_violation_over_1mrad': int((violation.max(axis=(1, 2)) > .001).sum()),
    }
    directory = root / method
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / 'evaluation_arrays.npz', time_s=times, qpos=qpos,
                        robot_body_positions=robot, source_body_positions=selected,
                        raw_source_body_positions=raw_selected, source_repaired_frame_mask=repaired,
                        bone_direction_error_deg=direction_errors, root_xy_error_m=root_xy_error,
                        signed_foot_ground_distance_m=ground_distance, signed_hip_seat_distance_m=seat_distance,
                        penetration_m=penetration, source_inferred_stance_mask=stance,
                        actual_ankle_xy_speed_m_s=ankle_speed, joint_limit_violation_rad=violation,
                        body_joint_velocity_rad_s=velocity[:, :, main_joints],
                        body_joint_acceleration_rad_s2=acceleration[:, :, main_joints])
    result = {'method': method, 'label': METHODS[method], 'status': 'full_kinematic_output_evaluated',
              'reference': str(path), 'reference_sha256': sha(path),
              'frames': n, 'actors': 2, 'duration_s': float(times[-1]), 'metrics': metrics,
              'elapsed_evaluation_s': time.monotonic() - started,
              'physics_rollout_executed': False, 'physics_success': None}
    (directory / 'evaluation.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/data/KinoGen/outputs/retarget_compare_20261028'))
    args = parser.parse_args()
    import numpy as np
    import mujoco as mj
    manifest_path = args.root / 'shared_input/source_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    with np.load(args.root / 'shared_input/common_observations.npz', allow_pickle=False) as archive:
        common = {key: archive[key] for key in archive.files}
    model_path = Path(manifest['model']).parent / 'two_vita_solver20.xml'
    model = mj.MjModel.from_xml_path(str(model_path))
    rows, unavailable, errors = [], [], []
    for method in METHODS:
        reference = find_reference(args.root, method)
        if reference is None:
            unavailable.append(method)
            continue
        try:
            result = evaluate(args.root, method, reference, common, manifest, model, mj, np)
            rows.append(result)
            print(json.dumps({'method': method, 'metrics': result['metrics']}), flush=True)
        except Exception as error:
            errors.append({'method': method, 'error': repr(error)})
    summary = {'schema': 'retarget_comparison.common_evaluation.v1',
               'protocol_sha256': sha(manifest_path), 'common_scene': str(model_path),
               'common_scene_sha256': sha(model_path), 'results': rows,
               'not_yet_available': unavailable, 'evaluation_errors': errors,
               'phase': 'retargeting_only; no policy/physics-success benchmark',
               'metric_notes': {
                   'pose': 'Direction error is morphology-tolerant; relativeMPJPE is intentionally unscaled and anatomy-dependent.',
                   'stance': 'Common inference from repaired human ankle height/speed, not observed contact groundtruth.',
                   'seat_gap': 'Signed collision distance of hip collision proxies to source-nearest fixed seat; near zero is geometric proximity, not load bearing.',
                   'penetration': 'Shared MuJoCo enabled collision pairs; not a complete independent self-collision certificate.',
                   'hands': 'No valid hand observations; cannot compare finger reconstruction or grasping accuracy.',
                   'timing': 'All172 original30Hz frames; common repairmask excludes actor2frames63-69 in the clean-subset metric.',
                   'scope': 'One clip with local embodiment adapters; no original-paper benchmark ranking or generalization claim.'}}
    (args.root / 'comparison_metrics.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    if errors:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
