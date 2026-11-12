"""Run the pinned upstream GMR solver on the shared two-person Vita52 input.

The Vita52 target mapping is a disclosed local embodiment adapter. The actual
solver is GeneralMotionRetargeting + Mink/DAQP, without scene/contact correction.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    with Path(path).open('rb') as stream:
        value = hashlib.sha256()
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(chunk)
        return value.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--support', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gmr-source', type=Path, default=Path('/data/grail-tools/gmr-pinned-source'))
    p.add_argument('--frames', type=int, default=172)
    p.add_argument('--solver', default='daqp')
    p.add_argument('--damping', type=float, default=0.5)
    p.add_argument('--initialization-passes', type=int, default=3)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('Use a new output directory: ' + str(a.output))
    a.output.mkdir(parents=True)
    begun = time.perf_counter()
    import numpy as np
    import mujoco as mj
    from scipy.spatial.transform import Rotation
    sys.path.insert(0, str(a.gmr_source))
    import general_motion_retargeting.motion_retarget as gmr_module

    model = mj.MjModel.from_xml_path(str(a.model.resolve()))
    if (model.nq, model.nv) != (59, 58):
        raise ValueError('Expected the shared free-root Vita52 model')
    jids = np.array([i for i in range(model.njnt) if model.jnt_type[i] == mj.mjtJoint.mjJNT_HINGE])
    jids = jids[np.argsort(model.jnt_qposadr[jids])]
    names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, int(i)) for i in jids]
    aux = np.array([model.jnt_qposadr[j] for j, n in zip(jids, names)
                    if n.startswith(('lh_', 'rh_', 'neck_'))], dtype=int)

    # SMPL-X neutral local axes: +X anatomical left, +Y up, +Z forward.
    # Vita body axes: +X forward, +Y left, +Z up. Keep scene/world unscaled.
    C = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
    spec = json.loads(a.support.read_text())
    U = np.asarray(spec['source_to_sim_U'], float)
    td = mj.MjData(model)
    td.qpos[:] = model.qpos0
    td.qpos[2] = 1.
    for side, sign in [('left', 1), ('right', -1)]:
        for suffix, value in [('shoulder_roll_joint', sign * np.pi / 2), ('elbow_joint', np.pi / 2)]:
            jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, side + '_' + suffix)
            td.qpos[model.jnt_qposadr[jid]] = value
    mj.mj_forward(model, td)

    matches = [
        ('base_link', 'pelvis', 0, 100., 10.),
        ('waist_pitch_link', 'spine3', 9, 10., 10.),
        ('left_hip_roll_link', 'left_hip', 1, 10., 5.),
        ('right_hip_roll_link', 'right_hip', 2, 10., 5.),
        ('left_knee_link', 'left_knee', 4, 10., 5.),
        ('right_knee_link', 'right_knee', 5, 10., 5.),
        ('left_ankle_roll_link', 'left_ankle', 7, 100., 50.),
        ('right_ankle_roll_link', 'right_ankle', 8, 100., 50.),
        ('left_shoulder_yaw_link', 'left_shoulder', 16, 10., 5.),
        ('right_shoulder_yaw_link', 'right_shoulder', 17, 10., 5.),
        ('left_elbow_link', 'left_elbow', 18, 10., 5.),
        ('right_elbow_link', 'right_elbow', 19, 10., 5.),
        ('left_wrist_pitch_link', 'left_wrist', 20, 20., 5.),
        ('right_wrist_pitch_link', 'right_wrist', 21, 20., 5.),
    ]
    table1, table2 = {}, {}
    for robot, human, _, position_weight, rotation_weight in matches:
        bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, robot)
        if bid < 0:
            raise ValueError('Missing target body: ' + robot)
        offset = Rotation.from_matrix(C @ td.xmat[bid].reshape(3, 3)).as_quat(scalar_first=True).tolist()
        table2[robot] = [human, position_weight, rotation_weight, [0., 0., 0.], offset]
        # Upstream's first stage emphasizes body orientations and feet/root.
        first_weight = position_weight if human in ('pelvis', 'left_ankle', 'right_ankle') else 0.
        table1[robot] = [human, first_weight, min(rotation_weight, 10.), [0., 0., 0.], offset]
    config = {'robot_root_name': 'base_link', 'human_root_name': 'pelvis',
              'ground_height': 0., 'human_height_assumption': 1.8,
              'use_ik_match_table1': True, 'use_ik_match_table2': True,
              'human_scale_table': {human: 1. for _, human, _, _, _ in matches},
              'ik_match_table1': table1, 'ik_match_table2': table2}
    config_path = a.output / 'smplx_to_vita52_local_adapter.json'
    config_path.write_text(json.dumps(config, indent=2) + '\n')
    target_key = 'vita52_two_sitting_local'
    gmr_module.ROBOT_XML_DICT[target_key] = a.model.resolve()
    gmr_module.IK_CONFIG_DICT.setdefault('smplx', {})[target_key] = config_path.resolve()

    times = np.asarray(np.load(a.data / 'timeline.npz', allow_pickle=False)['pts_seconds'][:a.frames], float)
    times = times - times[0]
    if len(times) < 2 or not np.all(np.diff(times) > 0):
        raise ValueError('Invalid shared timeline')
    parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
    allq, allj, allrot, person_reports, source_records = [], [], [], [], []
    for person in (1, 2):
        person_begun = time.perf_counter()
        source_dir = a.data / ('human/person_%02d' % person)
        params_path, mesh_path = source_dir / 'smplx_params_world.npz', source_dir / 'smplx_meshes_world.npz'
        params, meshes = np.load(params_path, allow_pickle=False), np.load(mesh_path, allow_pickle=False)
        joints = np.asarray(meshes['joints_world'][:len(times)], float) @ U.T
        pose = np.concatenate([params['global_orient'][:len(times), None],
                               params['body_pose'][:len(times)].reshape(len(times), 21, 3)], axis=1)
        local = Rotation.from_rotvec(pose.reshape(-1, 3)).as_matrix().reshape(len(times), 22, 3, 3)
        rotations = np.zeros_like(local)
        for j in range(22):
            rotations[:, j] = U @ local[:, j] if j == 0 else rotations[:, parents[j]] @ local[:, j]
        if not np.all(np.isfinite(joints)) or not np.all(np.isfinite(rotations)):
            raise ValueError('Shared source still contains nonfinite values')
        retargeter = gmr_module.GeneralMotionRetargeting('smplx', target_key,
                      actual_human_height=None, solver=a.solver, damping=a.damping,
                      verbose=False, use_velocity_limit=False)
        initial = model.qpos0.copy()
        initial[:3] = joints[0, 0]
        initial[3:7] = Rotation.from_matrix(rotations[0, 0] @ C).as_quat(scalar_first=True)
        for side in ('left', 'right'):
            for suffix, value in [('hip_pitch_joint', -1.2), ('knee_joint', 1.7), ('ankle_pitch_joint', -.3)]:
                jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, side + '_' + suffix)
                initial[model.jnt_qposadr[jid]] = value
        retargeter.configuration.update(initial)
        qrows, errors, durations, actual_targets = [], [], [], []
        for frame in range(len(times)):
            frame_begun = time.perf_counter()
            def human_data():
                return {human: [joints[frame, ji].copy(),
                                Rotation.from_matrix(rotations[frame, ji]).as_quat(scalar_first=True)]
                        for _, human, ji, _, _ in matches}
            for _ in range(a.initialization_passes if frame == 0 else 1):
                q = retargeter.retarget(human_data(), offset_to_ground=False)
            # Unobserved hands/neck retain a common neutral reference.
            q[aux] = model.qpos0[aux]
            retargeter.configuration.update(q)
            qrows.append(q.copy())
            errors.append([retargeter.error1(), retargeter.error2()])
            durations.append(time.perf_counter() - frame_begun)
            actual_targets.append([retargeter.scaled_human_data[human][0].copy()
                                   for _, human, _, _, _ in matches])
            if frame % 30 == 0 or frame == len(times)-1:
                print(json.dumps({'person': person, 'frame': frame,
                                  'error2': errors[-1][1], 'elapsed_s': time.perf_counter()-person_begun}), flush=True)
        qrows = np.asarray(qrows)
        if not np.all(np.isfinite(qrows)):
            raise ValueError('Nonfinite native GMR output')
        limit_error = np.maximum(model.jnt_range[jids, 0] - qrows[:, model.jnt_qposadr[jids]],
                                 qrows[:, model.jnt_qposadr[jids]] - model.jnt_range[jids, 1])
        np.savez_compressed(a.output / ('person_%02d_native.npz' % person), qpos=qrows,
                            times=times, dof_names=np.array(names), task_error=np.array(errors),
                            frame_wall_seconds=np.array(durations), target_positions=np.array(actual_targets))
        allq.append(qrows); allj.append(joints); allrot.append(rotations)
        person_reports.append({'person': person, 'frames': len(times),
                               'wall_seconds': time.perf_counter()-person_begun,
                               'median_frame_seconds': float(np.median(durations)),
                               'maximum_joint_limit_violation_rad': float(max(0., limit_error.max())),
                               'maximum_task2_error': float(np.max(np.asarray(errors)[:, 1]))})
        source_records.append({'params': str(params_path), 'params_sha256': sha(params_path),
                               'meshes': str(mesh_path), 'meshes_sha256': sha(mesh_path)})
    reference_path = a.output / 'reference.npz'
    np.savez_compressed(reference_path, qpos=np.asarray(allq), times=times,
                        dof_names=np.array(names), source_joints_zup=np.asarray(allj),
                        source_global_rotation_zup=np.asarray(allrot),
                        matched_robot_bodies=np.array([row[0] for row in matches]),
                        matched_human_indices=np.array([row[2] for row in matches]))
    versions = {}
    for package in ['mujoco', 'mink', 'qpsolvers', 'daqp', 'numpy', 'scipy']:
        try: versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package] = None
    source_py = Path(gmr_module.__file__)
    report = {'schema': 'retarget_compare.gmr_native.v1', 'method': 'GMR',
              'status': 'completed_native_kinematic_retargeting',
              'native_upstream_solver_executed': True, 'physical_simulation_executed': False,
              'robot': 'shared Vita52 local embodiment mapping', 'persons': 2,
              'frames_per_person': len(times), 'source_duration_seconds': float(times[-1]),
              'qpos_shape': list(np.asarray(allq).shape), 'quaternion_order': 'wxyz',
              'world_up': '+Z', 'world_scale': 1., 'world_origin_changed': False,
              'source_smoothing': False, 'scene_or_seat_correction_applied': False,
              'automatic_frame_ground_snap': False, 'source': source_records,
              'model': str(a.model), 'model_sha256': sha(a.model),
              'support': str(a.support), 'support_sha256': sha(a.support),
              'config': str(config_path), 'config_sha256': sha(config_path),
              'gmr_source': str(source_py), 'gmr_source_sha256': sha(source_py),
              'reference': str(reference_path), 'reference_sha256': sha(reference_path),
              'solver': a.solver, 'damping': a.damping, 'initialization_passes': a.initialization_passes,
              'versions': versions, 'persons_report': person_reports,
              'total_wall_seconds': time.perf_counter()-begun,
              'adaptations': ['Pinned upstream GMR has no Vita entry; a local Vita52 body mapping is registered in memory.',
                 'Rotation offsets calibrated from declared Vita neutral T-pose; source positions remain in the shared unscaled world.',
                 'Human root and all local scale table entries are1; no global scene scaling or root recentering.',
                 'Feet map SMPL-X ankles to Vita ankle-roll bodies rather than G1 toe bodies.',
                 'Two independent single-person GMR solves; no inter-person collision or support hard constraints.',
                 'Hands/neck held neutral because source hand observations are invalid; no hand accuracy claim.'],
              'limitations': ['Kinematic output only; not a physical rollout or a SONIC tracking result.',
                 'No original GMR paper benchmark reproduction; this is a Vita embodiment adapter experiment.',
                 'Task residual magnitudes depend on cost weights and are not cross-method comparison metrics.']}
    (a.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({key: report[key] for key in ['status', 'qpos_shape', 'total_wall_seconds']}), flush=True)


if __name__ == '__main__':
    main()
