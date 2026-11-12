"""Run the supplied UMR surface-correspondence method on the shared Vita52 case.

This is an input/model adapter and process runner. It does not substitute IK for
UMR, alter the shared scene, or execute a dynamics rollout.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def prepare(args):
    import numpy as np
    import mujoco
    import trimesh
    from scipy.spatial.transform import Rotation
    sys.path.insert(0, str(args.code / 'scripts'))
    from humanoid_retarget_config import load_config
    source_manifest = json.loads((args.shared / 'source_manifest.json').read_text())
    data = args.shared / 'rgb_174114_sim_handoff'
    model = Path(source_manifest['model'])
    support = Path(source_manifest['support_geometry'])
    m = mujoco.MjModel.from_xml_path(str(model))
    assert m.nq == 59 and m.nv == 58
    assert sha(model) == source_manifest['model_sha256']
    assert sha(support) == source_manifest['support_geometry_sha256']
    for name in ['inputs', 'configs', 'objects', 'results', 'logs', 'correspondence']:
        (args.output / name).mkdir(parents=True, exist_ok=True)
    timeline = np.load(data / 'timeline.npz', allow_pickle=False)
    frames, times = timeline['frame_ids'], timeline['pts_seconds']
    assert len(frames) == 172 and np.allclose(times, np.arange(172) / 30, atol=1e-6)
    spec = json.loads(support.read_text())
    objects = args.output / 'objects'
    object_rows = []
    import csv
    for seat in spec['seats']:
        name = seat['name']
        size = [2 * v for v in seat['half_size_xy_m']] + [seat['thickness_m']]
        center = seat['center_xy_m'] + [seat['top_z_m'] - seat['thickness_m'] / 2]
        mesh = objects / (name + '.obj')
        trimesh.creation.box(extents=size).export(mesh)
        half = ' '.join(str(v / 2) for v in size)
        xml = (f'<mujoco><asset><mesh name="{name}_mesh" file="{mesh}"/></asset>'
               f'<worldbody><body name="{name}"><freejoint/>'
               f'<geom type="mesh" mesh="{name}_mesh" group="2" contype="0" conaffinity="0"/>'
               f'<geom name="{name}_collision" type="box" size="{half}" group="3"/>'
               '</body></worldbody></mujoco>')
        (objects / (name + '.xml')).write_text(xml)
        quat = Rotation.from_euler('z', seat['yaw_rad']).as_quat()
        with (objects / ('prop_' + name + '.csv')).open('w') as stream:
            writer = csv.writer(stream)
            writer.writerow(['frame_id', 'timestamp', 'px', 'py', 'pz', 'qx', 'qy', 'qz', 'qw'])
            writer.writerows([int(f), float(t), *center, *quat] for f, t in zip(frames, times))
        object_rows.append({'name': name, 'center': center, 'size': size, 'yaw': seat['yaw_rad']})
    base = load_config(args.code / 'robot_configs/humanoid_retarget_vita02a_example.json')
    base.pop('_config_path', None)
    base.pop('_config_dir', None)
    base['smplx_model_dir'] = str(args.smplx)
    base['robot'].update(name='vita52_common', slot_name='vita52_common', xml=str(model),
                         point_cloud_center='body:base_link', height=1.0)
    base['correspondence']['slots'] = str(args.output / 'correspondence/correspondence_slots_final.npz')
    base['correspondence']['dataset'].update(out=str(args.output / 'correspondence/dataset.npz'),
            num_points=args.points, smpl_models=[{'type': 'smplx', 'dir': str(args.smplx), 'genders': ['neutral']}])
    base['correspondence']['train'].update(out_dir=str(args.output / 'correspondence'),
            template_name='smplx_neutral', device='cuda:0', epochs=args.epochs, log_every=50)
    base['retarget'].update(human_height=1.0, source_ground_align='none', smplx_device='cuda:0',
            smplx_batch_size=32, zero_source_finger_pose=True)
    base['hsi_hoi'] = {'source_sequence_dir': str(objects), 'object': {'output_up': 'z',
            'convert_y_up': False, 'ground_align': False, 'object_scale': 1., 'multiple_objects': True}}
    base['solver'].update(body_segment={'module': 'retarget_body_segment_surface_hoi_hsi',
            'schema': 'smplx_55_to_19_upper_arm_v1', 'upper_arm_split': {'ratio_start': 0., 'ratio_end': .3}},
            self_contact_map_cost=0., ground_contact_map_cost=30., ground_contact_anchor_cost=10.,
            object_contact_map_cost=30., object_contact_map_threshold=.10,
            object_contact_map_snap_threshold=.015, object_contact_map_samples=4096,
            object_contact_map_max_points=256, retarget_object_size='original',
            robot_object_penetration_soft_cost=1000., robot_object_hard_constraint=True,
            robot_object_margin=0., robot_object_hard_slack=True, robot_object_hard_slack_cost=3000.,
            robot_object_hard_failure_fallback_to_soft=True,
            robot_object_threshold=.06, robot_object_max_pairs=128,
            iters=args.iters, pose_init_iters=60, trajectory_warm_start_mode='sequential',
            trajectory_filter_mode='off', temporal_smooth_cost=.4)
    base['view'] = {'enabled': False}
    for person in [1, 2]:
        name = f'person_{person:02d}'
        params = np.load(data / f'human/{name}/smplx_params_world.npz', allow_pickle=False)
        poses = np.concatenate([params['global_orient'], params['body_pose'],
                 np.zeros((len(frames), 9)), params['left_hand_pose'], params['right_hand_pose']], axis=1)
        assert poses.shape == (172, 165) and np.isfinite(poses).all()
        assert np.isfinite(params['transl']).all()
        # Input adapter has been independently checked against provided world vertices.
        inp = args.output / 'inputs' / (name + '.npz')
        np.savez_compressed(inp, poses=poses.astype(np.float32), trans=params['transl'],
                betas=params['betas'][0], gender='neutral', mocap_frame_rate=30., output_up='y',
                frame_ids=frames, pts_seconds=times)
        config = json.loads(json.dumps(base))
        config['motion'] = {'data': str(inp), 'seq_key': name, 'start': 0, 'end': -1,
                            'stride': 1, 'max_frames': 0}
        config['retarget']['out'] = str(args.output / 'results' / (name + '_native.npz'))
        write(args.output / 'configs' / (name + '.json'), config)
    write(args.output / 'preparation.json', {
        'method': 'UMR supplied source with official NPZ loader compatibility update',
        'shared_input_manifest': str(args.shared / 'source_manifest.json'),
        'shared_input_manifest_sha256': sha(args.shared / 'source_manifest.json'),
        'model': str(model), 'model_sha256': sha(model), 'nq': m.nq, 'nv': m.nv,
        'source_scale': 1.0, 'source_timing': 'original172frames30fps0..5.7sec',
        'scale_configuration': 'robot.height=1 and retarget.human_height=1 enforce a ratio of1; these are scale parameters, not measured robot/human heights',
        'seats': object_rows, 'seat_policy': 'exact shared fitted seat top colliders; no separate refitting',
        'hand_observed_accuracy_claim': False, 'is_dynamics': False,
        'solver_iters': args.iters, 'correspondence_points': args.points,
        'correspondence_epochs': args.epochs, 'gpu_physical_index': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'limitations': ['Hands are unobserved/default source pose.',
                       'Native solver hard-object constraints may explicitly fall back to soft; final independent collision metrics must be checked.',
                       'Actors are independently retargeted; joint two-robot avoidance is not built into this initial baseline.']})


def run_pipeline(args, stage, person=1):
    command = [sys.executable, '-u', str(args.code / 'scripts/humanoid_retarget_pipeline.py'),
               '--config', str(args.output / f'configs/person_{person:02d}.json'),
               '--stage', stage, '--skip-view']
    log = args.output / 'logs' / f'{stage}_person{person:02d}.log'
    begin = time.monotonic()
    with log.open('w') as stream:
        process = subprocess.run(command, cwd=args.code, stdout=stream, stderr=subprocess.STDOUT)
    receipt = {'command': command, 'returncode': process.returncode,
               'wall_seconds': time.monotonic() - begin, 'log': str(log)}
    write(args.output / 'logs' / f'{stage}_person{person:02d}.receipt.json', receipt)
    if process.returncode:
        raise RuntimeError(f'{stage} failed: see {log}')
    return receipt


def export(args):
    import numpy as np
    spec = json.loads((args.shared / 'source_manifest.json').read_text())
    rows, native = [], []
    for person in [1, 2]:
        path = args.output / f'results/person_{person:02d}_native.npz'
        with np.load(path, allow_pickle=True) as data:
            q = np.asarray(data['qpos'], dtype=np.float64)
            assert q.shape == (172, 59), (path, q.shape)
            assert np.isfinite(q).all()
            assert abs(float(np.asarray(data['smpl_scale']).reshape(-1)[0]) - 1.) < 1e-6
            rows.append(q)
            native.append({'path': str(path), 'sha256': sha(path), 'fields': data.files})
    output = args.output / 'umr_vita52.npz'
    np.savez_compressed(output, qpos=np.stack(rows), times=np.arange(172) / 30.,
              dof_names=np.asarray(spec['dof_names']), frame_ids=np.arange(172),
              method=np.asarray('UMR learned surface correspondence'),
              source_manifest_sha256=np.asarray(sha(args.shared / 'source_manifest.json')))
    report = json.loads((args.output / 'preparation.json').read_text())
    report.update(status='all172_native_umr_retargeted', output=str(output), output_sha256=sha(output),
                  native_outputs=native, independent_common_evaluation='parent evaluator',
                  solver_receipts=[json.loads(path.read_text()) for path in sorted((args.output / 'logs').glob('*.receipt.json'))])
    write(args.output / 'method_report.json', report)
    print(json.dumps({'status': report['status'], 'output': str(output)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path('/data/KinoGen/outputs/retarget_compare_20261028')
    parser.add_argument('--output', type=Path, default=base / 'umr')
    parser.add_argument('--code', type=Path, default=base / 'umr/source')
    parser.add_argument('--shared', type=Path, default=base / 'shared_input')
    parser.add_argument('--smplx', type=Path, default=Path('/data/grail/imports/GEM-SMPL/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz'))
    parser.add_argument('--stage', choices=['prepare', 'build', 'train', 'retarget', 'export', 'all'], default='all')
    parser.add_argument('--points', type=int, default=4096)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--iters', type=int, default=6)
    args = parser.parse_args()
    if args.stage in ['prepare', 'all']:
        prepare(args)
    if args.stage in ['build', 'all']:
        print(run_pipeline(args, 'build'), flush=True)
    if args.stage in ['train', 'all']:
        print(run_pipeline(args, 'train'), flush=True)
    if args.stage in ['retarget', 'all']:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            for result in pool.map(lambda person: run_pipeline(args, 'retarget', person), [1, 2]):
                print(result, flush=True)
    if args.stage in ['export', 'all']:
        export(args)


if __name__ == '__main__':
    main()
