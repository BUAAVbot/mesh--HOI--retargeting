"""Verify downloaded final artifacts using the Python standard library only."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import struct
import zipfile


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def npz_shapes(path):
    result = {}
    with zipfile.ZipFile(path) as z:
        if z.testzip() is not None:
            raise ValueError('Corrupt NPZ archive')
        for name in z.namelist():
            with z.open(name) as f:
                if f.read(6) != b'\x93NUMPY':
                    raise ValueError('Invalid NPY magic')
                version = tuple(f.read(2))
                fmt = '<H' if version == (1, 0) else '<I'
                n = struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
                header = ast.literal_eval(f.read(n).decode('utf-8').strip())
                if 'O' in header['descr']:
                    raise ValueError('Object array is not permitted')
                result[name[:-4]] = list(header['shape'])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    a = p.parse_args()
    root = a.root.resolve()
    base = root/'outputs/fourth_vita_demo'
    trial = base/'physics_source_slide_v12_final'
    r = read(trial/'report.json')
    audit = read(trial/'independent_replay_audit.json')
    meta = read(base/'hoi_v12_final/metadata.json')
    checked = {}
    inputs = [(trial/'physical_trace.npz', r['artifacts']['trace']['sha256']),
              (trial/'physical_contacts.json', r['artifacts']['contacts']['sha256']),
              (base/'assets/vita_object_05_dynamic_ankle100.xml', r['input']['model']['sha256']),
              (base/'retarget_source_slide_v12.npz', r['input']['reference']['sha256']),
              (base/'hoi_v12_final/hoi_motion_physics.npz', meta['dataset']['sha256'])]
    for stem in ['fourth_vita_physics_v12', 'fourth_vita_ego_v12']:
        render = read(root/f'previews/{stem}.render.json')
        if render['trace_sha256'] != r['artifacts']['trace']['sha256']:
            raise ValueError('Video uses a different trace')
        inputs.append((root/f'previews/{stem}.mp4', render['video_sha256']))
        inputs.append((root/f'previews/{stem}_contactsheet.jpg', render['contactsheet_sha256']))
        if 'synthetic_ego_camera' in render:
            inputs.append((root/f'previews/{stem}.ego_camera.npz', render['synthetic_ego_camera']['sha256']))
    for path, expected in inputs:
        actual = sha(path)
        if actual != expected:
            raise ValueError('Download hash mismatch: '+str(path))
        checked[str(path.relative_to(root))] = actual
    if (not r['simulation_passed'] or len(r['checks']) != 12 or not all(r['checks'].values()) or
            audit['status'] != 'deterministic_physics_trace_verified' or
            audit['replayed_qpos_max_abs_error'] != 0 or audit['replayed_qvel_max_abs_error'] != 0):
        raise ValueError('Final physical or replay quality checks failed')
    if not meta['reference_adaptation']['synthetic_hand_contact_adaptation'] or meta['reference_adaptation']['source_object_trajectory_modified']:
        raise ValueError('Unexpected provenance labels')
    shapes = npz_shapes(base/'hoi_v12_final/hoi_motion_physics.npz')
    expected = {'state_time_s':[401], 'root_pose_xyz_wxyz':[401,7],
                'joint_position_rad':[401,52], 'joint_drive_torque_Nm':[400,52],
                'object_pose_xyz_wxyz':[401,7], 'reference_time_s':[31]}
    for key, shape in expected.items():
        if shapes.get(key) != shape:
            raise ValueError('Unexpected dataset dimension: '+key)
    out = {'status':'local_delivery_verified', 'checked_sha256':checked,
           'physical_checks_passed':12, 'replay_state_error':0,
           'dataset_shapes':shapes, 'local_simulation_run':False}
    path = root/'reports/fourth_vita_v12_delivery_verification.json'
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps({'status':out['status'], 'hashed_files':len(checked), 'dataset_arrays':len(shapes)}))


if __name__ == '__main__':
    main()
