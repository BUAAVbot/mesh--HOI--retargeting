"""Verify final downloads without loading a simulator or requiring NumPy locally."""
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


def shapes(path):
    out = {}
    with zipfile.ZipFile(path) as z:
        if z.testzip() is not None:
            raise ValueError('Corrupted NPZ CRC')
        for name in z.namelist():
            with z.open(name) as f:
                if f.read(6) != b'\x93NUMPY':
                    raise ValueError('Invalid array magic')
                version = tuple(f.read(2))
                fmt = '<H' if version == (1,0) else '<I'
                count = struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
                header = ast.literal_eval(f.read(count).decode('utf-8').strip())
                if not isinstance(header['descr'], str) or 'O' in header['descr']:
                    raise ValueError('No object arrays permitted')
                out[name[:-4]] = list(header['shape'])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trial', type=Path, required=True)
    p.add_argument('--export', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--video', type=Path, action='append', default=[])
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    r = read(a.trial/'report.json')
    audit = read(a.trial/'independent_replay_audit.json')
    source = read(a.trial/'source_provenance_audit.json')
    meta = read(a.export/'metadata.json')
    required = {'full_source_duration','finite_no_solver_warning','both_root_horizontal_tracking',
                'both_root_height_tracking','both_root_orientation_tracking','both_joint_tracking',
                'penetration','both_seated_support','shoe_source_tracking',
                'hand_shoe_manipulation_coupling','grasp_constraint_causal_motion',
                'grasp_constraint_force_plausible'}
    if (r.get('simulation_passed') is not True or not required.issubset(r.get('checks',{})) or
            any(v is not True for v in r['checks'].values()) or r['source_frames'] != 398):
        raise ValueError('Full physical quality checks have not passed')
    if audit['status'] != 'deterministic_physics_trace_verified' or source['status'] != 'raw_source_provenance_verified':
        raise ValueError('Physics replay and source provenance must pass')
    if audit.get('force_trace_verified') is not True:
        raise ValueError('Independent force trace replay must pass')
    files = [(a.trial/'model.mjb',r['model']['sha256']), (a.trial/'trace.npz',r['trace']['sha256']),
             (a.trial/'contacts.npz',r['contacts']['sha256']), (a.reference,r['input']['reference']['sha256']),
             (a.export/'hoi_motion_physics.npz',meta['dataset']['sha256'])]
    for video in a.video:
        render = read(video.with_suffix('.render.json'))
        if render['trace_sha256'] != r['trace']['sha256'] or render['model_sha256'] != r['model']['sha256']:
            raise ValueError('Video does not correspond to final frozen physics')
        if render['source_frames'] != 398:
            raise ValueError('Video uses a short diagnostic trial')
        files.append((video,render['video_sha256']))
        if 'synthetic_ego_camera' in render:
            files.append((video.with_suffix('.ego_camera.npz'),render['synthetic_ego_camera']['sha256']))
    checked = {}
    for path, expected in files:
        actual = sha(path)
        if actual != expected:
            raise ValueError('Download hash mismatch: '+str(path))
        checked[str(path)] = actual
    dim = shapes(a.export/'hoi_motion_physics.npz')
    n = r['physics_steps']
    for key, expected in {'root_pose_xyz_wxyz':[n+1,2,7], 'joint_position_rad':[n+1,2,52],
                          'joint_drive_torque_Nm':[n,2,52], 'shoe_pose_xyz_wxyz':[n+1,7]}.items():
        if dim.get(key) != expected:
            raise ValueError('Invalid dataset shape: '+key)
    for actor in (1,2):
        prefix = f'source_person{actor:02}_smplx_'
        if not any(k.startswith(prefix) and v and v[0] == 398 for k,v in dim.items()):
            raise ValueError('Missing full source SMPL-X parameters')
    out = {'status':'local_delivery_verified', 'simulation_run_locally':False,
           'source_frames':398, 'physics_steps':n, 'arrays':len(dim),
           'checked_sha256':checked, 'dataset_shapes':dim,
           'checks':r['checks'], 'source_shoe_correction_m_max':source['contact_optimized_shoe_position_correction_m_max']}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if k not in ('checked_sha256','dataset_shapes','checks')}))


if __name__ == '__main__':
    main()
