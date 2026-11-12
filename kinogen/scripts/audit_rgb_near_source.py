"""Verify full-frame provenance and quantify retarget/contact modifications."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--source-targets', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    import numpy as np
    ref = np.load(a.reference, allow_pickle=False)
    timeline = np.load(a.data/'timeline.npz', allow_pickle=False)
    obj = np.load(a.data/'objects/shoe/poses_world.npz', allow_pickle=False)
    source = np.load(a.source_targets, allow_pickle=False)
    if len(ref['time_s']) != 398 or not np.array_equal(ref['frame_ids'], timeline['frame_ids']):
        raise ValueError('Original 398 frame identifiers must be retained')
    time_error = float(np.max(np.abs(ref['time_s']-(timeline['pts_seconds']-timeline['pts_seconds'][0]))))
    rawshoe = np.c_[obj['position_m'], obj['quaternion_wxyz']]
    rawshoe_error = float(np.max(np.abs(ref['shoe_raw_source_pose']-rawshoe)))
    roots = []
    sources = {'timeline':{'path':str(a.data/'timeline.npz'), 'sha256':sha(a.data/'timeline.npz')},
               'shoe':{'path':str(a.data/'objects/shoe/poses_world.npz'), 'sha256':sha(a.data/'objects/shoe/poses_world.npz')}}
    joint_errors = []
    for actor in range(2):
        folder = a.data/f'human/person_{actor+1:02}'
        r = np.load(folder/'root_poses_world.npz', allow_pickle=False)
        roots.append(np.c_[r['position_m'],r['quaternion_wxyz']])
        mesh = np.load(folder/'smplx_meshes_world.npz', allow_pickle=False)
        vertices = mesh['vertices_world']
        reg = source['smplx_joint_regressor']
        independently_derived = np.empty((398,55,3), dtype=float)
        for joint in range(55):
            ids = np.flatnonzero(reg[joint])
            independently_derived[:,joint] = np.sum(vertices[:,ids,:]*reg[joint,ids][None,:,None], axis=1)
        joint_errors.append(float(np.max(np.abs(independently_derived-ref['raw_source_joint_targets'][:,actor]))))
        sources[f'human{actor+1}_mesh'] = {'path':str(folder/'smplx_meshes_world.npz'), 'sha256':sha(folder/'smplx_meshes_world.npz')}
        sources[f'human{actor+1}_roots'] = {'path':str(folder/'root_poses_world.npz'), 'sha256':sha(folder/'root_poses_world.npz')}
    root_error = float(np.max(np.abs(np.stack(roots,axis=1)-ref['raw_source_root_pose_xyz_wxyz'])))
    correction = np.linalg.norm(ref['object_target_pose'][:,:3]-rawshoe[:,:3],axis=1)
    passed = time_error < 1e-9 and rawshoe_error == 0 and root_error == 0 and max(joint_errors) < 2e-6
    result = {'status':'raw_source_provenance_verified' if passed else 'provenance_failed',
        'reference':{'path':str(a.reference), 'sha256':sha(a.reference)},
        'source_target_regressor_sha256':sha(a.source_targets), 'source_frames':398,
        'source_last_pts_s':float(timeline['pts_seconds'][-1]),
        'source_human_valid_frames':np.sum(timeline['human_valid'],axis=0).tolist(),
        'source_shoe_valid_frames':int(np.sum(obj['valid'])),
        'time_max_abs_difference_s':time_error, 'saved_raw_shoe_max_abs_difference':rawshoe_error,
        'saved_raw_roots_max_abs_difference':root_error, 'independent_mesh_regression_max_abs_difference_m':joint_errors,
        'contact_optimized_shoe_position_correction_m_max':float(np.max(correction)),
        'contact_optimized_shoe_position_correction_m_median':float(np.median(correction)),
        'shoe_orientation_observed_frames':int(np.sum(obj['orientation_observed'])),
        'interpretation':'Raw data preservation does not certify reconstruction accuracy. Contact-optimized targets are explicitly modified estimates.',
        'sources':sources}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps(result), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
