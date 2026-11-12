"""Create one traceable, repaired two-person input for all retargeting methods."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('/data/KinoGen/data/two_sitting_people/rgb_174114_sim_handoff'))
    parser.add_argument('--output', type=Path, default=Path('/data/KinoGen/outputs/retarget_compare_20261028/shared_input'))
    parser.add_argument('--assets', type=Path, default=Path('/data/KinoGen/outputs/two_sitting_pipeline/assets_deploy_v3'))
    args = parser.parse_args()
    import numpy as np
    import mujoco
    from scipy.spatial.transform import Rotation, Slerp

    start = time.monotonic()
    output = args.output / args.source.name
    if (args.output / 'source_manifest.json').exists():
        raise FileExistsError('Shared input is already frozen; use a new output directory')
    output.mkdir(parents=True, exist_ok=True)
    source_hashes = {}
    output_hashes = {}
    for relative in ['timeline.npz', 'human/person_01/smplx_params_world.npz',
                     'human/person_02/smplx_params_world.npz',
                     'human/person_01/smplx_meshes_world.npz',
                     'human/person_02/smplx_meshes_world.npz']:
        source_hashes[relative] = sha(args.source / relative)
    with np.load(args.source / 'timeline.npz', allow_pickle=False) as loaded:
        timeline = {key: loaded[key].copy() for key in loaded.files}
    times = np.asarray(timeline['pts_seconds'], float)
    times = times - times[0]
    count = len(times)
    if count != 172 or not np.all(np.diff(times) > 0):
        raise ValueError('Expected the complete172-frame synchronous example')
    repair_mask = np.zeros((count, 2), bool)
    repair_mask[63:70, 1] = True
    original_valid = timeline['human_valid'].copy()
    timeline['human_valid'] = np.ones_like(original_valid, bool)
    timeline['source_original_human_valid'] = original_valid
    timeline['source_repaired_frame_mask'] = repair_mask
    np.savez_compressed(output / 'timeline.npz', **timeline)
    joints_raw, joints_repaired = [], []
    poses = []
    for person in (1, 2):
        relative = f'human/person_{person:02d}'
        destination = output / relative
        destination.mkdir(parents=True)
        with np.load(args.source / relative / 'smplx_params_world.npz', allow_pickle=False) as loaded:
            params = {key: loaded[key].copy() for key in loaded.files}
        with np.load(args.source / relative / 'smplx_meshes_world.npz', allow_pickle=False) as loaded:
            meshes = {key: loaded[key].copy() for key in loaded.files}
        raw = meshes['joints_world'].copy()
        if person == 2:
            weights = ((times[63:70] - times[62]) / (times[70] - times[62]))
            for key in ('global_orient', 'body_pose'):
                original = params[key]
                shaped = original.reshape(count, -1, 3)
                for joint in range(shaped.shape[1]):
                    interpolation = Slerp(times[[62, 70]], Rotation.from_rotvec(shaped[[62, 70], joint]))
                    shaped[63:70, joint] = interpolation(times[63:70]).as_rotvec()
                params[key] = shaped.reshape(original.shape).astype(original.dtype)
            for key in ('transl', 'decoder_transl'):
                if key in params:
                    original = params[key]
                    params[key][63:70] = ((1 - weights[:, None]) * original[62]
                                           + weights[:, None] * original[70])
            for key in ('joints_world', 'vertices_world'):
                original = meshes[key]
                meshes[key][63:70] = ((1 - weights[:, None, None]) * original[62]
                                      + weights[:, None, None] * original[70])
        params['source_original_valid'] = params['valid'].copy()
        params['source_repaired_frame_mask'] = repair_mask[:, person - 1]
        params['valid'] = np.ones(count, bool)
        meshes['source_original_valid'] = meshes['valid'].copy()
        meshes['source_repaired_frame_mask'] = repair_mask[:, person - 1]
        meshes['valid'] = np.ones(count, bool)
        for data in (params, meshes):
            for key, values in data.items():
                if values.dtype.kind in 'fiu' and not np.isfinite(values).all():
                    raise ValueError(f'Nonfinite source array: {relative}/{key}')
        for name, data in [('smplx_params_world', params), ('smplx_meshes_world', meshes)]:
            np.savez_compressed(destination / f'{name}.npz', **data)
        joints_raw.append(raw)
        joints_repaired.append(meshes['joints_world'])
        poses.append({'person': person,
                      'parameter_convention': str(params.get('parameter_convention', '')),
                      'decode_convention': str(params.get('decode_convention', '')),
                      'coordinate_system': str(params.get('coordinate_system', '')),
                      'hand_valid_count': int(params['hand_valid'].sum())})
    # Link immutable, read-only inputs; adapters must never write these files.
    for relative in ['scene/pi3_mesh.npz', 'cameras_world.npz', 'coordinates.npz', 'input.mp4']:
        source = args.source / relative
        if not source.exists():
            continue
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
        source_hashes[relative] = sha(source)
    support = json.loads((args.assets / 'support_geometry.json').read_text())
    U = np.asarray(support['source_to_sim_U'], float)
    joints_raw = np.stack(joints_raw, axis=1) @ U.T
    joints_repaired = np.stack(joints_repaired, axis=1) @ U.T
    np.savez_compressed(args.output / 'common_observations.npz',
                        time_s=times, frame_ids=timeline['frame_ids'],
                        source_repaired_frame_mask=repair_mask,
                        raw_source_joints_zup=joints_raw,
                        repaired_source_joints_zup=joints_repaired,
                        source_to_sim_U=U,
                        source_hand_valid=np.zeros((count, 2, 2), bool))
    model_path = args.assets / 'vita_single_proxy.xml'
    model = mujoco.MjModel.from_xml_path(str(model_path))
    hinge = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
    hinge = hinge[np.argsort(model.jnt_qposadr[hinge])]
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(joint)) for joint in hinge]
    if model.nq != 59 or len(names) != 52:
        raise ValueError('All methods must target the commonVita52 model')
    for path in output.rglob('*.npz'):
        output_hashes[path.relative_to(output).as_posix()] = sha(path)
    protocol = {
        'schema': 'retarget_comparison.shared_input.v1', 'source': str(args.source),
        'prepared_input': str(output), 'frames': count, 'actors': 2,
        'duration_s': float(times[-1]), 'nominal_fps': 30,
        'frame_selection': 'all172frames; no cropped success selection',
        'source_to_sim_U': U.tolist(), 'source_global_scale': 1.0,
        'repair': {'person': 2, 'zero_based_frames_inclusive': [63, 69], 'anchor_frames': [62, 70],
                   'rotations': 'SO3Slerp', 'translations_and_world_mesh_joints': 'linear interpolation',
                   'smoothing': 'none in shared preprocessing',
                   'interpretation': 'algorithm-valid interpolated frames; not recovered observations'},
        'evaluation_subsets': ['all172', 'exclude_repaired165_actor2_frames'],
        'hand_observations_valid': False,
        'hand_policy': 'Report defaults/neutral and model compatibility, not observed hand accuracy',
        'model': str(model_path), 'model_sha256': sha(model_path),
        'dof_names': names, 'nq': model.nq, 'nv': model.nv,
        'support_geometry': str(args.assets / 'support_geometry.json'),
        'support_geometry_sha256': sha(args.assets / 'support_geometry.json'),
        'source_hashes': source_hashes, 'prepared_npz_hashes': output_hashes,
        'source_pose_conventions': poses,
        'method_output_contract': {'qpos': '[T,2,59] preferred; [2,T,59] accepted with explicitaxes',
                                   'time_s': '[T] original0..5.7s', 'dof_names': '52actualjointnames',
                                   'status': 'kinematic_retarg_output_not_physical_task_success'},
        'comparability': 'Native method-specific morphology/constraint adaptations must be recorded; no per-method furniture enlargement or whole-world rescaling.',
        'elapsed_s': time.monotonic() - start,
    }
    (args.output / 'source_manifest.json').write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({key: protocol[key] for key in ('prepared_input', 'frames', 'duration_s', 'elapsed_s', 'source_pose_conventions')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
