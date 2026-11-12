"""Audit shared SMPL-X parameters against supplied world meshes/joints on GPU0.

Interpolation-repaired rows are reported separately: SLERP parameter repair and
linear mesh repair are intentionally not an identical nonlinear decoder path.
"""
from __future__ import annotations

import argparse
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
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def metrics(errors, mask, np):
    values = errors[mask]
    if not len(values):
        return {'frames': 0, 'points': 0}
    return {'frames': int(mask.sum()), 'points': int(values.size),
            'median_m': float(np.median(values)), 'mean_m': float(np.mean(values)),
            'p95_m': float(np.quantile(values, .95)), 'p99_m': float(np.quantile(values, .99)),
            'max_m': float(np.max(values)),
            'frame_median_max_m': float(np.max(np.median(values, axis=1)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path('/data/KinoGen/outputs/retarget_compare_20261028')
    parser.add_argument('--shared', type=Path, default=base / 'shared_input')
    parser.add_argument('--output', type=Path, default=base / 'audit/shared_smplx_gpu0')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--tolerance-m', type=float, default=1e-4)
    args = parser.parse_args()
    begin = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'report.json').exists():
        raise FileExistsError('Use a fresh audit output rather than overwrite evidence')
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0', 'This audit explicitly uses physical GPU0'
    gpu_before = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu',
                                         '--format=csv,noheader,nounits'], text=True)
    row = next(line for line in gpu_before.splitlines() if line.startswith('0,'))
    values = [int(value.strip()) for value in row.split(',')]
    assert values[2] - values[1] >= 4096, ('GPU0 free memory too small', row)
    import numpy as np
    import torch
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.08, device=0)
    torch.cuda.reset_peak_memory_stats(0)
    decoder = base / 'umr/source/scripts/smplx_model_loader.py'
    sys.path.insert(0, str(decoder.parent))
    from smplx_model_loader import build_smplx_model
    model_path = Path('/data/grail/imports/GEM-SMPL/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz')
    model = build_smplx_model(model_path, 'neutral', 1).to('cuda:0').eval()
    manifest = json.loads((args.shared / 'source_manifest.json').read_text())
    data = args.shared / 'rgb_174114_sim_handoff'
    actors = []
    all_consistent = True
    actual_decoded_frames = 0
    for actor in [1, 2]:
        params_path = data / f'human/person_{actor:02d}/smplx_params_world.npz'
        mesh_path = data / f'human/person_{actor:02d}/smplx_meshes_world.npz'
        params = np.load(params_path, allow_pickle=False)
        supplied = np.load(mesh_path, allow_pickle=False)
        count = len(params['frame_ids'])
        assert count == 172
        repaired = np.asarray(params['source_repaired_frame_mask'], bool)
        assert np.isfinite(params['body_pose']).all() and np.isfinite(params['global_orient']).all()
        difference = np.linalg.norm(params['transl'] - params['decoder_transl'], axis=1)
        identical = np.array_equal(params['transl'], params['decoder_transl'])
        keys = ['transl'] if identical else ['transl', 'decoder_transl']
        actor_report = {'actor': actor, 'params_path': str(params_path), 'params_sha256': sha(params_path),
                        'mesh_path': str(mesh_path), 'mesh_sha256': sha(mesh_path),
                        'frames': count, 'repaired_frames_zero_based': np.where(repaired)[0].tolist(),
                        'transl_decoder_transl_bitwise_equal': bool(identical),
                        'translation_field_difference_max_m': float(difference.max()),
                        'translation_field_difference_median_m': float(np.median(difference)),
                        'faces_exact': bool(np.array_equal(model.faces, supplied['faces'])), 'variants': {}}
        for translation_key in keys:
            vertices, joints = [], []
            for start in range(0, count, args.batch_size):
                end = min(start + args.batch_size, count)
                n = end - start
                values = {key: torch.as_tensor(params[key][start:end], device='cuda:0', dtype=torch.float32)
                          for key in ['global_orient', 'body_pose', 'betas', 'left_hand_pose', 'right_hand_pose']}
                values['transl'] = torch.as_tensor(params[translation_key][start:end], device='cuda:0', dtype=torch.float32)
                values.update(jaw_pose=torch.zeros((n, 3), device='cuda:0'),
                              leye_pose=torch.zeros((n, 3), device='cuda:0'),
                              reye_pose=torch.zeros((n, 3), device='cuda:0'),
                              expression=torch.zeros((n, 10), device='cuda:0'))
                with torch.inference_mode():
                    result = model(**values, return_verts=True)
                vertices.append(result.vertices.cpu().numpy())
                joints.append(result.joints.cpu().numpy())
                actual_decoded_frames += n
            vertices = np.concatenate(vertices)
            joints = np.concatenate(joints)
            assert vertices.shape == supplied['vertices_world'].shape
            assert joints.shape == supplied['joints_world'].shape, (joints.shape, supplied['joints_world'].shape)
            vertex_error = np.linalg.norm(vertices - supplied['vertices_world'], axis=-1)
            joint_error = np.linalg.norm(joints - supplied['joints_world'], axis=-1)
            per_frame = {'vertex_median_m': np.median(vertex_error, axis=1).tolist(),
                         'vertex_max_m': vertex_error.max(axis=1).tolist(),
                         'joint_median_m': np.median(joint_error, axis=1).tolist(),
                         'joint_max_m': joint_error.max(axis=1).tolist()}
            variant = {'decoded_vertices_shape': list(vertices.shape), 'decoded_joints_shape': list(joints.shape),
                       'vertices': {name: metrics(vertex_error, mask, np) for name, mask in
                                    [('all', np.ones(count, bool)), ('unrepaired', ~repaired), ('repaired', repaired)]},
                       'joints': {name: metrics(joint_error, mask, np) for name, mask in
                                  [('all', np.ones(count, bool)), ('unrepaired', ~repaired), ('repaired', repaired)]},
                       'per_frame': per_frame}
            passed = bool(vertex_error[~repaired].max() <= args.tolerance_m and
                          joint_error[~repaired].max() <= args.tolerance_m and actor_report['faces_exact'])
            variant['unrepaired_consistency_passed'] = passed
            actor_report['variants'][translation_key] = variant
            all_consistent &= passed
        if identical:
            actor_report['decoder_transl_validation'] = 'Identical input array; the transl GPU decode exactly tests this variant too, without duplicate arithmetic.'
        actors.append(actor_report)
        print(json.dumps({'actor': actor, 'variants': keys, 'repaired_frames': int(repaired.sum())}), flush=True)
    torch.cuda.synchronize()
    report = {'schema': 'retarget_comparison.shared_smplx_gpu_audit.v1',
              'status': 'unrepaired_parameter_mesh_consistency_verified' if all_consistent else 'unrepaired_consistency_failed',
              'unrepaired_consistency_passed': bool(all_consistent), 'tolerance_m': args.tolerance_m,
              'source_frames_per_actor': 172, 'actors_count': 2, 'actual_gpu_decoded_frames': actual_decoded_frames,
              'physical_gpu_index': 0, 'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
              'torch_device': torch.cuda.get_device_name(0), 'torch_version': torch.__version__,
              'python_version': sys.version, 'numpy_version': np.__version__,
              'batch_size': args.batch_size, 'per_process_gpu_memory_fraction_cap': .08,
              'torch_peak_allocated_bytes': torch.cuda.max_memory_allocated(0),
              'torch_peak_reserved_bytes': torch.cuda.max_memory_reserved(0),
              'gpu_before_csv': gpu_before,
              'model_path': str(model_path), 'model_sha256': sha(model_path),
              'decoder_code_path': str(decoder), 'decoder_code_sha256': sha(decoder),
              'shared_manifest_sha256': sha(args.shared / 'source_manifest.json'),
              'coordinate_policy': 'Source Y-up absolute world space; no extra root transform or axis conversion applied.',
              'decode_policy': 'SMPLX use_pca=False flat_hand_mean=True, actual betas/body/global/absolute hands/transl; default zero jaw/eyes/expression.',
              'repaired_row_interpretation': 'Actor2 frames63..69: source parameter SO3 interpolation and separately linear world meshes/joints are nonlinear-inconsistent by construction; measured separately, never graded as a decode defect.',
              'hand_accuracy_claim': False,
              'scope': 'Parameter-to-supplied-mesh/joint decoder consistency only; not geometric reconstruction truth, contact ground truth, robot tracking or physical success.',
              'source_mutated': False, 'wall_seconds': time.monotonic() - begin, 'actors': actors}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for actor in actors:
        for translation, variant in actor['variants'].items():
            for axis, kind in zip(axes, ['vertex_median_m', 'joint_median_m']):
                axis.plot(np.arange(172) / 30, np.maximum(np.asarray(variant['per_frame'][kind]) * 1000, 1e-6),
                          label=f"Actor{actor['actor']} / {translation}")
    for axis, title in zip(axes, ['Vertex median error', 'Joint median error']):
        axis.axvspan(63 / 30, 69 / 30, color='#ffb74d', alpha=.25, label='Actor2 interpolated rows63-69')
        axis.set_yscale('log')
        axis.set_ylabel(title + ' [mm]')
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    axes[-1].set_xlabel('Original source time [seconds]')
    fig.suptitle('Shared SMPL-X parameter-to-world-mesh consistency | all2x172 frames decoded on GPU0\nInterpolated parameter/mesh rows are reported separately from untouched observations', fontsize=12)
    fig.tight_layout()
    fig.savefig(args.output / 'parameter_mesh_consistency.png', dpi=160)
    plt.close(fig)
    print(json.dumps({'status': report['status'], 'report': str(args.output / 'report.json'),
                      'gpu_decoded_frames': actual_decoded_frames, 'wall_seconds': report['wall_seconds']}), flush=True)


if __name__ == '__main__':
    main()
