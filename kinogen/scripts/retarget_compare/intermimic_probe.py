"""Bounded probes of the official InterMimic entry points on existing SDKs.

No substitute retargeter is implemented. A source contract probe is not a
successful InterMimic rollout, and data replay is not policy tracking.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_probe(command, cwd, log, env, timeout):
    started = time.monotonic()
    timed_out = False
    with log.open('w') as stream:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                code = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                code = process.wait()
    return {'command': command, 'pid': process.pid, 'returncode': code,
            'timeout': timed_out, 'wall_seconds': time.monotonic() - started,
            'log': str(log), 'log_sha256': sha(log),
            'last_lines': log.read_text(errors='replace').splitlines()[-20:]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path,
                        default=Path('/data/KinoGen/outputs/retarget_compare_20261028/intermimic'))
    parser.add_argument('--sdk-python', type=Path,
                        default=Path('/data/grail-envs/tracking45/bin/python'))
    parser.add_argument('--gpu', default='2')
    args = parser.parse_args()
    src = args.base / 'source_probe'
    logs = args.base / 'logs'
    logs.mkdir(exist_ok=True)
    readme = (src / 'README.md').read_text()
    tree = json.loads((args.base / 'official_tree.json').read_text())
    todo = [line for line in readme.splitlines() if 'retargeting as tracking' in line.lower()]
    files = sorted(src.rglob('*.py'))
    syntax = []
    for path in files:
        ast.parse(path.read_text())
        syntax.append({'path': str(path), 'sha256': sha(path)})
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, OMP_NUM_THREADS='2',
               MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONUNBUFFERED='1',
               ACCEPT_EULA='Y', OMNI_KIT_ACCEPT_EULA='YES')
    versions_code = """import importlib.metadata as m,importlib.util,json,sys
r={'python':sys.version,'modules':{}}
for p in ['torch','isaacsim','isaaclab','rl-games','gymnasium']:
 try:r['modules'][p]=m.version(p)
 except:r['modules'][p]=None
for p in ['isaaclab','isaacsim','isaacgym']:
 try:s=importlib.util.find_spec(p);r[p+'_path']=None if s is None else s.origin
 except Exception as e:r[p+'_path']=str(e)
print(json.dumps(r))"""
    probes = [run_probe([str(args.sdk_python), '-c', versions_code], src,
                        logs / 'sdk_versions.log', env, 30)]
    probes.append(run_probe([str(args.sdk_python), '-u', str(src / 'isaaclab/examples/data_replay.py'), '--help'],
                  src, logs / 'data_replay_help.log', env, 30))
    probes.append(run_probe([str(args.sdk_python), '-u', str(src / 'isaaclab/examples/test_policy_inference.py'), '--help'],
                  src, logs / 'policy_inference_help.log', env, 30))
    # Actual official entry, with the actual shared dataset path. No synthetic
    # policy/checkpoint and no fabricated robot conversion are supplied.
    shared = args.base.parent / 'shared_input/rgb_174114_sim_handoff'
    probes.append(run_probe([str(args.sdk_python), '-u', str(src / 'isaaclab/examples/data_replay.py'),
                  '--headless', '--num-envs', '1', '--motion-dir', str(shared)],
                  src, logs / 'data_replay_shared_case.log', env, 90))
    report = {
        'schema': 'retarget_comparison.intermimic_entry_probe.v1',
        'method': 'InterMimic official entry probes; no substitute IK implementation',
        'official_repository': 'https://github.com/Sirui-Xu/InterMimic',
        'official_retarg_tracking_todo_line': todo,
        'source_files_parsed': syntax,
        'gpu_visible_devices': args.gpu,
        'probes': probes,
        'official_isaaclab_model': {'body': 'SMPL-X', 'action_dim': 153,
                                    'configured_model': 'assets/smplx/omomo_isaaclab.xml'},
        'shared_comparison_model': {'body': 'Vita52', 'action_channels': 52, 'qpos_dim': 59},
        'vita_paths_in_official_tree': [item['path'] for item in tree if 'vita' in item['path'].lower()],
        'input_contract': {
            'native_loader': 'torch.load of preprocessed HOI motion tensors',
            'root_position_columns': [0, 3], 'root_quaternion_xyzw_columns': [3, 7],
            'dof_position_columns': [9, 162], 'body_positions52_columns': [162, 318],
            'object_position_quaternion_columns': [318, 325],
            'object_contact_column': [330, 331], 'human_contacts52_columns': [331, 383],
            'body_quaternions52_columns': [383, 591],
            'shared_input': 'two SMPL-X parameter/mesh NPZ actors plus two fixed seats; no measured contact labels',
        },
        'required_before_fair_vita52_comparison': [
            'A Vita52 asset, actuator and body-name/action/observation mapping for the official task.',
            'A Vita52-compatible trained teacher or a disclosed new training run; SMPL-X/G1 weights do not match Vita52.',
            'Conversion of both source actors and fixed furniture to the native motion/contact schema, preserving unobserved contact provenance.',
            'A compatible IsaacLab environment and all required official asset files.',
        ],
        'status': 'entry_probed_no_same_model_retarget_result',
        'same_input_same_vita52_output_available': False,
        'physics_rollout_pass_claim': False,
        'not_run_as_kinematic_baseline': 'Public entry is policy-based physics/replay; retarget-as-tracking remains listed unchecked in official README.',
    }
    path = args.base / 'intermimic_probe_report.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'report': str(path),
                      'probe_returncodes': [p['returncode'] for p in probes]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
