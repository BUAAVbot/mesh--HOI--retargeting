"""Run the existing IK implementation as fresh, explicitly labelled benchmarks."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/data/KinoGen/outputs/retarget_compare_20261028'))
    parser.add_argument('--contact', action='store_true')
    parser.add_argument('--max-nfev', type=int, default=45)
    args = parser.parse_args()
    import numpy as np
    method = 'custom_ik_contact' if args.contact else 'custom_ik'
    out = args.root / method
    out.mkdir(parents=True, exist_ok=True)
    source = args.root / 'shared_input/rgb_174114_sim_handoff'
    script = Path('/data/KinoGen/scripts/two_sitting_retarget.py')
    assets = Path('/data/KinoGen/outputs/two_sitting_pipeline/assets_deploy_v3')
    native = out / 'native_reference.npz'
    if native.exists():
        raise FileExistsError(native)
    command = [sys.executable, str(script), '--data', str(source), '--assets', str(assets),
               '--output', str(native), '--source-smoothing-sigma', '0', '--time-scale', '1',
               '--max-nfev', str(args.max_nfev)]
    if args.contact:
        command += ['--seat-xy-adaptation', '--continuous-foot-support']
    start = time.monotonic()
    receipt = {'method': method, 'status': 'running', 'pid': os.getpid(), 'command': command,
               'source_code_sha256': digest(script), 'shared_input_manifest_sha256': digest(args.root / 'shared_input/source_manifest.json'),
               'mode': 'kinematic_only', 'physics_success': None}
    (out / 'run_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    with (out / 'solver.log').open('w') as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True,
                       env={**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
    with np.load(native, allow_pickle=False) as reference:
        qpos = reference['qpos'].copy()
        times = reference['time_s'].copy()
        names = reference['dof_names'].copy()
    if qpos.shape != (172, 2, 59) or not np.isfinite(qpos).all():
        raise ValueError('Expected complete finite reference')
    np.savez_compressed(out / 'reference.npz', qpos=qpos, time_s=times, dof_names=names,
                        frame_ids=np.arange(len(times)), source_hand_valid=np.zeros((172, 2, 2), bool))
    receipt.update(status='completed', elapsed_s=time.monotonic() - start,
                   reference_sha256=digest(out / 'reference.npz'), frames=172, actors=2,
                   source_time_scale=1.0,
                   adaptations=['source-derived robot morphology height/limb scaling', 'upright root roll/pitch',
                                'pelvis-heading smoothing', 'model collision and foot target constraints']
                               + (['inferred continuous support', 'explicit seated XY adaptation'] if args.contact else []),
                   hands='Default unobserved hand/neck channels; this case does not validate hand IK accuracy',
                   limitations=['Kinematic output only; no SONIC or physical success is implied.',
                                'Native task-specific reference adaptations are disclosed and scored against unchanged common observations.'])
    (out / 'run_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
