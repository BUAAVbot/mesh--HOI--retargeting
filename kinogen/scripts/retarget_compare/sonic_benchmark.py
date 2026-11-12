"""Evaluate every canonical retarget reference with frozen SONIC in native physics."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/data/KinoGen/outputs/sonic_expert_20261028'))
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    import numpy as np
    compare = Path('/data/KinoGen/outputs/retarget_compare_20261028')
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / 'references').mkdir(exist_ok=True)
    protocol = json.loads((compare / 'shared_input/source_manifest.json').read_text())
    summary = json.loads((compare / 'comparison_metrics.json').read_text())
    common = dict(np.load(compare / 'shared_input/common_observations.npz', allow_pickle=False))
    labels = dict(np.load(compare / 'custom_ik/native_reference.npz', allow_pickle=False))
    records = []
    for row in summary['results']:
        method = row['method']
        arrays_path = compare / method / 'evaluation_arrays.npz'
        arrays = dict(np.load(arrays_path, allow_pickle=False))
        output = args.root / 'references' / (method + '.npz')
        values = dict(qpos=arrays['qpos'], time_s=arrays['time_s'],
                      source_time_s=common['time_s'], dof_names=np.asarray(protocol['dof_names']),
                      frame_ids=common['frame_ids'], source_hand_valid=common['source_hand_valid'],
                      source_repaired_frame_mask=common['source_repaired_frame_mask'],
                      foot_support_expected=labels['foot_support_expected'],
                      seat_support_expected=labels['seat_support_expected'],
                      contact_inferred=np.asarray(True))
        if not output.exists():
            np.savez_compressed(output, **values)
        else:
            old = np.load(output, allow_pickle=False)
            if any(not np.array_equal(old[key], value) for key, value in values.items()):
                raise RuntimeError('Existing normalized reference differs: ' + method)
        records.append({'method': method, 'label': row['label'], 'reference': str(output),
                        'reference_sha256': sha(output), 'canonical_arrays_sha256': sha(arrays_path)})
    model = Path('/data/KinoGen/outputs/two_sitting_pipeline/assets_deploy_v3/two_vita_solver20.xml')
    proxy = model.parent / 'vita_single_proxy.xml'
    checkpoint = Path('/data/KinoGen/outputs/two_sitting_pipeline/training_native_v1/checkpoint_000450.pt')
    contract = dict(method='Frozen dual new_ft15225 SONIC; zero learned residual',
                    model=str(model), model_sha256=sha(model), proxy_model=str(proxy),
                    proxy_sha256=sha(proxy), references=records, source_time_scale=1.,
                    end_hold_seconds=1., physics_dt_s=.005, policy_dt_s=.02,
                    baseline_policy_weights_changed=False, baseline_uses_previous_expert=False,
                    zero_residual_loader_checkpoint=str(checkpoint),
                    zero_residual_loader_explanation='Only supplies evaluator dimensions/shape. Residual is hard-zero; no previous expert action executes.',
                    contact_labels='Same source-derived inferred seat/foot labels copied from custom_ik/native_reference.npz for all methods; not measured ground truth.',
                    selection_rule='Lexicographic: functional pass, no-fall, completed fraction, final minimum seat-weight support, then lower root XY RMS, then penetration. No physics success inferred from kinematic score.')
    (args.root / 'benchmark_protocol.json').write_text(json.dumps(contract, indent=2) + '\n')
    if args.prepare_only:
        print(json.dumps({'prepared': len(records)})); return
    groups = [records[i::3] for i in range(3)]
    def worker(gpu, group):
        for item in group:
            output = args.root / 'sonic_baseline' / item['method']
            output.parent.mkdir(exist_ok=True)
            if (output / 'report.json').exists():
                continue
            command = ['/root/vita02a-mjlab/.venv310/bin/python',
                       '/data/KinoGen/scripts/two_sitting_eval_ppo.py',
                       '--model', str(model), '--proxy-model', str(proxy),
                       '--reference', item['reference'], '--checkpoint', str(checkpoint),
                       '--output', str(output), '--device', 'cuda:0', '--zero-residual']
            env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = str(gpu)
            began = time.monotonic()
            with (output.parent / (item['method'] + '.log')).open('w') as stream:
                result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
            receipt = dict(method=item['method'], physical_gpu=gpu, command=command,
                           returncode=result.returncode, wall_seconds=time.monotonic()-began)
            (output.parent / (item['method'] + '.receipt.json')).write_text(json.dumps(receipt, indent=2)+'\n')
            print(json.dumps(receipt), flush=True)
            if result.returncode:
                raise RuntimeError('Baseline failed to execute: '+item['method'])
    with ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(lambda item: worker(*item), zip((0,2,6),groups)))
    results = []
    for item in records:
        directory = args.root / 'sonic_baseline' / item['method']
        report = json.loads((directory / 'report.json').read_text())
        trace = np.load(directory / 'trace.npz', allow_pickle=False)
        root_rms = float(np.sqrt(np.mean(trace['root_xy_error_m']**2)))
        nofall = bool(np.min(report['root_height_min_m']) > .45 and
                      np.max(report['root_global_tilt_max_rad']) < .8)
        seat_fraction = float(np.min(np.asarray(report['final_seat_vertical_force_mean_N']) /
                                     np.asarray(report['robot_weight_N'])))
        key = [int(report['functional_task_success']), int(nofall),
               min(1., report['simulated_seconds']/6.7), seat_fraction,
               -root_rms, -report['max_penetration_m']]
        results.append(dict(method=item['method'], selection_key=key, no_fall=nofall,
                            root_xy_rms_m=root_rms, report=report))
    results.sort(key=lambda row: tuple(row['selection_key']), reverse=True)
    selection = dict(status='baseline_complete_reference_selected', selected_method=results[0]['method'],
                     results=results, reference=str(args.root/'references'/(results[0]['method']+'.npz')),
                     source='All10 canonical configurations; no retiming or reference corrections in this baseline.')
    (args.root / 'sonic_comparison.json').write_text(json.dumps(selection, indent=2)+'\n')
    print(json.dumps({'selected': selection['selected_method'],
                      'functional_successes': sum(row['report']['functional_task_success'] for row in results),
                      'tested': len(results)}), flush=True)


if __name__ == '__main__':
    main()
