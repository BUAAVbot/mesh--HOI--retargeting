"""Add corrected quaternion metric evidence without overwriting original trials."""
from pathlib import Path
import argparse
import hashlib
import json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    import numpy as np
    from two_sitting_physics import interp

    root = args.root
    destination = root / 'metrics_corrections/quaternion_shortest_arc'
    destination.mkdir(parents=True, exist_ok=True)
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    # The exact antipodal midpoint was undefined in the old interpolation.
    qt = np.zeros((2, 2, 59)); qt[:, :, 3] = 1; qt[1, :, 3] = -1
    reference = {'qpos': qt, 'time_s': np.array([0., 1.])}
    unit_tests = {}
    for t in (0., .25, .5, .75, 1., 2.):
        actual = interp(reference, t)[:, 3:7]
        unit_tests[str(t)] = bool(np.isfinite(actual).all() and np.allclose(np.abs(actual[:, 0]), 1.))
    if not all(unit_tests.values()):
        raise RuntimeError('Antipodal quaternion interpolation regression')
    records, skipped = [], []
    for trace_path in sorted(root.rglob('trace.npz')):
        report_path = trace_path.with_name('report.json')
        if not report_path.exists() or destination in trace_path.parents:
            continue
        report = json.loads(report_path.read_text())
        path = report.get('inputs', {}).get('reference')
        if path is None:
            skipped.append({'trial': str(trace_path.parent.relative_to(root)), 'reason': 'no inputs.reference'})
            continue
        reference_path = Path(path)
        trace = np.load(trace_path, allow_pickle=False)
        if 'robot_root_qadr' not in trace or 'root_orientation_error_rad' not in trace:
            skipped.append({'trial': str(trace_path.parent.relative_to(root)), 'reason': 'no root maps/old angle array'})
            continue
        raw = dict(np.load(reference_path, allow_pickle=False))
        desired = np.asarray([interp(raw, t) for t in trace['time']])
        actual = np.asarray([trace['qpos'][:, rq+3:rq+7] for rq in trace['robot_root_qadr']]).transpose(1, 0, 2)
        corrected = 2 * np.arccos(np.clip(np.abs(np.sum(actual * desired[:, :, 3:7], -1)), 0, 1))
        old = trace['root_orientation_error_rad']
        checks = dict(report.get('checks', {}))
        if 'root_orientation_error_below_0p65rad' in checks:
            checks['root_orientation_error_below_0p65rad'] = bool(corrected.max() < .65)
        row = {
            'trial': str(trace_path.parent.relative_to(root)),
            'original_report': str(report_path), 'original_report_sha256': sha(report_path),
            'original_trace_sha256': sha(trace_path), 'reference': str(reference_path),
            'reference_sha256': sha(reference_path),
            'old_max_root_orientation_error_rad': old.max(0).tolist(),
            'corrected_max_root_orientation_error_rad': corrected.max(0).tolist(),
            'max_metric_difference_rad': float(np.max(abs(old-corrected))),
            'corrected_checks': checks,
            'original_strict_success': report.get('success'),
            'corrected_strict_success': bool(all(checks.values())) if checks else None,
            'original_functional_task_success': report.get('functional_task_success'),
            'functional_metric_change': False,
            'physics_rerun': False, 'original_evidence_unchanged': True,
            'limitation': 'Only metric interpolation corrected. Original actual states, controls, contacts, source clock and functional task acceptance are preserved.'}
        out = destination / trace_path.parent.relative_to(root)
        out.mkdir(parents=True, exist_ok=True)
        (out / 'report.json').write_text(json.dumps(row, indent=2) + '\n')
        np.savez_compressed(out / 'corrected_orientation.npz', time_s=trace['time'],
                            root_orientation_error_rad=corrected)
        row['overlay_report'] = str(out / 'report.json')
        records.append(row)
    summary = {'status': 'metric_correction_overlay_complete_no_physics_rerun',
               'cause': 'Some evaluation callers passed raw sign-flipping quaternions to normalized linear interpolation; DualReference already canonicalizes adjacent signs.',
               'method': 'Choose shortest arc at each interpolated segment; preserve arithmetic for already continuous segments.',
               'antipodal_midpoint_regression_tests': unit_tests,
               'source_sha256': sha(Path(__file__)),
               'interpolation_source_sha256': sha(Path('/data/KinoGen/scripts/two_sitting_physics.py')),
               'records': records, 'skipped': skipped,
               'strict_success_changes': [r['trial'] for r in records if r['original_strict_success'] != r['corrected_strict_success']],
               'functional_success_changes': 0}
    (destination / 'correction_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({'corrected_trials': len(records), 'strict_success_changes': summary['strict_success_changes'],
                      'phc': [r for r in records if r['trial'] == 'sonic_baseline/phc']}))


if __name__ == '__main__':
    main()
