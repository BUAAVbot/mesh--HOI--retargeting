"""Expose actual adjacent-frame joint rates without central-difference cancellation."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path


def main():
    import numpy as np
    root = Path('/data/KinoGen/outputs/retarget_compare_20261028')
    summary_path = root / 'comparison_metrics.json'
    summary = json.loads(summary_path.read_text())
    protocol = json.loads((root / 'shared_input/source_manifest.json').read_text())
    names = protocol['dof_names']
    body = [i for i, name in enumerate(names) if not name.startswith(('lh_', 'rh_', 'neck_'))]
    checks = []
    for row in summary['results']:
        directory = root / row['method']
        with np.load(directory / 'evaluation_arrays.npz', allow_pickle=False) as archive:
            positions = archive['qpos'][:, :, 7:][:, :, body]
            times = archive['time_s']
        rates = np.diff(positions, axis=0) / np.diff(times)[:, None, None]
        peak_index = np.unravel_index(np.abs(rates).argmax(), rates.shape)
        peak = float(np.abs(rates[peak_index]))
        row['metrics']['peak_body_joint_interval_speed_rad_s'] = peak
        record = {'method': row['method'], 'peak_interval_speed_rad_s': peak,
                  'interval_start_frame': int(peak_index[0]), 'person': int(peak_index[1]) + 1,
                  'joint': names[body[int(peak_index[2])]],
                  'dt_s': float(times[peak_index[0] + 1] - times[peak_index[0]]),
                  'central_difference_max_rad_s': row['metrics']['max_body_joint_speed_rad_s'],
                  'hardware_velocity_limit_claim': False}
        np.savez_compressed(directory / 'interval_rate_audit.npz',
                            interval_time_s=times[:-1], body_joint_interval_velocity_rad_s=rates,
                            body_joint_names=np.asarray([names[i] for i in body]))
        (directory / 'interval_rate_audit.json').write_text(json.dumps(record, indent=2) + '\n')
        evaluation = directory / 'evaluation.json'
        detailed = json.loads(evaluation.read_text())
        detailed['metrics']['peak_body_joint_interval_speed_rad_s'] = peak
        evaluation.write_text(json.dumps(detailed, indent=2) + '\n')
        checks.append(record)
    summary['metric_notes']['velocity'] = 'max_body_joint_speed uses central differences; peak_body_joint_interval_speed uses actual adjacent-frame angle difference divided by source dt. Neither is a measured hardware limit.'
    summary['interval_rate_audit_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    summary_path.write_text(json.dumps(summary, indent=2) + '\n')
    (root / 'interval_rate_audit.json').write_text(json.dumps({'method': 'adjacent-frame finite difference, no smoothing or unwrap of bounded hinge joints', 'results': checks}, indent=2) + '\n')
    print(json.dumps(checks))


if __name__ == '__main__':
    main()
