"""Collect failed and successful trial evidence for the quality-repair loop."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    import numpy as np
    trials = []
    for path in sorted(a.base.glob('physics*/report.json')):
        r = json.loads(path.read_text())
        trial = path.parent
        item = {k:r.get(k) for k in ['status','end_reason','source_frames','source_duration_s',
            'duration_s','max_contact_penetration_m','shoe_reference_position_error_m_max',
            'raw_source_shoe_position_error_m_max','root_horizontal_error_m_max',
            'root_height_error_m_max','root_orientation_error_rad_max','seat_support_fraction',
            'hand_shoe_contact_duration_s','grasp_constraint_force_peak_N','elapsed_wall_s']}
        item.update(trial=str(trial), reference=r['input']['reference']['path'],
                    failed_checks=[k for k,v in r['checks'].items() if not v],
                    checks=r['checks'], quality_thresholds=r.get('quality_thresholds'))
        t = np.load(trial/'trace.npz', allow_pickle=False)
        item['max_recorded_elapsed_s'] = float(np.max(t['time']))
        item['recorded_time_monotonic'] = bool(np.all(np.diff(t['time']) > 0))
        item['accepted_intervals_before_reset'] = int(np.argmax(np.diff(t['time']) <= 0)) if not item['recorded_time_monotonic'] else len(t['time'])-1
        c = np.load(trial/'contacts.npz', allow_pickle=False)
        cutoff = item['accepted_intervals_before_reset']
        mask = c['interval_index'] < cutoff
        if np.any(mask):
            indices = np.flatnonzero(mask)
            i = int(indices[np.argmin(c['signed_distance_m'][mask])])
            item['valid_prefix_worst_contact'] = {
                'time_s':float(c['solver_time_s'][i]), 'penetration_m':max(0.,-float(c['signed_distance_m'][i])),
                'geoms':[str(c['geom_names'][g]) for g in c['geom_ids'][i]],
                'bodies':[str(c['body_names'][b]) for b in c['body_ids'][i]]}
        if isinstance(r.get('controller'), dict):
            item['controller'] = {k:r['controller'].get(k) for k in ['sonic_target_weight','arm_reference_override_weight','gravity_compensation_fraction']}
            item['inference_counts'] = [v.get('inference_count') for v in r['controller'].get('actors', [])]
        trials.append(item)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    out = {'schema':'rgb_near_quality_repair_history.v1', 'trials':trials,
           'interpretation':'Solver auto-reset states are failed evidence; valid-prefix metrics are explicitly distinguished.'}
    a.output.write_text(json.dumps(out, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps({'trials':len(trials), 'passed':[v['trial'] for v in trials if v['status']=='passed_assumed_physics_sequence']}))


if __name__ == '__main__':
    main()
