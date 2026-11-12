"""Read-only audit of source-relative training gates on a valid physical plan.

No training, mj_step, or live plant writes. This reproduces the declared gate
formulas on recorded states; it does not alter independent task acceptance.
"""
from pathlib import Path
import argparse
import hashlib
import json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    import numpy as np
    import torch
    import mujoco as mj
    from two_sitting_ppo_env import DualReference

    root = args.root
    trace_path = root / 'terminal_refined_full_v4/trace.npz'
    reference_path = root / 'references/phc_transfer_time1p5.npz'
    model_path = root / 'calibration/two_vita_expert.mjb'
    source_path = Path('/data/KinoGen/scripts/two_sitting_native_ppo_env.py')
    trace = np.load(trace_path, allow_pickle=False)
    model = mj.MjModel.from_binary_path(str(model_path))
    ref = DualReference(model, reference_path, 'cpu')
    # Same 20ms policy sampling; phase is accumulated in float32 as in training.
    indices = np.arange(4, len(trace['time']), 4)
    phase = torch.zeros(1)
    phases = []
    for _ in indices:
        phase += .02
        phases.append(float(phase[0]))
    target, _, _ = ref.sample(torch.tensor(phases))
    q = torch.tensor(trace['qpos'][indices], dtype=torch.float32)
    xy, height, angle = [], [], []
    for rq in ref.root_qadr:
        delta = q[:, rq:rq+3] - target[:, rq:rq+3]
        xy.append(delta[:, :2].norm(dim=-1))
        height.append(delta[:, 2].abs())
        angle.append(2 * torch.acos((q[:, rq+3:rq+7] * target[:, rq+3:rq+7]).sum(-1).abs().clamp(0, 1)))
    xy, height, angle = [torch.stack(a, 1).numpy() for a in (xy, height, angle)]
    masks = {'source_root_xy_above_0p65m': xy.max(-1) > .65,
             'source_root_z_above_0p30m': height.max(-1) > .30,
             'source_root_angle_above_1p2rad': angle.max(-1) > 1.2}
    failures = np.logical_or.reduce(list(masks.values()))
    first = int(np.flatnonzero(failures)[0]) if failures.any() else None
    gate = lambda mask: {'samples': int(mask.sum()),
                         'first_time_s': None if not mask.any() else float(trace['time'][indices[np.flatnonzero(mask)[0]]])}
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    report = {
        'status': 'recorded_state_training_gate_audit_complete_no_dynamics_or_training',
        'trace': str(trace_path), 'trace_sha256': sha(trace_path),
        'reference': str(reference_path), 'reference_sha256': sha(reference_path),
        'model_sha256': sha(model_path), 'audited_source_sha256': sha(source_path),
        'audit_source_sha256': sha(Path(__file__)),
        'sampling': 'Recorded actual states at each20ms boundary, float32 accumulated policy clock and exact DualReference.sample implementation',
        'source_relative_failure_gate': {k: gate(v) for k, v in masks.items()},
        'first_training_failure': None if first is None else {
            'actual_time_s': float(trace['time'][indices[first]]),
            'policy_phase_s': phases[first],
            'root_xy_m': xy[first].tolist(), 'root_z_m': height[first].tolist(),
            'root_angle_rad': angle[first].tolist(),
            'reasons': [k for k, v in masks.items() if v[first]]},
        'maximum_source_errors': {'root_xy_m': xy.max(0).tolist(),
                                  'root_z_m': height.max(0).tolist(),
                                  'root_angle_rad': angle.max(0).tolist()},
        'valid_physical_plan_would_be_cut_by_source_relative_training_done': bool(failures.any()),
        'limits': 'Read-only reproduction of declared numerical gates, not a new physical execution. Does not establish this is the only training failure cause. Does not relax PHC strict tracking acceptance.',
        'optional_separate_experiment': 'A physical-plan stabilization objective is a separate engineering comparison, not a demonstrated termination-bug fix. It must keep original source/PHC metrics in independent reporting.'}
    out = root / 'audit/physical_prior_training_objective'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez_compressed(out / 'gate_components.npz', time_s=trace['time'][indices],
                        source_root_xy_m=xy, source_root_z_m=height, source_root_angle_rad=angle,
                        training_failed=failures, **masks)
    print(json.dumps(report['first_training_failure']))


if __name__ == '__main__':
    main()
