"""Plot recorded failure evidence with FK only; never integrate or repair states."""
import argparse, hashlib, json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import mujoco as mj
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    trial = args.trial
    report = json.loads((trial / 'report.json').read_text())
    if mj.__version__ != report['mujoco_version']:
        raise ValueError('Use the recorded trial MuJoCo version')
    for name, key in [('model.mjb', 'model'), ('trace.npz', 'trace')]:
        if hashlib.sha256((trial / name).read_bytes()).hexdigest() != report[key]['sha256']:
            raise ValueError('Frozen input hash mismatch: ' + name)
    z = np.load(trial / 'trace.npz', allow_pickle=False)
    m = mj.MjModel.from_binary_path(str(trial / 'model.mjb'))
    d = mj.MjData(m)
    ankle = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, 'left_ankle_roll_link')
    base = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, 'base_link')
    selected = np.arange(0, len(z['time']), 5)
    actual_ankle, actual_com, target_ankle = [], [], []
    for q in z['qpos'][selected]:
        d.qpos[:] = q
        mj.mj_fwdPosition(m, d)
        actual_ankle.append(d.xpos[ankle].copy())
        actual_com.append(d.subtree_com[base].copy())
    for q in z['reference_qpos']:
        d.qpos[:] = q
        mj.mj_fwdPosition(m, d)
        target_ankle.append(d.xpos[ankle].copy())
    actual_ankle, actual_com, target_ankle = map(np.asarray, (actual_ankle, actual_com, target_ankle))
    t, tr, ts = z['source_time_s'][:-1], z['reference_source_time_s'], z['source_time_s'][selected]
    state_t = z['source_time_s'][1:]
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), constrained_layout=True)
    a = axes[0, 0]
    a.plot(tr, target_ankle[:, 1], label='Reference left ankle Y')
    a.plot(ts, actual_ankle[:, 1], label='Recorded left ankle Y')
    a.plot(ts, actual_com[:, 1], label='Recorded robot COM Y', alpha=.8)
    a.set(ylabel='World Y (m)', title='Planned stance anchor and recorded support')
    a = axes[0, 1]
    a.plot(state_t, z['root_horizontal_error_m'], label='Root XY error')
    a.axhline(.35, color='r', linestyle='--', label='Acceptance limit')
    a.set(ylabel='Error (m)', title='Recorded horizontal tracking')
    a = axes[1, 0]
    a.plot(t, z['foot_support_force_N'][:, 0], label='Left foot Fz')
    a.plot(t, z['foot_support_force_N'][:, 1], label='Right foot Fz')
    a.set(ylabel='Measured simulation force (N)', title='Actual foot/terrain supporting force')
    a = axes[1, 1]
    names = list(map(str, z['joint_names']))
    for side in ('left', 'right'):
        k = names.index(side + '_knee_joint')
        aid = int(z['actuator_ids'][k])
        a.plot(t, z['actuator_force'][:, aid], label=side.title() + ' knee')
    a.axhline(120, color='r', linestyle='--', label='Original motor limits')
    a.axhline(-120, color='r', linestyle='--')
    a.set(ylabel='Actual actuator torque (Nm)', title='Original knee effort limits retained')
    end = float(z['source_time_s'][-1])
    for a in axes.flat:
        a.set_xlim(0, end)
        a.set_xlabel('Original source time (s)')
        a.axvspan(3.2, min(3.6, end), color='orange', alpha=.15)
        a.grid(alpha=.25)
        a.legend(fontsize=8)
    fig.suptitle('RGB145 recorded contact-transition diagnostic — TASK NOT ACCEPTED', fontsize=14)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    args.output.with_suffix('.json').write_text(json.dumps({
        'mode': 'recorded_failure_evidence_fk_only', 'trial': str(trial.resolve()),
        'simulation_passed': report['simulation_passed'], 'new_physics_steps': 0,
        'trace_sha256': report['trace']['sha256'], 'model_sha256': report['model']['sha256'],
        'figure_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
        'force_timestamp': 'recorded source clock at the start of each 5ms solver interval',
        'root_error_timestamp': 'post-integration recorded state time',
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
