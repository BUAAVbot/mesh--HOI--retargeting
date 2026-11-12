"""Plot trace quality and contact forces from a frozen MuJoCo trial."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trial-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    r = json.loads((a.trial_dir/'report.json').read_text())
    c = json.loads((a.trial_dir/'physical_contacts.json').read_text())
    z = np.load(a.trial_dir/'physical_trace.npz', allow_pickle=False)
    ref = np.load(r['input']['reference']['path'], allow_pickle=False)
    t = z['time']-z['time'][0]
    tq = t[1:]
    rt = ref['time_s']-ref['time_s'][0]
    oq = int(z['object_qadr'])
    obj = z['qpos'][:, oq:oq+3]
    target = ref['object_target_pose'][:, :3]
    interp = np.column_stack([np.interp(t, rt, target[:, j]) for j in range(3)])
    force = np.asarray([v['object_hand_normal_force_N'] for v in c])
    penetration = np.asarray([v['max_penetration_m'] for v in c])
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    ax = axes[0, 0]
    for j, color in enumerate(['#c83349', '#1879be', '#2b8c4b']):
        ax.plot(t, (obj[:, j]-obj[0, j])*100, color=color, label='actual '+'XYZ'[j])
        ax.plot(t, (interp[:, j]-obj[0, j])*100, '--', color=color, alpha=.6)
    ax.set(title='Object displacement (solid actual, dashed target)', ylabel='cm', xlabel='time (s)')
    ax.legend(ncol=3, fontsize=8)
    ax = axes[0, 1]
    ax.plot(tq, force, color='#1879be', label='hand/object normal force')
    ax.set(title='Simulated hand/object contact', ylabel='N', xlabel='time (s)')
    ax.text(.02, .95, f"normal impulse = {r['object_hand_normal_impulse_Ns']:.3f} N s", transform=ax.transAxes, va='top')
    ax = axes[1, 0]
    ax.plot(tq, np.linalg.norm(obj[1:]-interp[1:], axis=1)*100, label='object target error')
    ax.plot(tq, penetration*100, label='max contact penetration')
    ax.axhline(r['quality_thresholds']['max_object_position_error_m']*100, color='#1879be', ls='--', alpha=.5)
    ax.axhline(r['quality_thresholds']['max_penetration_m']*100, color='#e08018', ls='--', alpha=.5)
    ax.set(title='Position and collision quality', ylabel='cm', xlabel='time (s)')
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    ax.plot(tq, np.max(np.abs(z['actuator_force']), axis=1), color='#874fa3')
    ax.set(title='Peak absolute actuator force (direct hinge motors)', ylabel='N m', xlabel='time (s)')
    fig.suptitle('Vita52 + bowl5 | '+r['status']+'\nAssumed mass/friction; native PD control; simulated forces', fontsize=12)
    for ax in axes.flat:
        ax.grid(alpha=.2)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output, dpi=170)
    plt.close(fig)
    print(str(a.output))


if __name__ == '__main__':
    main()
