"""Plot original-vs-optimized shoe motion, physics tracking and attachment loads."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trial', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    r = json.loads((a.trial/'report.json').read_text())
    z = np.load(a.trial/'trace.npz', allow_pickle=False)
    c = np.load(a.trial/'contacts.npz', allow_pickle=False)
    ref = np.load(r['input']['reference']['path'], allow_pickle=False)
    t = z['time']
    rt = ref['time_s']-ref['time_s'][0]
    oq = int(z['object_qadr'])
    fig, axs = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    ax = axs[0,0]
    ax.plot(t, z['qpos'][:,oq+2], label='actual shoe')
    ax.plot(rt, ref['object_target_pose'][:,2], '--', label='contact-optimized reference')
    ax.plot(rt, ref['shoe_raw_source_pose'][:,2], ':', label='original source estimate')
    ax.set(title='Shoe height: raw source, optimized reference and physics', xlabel='time (s)', ylabel='height (m)')
    ax.legend(fontsize=8)
    ax = axs[0,1]
    ax.plot(t[1:], z['shoe_reference_error_m']*100, label='error vs optimized reference')
    if 'raw_source_shoe_error_m' in z:
        ax.plot(t[1:], z['raw_source_shoe_error_m']*100, label='error vs original source')
    ax.axhline(r['quality_thresholds']['max_shoe_position_error_m']*100, ls='--', color='#777', label='optimized-reference gate')
    ax.set(title='Shoe tracking errors kept separate', xlabel='time (s)', ylabel='distance (cm)')
    ax.legend(fontsize=8)
    ax = axs[1,0]
    for actor, color in enumerate(['#1879be', '#ae8528']):
        ax.plot(t[1:], z['root_horizontal_error_m'][:,actor]*100, color=color, label=f'actor {actor+1} XY')
        ax.plot(t[1:], z['root_height_error_m'][:,actor]*100, '--', color=color, label=f'actor {actor+1} height')
    ax.set(title='Both robot roots vs optimized reference', xlabel='time (s)', ylabel='distance (cm)')
    ax.legend(fontsize=8, ncol=2)
    ax = axs[1,1]
    if 'eq_constraint_wrench_world_on_shoe' in z:
        force = np.linalg.norm(z['eq_constraint_wrench_world_on_shoe'][:,:,:3], axis=2)
        for j, name in enumerate(z['equality_names']):
            if np.max(force[:,j]) > .001:
                ax.plot(t[1:], force[:,j], label=str(name))
    shoe_body = next((i for i,n in enumerate(c['body_names']) if str(n)=='shoe'), -1)
    contact = np.zeros(len(t)-1)
    if shoe_body >= 0:
        mask = np.any(c['body_ids']==shoe_body, axis=1)
        np.add.at(contact, c['interval_index'][mask], c['normal_force_N'][mask])
    ax.plot(t[1:], contact, alpha=.6, label='all shoe collision normal forces')
    ax.set(title='Constraint loads and collision forces (simulated)', xlabel='time (s)', ylabel='force (N)')
    ax.legend(fontsize=7)
    for ax in axs.flat:
        ax.grid(alpha=.2)
    fig.suptitle('RGB near scene | '+r['status']+'\nMonocular source; assumed physical parameters; inferred wear/grasp constraints', fontsize=12)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output, dpi=160)
    plt.close(fig)
    print(str(a.output))


if __name__ == '__main__':
    main()
