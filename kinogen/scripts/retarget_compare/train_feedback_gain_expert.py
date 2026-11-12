"""Model-free CEM optimization of an actual-state joint feedback policy.

The immutable200Hz motor prior is augmented by50Hz q/v-error feedback, with a
three-physics-step feedback delay. CEM changes closed-loop gains, never roots,
model physics, or the nominal motor plan. Not the official SUGAR PPO trainer.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--population', type=int, default=24)
    p.add_argument('--generations', type=int, default=8)
    p.add_argument('--training-worlds', type=int, default=4)
    p.add_argument('--threads', type=int, default=6)
    p.add_argument('--seed', type=int, default=174210)
    p.add_argument('--noise', type=float, default=.002)
    p.add_argument('--max-offset', type=float, default=.25)
    p.add_argument('--screen-only', action='store_true')
    p.add_argument('--gains-json', type=Path)
    p.add_argument('--heldout', type=Path)
    a = p.parse_args()
    import numpy as np
    import mujoco as mj
    from two_sitting_physics import contacts

    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    prior_path = a.root / 'terminal_refined_full_v4/trace.npz'
    model_path = a.root / 'calibration/two_vita_expert.mjb'
    ref_path = a.root / 'references/phc_transfer_time1p5.npz'
    z = dict(np.load(prior_path, allow_pickle=False))
    model = mj.MjModel.from_binary_path(str(model_path))
    controls, planq, planv = z['ctrl'], z['qpos'], z['qvel']
    spec = mj.mjtState(int(z['state_spec']))
    names = list(map(str, z['joint_names']))
    cols = [i for i, name in enumerate(names) if not name.startswith(('lh_', 'rh_', 'neck_'))]
    qa = z['joint_qadr'][:, cols].ravel(); va = z['joint_vadr'][:, cols].ravel()
    aids = z['actuator_ids'][:, cols].ravel()
    if len(aids) != 56:
        raise ValueError('Expected56body motors')
    jids = model.actuator_trnid[aids, 0]
    lower, upper = model.jnt_range[jids, 0], model.jnt_range[jids, 1]
    rq, rv = z['robot_root_qadr'], z['robot_root_vadr']
    bn = list(map(str, z['body_names'])); gn = list(map(str, z['geom_names']))
    weights = np.array([model.body_subtreemass[mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, prefix+'base_link')]*9.81 for prefix in ('p1_', 'p2_')])
    target = np.load(ref_path, allow_pickle=False)['qpos'][-1, :, :2]
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    rng = np.random.default_rng(a.seed)
    states = []
    for _ in range(a.training_worlds):
        d = mj.MjData(model); mj.mj_setState(model, d, z['initial_state'], spec)
        d.qpos[qa] = np.clip(d.qpos[qa] + rng.uniform(-a.noise, a.noise, len(qa)), lower, upper)
        for root in rq:
            d.qpos[root:root+2] += rng.uniform(-a.noise, a.noise, 2)
        mj.mj_forward(model, d)
        s = np.empty(mj.mj_stateSize(model, spec)); mj.mj_getState(model, d, s, spec)
        states.append(s)
    np.savez_compressed(a.output/'training_initial_states.npz', integration_state=np.array(states), integration_state_spec=int(spec))
    manifest = {'method_kind': 'model_free_CEM_closed_loop_joint_feedback_gain_expert_not_PPO',
                'source_sha256': sha(Path(__file__)), 'prior': str(prior_path), 'prior_sha256': sha(prior_path),
                'model': str(model_path), 'model_sha256': sha(model_path),
                'reference': str(ref_path), 'reference_sha256': sha(ref_path),
                'policy': 'nominal200Hz motor packet + clip(kq*(planq-actualq)+kv*(planv-actualv), +/-offset) on56body motors;50Hz update/3step delay',
                'source_tracking_or_paper_reproduction_claim': False,
                'mid_episode_state_writes': 0, 'external_forces': False,
                'training_seed': a.seed, 'training_noise_joint_rad_root_xy_m': a.noise,
                'heldout_seed990013_not_used_for_learning': True,
                'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}}
    (a.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    began = time.monotonic()

    def evaluate(task):
        gains, initial, capture = task
        kq, kv = gains
        d = mj.MjData(model); mj.mj_setState(model, d, initial, spec)
        # Match the full-state evaluator's exact reset contract.
        mj.mj_forward(model, d); mj.mj_setState(model, d, initial, spec)
        initial_xy = np.array([d.qpos[r:r+2].copy() for r in rq])
        initial_z = d.qpos[rq+2].copy(); maximum_z = initial_z.copy()
        offset = np.zeros(56); previous = offset.copy()
        minheight, maxtilt, penetration, total_error = 10., 0., 0., 0.
        tailforces, tailseats, speeds, spins = [], [], [], []
        qhist = [d.qpos.copy()] if capture else []
        vhist = [d.qvel.copy()] if capture else []
        uhist = []
        terminated = 'completed'; largest = 0.
        for k, base in enumerate(controls):
            if k % 4 == 0:
                previous = offset.copy()
                offset = np.clip(kq*(planq[k, qa]-d.qpos[qa]) + kv*(planv[k, va]-d.qvel[va]), -a.max_offset, a.max_offset)
            applied = previous if k % 4 < 3 else offset
            command = base.copy()
            if np.any(applied):
                command[aids] = np.clip(command[aids]+applied, lower, upper)
            largest = max(largest, float(abs(command[aids]-base[aids]).max()))
            d.ctrl[:] = command; mj.mj_step(model, d)
            if capture:
                qhist.append(d.qpos.copy()); vhist.append(d.qvel.copy()); uhist.append(command)
            if np.any([warning.number for warning in d.warning]) or not np.isfinite(d.qpos).all():
                terminated = 'numeric_warning'; break
            minheight = min(minheight, float(d.qpos[rq+2].min()))
            maximum_z = np.maximum(maximum_z, d.qpos[rq+2])
            tilt = np.arccos(np.clip(1-2*(d.qpos[rq+4]**2+d.qpos[rq+5]**2), -1, 1))
            maxtilt = max(maxtilt, float(tilt.max()))
            if d.ncon:
                penetration = max(penetration, max(0., float(-np.min(d.contact.dist[:d.ncon]))))
            total_error += float(np.mean((d.qpos[rq[:, None]+np.arange(3)]-planq[k+1, rq[:, None]+np.arange(3)])**2))
            if k >= len(controls)-40:
                _, seats, _, force, _, _ = contacts(model, d, k, gn, bn)
                tailseats.append(seats); tailforces.append(force)
                speeds.append([np.linalg.norm(d.qvel[r:r+3]) for r in rv])
                spins.append([np.linalg.norm(d.qvel[r+3:r+6]) for r in rv])
            if minheight <= .45 or maxtilt >= .8 or penetration >= .05:
                terminated = 'physical_safety_gate'; break
        duration = (k+1)*model.opt.timestep
        xy = np.linalg.norm(np.array([d.qpos[r:r+2] for r in rq])-target, axis=-1)
        force = np.mean(tailforces, 0) if tailforces else np.zeros(2)
        fraction = np.mean(tailseats, 0) if tailseats else np.zeros(2)
        speed = np.max(speeds, 0) if speeds else np.full(2, 100.)
        spin = np.max(spins, 0) if spins else np.full(2, 100.)
        travel = np.linalg.norm(np.array([d.qpos[r:r+2] for r in rq])-initial_xy, axis=-1)
        checks = {'complete': terminated == 'completed', 'no_fall': minheight > .45 and maxtilt < .8,
                  'penetration_below_25mm': penetration < .025,
                  'both_rise': bool(np.all(maximum_z-initial_z > .12)),
                  'exchange': bool(np.all(travel > 1.2) and np.all(xy < .3)),
                  'seat_contact': bool(np.all(fraction >= .5)), 'seat_load': bool(np.all(force >= .1*weights)),
                  'final_stable': bool(np.all(speed < .15) and np.all(spin < .3))}
        rms = (total_error/(k+1))**.5
        score = 10*duration - 2*min(rms, 5.) + 100*int(all(checks.values()))
        row = {'gains': list(map(float, gains)), 'score': score, 'duration_s': duration,
               'functional_pass': all(checks.values()), 'checks': checks, 'termination': terminated,
               'physical_plan_root_xyz_rms_m': rms, 'final_xy_m': xy.tolist(),
               'seat_force_N': force.tolist(), 'max_penetration_m': penetration,
               'max_applied_feedback_rad': largest}
        if capture:
            row['_q'] = np.array(qhist); row['_v'] = np.array(vhist); row['_u'] = np.array(uhist)
        return row

    with ThreadPoolExecutor(max_workers=a.threads) as executor:
        def batch(gains, initials):
            rows = list(executor.map(evaluate, [(g, s, False) for g in gains for s in initials]))
            return [{'gains': list(map(float, g)), 'score': float(np.mean([x['score'] for x in rows[i*len(initials):(i+1)*len(initials)]])),
                     'functional_passes': sum(x['functional_pass'] for x in rows[i*len(initials):(i+1)*len(initials)]),
                     'episodes': rows[i*len(initials):(i+1)*len(initials)]} for i, g in enumerate(gains)]
        fixed = np.array([[0, 0], [.1, .005], [.5, .02], [1, .05], [2, .1], [3, .2], [-.5, .02], [0, .1]])
        screening = batch(fixed, states)
        (a.output/'fixed_gain_training_screen.json').write_text(json.dumps(screening, indent=2)+'\n')
        best = max(screening, key=lambda r: r['score']); center = np.array(best['gains']); std = np.array([1., .06])
        if a.gains_json:
            best = json.loads(a.gains_json.read_text())['best']; center = np.array(best['gains'])
        if not a.screen_only:
            for generation in range(a.generations):
                gains = np.clip(rng.normal(center, std, (a.population, 2)), [-.8, -.01], [5., .3])
                gains[0] = best['gains']
                results = sorted(batch(gains, states), key=lambda r: r['score'], reverse=True)
                if results[0]['score'] > best['score']:
                    best = results[0]
                elite = np.array([x['gains'] for x in results[:max(4, a.population//4)]])
                center = .25*center + .75*elite.mean(0)
                std = np.maximum([.02, .002], .25*std + .75*elite.std(0))
                row = {'generation': generation+1, 'best': best, 'center': center.tolist(), 'std': std.tolist(),
                       'wall_seconds': time.monotonic()-began}
                with (a.output/'generations.jsonl').open('a') as f:
                    f.write(json.dumps(row)+'\n')
                (a.output/'checkpoint.json').write_text(json.dumps(row, indent=2)+'\n')
                print(json.dumps({'generation': generation+1, 'score': best['score'], 'train_passes': best['functional_passes'], 'gains': best['gains']}), flush=True)
        nominal = evaluate((best['gains'], z['initial_state'], True))
        dq = float(np.max(abs(nominal.pop('_q')-planq))) if nominal['termination'] == 'completed' else None
        dv = float(np.max(abs(nominal.pop('_v')-planv))) if nominal['termination'] == 'completed' else None
        du = float(np.max(abs(nominal.pop('_u')-controls))) if nominal['termination'] == 'completed' else None
        nominal.update(max_q_vs_nominal=dq, max_v_vs_nominal=dv, max_control_vs_nominal=du,
                       inherited_nominal_not_learning_gain=True)
        if dq != 0 or dv != 0 or du != 0:
            raise RuntimeError('Feedback policy did not preserve exact nominal trajectory')
        heldout = None
        if a.heldout:
            bag = dict(np.load(a.heldout, allow_pickle=False))
            key = next(k for k in ('integration_state', 'initial_state', 'states') if k in bag)
            heldout = batch([best['gains']], bag[key])[0]
            heldout['initial_states_sha256'] = sha(a.heldout)
            heldout['training_used_these_states'] = False
        report = {'status': 'closed_loop_feedback_gain_experiment_complete', 'best': best, 'nominal_invariance': nominal,
                  'heldout': heldout, 'wall_seconds': time.monotonic()-began,
                  'strict_source_tracking_success_claim': False, 'PPO_success_claim': False}
        (a.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({'status': report['status'], 'train_passes': best['functional_passes'], 'heldout_passes': None if heldout is None else heldout['functional_passes']}), flush=True)


if __name__ == '__main__':
    main()
