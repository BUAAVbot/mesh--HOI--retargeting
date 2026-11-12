"""Replay a selected physical control sequence without policy or state writes.

Strict source tracking and functional seat exchange use the exact thresholds
in two_sitting_eval_ppo.py. Selecting a successful stochastic candidate never
implies the checkpoint succeeds reliably or reproduces the source trajectory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import mujoco as mj
import numpy as np

from two_sitting_physics import contacts, independent_replay, write_json, interp
from two_sitting_ppo_env import mapping


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metadata_from(candidate):
    for key in ('metadata', 'metadata_json'):
        if key in candidate:
            value = np.asarray(candidate[key])
            if value.size != 1 or value.dtype.kind not in 'US':
                raise ValueError('Candidate metadata must be a scalar Unicode JSON value')
            result = json.loads(str(value.item()))
            if not isinstance(result, dict):
                raise ValueError('Candidate metadata JSON must be an object')
            return result
    return {}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'candidate', 'reference', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    candidate = dict(np.load(a.candidate, allow_pickle=False))
    metadata = metadata_from(candidate)
    raw = dict(np.load(a.reference, allow_pickle=False))
    recorded_ctrl = np.asarray(candidate['ctrl'], dtype=np.float64)
    initial = np.asarray(candidate['initial_state'], dtype=np.float64)
    spec = mj.mjtState(int(np.asarray(candidate['state_spec']).item()))
    if int(spec) != int(mj.mjtState.mjSTATE_INTEGRATION):
        raise ValueError('Expected complete mjSTATE_INTEGRATION candidate state')
    if recorded_ctrl.ndim != 2 or recorded_ctrl.shape[1] != 104 or not len(recorded_ctrl):
        raise ValueError('Candidate ctrl must be nonempty[Nphysics,104]')
    if not np.isfinite(recorded_ctrl).all() or not np.isfinite(initial).all():
        raise ValueError('Nonfinite candidate state/control')
    m = mj.MjModel.from_binary_path(str(a.model.resolve()))
    if (m.nq, m.nv, m.nu) != (118, 116, 104) or abs(float(m.opt.timestep)-.005) > 1e-10:
        raise ValueError('Expected frozen dual-Vita118/116/104 plant at5ms')
    if m.nmocap:
        raise ValueError('Mocap forbidden')
    if len(initial) != mj.mj_stateSize(m, spec):
        raise ValueError('Candidate initial integration state has wrong size')
    names = list(map(str, raw['dof_names']))
    maps = [mapping(m, names, prefix) for prefix in ('p1_', 'p2_')]
    roots = {mp['root'] for mp in maps}
    if any(int(j) in roots for j in m.actuator_trnid[:, 0]):
        raise ValueError('Root actuators forbidden')
    if any(m.eq_type != mj.mjtEq.mjEQ_JOINT):
        raise ValueError('Only joint-coupling equalities permitted')
    # Hashes in the selected candidate may be flat or under an inputs object.
    for bag in (metadata, metadata.get('inputs', {})):
        if not isinstance(bag, dict):
            continue
        for key, path in (('model_sha256', a.model), ('reference_sha256', a.reference),
                          ('ref_sha256', a.reference)):
            expected = bag.get(key)
            if expected is not None and str(expected) != sha(path):
                raise ValueError('Candidate input hash mismatch: '+key)
    ref = SimpleNamespace(names=names, maps=maps,
                          root_qadr=np.asarray([x['rq'] for x in maps]),
                          root_vadr=np.asarray([x['rv'] for x in maps]),
                          body_cols=np.asarray([j for j,n in enumerate(names)
                                                if not n.startswith(('lh_', 'rh_', 'neck_'))]),
                          end=float(raw['time_s'][-1]))
    ref.horizon = ref.end+1.
    expected_physics_steps = int(np.ceil(ref.horizon/.02-1e-9))*4
    if len(recorded_ctrl) != expected_physics_steps:
        raise ValueError('Candidate must contain exactly the native-eval source+1s horizon: '+
                         str(expected_physics_steps)+' physics controls')
    a.output.mkdir(parents=True)
    binary = a.output/'model.mjb'; shutil.copyfile(a.model, binary)
    sources = a.output/'source_snapshot'; sources.mkdir()
    for name in ('two_sitting_replay_candidate.py', 'two_sitting_eval_ppo.py',
                 'two_sitting_physics.py', 'two_sitting_ppo_env.py'):
        shutil.copyfile(Path(__file__).with_name(name), sources/name)
    d = mj.MjData(m)
    mj.mj_setState(m, d, initial, spec)
    mj.mj_forward(m, d)
    if abs(float(d.time)) > 1e-9:
        raise ValueError('Full-source candidate must start at simulation time0')
    if np.any(d.qfrc_applied) or np.any(d.xfrc_applied):
        raise ValueError('Candidate contains applied external forces')
    np.savez_compressed(a.output/'initial_state.npz', state=initial, state_spec=int(spec),
                        qpos=d.qpos.copy(), qvel=d.qvel.copy())
    geom_names = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, i) or str(i) for i in range(m.ngeom)]
    bodies = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, i) or str(i) for i in range(m.nbody)]
    ts = [float(d.time)]; qs = [d.qpos.copy()]; vs = [d.qvel.copy()]
    controls, forces, torques, rows, seats, feet, sf, pens, ips = [], [], [], [], [], [], [], [], []
    reason, began = 'completed', time.monotonic()
    warning = np.asarray([w.number for w in d.warning])
    for step, u in enumerate(recorded_ctrl):
        d.ctrl[:] = u
        mj.mj_step(m, d)
        records, seat, foot, seatforce, pen, ip = contacts(m, d, len(controls), geom_names, bodies)
        rows.extend(records); seats.append(seat); feet.append(foot); sf.append(seatforce)
        pens.append(pen); ips.append(ip)
        controls.append(u.copy()); forces.append(d.actuator_force.copy()); torques.append(d.qfrc_actuator.copy())
        qs.append(d.qpos.copy()); vs.append(d.qvel.copy()); ts.append(float(d.time))
        if np.any(d.qfrc_applied) or np.any(d.xfrc_applied):
            raise ValueError('External force forbidden')
        if (not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all() or
                np.any(np.asarray([w.number for w in d.warning]) > warning)):
            reason = 'numeric_failure'; break
        if (step+1) % 4 == 0 and any(d.qpos[rq+2] < .15 for rq in ref.root_qadr):
            reason = 'root_below_0p15m'; break
        if (step+1) % 1000 == 0:
            print(json.dumps({'physics_steps': step+1, 'simulated_s': float(d.time)}), flush=True)
    qs, vs, ts, controls = np.asarray(qs), np.asarray(vs), np.asarray(ts), np.asarray(controls)
    desired = np.asarray([interp(raw, t) for t in ts])
    xy, ze, oe, je = [], [], [], []
    for i, mp in enumerate(ref.maps):
        actual = qs[:, mp['rq']:mp['rq']+7]
        xy.append(np.linalg.norm(actual[:, :2]-desired[:, i, :2], axis=-1))
        ze.append(abs(actual[:, 2]-desired[:, i, 2]))
        oe.append(2*np.arccos(np.clip(abs(np.sum(actual[:, 3:7]*desired[:, i, 3:7], axis=-1)), 0, 1)))
        je.append(np.sqrt(np.mean((qs[:, mp['qa'][ref.body_cols]]-desired[:, i, 7+ref.body_cols])**2, axis=-1)))
    xy, ze, oe, je = np.asarray(xy).T, np.asarray(ze).T, np.asarray(oe).T, np.asarray(je).T
    seats = np.asarray(seats); tail = ts[1:] >= ref.horizon-.2
    fraction = seats[tail].mean(0) if tail.any() else np.zeros(2)
    tail_states = ts >= ref.horizon-.2
    final_linear_speed = np.asarray([np.linalg.norm(vs[tail_states, rv:rv+3], axis=-1).max()
                                     for rv in ref.root_vadr]) if tail_states.any() else np.full(2, np.inf)
    final_angular_speed = np.asarray([np.linalg.norm(vs[tail_states, rv+3:rv+6], axis=-1).max()
                                      for rv in ref.root_vadr]) if tail_states.any() else np.full(2, np.inf)
    weight = np.asarray([m.body_subtreemass[mp['base']]*np.linalg.norm(m.opt.gravity) for mp in ref.maps])
    final_seat_force = np.asarray(sf)[tail].mean(0) if tail.any() else np.zeros(2)
    replay = independent_replay(binary, initial, spec, controls, qs, vs)
    # Exact conditions copied from two_sitting_eval_ppo.py, with no relaxation.
    checks = {'full_source_duration': bool(ts[-1] >= ref.end-1e-8),
        'completed_requested_duration': bool(reason == 'completed' and ts[-1] >= ref.horizon-1e-8),
        'root_xy_error_below_0p25m': bool(xy.max() < .25),
        'root_z_error_below_0p18m': bool(ze.max() < .18),
        'root_orientation_error_below_0p65rad': bool(oe.max() < .65),
        'penetration_below_0p025m': bool(max(pens, default=0.) < .025),
        'body_joint_rms_below_0p4rad': bool(np.median(je, axis=0).max() < .4),
        'final_both_people_have_true_seat_contact': bool(len(seats) and seats[-1].all() and np.all(fraction >= .5)),
        'final_roots_stable': bool(np.all(final_linear_speed < .15) and np.all(final_angular_speed < .3)),
        'final_seat_bears_at_least_ten_percent_body_weight': bool(np.all(final_seat_force >= .10*weight)),
        'independent_control_only_replay': replay['passed'],
        'no_state_overrides_or_external_forces': True}
    root_height = np.stack([qs[:, rq+2] for rq in ref.root_qadr], axis=1)
    root_tilt = np.stack([np.arccos(np.clip(1.-2.*(qs[:, rq+4]**2+qs[:, rq+5]**2), -1., 1.))
                          for rq in ref.root_qadr], axis=1)
    exchanged_distance = np.asarray([np.linalg.norm(qs[-1, rq:rq+2]-qs[0, rq:rq+2]) for rq in ref.root_qadr])
    functional_checks = {key:checks[key] for key in ('full_source_duration', 'completed_requested_duration',
        'penetration_below_0p025m', 'final_both_people_have_true_seat_contact', 'final_roots_stable',
        'final_seat_bears_at_least_ten_percent_body_weight', 'independent_control_only_replay',
        'no_state_overrides_or_external_forces')}
    functional_checks.update(both_rise_from_seat=bool(np.all(root_height.max(0)-root_height[0] > .12)),
        both_exchange_seat_positions=bool(np.all(exchanged_distance > 1.2) and np.all(xy[-1] < .30)),
        no_fall=bool(root_height.min() > .45 and root_tilt.max() < .8))
    consistency = {'available': False}
    if 'qpos_policy' in candidate and 'qvel_policy' in candidate:
        expected_q = np.asarray(candidate['qpos_policy']); expected_v = np.asarray(candidate['qvel_policy'])
        actual_q, actual_v = qs[::4], vs[::4]
        count = min(len(expected_q), len(actual_q), len(expected_v), len(actual_v))
        qerror = float(np.max(abs(actual_q[:count]-expected_q[:count]))) if count else None
        verror = float(np.max(abs(actual_v[:count]-expected_v[:count]))) if count else None
        tolerance = 1e-9 if expected_q.dtype.itemsize >= 8 and expected_v.dtype.itemsize >= 8 else 1e-5
        consistency = {'available': True, 'states_compared': count,
            'complete': bool(count == len(expected_q) == len(actual_q)),
            'candidate_qpos_dtype': str(expected_q.dtype), 'candidate_qvel_dtype': str(expected_v.dtype),
            'max_qpos_absolute_error': qerror, 'max_qvel_absolute_error': verror,
            'tolerance_for_candidate_storage_precision': tolerance,
            'passed': bool(count and qerror <= tolerance and verror <= tolerance and count == len(expected_q) == len(actual_q))}
    policy_time = np.asarray(candidate.get('policy_time_s', ts[::4]))
    if len(policy_time) == len(recorded_ctrl)//4+1:
        policy_time = policy_time[:-1]
    fields = dict(time=ts, qpos=qs, qvel=vs, ctrl=controls, actuator_force=np.asarray(forces),
        qfrc_actuator=np.asarray(torques), initial_state=initial, state_spec=int(spec),
        joint_names=np.asarray(ref.names), joint_qadr=np.asarray([x['qa'] for x in ref.maps]),
        joint_vadr=np.asarray([x['va'] for x in ref.maps]), actuator_ids=np.asarray([x['aids'] for x in ref.maps]),
        robot_root_qadr=ref.root_qadr, robot_root_vadr=ref.root_vadr,
        source_time_s=np.interp(ts, raw['time_s'], raw['source_time_s']), reference_qpos=desired,
        root_xy_error_m=xy, root_z_error_m=ze, root_orientation_error_rad=oe,
        seat_contact=seats, foot_contact=np.asarray(feet), seat_support_force_N=np.asarray(sf),
        geom_names=np.asarray(geom_names), body_names=np.asarray(bodies),
        policy_time_s=policy_time,
        candidate_policy_state_time_s=np.asarray(candidate.get('policy_time_s', [])),
        learned_residual_action=np.asarray(candidate.get('learned_residual_action', np.empty((0, 56)))))
    for key, i, shape, dtype in [('interval_index',0,(),int),('time_s',1,(),float),
            ('geom_ids',2,(2,),int),('body_ids',3,(2,),int),('position_world_m',4,(3,),float),
            ('force_world_on_geom2_N',5,(3,),float),('torque_world_on_geom2_Nm',6,(3,),float),
            ('normal_force_N',7,(),float),('signed_distance_m',8,(),float)]:
        fields['contact_'+key] = np.asarray([r[i] for r in rows], dtype=dtype).reshape((-1,)+shape)
    np.savez_compressed(a.output/'trace.npz', **fields)
    strict_passed, functional_passed = bool(all(checks.values())), bool(all(functional_checks.values()))
    status = ('passed' if strict_passed else 'functional_task_completed_strict_tracking_failed'
              if functional_passed else 'failed_physical_quality')
    report = dict(success=strict_passed, source_tracking_pass=strict_passed,
        functional_physical_demo_pass=functional_passed,
        status=status, checks=checks, termination=reason,
        functional_task_success=functional_passed, functional_task_checks=functional_checks,
        simulated_seconds=float(ts[-1]), source_duration_seconds=float(raw['source_time_s'][-1]),
        execution_reference_duration_seconds=ref.end, physics_steps=len(controls),
        max_root_xy_error_m=xy.max(0).tolist(), max_root_z_error_m=ze.max(0).tolist(),
        max_root_orientation_error_rad=oe.max(0).tolist(), max_penetration_m=max(pens,default=0.),
        max_interperson_penetration_m=max(ips,default=0.), final_0p2s_seat_contact_fraction=fraction.tolist(),
        final_root_linear_speed_max_m_s=final_linear_speed.tolist(),
        final_root_angular_speed_max_rad_s=final_angular_speed.tolist(),
        final_seat_vertical_force_mean_N=final_seat_force.tolist(), robot_weight_N=weight.tolist(),
        root_height_min_m=root_height.min(0).tolist(), root_global_tilt_max_rad=root_tilt.max(0).tolist(),
        exchanged_distance_m=exchanged_distance.tolist(), final_root_xy_error_m=xy[-1].tolist(),
        contact_records=len(rows), independent_replay=replay, candidate_policy_state_consistency=consistency,
        candidate_provenance={'candidate_path':str(a.candidate.resolve()),'candidate_sha256':sha(a.candidate),
                              'metadata':metadata,'recorded_control_steps':len(recorded_ctrl)},
        controller={'method':'Control-only replay of selected native physical stochastic rollout',
                    'policy_inference_during_replay':False,
                    'checkpoint':metadata.get('checkpoint_path',metadata.get('checkpoint')),
                    'checkpoint_sha256':metadata.get('checkpoint_sha256')},
        state_override_after_reset_count=0, external_force_steps=0, root_actuators=0, mocap_bodies=m.nmocap,
        mujoco_version=mj.__version__, wall_seconds=time.monotonic()-began,
        model_dimensions={'nq':m.nq,'nv':m.nv,'nu':m.nu},
        inputs={'reference':str(a.reference.resolve()),'reference_sha256':sha(a.reference),
                'model':str(a.model.resolve()),'model_sha256':sha(a.model)},
        artifacts={name:{'path':str((a.output/name).resolve()),'sha256':sha(a.output/name)}
                   for name in ('model.mjb','initial_state.npz','trace.npz')},
        policy_action_log_available=bool(len(fields['learned_residual_action'])),
        limitations=['Selected candidate among multiple stochastic attempts; not an independent policy success-rate estimate.',
                     'Replay executes recorded motor controls; it does not independently rerun closed-loop policy inference.',
                     'Functional completion and strict source-trajectory replication are reported separately.',
                     'Scene friction, collision support and scale remain reconstruction-based assumptions.'])
    write_json(a.output/'report.json', report)
    print(json.dumps({'report':str(a.output/'report.json'),'success':strict_passed,
                      'functional_task_success':functional_passed,'simulated_s':float(ts[-1])}),flush=True)


if __name__ == '__main__':
    main()
