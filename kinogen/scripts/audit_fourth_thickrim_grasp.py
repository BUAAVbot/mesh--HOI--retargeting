"""Independent, read-only audit of thickened dishes and actuator-only grasp traces.

Geometry variants are engineered proxies, not recovered ground-truth geometry.
Asset preservation, physics replay and actual lift are separate verdicts.
"""
from pathlib import Path
import argparse
import hashlib
import json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--variant', type=Path, required=True)
    p.add_argument('--trial', type=Path)
    p.add_argument('--initial-trace', type=Path,
                   help='Probe raw initial qpos from a recorded trace without stepping physics')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    import numpy as np
    import mujoco as mj
    from scipy.spatial import ConvexHull

    def load(path):
        return (mj.MjModel.from_binary_path(str(path.resolve())) if path.suffix == '.mjb'
                else mj.MjModel.from_xml_path(str(path.resolve())))

    b, m = load(a.base), load(a.variant)
    object_name = lambda name: (name or '').startswith('target_object_')
    invariant = {}
    for field in ['body_mass', 'body_inertia', 'body_ipos', 'body_iquat', 'body_pos',
                  'body_quat', 'jnt_type', 'jnt_qposadr', 'jnt_dofadr', 'jnt_range',
                  'jnt_pos', 'jnt_axis', 'dof_armature', 'dof_damping', 'eq_type',
                  'eq_obj1id', 'eq_obj2id', 'eq_data', 'eq_active0', 'eq_solref',
                  'eq_solimp', 'actuator_gainprm', 'actuator_biasprm', 'actuator_gear',
                  'actuator_forcerange', 'actuator_forcelimited', 'actuator_trnid']:
        invariant[field] = bool(np.array_equal(getattr(b, field), getattr(m, field)))
    preserved_old_geoms, missing_geoms, changed_geoms = [], [], []
    geom_fields = ['geom_type', 'geom_bodyid', 'geom_pos', 'geom_quat', 'geom_size',
                   'geom_contype', 'geom_conaffinity', 'geom_friction', 'geom_solref',
                   'geom_solimp', 'geom_rgba', 'geom_margin', 'geom_gap']

    def mesh_vertices(model, geom):
        mid = int(model.geom_dataid[geom]); start = int(model.mesh_vertadr[mid])
        return model.mesh_vert[start:start + model.mesh_vertnum[mid]]

    for g in range(b.ngeom):
        name = b.geom(g).name
        if name:
            ng = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, name)
        else:
            bodyname = b.body(int(b.geom_bodyid[g])).name
            ordinal = sum(not b.geom(j).name and b.geom_bodyid[j] == b.geom_bodyid[g] for j in range(g))
            matches = [j for j in range(m.ngeom) if not m.geom(j).name and m.body(int(m.geom_bodyid[j])).name == bodyname]
            ng = matches[ordinal] if ordinal < len(matches) else -1
            name = f'{bodyname}/unnamed_geom_{ordinal}'
        if ng < 0:
            missing_geoms.append(name)
            continue
        fields = [f for f in geom_fields if not np.array_equal(getattr(b, f)[g], getattr(m, f)[ng])]
        if b.geom_type[g] == mj.mjtGeom.mjGEOM_MESH:
            if not np.array_equal(mesh_vertices(b, g), mesh_vertices(m, ng)):
                fields.append('mesh_vertices')
        if fields:
            changed_geoms.append({'geom': name, 'fields': fields})
        else:
            preserved_old_geoms.append(name)
    added = [m.geom(g).name for g in range(m.ngeom) if m.geom(g).name and
             mj.mj_name2id(b, mj.mjtObj.mjOBJ_GEOM, m.geom(g).name) < 0]
    added_only_objects = all(object_name(m.body(int(m.geom_bodyid[m.geom(n).id])).name) for n in added)

    # Point-in-convex collider tests in the object's own frame. Cavity tests are
    # above the original base, where an open dish must not gain a hidden lid.
    objects = []
    for oid in range(1, 7):
        body = m.body(f'target_object_{oid:02d}').id
        old_body = b.body(f'target_object_{oid:02d}').id
        colliders = [g for g in range(m.ngeom) if m.geom_bodyid[g] == body and
                     (m.geom_contype[g] or m.geom_conaffinity[g])]
        old_colliders = [g for g in range(b.ngeom) if b.geom_bodyid[g] == old_body and
                         (b.geom_contype[g] or b.geom_conaffinity[g])]

        def local_vertices(model, g):
            if model.geom_type[g] != mj.mjtGeom.mjGEOM_MESH:
                return None
            rot = np.empty(9); mj.mju_quat2Mat(rot, model.geom_quat[g])
            return mesh_vertices(model, g) @ rot.reshape(3, 3).T + model.geom_pos[g]

        old_top = max(float(local_vertices(b, g)[:, 2].max()) for g in old_colliders
                      if b.geom_type[g] == mj.mjtGeom.mjGEOM_MESH)
        new_top = max(float(local_vertices(m, g)[:, 2].max()) for g in colliders
                      if m.geom_type[g] == mj.mjtGeom.mjGEOM_MESH)
        probes = [np.array([0., 0., old_top + dz]) for dz in [.0005, .002, .004]]
        occupied = []
        for point in probes:
            hits = []
            for g in colliders:
                vertices = local_vertices(m, g)
                if vertices is not None:
                    eq = ConvexHull(vertices).equations
                    inside = np.max(eq[:, :3] @ point + eq[:, 3]) <= 1e-8
                elif m.geom_type[g] == mj.mjtGeom.mjGEOM_CYLINDER:
                    rot = np.empty(9); mj.mju_quat2Mat(rot, m.geom_quat[g])
                    q = rot.reshape(3, 3).T @ (point - m.geom_pos[g])
                    inside = np.linalg.norm(q[:2]) <= m.geom_size[g, 0] and abs(q[2]) <= m.geom_size[g, 1]
                else:
                    raise ValueError('Unsupported collider type in cavity audit')
                if inside:
                    hits.append(m.geom(g).name)
            occupied.append({'point_object_xyz_m': point.tolist(), 'inside_colliders': hits})
        objects.append({'object_id': oid, 'old_top_z_m': old_top, 'new_top_z_m': new_top,
                        'top_extension_m': new_top - old_top,
                        'cavity_above_old_rim_empty': all(not r['inside_colliders'] for r in occupied),
                        'center_cavity_probes': occupied})

    out = {'schema': 'fourth.thickrim_independent_audit.v1',
           'base': {'path': str(a.base), 'sha256': sha(a.base)},
           'variant': {'path': str(a.variant), 'sha256': sha(a.variant)},
           'scope': 'Independent asset invariants plus optional deterministic physics replay; no task success inferred from geometry alone.',
           'model_arrays_preserved': invariant, 'old_geom_count': b.ngeom,
           'old_geoms_preserved_count': len(preserved_old_geoms),
           'missing_old_geoms': missing_geoms, 'changed_old_geoms': changed_geoms,
           'added_geom_count': len(added), 'added_geoms_only_on_six_objects': added_only_objects,
           'objects': objects,
           'asset_preservation_passed': all(invariant.values()) and not missing_geoms and not changed_geoms and added_only_objects and all(o['cavity_above_old_rim_empty'] for o in objects)}
    if a.initial_trace:
        raw = np.load(a.initial_trace, allow_pickle=False)['qpos'][0]
        penetration = []
        for model in [b, m]:
            d = mj.MjData(model); d.qpos[:] = raw; mj.mj_forward(model, d)
            table = {g for g in range(model.ngeom) if (model.geom(g).name or '').startswith('table_support')}
            rows = []
            for c in d.contact[:d.ncon]:
                g1, g2 = int(c.geom1), int(c.geom2)
                n1, n2 = model.body(int(model.geom_bodyid[g1])).name, model.body(int(model.geom_bodyid[g2])).name
                if (g1 in table and object_name(n2)) or (g2 in table and object_name(n1)):
                    rows.append({'object': n1 if object_name(n1) else n2,
                                 'gap_m': float(c.dist), 'geoms': [model.geom(g1).name, model.geom(g2).name]})
            penetration.append({'object_table_contacts': rows,
                                'penetration_max_m': max([max(0., -r['gap_m']) for r in rows], default=0.)})
        out['raw_initial_pose_probe'] = {'trace': str(a.initial_trace), 'trace_sha256': sha(a.initial_trace),
                                         'base': penetration[0], 'variant': penetration[1],
                                         'added_penetration_m': penetration[1]['penetration_max_m']-penetration[0]['penetration_max_m']}
    if a.trial:
        out['physics'] = trace_audit(a.trial, mj, np)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(a.output), 'asset_preservation_passed': out['asset_preservation_passed'],
                      'physics': out.get('physics', {}).get('verdict')}), flush=True)


def trace_audit(trial, mj, np):
    report = json.loads((trial / 'report.json').read_text())
    model_record = report.get('model', {})
    path = model_record if isinstance(model_record, str) else model_record.get('path')
    if not path:
        path = str(next(p for p in [trial/'effective_model.mjb', trial/'plant.mjb'] if p.exists()))
    m = mj.MjModel.from_binary_path(str(Path(path).resolve()))
    with np.load(trial / 'physical_trace.npz', allow_pickle=False) as loaded:
        z = {key: loaded[key] for key in loaded.files}
    t = z['time'] if 'time' in z else z['time_s']
    q, v, u = z['qpos'], z['qvel'], z['ctrl']
    external = {k: float(np.max(np.abs(z[k]))) if k in z else None for k in ['qfrc_applied', 'xfrc_applied']}
    d = mj.MjData(m)
    if 'initial_state' not in z or 'state_spec' not in z:
        raise ValueError('Initial integration state required for independent replay')
    mj.mj_setState(m, d, z['initial_state'], int(z['state_spec'])); mj.mj_forward(m, d)
    qerr = float(np.max(abs(d.qpos - q[0]))); verr = float(np.max(abs(d.qvel - v[0])))
    force_err = 0.
    table = {g for g in range(m.ngeom) if (m.geom(g).name or '').startswith('table_support')}
    objects = [m.body(f'target_object_{oid:02d}').id for oid in range(1, 7)]
    qadr = [int(m.jnt_qposadr[int(m.body_jntadr[body])]) for body in objects]
    table_up = np.zeros((len(u), 6)); hand_up = np.zeros_like(table_up)
    hand_contact = np.zeros_like(table_up, dtype=bool); object_table_pen = np.zeros(6)
    for step, ctrl in enumerate(u):
        d.ctrl[:] = ctrl
        mj.mj_step(m, d)
        qerr = max(qerr, float(np.max(abs(d.qpos-q[step+1]))))
        verr = max(verr, float(np.max(abs(d.qvel-v[step+1]))))
        force_err = max(force_err, float(np.max(abs(d.actuator_force-z['actuator_force'][step]))))
        for ci, c in enumerate(d.contact[:d.ncon]):
            g1, g2 = int(c.geom1), int(c.geom2)
            b1, b2 = int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])
            if b1 not in objects and b2 not in objects:
                continue
            f = np.empty(6); mj.mj_contactForce(m, d, ci, f)
            force2 = np.asarray(c.frame).reshape(3, 3).T @ f[:3]
            for oi, body in enumerate(objects):
                if body not in [b1, b2]:
                    continue
                otherg = g2 if body == b1 else g1
                otherb = b2 if body == b1 else b1
                force = -force2 if body == b1 else force2
                if otherg in table:
                    table_up[step, oi] += force[2]
                    object_table_pen[oi] = max(object_table_pen[oi], float(max(0., -c.dist)))
                if (m.body(otherb).name or '').startswith(('rh_', 'lh_')):
                    hand_up[step, oi] += force[2]
                    hand_contact[step, oi] = True
    mimic = []
    for e in range(m.neq):
        if m.eq_type[e] != mj.mjtEq.mjEQ_JOINT:
            continue
        child, parent = int(m.eq_obj1id[e]), int(m.eq_obj2id[e])
        if parent < 0 or not (m.joint(child).name or '').startswith(('rh_', 'lh_')):
            continue
        cq, pq = int(m.jnt_qposadr[child]), int(m.jnt_qposadr[parent])
        err = q[:, cq]-m.qpos0[cq]-np.polynomial.polynomial.polyval(q[:, pq]-m.qpos0[pq], m.eq_data[e, :5])
        mimic.append(float(np.max(abs(err))))
    hand_aids = [ai for ai in range(m.nu) if (m.actuator(ai).name or '').startswith(('motor_rh_', 'motor_lh_')) and np.any(m.actuator_gear[ai] != 0)]
    passive = np.flatnonzero(np.all(m.actuator_gear == 0, axis=1))
    hand_torque = float(np.max(abs(z['actuator_force'][:, hand_aids])))
    lifts = []
    dt = float(m.opt.timestep)
    baseline_mask = (t >= min(1.5, float(t[-1])*.25)) & (t <= min(2., float(t[-1])*.4))
    post_settle = t[1:] >= min(2., float(t[-1])*.4)
    for oi, adr in enumerate(qadr):
        baseline = float(np.median(q[baseline_mask, adr+2]))
        rise = q[1:, adr+2]-baseline
        good = post_settle & (rise >= .012) & (table_up[:, oi] < .15) & (hand_up[:, oi] >= .7*m.body_mass[objects[oi]]*9.81) & hand_contact[:, oi]
        streak = maximum = 0
        for value in good:
            streak = streak+1 if value else 0
            maximum = max(maximum, streak)
        lifts.append({'object_id': oi+1, 'baseline_z_m': baseline,
                      'max_rise_m': float(rise[post_settle].max()),
                      'rise_scope': 'After initial settling, relative to median1.5-2.0s height; raw initially suspended frames excluded.',
                      'qualified_lift_longest_s': maximum*dt,
                      'lift_at_least_12mm_held_80ms': bool(maximum*dt >= .08-1e-9),
                      'table_penetration_max_m': float(object_table_pen[oi]),
                      'hand_up_force_peak_N': float(hand_up[:, oi].max())})
    non_object_weld = all(m.eq_type[e] != mj.mjtEq.mjEQ_WELD or
                         (int(m.eq_obj1id[e]) not in objects and int(m.eq_obj2id[e]) not in objects) for e in range(m.neq))
    free_objects = all(m.jnt_type[int(m.body_jntadr[b])] == mj.mjtJoint.mjJNT_FREE and m.body_mocapid[b] < 0 for b in objects)
    replay_pass = max(qerr, verr, force_err) < 1e-10
    return {'trial': str(trial), 'duration_s': float(t[-1]), 'steps': len(u),
            'replay_qpos_max_error': qerr, 'replay_qvel_max_error': verr,
            'replay_actuator_force_max_error': force_err, 'deterministic_replay_passed': replay_pass,
            'external_force_maxima': external, 'zero_external_forces': all(v == 0 for v in external.values()),
            'object_state_overrides_excluded_by_replay': replay_pass,
            'objects_are_free_and_not_mocap': free_objects, 'no_object_weld': non_object_weld,
            'mimic_max_error_rad': max(mimic, default=0.), 'mimic_30mrad_passed': max(mimic, default=0.) <= .03,
            'active_hand_torque_max_abs_Nm': hand_torque, 'hard_0p4Nm_passed': hand_torque <= .4+1e-9,
            'passive_force_max_abs_Nm': float(np.max(abs(z['actuator_force'][:, passive]))),
            'objects': lifts, 'all_six_lift_passed': all(r['lift_at_least_12mm_held_80ms'] for r in lifts),
            'verdict': 'See separate replay, physics-contract and per-object lift verdicts; full stacking is not inferred.'}


if __name__ == '__main__':
    main()
