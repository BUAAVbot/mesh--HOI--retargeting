"""Independently replay controls and require a supported, sustained plate lift."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import mujoco as mj

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--trial', type=Path, required=True)
p.add_argument('--base-model', type=Path, required=True)
p.add_argument('--object-id', type=int, default=1)
p.add_argument('--clearance-m', type=float, default=.03)
p.add_argument('--hold-s', type=float, default=2.)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
m = mj.MjModel.from_binary_path(str(a.trial/'effective_model.mjb'))
b = (mj.MjModel.from_binary_path(str(a.base_model)) if a.base_model.suffix == '.mjb'
     else mj.MjModel.from_xml_path(str(a.base_model.resolve())))
z = dict(np.load(a.trial/'physical_trace.npz', allow_pickle=False))
times = next(z[k] for k in ['time_s', 'time', 'times'] if k in z)
if len(times) != len(z['ctrl'])+1 or np.any(np.diff(times) <= 0):
    raise ValueError('Incomplete or nonmonotonic physics trace')
def hand_shapes(model):
    result = {}
    for g in range(model.ngeom):
        name = model.body(int(model.geom_bodyid[g])).name or ''
        if not name.startswith(('rh_', 'lh_')):
            continue
        row = {key: np.asarray(getattr(model, key)[g]).tolist() for key in
               ['geom_type', 'geom_pos', 'geom_quat', 'geom_size', 'geom_contype',
                'geom_conaffinity', 'geom_friction', 'geom_margin', 'geom_gap']}
        if model.geom_type[g] == mj.mjtGeom.mjGEOM_MESH:
            mi = int(model.geom_dataid[g]); start = int(model.mesh_vertadr[mi])
            vertices = model.mesh_vert[start:start+int(model.mesh_vertnum[mi])]
            start = int(model.mesh_faceadr[mi]); faces = model.mesh_face[start:start+int(model.mesh_facenum[mi])]
            row['mesh_sha256'] = hashlib.sha256(vertices.tobytes()+faces.tobytes()).hexdigest()
        result[model.geom(g).name] = row
    return result
hands_unchanged = hand_shapes(m) == hand_shapes(b)
d, k = mj.MjData(m), mj.MjData(m)
mj.mj_setState(m, d, z['initial_state'], int(z['state_spec']))
mj.mj_forward(m, d)
obj = m.body(f'target_object_{a.object_id:02d}').id
obj_vadr = int(m.jnt_dofadr[int(m.body_jntadr[obj])])
geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == obj and m.geom_contype[g]]
table = {g for g in range(m.ngeom) if (m.geom(g).name or '').startswith('table_support')}
if not table:
    raise ValueError('No finite table collision geometry')
plane = np.array([.010271048830894367, .02292500590934548, .999684425036206, 1.0058665495441206])
plane /= np.linalg.norm(plane[:3]); n = plane[:3]
def clearance():
    values = []
    for g in geoms:
        R = k.geom_xmat[g].reshape(3, 3); center = k.geom_xpos[g]
        if m.geom_type[g] == mj.mjtGeom.mjGEOM_MESH:
            mi = int(m.geom_dataid[g]); start = int(m.mesh_vertadr[mi])
            vertices = m.mesh_vert[start:start+int(m.mesh_vertnum[mi])]
            values.append(float(np.min(vertices @ R.T @ n + center@n + plane[3])))
        elif m.geom_type[g] == mj.mjtGeom.mjGEOM_CYLINDER:
            axis = float(R[:, 2]@n)
            values.append(float(center@n+plane[3]-m.geom_size[g, 1]*abs(axis)
                                -m.geom_size[g, 0]*np.sqrt(max(0., 1-axis*axis))))
        else:
            raise ValueError('Unsupported object collider; add an exact support formula')
    return min(values)
qerror = verror = torque_error = 0.
rows = []; current_start = None; best = (0., None, None)
for step, ctrl in enumerate(z['ctrl']):
    d.ctrl[:] = ctrl; mj.mj_step(m, d)
    qerror = max(qerror, float(np.max(abs(d.qpos-z['qpos'][step+1]))))
    verror = max(verror, float(np.max(abs(d.qvel-z['qvel'][step+1]))))
    torque_error = max(torque_error, float(np.max(abs(d.actuator_force-z['actuator_force'][step]))))
    hand_up = table_load = nonhand_load = 0.; hand_contacts = 0
    for ci, c in enumerate(d.contact[:d.ncon]):
        gs = list(map(int, c.geom)); bs = list(map(int, m.geom_bodyid[gs]))
        if obj not in bs:
            continue
        oi = bs.index(obj); other = 1-oi; force = np.empty(6)
        mj.mj_contactForce(m, d, ci, force)
        world_force = (1 if oi == 1 else -1)*np.asarray(c.frame).reshape(3, 3).T@force[:3]
        if gs[other] in table:
            table_load += max(float(force[0]), 0.)
        if (m.body(bs[other]).name or '').startswith(('rh_', 'lh_')):
            hand_up += float(world_force[2]); hand_contacts += int(force[0] > .03)
        elif force[0] > 0:
            nonhand_load += float(force[0])
    k.qpos[:] = d.qpos; mj.mj_kinematics(m, k)
    gap = clearance(); upright = float(k.xmat[m.body('base_link').id].reshape(3, 3)[2, 2])
    linear_speed = float(np.linalg.norm(d.qvel[obj_vadr:obj_vadr+3]))
    angular_speed = float(np.linalg.norm(d.qvel[obj_vadr+3:obj_vadr+6]))
    good = (gap >= a.clearance_m and table_load < .15 and nonhand_load < .15
            and hand_contacts > 0 and hand_up >= .7*m.body_mass[obj]*9.81
            and upright >= .8 and linear_speed < .05 and angular_speed < .3)
    now = float(times[step+1])
    if good:
        if current_start is None: current_start = float(times[step])
        duration = now-current_start
        if duration > best[0]: best = (duration, current_start, now)
    else: current_start = None
    rows.append([now, gap, hand_up, table_load, hand_contacts, upright, int(good),
                 nonhand_load, linear_speed, angular_speed])
external_zero = all(key in z and not np.any(z[key]) for key in ['qfrc_applied', 'xfrc_applied'])
root = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, 'vita_demo_free_root')
root_free = root >= 0 and m.jnt_type[root] == mj.mjtJoint.mjJNT_FREE
objects_free = m.jnt_type[int(m.body_jntadr[obj])] == mj.mjtJoint.mjJNT_FREE and m.body_mocapid[obj] < 0
no_weld = not np.any(m.eq_type == mj.mjtEq.mjEQ_WELD)
replay = max(qerror, verror, torque_error) < 1e-10
hand_actuators = [ai for ai in range(m.nu) if (m.actuator(ai).name or '').startswith(('motor_rh_', 'motor_lh_'))
                  and np.any(m.actuator_gear[ai] != 0)]
passive_actuators = np.flatnonzero(np.all(m.actuator_gear == 0, axis=1))
hand_torque_max = float(np.max(abs(z['actuator_force'][:, hand_actuators])))
passive_torque_max = float(np.max(abs(z['actuator_force'][:, passive_actuators])))
mimic_error = 0.
for e in range(m.neq):
    if m.eq_type[e] != mj.mjtEq.mjEQ_JOINT:
        continue
    child, parent = int(m.eq_obj1id[e]), int(m.eq_obj2id[e])
    if parent < 0 or not (m.joint(child).name or '').startswith(('rh_', 'lh_')):
        continue
    cq, pq = int(m.jnt_qposadr[child]), int(m.jnt_qposadr[parent])
    residual = z['qpos'][:, cq]-m.qpos0[cq]-np.polynomial.polynomial.polyval(
        z['qpos'][:, pq]-m.qpos0[pq], m.eq_data[e, :5])
    mimic_error = max(mimic_error, float(np.max(abs(residual))))
hand_contract = hand_torque_max <= .4+1e-9 and passive_torque_max == 0 and mimic_error <= .03
result = {'scope': 'Independent control replay; exact lowest collider height over source tabletop plane, real hand support and unloaded table.',
          'trial': str(a.trial), 'object_id': a.object_id, 'height_requirement_m': a.clearance_m,
          'trace_sha256': hashlib.sha256((a.trial/'physical_trace.npz').read_bytes()).hexdigest(),
          'model_sha256': hashlib.sha256((a.trial/'effective_model.mjb').read_bytes()).hexdigest(),
          'hold_requirement_s': a.hold_s, 'longest_verified_hold_s': best[0],
          'maximum_linear_speed_m_s': .05, 'maximum_angular_speed_rad_s': .3,
          'hold_start_s': best[1], 'hold_end_s': best[2],
          'original_hand_shapes_unchanged': hands_unchanged, 'zero_external_forces': external_zero,
          'object_dynamic_free': bool(objects_free), 'no_welds': bool(no_weld),
          'robot_root_is_free': bool(root_free),
          'robot_root_free_joint': None if root < 0 else m.joint(root).name,
          'hand_torque_max_Nm': hand_torque_max, 'passive_torque_max_Nm': passive_torque_max,
          'mimic_max_error_rad': mimic_error, 'hand_contract_passed': bool(hand_contract),
          'replay_qpos_error': qerror, 'replay_qvel_error': verror, 'replay_torque_error': torque_error,
          'deterministic_replay_passed': replay,
          'success': bool(best[0] >= a.hold_s-1e-9 and hands_unchanged and external_zero
                          and objects_free and root_free and no_weld and replay and hand_contract)}
a.output.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(a.output.with_suffix('.metrics.npz'), values=np.asarray(rows),
                    columns=np.asarray(['time_s', 'lowest_table_plane_clearance_m', 'hand_up_N', 'table_load_N',
                                        'hand_contacts', 'upright', 'qualified', 'nonhand_load_N',
                                        'linear_speed_m_s', 'angular_speed_rad_s']))
a.output.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result), flush=True)
