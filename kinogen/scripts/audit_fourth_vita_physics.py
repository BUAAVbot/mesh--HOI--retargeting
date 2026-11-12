"""Independently replay frozen controls to audit a recorded physics trial."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--trace', type=Path, required=True)
    p.add_argument('--object-body', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    import numpy as np
    import mujoco

    z = np.load(args.trace, allow_pickle=False)
    qpos, qvel, times, controls = z['qpos'], z['qvel'], z['time'], z['ctrl']
    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    data = mujoco.MjData(model)
    if qpos.shape != (len(times), model.nq) or qvel.shape != (len(times), model.nv):
        raise ValueError('State dimensions disagree with frozen model')
    if controls.shape != (len(times)-1, model.nu):
        raise ValueError('Need one control for every recorded physics interval')
    if not all(np.isfinite(x).all() for x in [qpos,qvel,times,controls]):
        raise ValueError('Nonfinite physics state')
    dt_error = float(np.max(np.abs(np.diff(times)-model.opt.timestep)))
    if dt_error > 1e-9:
        raise ValueError('Recorded timeline does not match the model timestep')
    obj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, args.object_body)
    if obj < 0:
        raise ValueError('Target object body missing')
    obj_joints = np.flatnonzero(model.jnt_bodyid == obj)
    dynamic = len(obj_joints) == 1 and model.jnt_type[obj_joints[0]] == mujoco.mjtJoint.mjJNT_FREE
    object_mocap = int(model.body_mocapid[obj]) >= 0
    object_actuated = any(int(jid) in obj_joints for jid in model.actuator_trnid[:,0])
    applied_max = {}
    for key in ['qfrc_applied','xfrc_applied']:
        if key not in z:
            raise ValueError('Missing external-force record: '+key)
        applied_max[key] = float(np.max(np.abs(z[key])))
    if 'initial_state' in z and 'state_spec' in z:
        spec = int(z['state_spec'].reshape(-1)[0])
        mujoco.mj_setState(model, data, z['initial_state'], spec)
        initial_method = 'mjSTATE_INTEGRATION snapshot'
    else:
        data.qpos[:] = qpos[0]
        data.qvel[:] = qvel[0]
        data.time = float(times[0])
        initial_method = 'qpos/qvel/time only'
    mujoco.mj_forward(model, data)
    qerr = float(np.max(np.abs(data.qpos-qpos[0])))
    verr = float(np.max(np.abs(data.qvel-qvel[0])))
    max_force = 0.0
    contact_pairs = {}
    object_qadr = int(model.jnt_qposadr[obj_joints[0]]) if dynamic else None
    def object_position():
        return data.qpos[object_qadr:object_qadr+3].copy() if dynamic else data.xpos[obj].copy()
    object_positions = [object_position()]
    for i, ctrl in enumerate(controls):
        data.ctrl[:] = ctrl
        data.qfrc_applied[:] = 0
        data.xfrc_applied[:] = 0
        mujoco.mj_step(model, data)
        qerr = max(qerr,float(np.max(np.abs(data.qpos-qpos[i+1]))))
        verr = max(verr,float(np.max(np.abs(data.qvel-qvel[i+1]))))
        object_positions.append(object_position())
        active_pairs = set()
        for c in range(data.ncon):
            contact = data.contact[c]
            b1,b2 = int(model.geom_bodyid[contact.geom1]),int(model.geom_bodyid[contact.geom2])
            if obj not in (b1,b2):
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(model,data,c,force)
            maximum = float(np.linalg.norm(force[:3]))
            max_force = max(max_force,maximum)
            pair = tuple(sorted([mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,int(contact.geom1)) or str(contact.geom1),
                                 mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,int(contact.geom2)) or str(contact.geom2)]))
            pair_name = '|'.join(pair)
            active_pairs.add(pair_name)
            stats = contact_pairs.setdefault(pair_name, {'contact_points':0,'active_steps':0,'peak_force_N':0.0,'min_distance_m':0.0})
            stats['contact_points'] += 1
            stats['peak_force_N'] = max(stats['peak_force_N'],maximum)
            stats['min_distance_m'] = min(stats['min_distance_m'],float(contact.dist))
        for pair_name in active_pairs:
            contact_pairs[pair_name]['active_steps'] += 1
    positions = np.asarray(object_positions)
    passed = (qerr < 1e-7 and verr < 1e-6 and dynamic and not object_mocap and
              not object_actuated and all(v == 0 for v in applied_max.values()))
    report = {'status':'deterministic_physics_trace_verified' if passed else 'audit_failed',
              'audit_proves_tracking_success':False,
              'model_sha256':sha(args.model),'trace_sha256':sha(args.trace),
              'mujoco_version':mujoco.__version__,'steps':len(controls),'duration_s':float(times[-1]-times[0]),
              'initial_state_method':initial_method,'dt_max_error_s':dt_error,
              'replayed_qpos_max_abs_error':qerr,'replayed_qvel_max_abs_error':verr,
              'target_object_dynamic_freejoint':bool(dynamic),'target_object_mocap':object_mocap,
              'target_object_actuated':bool(object_actuated),'external_forces_max_abs':applied_max,
              'target_object_displacement_m':float(np.linalg.norm(positions[-1]-positions[0])),
              'target_object_position_range_m':np.ptp(positions,axis=0).tolist(),
              'target_contact_peak_force_N':max_force,'target_contact_pairs':contact_pairs}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(report),flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
