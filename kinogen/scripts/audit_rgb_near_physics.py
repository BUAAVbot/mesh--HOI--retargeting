"""Independently replay the dual-Vita bench trial, including logged attachment events."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trial', type=Path, required=True)
    a = p.parse_args()
    import numpy as np
    import mujoco
    r = json.loads((a.trial/'report.json').read_text())
    model_path = a.trial/'model.mjb'
    trace_path = a.trial/'trace.npz'
    if sha(model_path) != r['model']['sha256'] or sha(trace_path) != r['trace']['sha256']:
        raise ValueError('Frozen input hash mismatch')
    model = mujoco.MjModel.from_binary_path(str(model_path))
    data = mujoco.MjData(model)
    z = np.load(trace_path, allow_pickle=False)
    n = len(z['ctrl'])
    if z['qpos'].shape != (n+1, model.nq) or z['qvel'].shape != (n+1, model.nv):
        raise ValueError('State/control dimensions disagree')
    if z['ctrl'].shape != (n, model.nu) or not np.all(np.isfinite(z['qpos'])):
        raise ValueError('Invalid control/state array')
    dt_error = float(np.max(np.abs(np.diff(z['time'])-model.opt.timestep)))
    if dt_error > 1e-9:
        raise ValueError('Nonuniform physical timeline')
    for key in ('qfrc_applied', 'xfrc_applied'):
        if key not in z or np.any(z[key]):
            raise ValueError('Missing or nonzero external force record: '+key)
    if model.neq and 'eq_active' not in z:
        raise ValueError('Attachment-capable model requires interval equality activation log')
    if 'eq_active' in z and z['eq_active'].shape != (n, model.neq):
        raise ValueError('Equality activation dimensions disagree')
    if 'eq_data' in z and z['eq_data'].shape != (n, model.neq, model.eq_data.shape[1]):
        raise ValueError('Equality model parameter dimensions disagree')
    spec = int(z['state_spec'])
    mujoco.mj_setState(model, data, z['initial_state'], spec)
    mujoco.mj_forward(model, data)
    qerr = float(np.max(np.abs(data.qpos-z['qpos'][0])))
    verr = float(np.max(np.abs(data.qvel-z['qvel'][0])))
    force_errors = {key:0. for key in ('actuator_force','qfrc_constraint','qfrc_equality') if key in z}
    for i in range(n):
        if 'eq_data' in z:
            model.eq_data[:] = z['eq_data'][i]
        if 'eq_active' in z:
            data.eq_active[:] = z['eq_active'][i]
        data.ctrl[:] = z['ctrl'][i]
        data.qfrc_applied[:] = 0
        data.xfrc_applied[:] = 0
        mujoco.mj_step(model, data)
        qerr = max(qerr, float(np.max(np.abs(data.qpos-z['qpos'][i+1]))))
        verr = max(verr, float(np.max(np.abs(data.qvel-z['qvel'][i+1]))))
        for key in force_errors:
            if key == 'qfrc_equality':
                selected = np.zeros(data.nefc)
                ids = np.flatnonzero(data.efc_type[:data.nefc] == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY))
                selected[ids] = data.efc_force[ids]
                actual = np.zeros(model.nv)
                mujoco.mj_mulJacTVec(model,data,actual,selected)
            else:
                actual = getattr(data,key)
            force_errors[key] = max(force_errors[key],float(np.max(np.abs(actual-z[key][i]))))
    free_joints = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    shoe_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'shoe_free')
    shoe_body = int(model.jnt_bodyid[shoe_jid]) if shoe_jid >= 0 else -1
    roots = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, p+'vita_demo_free_root') for p in ('p1_','p2_')]
    no_object_actuator = shoe_jid >= 0 and not np.any(model.actuator_trnid[:,0] == shoe_jid)
    passed = (qerr < 1e-7 and verr < 1e-6 and len(free_joints) == 3 and min(roots) >= 0
              and shoe_body >= 0 and model.body_mocapid[shoe_body] < 0 and no_object_actuator
              and all(value < 1e-6 for value in force_errors.values()))
    out = {'status':'deterministic_physics_trace_verified' if passed else 'replay_failed',
           'replay_proves_task_success':False, 'model_sha256':sha(model_path), 'trace_sha256':sha(trace_path),
           'steps':n, 'duration_s':float(z['time'][-1]-z['time'][0]), 'dt_max_error_s':dt_error,
           'replayed_qpos_max_abs_error':qerr, 'replayed_qvel_max_abs_error':verr,
           'replayed_generalized_force_max_abs_errors':force_errors,
           'force_trace_verified':set(force_errors)=={'actuator_force','qfrc_constraint','qfrc_equality'} and all(v<1e-6 for v in force_errors.values()),
           'free_roots':len(free_joints), 'robot_free_roots':len(roots), 'shoe_mocap':False,
           'shoe_actuated':not bool(no_object_actuator), 'external_forces_zero':True,
           'attachment_capable_model':bool(model.neq),
           'attachment_activation_logged':'eq_active' in z,
           'attachment_parameters_logged':'eq_data' in z,
           'mujoco_version':mujoco.__version__}
    (a.trial/'independent_replay_audit.json').write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps(out), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
