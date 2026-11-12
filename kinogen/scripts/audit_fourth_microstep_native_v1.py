"""Read-only model contract and native one-millisecond control replay."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import mujoco as mj

p = argparse.ArgumentParser()
p.add_argument('--trial', type=Path, required=True)
p.add_argument('--trace', default='physical_trace.npz')
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
mp = a.trial/'effective_model.mjb'
model = mj.MjModel.from_binary_path(str(mp))
baseline = mj.MjModel.from_binary_path(str(root/'outputs/fourth_full_sequence/assets/vita_six_objects_o6_stiff_eq_v1.mjb'))
fields = ['mesh_vert','mesh_face','geom_type','geom_size','geom_pos','geom_quat',
          'geom_contype','geom_conaffinity','geom_friction','geom_solimp','geom_solref',
          'body_mass','body_inertia','body_ipos','body_iquat','body_pos','body_quat',
          'jnt_type','jnt_range','jnt_axis','dof_damping','dof_armature',
          'actuator_gainprm','actuator_biasprm','actuator_gear','actuator_forcerange',
          'actuator_forcelimited','eq_data','eq_solimp']
preserved = {name: bool(np.array_equal(getattr(model,name),getattr(baseline,name))) for name in fields}
z = dict(np.load(a.trial/a.trace, allow_pickle=False))
q, v, t, u = (z[k] for k in ['qpos','qvel','time_s','ctrl'])
assert q.shape == (len(u)+1,model.nq) and v.shape == (len(u)+1,model.nv)
assert len(t) == len(u)+1 and np.isfinite(q).all() and np.isfinite(u).all()
data = mj.MjData(model)
mj.mj_setState(model,data,z['initial_state'],int(z['state_spec']))
mj.mj_forward(model,data)
qe = float(np.max(abs(data.qpos-q[0])))
ve = float(np.max(abs(data.qvel-v[0])))
fe = 0.
for k, ctrl in enumerate(u):
    data.ctrl[:] = ctrl
    # Replaying only actual motor input. No state write or object correction.
    mj.mj_step(model,data)
    qe = max(qe,float(np.max(abs(data.qpos-q[k+1]))))
    ve = max(ve,float(np.max(abs(data.qvel-v[k+1]))))
    fe = max(fe,float(np.max(abs(data.actuator_force-z['actuator_force'][k]))))
external = {key:float(np.max(abs(z[key]))) for key in ['qfrc_applied','xfrc_applied']}
eq = np.flatnonzero(model.eq_type == mj.mjtEq.mjEQ_JOINT)
objects = []
for oid in range(1,7):
    body = model.body(f'target_object_{oid:02d}').id
    jid = int(model.body_jntadr[body])
    objects.append({'id':oid,'free_joint':bool(model.jnt_type[jid] == mj.mjtJoint.mjJNT_FREE),
                    'mocap':bool(model.body_mocapid[body]>=0),
                    'actuated':bool(np.any(model.actuator_trnid[:,0]==jid))})
packets = len(u)//5
held = bool(np.array_equal(u[:packets*5],np.repeat(u[:packets*5:5],5,axis=0)))
good = all(preserved.values()) and qe < 1e-8 and ve < 1e-7 and fe < 1e-6 and not any(external.values())
report = {'schema':'fourth.microstep_native_replay.v1','trial':str(a.trial),'trace':a.trace,
          'model_sha256':hashlib.sha256(mp.read_bytes()).hexdigest(),
          'mujoco_version':mj.__version__,'integration_dt_s':float(model.opt.timestep),
          'control_packet_dt_s':.005,'hand_feedback_dt_s':.02,
          'body_controller':'pd_groundaware','learned_policy_used':False,
          'explicit_body_control_delay_s':0.,
          'unused_source_sonic_queue_delay_s':.015,
          'clock_clarification':'SONIC was constructed for configuration only. Its 15ms queue and 50Hz actor were not used in this explicitly PD benchmark. Actual held control packets are5ms, physical integration1ms, hand feedback20ms.',
          'integration_steps':len(u),'repeated_held_control_exact':held,
          'timeline_step_max_error_s':float(np.max(abs(np.diff(t)-model.opt.timestep))),
          'native_control_only_replay_max_qpos_error':qe,
          'native_control_only_replay_max_qvel_error':ve,
          'native_control_only_replay_max_actuator_force_error':fe,
          'external_force_max_abs':external,'exact_preservation_vs_stiff_model':preserved,
          'eq_joint_solref':model.eq_solref[eq].tolist(),
          'robot_free_base':bool(model.jnt_type[0] == mj.mjtJoint.mjJNT_FREE),
          'objects':objects,'pass':good,
          'scope':'Physics provenance audit only. Does not assert task success; explicit reset benchmark displaces plate outward50mm and robot back60mm before time zero.'}
a.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['trace','integration_steps','native_control_only_replay_max_qpos_error','native_control_only_replay_max_qvel_error','native_control_only_replay_max_actuator_force_error','repeated_held_control_exact','pass']}),flush=True)
