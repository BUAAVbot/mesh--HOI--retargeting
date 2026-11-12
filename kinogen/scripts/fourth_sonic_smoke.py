"""Short real-physics check of the isolated SONIC 28->52 controller transfer."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import mujoco
from fourth_sonic_controller import Controller,freeze_effective_model

p=argparse.ArgumentParser()
p.add_argument('--model',type=Path,required=True)
p.add_argument('--reference',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--mode',choices=['tracking','velocity','newft15225'],default='tracking')
p.add_argument('--actuation',choices=['position','torque'],default='position')
p.add_argument('--arm-reference-override',action='store_true')
p.add_argument('--seconds',type=float,default=2.)
p.add_argument('--device',default='cuda:1')
a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
a.output.mkdir(parents=True)
model=mujoco.MjModel.from_xml_path(str(a.model.resolve()))
z=dict(np.load(a.reference,allow_pickle=False))
ctrl=Controller(model,z,{'mode':a.mode,'actuation':a.actuation,'arm_reference_override':a.arm_reference_override,'device':a.device})
freeze=freeze_effective_model(model,a.output/'model')
frozen=Path(freeze['binary'])
d=mujoco.MjData(model)
d.qpos[:z['qpos'].shape[1]]=z['qpos'][0]
if 'object_target_pose' in z:
    d.qpos[z['qpos'].shape[1]:]=np.asarray(z['object_target_pose'][0]).reshape(-1)
mujoco.mj_forward(model,d)
ctrl.reset(d)
spec=int(mujoco.mjtState.mjSTATE_INTEGRATION)
initial=np.empty(mujoco.mj_stateSize(model,spec));mujoco.mj_getState(model,d,initial,spec)
q=[d.qpos.copy()];v=[d.qvel.copy()];times=[float(d.time)];controls=[];tau=[];maxdepth=0.;end='duration'
warnings=np.array([w.number for w in d.warning])
for step in range(int(np.ceil(a.seconds/model.opt.timestep))):
    u=ctrl.action(d,d.time);d.ctrl[:]=u
    if np.any(d.qfrc_applied)or np.any(d.xfrc_applied):raise ValueError('External force was injected')
    controls.append(u.copy());previous=d.time;mujoco.mj_step(model,d)
    q.append(d.qpos.copy());v.append(d.qvel.copy());times.append(float(d.time));tau.append(d.actuator_force.copy())
    maxdepth=max(maxdepth,max((max(-c.dist,0.)for c in d.contact),default=0.))
    if not np.isfinite(d.qpos).all() or abs(d.time-previous-model.opt.timestep)>1e-9 or np.any(np.array([w.number for w in d.warning])>warnings):end='solver_failure';break
    if d.qpos[2]<q[0][2]-.25:end='fall';break
q=np.array(q);v=np.array(v);u=np.array(controls)
passed=end=='duration'and np.max(np.linalg.norm(q[:,:2]-q[0,:2],axis=1))<.25
report={**ctrl.report(),'status':'short_physics_survived'if passed else 'short_physics_failed',
    'strict_full_sequence_verified':False,'end_reason':end,'duration_s':times[-1],
    'root_xy_drift_m':float(np.max(np.linalg.norm(q[:,:2]-q[0,:2],axis=1))),
    'root_height_drop_m':float(q[0,2]-np.min(q[:,2])),'max_contact_penetration_m':float(maxdepth),
    'joint_rms_rad':float(np.sqrt(np.mean((q[:,ctrl.qa]-ctrl.interpolate(ctrl.body_q,z['time_s'][0]))**2))),
    'model':str(frozen),'reference':str(a.reference.resolve()),'trace':str(a.output/'trace.npz'),
    'model_sha256':hashlib.sha256(frozen.read_bytes()).hexdigest(),'freeze':freeze,'mujoco_version':mujoco.__version__,
    'physics_steps':len(u),'active_qpos_writes':0,'external_forces':False}
np.savez_compressed(a.output/'trace.npz',qpos=q,qvel=v,time=np.array(times),ctrl=u,actuator_force=np.array(tau),
    initial_state=initial,state_spec=np.asarray(spec),qfrc_applied=np.zeros((len(u),model.nv)),xfrc_applied=np.zeros((len(u),model.nbody,6)))
(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
