"""Read-only reuse of original SONIC plant: default-stand and new-reference tests."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import mujoco as mj
from fourth_sonic_controller import _ort,FROZEN
_ort()
sys.path.insert(0,'/root/vita02a-mjlab/vita_to_sonic_20261012_a1/src')
from vita_to.sonic_backend import SonicBackend

p=argparse.ArgumentParser();p.add_argument('--reference',type=Path);p.add_argument('--seconds',type=float,default=5.);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
a.output.mkdir(parents=True)
old=Path('/root/vita02a-mjlab/vita_to_sonic_20261012_a1/inputs/original_reference.npz')
b=SonicBackend(FROZEN/'source',old,FROZEN/'tracking.onnx',inference_threads=1)
m=b.model;d=mj.MjData(m);c=b.contract
times=np.arange(int(a.seconds/.02)+1)*.02
if a.reference:
    z=np.load(a.reference,allow_pickle=False);rt=z['time_s']-z['time_s'][0];names=list(map(str,z['dof_names']));q=np.column_stack([np.interp(times,rt,z['qpos'][:,7+names.index(n)])for n in b.joint_names]);root=np.column_stack([np.interp(times,rt,z['qpos'][:,i])for i in range(7)]);root[:,3:]/=np.linalg.norm(root[:,3:],axis=1)[:,None]
else:
    q=np.tile(c.default_pos,(len(times),1));root=np.tile([0.,0.,1.5,1.,0.,0.,0.],(len(times),1))
d.qpos[:7]=root[0];d.qpos[b.qpos_ids]=q[0];mj.mj_forward(m,d)
floor=mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,'floor')
feet=[i for i in range(m.ngeom)if(mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,i)or'').startswith('collision_foot_')and m.geom_contype[i]]
caps=[g for g in feet if m.geom_type[g]==mj.mjtGeom.mjGEOM_CAPSULE]
if len(caps)!=14:raise ValueError('Expected original 14 shoe capsules')
dist=min(float(d.geom_xpos[g,2]-m.geom_size[g,0]-m.geom_size[g,1]*abs(d.geom_xmat[g].reshape(3,3)[2,2]))for g in caps)
root[:,2]+=.001-dist
positions=[];quats=[]
for r,j in zip(root,q):
    d.qpos[:7]=r;d.qpos[b.qpos_ids]=j;mj.mj_forward(m,d);positions.append(d.xpos[b.body_ids].copy());quats.append(d.xquat[b.body_ids].copy())
positions=np.array(positions);quats=np.array(quats)
fields={'fps':np.asarray(50.),'joint_names':np.array(b.joint_names),'body_names':np.array(b.body_names),'joint_pos':q,'joint_vel':np.gradient(q,.02,axis=0),'body_pos_w':positions,'body_quat_w':quats,'body_lin_vel_w':np.gradient(positions,.02,axis=0),'body_ang_vel_w':np.zeros((len(times),len(b.body_names),3))}
b.close()
np.savez_compressed(a.output/'reference28.npz',**fields)
b=SonicBackend(FROZEN/'source',fields,FROZEN/'tracking.onnx',inference_threads=1)
r=b.rollout(record=True,horizon_steps=len(times)-1)
def safe(v):
    if isinstance(v,dict):return {k:safe(x)for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [safe(x)for x in v]
    if isinstance(v,np.ndarray):return safe(v.tolist())
    if isinstance(v,np.generic):return v.item()
    return v
report=safe({'source_reference':str(a.reference)if a.reference else 'metadata_default_standing','seconds_requested':a.seconds,'metrics':r.metrics,'diagnostics':r.diagnostics,'plant':'original 28DOF hardware SONIC with mesh+7capsule shoes','mujoco_version':mj.__version__})
(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');np.savez_compressed(a.output/'actual_trajectory.npz',**r.recording);print(json.dumps({k:report['metrics'].get(k)for k in ['first_task_termination','first_task_termination_s','elapsed_s','final_base_height_m','max_base_height_error_m','max_gravity_z_error','max_depth_m','accepted']}),flush=True)
