"""Build an explicit metadata-default standing reference for controller diagnosis."""
from pathlib import Path
import argparse
import numpy as np
import mujoco as mj
p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--template',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=float,default=5.);a=p.parse_args()
m=mj.MjModel.from_xml_path(str(a.model));d=mj.MjData(m);z=np.load(a.template,allow_pickle=False)
names=list(map(str,z['dof_names']));q=np.zeros(59);q[:3]=[3.,5.,.9];q[3]=1.
for i,n in enumerate(names):
    if 'hip_pitch'in n or 'ankle_pitch'in n:q[7+i]=-.2
    if 'knee'in n:q[7+i]=.4
    if 'hip_roll'in n:q[7+i]=.088 if n.startswith('left')else-.088
    if 'ankle_roll'in n:q[7+i]=-.088 if n.startswith('left')else.088
    if 'shoulder_roll'in n:q[7+i]=.4226 if n.startswith('left')else-.4226
    jid=mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,n)
    d.qpos[m.jnt_qposadr[jid]]=q[7+i]
d.qpos[:7]=q[:7]
if 'object_target_pose'in z:d.qpos[59:]=z['object_target_pose'][0].reshape(-1)
mj.mj_forward(m,d)
footids=[mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,n)for n in ['left_ankle_roll_link','right_ankle_roll_link']]
low=[]
for g in range(m.ngeom):
    if m.geom_bodyid[g]not in footids or not m.geom_contype[g]:continue
    size=m.geom_size[g];rz=d.geom_xmat[g].reshape(3,3)[2];typ=m.geom_type[g]
    if typ==mj.mjtGeom.mjGEOM_BOX:r=abs(rz)@size
    elif typ==mj.mjtGeom.mjGEOM_SPHERE:r=size[0]
    elif typ==mj.mjtGeom.mjGEOM_CAPSULE:r=size[0]+size[1]*abs(rz[2])
    elif typ==mj.mjtGeom.mjGEOM_CYLINDER:r=size[0]*np.linalg.norm(rz[:2])+size[1]*abs(rz[2])
    else:continue
    low.append(d.geom_xpos[g,2]-r)
floor=[g for g in range(m.ngeom)if m.geom_type[g]==mj.mjtGeom.mjGEOM_PLANE and m.geom_bodyid[g]==0]
if not floor or len(low)<4:raise ValueError('Actual floor / primitive soles missing')
q[2]+=d.geom_xpos[floor[0],2]+.001-min(low)
N=int(round(a.seconds/.02))+1
payload={'qpos':np.tile(q,(N,1)),'time_s':np.arange(N)*.02,'dof_names':np.array(names),'synthetic_default_stand_diagnostic':np.array(True),'floor_z':np.asarray(d.geom_xpos[floor[0],2])}
if 'object_target_pose'in z:payload['object_target_pose']=np.tile(z['object_target_pose'][0],(N,1,1))
a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,**payload);print('root',q[:7],'reference',str(a.output))
