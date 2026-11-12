"""Full-timeline tabletop adaptation with fixed standing feet and upper-body IK.

Preserves all 923 source timestamps, six object references and observation masks.
The source walking/root motion is explicitly replaced by a stationary standing
base; observed contact-phase hands remain Cartesian IK targets. This is a
separate task adaptation branch, never a claim of original walking fidelity.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import mujoco as mj
from scipy.optimize import least_squares
from scipy.ndimage import binary_dilation,gaussian_filter1d
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--root-x',type=float,default=.08);p.add_argument('--root-y',type=float,default=1.8);p.add_argument('--yaw',type=float,default=-np.pi);p.add_argument('--max-nfev',type=int,default=32);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
z=dict(np.load(a.source,allow_pickle=False));m=mj.MjModel.from_xml_path(str(a.model.resolve()));d=mj.MjData(m)
names=list(map(str,z['dof_names']));times=z['time_s'];N=len(times);jids=np.array([mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,n)for n in names]);qa=m.jnt_qposadr[jids]
q0=np.zeros(59);q0[:3]=[a.root_x,a.root_y,0.];q0[3:7]=[np.cos(a.yaw/2),0,0,np.sin(a.yaw/2)]
for i,n in enumerate(names):
    if 'hip_pitch'in n or 'ankle_pitch'in n:q0[7+i]=-.2
    if 'knee'in n:q0[7+i]=.4
    if 'hip_roll'in n:q0[7+i]=.088 if n.startswith('left')else-.088
    if 'ankle_roll'in n:q0[7+i]=-.088 if n.startswith('left')else.088
    if 'shoulder_roll'in n:q0[7+i]=.4226 if n.startswith('left')else-.4226
def fk(q,pose=None):
    d.qpos[:7]=q[:7];d.qpos[qa]=q[7:]
    if pose is not None:d.qpos[59:]=pose.reshape(-1)
    mj.mj_kinematics(m,d);mj.mj_comPos(m,d)
feet=[mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,n)for n in ['left_ankle_roll_link','right_ankle_roll_link']]
hands=[mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,n)for n in ['lh_hand_base_link','rh_hand_base_link']]
base=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'base_link');floor=mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,'assumed_floor');table=mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,'table_support')
if min(feet+hands+[base,floor,table])<0:raise ValueError('Named standing task bodies/geometries absent')
footgeoms=[g for g in range(m.ngeom)if m.geom_bodyid[g]in feet and m.geom_contype[g]]
def geom_extent(g):
    size=m.geom_size[g];r=d.geom_xmat[g].reshape(3,3);typ=m.geom_type[g]
    if typ==mj.mjtGeom.mjGEOM_BOX:return abs(r)@size
    if typ==mj.mjtGeom.mjGEOM_SPHERE:return np.ones(3)*size[0]
    if typ==mj.mjtGeom.mjGEOM_CAPSULE:return np.ones(3)*size[0]+abs(r[:,2])*size[1]
    if typ==mj.mjtGeom.mjGEOM_CYLINDER:return size[0]*np.sqrt(np.maximum(0,1-r[:,2]**2))+abs(r[:,2])*size[1]
    return None
fk(q0,z['object_target_pose'][0]);low=[d.geom_xpos[g,2]-geom_extent(g)[2]for g in footgeoms if geom_extent(g)is not None];q0[2]+=d.geom_xpos[floor,2]+.001-min(low);fk(q0)
points=[]
for g in footgeoms:
    ext=geom_extent(g)
    if ext is None or d.geom_xpos[g,2]-ext[2]>d.geom_xpos[floor,2]+.02:continue
    x,y=d.geom_xpos[g,:2]
    points.extend([[x+sx*ext[0],y+sy*ext[1]]for sx in [-1,1]for sy in [-1,1]])
hull=ConvexHull(np.asarray(points));A,B=hull.equations[:,:2],hull.equations[:,2]
rest_hands=d.xpos[hands].copy();rest_rot=d.xmat[hands].reshape(2,3,3).copy();footpos=d.xpos[feet].copy()
active=z['contact_candidate_observed'].any(axis=2)
weight=np.empty((N,2))
for s in range(2):weight[:,s]=np.maximum(active[:,s],gaussian_filter1d(binary_dilation(active[:,s],iterations=12).astype(float),5))
target=z['robot_hand_target_world']*weight[:,:,None]+rest_hands[None]*(1-weight[:,:,None])
rawori=z['robot_hand_orientation_target_world_wxyz']
goalrot=np.empty((N,2,3,3))
for i in range(N):
    for s in range(2):
        src=Rotation.from_quat(rawori[i,s,[1,2,3,0]]).as_matrix();rv=Rotation.from_matrix(rest_rot[s].T@src).as_rotvec();goalrot[i,s]=rest_rot[s]@Rotation.from_rotvec(rv*weight[i,s]).as_matrix()
selected=[i for i,n in enumerate(names)if n.startswith(('left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist','waist_'))]
selqa=7+np.asarray(selected);selj=jids[selected];lo=m.jnt_range[selj,0].copy();hi=m.jnt_range[selj,1].copy()
for j,i in enumerate(selected):
    if names[i]=='waist_pitch_joint':lo[j],hi[j]=max(lo[j],-.12),min(hi[j],.40)
    if names[i]=='waist_yaw_joint':lo[j],hi[j]=max(lo[j],-.65),min(hi[j],.65)
handgeoms=[g for g in range(m.ngeom)if m.geom_contype[g] and (mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[g]))or'').startswith(('lh_','rh_','left_wrist','right_wrist'))]
out=[];errors=[];com_margin=[];handfk=[];prev=q0[selqa].copy();start=prev.copy();maxrate=m.jnt_range[selj,1]*0+6.
for f in range(N):
    seed=q0.copy()
    for i,n in enumerate(names):
        if n.startswith('lh_'):seed[7+i]=z['qpos'][f,7+i]*weight[f,0]
        if n.startswith('rh_'):seed[7+i]=z['qpos'][f,7+i]*weight[f,1]
    lower,upper=lo.copy(),hi.copy()
    if f:
        lower=np.maximum(lower,prev-maxrate*(times[f]-times[f-1]));upper=np.minimum(upper,prev+maxrate*(times[f]-times[f-1]))
    def residual(v):
        q=seed.copy();q[selqa]=v;fk(q,z['object_target_pose'][f]);pos=(d.xpos[hands]-target[f]).reshape(-1)
        ori=np.concatenate([Rotation.from_matrix(goalrot[f,s].T@d.xmat[hands[s]].reshape(3,3)).as_rotvec() for s in range(2)])
        violation=np.maximum(A@d.subtree_com[base,:2]+B+.018,0)
        penetration=np.minimum(np.array([mj.mj_geomDistance(m,d,g,table,.5,np.zeros(6))for g in handgeoms])-.001,0)
        return np.r_[15*pos,.12*ori,.05*(v-prev),.025*(v-start),25*violation,12*penetration]
    fit=least_squares(residual,np.clip(prev,lower+1e-8,upper-1e-8),bounds=(lower,upper),max_nfev=a.max_nfev,diff_step=1e-4,ftol=2e-4,xtol=2e-4)
    result=seed.copy();result[selqa]=fit.x;prev=fit.x.copy();fk(result,z['object_target_pose'][f]);out.append(result);errors.append(np.linalg.norm(d.xpos[hands]-target[f],axis=1));com_margin.append(-np.max(A@d.subtree_com[base,:2]+B));handfk.append(d.xpos[hands].copy())
    if f%100==0:print('frame',f,'handerr',errors[-1],flush=True)
payload=dict(z);payload.update(qpos=np.asarray(out),walking_branch_reference_qpos=z['qpos'],walking_branch_robot_hand_target_world=z['robot_hand_target_world'],robot_hand_target_world=target,robot_hand_fk_world=np.asarray(handfk),robot_foot_target_world=np.tile(footpos,(N,1,1)),robot_foot_fk_world=np.tile(footpos,(N,1,1)),stationary_base_adaptation=np.asarray(True),stationary_base_pose_world_xyz_wxyz=q0[:7],stationary_hand_blend_weights=weight,stationary_source_hand_contact_target_retained=active,stationary_hand_ik_error_m=np.asarray(errors),stationary_com_margin_m=np.asarray(com_margin),source_full_sequence_time_objects_unchanged=np.asarray(True),adaptation_description=np.asarray('All923 timestamps/six object trajectories/masks preserved; source walking root+legs replaced by metadata-default planted standing base; observed contact hands use original world goals via bounded upperbody IK, inactive hand phases blend to safe rest; COM support constraint explicit.'))
plane=mj.mj_name2id(m,mj.mjtObj.mjOBJ_NUMERIC,'source_table_plane_world_abcd')
if plane>=0:
    adr=int(m.numeric_adr[plane]);size=int(m.numeric_size[plane])
    if size==4:payload['table_plane_world_abcd']=m.numeric_data[adr:adr+4].copy()
a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,**payload)
err=np.asarray(errors);activeerr=err[active]
report={'status':'stationary_full_timeline_reference_generated_not_physics_verified','frames':N,'duration_s':float(times[-1]-times[0]),'source':str(a.source),'source_sha256':hashlib.sha256(a.source.read_bytes()).hexdigest(),'model':str(a.model),'model_sha256':hashlib.sha256(a.model.read_bytes()).hexdigest(),'output':str(a.output),'root_pose':q0[:7].tolist(),'object_references_unchanged':bool(np.array_equal(payload['object_target_pose'],z['object_target_pose'])),'source_timing_unchanged':True,'contact_hand_ik_error_median_m':float(np.median(activeerr)),'contact_hand_ik_error_max_m':float(np.max(activeerr)),'static_com_margin_m_min':float(min(com_margin)),'root_legs_adapted':True,'original_walking_reproduced':False}
a.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
