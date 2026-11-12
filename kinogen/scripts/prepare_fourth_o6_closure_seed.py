"""Orient the measured O6 thumb-to-four closure axis before solving arm IK."""
from pathlib import Path
import argparse,json
import numpy as np
import mujoco as mj
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from fourth_arm_object_feedback import ArmObjectFeedback
p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--seed',type=Path,required=True);p.add_argument('--settled',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--radial-mode',action='store_true');a=p.parse_args()
m=mj.MjModel.from_xml_path(str(a.model));d=mj.MjData(m);ref=dict(np.load(a.reference));h=ArmObjectFeedback(m,ref)
q0=np.asarray(json.loads(a.seed.read_text())['best']['qpos']);d.qpos[:]=q0
for i,qa in enumerate(h.object_qadr):d.qpos[qa:qa+7]=np.load(a.settled)['final_object_pose_world_xyz_wxyz'][i]
for name,val in [('thumb_cmc_yaw',1.24),('thumb_cmc_pitch',.31),('index_mcp_pitch',.775),('middle_mcp_pitch',.775),('ring_mcp_pitch',.775),('pinky_mcp_pitch',.775)]:d.qpos[m.joint('rh_'+name).qposadr]=val
for e in range(m.neq):
 if m.eq_type[e]==mj.mjtEq.mjEQ_JOINT:
  child,parent=int(m.eq_obj1id[e]),int(m.eq_obj2id[e]);d.qpos[m.jnt_qposadr[child]]=m.eq_data[e,1]*d.qpos[m.jnt_qposadr[parent]]
mj.mj_forward(m,d);origin=d.xpos[h.object_bids[0]].copy();OR=d.xmat[h.object_bids[0]].reshape(3,3).copy();up=OR[:,2]
out=np.array([.35216658756772096,.9359373347617386,0]);out-=up*(out@up);out/=np.linalg.norm(out)
localaxis=np.array([-.44765,-.17671,.87657]);localaxis/=np.linalg.norm(localaxis);localtan=np.array([0.,1.,0.]);localtan-=localaxis*(localaxis@localtan);localtan/=np.linalg.norm(localtan)
L=np.column_stack([localaxis,localtan,np.cross(localaxis,localtan)])
selected=[j for j in range(m.njnt)if(m.joint(j).name or'').startswith('waist')]+h.chains[1]['joint_ids'];qa=m.jnt_qposadr[selected];lo=m.jnt_range[selected,0]+1e-5;hi=m.jnt_range[selected,1]-1e-5
site=m.site('rh_palm_control').id if mj.mj_name2id(m,mj.mjtObj.mjOBJ_SITE,'rh_palm_control')>=0 else None
localpalm=OR[:,0]*0
# Existing palm control site is transformed relative to the actual hand body.
HB=h.hand_bids[1];localpalm=d.xmat[HB].reshape(3,3).T@(h.palm(d,1)-d.xpos[HB])
base=d.qpos.copy();rows=[]
for az in [0.,.12,-.12,.25,-.25]:
 rw=Rotation.from_rotvec(up*az).apply(out);tw=np.cross(up,rw);wa=-rw if a.radial_mode else-up;wt=tw
 W=np.column_stack([wa,wt,np.cross(wa,wt)]);R=W@L.T
 midpoint=origin+rw*.098+up*.022
 if a.radial_mode:midpoint=origin+rw*.097+up*.022
 P=midpoint-R@np.array([.062,0.,.111])+R@localpalm
 def res(q):
  d.qpos[:]=base;d.qpos[qa]=q;mj.mj_kinematics(m,d);mj.mj_comPos(m,d)
  return np.r_[30*(h.palm(d,1)-P),Rotation.from_matrix(R@d.xmat[HB].reshape(3,3).T).as_rotvec()]
 fit=least_squares(res,base[qa],bounds=(lo,hi),max_nfev=180,ftol=1e-10,xtol=1e-10,gtol=1e-10);err=res(fit.x);mj.mj_forward(m,d)
 contacts=[]
 for c in d.contact:
  gg=list(map(int,c.geom));bb=[int(m.geom_bodyid[g])for g in gg]
  if any((m.body(b).name or'').startswith('rh_')for b in bb)and any((m.body(b).name or'').startswith('target_object_')for b in bb):contacts.append({'geom_names':[m.geom(g).name for g in gg],'distance_m':float(c.dist)})
 rows.append({'azimuth_offset_rad':az,'qpos':d.qpos.tolist(),'residual_norm':float(np.linalg.norm(err)),'palm_error_m':float(np.linalg.norm(err[:3])/30),'contacts':contacts,'target_rotation':R.tolist(),'target_palm':P.tolist()})
 print(json.dumps({k:v for k,v in rows[-1].items()if k!='qpos'}),flush=True)
a.output.write_text(json.dumps({'scope':'private closure-axis aligned IK; not a physical grasp','rows':rows,'best':min(rows,key=lambda row:row['residual_norm'])},indent=2)+'\n')
