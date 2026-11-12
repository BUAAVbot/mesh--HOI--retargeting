"""Private arm/waist reach and real-table collision probe for lower-side grasp."""
from pathlib import Path
import argparse,json
import numpy as np
import mujoco as mj
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from fourth_arm_object_feedback import ArmObjectFeedback
p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--trial',type=Path,required=True);p.add_argument('--settled-objects',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();m=mj.MjModel.from_xml_path(str(a.model));d=mj.MjData(m);ref=dict(np.load(a.reference));z=dict(np.load(a.trial/'physical_trace.npz'));initial=z['qpos'][0].copy();h=ArmObjectFeedback(m,ref);settled=np.load(a.settled_objects)['final_object_pose_world_xyz_wxyz'];plane=np.array([.010271048830894367,.02292500590934548,.999684425036206,1.0058665495441206]);out=np.array([.35216658756772096,.9359373347617386,0.]);out[2]=-plane[:2]@out[:2]/plane[2];out/=np.linalg.norm(out)
for k,adr in enumerate(h.object_qadr):initial[adr:adr+7]=settled[k]
initial[h.object_qadr[0]:h.object_qadr[0]+3]+=.05*out;names=[m.joint(i).name for i in range(m.njnt)];waist=[j for j,n in enumerate(names)if n and n.startswith('waist')];arm=waist+h.chains[1]['joint_ids'];qa=m.jnt_qposadr[arm];lower=m.jnt_range[arm,0].copy()+1e-5;upper=m.jnt_range[arm,1].copy()-1e-5;master=[names.index('rh_'+n)for n in ['thumb_cmc_yaw','thumb_cmc_pitch','index_mcp_pitch','middle_mcp_pitch','ring_mcp_pitch','pinky_mcp_pitch']]
for k,j in enumerate(waist):
    cap=1.8 if'yaw'in names[j]else .8;lower[k]=max(lower[k],-cap);upper[k]=min(upper[k],cap)
initial[m.jnt_qposadr[master]]=[1.2,.18,.8,.8,1.5,1.5]
def mimic():
    for e in range(m.neq):
        if m.eq_type[e]==mj.mjtEq.mjEQ_JOINT:
            child,parent=int(m.eq_obj1id[e]),int(m.eq_obj2id[e]);d.qpos[m.jnt_qposadr[child]]=m.eq_data[e,1]*d.qpos[m.jnt_qposadr[parent]]
d.qpos[:]=initial;mimic();mj.mj_kinematics(m,d);pobj=d.xpos[h.object_bids[0]].copy();Robj=d.xmat[h.object_bids[0]].reshape(3,3);rl=Robj.T@out;rl[2]=0;rl/=np.linalg.norm(rl);rw=Robj@rl;zw=Robj[:,2];tw=np.cross(zw,rw);theta=np.deg2rad(20);desiredR=np.column_stack([-rw*np.sin(theta)+zw*np.cos(theta),tw,-rw*np.cos(theta)-zw*np.sin(theta)]);goal=pobj+rw*.15;goal[2]=-(plane[:2]@goal[:2]+plane[3])/plane[2]-.048;rows=[]
for xy in [(-.25,2.),(-.30,2.05),(-.35,2.10),(-.30,2.15),(-.35,2.15),(-.40,2.15),(-.35,2.20)]:
    seed=initial.copy();seed[:2]=xy
    def residual(q):
        d.qpos[:]=seed;d.qpos[qa]=q;mimic();mj.mj_kinematics(m,d);R=d.xmat[h.hand_bids[1]].reshape(3,3);return np.r_[20*(h.palm(d,1)-goal),.8*Rotation.from_matrix(desiredR@R.T).as_rotvec(),.005*q[:len(waist)]]
    starts=[np.clip(seed[qa],lower,upper)]
    for wy in [-1.4,-.7,.7]:
        guess=starts[0].copy();guess[0]=np.clip(wy,lower[0],upper[0]);guess[1]=np.clip(.4,lower[1],upper[1]);starts.append(guess)
    fits=[least_squares(residual,start,bounds=(lower,upper),max_nfev=170,ftol=1e-8,xtol=1e-8,gtol=1e-8)for start in starts];fit=min(fits,key=lambda v:np.linalg.norm(v.fun));residual(fit.x);d.qvel[:]=0;mj.mj_forward(m,d);collisions=[]
    for c in d.contact[:d.ncon]:
        g1,g2=map(int,c.geom)
        if not({g1,g2}&set(h.table_geoms)):continue
        g=g2 if g1 in h.table_geoms else g1;bname=m.body(int(m.geom_bodyid[g])).name or''
        if bname.startswith('target_object'):continue
        if c.dist<0:collisions.append({'body':bname,'geom':m.geom(g).name,'penetration_m':float(-c.dist)})
    row={'root_xy':list(xy),'palm_goal_world':goal.tolist(),'actual_palm_world':h.palm(d,1).tolist(),'position_error_m':float(np.linalg.norm(fit.fun[:3])/20),'orientation_error_rad':float(np.linalg.norm(fit.fun[3:6])/.8),'waist':{names[j]:float(d.qpos[m.jnt_qposadr[j]])for j in waist},'robot_table_collisions':collisions,'robot_table_max_penetration_m':max((c['penetration_m']for c in collisions),default=0),'robot_COM_world':d.subtree_com[m.body('base_link').id].tolist(),'qpos':d.qpos.tolist()};rows.append(row);print(json.dumps({k:v for k,v in row.items()if k not in('qpos','robot_table_collisions')}),flush=True)
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({'scope':'Private FK/workspace only, hypothetical 5cm slide; not a physical initialization or grasp success','rows':rows},indent=2)+'\n')
