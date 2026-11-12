"""Frozen-state whole-body/arm diagnosis of object-relative tactile alignment."""
from pathlib import Path
import argparse,json
import numpy as np
import mujoco as mj
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
p=argparse.ArgumentParser();p.add_argument('--trial',type=Path,required=True);p.add_argument('--start',type=float,default=59.5);p.add_argument('--output',type=Path,required=True);a=p.parse_args();m=mj.MjModel.from_binary_path(str((a.trial/'effective_model.mjb').resolve()));z=dict(np.load(a.trial/'physical_trace.npz'));t=z['time_s'];q=z['qpos'];k=int(np.argmin(abs(t-a.start)));d=mj.MjData(m);template=json.loads((a.trial/'source_candidate.json').read_text())['best']['template'];site=m.site('rh_palm_control_site').id;hand=m.body('rh_hand_base_link').id;obj=m.body('target_object_01').id;oq=int(m.jnt_qposadr[int(m.body_jntadr[obj])]);arm=[j for j in range(m.njnt)if(m.joint(j).name or'').startswith(('right_shoulder','right_elbow','right_wrist'))];aa=m.jnt_qposadr[arm];waist=[m.joint('waist_yaw_joint').id,m.joint('waist_pitch_joint').id];wa=m.jnt_qposadr[waist];table={g for g in range(m.ngeom)if(m.geom(g).name or'').startswith('table_support')}
def quatR(q):return Rotation.from_quat(np.asarray(q)[[1,2,3,0]]).as_matrix()
def fk(pose):d.qpos[:]=pose;mj.mj_kinematics(m,d);return d.site_xpos[site].copy(),d.xmat[hand].reshape(3,3).copy()
p0,R0=fk(q[k]);samples=[]
for request in[a.start,63.5,65.5,67.5,68.5,float(t[-1])]:
    i=int(np.argmin(abs(t-request)));palm,R=fk(q[i]);OR=quatR(q[i,oq+3:oq+7]);want=q[i,oq:oq+3]+OR@np.asarray(template['palm_offset_in_object_m']);wantR=OR@quatR(template['hand_rotation_in_object_wxyz']);s=np.clip((t[i]-t[k])/4.,0,1);s=s*s*(3-2*s);goal=p0+s*(want-p0);goalR=Rotation.from_rotvec(s*Rotation.from_matrix(wantR@R0.T).as_rotvec()).as_matrix()@R0;rootdelta=(Rotation.from_quat(q[i,3:7][[1,2,3,0]])*Rotation.from_quat(q[k,3:7][[1,2,3,0]]).inv()).as_rotvec();samples.append({'time_s':float(t[i]),'palm_error_world_xyz_m':(palm-goal).tolist(),'palm_error_norm_m':float(np.linalg.norm(palm-goal)),'orientation_error_rad':float(np.linalg.norm(Rotation.from_matrix(goalR@R.T).as_rotvec())),'root_delta_from_tactile_start_xyz_m':(q[i,:3]-q[k,:3]).tolist(),'root_rotation_change_axisangle_rad':rootdelta.tolist(),'waist_actual_yaw_pitch_rad':q[i,wa].tolist(),'arm_limit_margins_rad':{m.joint(j).name:float(min(q[i,m.jnt_qposadr[j]]-m.jnt_range[j,0],m.jnt_range[j,1]-q[i,m.jnt_qposadr[j]]))for j in arm}})
base=q[-1].copy();OR=quatR(base[oq+3:oq+7]);goal=base[oq:oq+3]+OR@np.asarray(template['palm_offset_in_object_m']);goalR=OR@quatR(template['hand_rotation_in_object_wxyz']);private=[]
for weight in[.12,.03]:
    def res(x):
        pose=base.copy();pose[aa]=x;pp,R=fk(pose);return np.r_[pp-goal,weight*Rotation.from_matrix(goalR@R.T).as_rotvec()]
    low=m.jnt_range[arm,0]+1e-6;high=m.jnt_range[arm,1]-1e-6;sol=least_squares(res,np.clip(base[aa],low,high),bounds=(low,high),max_nfev=200,ftol=1e-11,xtol=1e-11,gtol=1e-11);pose=base.copy();pose[aa]=sol.x;pp,R=fk(pose);mj.mj_forward(m,d);coll=[]
    for c in d.contact[:d.ncon]:
        g1,g2=map(int,c.geom);b1,b2=map(int,m.geom_bodyid[[g1,g2]]);n1,n2=m.body(b1).name or'world',m.body(b2).name or'world'
        if(g1 in table and n2.startswith(('rh_','right_')))or(g2 in table and n1.startswith(('rh_','right_'))):coll.append({'geoms':[m.geom(g1).name,m.geom(g2).name],'penetration_m':float(max(0,-c.dist))})
    private.append({'orientation_weight_m':weight,'single_start_solver_success':bool(sol.success),'position_error_m':float(np.linalg.norm(pp-goal)),'orientation_error_rad':float(np.linalg.norm(Rotation.from_matrix(goalR@R.T).as_rotvec())),'right_robot_table_contacts':coll,'joint_targets_rad':sol.x.tolist()})
out={'schema':'fourth.tactile_alignment_body_arm_diagnosis.v1','scope':'Read-only frozen states plus private bounded IK. Single-start IK failure is not proof of unreachability; no physical trial or grasp success inferred.','trial':str(a.trial),'start_time_s':float(t[k]),'samples':samples,'private_final_pose_IK_comparison':private};a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
