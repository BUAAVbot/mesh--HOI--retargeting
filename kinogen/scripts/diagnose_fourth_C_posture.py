"""Read-only decomposition of C's pre-grasp world-palm tracking problem."""
from pathlib import Path
import argparse,json
import numpy as np
import mujoco as mj
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
p=argparse.ArgumentParser();p.add_argument('--trial',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();m=mj.MjModel.from_binary_path(str((a.trial/'effective_model.mjb').resolve()));z=np.load(a.trial/'physical_trace.npz');t=z['time_s'];q=z['qpos'];k=int(np.argmin(abs(t-40)));d=mj.MjData(m);sid=m.site('rh_palm_control_site').id;hb=m.body('rh_hand_base_link').id;waist=[j for j in range(m.njnt)if(m.joint(j).name or'').startswith('waist')];arm=[j for j in range(m.njnt)if(m.joint(j).name or'').startswith(('right_shoulder','right_elbow','right_wrist'))];wa=m.jnt_qposadr[waist];aa=m.jnt_qposadr[arm];table=[g for g in range(m.ngeom)if(m.geom(g).name or'').startswith('table_support')];hands=[g for g in range(m.ngeom)if(m.geom(g).name or'').startswith('collision_rh_')]
def fk(pose):
    d.qpos[:]=pose;d.qvel[:]=0;mj.mj_forward(m,d);return d.site_xpos[sid].copy(),d.xmat[hb].reshape(3,3).copy()
goal,goalR=fk(q[k]);last,lastR=fk(q[-1]);rootfixed=q[-1].copy();rootfixed[:7]=q[k,:7];A,_=fk(rootfixed);waistfixed=rootfixed.copy();waistfixed[wa]=q[k,wa];B,_=fk(waistfixed);armfixed=waistfixed.copy();armfixed[aa]=q[k,aa];C,_=fk(armfixed)
decomposition={'scope':'Ordered exact FK counterfactual decomposition, path-dependent; no causal dynamic claim. Restore root first, then waist, then arm.','root_motion_component_m':(last-A).tolist(),'waist_motion_component_m':(A-B).tolist(),'arm_motion_component_m':(B-C).tolist(),'other_component_m':(C-goal).tolist(),'total_error_vector_m':(last-goal).tolist(),'sum_residual_m':float(np.max(abs((last-A)+(A-B)+(B-C)+(C-goal)-(last-goal))))}
rows=[]
for time in[40.,41.,42.,43.,44.,45.]:
    i=int(np.argmin(abs(t-time)));palm,R=fk(q[i]);rr=Rotation.from_quat(q[i,3:7][[1,2,3,0]]);r0=Rotation.from_quat(q[k,3:7][[1,2,3,0]]);gaps=[]
    for hg in hands:
        gap=min(float(mj.mj_geomDistance(m,d,hg,tg,1.,None))for tg in table);gaps.append({'geom':m.geom(hg).name,'table_signed_gap_m':gap})
    rows.append({'time_s':float(t[i]),'palm_error_xyz_m':(palm-goal).tolist(),'palm_error_norm_m':float(np.linalg.norm(palm-goal)),'palm_orientation_error_rad':float(np.linalg.norm(Rotation.from_matrix(R@goalR.T).as_rotvec())),'root_translation_from40s_xyz_m':(q[i,:3]-q[k,:3]).tolist(),'root_rotation_change_from40s_axisangle_rad':(rr*r0.inv()).as_rotvec().tolist(),'waist_q_rad':q[i,wa].tolist(),'closest_hand_table':sorted(gaps,key=lambda r:r['table_signed_gap_m'])[:3]})
base=q[-1].copy()
def residual(x):
    pose=base.copy();pose[aa]=x;pp,R=fk(pose);return np.r_[pp-goal,.15*Rotation.from_matrix(goalR@R.T).as_rotvec()]
sol=least_squares(residual,np.clip(base[aa],m.jnt_range[arm,0]+1e-6,m.jnt_range[arm,1]-1e-6),bounds=(m.jnt_range[arm,0]+1e-6,m.jnt_range[arm,1]-1e-6),max_nfev=150,ftol=1e-12,xtol=1e-12,gtol=1e-12);pose=base.copy();pose[aa]=sol.x;pp,R=fk(pose);touch=[]
for c in d.contact[:d.ncon]:
    g1,g2=map(int,c.geom);b1,b2=map(int,m.geom_bodyid[[g1,g2]])
    if (g1 in table and(m.body(b2).name or'').startswith(('rh_','right_')))or(g2 in table and(m.body(b1).name or'').startswith(('rh_','right_'))):touch.append({'geoms':[m.geom(g1).name,m.geom(g2).name],'penetration_m':float(max(0,-c.dist))})
out={'schema':'fourth.C_posture_tracking_diagnosis.v1','trial':str(a.trial),'initial_world_palm_goal_m':goal.tolist(),'final_actual_palm_m':last.tolist(),'kinematic_decomposition':decomposition,'samples':rows,'private_pose_only_IK_at_final_body':{'scope':'IK without table penalties to test whether original world hold is geometrically reachable; private FK only.','success':bool(sol.success),'position_error_m':float(np.linalg.norm(pp-goal)),'orientation_error_rad':float(np.linalg.norm(Rotation.from_matrix(R@goalR.T).as_rotvec())),'arm_target_q':sol.x.tolist(),'right_arm_table_contacts':touch}};a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)

