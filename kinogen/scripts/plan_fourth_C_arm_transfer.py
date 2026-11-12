"""Private global arm IK and collision-audited configuration-space transfer."""
from pathlib import Path
import argparse,json
import numpy as np
import mujoco as mj
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from fourth_arm_object_feedback import ArmObjectFeedback
R=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);p.add_argument('--trial',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
c=json.loads(a.candidate.read_text());best=c['best'];template=best['template'];m=mj.MjModel.from_binary_path(str(a.trial/'effective_model.mjb'));trace=dict(np.load(a.trial/'physical_trace.npz'));i=int(np.argmin(abs(trace['time_s']-45.)));base=trace['qpos'][i].copy();d=mj.MjData(m);d.qpos[:]=base;mj.mj_forward(m,d)
ref=dict(np.load(R/'outputs/fourth_full_sequence/reference_o6_support_posture_stand_v2.npz'));h=ArmObjectFeedback(m,ref);qa=h.chains[1]['qa'];jids=h.chains[1]['joint_ids'];lower=m.jnt_range[jids,0]+1e-5;upper=m.jnt_range[jids,1]-1e-5;obj=h.object_bids[0];OR=d.xmat[obj].reshape(3,3);out=np.array([.3521665876,.9359373348,-.02508]);out/=np.linalg.norm(out);grasp=d.xpos[obj]+OR@np.asarray(template['palm_offset_in_object_m']);target=grasp+out*.08+np.array([0,0,.10]);targetR=OR@h.rot(template['hand_rotation_in_object_wxyz']);rng=np.random.default_rng(20261027)
armgeoms=[g for g in range(m.ngeom)if m.geom_contype[g]and(m.body(int(m.geom_bodyid[g])).name or'').startswith(('rh_','right_shoulder','right_elbow','right_wrist'))];scene=h.table_geoms+sum(h.object_geoms,[])
def gap(q):
 d.qpos[:]=base;d.qpos[qa]=q;mj.mj_forward(m,d);return min(float(mj.mj_geomDistance(m,d,g,sg,.2,np.zeros(6)))for g in armgeoms for sg in scene)
def pathgap(start,end,n=35):
 return min(gap(start+(end-start)*(s*s*(3-2*s)))for s in np.linspace(0,1,n))
def residual(q):
 d.qpos[:]=base;d.qpos[qa]=q;mj.mj_kinematics(m,d);return np.r_[30*(h.palm(d,1)-target),Rotation.from_matrix(targetR@d.xmat[h.hand_bids[1]].reshape(3,3).T).as_rotvec()]
rows=[];seeds=[np.asarray(best['qpos'])[qa],base[qa],np.clip(np.array([0,-.4226,0,1.0,0,0,0]),lower,upper)]
seeds += [rng.uniform(lower,upper)for _ in range(9)]
for k,seed in enumerate(seeds):
 fit=least_squares(residual,np.clip(seed,lower,upper),bounds=(lower,upper),max_nfev=220,ftol=1e-10,xtol=1e-10,gtol=1e-10);error=residual(fit.x);pos=float(np.linalg.norm(error[:3])/30);rot=float(np.linalg.norm(error[3:]));finalgap=gap(fit.x);pg=pathgap(base[qa],fit.x,25)if pos<.003 and rot<.05 and finalgap>.003 else None
 row={'seed':k,'arm_joint_target':fit.x.tolist(),'position_error_m':pos,'rotation_error_rad':rot,'final_arm_scene_gap_m':finalgap,'linear_configuration_path_minimum_gap_m':pg,'direct_path_pass':bool(pg is not None and pg>=.003),'actual_start_time_s':float(trace['time_s'][i]),'actual_start_arm_joint_angles':base[qa].tolist(),'target_palm_world':target.tolist(),'target_hand_rotation_world':targetR.tolist()};rows.append(row)
 print(json.dumps(row),flush=True)
 a.output.write_text(json.dumps({'scope':'Private no-state-write arm global IK/path planning, not executed transfer. Model original hands/objects.','source_trial':str(a.trial),'source_sample':i,'candidate':str(a.candidate),'arm_joint_names':h.chains[1]['names'],'rows':rows,'best':min(rows,key=lambda r:(not r['direct_path_pass'],r['position_error_m']+.02*r['rotation_error_rad']-min(r['final_arm_scene_gap_m'],.03)))},indent=2)+'\n')
 if row['direct_path_pass']:break
