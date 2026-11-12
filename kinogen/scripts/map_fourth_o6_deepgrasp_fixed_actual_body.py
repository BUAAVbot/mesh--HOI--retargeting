"""Map a deep-support template to the actual final pose of a recorded preslide.

All object poses stay identical to the trace. Robot IK is private planning only.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import mujoco as mj
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from fourth_arm_object_feedback import ArmObjectFeedback
p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);p.add_argument('--trial',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
src=json.loads(a.candidate.read_text());m=mj.MjModel.from_xml_path(src['model']);d=mj.MjData(m);h=ArmObjectFeedback(m,dict(np.load(a.reference)));trace=np.load(a.trial/'physical_trace.npz');actual=trace['qpos'][-1].copy();target=src['best']['template'];d.qpos[:]=actual
for name,value in target['finger_joint_targets'].items():d.qpos[m.joint(name).qposadr]=value
for e in range(m.neq):
 if m.eq_type[e]==mj.mjtEq.mjEQ_JOINT:
  child,parent=int(m.eq_obj1id[e]),int(m.eq_obj2id[e]);d.qpos[m.jnt_qposadr[child]]=m.eq_data[e,1]*d.qpos[m.jnt_qposadr[parent]]
mj.mj_forward(m,d);obj=h.object_bids[0];OP=d.xpos[obj].copy();OR=d.xmat[obj].reshape(3,3).copy();P=OP+OR@target['palm_offset_in_object_m'];R=OR@Rotation.from_quat(np.asarray(target['hand_rotation_in_object_wxyz'])[[1,2,3,0]]).as_matrix();base=d.qpos.copy()
selected=h.chains[1]['joint_ids'];qa=m.jnt_qposadr[selected];lo=m.jnt_range[selected,0]+1e-5;hi=m.jnt_range[selected,1]-1e-5;v8=np.asarray(src['best']['qpos'])[qa]
def res(q):
 d.qpos[:]=base;d.qpos[qa]=q;mj.mj_kinematics(m,d);mj.mj_comPos(m,d)
 return np.r_[40*(h.palm(d,1)-P),Rotation.from_matrix(R@d.xmat[h.hand_bids[1]].reshape(3,3).T).as_rotvec(),.001*(q-v8)]
fit=least_squares(res,np.clip(v8,lo,hi),bounds=(lo,hi),max_nfev=160,ftol=1e-10,xtol=1e-10,gtol=1e-10);err=res(fit.x);mj.mj_forward(m,d)
for adr in h.object_qadr:assert np.array_equal(d.qpos[adr:adr+7],actual[adr:adr+7])
row={'qpos':d.qpos.tolist(),'residual_norm':float(np.linalg.norm(err)),'position_error_m':float(np.linalg.norm(err[:3])/40),'orientation_error_rad':float(np.linalg.norm(err[3:6])),'robot_table_max_penetration_m':0.,'template':target}
out={'scope':'Private robot IK mapped onto recorded actual preslide terminal object poses. All six object poses are exact trace values; no physical grasp claimed.','model':src['model'],'model_sha256':src['model_sha256'],'actual_trial':str(a.trial),'actual_trace_sha256':hashlib.sha256((a.trial/'physical_trace.npz').read_bytes()).hexdigest(),'actual_terminal_time_s':float(trace['time_s'][-1]),'best':row,'rows':[row]};a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in row.items()if k not in ['qpos','template']}),flush=True)
