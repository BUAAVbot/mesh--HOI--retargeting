"""Fixed-wrist geometric closure sweep and full wrench audit; no physics state edited."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import mujoco as mj
from scipy.spatial.transform import Rotation
from fourth_arm_object_feedback import ArmObjectFeedback
from fourth_contact_wrench import contact_columns,wrench_lp
from fourth_convex_separation import ConvexSeparationOracle
p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--seed',type=Path,required=True);p.add_argument('--signature',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
m=mj.MjModel.from_xml_path(str(a.model));d=mj.MjData(m);h=ArmObjectFeedback(m,dict(np.load(a.reference)));oracle=ConvexSeparationOracle(m)
seed=json.loads(a.seed.read_text())['rows'][0];q0=np.asarray(seed['qpos']);sig=json.loads(a.signature.read_text())['thumb_pitch_free_common_gaps'];cmds={round(row['desired_directional_gap_m'],3):np.asarray(row['best']['six_command_rad'])for row in sig}
names=['rh_'+n for n in ['thumb_cmc_yaw','thumb_cmc_pitch','index_mcp_pitch','middle_mcp_pitch','ring_mcp_pitch','pinky_mcp_pitch']];master=[m.joint(n).id for n in names];qa=m.jnt_qposadr[master];finger=[j for j in range(m.njnt)if(m.joint(j).name or'').startswith('rh_')];P=np.zeros((m.nv,6));couplings=[]
for k,j in enumerate(master):P[m.jnt_dofadr[j],k]=1
for e in range(m.neq):
 if m.eq_type[e]==mj.mjtEq.mjEQ_JOINT:
  child,parent=int(m.eq_obj1id[e]),int(m.eq_obj2id[e]);ratio=float(m.eq_data[e,1]);couplings.append((child,parent,ratio))
  if parent in master:P[m.jnt_dofadr[child],master.index(parent)]=ratio
obj=h.object_bids[0];rows=[];margin=m.geom_margin.copy()
for fraction in [0.,.25,.5,.75,1.,1.125,1.25,1.5,1.75,2.]:
 command=cmds[.016]+fraction*(cmds[.012]-cmds[.016]);d.qpos[:]=q0;d.qpos[qa]=command
 for child,parent,ratio in couplings:d.qpos[m.jnt_qposadr[child]]=ratio*d.qpos[m.jnt_qposadr[parent]]
 d.qvel[:]=0;mj.mj_forward(m,d);penetrations=[]
 for c in d.contact:
  gg=list(map(int,c.geom));bb=[int(m.geom_bodyid[g])for g in gg]
  if any((m.body(b).name or'').startswith('rh_')for b in bb):penetrations.append({'geoms':[m.geom(g).name for g in gg],'distance_m':float(c.dist)})
 tablegap=min(oracle.query(d,g,h.table)[0]for g in h.hand_geoms[1]if m.geom_type[g]in(mj.mjtGeom.mjGEOM_MESH,mj.mjtGeom.mjGEOM_CYLINDER))
 m.geom_margin[h.hand_geoms[1]]=.0015;mj.mj_forward(m,d)
 cols=contact_columns(m,d,obj,('rh_thumb','rh_index','rh_middle','rh_ring','rh_pinky'),projection=P,max_gap_m=.003)
 lp=wrench_lp(cols['columns'],cols['normal_rows'],np.r_[m.body_mass[obj]*m.opt.gravity,np.zeros(3)],normal_cap_N=None,motor_torque_matrix=cols['motor_torque_matrix'],motor_gravity_bias_Nm=P.T@d.qfrc_bias,motor_cap_Nm=.4)
 m.geom_margin[:]=margin
 OR=d.xmat[obj].reshape(3,3);template={'hand':'right','palm_offset_in_object_m':(OR.T@(h.palm(d,1)-d.xpos[obj])).tolist(),'hand_rotation_in_object_wxyz':Rotation.from_matrix(OR.T@d.xmat[h.hand_bids[1]].reshape(3,3)).as_quat()[[3,0,1,2]].tolist(),'finger_joint_targets':{m.joint(j).name:float(d.qpos[m.jnt_qposadr[j]])for j in finger},'controlled_digits':['thumb','index','middle','ring','pinky'],'required_contact_digits':['thumb','index','middle']}
 row={'closure_fraction':fraction,'nominal_gap_m':.016-fraction*.004,'six_commands_rad':command.tolist(),'qpos':d.qpos.tolist(),'template':template,'wrench':lp,'prospective_contacts':cols['patches'],'physical_hand_contacts':penetrations,'maximum_hand_penetration_m':max((max(0,-r['distance_m'])for r in penetrations),default=0),'minimum_table_SAT_m':tablegap};rows.append(row)
 print(json.dumps({k:v for k,v in row.items()if k not in ['qpos','template']}),flush=True)
a.output.write_text(json.dumps({'scope':'private fixed-wrist closure only; candidate for actuator trial, not successful physics','model':str(a.model),'model_sha256':hashlib.sha256(a.model.read_bytes()).hexdigest(),'source_seed':str(a.seed),'rows':rows,'best':min(rows,key=lambda r:(not r['wrench']['feasible'],r['maximum_hand_penetration_m']>.0005,r['wrench']['residual_force_equivalent_L1'],r['maximum_hand_penetration_m']))},indent=2)+'\n')
