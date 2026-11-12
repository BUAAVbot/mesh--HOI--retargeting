"""Pure five-finger load-sharing LP on a frozen private geometry candidate."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import mujoco as mj
from scipy.optimize import linprog
from fourth_contact_wrench import contact_columns
p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
src=json.loads(a.candidate.read_text());m=mj.MjModel.from_xml_path(src['model']);d=mj.MjData(m);d.qpos[:]=src['best']['qpos'];mj.mj_forward(m,d);names=[m.joint(j).name or''for j in range(m.njnt)];masters=[names.index('rh_'+n)for n in ['thumb_cmc_yaw','thumb_cmc_pitch','index_mcp_pitch','middle_mcp_pitch','ring_mcp_pitch','pinky_mcp_pitch']];P=np.zeros((m.nv,6))
for k,j in enumerate(masters):P[m.jnt_dofadr[j],k]=1
for e in range(m.neq):
 if m.eq_type[e]==mj.mjtEq.mjEQ_JOINT and int(m.eq_obj2id[e])in masters:P[m.jnt_dofadr[int(m.eq_obj1id[e])],masters.index(int(m.eq_obj2id[e]))]=m.eq_data[e,1]
hand=[g for g in range(m.ngeom)if m.geom_contype[g]and(m.body(int(m.geom_bodyid[g])).name or'').startswith('rh_')];m.geom_margin[hand]=.0015;mj.mj_forward(m,d);obj=m.body('target_object_01').id;cols=contact_columns(m,d,obj,('rh_thumb','rh_index','rh_middle','rh_ring','rh_pinky'),projection=P,max_gap_m=.003)
A=np.asarray(cols['columns']).T;N=np.asarray(cols['normal_rows']).T;M=cols['motor_torque_matrix'];bias=P.T@d.qfrc_bias;K=A.shape[1];scale=np.array([1,1,1,10,10,10.]);external=np.r_[m.body_mass[obj]*m.opt.gravity,np.zeros(3)];present={int(row['digit'])for row in cols['patches']};floor=np.array([(.3 if i==0 else .15)if i in present else 0. for i in range(5)]);cap=.35
U=np.vstack([np.column_stack([N,-np.ones(5)]),np.column_stack([M,np.zeros(6)]),np.column_stack([-M,np.zeros(6)]),np.column_stack([-N,np.zeros(5)])]);B=np.r_[np.zeros(5),bias+cap,cap-bias,-floor]
r=linprog(np.r_[np.zeros(K),1.],A_ub=U,b_ub=B,A_eq=np.column_stack([scale[:,None]*A,np.zeros(6)]),b_eq=-scale*external,bounds=(0,None),method='highs');report={'scope':'Static private geometry load sharing only, not actual forces or a successful grasp','source_candidate':str(a.candidate),'source_sha256':hashlib.sha256(a.candidate.read_bytes()).hexdigest(),'feasible':bool(r.success),'motor_planning_budget_Nm':cap,'actual_model_hard_cap_Nm':.4,'normal_floor_N':floor.tolist()}
if r.success:
 weights=r.x[:-1];normal=N@weights;tau=bias-M@weights;scaled_tau=bias-M@(weights*1.08);report.update(normal_per_digit_N=normal.tolist(),digit_force_targets_N=dict(zip(['thumb','index','middle','ring','pinky'],normal.tolist())),required_motor_torque_Nm=tau.tolist(),scaled_108pct_motor_torque_Nm=scaled_tau.tolist(),max_normal_N=float(r.x[-1]),equilibrium_max_error=float(np.max(abs(A@weights+external))),prospective_contacts=cols['patches'])
a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
