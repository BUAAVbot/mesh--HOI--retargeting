"""Static minimax load-sharing and whole-robot collision audit of a candidate."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import mujoco as mj
from scipy.optimize import linprog
from fourth_contact_wrench import contact_columns

p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--budget',type=float,default=.35);a=p.parse_args()
r=json.loads(a.candidate.read_text());mfile=Path(r['model']);m=(mj.MjModel.from_binary_path(str(mfile))if mfile.suffix=='.mjb'else mj.MjModel.from_xml_path(str(mfile)));d=mj.MjData(m);d.qpos[:]=r['best']['qpos'];d.qvel[:]=0;digits=['thumb','index','middle','ring','pinky'];names=[m.joint(j).name or''for j in range(m.njnt)];masters=[names.index('rh_'+v)for v in ['thumb_cmc_yaw','thumb_cmc_pitch','index_mcp_pitch','middle_mcp_pitch','ring_mcp_pitch','pinky_mcp_pitch']];P=np.zeros((m.nv,6))
for k,j in enumerate(masters):P[m.jnt_dofadr[j],k]=1.
for e in range(m.neq):
 if m.eq_type[e] == mj.mjtEq.mjEQ_JOINT and int(m.eq_obj2id[e]) in masters:
  child = int(m.eq_obj1id[e])
  parent = int(m.eq_obj2id[e])
  P[int(m.jnt_dofadr[child]), masters.index(parent)] = m.eq_data[e,1]
handg=[g for g in range(m.ngeom)if m.geom_contype[g]and(m.body(int(m.geom_bodyid[g])).name or'').startswith('rh_')];margin=m.geom_margin.copy();m.geom_margin[handg]=.0015;mj.mj_forward(m,d);obj=m.body('target_object_01').id;items=contact_columns(m,d,obj,tuple('rh_'+v for v in digits),P,max_gap_m=.003);A=np.asarray(items['columns']).T;N=np.asarray(items['normal_rows']).T;M=items['motor_torque_matrix'];bias=P.T@d.qfrc_bias;K=A.shape[1];present=sorted(set(x['digit']for x in items['patches']));required=[k for k in present if k!=4];scale=np.array([1,1,1,10,10,10.]);U=np.r_[np.column_stack([N,-np.ones(5)]),np.column_stack([M,np.zeros(6)]),np.column_stack([-M,np.zeros(6)]),np.column_stack([-N[required],np.zeros(len(required))])];B=np.r_[np.zeros(5),bias+a.budget,a.budget-bias,-.1*np.ones(len(required))];eq=scale*np.r_[m.body_mass[obj]*m.opt.gravity,np.zeros(3)];fit=linprog(np.r_[np.zeros(K),1.],A_ub=U,b_ub=B,A_eq=np.column_stack([scale[:,None]*A,np.zeros(6)]),b_eq=-eq,bounds=(0,None),method='highs')
if not fit.success:raise RuntimeError('No minimax full-wrench solution within motor budget')
weights=fit.x[:-1];goals=N@weights;tau=bias-M@weights;scaled_tau=bias-M@(1.05*weights);m.geom_margin[:]=margin;mj.mj_forward(m,d);bad=[]
for c in d.contact[:d.ncon]:
 if c.dist>=-1e-5:continue
 gs=list(map(int,c.geom));bs=[int(m.geom_bodyid[g])for g in gs];bn=[m.body(b).name or'world'for b in bs];robot=[(i,n)for i,n in enumerate(bn)if n!='world'and not n.startswith('target_object')]
 if not robot:continue
 if all(any(x in n.lower()for x in ['foot','ankle'])for i,n in robot)and any('floor' in (m.geom(g).name or'')for g in gs):continue
 bad.append({'bodies':bn,'geoms':[m.geom(g).name for g in gs],'penetration_m':float(-c.dist)})
template=r['best']['template'];template['required_contact_digits']=[digits[k]for k in required];template['controlled_digits']=digits;template['digit_force_targets_N']={v:float(goals[k])if k in required else .05 for k,v in enumerate(digits)};r['balanced_loadshare']={'planning_motor_budget_Nm':a.budget,'physical_hard_limit_Nm':.4,'prospective_contact_digits':[digits[k]for k in present],'minimax_normal_N':goals.tolist(),'motor_torque_Nm':tau.tolist(),'motor_torque_scaled105_Nm':scaled_tau.tolist(),'equilibrium_max_error':float(np.max(abs(A@weights+np.r_[m.body_mass[obj]*m.opt.gravity,np.zeros(3)]))),'scope':'Prospective margins on private model copy only, restored before whole-scene collision audit; no actual grasp claim.'};r['whole_robot_scene_audit']={'unexpected_robot_penetrations':bad,'passed':not bad,'hand_colliders_unchanged':True};r['source_unenriched_candidate_sha256']=hashlib.sha256(a.candidate.read_bytes()).hexdigest();a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'output':str(a.output),'balanced':r['balanced_loadshare'],'whole_robot_scene_audit':r['whole_robot_scene_audit']}),flush=True)
