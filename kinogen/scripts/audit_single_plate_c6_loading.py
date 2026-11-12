"""Independent recorded-control replay and real-contact load budget diagnosis."""
from pathlib import Path
import argparse, json, hashlib
import numpy as np
import mujoco as mj
from scipy.optimize import linprog
from fourth_contact_wrench import contact_columns, wrench_lp

p=argparse.ArgumentParser();p.add_argument('--trial',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--balanced-module',action='store_true');a=p.parse_args()
if a.balanced_module:
 from fourth_contact_wrench_balanced_c9 import wrench_lp
m=mj.MjModel.from_binary_path(str(a.trial/'effective_model.mjb'));d=mj.MjData(m)
z=np.load(a.trial/'physical_trace.npz');h=np.load(a.trial/'grasp_history.npz');f=np.load(a.trial/'finger_history.npz');v=h['values'];dt=float(m.opt.timestep)
ok=v[:,-1]>0;edges=np.diff(np.r_[False,ok,False].astype(int));spans=[{'start_s':float(v[s,0]),'end_s':float(v[e-1,0]),'duration_s':float((e-s)*dt)}for s,e in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1))]
report={'scope':'Recorded original-model actual-point contact analysis; no new rollout or success claim.', 'trial':str(a.trial),'trace_sha256':hashlib.sha256((a.trial/'physical_trace.npz').read_bytes()).hexdigest(),'LP_feasible_spans':spans,'LP_longest_duration_s':max((x['duration_s']for x in spans),default=0),'LP_total_duration_s':float(np.sum(ok)*dt),'load_ramp_fraction_max':np.max(f['load_force_ramp_fraction'],axis=0).tolist(),'commanded_normal_goals_max_N':np.max(f['target_force_per_digit_N_vector'],axis=0).tolist(),'actual_digit_force_peaks_N':np.max(v[:,9:14],axis=0).tolist(),'final_palm_error_m':float(v[-1,8]),'actual_lift_verified':False,'contact_budget_checks':[]}
names=[m.joint(j).name or''for j in range(m.njnt)];masters=[names.index('rh_'+n)for n in ['thumb_cmc_yaw','thumb_cmc_pitch','index_mcp_pitch','middle_mcp_pitch','ring_mcp_pitch','pinky_mcp_pitch']];P=np.zeros((m.nv,6))
for k,j in enumerate(masters):P[m.jnt_dofadr[j],k]=1.
for ei in range(m.neq):
 if m.eq_type[ei]==mj.mjtEq.mjEQ_JOINT and int(m.eq_obj2id[ei])in masters:P[m.jnt_dofadr[int(m.eq_obj1id[ei])],masters.index(int(m.eq_obj2id[ei]))]=m.eq_data[ei,1]
obj=m.body('target_object_01').id;kin=mj.MjData(m);mj.mj_setState(m,d,z['initial_state'],int(z['state_spec']));mj.mj_forward(m,d)
times=z['time_s'];target_times=[70.87] if a.balanced_module else [70.87,81.37,81.45];wanted={int(np.argmin(abs(times-t))):t for t in target_times}
maxerr=0.
for k,ctrl in enumerate(z['ctrl']):
 d.ctrl[:]=ctrl;mj.mj_step(m,d)
 if k+1 not in wanted:continue
 maxerr=max(maxerr,float(np.max(abs(d.qpos-z['qpos'][k+1]))));items=contact_columns(m,d,obj,tuple('rh_'+v for v in ['thumb','index','middle','ring','pinky']),P,max_gap_m=0.);kin.qpos[:]=d.qpos;kin.qvel[:]=0;mj.mj_forward(m,kin);bias=P.T@kin.qfrc_bias
 row={'requested_s':wanted[k+1],'actual_s':float(d.time),'patches':items['patches'],'budgets':[]}
 for cap in [.35,.38,.39,.4]:
  lp=wrench_lp(items['columns'],items['normal_rows'],np.r_[m.body_mass[obj]*m.opt.gravity,np.zeros(3)],normal_cap_N=5.,motor_torque_matrix=items['motor_torque_matrix'],motor_gravity_bias_Nm=bias,motor_cap_Nm=cap)
  if items['columns']:
   A=np.asarray(items['columns']).T;N=np.asarray(items['normal_rows']).T;M=items['motor_torque_matrix'];scale=np.array([1,1,1,10,10,10.]);K=A.shape[1];mmU=np.r_[np.column_stack([N,-np.ones(N.shape[0])]),np.column_stack([M,np.zeros(M.shape[0])]),np.column_stack([-M,np.zeros(M.shape[0])])];mmB=np.r_[np.zeros(N.shape[0]),bias+cap,cap-bias]
   mm=linprog(np.r_[np.zeros(K),1.],A_ub=mmU,b_ub=mmB,A_eq=np.column_stack([scale[:,None]*A,np.zeros(6)]),b_eq=-scale*np.r_[m.body_mass[obj]*m.opt.gravity,np.zeros(3)],bounds=(0,None),method='highs')
   if mm.success:
    lp['minimax_normal_per_digit_N']=(N@mm.x[:-1]).tolist();lp['minimax_motor_torque_Nm']=(bias-M@mm.x[:-1]).tolist()
    if lp['feasible']:
     lp['reported_solution_peak_N']=float(np.max(lp['normal_per_digit_N']));lp['independent_minimax_peak_N']=float(mm.x[-1]);lp['normal_cap_satisfied']=bool(np.max(lp['normal_per_digit_N'])<=5.+1e-8);lp['motor_cap_satisfied']=bool(np.max(abs(np.asarray(lp['required_motor_torque_Nm'])))<=cap+1e-8);lp['full_wrench_residual_satisfied']=bool(lp['equilibrium_max_error']<1e-7)
  row['budgets'].append({'motor_budget_Nm':cap,'wrench':lp})
 report['contact_budget_checks'].append(row);print(json.dumps(row),flush=True)
 if len(report['contact_budget_checks'])==len(wanted):break
report['selected_replay_qpos_max_error']=maxerr;a.output.write_text(json.dumps(report,indent=2)+'\n');print(str(a.output),flush=True)
