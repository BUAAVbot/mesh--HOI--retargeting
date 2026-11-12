from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'scripts/run_fourth_o6_C_tactile_align_v3.py'
target=ROOT/'scripts/run_fourth_o6_C7_latched_actual_load.py'
text=source.read_text()
text=text.replace('from fourth_contact_feedback_o6_coupled import O6CoupledReferenceContactFeedback',
                  'from fourth_o6_load_hysteresis_c7 import O6CoupledReferenceContactFeedback')
text=text.replace("'fourth_contact_wrench.py']:", "'fourth_contact_wrench.py','fourth_o6_load_hysteresis_c7.py']:")
text=text.replace("default=24.);p.add_argument('--orientation-weight'", "default=32.);p.add_argument('--orientation-weight'")
text=text.replace("proof_since=None;verified=False;stop='duration';start=time.monotonic()", "proof_since=None;verified=False;last_good_lp=-1e9;last_seen=np.full(5,-np.inf);longest_proof=0.;stop='duration';start=time.monotonic()")
old=""" all_contact=all(Fdigit[v]>.05 for v in required)
 if not all_contact:lp_ok=False
 elif now-lastlp>=.1:
  items=contact_columns(m,d,obj,tuple('rh_'+v for v in digits),P,max_gap_m=0.);kin.qpos[:]=d.qpos;kin.qvel[:]=0;mj.mj_forward(m,kin);lp=wrench_lp(items['columns'],items['normal_rows'],np.r_[m.body_mass[obj]*m.opt.gravity,np.zeros(3)],normal_cap_N=5.,motor_torque_matrix=items['motor_torque_matrix'],motor_gravity_bias_Nm=P.T@kin.qfrc_bias,motor_cap_Nm=.4);lastlp=now;previous_ok=lp_ok;lp_ok=bool(lp['feasible']);lplog.append({'time_s':now,'LP':lp,'patches':items['patches']})
  if lp_ok and not previous_ok:load_started=now;finger.states[1]['load_contact_since']=now
"""
new=""" all_contact=all(Fdigit[v]>.05 for v in required)
 rawforces=np.asarray([Fdigit[v]for v in digits]);last_seen[rawforces>.03]=now
 gaps,_=finger._geometry(finger.hands[1],0)
 recent=all(now-last_seen[digits.index(v)]<=.100000001 and gaps[digits.index(v)]<.002 for v in required)
 if now-lastlp>=.1:
  items=contact_columns(m,d,obj,tuple('rh_'+v for v in digits),P,max_gap_m=0.);kin.qpos[:]=d.qpos;kin.qvel[:]=0;mj.mj_forward(m,kin)
  lp=wrench_lp(items['columns'],items['normal_rows'],np.r_[m.body_mass[obj]*m.opt.gravity,np.zeros(3)],normal_cap_N=4.8,motor_torque_matrix=items['motor_torque_matrix'],motor_gravity_bias_Nm=P.T@kin.qfrc_bias,motor_cap_Nm=.39);lastlp=now
  if lp['feasible']:
   last_good_lp=now
   desired={v:max(.05,1.01*float(x))for v,x in zip(digits,lp['normal_per_digit_N'])}
   if sum(desired.values())<=float(cfg['max_hand_normal_force_N']) and max(desired.values())<=float(cfg['max_force_per_digit_N']):
    targets=desired
  lplog.append({'time_s':now,'LP':lp,'patches':items['patches'],'actual_derived_targets_N':targets.copy()})
 lp_ok=bool(recent and now-last_good_lp<=.100000001)
"""
assert old in text
text=text.replace(old,new)
text=text.replace("proof=ready and rise>=.012 and tableforce<.15 and upforce>=.7*m.body_mass[obj]*9.81", "proof=ready and rise>=.03 and tableforce<.15 and upforce>=.7*m.body_mass[obj]*9.81")
text=text.replace("if d.time-proof_since+1e-9>=.08:verified=True", "longest_proof=max(longest_proof,float(d.time-proof_since))\n  if d.time-proof_since+1e-9>=2.:verified=True")
text=text.replace("'actual_LP_feasible_checks':sum", "'stable_3cm_support_max_duration_s':longest_proof,'load_contact_grace_s':.1,'actual_LP_budget_Nm':.39,'load_targets_from_actual_LP':True,'actual_LP_feasible_checks':sum")
target.write_text(text)
print(target)
