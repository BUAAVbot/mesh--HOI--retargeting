"""Prepare two declared load-certificate consistency comparisons."""
from pathlib import Path
R=Path(__file__).resolve().parents[1]
s=(R/'scripts/run_fourth_o6_stiff_eq_C11_segments_v1.py').read_text()
old="'min_clearance_m':clearance}"
new="'min_clearance_m':clearance,'desired_normal_N':targets,'applied_target_normal_N':finger.coupled_records[-1]['hands'][1]['target_force_per_digit_N_vector'].tolist(),'load_ramp_fraction':finger.coupled_records[-1]['hands'][1]['load_force_ramp_fraction'],'table_guard':finger.coupled_guard_records[-1][1]}"
# JSON cannot encode numpy scalars from the full guard dict; all guard fields
# are already Python scalar values in the source helper.
assert old in s;s=s.replace(old,new)
a=s.replace('motor_cap_Nm=.39);lastlp=now','motor_cap_Nm=.35);lastlp=now').replace('1.01*float(x)','1.08*float(x)')
needle="   if sum(desired.values())<=float(cfg['max_hand_normal_force_N']) and max(desired.values())<=float(cfg['max_force_per_digit_N']):"
replacement="""   bias=P.T@kin.qfrc_bias;scaled_tau=bias-1.08*(bias-np.asarray(lp['required_motor_torque_Nm']))
   lp['scaled_load_goal_motor_torque_Nm']=scaled_tau.tolist();lp['scaled_load_goal_motor_limit_pass']=bool(np.max(abs(scaled_tau))<=.4+1e-9)
   if lp['scaled_load_goal_motor_limit_pass'] and sum(desired.values())<=float(cfg['max_hand_normal_force_N']) and max(desired.values())<=float(cfg['max_force_per_digit_N']):"""
assert needle in a;a=a.replace(needle,replacement)
a=a.replace("'actual_LP_budget_Nm':.39","'actual_LP_budget_Nm':.35,'actual_LP_load_multiplier':1.08,'scaled_load_torque_rechecked':True")
b=s.replace('normal_cap_N=.95*np.asarray','normal_cap_N=np.asarray').replace('capped at95pct','capped at100pct')
b=b.replace("'actual_LP_budget_Nm':.39","'actual_LP_budget_Nm':.39,'measured_normal_certificate_factor':1.0")
for name,code in [('run_fourth_o6_stiff_C12_loadmargin_v1.py',a),('run_fourth_o6_stiff_C13_measured100_v1.py',b)]:
    p=R/'scripts'/name
    if p.exists():raise FileExistsError(p)
    compile(code,str(p),'exec');p.write_text(code);print(p)
