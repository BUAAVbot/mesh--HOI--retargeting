"""Replan a stale latched pressure allocation, independently of global ability."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
s=(ROOT/'scripts/run_fourth_single_plate_microstep_v9_pressure.py').read_text()
def once(old,new):
    global s
    assert s.count(old)==1,old
    s=s.replace(old,new)
once('lift_pause_samples=0','lift_pause_samples=0;pressure_caps_invalid_since=None;pressure_cap_checks=[];pressure_replan_events=[];pressure_plan_since=-1e9')
once("  if lp['feasible']:\n   last_good_lp=now", """  if lift_started is not None and lift_pressure_targets is not None:
   pressure_caps=[float(lift_pressure_targets[v])for v in digits]
   cap_lp=wrench_lp(items['columns'],items['normal_rows'],np.r_[m.body_mass[obj]*m.opt.gravity,np.zeros(3)],normal_cap_N=pressure_caps,motor_torque_matrix=items['motor_torque_matrix'],motor_gravity_bias_Nm=P.T@kin.qfrc_bias,motor_cap_Nm=.4)
   pressure_cap_checks.append({'time_s':now,'latched_caps_N':pressure_caps,'latched_caps_feasible':cap_lp['feasible'],'global_planning_feasible':lp['feasible'],'residual_force_equivalent_L1':cap_lp.get('residual_force_equivalent_L1')})
   if cap_lp['feasible']:pressure_caps_invalid_since=None
   elif pressure_caps_invalid_since is None:pressure_caps_invalid_since=now
   if pressure_caps_invalid_since is not None and now-pressure_caps_invalid_since>=.1 and now-pressure_plan_since>=.8 and lp['feasible']:
    allocation_latched=False;allocation_since=None;pressure_replan_events.append({'time_s':now,'reason':'Previous normal caps fail current6D balance for100ms although a new global motor-safe allocation exists','previous_caps_N':pressure_caps});pressure_caps_invalid_since=None
  if lp['feasible']:
   last_good_lp=now""")
once("  if ready and allocation_since is not None and allocation_since!=lift_pressure_allocation_since:", "  if allocation_latched and allocation_since is not None and allocation_since!=lift_pressure_allocation_since:")
once("lift_pressure_memory_updates.append({'time_s':float(d.time),'targets_N':targets.copy(),'allocation_time_s':allocation_since});finger.reseed_pressure_hold(1)", "lift_pressure_memory_updates.append({'time_s':float(d.time),'targets_N':targets.copy(),'allocation_time_s':allocation_since});pressure_plan_since=float(d.time);finger.reseed_pressure_hold(1)")
once('report.update(lift_pressure_policy=', "(a.output/'latched_pressure_checks.json').write_text(json.dumps(pressure_cap_checks,indent=2)+'\\n')\nreport.update(pressure_replanning_policy='During formal lift test current6D balance under remembered normal caps and unchanged.4Nm motors;100ms continuously infeasible triggers a new.35Nm plan only if scaled1.08rays pass original.4Nm projection; crossfade.8s and allow a full.8s before another pressure replan. Measured95pct certificate still gates wrist travel and final success.',pressure_replan_events=pressure_replan_events,lift_pressure_policy=")
target=ROOT/'scripts/run_fourth_single_plate_microstep_v10_caps.py'
target.write_text(s);print(target)
