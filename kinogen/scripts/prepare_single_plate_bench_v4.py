from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
text=(ROOT/'scripts/run_fourth_single_plate_benchmark_v3.py').read_text()
text=text.replace("longest_proof=0.;measured_support_checks=0;", "allocation_latched=False;allocation_since=None;lp_invalid_since=None;allocations=[];longest_proof=0.;measured_support_checks=0;")
old=""" all_contact=all(Fdigit[v]>.05 for v in required)
 rawforces=np.asarray([Fdigit[v]for v in digits]);last_seen[rawforces>.03]=now"""
new=""" pair_dot=min((float(n1@n2)for v1,n1,f1 in normals for v2,n2,f2 in normals if v1=='thumb'and v2!='thumb'and min(f1,f2)>.03),default=1.)
 pair_anchor=t>=6. and Fdigit['thumb']>.08 and sum(Fdigit[v]for v in digits if v!='thumb')>.15 and pair_dot<=-.25
 if locked_palm is None and pair_anchor:locked_palm=(arm.palm(d,1).copy(),d.xmat[arm.hand_bids[1]].reshape(3,3).copy())
 all_contact=all(Fdigit[v]>.05 for v in required)
 rawforces=np.asarray([Fdigit[v]for v in digits]);last_seen[rawforces>.03]=now"""
assert old in text
text=text.replace(old,new)
old="""   desired={v:max(.05,1.08*float(x))for v,x in zip(digits,lp['normal_per_digit_N'])}
   if sum(desired.values())<=float(cfg['max_hand_normal_force_N']) and max(desired.values())<=float(cfg['max_force_per_digit_N']):
    targets=desired
  lplog.append({'time_s':now,'LP':lp,'patches':items['patches'],'actual_derived_targets_N':targets.copy()})
 lp_ok=bool(recent and now-last_good_lp<=.100000001)
 if recent:lost_since=None
 else:
  if lost_since is None:lost_since=now
  if now-lost_since>.15:locked_palm=None"""
new="""   if not allocation_latched:
    desired={v:max(.05,1.08*float(x))for v,x in zip(digits,lp['normal_per_digit_N'])}
    if sum(desired.values())<=float(cfg['max_hand_normal_force_N']) and max(desired.values())<=float(cfg['max_force_per_digit_N']):
     targets=desired;allocation_latched=True;allocation_since=now;load_started=now
     allocations.append({'time_s':now,'normal_targets_N':targets.copy(),'unscaled_planning_motor_budget_Nm':.35,'scaled_motor_max_Nm':lp['scaled_load_max_motor_Nm']})
   lp_invalid_since=None
  else:
   if lp_invalid_since is None:lp_invalid_since=now
   if now-lp_invalid_since>.15:allocation_latched=False;allocation_since=None
  lplog.append({'time_s':now,'LP':lp,'patches':items['patches'],'actual_derived_targets_N':targets.copy(),'allocation_latched':allocation_latched})
 lp_ok=bool(allocation_latched and recent and now-last_good_lp<=.150000001)
 pair_recent=now-last_seen[0]<=.1 and gaps[0]<.002 and any(now-last_seen[k]<=.1 and gaps[k]<.002 for k in range(1,5))
 if pair_recent or recent:lost_since=None
 else:
  if lost_since is None:lost_since=now
  if now-lost_since>.15:locked_palm=None"""
assert old in text
text=text.replace(old,new)
text=text.replace("'palm_loading_policy':'Lock actual world palm/R at first actual full-wrench feasible contact; release latch after150ms real contact loss'",
                  "'palm_loading_policy':'Anchor actual palm on real opposing thumb-bank preload contact; full LP certification still required before loading/lift'")
text=text.replace("report.update(scope=", "report.update(load_allocation_policy='Latch first margin-certified actual LP normal targets; replan only after continuous150ms actual LP invalidity',load_allocations=allocations,scope=")
target=ROOT/'scripts/run_fourth_single_plate_benchmark_v4.py'
target.write_text(text)
print(target)
