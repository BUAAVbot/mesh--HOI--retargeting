from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
text=(ROOT/'scripts/run_fourth_o6_C7_latched_actual_load.py').read_text()
text=text.replace("last_good_lp=-1e9;last_seen", "locked_palm=None;lost_since=None;last_good_lp=-1e9;last_seen")
old=""" if lift_started is None:phase='tactile_align'if not lp_ok else'load';goal=pstart+s*(want-pstart);rotation=Rotation.from_rotvec(s*Rotation.from_matrix(wantR@Rstart.T).as_rotvec()).as_matrix()@Rstart
 else:phase='lift';goal=holdp+np.array([0,0,min(.08,.012*(now-lift_started))]);rotation=holdR
"""
new=""" if lift_started is None:
  phase='tactile_align'if not lp_ok else'load'
  if locked_palm is None:goal=pstart+s*(want-pstart);rotation=Rotation.from_rotvec(s*Rotation.from_matrix(wantR@Rstart.T).as_rotvec()).as_matrix()@Rstart
  else:goal=locked_palm[0];rotation=locked_palm[1]
 else:phase='lift';goal=holdp+np.array([0,0,min(.08,.012*(now-lift_started))]);rotation=holdR
"""
assert old in text
text=text.replace(old,new).replace("if now-lastlp>=.1:","if now-lastlp>=.019999999:")
text=text.replace("if lp['feasible']:\n   last_good_lp=now", "if lp['feasible']:\n   last_good_lp=now\n   if locked_palm is None:locked_palm=(arm.palm(d,1).copy(),d.xmat[arm.hand_bids[1]].reshape(3,3).copy())")
text=text.replace(" lp_ok=bool(recent and now-last_good_lp<=.100000001)", " lp_ok=bool(recent and now-last_good_lp<=.100000001)\n if recent:lost_since=None\n else:\n  if lost_since is None:lost_since=now\n  if now-lost_since>.15:locked_palm=None")
text=text.replace("'load_contact_grace_s':.1", "'load_contact_grace_s':.1,'actual_wrench_check_hz':50,'palm_loading_policy':'Lock actual world palm/R at first actual full-wrench feasible contact; release latch after150ms real contact loss'")
target=ROOT/'scripts/run_fourth_o6_C8_locked_actual_load.py'
target.write_text(text)
print(target)
