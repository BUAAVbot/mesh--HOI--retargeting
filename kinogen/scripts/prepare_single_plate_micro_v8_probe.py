"""Prepare an isolated loading-probe action; never alter the active v5b run."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / 'scripts/run_fourth_single_plate_microstep_v7_reaction.py').read_text()

def replace_once(old, new):
    global source
    assert source.count(old) == 1, old
    source = source.replace(old, new)

replace_once("stop='duration';start=time.monotonic()", "stop='duration';start=time.monotonic();probe_started=None;probe_candidate_since=None;probe_active=False;probe_height=0.;probe_peak_height=0.;probe_last_time=t0;probe_base=None;probe_R=None;probe_origin_z=None;probe_lost_since=None;probe_abort_reason=None;probe_events=[]")
replace_once(" now=float(d.time);t=now-t0", " now=float(d.time);t=now-t0;probe_dt=max(0.,now-probe_last_time);probe_last_time=now\n if probe_base is not None and lift_started is None:\n  probe_height=float(np.clip(probe_height+(.002 if probe_active else-.002)*probe_dt,0,.004));probe_peak_height=max(probe_peak_height,probe_height)")
replace_once(" else:phase='lift';goal=holdp+np.array([0,0,min(.08,.012*(now-lift_started))]);rotation=holdR", " else:phase='lift';goal=holdp+np.array([0,0,min(.08,.012*(now-lift_started))]);rotation=holdR\n if lift_started is None and probe_base is not None and(probe_active or probe_height>0):\n  phase='loading_probe'if probe_active else'loading_probe_retract';goal=probe_base+np.array([0.,0.,probe_height]);rotation=probe_R")
replace_once(" ready=lp_ok and all_contact and opposition<=-.25 and measured_ok", """ # A small physical weight-transfer motion is a control action, never a lift certificate.
 current_geometry_ok=bool(lastlp>=0 and now-lastlp<.03 and lp['feasible'])
 object_va=int(m.body_dofadr[obj]);object_slow=bool(np.linalg.norm(d.qvel[object_va:object_va+3])<.01 and np.linalg.norm(d.qvel[object_va+3:object_va+6])<.5)
 probe_candidate=bool(probe_started is None and lift_started is None and locked_palm is not None and allocation_latched and allocation_since is not None and now-allocation_since>=.8 and current_geometry_ok and all_contact and opposition<=-.25 and sum(Fdigit.values())>=1.3*m.body_mass[obj]*9.81 and tableforce<=.25*m.body_mass[obj]*9.81 and object_slow)
 if probe_candidate:
  if probe_candidate_since is None:probe_candidate_since=now
 else:probe_candidate_since=None
 if probe_candidate_since is not None and now-probe_candidate_since>=.1:
  probe_started=now;probe_active=True;probe_base=arm.palm(d,1).copy();probe_R=d.xmat[arm.hand_bids[1]].reshape(3,3).copy();probe_origin_z=float(d.qpos[oq+2]);probe_events.append({'time_s':now,'event':'start','table_force_N':tableforce,'measured_normal_N':Fdigit.copy(),'max_probe_height_m':.004,'speed_m_s':.002})
 if probe_active and lift_started is None:
  pair_live=bool(all_contact and opposition<=-.25)
  if pair_live:probe_lost_since=None
  elif probe_lost_since is None:probe_lost_since=now
  loss=probe_lost_since is not None and now-probe_lost_since>.1
  slip=probe_height>.001 and float(d.qpos[oq+2])<probe_origin_z-.003
  if loss or slip:
   probe_active=False;probe_abort_reason='opposing_raw_contact_lost_100ms'if loss else'object_dropped_3mm';probe_events.append({'time_s':now,'event':'retract','reason':probe_abort_reason,'probe_height_m':probe_height})
 ready=lp_ok and all_contact and opposition<=-.25 and measured_ok""")
replace_once("report.update(contact_reaction_feedforward=", "report.update(loading_probe_policy='After stable current geometric LP and true opposing load contacts, table residual<=0.25mg, and slow object, move palm up at2mm/s by at most4mm; retract upon100ms opposing-contact loss or3mm object drop. Formal lift still requires unchanged95pct measured-support LP.',loading_probe_events=probe_events,loading_probe_max_command_m=probe_peak_height,loading_probe_aborted_reason=probe_abort_reason,contact_reaction_feedforward=")

target = ROOT / 'scripts/run_fourth_single_plate_microstep_v8_probe.py'
target.write_text(source)
print(target)
