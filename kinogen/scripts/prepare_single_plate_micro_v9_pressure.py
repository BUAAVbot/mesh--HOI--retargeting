"""Independent thin+FF comparison: pressure retention and an incremental lift."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
s=(ROOT/'scripts/run_fourth_single_plate_microstep_v7_reaction.py').read_text()
def once(old,new):
    global s
    assert s.count(old)==1,old
    s=s.replace(old,new)
once('from fourth_o6_load_hysteresis_cached_c10 import O6CoupledReferenceContactFeedback','from fourth_o6_pressure_latch_v9 import O6CoupledReferenceContactFeedback\nfrom fourth_lift_clock_v9 import IncrementalLift')
once("stop='duration';start=time.monotonic()", "stop='duration';start=time.monotonic();lift_clock=IncrementalLift();lift_motion_permitted=False;lift_pressure_targets=None;lift_pressure_allocation_since=None;lift_pressure_memory_updates=[];lift_pair_lost_since=None;lift_pause_samples=0")
once(" else:phase='lift';goal=holdp+np.array([0,0,min(.08,.012*(now-lift_started))]);rotation=holdR", " else:\n  phase='lift'if lift_motion_permitted else'lift_paused';goal=holdp+np.array([0,0,lift_clock.advance(now,lift_motion_permitted)]);rotation=holdR")
once(" goals=targets if lp_ok else{v:min(.25,targets[v])for v in digits};plan=", " finger.set_pressure_hold(1,lift_started is not None)\n goals=lift_pressure_targets if lift_started is not None else(targets if lp_ok else{v:min(.25,targets[v])for v in digits});plan=")
once(" if lift_started is None and ready_since is not None and d.time-ready_since+1e-9>=.1:lift_started=float(d.time);holdp=arm.palm(d,1).copy();holdR=d.xmat[arm.hand_bids[1]].reshape(3,3).copy()", """ if lift_started is None and ready_since is not None and d.time-ready_since+1e-9>=.1:
  lift_started=float(d.time);holdp=arm.palm(d,1).copy();holdR=d.xmat[arm.hand_bids[1]].reshape(3,3).copy();lift_pressure_targets=targets.copy();lift_pressure_allocation_since=allocation_since;lift_pressure_memory_updates.append({'time_s':float(d.time),'targets_N':targets.copy(),'allocation_time_s':allocation_since});lift_clock.start(d.time)
 if lift_started is not None:
  if ready and allocation_since is not None and allocation_since!=lift_pressure_allocation_since:
   lift_pressure_targets=targets.copy();lift_pressure_allocation_since=allocation_since;lift_pressure_memory_updates.append({'time_s':float(d.time),'targets_N':targets.copy(),'allocation_time_s':allocation_since});finger.reseed_pressure_hold(1)
  lift_motion_permitted=bool(ready)
  lift_pause_samples+=int(not lift_motion_permitted)
  live_pair=bool(Fdigit['thumb']>.03 and sum(Fdigit[v]for v in digits if v!='thumb')>.03 and opposition<=-.25)
  if live_pair:lift_pair_lost_since=None
  elif lift_pair_lost_since is None:lift_pair_lost_since=float(d.time)
""")
once("report.update(contact_reaction_feedforward=", "report.update(lift_pressure_policy='Formal-lift pressure retains last certified allocation through LP or individual-digit loss; unchanged native force/torque/table guards remain. A new certified allocation replaces memory through original0.8s crossfade. Wrist increments pause unless strict ready is true and never catch up elapsed time; actual opposing thumb-bank loss150ms terminates the trial.',lift_latched_pressure_targets_N=lift_pressure_targets,lift_pressure_memory_updates=lift_pressure_memory_updates,lift_pause_samples=lift_pause_samples,lift_command_height_m=lift_clock.height,contact_reaction_feedforward=")
# Keep the final physical sample and all protection logic. Abort is a termination,
# never a successful grasp and never a change to the final two-second proof.
once("write_micro_trace(a.output/'physical_trace.npz',m,d)", " if lift_pair_lost_since is not None and d.time-lift_pair_lost_since>.15:stop='opposing_contact_lost_during_lift';break\nwrite_micro_trace(a.output/'physical_trace.npz',m,d)")
once("for v7_name in ['fourth_hand_reaction_ff_v7.py','fourth_ground_reaction_pd_loaded_v7.py']:", "for v7_name in ['fourth_hand_reaction_ff_v7.py','fourth_ground_reaction_pd_loaded_v7.py','fourth_o6_pressure_latch_v9.py','fourth_lift_clock_v9.py','fourth_single_plate_benchmark_groundaware.py']:")
target=ROOT/'scripts/run_fourth_single_plate_microstep_v9_pressure.py'
target.write_text(s);print(target)
