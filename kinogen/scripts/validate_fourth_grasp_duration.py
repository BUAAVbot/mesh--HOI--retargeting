"""Verify the nominal 0.1s physical-time ready boundary on recorded data."""
from pathlib import Path
import argparse,json
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--trial',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=json.loads((a.trial/'report.json').read_text());h=dict(np.load(a.trial/'sonic_contact_feedback.npz'));t=h['arm_feedback_time'];e=h['arm_feedback_grasp_evidence'][:,1];ready=(e[:,0]>.5)&(e[:,2]>=e[:,3]);dt=float(r['physics_dt_s']);first=None;start=None
for k,value in enumerate(ready):
    if not value:start=None;continue
    if start is None:start=k
    duration=float(t[k]-t[start]);ticks=int(round(t[k]/dt))-int(round(t[start]/dt))
    if duration+1e-9>=.1:
        first={'start_s':float(t[start]),'end_s':float(t[k]),'float_duration_s':duration,'integer_physics_ticks':ticks,'nominal_duration_from_ticks_s':ticks*dt,'strict_float_pass':bool(duration>=.1),'tolerant_float_pass':True};break
checks={'boundary_record_exists':first is not None,'nominal_duration_not_reduced':first is not None and first['integer_physics_ticks']>=20,'floating_error_reproduced':first is not None and not first['strict_float_pass'],'error_tolerance_tiny_relative_to_physics_dt':1e-9<dt/1e6};out={'status':'passed'if all(checks.values())else'failed','scope':'Frozen actual contact data; no physics or threshold changes','checks':checks,'first_ready_boundary':first,'duration_threshold_s':.1,'comparison_epsilon_s':1e-9};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2),flush=True)
if not all(checks.values()):raise SystemExit(2)
