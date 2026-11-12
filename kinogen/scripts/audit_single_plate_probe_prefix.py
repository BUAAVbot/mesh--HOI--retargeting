"""Verify FF-only and FF+probe are identical before the first probe action."""
from pathlib import Path
import json
import numpy as np
root=Path(__file__).resolve().parents[1]/'outputs/fourth_full_sequence'
a=root/'physics_single_plate_microstep_v7_reaction'
b=root/'physics_single_plate_microstep_v8_probe'
report=json.loads((b/'report.json').read_text());probe_time=report['loading_probe_events'][0]['time_s']
za=np.load(a/'physical_trace.npz');zb=np.load(b/'physical_trace.npz')
ta=za['time_s'];tb=zb['time_s'];count=int(np.searchsorted(tb,probe_time+1e-9,side='right'))
errors={k:float(np.max(abs(za[k][:count]-zb[k][:count])))for k in ['qpos','qvel']}
errors['time_s']=float(np.max(abs(ta[:count]-tb[:count])))
errors['ctrl']=float(np.max(abs(za['ctrl'][:count-1]-zb['ctrl'][:count-1])))
result={'success':max(errors.values())<1e-12,'prefix_through_s':float(tb[count-1]),
        'probe_start_time_s':probe_time,'microstates_compared':count,'maximum_absolute_difference':errors}
(b/'independent_probe_prefix.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
