"""Summarize actual lowest-collider height windows; no success inference."""
from pathlib import Path
import argparse,json
import numpy as np
p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args()
z=np.load(a.output/'grasp_history.npz');v=z['values'];names=z['columns'].tolist()
col={name:i for i,name in enumerate(names)};t=v[:,col['time_s']];h=v[:,col['object_min_clearance_from_table_plane_m']]
def duration(mask):
    start=None;longest=0.
    for ti,valid in zip(t,mask):
        if valid:
            if start is None:start=float(ti)
            longest=max(longest,float(ti)-start)
        else:start=None
    return longest
k=int(np.argmax(h));report={'scope':'Collider-height windows only; these are not a stable grasp certificate.',
    'maximum_lowest_collider_clearance_m':float(h[k]),'maximum_time_s':float(t[k]),
    'height_ge3cm_only_longest_window_s':duration(h>=.03),
    'height_ge3cm_and_table_unloaded_longest_window_s':duration((h>=.03)&(v[:,col['table_normal_N']]<.15)),
    'final_lowest_clearance_m':float(h[-1]),'final_phase':str(z['phase'][-1]),
    'maximum_frame':dict(zip(names,map(float,v[k])))}
(a.output/'independent_height_windows.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
