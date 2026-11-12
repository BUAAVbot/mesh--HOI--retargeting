"""Compare observed object motion onset with inferred hand contact windows."""
from pathlib import Path
import argparse,json
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path);a=p.parse_args()
r=np.load(a.reference,allow_pickle=False);t=r['time_s']-r['time_s'][0];poses=r['object_target_pose'];mask=r['object_observed_mask'];candidate=r['contact_candidate_observed']
cfg=Path(__file__).resolve().parents[1]/'configs/fourth_sonic_newft15225_object_feedback.json'
plane=json.loads(cfg.read_text()).get('arm_feedback',{}).get('table_plane_world_abcd')if cfg.exists()else None
plane=np.asarray(plane,float)if plane is not None else None
out=[]
for j,oid in enumerate(r['object_ids']):
    valid=np.flatnonzero(mask[:,j]);delta=np.linalg.norm(poses[:,j,:3]-poses[valid[0],j,:3],axis=1)
    moved=np.flatnonzero(mask[:,j]&(delta>.025));windows=[]
    for hand in range(2):
        ids=np.flatnonzero(candidate[:,hand,j]);splits=np.flatnonzero(np.diff(t[ids])>.55)+1
        for group in np.split(ids,splits):
            if len(group):windows.append({'hand':hand,'start_s':float(t[group[0]]),'end_s':float(t[group[-1]]),'frames':len(group)})
    velocity=np.linalg.norm(np.diff(poses[:,j,:3],axis=0)/np.diff(t)[:,None],axis=1);obsvel=mask[1:,j]&mask[:-1,j]
    out.append({'object_id':int(oid),'first_observed_motion_25mm_time_s':float(t[moved[0]])if len(moved)else None,
        'contact_windows':windows,'observed_speed_p95_m_s':float(np.quantile(velocity[obsvel],.95)),
        'observed_speed_max_m_s':float(velocity[obsvel].max()),'observed_displacement_max_m':float(delta[mask[:,j]].max()),
        'initial_observed_position_m':poses[valid[0],j,:3].tolist(),
        'final_observed_position_m':poses[valid[-1],j,:3].tolist(),
        'object_origin_height_above_table_range_m':([float(v)for v in [np.min(poses[valid,j,:3]@plane[:3]+plane[3]),np.max(poses[valid,j,:3]@plane[:3]+plane[3])]]if plane is not None else None)})
path=a.output or a.reference.with_name(a.reference.stem+'_contact_schedule.json');path.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
