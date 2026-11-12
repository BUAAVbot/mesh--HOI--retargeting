"""Read-only analysis of the valid prefix before any solver time reset."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('trial',type=Path);a=p.parse_args()
z=np.load(a.trial/'physical_trace.npz',allow_pickle=False)
t=z['time'];bad=np.flatnonzero(np.diff(t)<=0);stop=int(bad[0])+1 if len(bad) else len(t)
q=z['qpos'][:stop];v=z['qvel'][:stop];dt=float(np.median(np.diff(t[:stop])))
names=list(map(str,z['dof_names']));va=z['joint_vadr'].astype(int)
peaks=np.max(abs(v[:,va]),axis=0);order=np.argsort(peaks)[::-1]
def peak_rows(indices):
    return [{'joint':names[i],'dof':int(va[i]),'speed_peak_rad_s':float(peaks[i]),'at_s':float(t[np.argmax(abs(v[:,va[i]]))]),'last_speed_rad_s':float(v[-1,va[i]])}for i in indices]
body=[i for i in order if not names[i].startswith(('lh_','rh_','neck'))][:8]
aux=[i for i in order if names[i].startswith(('lh_','rh_','neck'))][:8]
root=[]
for target in [0,1,2,5,10,15,20,25,float(t[stop-1])]:
    i=int(np.argmin(abs(t[:stop]-target)));qt=q[i,3:7]/np.linalg.norm(q[i,3:7]);tilt=np.degrees(np.arccos(np.clip(1-2*(qt[1]**2+qt[2]**2),-1,1)))
    row={'time_s':float(t[i]),'root_position_m':q[i,:3].tolist(),'root_tilt_deg':float(tilt)}
    if i>0:row['source_root_position_error_m']=z['root_position_error_m'][i-1].tolist()
    root.append(row)
err=z['root_position_error_m'][:stop-1]
report={'valid_state_count':stop,'last_valid_time_s':float(t[stop-1]),'reset_state_count':len(t)-stop,
        'root_error_xy_max_m':float(np.max(np.linalg.norm(err[:,:2],axis=1))),
        'root_error_z_abs_max_m':float(np.max(abs(err[:,2]))),'root_drop_from_initial_m':float(q[0,2]-q[:,2].min()),
        'max_penetration_m':float(np.max(z['max_penetration_m'][:stop-1])),
        'body_speed_peaks':peak_rows(body),'aux_speed_peaks':peak_rows(aux),'root_samples':root,
        'hand_object_contact_duration_s':(np.sum(z['object_hand_normal_force_N'][:stop-1]>.01,axis=0)*dt).tolist()}
print(json.dumps(report,indent=2,allow_nan=False),flush=True)
