"""Read recorded states/contacts to compare force and distance-control segments."""
from pathlib import Path
import argparse,json
import numpy as np
import mujoco as mj
p=argparse.ArgumentParser();p.add_argument('--trial',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
m=mj.MjModel.from_binary_path(str(a.trial/'effective_model.mjb'));z=dict(np.load(a.trial/'physical_trace.npz'));c=dict(np.load(a.trial/'physical_contacts.npz'));d=mj.MjData(m)
times=z['time_s'];obj=m.body('target_object_01').id;groups={}
for i,bs in enumerate(c['body']):
    if obj not in bs:continue
    other=int(bs[0]if bs[1]==obj else bs[1]);name=m.body(other).name or''
    if not name.startswith('rh_thumb'):continue
    row=groups.setdefault(name,{'normal_impulse_Ns':0.,'normal_peak_N':0.,'rows':0,'first_s':None,'last_s':None})
    force=max(float(c['force'][i,0]),0.);t=float(times[int(c['step'][i])+1])
    row['normal_impulse_Ns']+=force*float(m.opt.timestep);row['normal_peak_N']=max(row['normal_peak_N'],force);row['rows']+=1
    row['first_s']=t if row['first_s']is None else min(row['first_s'],t);row['last_s']=t
thumb=[g for g in range(m.ngeom)if m.geom_contype[g]and(m.body(int(m.geom_bodyid[g])).name or'').startswith('rh_thumb')]
distal=[g for g in thumb if(m.body(int(m.geom_bodyid[g])).name or'')=='rh_thumb_distal'];objects=[g for g in range(m.ngeom)if m.geom_contype[g]and m.geom_bodyid[g]==obj]
samples=[]
for t in [59.5,64.,68.,72.,80.,float(times[-1])]:
    if t>times[-1]+1e-8:continue
    k=int(np.argmin(abs(times-t)));d.qpos[:]=z['qpos'][k];mj.mj_forward(m,d)
    def gap(geoms):
        values=[(float(mj.mj_geomDistance(m,d,g,o,1.,np.zeros(6))),m.geom(g).name,m.body(int(m.geom_bodyid[g])).name)for g in geoms for o in objects]
        return min(values)
    angles={name:float(d.qpos[m.joint(name).qposadr[0]])for name in ['rh_thumb_cmc_yaw','rh_thumb_cmc_pitch','rh_thumb_ip']}
    ctrl={name:float(z['ctrl'][max(k-1,0),m.actuator('motor_'+name).id])for name in ['rh_thumb_cmc_yaw','rh_thumb_cmc_pitch']}
    samples.append({'time_s':float(times[k]),'all_thumb_segment_gap':gap(thumb),'distal_only_gap':gap(distal),'actual_q_rad':angles,'ctrl_targets_rad':ctrl})
out={'scope':'Read-only recorded-contact and approximate geomDistance diagnostics; no simulated motion/grasp success claim.','trial':str(a.trial),'thumb_contact_body_groups':groups,'samples':samples}
a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
