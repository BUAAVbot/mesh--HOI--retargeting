"""Read-only diagnosis of an auxiliary feedback range/velocity fault."""
from pathlib import Path
import argparse
import json
import xml.etree.ElementTree as ET
import numpy as np
import mujoco as mj
from fourth_contact_feedback import DEFAULTS,DEFAULT_URDF

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--trial-dir',type=Path,required=True);p.add_argument('--time',type=float,default=7.02)
p.add_argument('--output',type=Path,required=True);a=p.parse_args()
r=json.loads((a.trial_dir/'report.json').read_text());mp=Path(r['model']['path'])
m=mj.MjModel.from_binary_path(str(mp))if mp.suffix=='.mjb'else mj.MjModel.from_xml_path(str(mp))
z=np.load(a.trial_dir/'physical_trace.npz',allow_pickle=False)
t,q,v,ctrl=z['time'],z['qpos'],z['qvel'],z['ctrl'];index=int(np.argmin(abs(t-a.time)))
definitions={j.get('name'):j for j in ET.parse(DEFAULT_URDF).getroot().findall('joint')}
rows=[]
for j in range(m.njnt):
    name=mj.mj_id2name(m,mj.mjtObj.mjOBJ_JOINT,j)or''
    if not name.startswith('rh_'):continue
    qa=int(m.jnt_qposadr[j]);va=int(m.jnt_dofadr[j]);aid=mj.mj_name2id(m,mj.mjtObj.mjOBJ_ACTUATOR,'motor_'+name)
    speed=float(definitions[name].find('limit').get('velocity'))
    limit=max(speed*DEFAULTS['joint_velocity_fault_multiplier'],DEFAULTS['joint_velocity_fault_floor_rad_s'])
    outside=(q[:,qa]<m.jnt_range[j,0]-DEFAULTS['joint_range_fault_margin_rad'])|(q[:,qa]>m.jnt_range[j,1]+DEFAULTS['joint_range_fault_margin_rad'])
    fast=abs(v[:,va])>limit;bad=np.flatnonzero(outside|fast)
    rows.append({'joint':name,'dof':va,'qpos_rad':float(q[index,qa]),'qvel_rad_s':float(v[index,va]),
        'model_range_rad':m.jnt_range[j].tolist(),'urdf_velocity_limit_rad_s':speed,'feedback_velocity_fault_threshold_rad_s':limit,
        'range_fault_at_requested_time':bool(outside[index]),'velocity_fault_at_requested_time':bool(fast[index]),
        'first_range_or_velocity_fault_s':float(t[bad[0]])if len(bad)else None,
        'ctrl_before_rad':float(ctrl[max(0,index-5),aid]),'ctrl_at_time_rad':float(ctrl[min(index,len(ctrl)-1),aid]),
        'actual_actuator_force_at_time_Nm':float(z['actuator_force'][min(index,len(ctrl)-1),aid])})
c={k:v for k,v in np.load(a.trial_dir/'physical_contacts.npz',allow_pickle=False).items()}
intervals=c['interval_index'];window=(intervals>=max(0,index-8))&(intervals<=index+2)
names=[mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,b)or'world'for b in range(m.nbody)]
rhs=np.asarray([name.startswith('rh_')for name in names])
selected=np.flatnonzero(window&(rhs[c['body_ids'][:,0]]|rhs[c['body_ids'][:,1]]))
selected=selected[np.argsort(c['normal_force_N'][selected])[::-1]][:12]
contacts=[{'time_s':float(t[int(intervals[k])]),'bodies':[names[int(b)]for b in c['body_ids'][k]],
           'geoms':[mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,int(g))for g in c['geom_ids'][k]],
           'normal_force_N':float(c['normal_force_N'][k]),'distance_m':float(c['signed_distance_m'][k])}for k in selected]
result={'schema':'fourth.finger_feedback_fault_diagnosis.v1','trial':str(a.trial_dir),'requested_time_s':a.time,
        'actual_sample_time_s':float(t[index]),'right_hand_joints':rows,'nearby_right_hand_contacts':contacts,
        'controller_or_model_modified':False,'scope':'Recorded state, commanded target and collision evidence only; no new physical success claimed.'}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'faulting_joints':[row for row in rows if row['range_fault_at_requested_time']or row['velocity_fault_at_requested_time']],
                  'largest_contacts':contacts[:4]}),flush=True)
