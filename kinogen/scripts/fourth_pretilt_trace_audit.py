"""Read-only force/tracking diagnosis of an actual partial-tilt trace."""
from pathlib import Path
import argparse,json
import numpy as np
import mujoco as mj
p=argparse.ArgumentParser();p.add_argument('--trial',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
z=dict(np.load(a.trial/'physical_trace.npz',allow_pickle=False));m=mj.MjModel.from_binary_path(str(a.trial/'plant.mjb'));t=z['time'][1:];metric=z['metrics'];labels=list(map(str,z['metric_names']));out=[]
for start,end in [(3.5,6.),(6.,8.),(8.,12.)]:
    mask=(t>=start)&(t<=end)
    if not mask.any():continue
    row={'window_s':[start,end]}
    for name in ['index_up_N','table_up_N','tilt_rad','actual_palm_target_error_m','filtered_index_up_N']:
        if name in labels:
            values=metric[mask,labels.index(name)];row[name]={'mean':float(values.mean()),'p50':float(np.median(values)),'p95':float(np.quantile(values,.95)),'max':float(values.max())}
    out.append(row)
contacts=[]
if'object_contacts'in z:
    c=z['object_contacts'];steps=c[:,0].astype(int);ct=z['time'][steps]
    for start,end in [(4.,6.),(6.,8.),(8.,12.)]:
        selected=(ct>=start)&(ct<=end);groups={};vectors={}
        for row in c[selected]:
            g1,g2=int(row[1]),int(row[2]);names=[m.body(int(m.geom_bodyid[g])).name for g in [g1,g2]];hand=next((name for name in names if name.startswith('rh_')),None)
            if hand is None:continue
            groups.setdefault(hand,[]).append(float(row[8]));vectors.setdefault(hand,[]).append(row[6:9])
        contacts.append({'window_s':[start,end],'parts':[{'body':name,'contact_samples':len(v),'mean_force_world_N':(np.sum(vectors[name],axis=0)*m.opt.timestep/(end-start)).tolist(),'sum_forceZ_impulse_Ns':float(np.sum(v)*m.opt.timestep),'forceZ_min_N':float(min(v)),'forceZ_max_N':float(max(v))}for name,v in groups.items()]})
finger=[]
for ai in range(m.nu):
    ji=int(m.actuator_trnid[ai,0]);name=m.joint(ji).name
    if not name.startswith('rh_'):continue
    qa=int(m.jnt_qposadr[ji]);finger.append({'joint':name,'gain':m.actuator_gainprm[ai,:3].tolist(),'bias':m.actuator_biasprm[ai,:3].tolist(),'q_initial':float(z['qpos'][0,qa]),'q_final':float(z['qpos'][-1,qa]),'last_target':float(z['ctrl'][-1,ai]),'max_force':float(np.max(np.abs(z['actuator_force'][:,ai]))),'max_abs_q_target_error':float(np.max(np.abs(z['qpos'][1:,qa]-z['ctrl'][:,ai])))})
oi=int(m.jnt_qposadr[int(m.body_jntadr[m.body('target_object_01').id])])
friction={m.geom(g).name:m.geom_friction[g].tolist()for g in range(m.ngeom)if m.geom(g).name.startswith(('table_support','target_object_01_bottom','target_object_01_wall_00','collision_rh_index'))}
report={'scope':'Actual recorded step forces/tracking only, not new simulation','trial':str(a.trial),'contact_material_friction':friction,'object1_reset_to_final_translation_m':(z['qpos'][-1,oi:oi+3]-z['qpos'][0,oi:oi+3]).tolist(),'robot_hand_actuators':finger,'windows':out,'part_contact_forceZ':contacts};a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'contact_material_friction':friction,'movement':report['object1_reset_to_final_translation_m'],'forces':contacts}),flush=True)
