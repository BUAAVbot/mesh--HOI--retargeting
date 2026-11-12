"""Read-only contact and coupling diagnosis from frozen grasp recordings."""
from pathlib import Path
import argparse,json
import numpy as np
import mujoco as mj
p=argparse.ArgumentParser();p.add_argument('--trial',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--late-start',type=float,default=11.8);a=p.parse_args()
m=mj.MjModel.from_binary_path(str((a.trial/'effective_model.mjb').resolve()));z=dict(np.load(a.trial/'physical_trace.npz'));c=dict(np.load(a.trial/'physical_contacts.npz'));t=z['time_s']if'time_s'in z else z['time'];q=z['qpos'];faults=[]
for e in range(m.neq):
    if m.eq_type[e]!=mj.mjtEq.mjEQ_JOINT:continue
    child,parent=int(m.eq_obj1id[e]),int(m.eq_obj2id[e]);name=m.joint(child).name or''
    if parent<0 or not name.startswith(('rh_','lh_')):continue
    cq,pq=int(m.jnt_qposadr[child]),int(m.jnt_qposadr[parent]);err=q[:,cq]-m.qpos0[cq]-np.polynomial.polynomial.polyval(q[:,pq]-m.qpos0[pq],m.eq_data[e,:5]);k=int(np.argmax(abs(err)));ai=m.actuator('motor_'+m.joint(parent).name).id;cv,pv=int(m.jnt_dofadr[child]),int(m.jnt_dofadr[parent]);rows=[]
    for i in range(max(1,k-12),min(len(t),k+1)):
        rows.append({'time_s':float(t[i]),'mimic_error_rad':float(err[i]),'parent_q_rad':float(q[i,pq]),'child_q_rad':float(q[i,cq]),'parent_dq_rad_s':float(z['qvel'][i,pv]),'child_dq_rad_s':float(z['qvel'][i,cv]),'parent_ctrl':float(z['ctrl'][i-1,ai]),'parent_actual_force_Nm':float(z['actuator_force'][i-1,ai])})
    faults.append({'child':name,'parent':m.joint(parent).name,'max_error_rad':float(abs(err[k])),'peak_time_s':float(t[k]),'first_over_30mrad_s':float(t[np.flatnonzero(abs(err)>.03)[0]])if np.any(abs(err)>.03)else None,'near_peak':rows})
faults.sort(key=lambda r:-r['max_error_rad']);ct=t[np.minimum(c['step'].astype(int),len(t)-1)];contacts=[]
bnames=[m.body(i).name or'world'for i in range(m.nbody)];gnames=[m.geom(i).name or'unnamed'for i in range(m.ngeom)];handids=[i for i,name in enumerate(bnames)if name.startswith(('rh_','lh_'))];handmask=np.isin(c['body'][:,0],handids)|np.isin(c['body'][:,1],handids)
for start,end in [(0.,2.),(a.late_start,float(t[-1])+.001)]:
    select=(ct>=start)&(ct<=end)&handmask;groups={}
    for i in np.flatnonzero(select):
        b1,b2=map(int,c['body'][i]);n1,n2=bnames[b1],bnames[b2];hand=n1 if n1.startswith(('rh_','lh_'))else n2 if n2.startswith(('rh_','lh_'))else None
        if hand is None:continue
        g1,g2=map(int,c['geom'][i]);other=n2 if hand==n1 else n1;othergeom=gnames[g2 if hand==n1 else g1];key=(hand,other,othergeom)
        # World contact force acts on geom2. Report force on the non-hand.
        f=c['frame'][i].reshape(3,3).T@c['force'][i,:3];f=f if hand==n1 else-f
        row=groups.setdefault(key,{'hand':hand,'other':other,'other_geom':othergeom,'first_s':float(ct[i]),'last_s':float(ct[i]),'normal_force_peak_N':0.,'normal_impulse_Ns':0.,'world_impulse_Ns':np.zeros(3),'contact_count':0,'penetration_max_m':0.})
        row['last_s']=float(ct[i]);row['normal_force_peak_N']=max(row['normal_force_peak_N'],float(c['force'][i,0]));row['normal_impulse_Ns']+=float(c['force'][i,0])*m.opt.timestep;row['world_impulse_Ns']+=f*m.opt.timestep;row['contact_count']+=1;row['penetration_max_m']=max(row['penetration_max_m'],float(max(0,-c['distance'][i])))
    for row in groups.values():row['world_impulse_Ns']=row['world_impulse_Ns'].tolist()
    contacts.append({'window_s':[start,end],'groups':sorted(groups.values(),key=lambda r:-r['normal_impulse_Ns'])})
out={'scope':'Frozen recorded data only; no physical model/controller changes','trial':str(a.trial),'mimic_joints':faults,'hand_contacts':contacts};a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'output':str(a.output),'fault':faults[0],'contacts':contacts}),flush=True)
