"""Read-only actual support, IK, and tracking diagnosis around the first leg lift."""
import argparse,json
from pathlib import Path
import numpy as np
import mujoco as mj
p=argparse.ArgumentParser();p.add_argument('trial',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
m=mj.MjModel.from_binary_path(str(a.trial/'model.mjb'));d=mj.MjData(m);r=mj.MjData(m)
z=np.load(a.trial/'trace.npz');c=np.load(a.trial/'contacts.npz');leg=np.load(a.trial/'world_leg_feedback.npz');fb=np.load(a.trial/'root_acceleration_feedback.npz')
names=list(map(str,z['joint_names']));geom_names=list(map(str,c['geom_names']));body_names=list(map(str,c['body_names']));base=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'p1_base_link')
feet=[mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'p1_'+side+'_ankle_roll_link')for side in ['left','right']];rows=[]
for t in [8.,8.5,8.8,8.9,9.,9.2,9.4,9.6,10.]:
    s=int(np.argmin(abs(z['time']-t)));step=min(max(s-1,0),len(z['ctrl'])-1);t=float(z['time'][s]);d.qpos[:]=z['qpos'][s];d.qvel[:]=z['qvel'][s];r.qpos[:]=d.qpos
    for actor in range(2):
        q=np.asarray([np.interp(t,z['reference_time_s'],z['reference_qpos'][:,actor,j])for j in range(59)])
        adr=int(z['robot_root_qadr'][actor]);r.qpos[adr:adr+7]=q[:7];r.qpos[z['joint_qadr'][actor]]=q[7:]
    mj.mj_kinematics(m,d);mj.mj_comPos(m,d);mj.mj_kinematics(m,r);mj.mj_comPos(m,r)
    supports={'seat':0.,'right_foot':0.,'left_foot':0.,'shoe':0.};points=[]
    for ci in np.flatnonzero(c['interval_index']==step):
        gi=c['geom_ids'][ci];bi=c['body_ids'][ci];gn=[geom_names[g]for g in gi];bn=[body_names[b]for b in bi]
        for k in range(2):
            force=c['force_world_on_geom2_N'][ci]*(1 if k==1 else-1)
            other=1-k
            if 'bench_seat'in gn[other]and bn[k].startswith('p1_'):key='seat'
            elif any(x in gn[other]for x in ['ground','floor']):
                if bn[k].startswith('p1_right_ankle'):key='right_foot'
                elif bn[k].startswith('p1_left_ankle'):key='left_foot'
                elif bn[k]=='shoe':key='shoe'
                else:continue
            else:continue
            supports[key]+=float(force[2]);points.append(c['position_world_m'][ci].tolist())
    left=np.flatnonzero((leg['actor']==0)&(leg['side']=='left'));li=left[np.argmin(abs(leg['time_s'][left]-t))];fi=int(np.argmin(abs(fb['time_s']-t)))
    qref=z['reference_qpos'][:,0,7:];speed=np.diff(qref,axis=0)/np.diff(z['reference_time_s'])[:,None];frame=min(max(np.searchsorted(z['reference_time_s'],t)-1,0),len(speed)-1)
    legs=[j for j,n in enumerate(names)if n.startswith(('left_hip','left_knee','left_ankle','right_hip','right_knee','right_ankle'))]
    rows.append({'time_s':t,'root_orientation_error_rad':z['root_orientation_error_rad'][step].tolist(),'root_horizontal_error_m':z['root_horizontal_error_m'][step].tolist(),
        'shoe_error_m':float(z['shoe_reference_error_m'][step]),'actual_foot_xyz_m':[d.xpos[b].tolist()for b in feet],'reference_foot_xyz_m':[r.xpos[b].tolist()for b in feet],
        'p1_left_private_IK_solved_error_m':float(leg['solved_position_error_m'][li]),'support_vertical_force_N':supports,'actual_COM_world':d.subtree_com[base].tolist(),'reference_COM_world':r.subtree_com[base].tolist(),'support_contact_points':points,
        'source_leg_velocity_rad_s':{names[j]:float(speed[frame,j])for j in legs},
        'actual_leg_tracking_rad':{names[j]:float(d.qpos[z['joint_qadr'][0,j]]-r.qpos[z['joint_qadr'][0,j]])for j in legs},
        'root_feedback_error_local_rad':fb['root_error_local_rad'][fi,0].tolist(),'root_feedback_acceleration_desired':fb['root_acceleration_desired'][fi,:6].tolist(),'root_feedback_acceleration_predicted':fb['root_acceleration_linearized'][fi,:6].tolist()})
out={'trial':str(a.trial),'feet_order':['p1left','p1right'],'samples':rows};a.output.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps([{k:v for k,v in row.items()if k not in ['source_leg_velocity_rad_s','actual_leg_tracking_rad','support_contact_points']}for row in rows],indent=2))
