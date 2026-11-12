"""Private vertical-palm contact sweep from a recorded safe raw-scene hold."""
from pathlib import Path
import argparse,json
import numpy as np
import mujoco as mj
from fourth_arm_object_feedback import ArmObjectFeedback
p=argparse.ArgumentParser();p.add_argument('--trial',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--inward-m',type=float,default=0.);p.add_argument('--goal-from-initial',action='store_true');a=p.parse_args();report=json.loads((a.trial/'report.json').read_text());m=mj.MjModel.from_binary_path(report['model']['path']);z=dict(np.load(a.trial/'physical_trace.npz'));ref=dict(np.load(a.reference));d=mj.MjData(m);k=int(np.argmin(abs(z['time']-1.)));seed=z['qpos'][k].copy();d.qpos[:]=seed;d.qvel[:]=0;mj.mj_forward(m,d);h=ArmObjectFeedback(m,ref,{'initial_geometric_seed':'actual_state','ik_iterations':25,'orientation_weight_m':.10,'arm_speed_rad_s':3.});palm=h.palm(d,1);R=d.xmat[h.hand_bids[1]].reshape(3,3).copy();chain=h.chains[1];obj=h.object_bids[0];base=np.zeros(m.nu);names=list(map(str,ref['dof_names']));aids=[m.actuator('motor_'+n).id for n in names];base[aids]=ref['qpos'][0,7:];rows=[]
if a.goal_from_initial:
    d.qpos[:]=z['qpos'][0];mj.mj_kinematics(m,d);palm=h.palm(d,1);R=d.xmat[h.hand_bids[1]].reshape(3,3).copy();d.qpos[:]=seed;mj.mj_kinematics(m,d)
outward=np.array([.35216658756772096,.9359373347617386,-.02508]);outward/=np.linalg.norm(outward);palm=palm-a.inward_m*outward
for rise in np.arange(0,.0321,.002):
    target=palm+np.array([0,0,rise]);q,error=h.solve(d,1,target,R,base);h.last_targets[1]=q.copy();d.qpos[chain['qa']]=q;mj.mj_kinematics(m,d);mj.mj_comPos(m,d);mj.mj_collision(m,d);entries=[];table=[]
    for c in d.contact[:d.ncon]:
        g1,g2=map(int,c.geom);b1,b2=map(int,m.geom_bodyid[[g1,g2]])
        if obj in(b1,b2):
            obj2=b2==obj;hg=g1 if obj2 else g2;hb=b1 if obj2 else b2;name=m.body(hb).name or''
            if name.startswith(('rh_','right_')):
                n=np.asarray(c.frame).reshape(3,3)[0]*(1 if obj2 else-1);entries.append({'hand_geom':m.geom(hg).name,'normal_on_object':n.tolist(),'upward_normal':float(n[2]),'normal_xy_over_up':float(np.linalg.norm(n[:2])/max(n[2],1e-9)),'signed_gap_m':float(c.dist)})
        if{g1,g2}&set(h.table_geoms):
            other=g2 if g1 in h.table_geoms else g1;b=m.body(int(m.geom_bodyid[other])).name or'world'
            if b!='world'and not b.startswith('target_object')and c.dist<0:table.append({'body':b,'penetration_m':float(-c.dist)})
    row={'requested_rise_m':float(rise),'geometric_rise_m':float(h.palm(d,1)[2]-palm[2]),'palm_error_m':float(error),'contacts':entries,'robot_table_penetration_max_m':max((v['penetration_m']for v in table),default=0),'qpos':d.qpos.tolist()};rows.append(row);print(json.dumps({k:v for k,v in row.items()if k!='qpos'}),flush=True)
    if entries and min(v['signed_gap_m']for v in entries)<-.002:break
out={'scope':'Private FK/collision probe only; all raw object states come from actual recorded no-contact hold at1s and stay fixed. No actual lifting or forces claimed.','model':report['model']['path'],'source_trial':str(a.trial),'inward_offset_m':a.inward_m,'goal_from_initial_reference':a.goal_from_initial,'samples':rows};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n')
