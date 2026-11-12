"""Read-only initial hybrid-target and contact timing diagnosis."""
from pathlib import Path
import argparse,json
import numpy as np
import mujoco as mj
p=argparse.ArgumentParser();p.add_argument('--trial',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=json.loads((a.trial/'report.json').read_text());m=mj.MjModel.from_binary_path(r['model']['path']);z=np.load(a.trial/'physical_trace.npz');h=np.load(a.trial/'sonic_contact_feedback.npz');c=np.load(a.trial/'physical_contacts.npz');names=list(r['sonic']['body_joint_names']);arm=[n for n in names if n.startswith(('right_shoulder','right_elbow','right_wrist'))];jids=[mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,n)for n in arm];qa=m.jnt_qposadr[jids];idx=[names.index(n)for n in arm];q0=z['qpos'][0,qa];learned=h['hybrid_policy_learned_body_target'][0,idx];time=h['arm_feedback_time'];rows=[]
for target in [0,.02,.04,.08,.12,.2,.26,.28,.4,.6]:
    k=int(np.argmin(abs(time-target)));goal=h['arm_feedback_goal'][k,1];actual=h['arm_feedback_actual_palm'][k,1];geometric=h['arm_feedback_geometric_palm'][k,1];joint=h['arm_feedback_geometric_joint_targets'][k,1];physical=int(np.argmin(abs(z['time']-time[k])));difference=joint-z['qpos'][physical,qa]
    rows.append({'time_s':float(time[k]),'actual_palm_goal_error_m':float(np.linalg.norm(actual-goal)),'geometric_palm_goal_error_m':float(np.linalg.norm(geometric-goal)),'geometric_joint_minus_actual_rad':difference.tolist(),'actual_palm':actual.tolist(),'geometric_palm':geometric.tolist(),'goal':goal.tolist()})
targetbody=int(z['object_body_ids'][0]);bodies=c['body_ids'];gnames=c['geom_names'];bnames=c['body_names'];interval=c['interval_index'];thumb=[];allhand=[]
for row in np.flatnonzero((bodies[:,0]==targetbody)|(bodies[:,1]==targetbody)):
    g1,g2=c['geom_ids'][row];b1,b2=bodies[row];hg=int(g2 if b1==targetbody else g1);hb=int(b2 if b1==targetbody else b1)
    if not str(bnames[hb]).startswith('rh_'):continue
    entry={'time_s':float(z['time'][min(int(interval[row])+1,len(z['time'])-1)]),'normal_force_N':float(c['normal_force_N'][row]),'hand_geom':str(gnames[hg]),'penetration_m':float(max(-c['signed_distance_m'][row],0))}
    if entry['time_s']<=.4:allhand.append(entry)
    if'thumb'in str(gnames[hg]):thumb.append(entry)
out={'scope':'Frozen logs only, no physics stepping/model changes','trial':str(a.trial),'right_arm_names':arm,'first_learned_position_motor_target_minus_actual_q_rad':(learned-q0).tolist(),'initial_geometric_ik_seed_in_current_code':'policy motor target when last_targets is None (contains learned offsets); target rate-limited relative to that seed','initialization_samples':rows,'first_thumb_contact_s':min((v['time_s']for v in thumb if v['normal_force_N']>.01),default=None),'maximum_early_hand_object_contacts':sorted(allhand,key=lambda x:x['normal_force_N'],reverse=True)[:10],'finger_commands_change_before_fault_rad':np.max(abs(h['finger_feedback_independent_command_rad'][h['finger_feedback_time_s']<=.265,1]-h['finger_feedback_independent_command_rad'][0,1]),axis=0).tolist()};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2),flush=True)
