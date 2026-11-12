"""Independent continuous-push prefix and subsequent reference-transition audit."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import mujoco as mj
p=argparse.ArgumentParser();p.add_argument('--trial',type=Path,required=True);p.add_argument('--prefix',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
z=np.load(a.trial/'physical_trace.npz');b=np.load(a.prefix/'physical_trace.npz');n=len(b['ctrl']);keys=['qpos','qvel','ctrl','actuator_force','qfrc_applied','xfrc_applied'];prefix={}
for key in keys:
    size=len(b[key]);prefix[key]=float(np.max(abs(z[key][:size]-b[key])))
t=z['time_s'];tb=b['time_s'];prefix['time_s']=float(np.max(abs(t[:len(tb)]-tb)));dt=np.diff(t);m=mj.MjModel.from_binary_path(str((a.trial/'effective_model.mjb').resolve()));ref=np.load(a.trial/'adapted_robot_reference.npz');rt=ref['time_s'];rq=ref['qpos'];where=np.flatnonzero(rt>=tb[-1]-1e-8);start=int(where[0]);waist=[j for j in range(m.njnt)if(m.joint(j).name or'').startswith('waist')];legs=[j for j in range(m.njnt)if(m.joint(j).name or'').startswith(('left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle'))];end=min(len(rt)-1,int(np.argmin(abs(rt-(tb[-1]+5)))));waistrows=[]
for j in waist:
    adr=int(m.jnt_qposadr[j]);waistrows.append({'joint':m.joint(j).name,'target_start_rad':float(rq[start,adr]),'target_end_rad':float(rq[end,adr]),'max_target_speed_rad_s':float(np.max(abs(np.diff(rq[start:end+1,adr])/np.diff(rt[start:end+1])))),'actual_at_prefix_rad':float(z['qpos'][n,adr]),'actual_final_rad':float(z['qpos'][-1,adr])})
legrows=[{'joint':m.joint(j).name,'reference_5s_change_rad':float(rq[end,m.jnt_qposadr[j]]-rq[start,m.jnt_qposadr[j]])}for j in legs];h=np.load(a.trial/'grasp_history.npz');hv=h['values'];col=list(map(str,h['columns']));phase=h['phase'];faults=[]
for kind,name,op,bound in [('root_drift','root_drift_m',np.greater,.2),('root_up','root_up',np.less,.8),('mimic','mimic_error_rad',np.greater,.03)]:
    selected=np.flatnonzero(op(hv[:,col.index(name)],bound))
    if len(selected):i=int(selected[0]);faults.append({'criterion':kind,'time_s':float(hv[i,0]),'phase':str(phase[i]),'value':float(hv[i,col.index(name)]),'threshold':bound})
faults.sort(key=lambda r:r['time_s']);out={'schema':'fourth.C_continuous_prefix_transition_audit.v1','scope':'Read-only frozen traces and target references; control-only deterministic replay is recorded separately. No reset at push-to-grasp boundary permitted.','trial':str(a.trial),'prefix_trial':str(a.prefix),'prefix_steps':n,'prefix_max_abs_errors':prefix,'prefix_all_exactly_equal':all(v==0 for v in prefix.values()),'timeline_monotonic':bool(np.all(dt>0)),'timeline_step_max_error_s':float(np.max(abs(dt-m.opt.timestep))),'boundary_time_s':float(t[n]),'boundary_previous_time_s':float(t[n-1]),'boundary_next_time_s':float(t[n+1]),'waist_target_transition':waistrows,'leg_target_changes':legrows,'max_leg_reference_change_rad':max(abs(r['reference_5s_change_rad'])for r in legrows),'root_quat_reference_change_norm':float(np.linalg.norm(rq[end,3:7]-rq[start,3:7])),'first_protection_criteria':faults,'phase_counts':{str(s):int(np.sum(phase==s))for s in np.unique(phase)},'last_continuation_metrics':dict(zip(col,map(float,hv[-1])))};a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
