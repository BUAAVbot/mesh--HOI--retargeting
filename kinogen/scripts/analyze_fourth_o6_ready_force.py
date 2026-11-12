"""Quantify ready dwell, contact-force oscillation and actual motor gains."""
from pathlib import Path
import argparse,json
import numpy as np
import mujoco as mj
p=argparse.ArgumentParser();p.add_argument('--trial',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=json.loads((a.trial/'report.json').read_text());m=mj.MjModel.from_binary_path(r['model']['path']);h=dict(np.load(a.trial/'sonic_contact_feedback.npz'));z=dict(np.load(a.trial/'physical_trace.npz'));c=dict(np.load(a.trial/'physical_contacts.npz'));t=h['arm_feedback_time'];e=h['arm_feedback_grasp_evidence'][:,1];ready=(e[:,0]>.5)&(e[:,2]>=e[:,3]);runs=[];start=None
for k,value in enumerate(ready):
    if value and start is None:start=k
    if start is not None and(not value or k==len(ready)-1):
        end=k if value else k-1;runs.append({'start_s':float(t[start]),'end_s':float(t[end]),'duration_s':float(t[end]-t[start]),'samples':end-start+1});start=None
dt=float(r['physics_dt_s']);target=int(z['object_body_ids'][0]);N=len(z['ctrl']);forces={digit:np.zeros(N)for digit in ['thumb','index','middle','ring','pinky']};up={digit:np.zeros(N)for digit in forces};counts={digit:np.zeros(N,bool)for digit in forces};ii=c['interval_index'];bb=c['body_ids'];gg=c['geom_ids'];names=c['geom_names'];bnames=c['body_names']
for row in np.flatnonzero((bb[:,0]==target)|(bb[:,1]==target)):
    obj2=bb[row,1]==target;other=int(bb[row,0]if obj2 else bb[row,1]);geom=int(gg[row,0]if obj2 else gg[row,1]);name=str(names[geom])
    if not str(bnames[other]).startswith('rh_'):continue
    digit=next((d for d in forces if d in name),None)
    if digit is None:continue
    k=int(ii[row]);forces[digit][k]+=float(c['normal_force_N'][row]);up[digit][k]+=float(c['force_world_on_geom2_N'][row,2])*(1 if obj2 else-1);counts[digit][k]=True
time=z['time'][1:];take=(time>=3.1)&(time<=6.);force_total=forces['thumb']+forces['index'];signal=force_total[take];freq=np.fft.rfftfreq(len(signal),dt);power=abs(np.fft.rfft(signal-signal.mean()))**2;eligible=(freq>=1)&(freq<=80);peak=float(freq[eligible][np.argmax(power[eligible])]);stats={}
for digit in forces:
    stats[digit]={'mean_normal_force_N':float(np.mean(forces[digit][take])),'max_normal_force_N':float(np.max(forces[digit][take])),'normal_force_std_N':float(np.std(forces[digit][take])),'mean_upward_force_N':float(np.mean(up[digit][take])),'contact_duration_s':float(np.sum(counts[digit][take])*dt)}
motor={}
for name in ['rh_thumb_cmc_yaw','rh_thumb_cmc_pitch','rh_index_mcp_pitch']:
    aid=mj.mj_name2id(m,mj.mjtObj.mjOBJ_ACTUATOR,'motor_'+name);kp=float(m.actuator_gainprm[aid,0]);kd=float(-m.actuator_biasprm[aid,2]);motor[name]={'kp_Nm_rad':kp,'kd_Nms_rad':kd,'kd_over_kp_s':kd/kp,'hard_forcerange':m.actuator_forcerange[aid].tolist()}
out={'scope':'Frozen data only, no rollout/model changes','maximum_ready_dwell_s':max((v['duration_s']for v in runs),default=0.),'top_ready_runs':sorted(runs,key=lambda v:v['duration_s'],reverse=True)[:10],'actual_controlled_force_mean_N':float(np.mean(signal)),'actual_controlled_force_std_N':float(np.std(signal)),'actual_controlled_force_oscillation_peak_Hz':peak,'by_digit_actual_physics':stats,'native_motor_contract':motor};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2),flush=True)
