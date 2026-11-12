"""Read-only dynamic-trace evidence for uncompensated arm gravity sag."""
from pathlib import Path
import argparse,json
import numpy as np
import mujoco as mj
from fourth_arm_object_feedback import ArmObjectFeedback
p=argparse.ArgumentParser();p.add_argument('--trial',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
r=json.loads((a.trial/'report.json').read_text());m=mj.MjModel.from_binary_path(r['model']['path']);z=np.load(a.trial/'physical_trace.npz');ref=dict(np.load(a.reference,allow_pickle=False));h=ArmObjectFeedback(m,ref);d=mj.MjData(m);v=mj.MjData(m)
history=np.load(a.trial/'sonic_contact_feedback.npz');times=history['arm_feedback_time'];phase=history['arm_feedback_phase'];rows=[[],[]]
for i,t in enumerate(z['time']):
    if not 3.3<=t<=6.7:continue
    hi=int(np.argmin(abs(times-t)));ctrl=z['ctrl'][min(i,len(z['ctrl'])-1)]
    d.qpos[:]=z['qpos'][i];d.qvel[:]=0;mj.mj_forward(m,d)
    v.qpos[:]=d.qpos;v.qvel[:]=0
    for c in h.chains:v.qpos[c['qa']]=ctrl[c['aids']]
    mj.mj_forward(m,v)
    for side,c in enumerate(h.chains):
        if phase[hi,side]=='idle':continue
        delta=ctrl[c['aids']]-d.qpos[c['qa']];gravity=d.qfrc_bias[c['va']].copy();kp=m.actuator_gainprm[c['aids'],0];estimate=gravity/kp
        rows[side].append({'time_s':float(t),'actual_palm_z_m':float(h.palm(d,side)[2]),'applied_target_fk_palm_z_m':float(h.palm(v,side)[2]),'palm_z_sag_m':float(h.palm(v,side)[2]-h.palm(d,side)[2]),'joint_target_minus_actual_rad':delta.tolist(),'static_gravity_torque_Nm':gravity.tolist(),'gravity_compensation_offset_rad':estimate.tolist(),'arm_actual_velocity_rad_s':z['qvel'][i,c['va']].tolist()})
out=[]
for side,rr in enumerate(rows):
    if not rr:out.append({'hand':side,'samples':0});continue
    sag=np.array([q['palm_z_sag_m']for q in rr]);err=np.array([q['joint_target_minus_actual_rad']for q in rr]);g=np.array([q['gravity_compensation_offset_rad']for q in rr]);vel=np.array([q['arm_actual_velocity_rad_s']for q in rr]);slow=np.sqrt(np.mean(vel**2,axis=1))<.7
    out.append({'hand':side,'samples':len(rr),'palm_sag_z_m_median':float(np.median(sag)),'palm_sag_z_m_p95':float(np.percentile(sag,95)),'joint_error_rad_rms':float(np.sqrt(np.mean(err**2))),'gravity_offset_rad_rms':float(np.sqrt(np.mean(g**2))),'joint_error_vs_gravity_offset_correlation':float(np.corrcoef(err.ravel(),g.ravel())[0,1]),'slow_arm_samples':int(sum(slow)),'slow_palm_sag_median_m':float(np.median(sag[slow]))if slow.any()else None,'example':rr[len(rr)//2]})
report={'scope':'Read-only pre-fault physical trace analysis; static gravity from frozen plant qvel=0, no simulation steps','model':r['model']['path'],'time_window_s':[3.3,6.7],'hands':out};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
