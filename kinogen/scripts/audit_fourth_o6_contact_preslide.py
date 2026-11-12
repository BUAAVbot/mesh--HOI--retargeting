"""Replay a true-contact pre-slide diagnostic and locate first mechanical fault."""
from pathlib import Path
import argparse,json
import numpy as np
import mujoco as mj
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trial',type=Path,required=True);a=p.parse_args();r=json.loads((a.trial/'report.json').read_text());m=mj.MjModel.from_binary_path(r['model']['path']);z=dict(np.load(a.trial/'physical_trace.npz'));c=dict(np.load(a.trial/'physical_contacts.npz'));d=mj.MjData(m);mj.mj_setState(m,d,z['initial_state'],int(z['state_spec']));mj.mj_forward(m,d);errors=[]
for step,ctrl in enumerate(z['ctrl']):
    d.ctrl[:]=ctrl;mj.mj_step(m,d);errors.append([float(np.max(abs(d.qpos-z['qpos'][step+1]))),float(np.max(abs(d.qvel-z['qvel'][step+1])))])
faults=[]
for e in range(m.neq):
    if m.eq_type[e]!=mj.mjtEq.mjEQ_JOINT:continue
    child,parent=int(m.eq_obj1id[e]),int(m.eq_obj2id[e]);res=z['qpos'][:,int(m.jnt_qposadr[child])]-m.eq_data[e,1]*z['qpos'][:,int(m.jnt_qposadr[parent])];bad=np.flatnonzero(abs(res)>.03)
    faults.append({'child':m.joint(child).name,'max_abs_error_rad':float(max(abs(res))),'first_threshold_time_s':float(z['time_s'][bad[0]])if len(bad)else None})
bn=[m.body(i).name or'world'for i in range(m.nbody)];obj=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'target_object_01');impulses={};early={};dt=float(m.opt.timestep)
for k,bs in enumerate(c['body']):
    if obj not in bs:continue
    other=int(bs[1])if bs[0]==obj else int(bs[0]);name=bn[other]
    if not name.startswith(('rh_','lh_')):continue
    fn=max(float(c['force'][k,0]),0);impulses[name]=impulses.get(name,0)+fn*dt
    if c['step'][k]*dt<.5:early[name]=early.get(name,0)+fn*dt
report={'schema':'fourth.o6_contact_preslide_independent_replay_audit.v1','trial':str(a.trial),'qpos_replay_max_error':float(np.max(np.asarray(errors)[:,0])),'qvel_replay_max_error':float(np.max(np.asarray(errors)[:,1])),
        'native_mimic_actual_residuals':faults,'hand_object1_normal_impulse_Ns_by_body':impulses,'first_0_5s_object1_hand_impulse_Ns_by_body':early,
        'no_external_force':bool(np.all(z['qfrc_applied']==0)and np.all(z['xfrc_applied']==0)),'physical_preslide_success':r['success'],'post_reset_state_override_claim':r['post_reset_state_overrides']}
(a.trial/'independent_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
