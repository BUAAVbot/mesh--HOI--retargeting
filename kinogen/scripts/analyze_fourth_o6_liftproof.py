"""Read actual arm/force/command evidence from a frozen O6 lift trial."""
from pathlib import Path
import argparse,json
import numpy as np
import mujoco as mj
def finite_columns(values,stat):
    return [float(stat(v[np.isfinite(v)]))if np.isfinite(v).any()else None for v in values.T]
p=argparse.ArgumentParser();p.add_argument('--trial',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=json.loads((a.trial/'report.json').read_text());ff=json.loads((a.trial/'sonic_contact_feedback.json').read_text());h=np.load(a.trial/'sonic_contact_feedback.npz');z=np.load(a.trial/'physical_trace.npz');m=mj.MjModel.from_binary_path(r['model']['path']);t=h['arm_feedback_time'];phase=h['arm_feedback_phase'][:,1];ev=h['arm_feedback_grasp_evidence'][:,1];take=(t>=2.94)&(t<=6.);real=h['arm_feedback_actual_palm'][:,1];geom=h['arm_feedback_geometric_palm'][:,1];goal=h['arm_feedback_goal'][:,1];ft=h['finger_feedback_time_s'];fp=h['finger_feedback_phase_codes'][:,1];dist=h['finger_feedback_constraint_gap_m'][:,1];commands=h['finger_feedback_independent_command_rad'][:,1];expanded=h['finger_feedback_native_expanded_target_rad'][:,1];ri=[mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,n)for n in ff['finger']['o6_coupling']['hands'][1]['independent_names']];qa=m.jnt_qposadr[ri];va=m.jnt_dofadr[ri];ii=np.minimum(np.searchsorted(z['time'],ft),len(z['time'])-1);actual=z['qpos'][ii][:,qa];err=commands-actual;initial=commands[0];active=ft>=2.94
phaseinfo=[]
ready=(ev[:,0]>.5)&(ev[:,2]>=ev[:,3]);longest=0.;started=None
for k,yes in enumerate(ready):
    if yes:
        if started is None:started=k
        longest=max(longest,float(t[k]-t[started]))
    else:started=None
for code in np.unique(fp):
    tx=ft[fp==code];phaseinfo.append({'code':int(code),'first_s':float(tx[0]),'last_s':float(tx[-1]),'calls':len(tx)})
oqa=int(z['object_qadr'][0]);oi=int(z['object_body_ids'][0]);mass=float(m.body_mass[oi]);w=mass*np.linalg.norm(m.opt.gravity);baseline=int(np.argmin(abs(z['time']-2.8)));rise=z['qpos'][:,oqa+2]-z['qpos'][baseline,oqa+2]
out={'scope':'Frozen actual physics/actuator logs; diagnostic, not full sequence success','trial':str(a.trial),'mass_kg':mass,'weight_N':float(w),'actual_object_z_rise_after2p8s_max_m':float(rise[baseline:].max()),'arm_phases':{str(v):int(np.sum(phase==v))for v in np.unique(phase)},'lift_confirmed_calls':int(np.sum(h['arm_feedback_lift_confirmed'])),'arm_goal_error_active_median_m':float(np.median(np.linalg.norm(goal[take]-real[take],axis=1))),'arm_geometric_z_minus_actual_active_median_m':float(np.median(geom[take,2]-real[take,2])),'grasp_evidence_columns':ff['arm']['grasp_evidence_columns'],'grasp_evidence_active_min':ev[take].min(axis=0).tolist(),'grasp_evidence_active_max':ev[take].max(axis=0).tolist(),'grasp_evidence_active_median':np.median(ev[take],axis=0).tolist(),'finger_phase':phaseinfo,'finger_events':ff['finger']['events'],'finger_independent_names':ff['finger']['o6_coupling']['hands'][1]['independent_names'],'finger_command_initial_rad':initial.tolist(),'finger_command_active_min_rad':commands[active].min(axis=0).tolist(),'finger_command_active_max_rad':commands[active].max(axis=0).tolist(),'finger_command_minus_actual_active_median_rad':np.median(err[active],axis=0).tolist(),'finger_native_gap_active_min_m':finite_columns(dist[active],np.min),'finger_native_gap_active_median_m':finite_columns(dist[active],np.median),'coupling_report':ff['finger']['o6_coupling'],'table_guard':ff['finger']['table_clearance_guard'],'checks':r['checks']};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps(out,indent=2,allow_nan=False),flush=True)
