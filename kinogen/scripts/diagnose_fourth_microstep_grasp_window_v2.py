"""Inspect the frozen true grasp window without changing a running trial."""
from pathlib import Path
import json,numpy as np,mujoco as mj
root=Path(__file__).resolve().parents[1]
reports=[]
for case in ['cached','segments']:
    trial=root/f'outputs/fourth_full_sequence/physics_single_plate_microstep_v2_{case}'
    z=dict(np.load(trial/'physical_trace.npz'))
    h=dict(np.load(trial/'grasp_history.npz'))
    f=dict(np.load(trial/'finger_history.npz'))
    c=dict(np.load(trial/'physical_contacts.npz'))
    model=mj.MjModel.from_binary_path(str(trial/'effective_model.mjb'))
    data=mj.MjData(model)
    cfg=json.loads((trial/'finger_config.json').read_text())
    mingap=float(cfg.get('min_requested_gap_m',-.003));gain=float(cfg['force_to_gap_gain_m_per_N'])
    fields=list(map(str,h['columns']))
    names=['rh_thumb_cmc_yaw','rh_thumb_cmc_pitch','rh_index_mcp_pitch','rh_middle_mcp_pitch','rh_ring_mcp_pitch','rh_pinky_mcp_pitch']
    joints=np.array([model.joint(n).id for n in names]);qa=model.jnt_qposadr[joints]
    aids=np.array([model.actuator('motor_'+n).id for n in names])
    armj=[j for j in range(model.njnt)if(model.joint(j).name or'').startswith('right_')and not(model.joint(j).name or'').startswith(('right_hip','right_knee','right_ankle'))]
    armva=model.jnt_dofadr[armj]
    rows=[]
    for when in [12.,16.,22.,28.,34.,38.]:
        hi=int(np.argmin(abs(h['values'][:,0]-when)));metrics=dict(zip(fields,map(float,h['values'][hi])))
        ti=int(np.argmin(abs(z['time_s']-metrics['time_s'])));fi=int(np.argmin(abs(f['time_s']-(metrics['time_s']-.005))))
        gaps=f['constraint_gap_m'][fi,1];force=f['all_physical_digit_normal_force_N'][fi,1];target=f['target_force_per_digit_N_vector'][fi,1]
        desired=np.clip(gaps+(force-target)*gain,mingap,float(cfg.get('max_backoff_gap_m',.004)))
        false_open=(force+.05<target)&(desired>gaps+1e-6)
        step=ti-1;ci=np.flatnonzero(c['step']==step)
        data.qpos[:]=z['qpos'][step];data.qvel[:]=z['qvel'][step];mj.mj_forward(model,data)
        qr=np.zeros(model.nv);jp=np.zeros((3,model.nv));jr=np.zeros_like(jp)
        for k in ci:
            b1,b2=map(int,c['body'][k]);n1=model.body(b1).name or '';n2=model.body(b2).name or ''
            if n1.startswith(('lh_','rh_'))and n2.startswith('target_object_'):body,sgn=b1,-1.
            elif n2.startswith(('lh_','rh_'))and n1.startswith('target_object_'):body,sgn=b2,1.
            else:continue
            frame=c['frame'][k].reshape(3,3);forcew=sgn*(frame.T@c['force'][k,:3]);torquew=sgn*(frame.T@c['force'][k,3:])
            mj.mj_jac(model,data,jp,jr,c['point'][k],body)
            qr+=jp.T@forcew+jr.T@torquew
        rows.append({'time_s':metrics['time_s'],'metrics':metrics,'force_control':{'gap_m':gaps.tolist(),'actual_N':force.tolist(),'target_ramped_N':target.tolist(),'requested_gap_m':desired.tolist(),'force_increase_blocked_by_absolute_gap_floor':false_open.tolist(),'ramp_fraction':float(f['load_force_ramp_fraction'][fi,1]),'phase_code':int(f['phase_codes'][fi,1])},'actual_hand_masters':dict(zip(names,map(float,z['qpos'][ti,qa]))),'ctrl_hand_masters':dict(zip(names,map(float,z['ctrl'][step,aids]))),'hand_native_motor_tau_Nm':z['actuator_force'][step,aids].tolist(),'hand_master_distance_to_lower_rad':(z['qpos'][ti,qa]-model.jnt_range[joints,0]).tolist(),'hand_master_distance_to_upper_rad':(model.jnt_range[joints,1]-z['qpos'][ti,qa]).tolist(),'actual_hand_contact_generalized_reaction_RH_arm_Nm':qr[armva].tolist(),'arm_joint_names':[model.joint(j).name for j in armj],'table_guard_deferred':bool(f['table_guard_release_deferred'][fi,1]),'table_guard_clearance_m':float(f['table_guard_output_clearance_m'][fi,1])})
    blocked=[]
    mask=f['time_s']>=12
    gaps=f['constraint_gap_m'][mask,1];force=f['all_physical_digit_normal_force_N'][mask,1];target=f['target_force_per_digit_N_vector'][mask,1]
    desired=np.clip(gaps+(force-target)*gain,mingap,float(cfg.get('max_backoff_gap_m',.004)))
    blocked=((force+.05<target)&(desired>gaps+1e-6)).mean(axis=0)
    report={'case':case,'min_requested_gap_m':mingap,'gain_m_per_N':gain,'digit_order':['thumb','index','middle','ring','pinky'],'fraction_after12s_positive_force_error_reversed_to_open_by_gap_floor':blocked.tolist(),'samples':rows}
    reports.append(report)
out=root/'reports/single_plate_microstep_v2_actual_window_diagnosis.json'
out.write_text(json.dumps({'scope':'Read-only frozen actual traces. Servo and contact observations are not task-success claims.','trials':reports},indent=2)+'\n')
print(json.dumps([{'case':r['case'],'absolute_gap_floor_reverse_fraction':r['fraction_after12s_positive_force_error_reversed_to_open_by_gap_floor'],'last_force_control':r['samples'][-1]['force_control'],'last_palm_error_m':r['samples'][-1]['metrics']['palm_error_m']}for r in reports]),flush=True)
