"""Read-only coupling/contact/control diagnosis of the actual seed-fixed trial."""
from pathlib import Path
import argparse,json
import numpy as np
import mujoco as mj

R=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--trial',type=Path,default=R/'outputs/fourth_full_sequence/physics_sonic15225_o6_obj1_seedfix_v2_6s')
p.add_argument('--output',type=Path,default=R/'reports/fourth_o6_seedfix_coupling_contact_diagnosis.json');a=p.parse_args()
r=json.loads((a.trial/'report.json').read_text());m=mj.MjModel.from_binary_path(r['model']['path'])
z=dict(np.load(a.trial/'physical_trace.npz',allow_pickle=False));f=dict(np.load(a.trial/'sonic_contact_feedback.npz',allow_pickle=False));c=dict(np.load(a.trial/'physical_contacts.npz',allow_pickle=False))
time=z['time'];mimic=[]
for e in range(m.neq):
    if m.eq_type[e]!=mj.mjtEq.mjEQ_JOINT:continue
    child,parent=int(m.eq_obj1id[e]),int(m.eq_obj2id[e]);name=m.joint(child).name or''
    if not name.startswith(('lh_','rh_')):continue
    cq,pq=int(m.jnt_qposadr[child]),int(m.jnt_qposadr[parent]);res=z['qpos'][:,cq]-m.qpos0[cq]-np.polynomial.polynomial.polyval(z['qpos'][:,pq]-m.qpos0[pq],m.eq_data[e,:5])
    bad=np.flatnonzero(abs(res)>.03);peak=int(np.argmax(abs(res)))
    mimic.append({'joint':name,'first_error_above_unchanged_threshold_s':float(time[bad[0]])if len(bad)else None,'maximum_absolute_error_rad':float(abs(res[peak])),'peak_time_s':float(time[peak])})
first=min([x['first_error_above_unchanged_threshold_s']for x in mimic if x['first_error_above_unchanged_threshold_s']is not None],default=None)
rows=[]
if first is not None:
    interval=c['interval_index'];bt=list(map(str,c['body_names']));gn=list(map(str,c['geom_names']))
    select=(interval*m.opt.timestep>=first-.12)&(interval*m.opt.timestep<=first+.03)
    ishand=np.asarray([x.startswith('rh_')for x in bt])
    select&=ishand[c['body_ids'][:,0]]|ishand[c['body_ids'][:,1]]
    for i in np.flatnonzero(select):
        rows.append({'time_s':float(interval[i]*m.opt.timestep),'bodies':[bt[int(b)]for b in c['body_ids'][i]],'geoms':[gn[int(g)]for g in c['geom_ids'][i]],
                     'normal_force_N':float(c['normal_force_N'][i]),'signed_distance_m':float(c['signed_distance_m'][i])})
    rows=sorted(rows,key=lambda x:x['normal_force_N'],reverse=True)[:25]
tt=f['finger_feedback_time_s'];phase=f['finger_feedback_phase_codes'][:,1];active=(tt>=2.93)&(tt<=3.7)
report={'schema':'fourth.o6_seedfix_readonly_coupling_contacts.v1','trial':str(a.trial),'mimic_constraints':mimic,
        'first_fault_time_s':first,'largest_contacts_near_first_fault':rows,
        'active_command_min_rad':f['finger_feedback_independent_command_rad'][active,1].min(0).tolist(),
        'active_command_max_rad':f['finger_feedback_independent_command_rad'][active,1].max(0).tolist(),
        'active_constraint_gap_min_m':np.nanmin(f['finger_feedback_constraint_gap_m'][active,1],axis=0).tolist(),
        'table_guard_unsafe_during_acquire_or_grasp_calls':int(np.count_nonzero((phase==1)|(phase==2))and np.count_nonzero(np.isin(phase,[1,2])&(f['finger_feedback_table_guard_output_clearance_m'][:,1]<.002))),
        'table_guard_unresolved_calls':int(np.count_nonzero(f['finger_feedback_table_guard_no_safe_command_in_current_rate_bounds'][:,1])),
        'source_offset_slack_max_rad':float(np.max(f['finger_feedback_source_offset_budget_native_slack_rad'])),
        'actual_state_or_controller_modified':False}
a.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report),flush=True)
