"""Compare the physical gravity-feedforward A/B trials and contact plans."""
from pathlib import Path
import argparse,json
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--off',type=Path,required=True);p.add_argument('--on',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();result={}
for name,path in [('off',a.off),('on',a.on)]:
    report=json.loads((path/'report.json').read_text());h=np.load(path/'sonic_contact_feedback.npz');feedback=json.loads((path/'sonic_contact_feedback.json').read_text());t=h['arm_feedback_time'];phase=h['arm_feedback_phase'];ids=h['arm_feedback_ids'];actual=h['arm_feedback_actual_palm'];geometric=h['arm_feedback_geometric_palm'];goal=h['arm_feedback_goal'];off=h['arm_feedback_actuator_gravity_offset_rad'];force=h['arm_feedback_force'];kind=h['arm_feedback_contact_kind'];hands=[]
    for side in range(2):
        mask=(t>=3.3)&(t<=6.7)&(phase[:,side]!='idle');sag=geometric[mask,side,2]-actual[mask,side,2];err=np.linalg.norm(goal[mask,side]-actual[mask,side],axis=1)
        events=[]
        for oi in np.unique(ids[:,side]):
            if oi<0:continue
            choose=ids[:,side]==oi
            for ph in np.unique(phase[choose,side]):
                take=choose&(phase[:,side]==ph);tx=t[take];events.append({'object_id':int(oi),'phase':str(ph),'first_s':float(tx[0]),'last_s':float(tx[-1]),'records':int(take.sum()),'force_max_N':float(np.max(force[take,side])),'grasp_kinds':{str(k):int(np.sum(kind[take,side]==k))for k in np.unique(kind[take,side])}})
        hands.append({'side':side,'active_3p3_6p7_samples':int(mask.sum()),'geometric_palm_z_minus_actual_m_median':float(np.median(sag))if len(sag)else None,'actual_palm_to_goal_error_m_median':float(np.median(err))if len(err)else None,'gravity_offset_max_rad':float(np.max(abs(off[:,side]))),'events':events})
    ftime=h['finger_feedback_time_s'];selected=h['finger_feedback_selected_object_ids'];finger_offsets=h['finger_feedback_finger_offsets_rad'];distances=h['finger_feedback_selected_tip_distance_m'];finger_stats=[]
    for side in range(2):
        mask=(selected[:,side]==1)&(ftime>=2.93)&(ftime<=6.55)
        if not mask.any():continue
        cap=float(feedback['finger']['configuration']['max_offset_from_reference_rad']);v=finger_offsets[mask,side];dd=distances[mask,side]
        finger_stats.append({'side':side,'object_id':1,'samples':int(mask.sum()),'digit_order':['thumb','index','middle','ring','pinky'],'distance_m_min_by_digit':np.nanmin(dd,axis=0).tolist(),'distance_m_median_by_digit':np.nanmedian(dd,axis=0).tolist(),'offset_max_abs_rad_by_native_joint':np.max(abs(v),axis=0).tolist(),'offset_saturated_fraction_by_native_joint':np.mean(abs(v)>=cap-1e-5,axis=0).tolist(),'configured_offset_cap_rad':cap,'target_force_per_digit_N_median':float(np.median(h['finger_feedback_target_normal_force_per_digit_N'][mask,side])),'required_total_normal_force_N_median':float(np.median(h['finger_feedback_gravity_friction_required_total_normal_force_N'][mask,side])),'native_joint_order':[q['name']for q in feedback['finger']['joint_contracts'][side]]})
    result[name]={'checks':report['checks'],'root_horizontal_error_max_m':report['root_horizontal_error_max_m'],'max_penetration_m':report['max_penetration_m'],'object_metrics':report['object_metrics'],'handplans':hands,'finger_events':feedback['finger']['events'],'finger_object1_stats':finger_stats,'finger_torque_guard_infeasible_calls':feedback['finger']['infeasible_estimated_motor_torque_guards'],'table_guard':feedback['finger'].get('table_clearance_guard')}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({n:{k:v[k]for k in ['root_horizontal_error_max_m','max_penetration_m','finger_object1_stats','finger_torque_guard_infeasible_calls']}for n,v in result.items()},indent=2),flush=True)
