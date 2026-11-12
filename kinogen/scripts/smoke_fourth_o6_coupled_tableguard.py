"""Private coupled-hand release fixture; no physical rollout or lift claim."""
from pathlib import Path
import argparse,json
import numpy as np
import mujoco as mj
from fourth_contact_feedback_o6_coupled import O6CoupledReferenceContactFeedback

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--model',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);a=p.parse_args()
m=mj.MjModel.from_xml_path(str(a.model));ref=dict(np.load(a.reference,allow_pickle=False))
hook=O6CoupledReferenceContactFeedback(m,ref,{'enabled':False,'geometry_backend':'convex_sat_o6'})
d=mj.MjData(m);n=hook._plane[:3];u=np.asarray([1.,0.,-n[0]/n[2]]);u/=np.linalg.norm(u)
desired=np.column_stack([u,np.cross(-n,u),-n])
for h in hook.hands:
    roots=np.asarray([.6,.4,.6,.6,.6,.6]);d.qpos[h['qa']]=hook._expand(h,roots)
mj.mj_kinematics(m,d);base=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'rh_hand_base_link')
rotation=desired@d.xmat[base].reshape(3,3).T;quat=np.empty(4);mj.mju_mat2Quat(quat,rotation.reshape(-1));d.qpos[3:7]=quat
mj.mj_kinematics(m,d);palm=mj.mj_name2id(m,mj.mjtObj.mjOBJ_SITE,'rh_palm_control_site')
centre=np.asarray([-.57,1.8,0.]);centre[2]=(-hook._plane[3]-n[:2]@centre[:2])/n[2]
d.qpos[:3]+=centre+.10*n-d.site_xpos[palm];mj.mj_forward(m,d)
ctrl=np.zeros(m.nu)
for h in hook.hands:ctrl[h['root_aids']]=d.qpos[h['root_qa']]
previous=hook.apply(d,ctrl,0.);hand=hook.hands[1]
minimum=1.;immutable=True;preserved=True;zero=True;coupled=True;range_ok=True;rate_ok=True;previous_native=None;maximum_estimated=0.
rate_violations=[]
for step in range(1,221):
    if step==121:d.qpos[:3]+=.05*n
    d.time=step*.02
    for h in hook.hands:d.qpos[h['qa']]=hook._expand(h,previous[h['root_aids']])
    mj.mj_forward(m,d);request=ctrl.copy();request[hand['root_aids']]=hand['independent_lower']
    saved={k:getattr(d,k).copy()for k in ['qpos','qvel','ctrl','qfrc_applied','xfrc_applied']}
    out=hook.apply(d,request,d.time);record=hook.coupled_records[-1]['hands'][1]
    native=record['native_expanded_target_rad'];distance,ff,unknown=hook._clearance(d,1,native)
    if step<=120:minimum=min(minimum,distance)
    immutable&=all(np.array_equal(v,getattr(d,k))for k,v in saved.items())
    preserved&=np.array_equal(out[hook.preserved_actuator_ids],request[hook.preserved_actuator_ids])
    zero&=bool(np.all(out[hook.passive_actuator_ids]==0))
    coupled&=bool(np.max(abs(native-hook._expand(hand,out[hand['root_aids']])))<1e-13)
    range_ok&=bool(np.all(native>=hand['lower']-1e-9)and np.all(native<=hand['upper']+1e-9))
    if previous_native is not None:
        differences=abs(native-previous_native);rate_ok&=bool(np.max(differences)<=.02000001)
        if np.max(differences)>.02000001:
            rate_violations.append({'step':step,'time_s':d.time,'maximum_native_delta_rad':float(np.max(differences)),
                'joint':hand['names'][int(np.argmax(differences))],'previous_native':previous_native.tolist(),
                'new_native':native.tolist(),'torque_goal_feasible':record['motor_torque_goal_feasible']})
    maximum_estimated=max(maximum_estimated,float(np.max(abs(record['estimated_independent_motor_torque_Nm']))))
    previous_native=native.copy();previous=out
    if step==120:held=native.copy()
final=hook.coupled_records[-1]['hands'][1]['native_expanded_target_rad'];released=bool(np.max(abs(final-hand['lower']))<.005)
report={'schema':'fourth.o6_coupled_tableguard_private_fixture.v1','physical_rollout_performed':False,
        'fixture':'Coupled closed hand at10cm palm height, request open, then manually raise private FK fixture5cm and continue opening.',
        'actual_test_state_unchanged_by_hook':immutable,'body_neck_ctrl_preserved':preserved,'passive_ctrl_zero':zero,
        'all_native_targets_mimic_consistent':coupled,'native_range_ok':range_ok,'native_rate_ok':rate_ok,
        'minimum_right_hand_clearance_before_private_fixture_raise_m':minimum,
        'held_native_targets_rad':held.tolist(),'final_native_targets_rad':final.tolist(),
        'released_after_private_fixture_raise':released,'maximum_estimated_active_motor_torque_Nm':maximum_estimated,
        'native_rate_violations':rate_violations,
        'helper_report':hook.report()}
report['status']='passed'if all([immutable,preserved,zero,coupled,range_ok,rate_ok,released,minimum>=.0018,maximum_estimated<=.400001])else'failed'
a.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in report.items()if k!='helper_report'}),flush=True)
if report['status']!='passed':raise SystemExit(2)
