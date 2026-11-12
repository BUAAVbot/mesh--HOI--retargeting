"""Private-FK contracts and feasible six-command Jacobians; not a rollout."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import mujoco as mj
from fourth_contact_feedback_o6_coupled import O6CoupledReferenceContactFeedback

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--model',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);a=p.parse_args()
m=mj.MjModel.from_xml_path(str(a.model));ref=dict(np.load(a.reference,allow_pickle=False))
properties=['body_mass','body_inertia','body_pos','body_quat','jnt_range','actuator_gainprm','actuator_biasprm','actuator_gear','eq_data']
original={k:getattr(m,k).copy()for k in properties}
hook=O6CoupledReferenceContactFeedback(m,ref,{'geometry_backend':'convex_sat_o6'})
if any(not np.array_equal(getattr(m,k),v)for k,v in original.items()):raise ValueError('Constructor mutated live model')
qa=[int(m.jnt_qposadr[mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,str(n))])for n in ref['dof_names']]
aid=[mj.mj_name2id(m,mj.mjtObj.mjOBJ_ACTUATOR,'motor_'+str(n))for n in ref['dof_names']]
oqa=[int(m.jnt_qposadr[mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,f'target_object_{int(i):02d}_free')])for i in ref['object_ids']]
checks=[];gradients=[]
for frame in [88,147,353,485,710]:
    hook.reset();d=mj.MjData(m);d.qpos[:7]=ref['qpos'][frame,:7];d.qpos[qa]=ref['qpos'][frame,7:]
    for i,q in enumerate(oqa):d.qpos[q:q+7]=ref['object_target_pose'][frame,i]
    for h in hook.hands:
        u=np.clip(d.qpos[h['root_qa']],h['independent_lower'],h['independent_upper']);d.qpos[h['qa']]=hook._expand(h,u)
    mj.mj_forward(m,d);ctrl=np.zeros(m.nu);ctrl[aid]=d.qpos[qa];ctrl[hook.passive_actuator_ids]=1.
    previous=None
    for tick in range(3):
        d.time=float(ref['time_s'][frame])+tick*.02
        saved={k:getattr(d,k).copy()for k in ['qpos','qvel','ctrl','qfrc_applied','xfrc_applied','actuator_force','qfrc_actuator']}
        packet=ctrl.copy();out=hook.apply(d,ctrl,float(ref['time_s'][frame]))
        immutable=all(np.array_equal(v,getattr(d,k))for k,v in saved.items())and np.array_equal(packet,ctrl)
        preserved=np.array_equal(out[hook.preserved_actuator_ids],ctrl[hook.preserved_actuator_ids])
        zero=bool(np.all(out[hook.passive_actuator_ids]==0))
        ranges=True;rates=True;coupled=True;torque=True
        for side,h in enumerate(hook.hands):
            rec=hook.coupled_records[-1]['hands'][side];native=rec['native_expanded_target_rad']
            coupled&=bool(np.max(abs(native-hook._expand(h,out[h['root_aids']])))<1e-13)
            ranges&=bool(np.all(native>=h['lower']-1e-9)and np.all(native<=h['upper']+1e-9))
            if previous is not None:rates&=bool(np.max(abs(native-previous[side]))<=.02000001)
            torque&=bool(rec['motor_torque_goal_feasible'])
        previous=[x['native_expanded_target_rad'].copy()for x in hook.coupled_records[-1]['hands']]
        checks.append({'frame':frame,'tick':tick,'state_and_input_unchanged':bool(immutable),'body_neck_unchanged':bool(preserved),
                       'passive_ctrl_zero':zero,'native_target_exactly_mimic_coupled':bool(coupled),
                       'native_ranges_ok':bool(ranges),'native_velocity_rate_ok':bool(rates),'active_motor_torque_goals_feasible':bool(torque)})
        if not all([immutable,preserved,zero,coupled,ranges,rates,torque]):raise ValueError(checks[-1])
    selected=np.argwhere(ref['contact_candidate_observed'][frame])
    if len(selected):
        side,oi=map(int,selected[0]);h=hook.hands[side];hook._kin.qpos[:]=d.qpos;mj.mj_forward(m,hook._kin)
        gap,J=hook._geometry(h,oi,True);Jc=J@h['coupling_matrix'];numeric=np.zeros_like(Jc);eps=1e-5
        native0=hook._kin.qpos[h['qa']].copy()
        for j in range(6):
            hook._kin.qpos[h['qa']]=native0+eps*h['coupling_matrix'][:,j];mj.mj_kinematics(m,hook._kin);mj.mj_comPos(m,hook._kin)
            plus,_=hook._geometry(h,oi)
            hook._kin.qpos[h['qa']]=native0-eps*h['coupling_matrix'][:,j];mj.mj_kinematics(m,hook._kin);mj.mj_comPos(m,hook._kin)
            minus,_=hook._geometry(h,oi);numeric[:,j]=(plus-minus)/(2*eps)
        hook._kin.qpos[h['qa']]=native0;mj.mj_kinematics(m,hook._kin);mj.mj_comPos(m,hook._kin)
        error=float(np.max(abs(Jc-numeric)))
        gradients.append({'frame':frame,'hand':side,'six_independent_columns_checked':True,'chain_rule_max_error_m_per_rad':error,'passed':error<.005})
probe=mj.MjData(m);probe.ctrl[hook.passive_actuator_ids]=1e6;mj.mj_forward(m,probe)
passive_force_zero=bool(np.all(probe.actuator_force[hook.passive_actuator_ids]==0))
hard_torque_caps=[]
for sign in [-1,1]:
    probe.ctrl[hook.active_finger_actuator_ids]=sign*1e6;mj.mj_forward(m,probe)
    hard_torque_caps.append(float(np.max(abs(probe.actuator_force[hook.active_finger_actuator_ids]))))
unchanged=all(np.array_equal(getattr(m,k),v)for k,v in original.items())
report={'schema':'fourth.o6_coupled_feedback_kinematic_smoke.v1','model':str(a.model),'model_sha256':hashlib.sha256(a.model.read_bytes()).hexdigest(),
        'contracts':checks,'coupled_chain_rule_gradient_tests':gradients,'live_model_arrays_unchanged':unchanged,
        'passive_actuator_force_zero_with_arbitrary_ctrl_probe':passive_force_zero,'helper_report':hook.report(),
        'actual_hard_active_motor_torque_caps_with_arbitrary_ctrl_Nm':hard_torque_caps,
        'physical_rollout_performed':False,'physical_contact_or_grasp_success_claimed':False,
        'status':'passed'if unchanged and passive_force_zero and max(hard_torque_caps)<=.400000001 and all(g['passed']for g in gradients)else'failed'}
a.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in report.items()if k!='helper_report'}),flush=True)
if report['status']!='passed':raise SystemExit(2)
