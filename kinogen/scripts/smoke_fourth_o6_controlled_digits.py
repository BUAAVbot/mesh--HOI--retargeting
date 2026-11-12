"""Private mask contracts and default-all shadow regression, not a rollout."""
from pathlib import Path
import argparse,importlib.util,json
import numpy as np
import mujoco as mj
from fourth_contact_feedback_o6_coupled import O6CoupledReferenceContactFeedback

R=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--trial',type=Path,default=R/'outputs/fourth_full_sequence/physics_sonic15225_o6_obj1_seedfix_v2_6s')
p.add_argument('--output',type=Path,default=R/'reports/fourth_o6_controlled_digits_smoke.json')
p.add_argument('--shadow-duration-s',type=float,default=4.);a=p.parse_args()
record=json.loads((a.trial/'report.json').read_text());m=mj.MjModel.from_binary_path(record['model']['path'])
ref=Path(record['reference']['path']);cfg=json.loads((a.trial/'source_sonic_config.json').read_text()).get('finger_feedback',{})
candidate=json.loads((R/'reports/fourth_o6_static_grasp_refined_v1.json').read_text())['best'];d=mj.MjData(m);d.qpos[:]=candidate['qpos']
hook=O6CoupledReferenceContactFeedback(m,ref,cfg);baseline=np.zeros(m.nu)
for j in range(m.njnt):
    name=m.joint(j).name;aid=mj.mj_name2id(m,mj.mjtObj.mjOBJ_ACTUATOR,'motor_'+str(name))
    if aid>=0:baseline[aid]=d.qpos[int(m.jnt_qposadr[j])]
baseline[hook.passive_actuator_ids]=0.;d.ctrl[:]=baseline;mj.mj_forward(m,d)
plan={'active_object_ids_by_hand':[-1,1],'phase_by_hand':['idle','acquire'],
      'controlled_digits_by_hand':[['thumb','index','middle','ring','pinky'],['thumb','index']]}
contracts=[];previous=None
for step in range(51):
    d.time=step*.02
    saved={k:getattr(d,k).copy()for k in ['qpos','qvel','ctrl','qfrc_applied','xfrc_applied']}
    out=hook.apply(d,baseline,2.94,plan);right=hook.hands[1];h=hook.coupled_records[-1]['hands'][1]
    zero=bool(np.all(out[hook.passive_actuator_ids]==0));immutable=all(np.array_equal(v,getattr(d,k))for k,v in saved.items())
    inactive=np.asarray([3,4,5]);retained=bool(np.max(abs(out[right['root_aids'][inactive]]-baseline[right['root_aids'][inactive]]))<1e-12)
    native=h['native_expanded_target_rad'];rate=True if previous is None else bool(np.max(abs(native-previous))<=.020000001)
    coupled=bool(np.max(abs(native-hook._expand(right,out[right['root_aids']])))<1e-12)
    ranges=bool(np.all(native>=right['lower']-1e-10)and np.all(native<=right['upper']+1e-10))
    mask=bool(np.array_equal(h['controlled_digit_mask'],[1,1,0,0,0]))
    active=bool(h['phase']=='acquire');previous=native.copy()
    contracts.append({'call':step,'data_unchanged':immutable,'passive_zero':zero,'uncontrolled_prescribed_targets_retained':retained,
                      'native_rate_ok':rate,'native_coupled':coupled,'native_ranges_ok':ranges,'two_row_mask_correct':mask,'acquire_active':active})
    if not all([zero,immutable,retained,rate,coupled,ranges,mask,active]):raise ValueError(contracts[-1])
# Per-object configuration must resolve identically without an explicit plan mask.
configured=O6CoupledReferenceContactFeedback(m,ref,{**cfg,'controlled_digits_by_object':{'1':{'right':['thumb','index']}}})
object_rule_correct=bool(np.array_equal(configured._controlled_mask(1,{'active_object_ids_by_hand':[-1,1]}),[1,1,0,0,0]))
oldpath=R/'scripts/fourth_contact_feedback_o6_coupled_2a8711bc684c.py'
spec=importlib.util.spec_from_file_location('frozen_o6_feedback_2a871',oldpath);oldmodule=importlib.util.module_from_spec(spec);spec.loader.exec_module(oldmodule)
old=oldmodule.O6CoupledReferenceContactFeedback(m,ref,cfg);new=O6CoupledReferenceContactFeedback(m,ref,cfg)
z=dict(np.load(a.trial/'physical_trace.npz',allow_pickle=False));f=dict(np.load(a.trial/'sonic_contact_feedback.npz',allow_pickle=False))
live=mj.MjData(m);mj.mj_setState(m,live,z['initial_state'],int(np.asarray(z['state_spec']).reshape(-1)[0]));mj.mj_forward(m,live)
maximum=0.;trace_difference=0.;qerror=0.;steps=min(len(z['ctrl']),int(round(a.shadow_duration_s/m.opt.timestep)))
for step in range(steps):
    now=float(live.time);k=max(0,int(np.searchsorted(f['arm_feedback_time'],now+1e-9,side='right')-1))
    armplan={'active_object_ids_by_hand':list(map(int,f['arm_feedback_ids'][k])), 'phase_by_hand':list(map(str,f['arm_feedback_phase'][k]))}
    packet=f['finger_preshape_template_baseline'][step]
    one=old.apply(live,packet,now,armplan);two=new.apply(live,packet,now,armplan)
    maximum=max(maximum,float(np.max(abs(one-two))));trace_difference=max(trace_difference,float(np.max(abs(two-z['ctrl'][step]))))
    live.ctrl[:]=z['ctrl'][step];mj.mj_step(m,live);qerror=max(qerror,float(np.max(abs(live.qpos-z['qpos'][step+1]))))
report={'schema':'fourth.o6_controlled_digit_mask_private_smoke.v1','mask_contract_calls':contracts,
        'per_object_rule_resolves_correctly':object_rule_correct,'default_all_shadow_steps':steps,
        'default_all_control_difference_to_frozen_helper_max_rad':maximum,
        'default_all_control_difference_to_recorded_packet_max_rad':trace_difference,
        'shadow_frozen_state_replay_qpos_max_abs_error':qerror,'masked_helper_report':hook.report(),
        'masked_physics_rollout_performed':False,'grasp_or_lift_success_claimed':False,
        'status':'passed'if object_rule_correct and maximum<1e-12 and trace_difference<1e-10 and qerror<1e-7 else'failed'}
a.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in report.items()if k not in ['mask_contract_calls','masked_helper_report']}),flush=True)
if report['status']!='passed':raise SystemExit(2)
