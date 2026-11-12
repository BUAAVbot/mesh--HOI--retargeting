"""Private staged per-digit force contracts, never physical-grasp evidence."""
from pathlib import Path
import json,hashlib
import numpy as np
import mujoco as mj
from fourth_contact_feedback_o6_coupled import O6CoupledReferenceContactFeedback
R=Path(__file__).resolve().parents[1];trial=R/'outputs/fourth_full_sequence/physics_sonic15225_o6_obj1_pinchmask_v3_6s';record=json.loads((trial/'report.json').read_text());m=mj.MjModel.from_binary_path(record['model']['path']);cfg=json.loads((trial/'source_sonic_config.json').read_text())['finger_feedback'];ref=record['reference']['path']
hook=O6CoupledReferenceContactFeedback(m,ref,cfg);mask=np.asarray([1,1,1,0,0],bool);targets={'thumb':2.514,'index':1.682,'middle':2.641};plan={'active_object_ids_by_hand':[-1,1],'controlled_digits_by_hand':[['thumb','index','middle','ring','pinky'],['thumb','index','middle']], 'digit_force_targets_N_by_hand':[None,targets], 'required_contact_digits_by_hand':[None,['thumb','index','middle']]};contracts={}
desired,required=hook._explicit_load_targets(1,0,mask,plan);contracts['vector_maps_to_thumb_index_middle']=bool(np.array_equal(desired,np.asarray([2.514,1.682,2.641,0,0])));contracts['requires_exact_three_contacts']=bool(np.array_equal(required,mask))
sequence=[(0.,[.2,.2,0,0,0]),(.1,[.2,.2,.2,0,0]),(.25,[.2,.2,.2,0,0]),(.4,[.2,.2,.2,0,0]),(.405,[.2,.2,0,0,0])];rows=[]
for t,forces in sequence:
    vector,present,fraction=hook._ramp_load_targets(1,0,mask,desired,required,np.asarray(forces),t);rows.append({'time_s':t,'synthetic_contract_forces_N':forces,'present':present,'fraction':fraction,'target_vector_N':vector.tolist()})
contracts['two_contacts_only_low_preload']=bool(np.max(abs(np.asarray(rows[0]['target_vector_N'])-[.5,.5,.5,0,0]))<1e-12 and not rows[0]['present'])
contracts['third_contact_starts_at_low_preload']=bool(rows[1]['present']and rows[1]['fraction']==0.)
contracts['half_ramp_smoothstep_halfway']=bool(abs(rows[2]['fraction']-.5)<1e-12 and np.max(abs(np.asarray(rows[2]['target_vector_N'])-(np.where(mask,.5,0)+desired)/2))<1e-12)
contracts['full_targets_after_0_3s']=bool(np.max(abs(np.asarray(rows[3]['target_vector_N'])-desired))<1e-12)
contracts['contact_loss_resets_preload']=bool(rows[4]['fraction']==0 and not rows[4]['present'])
contracts['configured_budgets_retained']=bool(max(desired)<=3 and desired.sum()<=12)
zero,_=hook._explicit_load_targets(1,0,mask,{'active_object_ids_by_hand':[-1,1]});contracts['no_explicit_key_disables_new_path']=zero is None
invalid=[]
for bad in [{'thumb':2,'index':1},{'thumb':3.1,'index':1,'middle':1},{'thumb':float('nan'),'index':1,'middle':1}]:
    try:hook._explicit_load_targets(1,0,mask,{**plan,'digit_force_targets_N_by_hand':[None,bad]});invalid.append(False)
    except ValueError:invalid.append(True)
contracts['missing_controlled_digit_rejected']=invalid[0];contracts['per_digit_overbudget_rejected']=invalid[1];contracts['nonfinite_force_rejected']=invalid[2]
object_hook=O6CoupledReferenceContactFeedback(m,ref,{**cfg,'digit_force_targets_N_by_object':{'1':{'right':targets}}});object_target,_=object_hook._explicit_load_targets(1,0,mask,None);contracts['per_object_configuration_resolves']=bool(np.array_equal(object_target,desired))
contracts['left_plan_none_retains_legacy']=hook._explicit_load_targets(0,0,np.ones(5,bool),plan)[0]is None
contracts['physical_model_not_modified']=bool(np.all(m.actuator_gear[hook.passive_actuator_ids]==0)and np.all(m.actuator_gainprm[hook.passive_actuator_ids]==0))
report={'schema':'fourth.o6_staged_load_force_private_contracts.v1','contracts':contracts,'all_passed':all(contracts.values()),'ramp_contract_rows':rows,'helper_sha256':hashlib.sha256((R/'scripts/fourth_contact_feedback_o6_coupled.py').read_bytes()).hexdigest(),'synthetic_contract_only_no_rollout':True,'physical_lift_success_claimed':False}
(R/'reports/fourth_o6_staged_load_force_contracts.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if not report['all_passed']:raise SystemExit(2)
