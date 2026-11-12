"""Private-fixture contract checks for optional initial-world palm hold.

The candidate module is deliberately separate from the execution module until
the parent coordinates the next source-freeze boundary.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import mujoco as mj
from fourth_arm_object_feedback_absolute_hold_candidate import ArmObjectFeedback
from fourth_contact_feedback_o6_coupled import O6CoupledReferenceContactFeedback
p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();m=mj.MjModel.from_xml_path(str(a.model));ref=dict(np.load(a.reference));d=mj.MjData(m);names=list(map(str,ref['dof_names']));qa=np.array([m.jnt_qposadr[m.joint(n).id]for n in names]);aids=np.array([m.actuator('motor_'+n).id for n in names]);d.qpos[:7]=ref['qpos'][0,:7];d.qpos[qa]=ref['qpos'][0,7:]
for oi in range(6):adr=m.jnt_qposadr[m.joint(f'target_object_{oi+1:02d}_free').id];d.qpos[adr:adr+7]=ref['object_target_pose'][0,oi]
d.qvel[:]=0;mj.mj_forward(m,d);seed=d.qpos.copy();base=np.zeros(m.nu);base[aids]=ref['qpos'][0,7:];passive=np.flatnonzero(np.all(m.actuator_gear==0,axis=1));base[passive]=0;cfg={'absolute_reference_hold':True,'active_hands':[1],'initial_geometric_seed':'actual_state','minimum_source_contact_confidence':2.,'gravity_compensation':True,'ik_iterations':15};arm=ArmObjectFeedback(m,ref,cfg);finger=O6CoupledReferenceContactFeedback(m,ref,{'min_candidate_confidence':.3,'geometry_backend':'convex_sat_o6','finger_urdf':'/data/KinoGen/outputs/fourth_full_sequence/assets/o6_calibrated_linear_joint_contracts_v2.urdf'});checks=[];goals=[]
for k in range(12):
    d.qpos[:]=seed;d.qpos[0]+=.0002*k;d.qvel[:]=0;mj.mj_forward(m,d);before=(d.qpos.copy(),d.qvel.copy(),d.qfrc_applied.copy(),d.xfrc_applied.copy());out=arm.apply(d,base,.02*k);before_finger=out.copy();out=finger.apply(d,out,.02*k,contact_plan=arm.contact_plan);ch=arm.chains[1];other=np.setdiff1d(np.arange(m.nu),ch['aids']);record=arm.records[-1];goals.append(record['goal'][1].copy())
    checks.append({'actual_state_and_external_forces_unchanged':all(np.array_equal(x,y)for x,y in zip(before,[d.qpos,d.qvel,d.qfrc_applied,d.xfrc_applied])),'only_right_arm_changed_before_finger':bool(np.array_equal(before_finger[other],base[other])),'guard_only_finger_packet_equals_baseline':bool(np.array_equal(out,before_finger)),'passive_ctrl_zero':bool(np.all(out[passive]==0)),'arm_phase_hold':str(record['phase'][1])=='hold','no_source_object_ownership':arm.contact_plan['active_object_ids_by_hand']==[-1,-1],'finger_phase_idle':all(s is None or s['phase']=='idle'for s in finger.states),'geometric_palm_error_m':float(np.linalg.norm(record['geometric_palm'][1]-record['goal'][1]))})
good=all(all(v for key,v in row.items()if key!='geometric_palm_error_m')and row['geometric_palm_error_m']<.003 for row in checks)and np.array_equal(np.asarray(goals),np.repeat(goals[0][None,:],len(goals),axis=0));out={'status':'pass'if good else'fail','scope':'Private FK fixture only, no actual physics rollout or force correction','checks':checks,'fixed_world_goal_exact_across_private_base_perturbation':bool(np.array_equal(np.asarray(goals),np.repeat(goals[0][None,:],len(goals),axis=0))),'arm_report':arm.report()};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'status':out['status'],'calls':len(checks),'max_geometric_palm_error_m':max(row['geometric_palm_error_m']for row in checks),'all_checks':good}),flush=True)
if not good:raise SystemExit(1)

