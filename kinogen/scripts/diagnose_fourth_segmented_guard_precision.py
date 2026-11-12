"""Kinematic-only table-release guard fixture; no dynamic success is claimed."""
from pathlib import Path
import argparse
import json
import numpy as np
import mujoco as mj
from fourth_contact_feedback_segmented import SegmentedReferenceContactFeedback as ReferenceContactFeedbackWithTableGuard

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--model',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);a=p.parse_args()
m=mj.MjModel.from_xml_path(str(a.model));ref=dict(np.load(a.reference,allow_pickle=False))
m.opt.ccd_iterations=100;m.opt.ccd_tolerance=1e-8
hook=ReferenceContactFeedbackWithTableGuard(m,ref,{'enabled':False})
d=mj.MjData(m);hand=hook.hands[1];normal=np.asarray([.010271048830894367,.02292500590934548,.999684425036206])
u=np.asarray([1.,0.,-normal[0]/normal[2]]);u/=np.linalg.norm(u)
desired=np.column_stack([u,np.cross(-normal,u),-normal])
for h in hook.hands:
    d.qpos[h['qa']]=h['lower']
    for i,name in enumerate(h['names']):
        if 'mcp_pitch'in name:d.qpos[h['qa'][i]]=1.
        elif 'thumb_cmc_yaw'in name:d.qpos[h['qa'][i]]=.6
        elif 'thumb_cmc_pitch'in name:d.qpos[h['qa'][i]]=.4
        elif 'thumb_ip'in name:d.qpos[h['qa'][i]]=.6
mj.mj_kinematics(m,d)
base=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'rh_hand_base_link')
rotation=desired@d.xmat[base].reshape(3,3).T
quat=np.empty(4);mj.mju_mat2Quat(quat,rotation.reshape(-1));d.qpos[3:7]=quat
mj.mj_kinematics(m,d)
palm=mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,'collision_rh_palm_mesh')
site=d.geom_xpos[palm]+d.geom_xmat[palm].reshape(3,3)@m.geom_aabb[palm,:3]
centre=np.asarray([-.57,1.8,0.]);centre[2]=(-1.0058665495441206-normal[:2]@centre[:2])/normal[2]
d.qpos[:3]+=centre+.10*normal-site;mj.mj_kinematics(m,d)
ctrl=np.zeros(m.nu)
for h in hook.hands:ctrl[h['aids']]=d.qpos[h['qa']]
initial=hook.apply(d,ctrl,0.);preserved=True;immutable=True;range_ok=True;rate_ok=True
previous=initial.copy();minimum=1.;first_stage=[]
for step in range(1,121):
    d.time=step*.02
    for h in hook.hands:d.qpos[h['qa']]=previous[h['aids']]
    mj.mj_kinematics(m,d)
    request=ctrl.copy();request[hand['aids']]=hand['lower']
    before={key:getattr(d,key).copy()for key in ['qpos','qvel','ctrl','qfrc_applied','xfrc_applied']}
    out=hook.apply(d,request,d.time)
    immutable&=all(np.array_equal(value,getattr(d,key))for key,value in before.items())
    preserved&=np.array_equal(out[hook.preserved_actuator_ids],request[hook.preserved_actuator_ids])
    range_ok&=bool(np.all(out[hand['aids']]>=hand['lower']-1e-8)and np.all(out[hand['aids']]<=hand['upper']+1e-8))
    rate_ok&=bool(np.all(np.abs(out[hand['aids']][hand['geometry_controlled']]-previous[hand['aids']][hand['geometry_controlled']])<=.02000001))
    distance,ff,unknown=hook._clearance(d,1,out[hand['aids']]);minimum=min(minimum,distance)
    first_stage.append(distance);previous=out
held=previous[hand['aids']].copy()
# A private kinematic lift supplies the clearance a real arm must create.
# This fixture movement is not part of the control hook or a physics rollout.
d.qpos[:3]+=.05*normal
for step in range(121,221):
    d.time=step*.02
    for h in hook.hands:d.qpos[h['qa']]=previous[h['aids']]
    mj.mj_kinematics(m,d);request=ctrl.copy();request[hand['aids']]=hand['lower']
    before=d.qpos.copy();out=hook.apply(d,request,d.time);immutable&=np.array_equal(before,d.qpos)
    preserved&=np.array_equal(out[hook.preserved_actuator_ids],request[hook.preserved_actuator_ids]);previous=out
released=previous[hand['aids']];control=hand['geometry_controlled']
release_finished=bool(np.max(abs(released[control]-hand['lower'][control]))<.005)
result={'schema':'fourth.table_guard_kinematic_fixture.v1','physical_rollout_performed':False,
        'fixture':'Closed right fingers at assumed palm height10cm, request open; then manually lift private fixture5cm and request open.',
        'actual_test_data_unchanged_by_hook':bool(immutable),'body_neck_ctrl_preserved':bool(preserved),
        'controlled_finger_range_ok':bool(range_ok),'controlled_finger_rate_ok':bool(rate_ok),
        'minimum_right_hand_clearance_before_fixture_lift_m':minimum,
        'held_finger_targets_rad':held.tolist(),'final_targets_after_lift_rad':released.tolist(),
        'release_finished_after_fixture_lift':release_finished,'report':hook.report()}
result['status']='passed'if immutable and preserved and range_ok and rate_ok and minimum>=.0018 and release_finished else'fixture_failed'
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items()if k!='report'}),flush=True)
if result['status']!='passed':raise SystemExit(2)



