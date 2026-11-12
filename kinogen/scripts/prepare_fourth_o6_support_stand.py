"""Nominal 923-clock support-posture regression; raw objects are unchanged.

The robot holds a prospective post-slide absolute pose while the scene starts
with the original object poses. This only validates the new robot posture and
SONIC ABI; it neither executes the slide nor claims any grasp or six-object task.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import mujoco as mj
from scipy.spatial.transform import Rotation
from fourth_arm_object_feedback import ArmObjectFeedback
p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest();receipt=json.loads(a.candidate.read_text());source=dict(np.load(a.reference));m=mj.MjModel.from_xml_path(str(a.model));d=mj.MjData(m);h=ArmObjectFeedback(m,source);d.qpos[:]=receipt['best']['qpos']
if sha(a.model)!=receipt['model_sha256']:raise ValueError('Model SHA mismatch')
for oi,adr in enumerate(h.object_qadr):d.qpos[adr:adr+7]=source['object_target_pose'][0,oi]
d.qvel[:]=0;mj.mj_forward(m,d);names=list(map(str,source['dof_names']));jids=np.array([m.joint(n).id for n in names]);qa=m.jnt_qposadr[jids];q=np.r_[d.qpos[:7],d.qpos[qa]];n=len(source['time_s']);payload=dict(source)
for key in list(payload):
    if key.startswith('robot_')and key!='robot_palm_anatomical_frame_in_hand_base':payload['support_stand_source_'+key]=payload.pop(key)
payload['support_stand_input_qpos']=source['qpos'];payload['qpos']=np.repeat(q[None,:],n,axis=0);payload.setdefault('source_qpos',source['qpos']);payload['support_posture_regression_only']=np.array(True);payload['support_posture_absolute_robot_model_qpos']=d.qpos[:59].copy();payload['support_posture_candidate_sha256']=np.array(sha(a.candidate));payload['support_posture_description']=np.array('Internal SONIC hold of absolute support posture; raw initial objects and all923 raw/source targets unchanged. No slide, grasp or full-task success claim.')
for key in ['robot_dq','qvel','pregrasp_projected_joint_velocity_rad_s','o6_projected_joint_velocity_rad_s']:
    if key in source:payload['support_stand_source_'+key]=source[key];payload[key]=np.zeros_like(source[key])
payload['robot_hand_fk_world']=np.repeat(d.xpos[h.hand_bids][None,:,:],n,axis=0);payload['robot_hand_target_world']=payload['robot_hand_fk_world'].copy();palms=np.asarray([h.palm(d,side)for side in range(2)]);payload['robot_palm_fk_world']=np.repeat(palms[None,:,:],n,axis=0);payload['robot_palm_target_world']=payload['robot_palm_fk_world'].copy();quat=Rotation.from_matrix(d.xmat[h.hand_bids].reshape(2,3,3)).as_quat()[:,[3,0,1,2]];payload['robot_hand_orientation_target_world_wxyz']=np.repeat(quat[None,:,:],n,axis=0)
feet=[m.body(side+'_ankle_roll_link').id for side in ['left','right']];wrists=[m.body(side+'_wrist_pitch_link').id for side in ['left','right']];payload['robot_foot_fk_world']=np.repeat(d.xpos[feet][None,:,:],n,axis=0);payload['robot_foot_target_world']=payload['robot_foot_fk_world'].copy();payload['robot_wrist_target_world']=np.repeat(d.xpos[wrists][None,:,:],n,axis=0);com=d.subtree_com[m.body('base_link').id];payload['robot_com_fk_world_m']=np.repeat(com[None,:],n,axis=0);payload['robot_com_target_world_xy_m']=payload['robot_com_fk_world_m'][:,:2].copy();payload['planned_stance_mask']=np.ones((n,2),bool);payload['robot_stance_foot_speed_m_s']=np.zeros((n,2));payload['support_adaptation_description']=payload['support_posture_description'];payload['collision_model_path']=np.array(str(a.model));payload['collision_model_sha256']=np.array(sha(a.model));payload['physical_ready']=np.array(False)
if'raw_source_root_position_m'in source:payload['root_alignment_offset_xyz_m']=payload['qpos'][:,:3]-source['raw_source_root_position_m'];payload['root_deviation_from_raw_source_xyz_m']=payload['root_alignment_offset_xyz_m'].copy()
collisions=[]
for c in d.contact[:d.ncon]:
    bs=list(map(int,m.geom_bodyid[list(c.geom)]));bn=[m.body(b).name or'world'for b in bs];robot=[b>0 and not name.startswith('target_object')for b,name in zip(bs,bn)]
    if any(robot)and not all(robot)and c.dist<-.001:collisions.append({'bodies':bn,'signed_distance_m':float(c.dist)})
if collisions:raise ValueError('Unsafe raw-scene initial robot pose: '+json.dumps(collisions))
protected=[k for k in source if k.startswith(('raw_source_','human_','object_','contact_candidate'))or k in ['time_s','source_frame_ids','frame_ids']];equal=lambda x,y:np.array_equal(x,y,equal_nan=True)if x.dtype.kind in'fc'else np.array_equal(x,y)
if not all(equal(payload[k],source[k])for k in protected):raise ValueError('Source provenance changed')
a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,**payload);report={'role':'Internal new-posture SONIC standing regression only','frames':n,'time_s':float(source['time_s'][-1]-source['time_s'][0]),'raw_initial_objects_and_all_source_targets_unchanged':True,'protected_arrays':len(protected),'robot_initial_root':q[:7].tolist(),'waist_reference_rad':{name:float(q[7+names.index(name)])for name in names if name.startswith('waist')},'robot_scene_penetrations_gt1mm':collisions,'model_sha256':sha(a.model),'candidate_sha256':sha(a.candidate),'output_sha256':sha(a.output)};a.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
