"""Project all source frames onto the frozen O6 linear mimic manifold.

Only hand commands are clipped to the joint-limit intersection, then ten
passive positions/velocities are expanded from twelve independent commands.
Source motion, source timestamps, object trajectories and masks remain saved.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import mujoco as mj
p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
m=mj.MjModel.from_xml_path(str(a.model.resolve()));z=dict(np.load(a.reference,allow_pickle=False));names=list(map(str,z['dof_names']));q=z['qpos'].copy();source=q.copy();time=np.asarray(z['time_s']);relations=[];masters={}
for e in range(m.neq):
    if m.eq_type[e]!=mj.mjtEq.mjEQ_JOINT:continue
    slave,master=int(m.eq_obj1id[e]),int(m.eq_obj2id[e]);sn=mj.mj_id2name(m,mj.mjtObj.mjOBJ_JOINT,slave);mn=mj.mj_id2name(m,mj.mjtObj.mjOBJ_JOINT,master)
    if not sn.startswith(('lh_','rh_')):continue
    coef=m.eq_data[e,:5].copy()
    if np.max(abs(coef[2:]))>1e-12 or coef[1]<=0:raise ValueError('Expected positive linear URDF mimic coupling')
    sj,mjidx=names.index(sn),names.index(mn);ratio=float(coef[1]);offset=float(m.qpos0[m.jnt_qposadr[slave]]+coef[0]-ratio*m.qpos0[m.jnt_qposadr[master]])
    lo=max(float(m.jnt_range[master,0]),float((m.jnt_range[slave,0]-offset)/ratio));hi=min(float(m.jnt_range[master,1]),float((m.jnt_range[slave,1]-offset)/ratio))
    if lo>hi:raise ValueError('No mimic/joint-limit intersection')
    masters[mjidx]=(max(lo,masters.get(mjidx,(-np.inf,np.inf))[0]),min(hi,masters.get(mjidx,(-np.inf,np.inf))[1]))
    aid=mj.mj_name2id(m,mj.mjtObj.mjOBJ_ACTUATOR,'motor_'+sn)
    if aid<0 or np.any(m.actuator_gear[aid])or np.any(m.actuator_gainprm[aid])or np.any(m.actuator_biasprm[aid]):raise ValueError('Mimic child motor must remain a passive zero slot')
    relations.append({'slave':sn,'master':mn,'slave_column':sj,'master_column':mjidx,'offset_rad':offset,'multiplier':ratio,'passive_actuator_id':int(aid)})
if len(relations)!=10:raise ValueError('Expected exactly ten O6 joint equalities')
slave_names={v['slave']for v in relations}
for j,name in enumerate(names):
    if not name.startswith(('lh_','rh_'))or name in slave_names:continue
    jid=mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,name)
    if j not in masters:masters[j]=(float(m.jnt_range[jid,0]),float(m.jnt_range[jid,1]))
if len(masters)!=12:raise ValueError('Expected twelve independent hand commands, including two uncoupled thumb-yaw axes')
clipped={}
for j,(lo,hi)in masters.items():
    original=q[:,7+j].copy();q[:,7+j]=np.clip(original,lo,hi);clipped[names[j]]={'frames_clipped':int(np.sum(q[:,7+j]!=original)),'max_correction_rad':float(np.max(abs(q[:,7+j]-original))),'allowed_master_range_rad':[lo,hi]}
for r in relations:q[:,7+r['slave_column']]=r['offset_rad']+r['multiplier']*q[:,7+r['master_column']]
dq=np.gradient(q[:,7:],time,axis=0);residuals=np.array([q[:,7+r['slave_column']]-r['offset_rad']-r['multiplier']*q[:,7+r['master_column']]for r in relations]);velocity_residual=np.array([dq[:,r['slave_column']]-r['multiplier']*dq[:,r['master_column']]for r in relations]);hand=[j for j,n in enumerate(names)if n.startswith(('lh_','rh_'))];nonhand=np.array([j for j in range(q.shape[1])if j<7 or j-7 not in hand])
payload=dict(z);payload['qpos']=q;payload['o6_input_reference_qpos']=source;payload.setdefault('source_qpos',source);payload['o6_projected_joint_velocity_rad_s']=dq;payload['o6_mimic_manifold_projection']=np.asarray(True);payload['o6_independent_command_count']=np.asarray(12);payload['o6_physical_finger_joint_count']=np.asarray(22)
if'robot_dq'in payload:payload['o6_input_robot_dq']=payload['robot_dq'];payload['robot_dq']=dq
if'qvel'in payload:
    old=payload['qvel'];payload['o6_input_qvel']=old.copy()
    if old.shape==(len(q),52):payload['qvel']=dq
    elif old.shape==(len(q),58):payload['qvel']=old.copy();payload['qvel'][:,6:]=dq
    else:raise ValueError('Unexpected source qvel shape; retain source explicitly before adapting')
object_same=bool(np.array_equal(payload['object_target_pose'],z['object_target_pose']));body_same=bool(np.array_equal(q[:,nonhand],source[:,nonhand]));a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,**payload)
report={'status':'all_frames_projected_to_actual_o6_equality_manifold_not_physics_verified','frames':len(q),'duration_s':float(time[-1]-time[0]),'source_reference':str(a.reference),'source_sha256':hashlib.sha256(a.reference.read_bytes()).hexdigest(),'model':str(a.model),'model_sha256':hashlib.sha256(a.model.read_bytes()).hexdigest(),'output':str(a.output),'output_sha256':hashlib.sha256(a.output.read_bytes()).hexdigest(),'independent_hand_commands':12,'finger_physical_joints':22,'effective_robot_drives':42,'passive_zero_slots':10,'maximum_mimic_position_residual_rad':float(np.max(abs(residuals))),'maximum_mimic_velocity_residual_rad_s':float(np.max(abs(velocity_residual))),'root_body_neck_unchanged':body_same,'all_object_trajectories_unchanged':object_same,'source_time_unchanged':bool(np.array_equal(payload['time_s'],z['time_s'])),'source_hand_observation_masks_unchanged':all(np.array_equal(payload[k],z[k])for k in z if'contact_candidate'in k),'master_clipping':clipped,'linear_relations':relations,'scope':'Source data retained; URDF linear mimic only. Quadratic calibration curves and partial-calibration JSON do not alter this projection.'}
a.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
