"""Export the entire source timeline with six-object dynamics and quality masks."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trial-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--allow-failed-diagnostic',action='store_true')
    p.add_argument('--source-cameras',type=Path,default=Path('/data/KinoGen/data/fourth_sim_handoff/cameras_sim.npz'))
    a=p.parse_args()
    import numpy as np
    import mujoco as mj
    r=json.loads((a.trial_dir/'report.json').read_text());audit=json.loads((a.trial_dir/'independent_replay_audit.json').read_text())
    if not audit['replay_verified']or not audit['six_dynamic_objects']or not audit['all_923_source_frames_covered']:
        raise ValueError('An independently verified full 923-frame/six-free-object trace is required')
    if not audit.get('robot_hand_mimic_passive_actuation_contract_passed',True):
        raise ValueError('Robot hand mimic/passive-actuation contract failed')
    if not r['simulation_passed']and not a.allow_failed_diagnostic:
        raise ValueError('Quality gates failed; only explicit diagnostic export is permitted')
    z={k:v for k,v in np.load(a.trial_dir/'physical_trace.npz',allow_pickle=False).items()}
    c={k:v for k,v in np.load(a.trial_dir/'physical_contacts.npz',allow_pickle=False).items()}
    ref={k:v for k,v in np.load(r['reference']['path'],allow_pickle=False).items()}
    for path,expected in [(a.trial_dir/'physical_trace.npz',r['artifacts']['trace']['sha256']),
                          (a.trial_dir/'physical_contacts.npz',r['artifacts']['contacts']['sha256']),
                          (Path(r['model']['path']),r['model']['sha256']),
                          (Path(r['reference']['path']),r['reference']['sha256'])]:
        if sha(path)!=expected:raise ValueError('Frozen SHA mismatch: '+str(path))
    mp=Path(r['model']['path']);model=mj.MjModel.from_binary_path(str(mp))if mp.suffix=='.mjb'else mj.MjModel.from_xml_path(str(mp))
    data=mj.MjData(model);idx=z['source_state_index'];qa=z['joint_qadr'];va=z['joint_vadr'];aids=z['actuator_ids']
    states=z['qpos'][idx];vel=z['qvel'][idx]
    positions=[];quaternions=[]
    for q in states:
        data.qpos[:]=q;mj.mj_forward(model,data);positions.append(data.xpos.copy());quaternions.append(data.xquat.copy())
    interval=np.maximum(idx-1,0);interval=np.minimum(interval,len(z['ctrl'])-1)
    rootup=1-2*(states[:,4]**2+states[:,5]**2)
    framepen=z['max_penetration_m'][interval]
    mask=np.asarray(ref['object_observed_mask'],bool)
    error=z['source_frame_object_position_error_m']
    object_tracking=(error<=r['quality_thresholds']['observed_object_error_max_m'])&mask
    physical_mask=(rootup>=.5)&(framepen<=r['quality_thresholds']['max_penetration_m'])
    payload=dict(source_frame_ids=ref.get('source_frame_ids',np.arange(923)),source_time_s=z['source_time_s'],
        actual_sample_time_s=z['time'][idx],actual_robot_root_pose_xyz_wxyz=states[:,:7],joint_names=z['dof_names'],
        actual_joint_position_rad=states[:,qa],actual_joint_velocity_rad_s=vel[:,va],
        actual_joint_drive_torque_Nm=z['actuator_force'][interval][:,aids]*model.actuator_gear[aids,0],
        actual_joint_control=z['ctrl'][interval][:,aids],object_ids=z['object_ids'],
        actual_object_pose_xyz_wxyz=z['source_frame_object_actual_pose'],
        object_observed_position_error_m=error,object_observed_tracking_pass_mask=object_tracking,
        physical_frame_quality_mask=physical_mask,physical_frame_max_penetration_m=framepen,
        body_names=c['body_names'],body_position_world_m=np.asarray(positions),body_orientation_wxyz=np.asarray(quaternions),
        physics_time_s=z['time'],physics_qpos=z['qpos'],physics_qvel=z['qvel'],
        physics_joint_drive_torque_Nm=z['actuator_force'][:,aids]*model.actuator_gear[aids,0],
        physics_joint_control=z['ctrl'][:,aids])
    for k,v in ref.items():payload['reference_'+k]=v
    for k,v in c.items():payload['contact_'+k]=v
    camera_receipt=None
    if a.source_cameras.exists():
        for k,v in np.load(a.source_cameras,allow_pickle=False).items():payload['source_camera_'+k]=v
        camera_receipt={'path':str(a.source_cameras),'sha256':sha(a.source_cameras),'provenance':'Original supplied reconstruction cameras; not assumed robot-head calibration'}
    a.output_dir.mkdir(parents=True,exist_ok=True);out=a.output_dir/'hoi_full_923_six_objects.npz';np.savez_compressed(out,**payload)
    scalar_meta={}
    for key,value in ref.items():
        if value.shape==():
            item=value.item()
            scalar_meta[key]=item.decode('utf-8',errors='replace')if isinstance(item,bytes)else item
    active_actuators=np.any(model.actuator_gear!=0,axis=1)&(np.any(model.actuator_gainprm!=0,axis=1)|np.any(model.actuator_biasprm!=0,axis=1))
    meta={'schema':'fourth.full_hoi_923.v1','status':'accepted_full_sequence'if r['simulation_passed']else'diagnostic_quality_failed',
        'source_frames':923,'objects':6,'robot_dof':52,'physics_steps':len(z['ctrl']),
        'declared_actuator_slots':model.nu,'active_actuator_count':int(active_actuators.sum()),
        'disabled_actuator_count':int((~active_actuators).sum()),
        'hand_mimic_constraint_audit':audit.get('robot_hand_mimic_constraints',[]),
        'source_duration_s':r['source_duration_s'],'physics_duration_s':r['actual_duration_s'],
        'continuous_rollout':r['continuous_rollout'],'controller':r.get('sonic',{}).get('controller',r['controller']),
        'controller_details':r.get('sonic'),'collision_configuration':audit.get('collision_configuration'),
        'reference_scalar_metadata':scalar_meta,'quality_checks':r['checks'],
        'physical_quality_frames':int(physical_mask.sum()),'observed_tracking_pass_frames_per_object':object_tracking.sum(axis=0).tolist(),
        'source_human_observed_frames':int(np.asarray(ref['human_observed_mask']).sum()),
        'source_object_observed_frames':mask.sum(axis=0).tolist(),'source_object_imputed_frames':(~mask).sum(axis=0).tolist(),
        'object_metrics':r['object_metrics'],'quality_thresholds':r['quality_thresholds'],
        'contact_convention':'World wrench on geom2 at contact point; geom1 opposite. One graph edge per contact point and physical interval.',
        'timing':'Contact interval k is solved for [physics_time[k],physics_time[k+1]]; source samples use nearest following state and record actual time.',
        'coordinate_system':'Z-up, metres, quaternion wxyz','physical_parameters':'assumed; not measured',
        'forces':'simulated solver contact forces and actuator torques, not sensor ground truth',
        'source_cameras':camera_receipt,'reference':r['reference'],'model':r['model'],
        'trace_sha256':r['artifacts']['trace']['sha256'],'replay_audit_sha256':sha(a.trial_dir/'independent_replay_audit.json'),
        'dataset':{'path':str(out),'bytes':out.stat().st_size,'sha256':sha(out)},
        'limitations':r['limitations']}
    (a.output_dir/'metadata.json').write_text(json.dumps(meta,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:meta[k]for k in ['status','source_frames','objects','source_object_observed_frames','physical_quality_frames','dataset']}),flush=True)


if __name__=='__main__':main()
