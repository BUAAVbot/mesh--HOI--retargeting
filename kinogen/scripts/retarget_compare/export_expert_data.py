"""Export accepted integrated expert execution as named commands and distillation data."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    import numpy as np,mujoco as mj
    report=json.loads((a.trial/'report.json').read_text())
    if not report.get('functional_task_success',False):
        raise RuntimeError('Rejected: failed execution cannot be exported as positive expert demonstration')
    if not report.get('independent_replay',{}).get('passed',False):
        raise RuntimeError('Rejected: independent native control replay is required')
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    z=np.load(a.trial/'trace.npz',allow_pickle=False)
    model=mj.MjModel.from_binary_path(str(a.trial/'model.mjb'));data=mj.MjData(model)
    times=z['time'];requested_policy_times=z['policy_time_s']
    # Float32 accumulated policy phase can drift a few microseconds from the
    # native float64 integration clock. Select the nearest recorded state,
    # never the next5ms state merely because its timestamp rounds upward.
    right=np.searchsorted(times,requested_policy_times).clip(0,len(times)-1)
    left=(right-1).clip(0,len(times)-1)
    indices=np.where(abs(times[left]-requested_policy_times)<=abs(times[right]-requested_policy_times),left,right)
    clock_error=float(np.max(abs(times[indices]-requested_policy_times)))
    if clock_error>=.5*float(np.min(np.diff(times)))or np.any(np.diff(indices)<=0):
        raise ValueError('Policy/state clocks do not have a unique nearest physics sample')
    policy_times=times[indices]
    names=z['joint_names'];qa=z['joint_qadr'];rq=z['robot_root_qadr'];aids=z['actuator_ids']
    actual=z['qpos'][indices];qvel=z['qvel'][indices]
    qpos=np.stack([np.concatenate([actual[:,root:root+7],actual[:,qa[actor]]],axis=-1)for actor,root in enumerate(rq)],axis=1)
    bases=[mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,prefix+'base_link')for prefix in ('p1_','p2_')]
    world_velocity=[]
    for q,v in zip(actual,qvel):
        data.qpos[:]=q;data.qvel[:]=v
        mj.mj_kinematics(model,data);mj.mj_comPos(model,data);mj.mj_comVel(model,data)
        row=[]
        for body in bases:
            velocity=np.zeros(6);mj.mj_objectVelocity(model,data,mj.mjtObj.mjOBJ_BODY,body,velocity,0);row.append(velocity)
        world_velocity.append(row)
    world_velocity=np.asarray(world_velocity)
    control_indices=np.minimum(indices+3,len(z['ctrl'])-1)
    actual_targets=z['ctrl'][control_indices][:,aids]
    active=np.linalg.norm(model.actuator_gear[aids],axis=-1)>0
    body_mask=np.asarray([not str(name).startswith(('lh_','rh_','neck_'))for name in names])
    state_contact_indices=np.clip(indices-1,0,len(z['seat_contact'])-1)
    np.savez_compressed(a.output/'expert_commands.npz',time_s=policy_times,
        requested_policy_phase_s=requested_policy_times,
        source_time_s=z['source_time_s'][indices],frame_ids=np.arange(len(indices)),dof_names=names,
        qpos=qpos,actual_joint_positions_rad=qpos[:,:,7:],
        command_joint_position_rad=qpos[:,:,7:],
        actual_motor_target_position_rad=actual_targets,
        active_joint_motor_mask=active,
        expert_controlled_body_joint_mask=body_mask,
        command_root_linear_velocity_world_m_s=world_velocity[:,:,3:],
        command_root_angular_velocity_world_rad_s=world_velocity[:,:,:3],
        actual_seat_contact=z['seat_contact'][state_contact_indices],
        actual_foot_contact=z['foot_contact'][state_contact_indices],
        actual_seat_support_force_N=z['seat_support_force_N'][state_contact_indices],
        actual_contact_valid=indices>0,
        observed_state_full_qpos=actual,observed_state_full_qvel=qvel,
        inferred_or_synthetic_hand_reference=np.asarray(True),
        functional_task_pass=np.asarray(True),strict_reference_tracking_pass=np.asarray(report['success']))
    metadata={'status':'accepted_physical_execution_data_ready_for_downstream_training',
        'trial':str(a.trial.resolve()),'trace_sha256':sha(a.trial/'trace.npz'),
        'model_sha256':sha(a.trial/'model.mjb'),'report_sha256':sha(a.trial/'report.json'),
        'commands_sha256':sha(a.output/'expert_commands.npz'),'policy':report['controller'],
        'source_method_kind':report.get('method_kind','see_controller_provenance'),
        'trained_policy_checkpoint_present':bool(report['controller'].get('checkpoint')or report['controller'].get('policy')),
        'learned_rl_expert_success_claim':bool(report.get('rl_incremental_gain_proven',False)),
        'nominal_capability_inherited_from_prior':report.get('nominal_capability_inherited_from_prior'),
        'functional_task_pass':report['functional_task_success'],'strict_reference_tracking_pass':report['success'],
        'policy_samples':len(indices),'sample_rate_hz':50,'source_time_s': 'explicit mapping saved; may include declared slowing and final hold',
        'clock_alignment':{'method':'unique nearest recorded physics state; time_s uses actual integration clock',
                           'max_requested_phase_minus_actual_time_abs_s':clock_error,
                           'original_requested_phase_preserved':True},
        'command_semantics':'SUGAR-style command joints are actual executed robot positions, and root velocities are actual world-frame body velocities. Actual motor targets are saved separately as action labels.',
        'coordinate_contract':{'up_axis':'Z','length':'m','root_quaternion':'wxyz','root_velocities':'world-frame base_link velocity queried by mj_objectVelocity'},
        'downstream_status':'Data exported; SONIC weights have not been distilled or updated by this export.',
        'quality_scope':'Positive demonstration is functionally accepted in the saved simulator scenario; strict source error, robustness and hand observation limits remain separately labeled.'}
    (a.output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({'output':str(a.output),'samples':len(indices),'strict_pass':report['success']}))


if __name__=='__main__':main()
