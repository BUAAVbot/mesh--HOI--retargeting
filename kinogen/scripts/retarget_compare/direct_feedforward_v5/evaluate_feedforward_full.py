"""Full200Hz physical-feedforward specialist evaluation with independent control replay."""
import argparse,hashlib,json,os,shutil,time
from pathlib import Path

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--zero-residual',action='store_true')
    p.add_argument('--initial-state',type=Path);p.add_argument('--initial-states',type=Path);p.add_argument('--case-id',type=int,default=0)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    import numpy as np,mujoco as mj,torch
    from centered_actor import ActorCritic
    from two_sitting_physics import contacts,independent_replay,write_json,interp
    torch.set_num_threads(2)
    checkpoint=torch.load(a.checkpoint,map_location=a.device,weights_only=False)
    if checkpoint.get('physical_feedforward_prior'):os.environ['PHYSICAL_FEEDFORWARD_PRIOR']=checkpoint['physical_feedforward_prior']
    from physical_feedforward_env import PhysicalFeedforwardEnv,PRIOR
    if checkpoint.get('physical_feedforward_sha256') and sha(PRIOR)!=checkpoint['physical_feedforward_sha256']:raise ValueError('Physical prior changed')
    if checkpoint.get('expert_observation_contract')!='direct_feedforward_centered_proprio_seat_v5':raise ValueError('Wrong observation contract')
    a.model=Path(checkpoint['model_path']);a.reference=Path(checkpoint['reference_path'])
    if sha(a.model)!=checkpoint['model_sha256']or sha(a.reference)!=checkpoint['reference_sha256']:raise ValueError('Model/reference changed')
    initial_path=a.initial_state or Path(checkpoint['fixed_initial_state_path'])
    os.environ['DIRECT_INITIAL_STATE_PATH']=str(initial_path);os.environ['FF_TRAIN_INITIAL_NOISE']='0'
    env=PhysicalFeedforwardEnv(a.model,a.reference,n=1,seed=174145,action_scale=checkpoint['action_scale'],sonic_base=False,threads=1)
    env.reset(torch.arange(1,device=env.device),force_start=True)
    m,d,ref=env.model,env.datas[0],env.ref
    if a.initial_states:
        bag=dict(np.load(a.initial_states,allow_pickle=False));state=bag['initial_states'][a.case_id]
        mj.mj_setState(m,d,state,mj.mjtState(int(bag['state_spec'])))
        env._cache_one(0);env._generation+=1;env._host_generation=env._generation;env.refresh_policy_sensors()
    actor=ActorCritic(checkpoint['obs_dim'],56).to(env.device);actor.load_state_dict(checkpoint['model']);actor.eval()
    a.output.mkdir(parents=True);binary=a.output/'model.mjb';mj.mj_saveModel(m,str(binary),None)
    sources=a.output/'source_snapshot';sources.mkdir()
    for name in ('physical_feedforward_env.py','direct_proprio.py','sugar_expert_env.py'):
        shutil.copy2(Path(__file__).with_name(name),sources/name)
    shutil.copy2(__file__,sources/'evaluate_feedforward_full.py')
    env.enable_capture()
    spec=mj.mjtState.mjSTATE_INTEGRATION;initial=np.empty(mj.mj_stateSize(m,spec));mj.mj_getState(m,d,initial,spec)
    np.savez_compressed(a.output/'initial_state.npz',state=initial,state_spec=int(spec))
    names=env.capture_geom_names;bodies=env.capture_body_names
    ts=[float(d.time)];qs=[d.qpos.copy()];vs=[d.qvel.copy()];controls=[];forces=[];torques=[];rows=[];seats=[];feet=[];sf=[];pens=[];ips=[];actions=[];elapsed=[]
    horizon=float(env.prior['time'][-1]);reason='completed';began=time.monotonic();base=None;use_sonic=False;seating=None;a.hold_last_seconds=0.
    for step in range(int(round(horizon/.02))):
        obs=env.observations()
        with torch.no_grad():
            action=torch.zeros((1,56),device=env.device)if a.zero_residual else actor.centered_mean(actor.normalize(obs),actor.normalize(env.counterfactual_observations()))
            if not torch.isfinite(action).all():reason='nonfinite_observation';break
            env.step(action)
        actions.append(action[0].cpu().numpy());elapsed.append(step*.02)
        for row in env.current_capture:
            qs.append(row['qpos']);vs.append(row['qvel']);ts.append(row['time']);controls.append(row['ctrl'])
            forces.append(row['actuator_force']);torques.append(row['qfrc_actuator'])
            records,seat,foot,seatforce,pen,ip=row['contacts'];rows.extend(records);seats.append(seat);feet.append(foot);sf.append(seatforce);pens.append(pen);ips.append(ip)
        if np.any(d.qfrc_applied)or np.any(d.xfrc_applied):raise ValueError('External assisting force forbidden')
        if env._numeric_fault[0]:reason='numeric_failure';break
        if any(d.qpos[rq+2]<.15 for rq in ref.root_qadr):reason='root_below_0p15m';break
    qs=np.asarray(qs);vs=np.asarray(vs);ts=np.asarray(ts);controls=np.asarray(controls)
    raw=dict(np.load(a.reference,allow_pickle=False));desired=np.asarray([interp(raw,t)for t in ts])
    xy=[];ze=[];oe=[];je=[]
    for i,mp in enumerate(ref.maps):
        actual=qs[:,mp['rq']:mp['rq']+7]
        xy.append(np.linalg.norm(actual[:,:2]-desired[:,i,:2],axis=-1));ze.append(abs(actual[:,2]-desired[:,i,2]))
        oe.append(2*np.arccos(np.clip(abs(np.sum(actual[:,3:7]*desired[:,i,3:7],axis=-1)),0,1)))
        je.append(np.sqrt(np.mean((qs[:,mp['qa'][ref.body_cols]]-desired[:,i,7+ref.body_cols])**2,axis=-1)))
    xy=np.asarray(xy).T;ze=np.asarray(ze).T;oe=np.asarray(oe).T;je=np.asarray(je).T
    seats=np.asarray(seats);tail=ts[1:]>=horizon-.2;fraction=seats[tail].mean(0)if tail.any()else np.zeros(2)
    tail_states=ts>=horizon-.2
    final_linear_speed=np.asarray([np.linalg.norm(vs[tail_states,rv:rv+3],axis=-1).max()for rv in ref.root_vadr])if tail_states.any()else np.full(2,np.inf)
    final_angular_speed=np.asarray([np.linalg.norm(vs[tail_states,rv+3:rv+6],axis=-1).max()for rv in ref.root_vadr])if tail_states.any()else np.full(2,np.inf)
    weight=np.asarray([m.body_subtreemass[mp['base']]*np.linalg.norm(m.opt.gravity)for mp in ref.maps])
    final_seat_force=np.asarray(sf)[tail].mean(0)if tail.any()else np.zeros(2)
    replay=independent_replay(binary,initial,spec,controls,qs,vs)
    checks={'full_source_duration':bool(ts[-1]>=ref.end-1e-8),'completed_requested_duration':bool(reason=='completed'and ts[-1]>=horizon-1e-8),
        'root_xy_error_below_0p25m':bool(xy.max()<.25),'root_z_error_below_0p18m':bool(ze.max()<.18),
        'root_orientation_error_below_0p65rad':bool(oe.max()<.65),'penetration_below_0p025m':bool(max(pens,default=0.)<.025),
        'body_joint_rms_below_0p4rad':bool(np.median(je,axis=0).max()<.4),
        'final_both_people_have_true_seat_contact':bool(len(seats)and seats[-1].all()and np.all(fraction>=.5)),
        'final_roots_stable':bool(np.all(final_linear_speed<.15)and np.all(final_angular_speed<.3)),
        'final_seat_bears_at_least_ten_percent_body_weight':bool(np.all(final_seat_force>=.10*weight)),
        'independent_control_only_replay':replay['passed'],'no_state_overrides_or_external_forces':True}
    root_height=np.stack([qs[:,rq+2]for rq in ref.root_qadr],axis=1)
    root_tilt=np.stack([np.arccos(np.clip(1.-2.*(qs[:,rq+4]**2+qs[:,rq+5]**2),-1.,1.))for rq in ref.root_qadr],axis=1)
    exchanged_distance=np.asarray([np.linalg.norm(qs[-1,rq:rq+2]-qs[0,rq:rq+2])for rq in ref.root_qadr])
    # Functional task completion is reported separately from source-trajectory
    # accuracy. It never changes the strict success flag above.
    functional_checks={key:checks[key]for key in ('full_source_duration','completed_requested_duration',
        'penetration_below_0p025m','final_both_people_have_true_seat_contact','final_roots_stable',
        'final_seat_bears_at_least_ten_percent_body_weight','independent_control_only_replay','no_state_overrides_or_external_forces')}
    functional_checks.update(both_rise_from_seat=bool(np.all(root_height.max(0)-root_height[0]>.12)),
        both_exchange_seat_positions=bool(np.all(exchanged_distance>1.2)and np.all(xy[-1]<.30)),
        no_fall=bool(root_height.min()>.45 and root_tilt.max()<.8))
    fields=dict(time=ts,qpos=qs,qvel=vs,ctrl=controls,actuator_force=np.asarray(forces),qfrc_actuator=np.asarray(torques),initial_state=initial,state_spec=int(spec),
        joint_names=np.asarray(ref.names),joint_qadr=np.asarray([x['qa']for x in ref.maps]),joint_vadr=np.asarray([x['va']for x in ref.maps]),
        actuator_ids=np.asarray([x['aids']for x in ref.maps]),robot_root_qadr=ref.root_qadr,robot_root_vadr=ref.root_vadr,
        source_time_s=np.interp(ts,env.prior['time'],env.prior['source_time_s']),reference_qpos=desired,root_xy_error_m=xy,root_z_error_m=ze,root_orientation_error_rad=oe,
        seat_contact=seats,foot_contact=np.asarray(feet),seat_support_force_N=np.asarray(sf),geom_names=np.asarray(names),body_names=np.asarray(bodies),
        policy_time_s=np.asarray(elapsed),learned_residual_action=np.asarray(actions))
    for key,i,shape,dtype in [('interval_index',0,(),int),('time_s',1,(),float),('geom_ids',2,(2,),int),('body_ids',3,(2,),int),('position_world_m',4,(3,),float),('force_world_on_geom2_N',5,(3,),float),('torque_world_on_geom2_Nm',6,(3,),float),('normal_force_N',7,(),float),('signed_distance_m',8,(),float)]:
        fields['contact_'+key]=np.asarray([r[i]for r in rows],dtype=dtype).reshape((-1,)+shape)
    np.savez_compressed(a.output/'trace.npz',**fields)
    report=dict(success=all(checks.values()),status='passed'if all(checks.values())else'failed_physical_quality',checks=checks,termination=reason,
        functional_task_success=all(functional_checks.values()),functional_task_checks=functional_checks,
        simulated_seconds=float(ts[-1]),source_duration_seconds=float(raw['source_time_s'][-1]),execution_reference_duration_seconds=ref.end,physics_steps=len(controls),
        max_root_xy_error_m=xy.max(0).tolist(),max_root_z_error_m=ze.max(0).tolist(),max_root_orientation_error_rad=oe.max(0).tolist(),
        max_penetration_m=max(pens,default=0.),max_interperson_penetration_m=max(ips,default=0.),final_0p2s_seat_contact_fraction=fraction.tolist(),
        final_root_linear_speed_max_m_s=final_linear_speed.tolist(),final_root_angular_speed_max_rad_s=final_angular_speed.tolist(),
        final_seat_vertical_force_mean_N=final_seat_force.tolist(),robot_weight_N=weight.tolist(),
        root_height_min_m=root_height.min(0).tolist(),root_global_tilt_max_rad=root_tilt.max(0).tolist(),exchanged_distance_m=exchanged_distance.tolist(),final_root_xy_error_m=xy[-1].tolist(),
        contact_records=len(rows),independent_replay=replay,controller={'method':('zero residual contract check'if a.zero_residual else'PPO56 learned joint target residual')+(' on frozen dual SONIC'if use_sonic else' on declared kinematic reference PD'),'policy':str(a.checkpoint),'policy_sha256':sha(a.checkpoint),'training_iteration':checkpoint['iteration'],'zero_residual_override':a.zero_residual,
        'privileged_root_state_observation':True,'action_scale_rad':checkpoint['action_scale'],'body_delay_physics_steps':3,'sonic_internal_delay':0 if use_sonic else None,'base':base.report()if base is not None else'reference PD; no frozen SONIC actor','terminal_seating_feedback':seating.report()if seating is not None else None},
        state_override_after_reset_count=0,external_force_steps=0,root_actuators=0,mocap_bodies=m.nmocap,
        mujoco_version=mj.__version__,wall_seconds=time.monotonic()-began,model_dimensions={'nq':m.nq,'nv':m.nv,'nu':m.nu},
        inputs={'reference':str(a.reference),'reference_sha256':sha(a.reference),'model':str(a.model),'checkpoint':str(a.checkpoint)},
        limitations=['Single source-specific policy with privileged simulator localization; not tested on hardware or unseen trajectories.','Scene friction and support geometry are reconstruction-based assumptions.'])
    if seating is not None:
        write_json(a.output/'seating_feedback.json',seating.records)
    report['requested_final_hold_seconds']=a.hold_last_seconds
    report['controller'].update(method='Historical actual200Hz feedforward plus new learned feedback',sonic_executed=False,physical_feedforward_prior=str(PRIOR),physical_feedforward_sha256=sha(PRIOR),historical_nominal_delay_added=False,new_feedback_delay_physics_steps=3,observation_contract='direct_feedforward_centered_proprio_seat_v5')
    report['historical_nominal_execution_seconds']=float(env.prior['time'][-1])
    report['new_terminal_hold_added_seconds']=0.
    report['initial_state_override']={'path':str(initial_path),'sha256':sha(initial_path),'heldout_file':None if a.initial_states is None else str(a.initial_states),'heldout_sha256':None if a.initial_states is None else sha(a.initial_states),'case_id':a.case_id}
    report['source_accuracy_scope']='PHC/source metrics retained; historical source clock and actual controls preserved; prior replay is not newly trained expert success.'
    report['actual_initial_integration_sha256']=hashlib.sha256(initial.tobytes()).hexdigest()
    report['controller']['base']='Immutable recorded actual200Hz physical controls; no SONIC actor'
    report['controller']['sampler_unused_in_deterministic_evaluation']=True
    env.close()
    report.update(method_kind='PPO_feedback_prototype_with_frozen_CEM_prior',nominal_capability_inherited_from_prior=True,rl_incremental_gain_proven=False,policy_checkpoint_present=True)
    write_json(a.output/'report.json',report)
    print(json.dumps({'report':str(a.output/'report.json'),'success':report['success'],'simulated_s':ts[-1]}),flush=True)


if __name__=='__main__':main()
