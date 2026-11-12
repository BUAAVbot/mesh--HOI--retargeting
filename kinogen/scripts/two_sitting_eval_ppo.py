"""Evaluate a learned dual-Vita residual in native CPU MuJoCo, without resets.

The frozen SONIC actor remains active. Only joint target residuals are learned;
world-root error is a privileged simulation observation. Every physical control
is retained for an independent control-only replay of the frozen binary plant.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','proxy-model','reference','checkpoint','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--zero-residual',action='store_true',help='Validate the frozen SONIC batch contract with exactly zero residual')
    p.add_argument('--seating-feedback',action='store_true',help='Disclosed motor-only terminal seating feedback')
    p.add_argument('--hold-last-seconds',type=float,default=1.)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    import mujoco as mj,numpy as np,torch
    from two_sitting_ppo_env import Reference,ActorCritic
    from two_sitting_batch_sonic import DualBatchSonic
    from two_sitting_physics import contacts,independent_replay,write_json,interp
    torch.set_num_threads(2)
    checkpoint=torch.load(a.checkpoint,map_location=a.device,weights_only=False)
    if checkpoint.get('action_dim')!=56:raise ValueError('Expected trained56-output dual actor')
    if not a.zero_residual and checkpoint.get('reference_sha256')!=sha(a.reference):raise ValueError('Reference hash mismatch')
    if checkpoint.get('model_sha256')!=sha(a.model):raise ValueError('Training/evaluation source model hash mismatch')
    m=(mj.MjModel.from_binary_path(str(a.model.resolve())) if a.model.suffix=='.mjb'
       else mj.MjModel.from_xml_path(str(a.model.resolve())))
    ref=Reference(m,a.reference,a.device)
    use_sonic=bool(checkpoint.get('sonic_base',False))
    base=DualBatchSonic(m,ref,1,a.device,a.proxy_model)if use_sonic else None
    ref.refresh_actuation(m)
    seating=None
    if a.seating_feedback:
        from two_sitting_seat_feedback import SeatingFeedback
        seating=SeatingFeedback(m,dict(np.load(a.reference,allow_pickle=False)),{})
    actor=ActorCritic(checkpoint['obs_dim'],action_dim=56).to(a.device)
    actor.load_state_dict(checkpoint['model'],strict=True);actor.eval()
    a.output.mkdir(parents=True)
    binary=a.output/'model.mjb';mj.mj_saveModel(m,str(binary),None)
    sources=a.output/'source_snapshot';sources.mkdir()
    for n in ('two_sitting_eval_ppo.py','two_sitting_ppo_env.py','two_sitting_batch_sonic.py','fourth_sonic_controller.py','two_sitting_physics.py'):
        shutil.copyfile(Path(__file__).with_name(n),sources/n)
    if seating is not None:shutil.copyfile(Path(__file__).with_name('two_sitting_seat_feedback.py'),sources/'two_sitting_seat_feedback.py')
    d=mj.MjData(m);d.qpos[:]=ref.q[0].cpu().numpy();d.qvel[:]=0.;mj.mj_forward(m,d)
    sensor=mj.MjData(m)
    spec=mj.mjtState.mjSTATE_INTEGRATION;initial=np.empty(mj.mj_stateSize(m,spec));mj.mj_getState(m,d,initial,spec)
    np.savez_compressed(a.output/'initial_state.npz',state=initial,state_spec=int(spec))
    previous=torch.zeros((1,56),device=a.device)
    oldbody=ref.controls(torch.zeros(1,device=a.device),previous,0.)[:,ref.body_aids_t].clone()
    names=[mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,i)or str(i)for i in range(m.ngeom)]
    bodies=[mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,i)or str(i)for i in range(m.nbody)]
    ts=[0.];qs=[d.qpos.copy()];vs=[d.qvel.copy()];controls=[];forces=[];torques=[];rows=[];seats=[];feet=[];sf=[];pens=[];ips=[];actions=[]
    elapsed=[];reason='completed';began=time.monotonic()
    warning=np.array([w.number for w in d.warning])
    if a.hold_last_seconds<0:raise ValueError('Negative final hold')
    horizon=ref.end+a.hold_last_seconds;cycles=int(np.ceil(horizon/.02-1e-9))
    for step in range(cycles):
        t=float(d.time)
        sensor.qpos[:]=d.qpos;sensor.qvel[:]=d.qvel
        mj.mj_kinematics(m,sensor);mj.mj_comPos(m,sensor);mj.mj_comVel(m,sensor)
        q=torch.as_tensor(d.qpos.copy(),device=a.device,dtype=torch.float32)[None]
        v=torch.as_tensor(d.qvel.copy(),device=a.device,dtype=torch.float32)[None]
        xp=torch.as_tensor(sensor.xpos.copy(),device=a.device,dtype=torch.float32)[None]
        rot=torch.as_tensor(sensor.xmat.copy().reshape(m.nbody,3,3),device=a.device,dtype=torch.float32)[None]
        cv=torch.as_tensor(sensor.cvel.copy(),device=a.device,dtype=torch.float32)[None]
        phase=torch.tensor([min(t,ref.horizon)],device=a.device)
        with torch.inference_mode():
            nominal=base.targets(q,v,phase,rot,cv).clone()if base is not None else ref.controls(phase,torch.zeros_like(previous),0.)
            target,_,_=ref.sample(phase)
            obs=ref.observations(q,v,xp,phase,previous)
            if base is not None:obs=torch.cat([obs,nominal[:,ref.body_aids_t]-target[:,ref.qa_t]],dim=-1)
            if not torch.isfinite(obs).all():reason='nonfinite_observation';break
            action=torch.zeros((1,56),device=a.device)if a.zero_residual else actor.actor(actor.normalize(obs))
            prelimit=nominal.clone()
            prelimit[:,ref.body_aids_t]+=checkpoint['action_scale']*torch.tanh(action)
            if seating is not None:prelimit=seating.apply(prelimit,d,t)
            packet=prelimit.clone()
            packet[:,ref.body_aids_t]=packet[:,ref.body_aids_t].clamp(ref.lower,ref.upper)
            packet[:,ref.passive_t]=0.
            delayed=packet.clone();delayed[:,ref.body_aids_t]=oldbody
            if step==0:delayed[:,ref.body_aids_t]=packet[:,ref.body_aids_t]
            oldbody=packet[:,ref.body_aids_t].clone()
            uold=delayed[0].cpu().numpy();unew=packet[0].cpu().numpy()
        for sub in range(4):
            u=uold if sub<3 else unew
            d.ctrl[:]=u;mj.mj_step(m,d)
            records,seat,foot,seatforce,pen,ip=contacts(m,d,len(controls),names,bodies)
            rows.extend(records);seats.append(seat);feet.append(foot);sf.append(seatforce);pens.append(pen);ips.append(ip)
            controls.append(u.copy());forces.append(d.actuator_force.copy());torques.append(d.qfrc_actuator.copy())
            qs.append(d.qpos.copy());vs.append(d.qvel.copy());ts.append(float(d.time))
            if np.any(d.qfrc_applied)or np.any(d.xfrc_applied):raise ValueError('External force forbidden')
            if not np.isfinite(d.qpos).all()or not np.isfinite(d.qvel).all()or np.any(np.array([w.number for w in d.warning])>warning):reason='numeric_failure';break
        if base is not None:base.record_applied(prelimit)
        previous=action.clone();actions.append(action[0].cpu().numpy());elapsed.append(t)
        if reason!='completed':break
        if any(d.qpos[rq+2]<.15 for rq in ref.root_qadr):reason='root_below_0p15m';break
        if step%100==0:print(json.dumps({'policy_update':step,'simulated_s':d.time,'horizon_s':horizon}),flush=True)
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
        source_time_s=np.interp(ts,raw['time_s'],raw['source_time_s']),reference_qpos=desired,root_xy_error_m=xy,root_z_error_m=ze,root_orientation_error_rad=oe,
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
    write_json(a.output/'report.json',report)
    print(json.dumps({'report':str(a.output/'report.json'),'success':report['success'],'simulated_s':ts[-1]}),flush=True)


if __name__=='__main__':main()
