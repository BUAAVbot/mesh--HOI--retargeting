"""Independent per-pose frozen-SONIC static tracking screen, not feasibility.

350 isolated GPU worlds start at source-reference poses with zero velocity.
They hold static targets for two seconds without resets or state repairs.
Contact forces, penetration, and continuous motion feasibility are not audited.
"""
from pathlib import Path
import argparse,hashlib,json,shutil,time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=2.);p.add_argument('--settle-window-s',type=float,default=.4)
    p.add_argument('--save-settled-candidates',action='store_true');a=p.parse_args()
    if a.output.exists()or a.seconds<=0 or not 0<a.settle_window_s<=a.seconds:raise ValueError('Choose fresh output and valid screen durations')
    import numpy as np,torch,mujoco as mj
    from rgb145_ppo_env import WarpTracking
    torch.set_num_threads(2);started=time.monotonic()
    source=np.load(a.reference,allow_pickle=False)
    if source['qpos'].shape!=(350,59)or source['frame_ids'].shape!=(350,):raise ValueError('Requires all350 source-reference poses')
    if not np.isfinite(source['qpos']).all():raise ValueError('Input target poses must be finite')
    a.output.mkdir(parents=True);shutil.copy2(a.reference,a.output/'original_source_reference.npz');shutil.copy2(__file__,a.output/'generator.py')
    starts=np.arange(350,dtype=float)*2.
    bank={'qpos':np.repeat(source['qpos'],2,axis=0),'dof_names':source['dof_names'],'frame_ids':np.repeat(source['frame_ids'],2),'source_time_s':np.repeat(source['source_time_s'],2),'time_s':np.c_[starts,starts+.99].ravel(),'diagnostic_not_source_motion':np.asarray(True),'diagnostic_reference_indexing_only':np.asarray(True)}
    for key in ('wrist_reference_joint_names','wrist_reference_joint_targets_rad'):
        if key in source:bank[key]=source[key]
    bank_path=a.output/'static_bank_reference.npz';np.savez_compressed(bank_path,**bank)
    env=WarpTracking(a.model,bank_path,350,145317,0.,True);wp=env.wp
    selected=torch.arange(350,device=env.device,dtype=torch.long)*2
    target=env.ref.q[selected].clone()
    # Exactly one declared initialization for each independently screened pose.
    mask=torch.ones(350,device=env.device,dtype=torch.bool)
    env.mw.reset_data(env.m,env.d,wp.from_torch(mask,dtype=wp.bool))
    env.q.copy_(target);env.v.zero_();env.phase.copy_(torch.as_tensor(starts,device=env.device,dtype=torch.float32))
    env.previous.zero_();env.steps.zero_();env.from_start.zero_();env.sonic.reset(torch.arange(350,device=env.device))
    env.ref.dq.zero_();env.sonic.dqf.zero_();env.sonic.c.body_dq[:]=0.
    env.u.copy_(env.ref.controls(env.phase,torch.zeros((350,28),device=env.device),0.))
    env.mw.forward(env.m,env.d);env.refresh_policy_sensors();wp.synchronize()
    initial_q=env.q.cpu().numpy().copy();initial_v=env.v.cpu().numpy().copy()
    wrist=[]
    for name,limits in env.ref.software_wrist_command_windows.items():
        aid=mj.mj_name2id(env.model,mj.mjtObj.mjOBJ_ACTUATOR,'motor_'+name);wrist.append((aid,float(limits[0]),float(limits[1])))
    count=int(np.ceil(a.seconds/env.dt));elapsed=[];errors=[];velocities=[];root_positions=[];root_quaternions=[];clock_errors=[];physical_finite=[]
    print(json.dumps({'stage':'static_tracking_screen_started','worlds':350,'seconds':a.seconds,'policy_dt_s':env.dt,'actual_physics_dt_s':float(env.model.opt.timestep),'reference_phase_is_fixed':True}),flush=True)
    for step in range(count):
        finite=torch.isfinite(env.q).all(1)&torch.isfinite(env.v).all(1)&torch.isfinite(env.rotation.flatten(1)).all(1)&torch.isfinite(env.cvel.flatten(1)).all(1)
        # Invalidate only inference caches to update history at every50Hz.
        # Already nonfinite worlds keep their last finite command; their real
        # physics state is never replaced, repaired, or restarted.
        env.sonic.phase.fill_(-1.);env.sonic.phase[~finite]=env.phase[~finite];env.sonic.valid[~finite]=True
        b=env.sonic.c.base
        with torch.inference_mode():
            base=env.sonic.targets(env.q,env.v,env.phase,env.rotation[:,b],env.cvel[:,b,:3])
            packet=env.ref.controls(env.phase,torch.zeros((350,28),device=env.device),0.);packet[:,env.sonic.aids]=base
            # Retain the frozen actor's original body commands. Only the
            # declared wrist software windows alter those28 target channels.
            for aid,lo,hi in wrist:packet[:,aid].clamp_(lo,hi)
            packet[:,env.ref.passive_t]=0.;env.u.copy_(packet);env.sonic.record_applied(packet)
            env.wp.capture_launch(env.graph);env.refresh_policy_sensors()
            delta=env.q[:,:3]-target[:,:3]
            actual=env.q[:,3:7]/env.q[:,3:7].norm(dim=1,keepdim=True).clamp_min(1e-9)
            wanted=target[:,3:7]/target[:,3:7].norm(dim=1,keepdim=True).clamp_min(1e-9)
            orientation=2*torch.acos((actual*wanted).sum(1).abs().clamp(0,1))
            body_rms=(env.q[:,env.ref.qa_t]-target[:,env.ref.qa_t]).square().mean(1).sqrt()
            metric=torch.stack([delta[:,:2].norm(dim=1),delta[:,2].abs(),orientation,body_rms],dim=1)
            speed=torch.stack([env.v[:,:3].norm(dim=1),env.v[:,3:6].norm(dim=1),env.v[:,env.ref.va_t].square().mean(1).sqrt()],dim=1)
        t=(step+1)*env.dt;elapsed.append(t);errors.append(metric.cpu().numpy());velocities.append(speed.cpu().numpy())
        root_positions.append(env.q[:,:3].cpu().numpy().copy());root_quaternions.append(env.q[:,3:7].cpu().numpy().copy())
        physical_finite.append((torch.isfinite(env.q).all(1)&torch.isfinite(env.v).all(1)).cpu().numpy())
        clock_errors.append((wp.to_torch(env.d.time)-t).abs().cpu().numpy())
        if(step+1)%25==0:print(json.dumps({'stage':'static_tracking_screen_progress','elapsed_s':t,'policy_steps':step+1,'total_policy_steps':count}),flush=True)
    wp.synchronize();final_q=env.q.cpu().numpy().copy();final_v=env.v.cpu().numpy().copy()
    error=np.asarray(errors);speed=np.asarray(velocities);ts=np.asarray(elapsed);position=np.asarray(root_positions);quats=np.asarray(root_quaternions);clock=np.asarray(clock_errors)
    threshold=np.asarray([.35,.20,.6,.6]);over=(error>threshold[None,None])|~np.isfinite(error)
    finite_final=np.isfinite(final_q).all(1)&np.isfinite(final_v).all(1)
    nonfinite=(~np.asarray(physical_finite))|(~np.isfinite(error).all(2))|(~np.isfinite(speed).all(2));bad_clock=clock>1e-4
    def first(mask):
        present=mask.any(0);out=np.full(350,np.nan);out[present]=ts[mask[:,present].argmax(0)];return out
    first_per_metric=np.stack([first(over[:,:,k])for k in range(4)],axis=1)
    first_any=first(over.any(2)|nonfinite|bad_clock)
    static_pass=np.isnan(first_any)&finite_final
    window=ts>=ts[-1]-a.settle_window_s-1e-9
    end_speed_mean=speed[window].mean(0);end_speed_max=np.max(speed[window],axis=0)
    drift=np.linalg.norm(position[-1]-position[np.flatnonzero(window)[0]],axis=1)
    qstart=quats[np.flatnonzero(window)[0]];qend=quats[-1]
    qstart=qstart/np.maximum(np.linalg.norm(qstart,axis=1)[:,None],1e-9);qend=qend/np.maximum(np.linalg.norm(qend,axis=1)[:,None],1e-9)
    angle_drift=2*np.arccos(np.clip(abs(np.sum(qstart*qend,axis=1)),0,1))
    heuristic={'root_linear_speed_mean_max_m_s':.10,'root_angular_speed_mean_max_rad_s':.25,'body_joint_rms_speed_mean_max_rad_s':.5,'root_position_drift_max_m':.05,'root_orientation_drift_max_rad':.10}
    settled=static_pass&(end_speed_mean[:,0]<=.10)&(end_speed_mean[:,1]<=.25)&(end_speed_mean[:,2]<=.5)&(drift<=.05)&(angle_drift<=.10)
    maximum=np.max(np.where(np.isfinite(error),error,np.inf),axis=0)
    def number(v):return float(v)if np.isfinite(v)else None
    rows=[]
    for i in range(350):
        rows.append({'frame_id':int(source['frame_ids'][i]),'source_time_s':float(source['source_time_s'][i]),'independently_initialized_pose':True,'static_tracking_screen_passed':bool(static_pass[i]),'settled_candidate_heuristic_passed':bool(settled[i]),'first_any_threshold_exceedance_elapsed_s':number(first_any[i]),'first_root_xy_exceedance_elapsed_s':number(first_per_metric[i,0]),'first_root_z_exceedance_elapsed_s':number(first_per_metric[i,1]),'first_root_orientation_exceedance_elapsed_s':number(first_per_metric[i,2]),'first_body_rms_exceedance_elapsed_s':number(first_per_metric[i,3]),'root_xy_error_max_m':number(maximum[i,0]),'root_z_error_max_m':number(maximum[i,1]),'root_orientation_error_max_rad':number(maximum[i,2]),'body_joint_rms_error_max_rad':number(maximum[i,3]),'final_state_finite':bool(finite_final[i]),'last_window_root_linear_speed_mean_m_s':number(end_speed_mean[i,0]),'last_window_root_linear_speed_max_m_s':number(end_speed_max[i,0]),'last_window_root_angular_speed_mean_rad_s':number(end_speed_mean[i,1]),'last_window_body_joint_rms_speed_mean_rad_s':number(end_speed_mean[i,2]),'last_window_root_position_drift_m':number(drift[i]),'last_window_root_orientation_drift_rad':number(angle_drift[i])})
    state_path=a.output/'screen_states.npz'
    np.savez_compressed(state_path,frame_ids=source['frame_ids'],source_time_s=source['source_time_s'],target_qpos=target.cpu().numpy(),initial_actual_qpos=initial_q,initial_actual_qvel=initial_v,final_actual_qpos=final_q,final_actual_qvel=final_v,elapsed_time_s=ts,tracking_errors=error,velocities=speed,root_position_world_m=position,root_quaternion_wxyz=quats,first_threshold_exceedance_elapsed_s=first_per_metric,first_any_exceedance_elapsed_s=first_any,static_tracking_screen_passed=static_pass,settled_candidate_heuristic_passed=settled,clock_error_s=clock,joint_names=source['dof_names'],not_continuous_trajectory=np.asarray(True))
    if a.save_settled_candidates:
        candidates=np.full_like(final_q,np.nan);candidates[settled]=final_q[settled]
        candidate_v=np.full_like(final_v,np.nan);candidate_v[settled]=final_v[settled]
        np.savez_compressed(a.output/'independent_settled_candidates.npz',frame_ids=source['frame_ids'],source_time_s=source['source_time_s'],qpos=candidates,qvel=candidate_v,valid=settled,independent_resets=np.asarray(True),not_continuous_trajectory=np.asarray(True),not_dynamic_feasibility=np.asarray(True))
    result={'schema':'rgb145.static_tracking_screen.v1','mode':'static_tracking_screen','not_continuous_trajectory':True,'not_dynamic_feasibility':True,'full_reproduction_passed':False,'worlds':350,'duration_s':float(ts[-1]),'source_frame_ids_preserved':True,'physics_dt_s':float(env.model.opt.timestep),'policy_dt_s':env.dt,'source_reference':{'path':str(a.reference.resolve()),'sha256':sha(a.reference)},'model_input':{'path':str(a.model.resolve()),'sha256':sha(a.model)},'generator_sha256':sha(__file__),'static_bank_sha256':sha(bank_path),'screen_states_sha256':sha(state_path),'frozen_sonic':env.sonic.c.report(),'policy_rows_evaluated':env.sonic.inference_count,'software_wrist_command_windows':env.ref.software_wrist_command_windows,'initial_velocity':'exactly_zero','physical_state_initializations':350,'physical_state_repairs_after_initialization':0,'reference_indexing':'Each pose occupies2i through2i+.99; phase fixed at2i; future+.9 remains that static pose; all future joint velocities zeroed. This clock is a diagnostic index only.','raw_source_human_fields':'Preserved unchanged in original_source_reference.npz','screen_thresholds':{'root_xy_m':.35,'root_z_m':.20,'root_orientation_rad':.6,'body_joint_rms_rad':.6},'last_window_s':a.settle_window_s,'settled_candidate_heuristic':heuristic,'static_tracking_passed_count':int(static_pass.sum()),'settled_candidate_count':int(settled.sum()),'first_failed_frame_ids':source['frame_ids'][~static_pass][:30].tolist(),'mujoco_version':mj.__version__,'wall_s':time.monotonic()-started,'limitations':['Independent static initialization at every source pose, not a continuous motion test.','No contact-force, support-surface, penetration, joint-limit or seating-load acceptance is performed.','A held pose within these tracking bounds is not a proof of whole-body dynamic feasibility or full reproduction.','GPU FP32 screen results require native validation before use as task acceptance.'],'phases':rows}
    (a.output/'screen_report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k]for k in ('mode','worlds','duration_s','static_tracking_passed_count','settled_candidate_count','wall_s','not_dynamic_feasibility','full_reproduction_passed')}),flush=True)


if __name__=='__main__':main()
