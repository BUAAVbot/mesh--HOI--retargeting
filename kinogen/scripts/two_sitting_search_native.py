"""Search frozen-policy native physical rollouts and export review candidates.

Every world starts at source frame0 with zero velocity and runs without resets.
No trajectory is called successful here: candidates require independent5ms
control-only replay. All failed trials remain in the full search summary.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_ready(x):
    if isinstance(x,dict):return {k:json_ready(v)for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [json_ready(v)for v in x]
    if isinstance(x,float)and not math.isfinite(x):return None
    return x


def write_json(path,value):
    Path(path).write_text(json.dumps(json_ready(value),indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','proxy-model','reference','checkpoint','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--num-envs',type=int,default=128);p.add_argument('--threads',type=int,default=16)
    p.add_argument('--seed',type=int,default=174114);p.add_argument('--std-scale',type=float,default=1.)
    p.add_argument('--tail-std-scale',type=float,default=0.)
    p.add_argument('--tail-seconds',type=float,default=1.)
    p.add_argument('--deterministic',action='store_true');p.add_argument('--top-k',type=int,default=16)
    p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if min(a.num_envs,a.threads,a.top_k)<1 or any(not math.isfinite(x)or x<0 for x in (a.std_scale,a.tail_std_scale,a.tail_seconds)):
        p.error('Positive counts and finite nonnegative noise scales/tail-seconds required')
    import numpy as np,torch,mujoco as mj
    from two_sitting_native_ppo_env import NativeTracking,ActorCritic
    torch.set_num_threads(2)
    checkpoint=torch.load(a.checkpoint,map_location=a.device,weights_only=False)
    if checkpoint.get('action_dim')!=56:raise ValueError('Expected56-output frozen residual actor')
    if checkpoint.get('reference_sha256')!=sha(a.reference):raise ValueError('Checkpoint/reference hash mismatch')
    if checkpoint.get('model_sha256')!=sha(a.model):raise ValueError('Checkpoint/model XML hash mismatch')
    a.output.mkdir(parents=True);sources=a.output/'source_snapshot';sources.mkdir()
    for name in ('two_sitting_search_native.py','two_sitting_native_ppo_env.py','two_sitting_ppo_env.py','two_sitting_batch_sonic.py','fourth_sonic_controller.py'):
        shutil.copyfile(Path(__file__).with_name(name),sources/name)
    rules={'minimum_final_seat_upward_force_fraction_of_weight':.10,'last_window_seconds':.20,
           'minimum_last_window_seat_support_fraction':.90,'last_window_root_linear_speed_m_s_max':.15,
           'last_window_root_angular_speed_rad_s_max':.30,'final_root_horizontal_error_m_max':.30,
           'minimum_each_actor_exchange_progress_m':1.20,'minimum_each_actor_root_rise_m':.12,
           'minimum_root_height_all_sampled_states_m':.45,'maximum_root_upright_tilt_all_sampled_states_rad':.80,
           'maximum_sampled_contact_penetration_m':.025,'numerical_warning_or_time_reset_allowed':False,
           'full_source_tracking_required_for_preliminary_selection':False,
           'disclosure':'Functional physical-demo screening only. Source tracking error is reported separately; all criteria require independent5ms replay before acceptance.'}
    search_id=a.output.name
    config={'schema':'two_sitting_native_policy_search.v1','status':'pending_full_5ms_replay','search_id':search_id,
            'args':{k:str(v)if isinstance(v,Path)else v for k,v in vars(a).items()},'physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),
            'checkpoint_iteration':checkpoint.get('iteration'),'checkpoint_sha256':sha(a.checkpoint),
            'reference_sha256':sha(a.reference),'input_model_xml_sha256':sha(a.model),'proxy_model_sha256':sha(a.proxy_model),
            'selectionrules':rules,'total_attempts':a.num_envs,'seed':a.seed,'std_scale':a.std_scale,
            'sampling':'deterministic actor mean'if a.deterministic else'frozen Normal(mean,clamped_checkpoint_std*scheduled_std_scale); independent56-dimensional samples each policy update for all worlds',
            'source_files':{f.name:sha(f)for f in sources.iterdir()},
            'prohibition':'No reset after initialization, external force, root actuation, model alteration during rollout, weight update, or automatic success claim.'}
    write_json(a.output/'run_config.json',config)
    print(json.dumps({'stage':'construct_native_persistent_worlds','worlds':a.num_envs,'threads':a.threads,'checkpoint_iteration':checkpoint.get('iteration')}),flush=True)
    env=NativeTracking(a.model,a.reference,a.num_envs,a.seed,float(checkpoint['action_scale']),bool(checkpoint.get('sonic_base',False)),proxy_path=a.proxy_model,threads=a.threads,device=a.device)
    actor=ActorCritic(int(checkpoint['obs_dim']),action_dim=56).to(a.device)
    actor.load_state_dict(checkpoint['model'],strict=True);actor.eval()
    for param in actor.parameters():param.requires_grad_(False)
    torch.manual_seed(a.seed)
    if str(a.device).startswith('cuda'):torch.cuda.manual_seed_all(a.seed)
    env.reset(torch.arange(env.n,device=a.device),force_start=True)
    if any(np.any(d.qvel)for d in env.datas):raise ValueError('All force-start velocities must be zero')
    reset_events_before=env.state_write_reset_events
    state_spec=mj.mjtState.mjSTATE_INTEGRATION
    initial=np.empty((env.n,mj.mj_stateSize(env.model,state_spec)),dtype=np.float64)
    for i,d in enumerate(env.datas):mj.mj_getState(env.model,d,initial[i],state_spec)
    binary=a.output/'model.mjb';mj.mj_saveModel(env.model,str(binary),None)
    binary_hash=sha(binary)
    weight=[]
    for prefix in ('p1_','p2_'):
        ids=[b for b in range(env.model.nbody)if(mj.mj_id2name(env.model,mj.mjtObj.mjOBJ_BODY,b)or'').startswith(prefix)]
        weight.append(float(env.model.body_mass[ids].sum()*abs(env.model.opt.gravity[2])))
    weight=np.asarray(weight)
    if np.any(weight<=0):raise ValueError('Missing positive actor weight')
    config.update(model_sha256=binary_hash,model_binary_sha256=binary_hash,model_mjb_path=str(binary.resolve()),model_mujoco_version=mj.__version__,actor_weight_N=weight.tolist())
    tail_start=max(float(env.ref.end),float(env.ref.horizon)-a.tail_seconds)
    config['noise_schedule']={'motion_std_scale':0. if a.deterministic else a.std_scale,
                              'tail_std_scale':0. if a.deterministic else a.tail_std_scale,
                              'requested_tail_seconds':a.tail_seconds,
                              'effective_tail_seconds':max(0.,float(env.ref.horizon)-tail_start),
                              'tail_start_execution_seconds':tail_start,'reference_end_execution_seconds':float(env.ref.end),
                              'unchanged_horizon_execution_seconds':float(env.ref.horizon),
                              'semantics':'Policy-time step schedule. Tail is restricted to the existing post-reference hold; source motion timing, hold duration, selection thresholds, checkpoint and engine are unchanged.'}
    write_json(a.output/'run_config.json',config)
    def live():return np.stack([d.qpos.copy()for d in env.datas]),np.stack([d.qvel.copy()for d in env.datas])
    q0,v0=live();qs=[q0];vs=[v0];controls=[];actions=[];seat=[];pen=[];cross=[];done=[];fault=np.zeros(env.n,bool);faultsteps=np.full(env.n,-1,int);first_training_fail=np.full(env.n,-1,int)
    rows=[];phase=[];noise_scales=[];obs=None;wall=time.monotonic();cycles=int(np.ceil(env.ref.horizon/env.dt-1e-9));numeric_sanitized=np.zeros(env.n,int)
    try:
        with torch.inference_mode():
            obs=env.observations()
            for step in range(cycles):
                distribution=actor.distribution(actor.normalize(obs));mean=distribution.mean
                scale=torch.where(env.phase>=tail_start,a.tail_std_scale,a.std_scale)
                if a.deterministic:scale=torch.zeros_like(scale)
                action=mean if a.deterministic else mean+scale[:,None]*distribution.stddev*torch.randn_like(mean)
                noise_scales.append(scale.detach().cpu().numpy().copy())
                finite_action=torch.isfinite(action).all(-1)
                if not finite_action.all():
                    ids=(~finite_action).detach().cpu().numpy();fault|=ids;numeric_sanitized[ids]+=1
                    action=torch.nan_to_num(action,nan=0.,posinf=0.,neginf=0.)
                nxt,reward,finished,info=env.step(action)
                old=env._control_old_cpu.copy();new=env._control_new_cpu.copy()
                controls.extend((old.copy(),old.copy(),old.copy(),new.copy()))
                q,v=live();qs.append(q);vs.append(v);actions.append(action.detach().cpu().numpy().copy())
                seat.append(env._seat_cpu.copy());pen.append(env._penetration_cpu.copy());cross.append(env._cross_cpu.copy());phase.append(float((step+1)*env.dt))
                finite=np.isfinite(q).all(1)&np.isfinite(v).all(1)
                clock=np.array([d.time for d in env.datas]);wrongclock=abs(clock-(step+1)*env.dt)>1e-7
                nowfault=env._numeric_fault.copy()|~finite|wrongclock
                faultsteps[(faultsteps<0)&nowfault]=step;fault|=nowfault
                failed=info['failed'].detach().cpu().numpy();done.append(failed)
                first_training_fail[(first_training_fail<0)&failed]=step
                if env.state_write_reset_events!=reset_events_before:raise ValueError('Unexpected episode reset during physical search')
                obs=nxt
                if step%50==0:print(json.dumps({'policy_step':step+1,'time_s':phase[-1],'numeric_fault_worlds':int(fault.sum()),'current_both_seated_worlds':int(np.all(env._seat_cpu>=.10*weight,axis=1).sum()),'root_xy_error_max_m':float(info['root_xy'].max())}),flush=True)
    finally:
        env.close()
    if not controls:raise RuntimeError('No native physical step executed')
    q=np.asarray(qs);v=np.asarray(vs);u=np.asarray(controls);seat=np.asarray(seat);pen=np.asarray(pen);cross=np.asarray(cross);phase=np.asarray(phase)
    np.savez_compressed(a.output/'all_rollouts.npz',initial_state=initial,state_spec=int(state_spec),ctrl=u,qpos_policy=q,qvel_policy=v,
        policy_time_s=np.r_[0.,phase],physical_control_time_s=np.arange(len(u))*.005,actions=np.asarray(actions),policy_std_scale=np.asarray(noise_scales),seat_upward_force_N=seat,
        sampled_max_penetration_m=pen,sampled_cross_actor_contact_penalty=cross,training_failed_flag=np.asarray(done),numeric_fault_world=fault,
        root_qadr=env.ref.root_qadr,root_vadr=env.ref.root_vadr,metadata=np.asarray(json.dumps(config)))
    actual_duration=len(u)*.005;full=actual_duration>=env.ref.horizon-.005
    qref=env.ref.q[-1].detach().cpu().numpy();window=max(1,int(np.ceil(.20/env.dt)))
    roots=np.stack([q[:,:,int(rq):int(rq)+3]for rq in env.ref.root_qadr],axis=2)
    quats=np.stack([q[:,:,int(rq)+3:int(rq)+7]for rq in env.ref.root_qadr],axis=2)
    norm=np.linalg.norm(quats,axis=-1);zup=1-2*(quats[:,:,:,1]**2+quats[:,:,:,2]**2)/np.maximum(norm**2,1e-12)
    tilt=np.arccos(np.clip(zup,-1,1));linear=np.stack([np.linalg.norm(v[-window:,:,int(rv):int(rv)+3],axis=-1)for rv in env.ref.root_vadr],axis=-1)
    angular=np.stack([np.linalg.norm(v[-window:,:,int(rv)+3:int(rv)+6],axis=-1)for rv in env.ref.root_vadr],axis=-1)
    finaltargets=np.stack([qref[int(rq):int(rq)+3]for rq in env.ref.root_qadr])
    finalerror=np.linalg.norm(roots[-1,:,:,:2]-finaltargets[None,:,:2],axis=-1)
    axis=roots[0,0,1,:2]-roots[0,0,0,:2];axis/=max(np.linalg.norm(axis),1e-12)
    progress=np.stack([(roots[-1,:,0,:2]-roots[0,:,0,:2])@axis,(roots[-1,:,1,:2]-roots[0,:,1,:2])@(-axis)],axis=-1)
    rise=np.max(roots[:,:,:,2]-roots[0:1,:,:,2],axis=0)
    seatfraction=(seat[-window:]>=.10*weight[None,None,:]).mean(0)
    maxxy=np.zeros((env.n,2))
    for step in range(len(q)):
        tt=min(step*env.dt,float(env.ref.end));i,j,w=env.ref.indices(torch.tensor([tt],device=env.device));refq=(env.ref.q[i]+w[:,None]*(env.ref.q[j]-env.ref.q[i]))[0].detach().cpu().numpy()
        goal=np.stack([refq[int(rq):int(rq)+2]for rq in env.ref.root_qadr]);maxxy=np.maximum(maxxy,np.linalg.norm(roots[step,:,:,:2]-goal[None],axis=-1))
    for candidate in range(env.n):
        checks={'full_duration':full,'no_numeric_fault':not bool(fault[candidate]),
                'both_final_seated_with_weight_fraction':bool(np.all(seatfraction[candidate]>=.90)),
                'last_0p2s_linear_speed':bool(np.max(linear[:,candidate])<.15),'last_0p2s_angular_speed':bool(np.max(angular[:,candidate])<.30),
                'both_final_root_xy':bool(np.all(finalerror[candidate]<.30)),'both_exchange_progress':bool(np.all(progress[candidate]>1.20)),
                'both_root_rise':bool(np.all(rise[candidate]>.12)),'no_sampled_fall':bool(np.min(roots[:,candidate,:,2])>.45),
                'sampled_root_tilt':bool(np.max(tilt[:,candidate])<.80),'sampled_penetration':bool(np.max(pen[:,candidate])<.025)}
        preliminary=all(checks.values())
        score=float(sum(checks.values())+np.minimum(progress[candidate],1.8).sum()-.5*finalerror[candidate].sum()+seatfraction[candidate].sum()-np.maximum(tilt[:,candidate].max()-.8,0))
        if fault[candidate]or not math.isfinite(score):score=-1e6+sum(checks.values())
        rows.append({'candidate_id':candidate,'search_id':search_id,'status':'pending_full_5ms_replay','preliminary_selected':preliminary,
                     'selection_checks':checks,'ranking_score':score,'final_root_xy_error_m':finalerror[candidate].tolist(),
                     'maximum_source_root_xy_error_m':maxxy[candidate].tolist(),'exchange_progress_m':progress[candidate].tolist(),'root_rise_m':rise[candidate].tolist(),
                     'minimum_root_height_m':np.min(roots[:,candidate,:,2],axis=0).tolist(),'maximum_root_tilt_rad':np.max(tilt[:,candidate],axis=0).tolist(),
                     'sampled_max_penetration_m':float(np.max(pen[:,candidate])),'last_0p2s_seat_weight_support_fraction':seatfraction[candidate].tolist(),
                     'last_0p2s_root_linear_speed_peak_m_s':np.max(linear[:,candidate],axis=0).tolist(),'last_0p2s_root_angular_speed_peak_rad_s':np.max(angular[:,candidate],axis=0).tolist(),
                     'final_seat_upward_force_N':seat[-1,candidate].tolist(),'first_training_failed_policy_step':int(first_training_fail[candidate]),
                     'numeric_fault':bool(fault[candidate]),'first_numeric_fault_policy_step':int(faultsteps[candidate]),'invalid_action_sanitized_count':int(numeric_sanitized[candidate])})
    ranked=sorted(rows,key=lambda row:(row['preliminary_selected'],row['ranking_score']),reverse=True);top=ranked[:min(a.top_k,env.n)];folder=a.output/'candidates';folder.mkdir()
    for rank,row in enumerate(top):
        candidate=int(row['candidate_id']);name=f'candidate_{candidate:04d}.npz'
        metadata={**config,**row,'rank':rank,'total_attempts':env.n,'numenvs':env.n,'seed':a.seed,'simulated_seconds':actual_duration,
                  'qpos_policy_sampling':'FP64 native states every4 physics steps; independent replay should compare every4steps.',
                  'not_a_final_pass':True,'model_path':str(binary.resolve()),'reference_path':str(a.reference.resolve()),'checkpoint_path':str(a.checkpoint.resolve())}
        np.savez_compressed(folder/name,initial_state=initial[candidate],state_spec=int(state_spec),ctrl=u[:,candidate],
            qpos_policy=q[:,candidate],qvel_policy=v[:,candidate],policy_time_s=np.r_[0.,phase],
            learned_residual_action=np.asarray(actions)[:,candidate],
            policy_std_scale=np.asarray(noise_scales)[:,candidate],
            seat_upward_force_N=seat[:,candidate],sampled_max_penetration_m=pen[:,candidate],metadata=np.asarray(json.dumps(json_ready(metadata),allow_nan=False)))
        row['candidate_file']=str((folder/name).resolve());row['candidate_sha256']=sha(folder/name)
    summary={**config,'status':'pending_full_5ms_replay','full_source_tracking_pass_claimed':False,'independent_physical_success_count':None,
             'preliminary_selected_count':sum(row['preliminary_selected']for row in rows),'total_attempts':env.n,'sampled_selection_fraction':sum(row['preliminary_selected']for row in rows)/env.n,
             'simulated_seconds_per_world':actual_duration,'physical_steps_per_world':len(u),'policy_steps_per_world':len(q)-1,
             'wall_seconds':time.monotonic()-wall,'no_resets_after_initialization':True,'numeric_fault_world_count':int(fault.sum()),
             'statistics_disclosure':'All attempts below, including failures. Preliminary selection fraction is not a success rate. Only independent full5ms replay may establish acceptance.',
             'backend':env.backend_report(),'top_candidate_ids':[row['candidate_id']for row in top],'all_attempts':rows,'ranked_top_candidates':top}
    write_json(a.output/'summary.json',summary)
    print(json.dumps({'status':summary['status'],'total_attempts':env.n,'preliminary_selected_count':summary['preliminary_selected_count'],'simulated_seconds':actual_duration,'numeric_fault_world_count':int(fault.sum()),'top_candidate_ids':summary['top_candidate_ids'],'summary':str(a.output/'summary.json')}),flush=True)


if __name__=='__main__':main()
