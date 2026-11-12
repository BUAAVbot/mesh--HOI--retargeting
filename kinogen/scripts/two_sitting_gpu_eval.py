"""Deterministic, uninterrupted GPU evaluation of a dual-Vita residual policy.

GPU completion is not native MuJoCo acceptance. All physical qpos/qvel/control
samples are saved. Training termination flags are recorded but never reset an
actor or stop this evaluation; only numerical failure or root safety does.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    def clean(x):
        if isinstance(x,dict):return {k:clean(v)for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [clean(v)for v in x]
        if isinstance(x,float)and not __import__('math').isfinite(x):return None
        return x
    Path(path).write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('model','proxy-model','reference','checkpoint','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--num-envs',type=int,default=1);p.add_argument('--seed',type=int,default=174114)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if a.num_envs<1:p.error('Positive num-envs required')
    import numpy as np,torch,mujoco as mj
    from two_sitting_ppo_env import WarpTracking,ActorCritic
    torch.set_num_threads(2)
    checkpoint=torch.load(a.checkpoint,map_location='cuda:0',weights_only=False)
    if checkpoint.get('action_dim')!=56:raise ValueError('Expected56-output dual residual actor')
    if checkpoint.get('reference_sha256')!=sha(a.reference):raise ValueError('Checkpoint/reference hash mismatch')
    if checkpoint.get('model_sha256')!=sha(a.model):raise ValueError('Checkpoint/model XML hash mismatch')
    a.output.mkdir(parents=True)
    source=a.output/'source_snapshot';source.mkdir()
    for name in ('two_sitting_gpu_eval.py','two_sitting_ppo_env.py','two_sitting_batch_sonic.py','fourth_sonic_controller.py'):
        shutil.copyfile(Path(__file__).with_name(name),source/name)
    config={'status':'gpu_only_pending_native','device':'cuda:0','physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),
            'args':{k:str(v)if isinstance(v,Path)else v for k,v in vars(a).items()},
            'checkpoint_iteration':checkpoint.get('iteration'),'action_scale':checkpoint['action_scale'],
            'inputs':{key:{'path':str(getattr(a,key)),'sha256':sha(getattr(a,key))}for key in ('model','proxy_model','reference','checkpoint')},
            'sources':{name.name:sha(name)for name in source.iterdir()},
            'deterministic_action':'actor.actor(actor.normalize(env.observations())); frozen checkpoint observation statistics',
            'reset_contract':'Force source frame0 and zero qvel once before evaluation; no subsequent reset.',
            'training_done_handling':'Flags retained, never used to stop/reset this evaluation.',
            'trace_contract':'qpos/qvel/control at every5ms physical substep; policy observations/actions and diagnostics at20ms.'}
    write_json(a.output/'run_config.json',config)
    print(json.dumps({'stage':'create_gpu_eval_environment','num_envs':a.num_envs}),flush=True)
    env=WarpTracking(a.model,a.reference,a.num_envs,a.seed,float(checkpoint['action_scale']),bool(checkpoint.get('sonic_base',False)),proxy_path=a.proxy_model)
    actor=ActorCritic(int(checkpoint['obs_dim']),action_dim=56).to(env.device)
    actor.load_state_dict(checkpoint['model'],strict=True);actor.eval()
    for param in actor.parameters():param.requires_grad_(False)
    # The three-step graph has the same mw.step operations as training. Device
    # copies add read-only substep snapshots, with no extra forward/solve call.
    traceq=torch.empty((3,env.n,env.model.nq),device=env.device)
    tracev=torch.empty((3,env.n,env.model.nv),device=env.device)
    traceu=torch.empty((3,env.n,env.model.nu),device=env.device)
    qviews=[env.wp.from_torch(traceq[k])for k in range(3)]
    vviews=[env.wp.from_torch(tracev[k])for k in range(3)]
    uviews=[env.wp.from_torch(traceu[k])for k in range(3)]
    with env.wp.ScopedCapture()as capture:
        for k in range(3):
            env.mw.step(env.m,env.d)
            env.wp.copy(qviews[k],env.d.qpos)
            env.wp.copy(vviews[k],env.d.qvel)
            env.wp.copy(uviews[k],env.d.ctrl)
    env.graph3=capture.graph
    env.reset(torch.arange(env.n,device=env.device),force_start=True)
    if torch.any(env.v!=0):raise ValueError('Force-start evaluation must have zero initial velocity')
    env.wp.synchronize()
    mj.mj_saveModel(env.model,str(a.output/'model.mjb'),None)
    q=[env.q.detach().cpu().numpy().copy()];v=[env.v.detach().cpu().numpy().copy()];u=[]
    actions=[];observations=[];policytime=[];xy=[];ze=[];ori=[];seat=[];pen=[];cross=[];doneflags=[];failedflags=[];rewards=[]
    initialq=q[0].copy();initialv=v[0].copy();initialctrl=env.u.detach().cpu().numpy().copy()
    nominal=[];applied=[];reason='completed_reference_and_hold';wall=time.monotonic();cycles=int(np.ceil(env.ref.horizon/env.dt-1e-9))
    with torch.inference_mode():
        obs=env.observations()
        for cycle in range(cycles):
            phase=env.phase.detach().cpu().numpy().copy()
            if not torch.isfinite(obs).all():reason='nonfinite_policy_observation';break
            action=actor.actor(actor.normalize(obs))
            if not torch.isfinite(action).all():reason='nonfinite_policy_action';break
            observations.append(obs.detach().cpu().numpy().copy());actions.append(action.detach().cpu().numpy().copy());policytime.append(phase)
            # Clone the nominal teacher before step; this is exactly the cached
            # teacher packet subsequently consumed by env.step at this phase.
            baseline=env.base_controls().clone()
            command=baseline.clone();command[:,env.ref.body_aids_t]+=env.action_scale*torch.tanh(action)
            nominal.append(command.detach().cpu().numpy().copy())
            nxt,reward,done,info=env.step(action)
            env.wp.synchronize()
            q.extend(traceq.detach().cpu().numpy().copy());q.append(env.q.detach().cpu().numpy().copy())
            v.extend(tracev.detach().cpu().numpy().copy());v.append(env.v.detach().cpu().numpy().copy())
            u.extend(traceu.detach().cpu().numpy().copy());u.append(env.u.detach().cpu().numpy().copy())
            target,_,_=env.ref.sample(env.phase)
            heights=[];angles=[]
            for rq in env.ref.root_qadr:
                heights.append((env.q[:,rq+2]-target[:,rq+2]).abs())
                actual=env.q[:,rq+3:rq+7];desired=target[:,rq+3:rq+7]
                cosine=(actual*desired).sum(-1).abs()/(actual.norm(dim=-1)*desired.norm(dim=-1)).clamp_min(1e-9)
                angles.append(2*torch.acos(cosine.clamp(0,1)))
            xy.append(info['root_xy_per_actor'].detach().cpu().numpy().copy())
            ze.append(torch.stack(heights,-1).detach().cpu().numpy().copy());ori.append(torch.stack(angles,-1).detach().cpu().numpy().copy())
            seat.append(info['seat_support_force_N'].detach().cpu().numpy().copy())
            pen.append(info['max_penetration_m'].detach().cpu().numpy().copy())
            cross.append(info['cross_actor_contact_penalty'].detach().cpu().numpy().copy())
            doneflags.append(done.detach().cpu().numpy().copy());failedflags.append(info['failed'].detach().cpu().numpy().copy());rewards.append(reward.detach().cpu().numpy().copy())
            obs=nxt
            if not torch.isfinite(env.q).all()or not torch.isfinite(env.v).all():reason='nonfinite_physical_state';break
            if any(bool(torch.any(env.q[:,int(rq)+2]<.15))for rq in env.ref.root_qadr):reason='root_below_0p15m';break
            if cycle%100==0:print(json.dumps({'cycle':cycle,'time_s':float(env.phase[0]),'root_xy_max_m':float(info['root_xy'].max()),'max_penetration_m':float(info['max_penetration_m'].max())}),flush=True)
    q=np.asarray(q);v=np.asarray(v);u=np.asarray(u);xy=np.asarray(xy);ze=np.asarray(ze);ori=np.asarray(ori);seat=np.asarray(seat);pen=np.asarray(pen);cross=np.asarray(cross)
    duration=len(u)*float(env.model.opt.timestep);physicaltime=np.arange(len(q))*float(env.model.opt.timestep)
    np.savez_compressed(a.output/'trace.npz',time=physicaltime,qpos=q,qvel=v,ctrl=u,
        policy_time_s=np.asarray(policytime),policy_observations=np.asarray(observations),deterministic_residual_actions=np.asarray(actions),
        nominal_pre_limit_pre_delay_controls=np.asarray(nominal),root_horizontal_error_m=xy,root_height_error_m=ze,root_orientation_error_rad=ori,
        seat_upward_force_N=seat,max_penetration_m=pen,cross_actor_contact_penalty=cross,training_done_flag=np.asarray(doneflags),training_failed_flag=np.asarray(failedflags),reward=np.asarray(rewards),
        robot_root_qadr=env.ref.root_qadr,robot_root_vadr=env.ref.root_vadr,body_actuator_ids=env.ref.body_aids,all_actuator_ids=env.ref.all_aids,
        initial_qpos=initialq,initial_qvel=initialv,initial_ctrl=initialctrl,
        reference_qpos=env.ref.q.detach().cpu().numpy(),reference_time_s=env.ref.times.detach().cpu().numpy(),source_time_s=env.ref.source_times.detach().cpu().numpy())
    if not len(xy):raise RuntimeError('No physical cycle executed')
    finalforce=seat[-min(10,len(seat)):];finalfraction=(finalforce>1.).mean(0)
    checks={'completed_full_reference_and_hold':reason=='completed_reference_and_hold'and duration>=env.ref.horizon-.005,
            'root_xy_below_0p25m':bool(np.max(xy)<=.25),'root_z_below_0p18m':bool(np.max(ze)<=.18),
            'root_orientation_below_0p65rad':bool(np.max(ori)<=.65),'penetration_below_0p025m':bool(np.max(pen)<=.025),
            'final_both_seated_upward_force':bool(np.all(finalfraction>=.9))}
    report={**config,'status':'gpu_only_pending_native','gpu_quality_passed':all(checks.values()),'gpu_checks':checks,
            'termination':reason,'simulated_seconds':duration,'source_seconds_covered':min(float(env.ref.source_clock(env.phase).max()),float(env.ref.source_times[-1])),
            'physics_steps':len(u),'policy_updates':len(actions),'wall_seconds':time.monotonic()-wall,
            'max_root_xy_error_m_per_actor':np.max(xy,axis=(0,1)).tolist(),'max_root_z_error_m_per_actor':np.max(ze,axis=(0,1)).tolist(),
            'max_root_orientation_error_rad_per_actor':np.max(ori,axis=(0,1)).tolist(),'max_penetration_m':float(np.max(pen)),
            'last_root_xy_error_m':xy[-1].tolist(),'final_0p2s_upward_seat_force_fraction':finalfraction.tolist(),
            'final_upward_seat_force_N':seat[-1].tolist(),'final_root_positions_world_m':[[q[-1,k,int(rq):int(rq)+3].tolist()for rq in env.ref.root_qadr]for k in range(env.n)],
            'training_done_flags_ignored_count':int(np.sum(doneflags)),'no_reset_after_initialization':True,
            'state_writes_after_initialization':0,'applied_external_force_steps':0,'root_actuator_count':0,
            'physics_sensor_contract':env.policy_sensor_contract,'contact_force_contract':env.contact_contract,
            'native_validation_passed':False,'native_validation_status':'not_performed_in_this_gpu_evaluation',
            'trace_sampling':'All qpos/qvel/ctrl every5ms; graph3 identical3 mw.step operations plus read-only device snapshots.',
            'artifacts':{name:{'path':str(a.output/name),'sha256':sha(a.output/name)}for name in ('model.mjb','trace.npz')}}
    write_json(a.output/'report.json',report)
    print(json.dumps({k:report[k]for k in ('status','gpu_quality_passed','termination','simulated_seconds','max_root_xy_error_m_per_actor','max_root_orientation_error_rad_per_actor','max_penetration_m','final_0p2s_upward_seat_force_fraction')}),flush=True)


if __name__=='__main__':main()
