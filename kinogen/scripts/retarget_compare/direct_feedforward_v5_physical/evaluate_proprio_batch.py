"""Screen a specialist checkpoint on deterministic closed-loop native worlds without resets."""
from __future__ import annotations
import argparse, hashlib, json, time, os
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--initial-state',type=Path)
    parser.add_argument('--num-envs',type=int,default=8);parser.add_argument('--seed',type=int,default=88001)
    parser.add_argument('--joint-noise',type=float,default=.01);parser.add_argument('--root-noise',type=float,default=.01)
    args=parser.parse_args()
    import numpy as np,mujoco as mj,torch
    from direct_proprio import DirectProprioEnv as NativeTracking
    from two_sitting_ppo_env import ActorCritic
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    torch.set_num_threads(2);checkpoint=torch.load(args.checkpoint,map_location='cuda:0',weights_only=False)
    initial=args.initial_state or checkpoint.get('fixed_initial_state_path')
    if initial:os.environ['DIRECT_INITIAL_STATE_PATH']=str(initial)
    env=NativeTracking(Path(checkpoint['model_path']),Path(checkpoint['reference_path']),n=args.num_envs,
        seed=args.seed,action_scale=checkpoint['action_scale'],sonic_base=checkpoint['sonic_base'],threads=min(8,args.num_envs),
        proxy_path=Path('/data/KinoGen/outputs/two_sitting_pipeline/assets_deploy_v3/vita_single_proxy.xml'))
    env.reset(torch.arange(env.n,device=env.device),force_start=True)
    rng=np.random.default_rng(args.seed);initial=[]
    for i,data in enumerate(env.datas):
        if i:
            data.qpos[env.ref.qa]+=rng.uniform(-args.joint_noise,args.joint_noise,len(env.ref.qa))
            data.qpos[env.ref.qa]=np.clip(data.qpos[env.ref.qa],env.model.jnt_range[env.ref.jids,0],env.model.jnt_range[env.ref.jids,1])
            for rq in env.ref.root_qadr:data.qpos[rq:rq+2]+=rng.uniform(-args.root_noise,args.root_noise,2)
        mj.mj_forward(env.model,data);env._cache_one(i)
        initial.append(data.qpos.copy())
    env._generation+=1;env._host_generation=env._generation;env.refresh_policy_sensors()
    actor=ActorCritic(checkpoint['obs_dim'],action_dim=56).to(env.device)
    actor.load_state_dict(checkpoint['model']);actor.eval()
    obs=env.observations();began=time.monotonic();history={k:[]for k in ('q','v','xy','z','ori','pose','pen','seat','numeric')}
    for step in range(int(np.ceil(env.ref.horizon/.02-1e-8))):
        with torch.no_grad():action=actor.actor(actor.normalize(obs));obs,reward,done,info=env.step(action)
        target,_,_=env.ref.sample(env.phase)
        z=torch.stack([(env.q[:,rq+2]-target[:,rq+2]).abs()for rq in env.ref.root_qadr],1)
        for key,value in [('q',env.q),('v',env.v),('xy',info['root_xy_per_actor']),
                          ('z',z),('ori',info['root_angle_per_actor']),('pose',info['pose_mse']),
                          ('pen',info['max_penetration_m']),('seat',info['seat_support_force_N']),
                          ('numeric',torch.as_tensor(env._numeric_fault,device=env.device))]:
            history[key].append(value.detach().cpu().numpy().copy())
    h={key:np.asarray(value)for key,value in history.items()}
    q,v=h['q'],h['v'];tail=slice(-10,None)
    heights=np.stack([q[:,:,rq+2]for rq in env.ref.root_qadr],-1)
    tilt=np.stack([np.arccos(np.clip(1-2*(q[:,:,rq+4]**2+q[:,:,rq+5]**2),-1,1))for rq in env.ref.root_qadr],-1)
    weight=np.asarray([env.model.body_subtreemass[m['base']]*9.81 for m in env.ref.maps])
    linear=np.stack([np.linalg.norm(v[tail,:,rv:rv+3],axis=-1).max(0)for rv in env.ref.root_vadr],-1)
    angular=np.stack([np.linalg.norm(v[tail,:,rv+3:rv+6],axis=-1).max(0)for rv in env.ref.root_vadr],-1)
    initial=np.asarray(initial);travel=np.stack([np.linalg.norm(q[-1,:,rq:rq+2]-initial[:,rq:rq+2],axis=-1)for rq in env.ref.root_qadr],-1)
    rise=heights.max(0)-np.stack([initial[:,rq+2]for rq in env.ref.root_qadr],-1)
    seatforce=h['seat'][tail].mean(0);fraction=(h['seat'][tail]>1.).mean(0)
    results=[]
    for i in range(env.n):
        checks=dict(full_duration=True,no_fall=bool(heights[:,i].min()>.45 and tilt[:,i].max()<.8),
            penetration_below_25mm=bool(h['pen'][:,i].max()<.025),both_rise=bool(np.all(rise[i]>.12)),
            exchanged_seats=bool(np.all(travel[i]>1.2)and np.all(h['xy'][-1,i]<.30)),
            final_actual_seat_contact=bool(np.all(fraction[i]>=.5)),
            final_seat_load=bool(np.all(seatforce[i]>=.10*weight)),
            final_stable=bool(np.all(linear[i]<.15)and np.all(angular[i]<.30)),
            finite=bool(np.isfinite(q[:,i]).all()and np.isfinite(v[:,i]).all()),
            no_numeric_warnings=bool(not h['numeric'][:,i].any()))
        strict=bool(all(checks.values())and h['xy'][:,i].max()<.25 and h['z'][:,i].max()<.18 and h['ori'][:,i].max()<.65)
        results.append(dict(index=i,functional_pass=all(checks.values()),strict_pass=strict,checks=checks,
            root_xy_rms_m=float(np.sqrt(np.mean(h['xy'][:,i]**2))),root_xy_max_m=h['xy'][:,i].max(0).tolist(),
            final_xy_m=h['xy'][-1,i].tolist(),max_penetration_m=float(h['pen'][:,i].max()),
            seat_force_N=seatforce[i].tolist(),seat_contact_fraction=fraction[i].tolist(),
            final_linear_speed_m_s=linear[i].tolist(),final_angular_speed_rad_s=angular[i].tolist(),
            min_root_height_m=float(heights[:,i].min()),max_tilt_rad=float(tilt[:,i].max())))
    report=dict(status='checkpoint_screen_complete_not_independent_replay',checkpoint=str(args.checkpoint),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),iteration=checkpoint['iteration'],
        expert_observation_contract=checkpoint.get('expert_observation_contract'),fixed_initial_state_path=str(initial),
        reference=checkpoint['reference_path'],seed=args.seed,num_envs=env.n,
        perturbation={'unperturbed_trial':0,'other_trials_joint_noise_rad':args.joint_noise,
                      'other_trials_root_xy_noise_m':args.root_noise},
        functional_passes=sum(r['functional_pass']for r in results),strict_passes=sum(r['strict_pass']for r in results),
        original_reference_duration_s=env.ref.end,results=results,wall_seconds=time.monotonic()-began,
        sample_rate_hz=50,application_mid_episode_resets=0,numerical_warning_worlds=int(h['numeric'].any(0).sum()),sonic_executed=checkpoint['sonic_base'],
        limitation='Screen samples physics state/contact at50Hz; selected final execution must pass full200Hz independent-control replay before acceptance.')
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(args.output/'screen_states.npz',qpos=q,qvel=v,time_s=np.arange(1,len(q)+1)*.02,
                        root_xy_error_m=h['xy'],seat_support_force_N=h['seat'])
    print(json.dumps({key:report[key]for key in ('iteration','num_envs','functional_passes','strict_passes','wall_seconds')}),flush=True)
    env.close()


if __name__=='__main__':main()
