"""Train a direct privileged dual-Vita specialist, inspired by SUGAR's Refiner."""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, sys, time
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def calibrated_model(root):
    import numpy as np,mujoco as mj
    output=root/'calibration';output.mkdir(exist_ok=True)
    path=output/'two_vita_expert.mjb'
    if path.exists():return path
    source=Path('/data/KinoGen/outputs/two_sitting_pipeline/assets_deploy_v3/two_vita_solver20.xml')
    baseline=root/'sonic_baseline/phc/model.mjb'
    frozen=mj.MjModel.from_binary_path(str(baseline))
    keys=('actuator_gainprm','actuator_biasprm','actuator_dynprm','actuator_gaintype',
          'actuator_biastype','actuator_dyntype','actuator_ctrllimited','actuator_ctrlrange',
          'actuator_forcelimited','actuator_forcerange','actuator_gear','dof_armature',
          'dof_frictionloss','dof_damping','jnt_range','jnt_limited')
    shutil.copy2(baseline,path)
    reloaded=mj.MjModel.from_binary_path(str(path))
    checks={key:float(np.max(np.abs(getattr(reloaded,key).astype(float)-getattr(frozen,key).astype(float))))
            for key in (*keys,'body_mass','body_inertia','body_pos','body_quat','geom_pos','geom_size','geom_friction')}
    if max(checks.values())>1e-4:raise RuntimeError('Calibrated plant changed materially: '+str(checks))
    (output/'report.json').write_text(json.dumps({'source_model':str(source),'source_sha256':sha(source),
        'baseline_frozen_model':str(baseline),'baseline_sha256':sha(baseline),'expert_model':str(path),
        'expert_sha256':sha(path),'max_parameter_reload_absolute_error':checks,
        'no_root_actuators':True,'same_sonic_calibrated_motor_rotor_joint_contract':True,
        'expert_executes_sonic':False},indent=2)+'\n')
    return path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-override',type=Path)
    parser.add_argument('--root',type=Path,default=Path('/data/KinoGen/outputs/sonic_expert_20261028'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--iterations',type=int,default=600)
    parser.add_argument('--num-envs',type=int,default=64);parser.add_argument('--threads',type=int,default=8)
    parser.add_argument('--rollout',type=int,default=32);parser.add_argument('--epochs',type=int,default=4)
    parser.add_argument('--seed',type=int,default=174114);parser.add_argument('--action-scale',type=float,default=.7)
    parser.add_argument('--learning-rate',type=float,default=3e-4);parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--pool',type=Path);parser.add_argument('--time-scale',type=float,default=1.)
    parser.add_argument('--save-every',type=int,default=100)
    parser.add_argument('--sonic-base',action='store_true')
    parser.add_argument('--warm-start',type=Path)
    parser.add_argument('--initial-std',type=float)
    parser.add_argument('--entropy',type=float,default=.005)
    args=parser.parse_args()
    import torch,numpy as np,mujoco as mj
    from two_sitting_ppo_env import ActorCritic
    from residual_seeded_seat_env import ResidualSeededSeatEnv as RefinerEnv
    torch.set_num_threads(2);torch.manual_seed(args.seed)
    selection=json.loads((args.root/'sonic_comparison.json').read_text())
    model=calibrated_model(args.root);reference=Path(selection['reference'])
    if args.time_scale!=1.:
        z=dict(np.load(reference,allow_pickle=False));z['time_s']=z['time_s']*args.time_scale
        target=args.root/'references'/f"{selection['selected_method']}_time_scale{args.time_scale:g}.npz"
        if not target.exists():np.savez_compressed(target,**z)
        reference=target
    if args.reference_override is not None:
        old=dict(np.load(reference,allow_pickle=False));new=dict(np.load(args.reference_override,allow_pickle=False))
        for key in ('qpos','time_s','source_time_s','dof_names'):
            if not np.array_equal(old[key],new[key]):raise ValueError('Contact-only variant changed '+key)
        reference=args.reference_override
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    for name in ('train_residual_refined_contact_20261028.py','sugar_expert_env.py','residual_heading_env.py','residual_soft_seat_env.py','residual_seeded_seat_env.py'):
        source=Path(__file__).with_name(name)
        if source.exists():shutil.copy2(source,args.output/('source_'+name))
    env=RefinerEnv(model,reference,n=args.num_envs,seed=args.seed,action_scale=args.action_scale,
                   sonic_base=args.sonic_base,threads=args.threads,
                   proxy_path=Path('/data/KinoGen/outputs/two_sitting_pipeline/assets_deploy_v3/vita_single_proxy.xml'))
    env.start_probability=.4
    if args.pool:
        z=np.load(args.pool,allow_pickle=False)
        for i in range(len(z['phase'])):
            entry={key:z[key][i].copy()if hasattr(z[key][i],'copy')else z[key][i]for key in z.files}
            entry['phase']=float(entry['phase']);env.pool.setdefault(int(entry['phase']/.1),[]).append(entry)
    obs=env.observations();dim=obs.shape[-1]
    actor=ActorCritic(dim,action_dim=56).to(env.device);actor.log_std.data.fill_(-.7)
    optimizer=torch.optim.Adam(actor.parameters(),lr=args.learning_rate)
    start=0
    if args.warm_start:
        saved=torch.load(args.warm_start,map_location=env.device,weights_only=False)
        if saved['obs_dim']!=dim or bool(saved['sonic_base'])!=args.sonic_base:
            raise ValueError('Warm-start policy architecture differs')
        actor.load_state_dict(saved['model'])
    if args.checkpoint:
        saved=torch.load(args.checkpoint,map_location=env.device,weights_only=False)
        if saved['reference_sha256']!=sha(reference):raise ValueError('Reference changed on resume')
        actor.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);start=saved['iteration']
        for group in optimizer.param_groups:group['lr']=args.learning_rate
    if args.initial_std is not None:actor.log_std.data.fill_(float(np.log(args.initial_std)))
    if args.checkpoint is None:
        actor.actor[-1].weight.data.zero_();actor.actor[-1].bias.data.zero_()
    original={key:value.detach().clone()for key,value in actor.state_dict().items()}
    method=('Independent task PPO residual expert on strictly frozen SONIC; SUGAR-inspired state pool and contact refinement; SONIC weights unchanged' if args.sonic_base else
            'Direct privileged dual-Vita PPO expert around selected kinematic reference; SUGAR-inspired state pool and contact refinement; no SONIC inference')
    manifest={'method':method,'selected_method':selection['selected_method'],
        'reference':str(reference),'reference_sha256':sha(reference),'model':str(model),'model_sha256':sha(model),
        'physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'args':{k:str(v)if isinstance(v,Path)else v for k,v in vars(args).items()},
        'mujoco_version':mj.__version__,'torch_version':torch.__version__,'policy_sensor_contract':env.policy_sensor_contract,
        'sugar_repository':'https://github.com/tianshuwu/SUGAR','sugar_commit':'01fe1234b57188412ec38775150f008f133e0ad4',
        'adaptation':'Own dual-Vita MuJoCo3.5 implementation, not official G1/IsaacLab5.1 trainer. Uses privileged future reference, actual contacts and validated-state reset pool. Exact SUGAR reward/hyperparameters and autonomous generator not reproduced.',
        'live_state_writes':'episode resets only; no root actuation, weld, mocap or external assisting wrench',
        'warmup_initialization':'40% first state plus conventional reference-state initialization; RSI is not labeled a physically validated state',
        'reward':'Existing root/joint/foot mimic terms plus root-relative body, seat-relative geometric target, actual upward contact load and action/velocity regularization; weights in source.',
        'no_domain_randomization_claim':True,'source_hand_ground_truth_available':False}
    manifest['warm_start']=None if args.warm_start is None else {'path':str(args.warm_start),'sha256':sha(args.warm_start),'old_reference_sha256':saved['reference_sha256'],'new_training_iteration_starts_at':0}
    manifest['sonic_weights_updated']=False
    manifest['physical_refined_contact_schedule']={'reference':str(reference),'sha256':sha(reference),'diagnostic':'/data/KinoGen/outputs/sonic_expert_20261028/physical_contact_schedule_diagnostic.json','qpos_time_dof_names_unchanged':True,'coarse_labels_preserved_in_reference':True,'source':'Actual independently replayed oldcandidate156 seat contact/load aligned by source clock, not human ground truth or newexpert success'}
    manifest['zero_actor_output_head_initialization']={'applied':args.checkpoint is None,'purpose':'Initial deterministic residual exactly zero, so paired physical mean behavior equals frozen SONIC; old task actor output is not inherited. Hidden/critic/normalization initialization is retained as disclosed.'}
    manifest['physical_prior_pool']=None if args.pool is None else {'path':str(args.pool),'sha256':sha(args.pool),'source':'Audited old actual candidate156 states, explicitly not new-PHC success; task-only source path tolerances declared in export receipt.'}
    manifest['physical_initial_state']={'path':str(env.fixed_initial_path),'sha256':sha(env.fixed_initial_path),'source':'Previously independently replayed candidate156; this changes episode startup explicitly, not a new-task success claim; only episode reset writes.'}
    manifest['soft_seat_source_root_adaptation']='For late actors with inferred seated label, remove original0.30*exp(-XY²/.15²)-0.20*XY² and new0.10*XY source-root terms; retain body-relative pose and heading; add0.30 actual load-.08 seat XY distance. Final validation thresholds unchanged.'
    manifest['additional_reward_ablation']='Dense full root orientation penalty0.08*mean(angle**2), root XY penalty0.10*mean(error); after55pct phase add0.40 actual seat load +0.30 seat geometry +0.15 contact-conditioned root stability. Explicit own engineering ablation.'
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    def save(name,iteration):
        path=args.output/name
        torch.save(dict(model=actor.state_dict(),optimizer=optimizer.state_dict(),iteration=iteration,
            obs_dim=dim,action_dim=56,action_scale=args.action_scale,reference_path=str(reference),
            reference_sha256=sha(reference),model_path=str(model),model_sha256=sha(model),method=method,
            privileged_observations=True,sonic_base=args.sonic_base,fixed_initial_state_path=str(env.fixed_initial_path),fixed_initial_state_sha256=sha(env.fixed_initial_path)),path)
        return path
    save('initial.pt',start);log=(args.output/'metrics.jsonl').open('a',buffering=1)
    began=time.monotonic();updates=0
    for iteration in range(start,start+args.iterations):
        tick=time.monotonic();env.iteration=iteration;actor.update_stats(obs)
        buffer={key:[]for key in ('obs','act','lp','value','reward','done')};xy=[];loads=[]
        with torch.no_grad():
            for step in range(args.rollout):
                normalized=actor.normalize(obs);dist=actor.distribution(normalized);action=dist.sample()
                value=actor.critic(normalized).squeeze(-1);nxt,reward,done,info=env.step(action)
                terminal=actor.critic(actor.normalize(nxt)).squeeze(-1)
                reward=torch.nan_to_num(reward,nan=-2.,posinf=-2.,neginf=-2.)+.99*terminal*(info['timeout']&~info['failed']).float()
                for key,item in [('obs',normalized),('act',action),('lp',dist.log_prob(action).sum(-1)),
                                 ('value',value),('reward',reward),('done',done)]:buffer[key].append(item.clone())
                xy.append(info['root_xy']);loads.append(info['seat_load_fraction'])
                ids=done.nonzero().flatten();env.reset(ids);obs=env.observations()if len(ids)else nxt
            last=actor.critic(actor.normalize(obs)).squeeze(-1)
        b={key:torch.stack(items)for key,items in buffer.items()}
        adv=torch.zeros_like(b['reward']);gae=torch.zeros(env.n,device=env.device)
        for t in reversed(range(args.rollout)):
            nxt=last if t==args.rollout-1 else b['value'][t+1];mask=(~b['done'][t]).float()
            delta=b['reward'][t]+.99*nxt*mask-b['value'][t];gae=delta+.99*.95*mask*gae;adv[t]=gae
        returns=(adv+b['value']).flatten();advantages=adv.flatten();advantages=(advantages-advantages.mean())/(advantages.std()+1e-8)
        flat={key:item.flatten(0,1)for key,item in b.items()};batch=len(advantages);losses=[]
        for epoch in range(args.epochs):
            for ids in torch.randperm(batch,device=env.device).split(max(1,batch//4)):
                dist=actor.distribution(flat['obs'][ids]);ratio=(dist.log_prob(flat['act'][ids]).sum(-1)-flat['lp'][ids]).exp()
                policy=torch.maximum(-advantages[ids]*ratio,-advantages[ids]*ratio.clamp(.8,1.2)).mean()
                value=actor.critic(flat['obs'][ids]).squeeze(-1);old=flat['value'][ids]
                clipped=old+(value-old).clamp(-.2,.2)
                vl=torch.maximum((value-returns[ids]).square(),(clipped-returns[ids]).square()).mean()
                loss=policy+.5*vl-args.entropy*dist.entropy().sum(-1).mean()
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite expert PPO loss')
                optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(actor.parameters(),1.)
                optimizer.step();updates+=1;losses.append(float(loss.detach()))
        row={'iteration':iteration+1,'transitions_this_run':(iteration-start+1)*env.n*args.rollout,
            'reward':float(b['reward'].mean()),'root_xy_mean_m':float(torch.stack(xy).mean()),
            'seat_load_mean':float(torch.stack(loads).mean()),'training_start_successes':env.start_successes,
            'training_start_episodes':env.starts,'pool_states':sum(map(len,env.pool.values())),
            'pool_bins':len(env.pool),'pool_resets':env.pool_resets,'nonfinite_episodes':env.nan_episodes,
            'wall_s':time.monotonic()-began,'iteration_s':time.monotonic()-tick,
            'loss':sum(losses)/len(losses),'std':float(actor.log_std.exp().mean())}
        log.write(json.dumps(row)+'\n')
        if (iteration+1)%10==0:print(json.dumps(row),flush=True)
        if (iteration+1)%args.save_every==0:
            save(f'checkpoint_{iteration+1:06d}.pt',iteration+1)
            pool=env.save_pool(args.output/f'pool_{iteration+1:06d}.npz')
            (args.output/'state_pool_report.json').write_text(json.dumps(pool,indent=2)+'\n')
    final=save('final.pt',start+args.iterations);pool=env.save_pool(args.output/'state_pool.npz')
    result={'status':'expert_ppo_training_finished_pending_independent_physics_validation','checkpoint':str(final),
        'checkpoint_sha256':sha(final),'optimizer_updates':updates,'pool':pool,'backend':env.backend_report(),
        'changed_tensor_keys':[key for key,value in actor.state_dict().items()if not torch.equal(value,original[key])],
        'wall_seconds':time.monotonic()-began,'selected_reference':str(reference)}
    (args.output/'training_receipt.json').write_text(json.dumps(result,indent=2)+'\n');env.close()
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
