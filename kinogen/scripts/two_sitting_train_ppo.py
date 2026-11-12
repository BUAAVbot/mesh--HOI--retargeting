"""Train a shared-world two-Vita 56-body residual policy in real GPU MuJoCo dynamics."""
import argparse,hashlib,json,time,os,shutil
from pathlib import Path


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--num-envs',type=int,default=512);p.add_argument('--iterations',type=int,default=500);p.add_argument('--rollout',type=int,default=32);p.add_argument('--epochs',type=int,default=4);p.add_argument('--minibatches',type=int,default=4);p.add_argument('--save-every',type=int,default=50);p.add_argument('--seed',type=int,default=174114);p.add_argument('--checkpoint',type=Path);p.add_argument('--action-scale',type=float,default=.25)
    p.add_argument('--proxy-model',type=Path,required=True);p.add_argument('--sonic-base',action='store_true');p.add_argument('--initial-std',type=float)
    p.add_argument('--learning-rate',type=float,default=3e-4);p.add_argument('--entropy-coefficient',type=float,default=.002);p.add_argument('--start-probability',type=float,default=.30);p.add_argument('--strict-training',action='store_true')
    p.add_argument('--backend',choices=['warp','native'],default='warp');p.add_argument('--threads',type=int,default=16)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    import torch,mujoco as mj,warp as wp
    from two_sitting_ppo_env import WarpTracking,ActorCritic
    if a.backend=='native':
        from two_sitting_native_ppo_env import NativeTracking as WarpTracking
    torch.set_num_threads(4)
    method='Learned PPO residual on all56 dual frozen SONIC body outputs'if a.sonic_base else'Task-specific PPO residual motion imitation around the declared joint reference; not a SONIC checkpoint fine-tune'
    manifest={'method':method,'physics':'MuJoCo-Warp real free-root/contact dynamics','mujoco_version':mj.__version__,'args':{k:str(v)if isinstance(v,Path)else v for k,v in vars(a).items()},'physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'inputs':{'model_sha256':sha(a.model),'reference_sha256':sha(a.reference)},'active_external_wrenches':False,'active_root_state_writes':False,'state_writes':'Only during explicitly declared episode resets','privileged_observations':'Simulator root pose/velocity and world foot positions are used; real deployment would need localization.','proxy_model_sha256':sha(a.proxy_model),'actors':2,'action_dim':56,'source_files':{str(Path(__file__).name):sha(__file__),'two_sitting_ppo_env.py':sha(Path(__file__).with_name('two_sitting_ppo_env.py'))}}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for name in ('two_sitting_train_ppo.py','two_sitting_ppo_env.py'):shutil.copy2(Path(__file__).with_name(name),a.output/('source_'+name))
    if a.backend=='native':shutil.copy2(Path(__file__).with_name('two_sitting_native_ppo_env.py'),a.output/'source_two_sitting_native_ppo_env.py')
    if a.sonic_base:shutil.copy2(Path(__file__).with_name('two_sitting_batch_sonic.py'),a.output/'source_two_sitting_batch_sonic.py')
    print(json.dumps({'stage':'create_gpu_environment','num_envs':a.num_envs}),flush=True)
    env_kwargs={'proxy_path':a.proxy_model}
    if a.backend=='native':env_kwargs['threads']=a.threads
    env=WarpTracking(a.model,a.reference,a.num_envs,a.seed,a.action_scale,a.sonic_base,**env_kwargs);obs=env.observations();dim=obs.shape[-1]
    env.start_probability=a.start_probability;env.strict_training=a.strict_training
    manifest['training_curriculum']=env.training_curriculum_contract
    manifest['policy_sensor_contract']=env.policy_sensor_contract;manifest['software_wrist_command_windows']=env.ref.software_wrist_command_windows
    manifest['physical_body_control_delay_steps']=3
    manifest['contact_reward_contract']=env.contact_contract
    manifest['frozen_sonic_base']=env.sonic.report()if env.sonic is not None else None
    if a.backend=='native':manifest['physics']='Native CPU MuJoCo FP64 persistent worlds with GPU batched policy inference; no Warp dynamics'
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    actor=ActorCritic(dim,action_dim=56).to(env.device);optim=torch.optim.Adam(actor.parameters(),lr=a.learning_rate);start=0
    if a.sonic_base:actor.log_std.data.fill_(-2.3)
    if a.initial_std is not None:
        if a.initial_std<=0:raise ValueError('initial-std must be positive')
        actor.log_std.data.fill_(float(torch.tensor(a.initial_std).log()))
    if a.checkpoint:
        old=torch.load(a.checkpoint,map_location=env.device,weights_only=False);actor.load_state_dict(old['model']);optim.load_state_dict(old['optimizer']);start=old['iteration']
        for group in optim.param_groups:group['lr']=a.learning_rate
    initial={k:v.detach().clone()for k,v in actor.state_dict().items()};updates=0
    def save(name,iteration):
        path=a.output/name;temp=path.with_suffix('.partial');torch.save({'model':actor.state_dict(),'optimizer':optim.state_dict(),'iteration':iteration,'obs_dim':dim,'action_dim':56,'action_scale':a.action_scale,'reference_path':str(a.reference.resolve()),'reference_sha256':sha(a.reference),'model_path':str(a.model.resolve()),'model_sha256':sha(a.model),'method':manifest['method'],'privileged_observations':True,'sonic_base':a.sonic_base},temp);temp.replace(path);return path
    mj.mj_saveModel(env.model,str(a.output/'model.mjb'));mj.mj_saveLastXML(str(a.output/'model.xml'),env.model)
    save('initial.pt',start);began=time.monotonic();log=(a.output/'metrics.jsonl').open('a',buffering=1)
    for iteration in range(start,start+a.iterations):
        tick=time.monotonic();actor.update_stats(obs);buf={k:[]for k in ('obs','act','lp','value','reward','done')};errors=[];angles=[];terminals=0
        with torch.no_grad():
            for _ in range(a.rollout):
                normalized=actor.normalize(obs);dist=actor.distribution(normalized);act=dist.sample();value=actor.critic(normalized).squeeze(-1)
                nxt,reward,done,info=env.step(act)
                terminal_value=actor.critic(actor.normalize(nxt)).squeeze(-1);reward=reward+.99*terminal_value*(info['timeout']&~info['failed']).float()
                for k,v in [('obs',normalized),('act',act),('lp',dist.log_prob(act).sum(-1)),('value',value),('reward',reward),('done',done)]:buf[k].append(v.clone())
                errors.append(info['root_xy']);angles.append(info['root_angle']);ids=done.nonzero().flatten();terminals+=len(ids)
                env.reset(ids);obs=env.observations()if len(ids)else nxt
            last=actor.critic(actor.normalize(obs)).squeeze(-1)
        b={k:torch.stack(v)for k,v in buf.items()};adv=torch.zeros_like(b['reward']);gae=torch.zeros(env.n,device=env.device)
        for t in reversed(range(a.rollout)):
            nxt=last if t==a.rollout-1 else b['value'][t+1];mask=(~b['done'][t]).float();delta=b['reward'][t]+.99*nxt*mask-b['value'][t];gae=delta+.99*.95*mask*gae;adv[t]=gae
        returns=(adv+b['value']).flatten();advantages=adv.flatten();advantages=(advantages-advantages.mean())/(advantages.std()+1e-8);flat={k:v.flatten(0,1)for k,v in b.items()};batch=len(advantages);losses=[]
        for _ in range(a.epochs):
            for ids in torch.randperm(batch,device=env.device).split(max(1,batch//a.minibatches)):
                dist=actor.distribution(flat['obs'][ids]);lp=dist.log_prob(flat['act'][ids]).sum(-1);ratio=(lp-flat['lp'][ids]).exp();policy=torch.maximum(-advantages[ids]*ratio,-advantages[ids]*ratio.clamp(.8,1.2)).mean()
                value=actor.critic(flat['obs'][ids]).squeeze(-1);old=flat['value'][ids];clip=old+(value-old).clamp(-.2,.2);vl=torch.maximum((value-returns[ids]).square(),(clip-returns[ids]).square()).mean()
                loss=policy+.5*vl-a.entropy_coefficient*dist.entropy().sum(-1).mean()
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite PPO loss')
                optim.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(actor.parameters(),1.);optim.step();updates+=1;losses.append(float(loss.detach()))
        row={'iteration':iteration+1,'transitions':(iteration+1)*env.n*a.rollout,'wall_s':time.monotonic()-began,'iteration_s':time.monotonic()-tick,'reward':float(b['reward'].mean()),'root_xy_mean_m':float(torch.stack(errors).mean()),'root_angle_mean_rad':float(torch.stack(angles).mean()),'terminals':terminals,'from_start_successes':env.start_successes,'from_start_episodes':env.starts,'max_from_start_source_time_s':env.max_from_start_source_s,'nonfinite_episodes':env.nan_episodes,'loss':sum(losses)/len(losses),'std':float(actor.log_std.exp().mean())}
        log.write(json.dumps(row,allow_nan=False)+'\n')
        if iteration%10==0:print(json.dumps(row),flush=True)
        if (iteration+1)%a.save_every==0:save(f'checkpoint_{iteration+1:06d}.pt',iteration+1)
    final=save('final.pt',start+a.iterations)
    result={'status':'ppo_updates_finished_pending_independent_physics_validation','checkpoint':str(final),'sha256':sha(final),'optimizer_updates':updates,'changed_tensor_keys':[k for k,v in actor.state_dict().items()if not torch.equal(v,initial[k])],'from_start_successes_during_training':env.start_successes,'wall_s':time.monotonic()-began}
    (a.output/'training_receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
