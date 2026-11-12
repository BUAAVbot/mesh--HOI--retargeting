"""Train a new 28-body residual policy in real GPU MuJoCo dynamics."""
import argparse,hashlib,json,time,os,shutil,math
from pathlib import Path


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sanitize_metrics(row,sample_nonfinite_counts=None):
    """Keep failed-episode diagnostics honest and JSON-safe, never replace with0."""
    result=dict(row);bad=[]
    for key,value in row.items():
        if isinstance(value,float)and not math.isfinite(value):result[key]=None;bad.append(key)
    counts=dict(sample_nonfinite_counts or {})
    for key in bad:counts[key]=max(1,int(counts.get(key,0)))
    result['nonfinite_aggregate_fields']=bad;result['nonfinite_metric_sample_counts']=counts
    return result


def initial_checkpoint_provenance(checkpoint,path,current,previous_protocol):
    """Validate policy ABI and declare changed-task weight/optimizer initialization."""
    for name in ('obs_dim','action_dim'):
        if name not in checkpoint or checkpoint[name]!=current[name]:raise ValueError('Initial checkpoint '+name+' differs from the current policy ABI')
    if bool(checkpoint.get('sonic_base',False))!=current['sonic_base']:raise ValueError('Initial checkpoint sonic_base differs from the current policy ABI')
    if checkpoint.get('policy_sensor_contract')!=current['policy_sensor_contract']:raise ValueError('Initial checkpoint sensor sampling contract differs from the current policy ABI')
    previous_abi=None if previous_protocol is None else{k:previous_protocol[k]for k in ('heading_mode','reference_velocity_mode')}
    if previous_abi!=current['sonic_protocol']:raise ValueError('Initial checkpoint SONIC ABI differs; the environment protocol will not be silently changed')
    previous_mode=checkpoint.get('control_bound_mode','legacy_joint')
    if previous_mode not in ('legacy_joint','actuator_ctrl'):raise ValueError('Unsupported initial checkpoint control_bound_mode')
    prior={k:checkpoint.get(k)for k in ('model_path','model_sha256','reference_path','reference_sha256','iteration','obs_dim','action_dim','action_scale','policy_sensor_contract','episode_reset_rule')}
    prior.update(sonic_base=bool(checkpoint.get('sonic_base',False)),sonic_protocol=previous_abi,
                 sonic_protocol_metadata=previous_protocol,control_bound_mode=previous_mode,
                 control_bound_mode_source='explicit_checkpoint_metadata'if'control_bound_mode'in checkpoint else'legacy_missing_metadata_fallback')
    prior['physical_quality_training']=bool(checkpoint.get('physical_quality_training',False));prior['physical_quality_training_contract']=checkpoint.get('physical_quality_training_contract')
    differences={}
    for name in ('model_sha256','reference_sha256','action_scale','control_bound_mode','episode_reset_rule','physical_quality_training','physical_quality_training_contract'):
        if prior[name]!=current[name]:differences[name]={'initial_checkpoint':prior[name],'current_training':current[name]}
    task_changed=any(name in differences for name in ('model_sha256','reference_sha256','action_scale','control_bound_mode','physical_quality_training'))
    return {'schema':'rgb145.initial_checkpoint_provenance.v1','mode':'fine_tune_from_checkpoint_with_changed_task_or_command_contract'if task_changed else'continue_weights_and_optimizer_with_fresh_episode_state',
            'checkpoint_path':str(Path(path).resolve()),'checkpoint_sha256':sha(path),'initial_checkpoint_metadata':prior,
            'current_training_target':current,'declared_differences':differences,'policy_abi_checks_passed':True,
            'actor_and_normalization_state_loaded':True,'optimizer_state_loaded':True,
            'environment_rng_or_episode_state_restored':False,'exact_training_resume_claimed':False,
            'iteration_counter_starts_at':int(checkpoint['iteration']),
            'environment_protocol_changed_from_current_cli':False,'actor_log_std_loaded_from_checkpoint':True,
            'initial_std_cli_overrides_checkpoint_log_std':False}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--num-envs',type=int,default=512);p.add_argument('--iterations',type=int,default=500);p.add_argument('--rollout',type=int,default=32);p.add_argument('--epochs',type=int,default=4);p.add_argument('--minibatches',type=int,default=4);p.add_argument('--save-every',type=int,default=50);p.add_argument('--seed',type=int,default=145317);p.add_argument('--checkpoint',type=Path);p.add_argument('--action-scale',type=float,default=.45)
    p.add_argument('--sonic-base',action='store_true');p.add_argument('--initial-std',type=float)
    p.add_argument('--learning-rate',type=float,default=3e-4);p.add_argument('--entropy-coefficient',type=float,default=.002);p.add_argument('--start-probability',type=float,default=.30);p.add_argument('--strict-training',action='store_true')
    p.add_argument('--reset-source-window',type=float,nargs=2,metavar=('START','END'),help='Optional source-time window for random reference starts; from-start episodes still start at zero and run to the full horizon')
    p.add_argument('--control-bound-mode',choices=('legacy_joint','actuator_ctrl'),default='legacy_joint',help='PD command bounds; actuator_ctrl uses true actuator control limits plus the wrist window, while physical joint/force checks remain unchanged')
    p.add_argument('--physical-quality-training',action='store_true',help='Add true52-hinge limit and real Warp contact-penetration penalties/termination; root bounds and deployment QC stay unchanged')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    import torch,mujoco as mj,warp as wp
    from rgb145_ppo_env import WarpTracking,ActorCritic
    torch.set_num_threads(4)
    method='Learned PPO residual on all28 frozen SONIC body outputs'if a.sonic_base else'Task-specific PPO residual motion imitation around the declared joint reference; not a SONIC checkpoint fine-tune'
    manifest={'method':method,'physics':'MuJoCo-Warp real free-root/contact dynamics','mujoco_version':mj.__version__,'args':{k:str(v)if isinstance(v,Path)else v for k,v in vars(a).items()},'physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'inputs':{'model_sha256':sha(a.model),'reference_sha256':sha(a.reference)},'active_external_wrenches':False,'active_root_state_writes':False,'state_writes':'Only during explicitly declared episode resets','privileged_observations':'Simulator root pose/velocity and world foot positions are used; real deployment would need localization.','source_files':{str(Path(__file__).name):sha(__file__),'rgb145_ppo_env.py':sha(Path(__file__).with_name('rgb145_ppo_env.py'))}}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for name in ('rgb145_train_ppo.py','rgb145_ppo_env.py','rgb145_task_controller.py'):
        source=Path(__file__).with_name(name);shutil.copy2(source,a.output/('source_'+name));manifest['source_files'][name]=sha(source)
    if a.sonic_base:
        for name in ('rgb145_batch_sonic.py','rgb145_sonic_controller.py'):
            source=Path(__file__).with_name(name);shutil.copy2(source,a.output/('source_'+name));manifest['source_files'][name]=sha(source)
    print(json.dumps({'stage':'create_gpu_environment','num_envs':a.num_envs}),flush=True)
    env=WarpTracking(a.model,a.reference,a.num_envs,a.seed,a.action_scale,a.sonic_base,reset_source_window=a.reset_source_window,start_probability=a.start_probability,control_bound_mode=a.control_bound_mode,physical_quality_training=a.physical_quality_training);obs=env.observations();dim=obs.shape[-1]
    env.start_probability=a.start_probability;env.strict_training=a.strict_training
    manifest['policy_sensor_contract']=env.policy_sensor_contract;manifest['software_wrist_command_windows']=env.ref.software_wrist_command_windows
    manifest['episode_reset_rule']=env.reset_rule
    manifest['control_bound_mode']=env.ref.control_bound_mode;manifest['control_bound_contract']=env.ref.control_bound_contract
    manifest['physical_quality_training']=env.physical_quality_training;manifest['physical_quality_training_contract']=env.physical_quality_training_contract
    sonic_protocol=None
    if env.sonic is not None:
        sonic_protocol={'heading_mode':env.sonic.c.heading_mode,'reference_velocity_mode':env.sonic.c.reference_velocity_mode}
        if env.sonic.heading_mode!=sonic_protocol['heading_mode']or env.sonic.reference_velocity_mode!=sonic_protocol['reference_velocity_mode']:raise ValueError('Batch/native SONIC protocol metadata disagree')
    manifest['sonic_protocol']=sonic_protocol;manifest['protocol_metadata_version']=1
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    actor=ActorCritic(dim).to(env.device);optim=torch.optim.Adam(actor.parameters(),lr=a.learning_rate);start=0
    if a.sonic_base:actor.log_std.data.fill_(-2.3)
    if a.initial_std is not None:
        if a.initial_std<=0:raise ValueError('initial-std must be positive')
        actor.log_std.data.fill_(float(torch.tensor(a.initial_std).log()))
    target_metadata={'model_path':str(a.model.resolve()),'model_sha256':sha(a.model),'reference_path':str(a.reference.resolve()),'reference_sha256':sha(a.reference),'obs_dim':dim,'action_dim':28,'sonic_base':bool(a.sonic_base),'sonic_protocol':sonic_protocol,'policy_sensor_contract':env.policy_sensor_contract,'action_scale':a.action_scale,'control_bound_mode':env.ref.control_bound_mode,'episode_reset_rule':env.reset_rule,'physical_quality_training':env.physical_quality_training,'physical_quality_training_contract':env.physical_quality_training_contract,'learning_rate':a.learning_rate,'additional_iterations_requested':a.iterations}
    initialization={'schema':'rgb145.initial_checkpoint_provenance.v1','mode':'fresh_weights','checkpoint_path':None,'checkpoint_sha256':None,'current_training_target':target_metadata,'actor_and_normalization_state_loaded':False,'optimizer_state_loaded':False,'environment_protocol_changed_from_current_cli':False,'exact_training_resume_claimed':False}
    if a.checkpoint:
        from rgb145_task_controller import checkpoint_sonic_protocol
        old=torch.load(a.checkpoint,map_location=env.device,weights_only=False)
        initialization=initial_checkpoint_provenance(old,a.checkpoint,target_metadata,checkpoint_sonic_protocol(old))
        actor.load_state_dict(old['model']);optim.load_state_dict(old['optimizer']);start=int(old['iteration'])
        for group in optim.param_groups:group['lr']=a.learning_rate
    manifest['initialization_provenance']=initialization
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    initial={k:v.detach().clone()for k,v in actor.state_dict().items()};updates=0
    def save(name,iteration):
        path=a.output/name;temp=path.with_suffix('.partial');torch.save({'model':actor.state_dict(),'optimizer':optim.state_dict(),'iteration':iteration,'obs_dim':dim,'action_dim':28,'action_scale':a.action_scale,'reference_path':str(a.reference.resolve()),'reference_sha256':sha(a.reference),'model_path':str(a.model.resolve()),'model_sha256':sha(a.model),'method':manifest['method'],'privileged_observations':True,'sonic_base':a.sonic_base,'protocol_metadata_version':1,'sonic_protocol':sonic_protocol,'heading_mode':sonic_protocol['heading_mode']if sonic_protocol else None,'reference_velocity_mode':sonic_protocol['reference_velocity_mode']if sonic_protocol else None,'policy_sensor_contract':env.policy_sensor_contract,'episode_reset_rule':env.reset_rule,'control_bound_mode':env.ref.control_bound_mode,'control_bound_contract':env.ref.control_bound_contract,'physical_quality_training':env.physical_quality_training,'physical_quality_training_contract':env.physical_quality_training_contract,'initialization_provenance':initialization},temp);temp.replace(path);return path
    mj.mj_saveModel(env.model,str(a.output/'model.mjb'));mj.mj_saveLastXML(str(a.output/'model.xml'),env.model)
    save('initial.pt',start);began=time.monotonic();log=(a.output/'metrics.jsonl').open('a',buffering=1)
    for iteration in range(start,start+a.iterations):
        tick=time.monotonic();actor.update_stats(obs);buf={k:[]for k in ('obs','act','lp','value','reward','done')};errors=[];angles=[];terminals=0;joint_quality=[];contact_quality=[]
        with torch.no_grad():
            for _ in range(a.rollout):
                normalized=actor.normalize(obs);dist=actor.distribution(normalized);act=dist.sample();value=actor.critic(normalized).squeeze(-1)
                nxt,reward,done,info=env.step(act)
                terminal_value=actor.critic(actor.normalize(nxt)).squeeze(-1);reward=reward+.99*terminal_value*(info['timeout']&~info['failed']).float()
                for k,v in [('obs',normalized),('act',act),('lp',dist.log_prob(act).sum(-1)),('value',value),('reward',reward),('done',done)]:buf[k].append(v.clone())
                errors.append(info['root_xy']);angles.append(info['root_angle']);ids=done.nonzero().flatten();terminals+=len(ids)
                if env.physical_quality_training:joint_quality.append(info['physical_joint_violation_rad']);contact_quality.append(info['physical_contact_penetration_m'])
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
        xy_samples=torch.stack(errors);angle_samples=torch.stack(angles);std_samples=actor.log_std.exp()
        row={'iteration':iteration+1,'transitions':(iteration+1)*env.n*a.rollout,'wall_s':time.monotonic()-began,'iteration_s':time.monotonic()-tick,'reward':float(b['reward'].mean()),'root_xy_mean_m':float(xy_samples.mean()),'root_angle_mean_rad':float(angle_samples.mean()),'terminals':terminals,'from_start_successes':env.start_successes,'from_start_episodes':env.starts,'max_from_start_source_time_s':env.max_from_start_source_s,'nonfinite_episodes':env.nan_episodes,'nonfinite_episode_resets':env.nan_episodes,'nonfinite_quality_episode_resets':env.nonfinite_quality_episodes,'invalid_contact_quality_episode_resets':env.invalid_contact_quality_episodes,'physical_quality_terminated_episode_resets':env.quality_terminated_episodes,'joint_limit_terminated_episode_resets':env.joint_limit_terminated_episodes,'penetration_terminated_episode_resets':env.penetration_terminated_episodes,'loss':sum(losses)/len(losses),'std':float(std_samples.mean())}
        sample_counts={key:int((~torch.isfinite(values)).sum())for key,values in [('reward',b['reward']),('root_xy_mean_m',xy_samples),('root_angle_mean_rad',angle_samples),('std',std_samples)]}
        sample_counts['loss']=sum(not math.isfinite(x)for x in losses)
        if env.physical_quality_training:
            for key,values in [('physical_joint_violation_rad_max',torch.stack(joint_quality)),('physical_contact_penetration_m_max',torch.stack(contact_quality))]:row[key]=float(values.max());sample_counts[key]=int((~torch.isfinite(values)).sum())
        row=sanitize_metrics(row,sample_counts)
        log.write(json.dumps(row,allow_nan=False)+'\n')
        if iteration%10==0:print(json.dumps(row),flush=True)
        if (iteration+1)%a.save_every==0:save(f'checkpoint_{iteration+1:06d}.pt',iteration+1)
    final=save('final.pt',start+a.iterations)
    result={'status':'ppo_updates_finished_pending_independent_physics_validation','checkpoint':str(final),'sha256':sha(final),'optimizer_updates':updates,'changed_tensor_keys':[k for k,v in actor.state_dict().items()if not torch.equal(v,initial[k])],'from_start_successes_during_training':env.start_successes,'wall_s':time.monotonic()-began,'initialization_provenance':initialization}
    (a.output/'training_receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
