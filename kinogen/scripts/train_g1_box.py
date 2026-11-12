"""Actual GPU PhysX + PPO for floating G1 five-finger bilateral box lifting."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import shutil

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['initcheck','preflight','train','eval'],default='preflight')
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--num-envs',type=int,default=256)
    p.add_argument('--iterations',type=int,default=1000)
    p.add_argument('--rollout',type=int,default=32)
    p.add_argument('--seed',type=int,default=20261021)
    p.add_argument('--mass',type=float,default=1.0)
    p.add_argument('--noise',type=float,default=.01)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--asset-config',type=Path,default=Path('data/robots/g1_box_training/training_asset.json'))
    p.add_argument('--table-x-size',type=float,default=.36)
    p.add_argument('--stage-box-before-reset',action='store_true')
    p.add_argument('--save-every',type=int,default=100)
    p.add_argument('--epochs',type=int,default=4)
    p.add_argument('--minibatches',type=int,default=4)
    p.add_argument('--learning-rate',type=float,default=3e-4)
    args=p.parse_args()
    out=(ROOT/args.run_dir).resolve()
    if not out.is_relative_to(ROOT/'outputs'):
        raise ValueError('Run directory must be inside KinoGen/outputs')
    out.mkdir(parents=True,exist_ok=False)
    if os.environ.get('CUDA_VISIBLE_DEVICES')not in ('6','7'):raise RuntimeError('Reserve physical GPU6 or7 explicitly')
    physical_gpu=int(os.environ['CUDA_VISIBLE_DEVICES'])
    if hasattr(os,'sched_getaffinity'):
        allowed=os.sched_getaffinity(0);selected=set(range(32,40)if physical_gpu==6 else range(40,48))&allowed
        if selected:os.sched_setaffinity(0,selected)
    os.environ.setdefault('OMNI_KIT_ACCEPT_EULA','Yes')
    os.environ.setdefault('OMNI_KIT_ALLOW_ROOT','1')
    asset_config=(ROOT/args.asset_config).resolve()
    if not asset_config.is_relative_to(ROOT):raise ValueError('Asset config outside project')
    asset_spec=json.loads(asset_config.read_text())
    base_asset_path=None
    if 'source_training_asset_path'in asset_spec:
        base_asset_path=(ROOT/asset_spec['source_training_asset_path']).resolve()
        if not base_asset_path.is_relative_to(ROOT)or sha(base_asset_path)!=asset_spec['source_training_asset_sha256']:
            raise RuntimeError('Base training asset SHA differs from reset patch')
        asset_spec={**json.loads(base_asset_path.read_text()),**asset_spec}
    manifest={'created_at_utc':datetime.now(timezone.utc).isoformat(),'args':{k:str(v)if isinstance(v,Path)else v for k,v in vars(args).items()},
        'physical_gpu':physical_gpu,'local_training':False,'physics':'IsaacSim4.5 GPU PhysX',
        'method':'PPO residual joint-target controller around a declared IK close/lift reference; no policy-pretrained manipulation weights',
        'sources':{str(x.relative_to(ROOT)):sha(x)for x in [Path(__file__),ROOT/'scripts/g1_box_physics.py',asset_config,ROOT/asset_spec['urdf'],ROOT/asset_spec['coupling_lut']]+([base_asset_path]if base_asset_path else[])},
        'success_protocol':'Actual box oriented bottom >= table+10cm, bilateral box-filtered contact >0.5N, table contact <0.2N through final2s, free standing robot, no active pose/joint-state writes or wrenches.'}
    write_json(out/'run_manifest.json',manifest)
    for name,digest in manifest['sources'].items():
        src=ROOT/name
        if sha(src)!=digest:raise RuntimeError('Source changed during run setup')
        target=out/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,target)
    from isaaclab.app import AppLauncher
    class PinnedAppLauncher(AppLauncher):
        def _resolve_device_settings(self, launcher_args):
            super()._resolve_device_settings(launcher_args)
            launcher_args.update(active_gpu=physical_gpu,physics_gpu=0,multi_gpu=False)
    launcher=PinnedAppLauncher(headless=True,device='cuda:0',
        kit_args='--allow-root --/app/extensions/registryEnabled=false --/app/extensions/preferLocalVersions=true --/log/file='+str(out/'kit.log'))
    import torch
    import numpy as np
    from torch import nn
    from torch.distributions import Normal
    from g1_box_physics import G1BoxPhysics
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    env=G1BoxPhysics(args.num_envs,seed=args.seed,mass=args.mass,noise=args.noise,physical_gpu=physical_gpu,asset_config=asset_config,table_x_size=args.table_x_size,
        stage_box_before_reset=args.stage_box_before_reset)
    obs=env.observations()
    if not torch.isfinite(obs).all():raise RuntimeError('Initial state nonfinite')
    obs_dim=obs.shape[1]

    class ActorCritic(nn.Module):
        def __init__(self):
            super().__init__()
            def net(output):
                return nn.Sequential(nn.Linear(obs_dim,256),nn.ELU(),nn.Linear(256,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,output))
            self.actor=net(41);self.critic=net(1);self.log_std=nn.Parameter(torch.full((41,),-.8))
            for layer in self.modules():
                if isinstance(layer,nn.Linear):
                    nn.init.orthogonal_(layer.weight,2**.5);nn.init.zeros_(layer.bias)
            nn.init.orthogonal_(self.actor[-1].weight,.01);nn.init.orthogonal_(self.critic[-1].weight,1.)
            self.register_buffer('obs_mean',torch.zeros(obs_dim))
            self.register_buffer('obs_var',torch.ones(obs_dim))
            self.register_buffer('obs_count',torch.tensor(1e-4))
        def normalize(self,x):return ((x-self.obs_mean)/(self.obs_var+1e-6).sqrt()).clamp(-10.,10.)
        @torch.no_grad()
        def update_stats(self,x):
            var,mean=torch.var_mean(x,dim=0,unbiased=False);count=x.shape[0]
            delta=mean-self.obs_mean;total=self.obs_count+count
            self.obs_var.copy_((self.obs_var*self.obs_count+var*count+delta.square()*self.obs_count*count/total)/total)
            self.obs_mean.add_(delta*count/total);self.obs_count.copy_(total)
        def distribution(self,x):return Normal(self.actor(x),self.log_std.clamp(-3.,.5).exp())

    model=ActorCritic().to(env.device)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.learning_rate)
    start_iteration=0
    if args.checkpoint:
        source=(ROOT/args.checkpoint).resolve()
        ckpt=torch.load(source,map_location=env.device,weights_only=False)
        if ckpt['obs_dim']!=obs_dim or ckpt['action_dim']!=41:raise RuntimeError('Checkpoint interface changed')
        model.load_state_dict(ckpt['model'],strict=True)
        write_json(out/'checkpoint_load_receipt.json',{'checkpoint_path':str(source),'checkpoint_sha256':sha(source),
            'strict_model_load':True,'training_asset_sha256':ckpt['training_asset_sha256'],
            'current_asset_config_sha256':sha(asset_config),
            'asset_or_reset_config_changed':ckpt['training_asset_sha256']!=sha(asset_config),
            'original_model_interface_preserved':True})
        if args.mode=='train':
            optimizer.load_state_dict(ckpt['optimizer']);start_iteration=ckpt['iteration']
    initial={k:v.detach().clone()for k,v in model.state_dict().items()}
    def actor_sha():
        h=hashlib.sha256()
        for name,tensor in sorted(model.actor.state_dict().items()):
            h.update(name.encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    initial_actor_sha=actor_sha()
    original_training=ckpt.get('training',{})if args.checkpoint else {}
    optimizer_updates=int(original_training.get('optimizer_steps',0))if args.mode=='train'else 0
    inference_calls=0

    def save(name,iteration):
        path=out/name;temp=path.with_suffix('.tmp')
        checkpoint={'model':model.state_dict(),'optimizer':optimizer.state_dict(),
            'iteration':iteration,'env_steps':iteration*args.rollout*env.n,'obs_dim':obs_dim,'action_dim':41,
            'body_joint_names':env.body_names,'hand_joint_names':env.hand_names,
            'training_asset_sha256':sha(asset_config),
            'run_manifest_sha256':sha(out/'run_manifest.json'),'torch_version':torch.__version__,
            'training':{'optimizer_steps':optimizer_updates,'environment_steps':iteration*args.rollout*env.n,
                'initial_actor_sha256':original_training.get('initial_actor_sha256',initial_actor_sha),
                'final_actor_sha256':actor_sha()}}
        torch.save(checkpoint,temp);temp.replace(path)
        return path

    baseline=save('initial.pt',start_iteration)
    print(json.dumps({'ready':True,'mode':args.mode,'num_envs':env.n,'obs_dim':obs_dim,'action_dim':41,
        'initial_checkpoint':str(baseline),'physics':env.live_audit()}),flush=True)
    write_json(out/'live_physics_initial.json',env.live_audit())
    if args.mode in ('initcheck','preflight','eval'):
        if args.mode!='initcheck':
            from export_g1_box_render_geometry import export
            export(env,env.robot,env.box,out)
        source=(ROOT/args.checkpoint).resolve()if args.checkpoint else baseline
        seeds=[args.seed+i for i in range(env.n)]
        env.reset(torch.arange(env.n,device=env.device),seeds=seeds)
        # True open-hand physical settling, with the close/lift phase still zero.
        init_records=[]
        for initialization_step in range(51):
            s=env.state()
            init_records.append({**{k:v.detach().cpu().numpy()for k,v in s.items()if torch.is_tensor(v)},
                'robot_box_forces':env.scene['all_robot_box_contact'].data.force_matrix_w.detach().cpu().numpy(),
                'table_pos':(env.scene['table'].data.root_pos_w-env.scene.env_origins).detach().cpu().numpy()})
            if initialization_step==50:break
            env.scene.write_data_to_sim();env.sim.step(render=False);env.scene.update(env.physics_dt)
        init_arrays={k:np.stack([r[k]for r in init_records])for k in init_records[0]}
        np.savez_compressed(out/'initialization_trace.npz',**init_arrays,
            timestamps=np.arange(51)*env.physics_dt,seed_ids=np.asarray(seeds),
            contact_body_names=np.asarray(env.contact_body_names),joint_names=np.asarray(env.names))
        peak=np.linalg.norm(init_arrays['robot_box_forces'],axis=-1).max(axis=(0,2))
        init_report={'seed_ids':seeds,'environment_origins':env.scene.env_origins.cpu().tolist(),
            'table_physx_paths':env.scene['table'].root_physx_view.prim_paths,
            'box_physx_paths':env.box.root_physx_view.prim_paths,
            'robot_physx_paths':env.robot.root_physx_view.prim_paths,
            'initial_table_pos_relative':init_arrays['table_pos'][0].tolist(),
            'last_table_pos_relative':init_arrays['table_pos'][-1].tolist(),
            'initial_box_pos_relative':init_arrays['box_pos'][0].tolist(),
            'last_box_pos_relative':init_arrays['box_pos'][-1].tolist(),
            'last_box_velocity':init_arrays['box_vel'][-1].tolist(),
            'last_box_bottom_z':init_arrays['box_bottom_z'][-1].tolist(),
            'peak_robot_contacts_by_episode':[{'seed':seeds[i],'max_force_N':float(peak[i].max()),
                'contact_body_names':[env.contact_body_names[j]for j in np.flatnonzero(peak[i]>.01)]}for i in range(env.n)]}
        write_json(out/'initialization_report.json',init_report)
        if args.mode=='initcheck':
            print(json.dumps({'initialization_diagnostic_completed':True,'report':str(out/'initialization_report.json')}),flush=True)
            os._exit(0)
        obs=env.observations()
        records=[];finished=torch.zeros(env.n,dtype=torch.bool,device=env.device)
        def sample(done,initial=False):
            s=env.state();active=(~finished).nonzero().flatten()
            if not len(active):return
            row={k:v[active].detach().cpu().numpy()for k,v in s.items()if torch.is_tensor(v)}
            ids=active.cpu().numpy()
            row.update(episode_id=ids,seed_id=np.asarray(seeds,dtype=np.int64)[ids],
                timestamps=np.full(len(ids),0. if initial else float(env.steps[0])*env.dt),
                done=done[active].cpu().numpy(),reset_events=np.full(len(ids),initial))
            records.append(row)
        sample(torch.zeros_like(finished),initial=True)
        with torch.inference_mode():
            for step in range(env.max_steps):
                if args.mode=='preflight':action=torch.zeros((env.n,41),device=env.device)
                else:
                    action=model.actor(model.normalize(obs));inference_calls+=1
                obs,reward,done,info=env.step(action)
                sample(done)
                finished |= done
                if step%50==0:
                    print(json.dumps({'eval_step':step,'finished':int(finished.sum()),'max_lift_m':float(info['lift_m'].max()),
                        'double_contact':int(info['double_contact'].sum()),'upright_min':float(info['upright'].min())}),flush=True)
                if finished.all():break
        arrays={k:np.concatenate([r[k]for r in records],axis=0)for k in records[0]}
        arrays.update(joint_names=np.asarray(env.names),body_names=np.asarray(env.robot.body_names),
            joint_units=np.asarray(env.live_audit()['joint_units']),
            source_checkpoint_sha256=np.asarray(sha(source)),quaternion_order=np.asarray('wxyz'))
        trace=out/'trace.npz';np.savez_compressed(trace,**arrays)
        live=env.live_audit();active=sum(x['active_calls']for x in env.guard_counts.values())
        protocol={**live,'training':original_training,'sources':{'trace_sha256':sha(trace),'checkpoint_sha256':sha(source),'checkpoint_path':str(source)},
            'trace_layout':{'quaternion_order':'wxyz','layout':'flattened_per_sample','body_quaternion_order':'wxyz'},
            'stage':{'gravity_m_s2':[0.,0.,-9.81],
                'box':{'mass_kg':args.mass,'size_xyz_m':env.box_size.cpu().tolist(),'dynamic':True,'kinematic':False,'gravity_enabled':True},
                'robot':{'free_base':not env.robot.is_fixed_base,'root_kinematic':False},
                'table':{'top_z_m':env.table_z},'object_attachment_joint_count':len(live['box_connected_joints']),
                'artificial_grasp_assist':False},
            'contacts':{'hand_box_filtered':True,'table_box_filtered':True},
            'guards':{'active_state_write_calls':active,'active_external_wrench_calls':active,
                'instrumented_apis':list(env.guard_counts),'started_before_active':True,'per_api':env.guard_counts},
            'evaluation':{'seeds':seeds,'sample_phase':'post_physics_pre_reset','mode':args.mode,
                'actor_sha256':actor_sha(),'policy_inference_calls':inference_calls,
                'initial_physical_settle_s':.25,'box_xy_randomization_m':args.noise,'first_terminal_only':True}}
        write_json(out/'live_physics.json',protocol)
        print(json.dumps({'evaluation_complete':True,'trace':str(trace),'trace_sha256':sha(trace),
            'all_episodes_terminal':bool(finished.all()),'actual_samples':len(arrays['timestamps']),
            'max_hand_contacts':[float(arrays['hand_contact_force_left_N'].max()),float(arrays['hand_contact_force_right_N'].max())],
            'min_root_z':float(arrays['root_pos'][:,2].min()),'max_box_z':float(arrays['box_pos'][:,2].max())}),flush=True)
        os._exit(0)

    model.train()
    started=time.monotonic();episode_count=0;successful_terminal=0
    log=(out/'metrics.jsonl').open('a',buffering=1)
    for iteration in range(start_iteration,start_iteration+args.iterations):
        iteration_start=time.monotonic()
        model.update_stats(obs)
        buffers={k:[]for k in ['obs','actions','logprob','values','rewards','dones']}
        lifts=[];contacts=[];holds=[]
        with torch.no_grad():
            for _ in range(args.rollout):
                normalized=model.normalize(obs)
                distribution=model.distribution(normalized);actions=distribution.sample()
                value=model.critic(normalized).squeeze(-1)
                nxt,reward,done,info=env.step(actions)
                terminal_value=model.critic(model.normalize(torch.nan_to_num(nxt))).squeeze(-1)
                terminal_value=torch.where(info['failed'],torch.zeros_like(terminal_value),terminal_value)
                adjusted=reward+.99*terminal_value*(info['timeout']&~info['failed']).float()
                for k,v in [('obs',normalized),('actions',actions),('logprob',distribution.log_prob(actions).sum(-1)),
                    ('values',value),('rewards',adjusted),('dones',done)]:buffers[k].append(v.clone())
                terminal=done.nonzero().flatten()
                episode_count+=len(terminal);successful_terminal+=int((done&info['success']).sum())
                lifts.append(info['lift_m']);contacts.append(info['double_contact']);holds.append(info['held'])
                env.reset(terminal)
                obs=env.observations()if len(terminal)else nxt
            last_value=model.critic(model.normalize(obs)).squeeze(-1)
        b={k:torch.stack(v)for k,v in buffers.items()}
        advantage=torch.zeros_like(b['rewards']);last_gae=torch.zeros(env.n,device=env.device)
        for t in reversed(range(args.rollout)):
            following=last_value if t==args.rollout-1 else b['values'][t+1]
            mask=(~b['dones'][t]).float()
            delta=b['rewards'][t]+.99*following*mask-b['values'][t]
            last_gae=delta+.99*.95*mask*last_gae;advantage[t]=last_gae
        returns=advantage+b['values']
        adv=advantage.flatten();adv=(adv-adv.mean())/(adv.std()+1e-8)
        flat={k:v.flatten(0,1)for k,v in b.items()}
        ret=returns.flatten();batch=len(adv);batch_size=max(1,batch//args.minibatches)
        losses=[];kls=[]
        for _ in range(args.epochs):
            permutation=torch.randperm(batch,device=env.device)
            for indices in permutation.split(batch_size):
                dist=model.distribution(flat['obs'][indices]);lp=dist.log_prob(flat['actions'][indices]).sum(-1)
                ratio=(lp-flat['logprob'][indices]).exp()
                surrogate=torch.maximum(-adv[indices]*ratio,-adv[indices]*ratio.clamp(.8,1.2)).mean()
                value=model.critic(flat['obs'][indices]).squeeze(-1)
                old=flat['values'][indices];clipped=old+(value-old).clamp(-.2,.2)
                value_loss=torch.maximum((value-ret[indices]).square(),(clipped-ret[indices]).square()).mean()
                entropy=dist.entropy().sum(-1).mean()
                loss=surrogate+.5*value_loss-.005*entropy
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite PPO objective')
                optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step()
                optimizer_updates+=1
                losses.append(float(loss.detach()));kls.append(float((flat['logprob'][indices]-lp).mean().detach()))
        row={'iteration':iteration+1,'env_steps':(iteration+1)*args.rollout*env.n,
            'elapsed_s':time.monotonic()-started,'iteration_s':time.monotonic()-iteration_start,
            'reward_mean':float(b['rewards'].mean()),'lift_mean_m':float(torch.stack(lifts).mean()),
            'bilateral_contact_fraction':float(torch.stack(contacts).float().mean()),
            'hold_fraction':float(torch.stack(holds).float().mean()),'episodes':episode_count,
            'training_terminal_hold_successes':successful_terminal,'loss_mean':sum(losses)/len(losses),
            'approx_kl_mean':sum(kls)/len(kls),'policy_std_mean':float(model.log_std.exp().mean().detach())}
        log.write(json.dumps(row,allow_nan=False)+'\n')
        if iteration%10==0:print(json.dumps(row),flush=True)
        if (iteration+1)%args.save_every==0:save(f'checkpoint_{iteration+1:06d}.pt',iteration+1)
    final=save('final.pt',start_iteration+args.iterations)
    changed=[k for k,v in model.state_dict().items()if not torch.equal(v,initial[k])]
    all_finite=all(bool(torch.isfinite(v).all())for v in model.state_dict().values())
    write_json(out/'training_receipt.json',{'status':'ppo_updates_completed_not_success_evaluation',
        'checkpoint':str(final),'checkpoint_sha256':sha(final),'initial_checkpoint_sha256':sha(baseline),
        'iterations':args.iterations,'optimizer_updates':optimizer_updates,
        'initial_actor_sha256':initial_actor_sha,'final_actor_sha256':actor_sha(),
        'environment_transitions':args.iterations*args.rollout*env.n,
        'changed_model_tensor_keys':changed,'all_tensors_finite':all_finite,
        'wall_time_s':time.monotonic()-started,'guard_counts':env.guard_counts,
        'success_requires_independent_heldout_evaluation':True})
    print('G1_BOX_TRAINING_COMPLETED '+str(final),flush=True)
    os._exit(0)


if __name__=='__main__':main()
