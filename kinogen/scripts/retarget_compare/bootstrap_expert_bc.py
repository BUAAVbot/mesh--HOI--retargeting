"""Initialize an independent expert from stable actual SONIC executions, then release to RL."""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/data/KinoGen/outputs/sonic_expert_20261028'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--num-envs',type=int,default=16)
    p.add_argument('--epochs',type=int,default=300);p.add_argument('--action-scale',type=float,default=1.5)
    a=p.parse_args()
    import numpy as np,torch,mujoco as mj
    from two_sitting_native_ppo_env import NativeTracking
    from two_sitting_ppo_env import ActorCritic
    from train_sugar_expert_20261028 import calibrated_model
    torch.set_num_threads(2);torch.manual_seed(174120)
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    model=calibrated_model(a.root);reference=a.root/'references/phc.npz'
    proxy=Path('/data/KinoGen/outputs/two_sitting_pipeline/assets_deploy_v3/vita_single_proxy.xml')
    env=NativeTracking(model,reference,n=a.num_envs,seed=174120,action_scale=a.action_scale,
                       sonic_base=True,proxy_path=proxy,threads=8)
    env.reset(torch.arange(env.n,device=env.device),force_start=True)
    rng=np.random.default_rng(174120)
    for i,d in enumerate(env.datas):
        if i:
            d.qpos[env.ref.qa]+=rng.uniform(-.01,.01,len(env.ref.qa))
            for rq in env.ref.root_qadr:d.qpos[rq:rq+2]+=rng.uniform(-.01,.01,2)
        mj.mj_forward(env.model,d);env._cache_one(i)
    env._generation+=1;env._host_generation=env._generation;env.refresh_policy_sensors()
    previous=torch.zeros((env.n,56),device=env.device)
    observations=[];labels=[];states=[];velocities=[];phases=[];valids=[]
    began=time.monotonic()
    for step in range(int(np.ceil(env.ref.horizon/.02-1e-8))):
        obs=env.ref.observations(env.q,env.v,env.xpos,env.phase,previous)
        target,_,_=env.ref.sample(env.phase)
        q=env.q.detach().cpu().numpy().copy();v=env.v.detach().cpu().numpy().copy();phase=env.phase.cpu().numpy().copy()
        with torch.no_grad():_,_,_,info=env.step(torch.zeros((env.n,56),device=env.device))
        delta=(env.u[:,env.ref.body_aids_t]-target[:,env.ref.qa_t])/a.action_scale
        action=torch.atanh(delta.clamp(-.98,.98))
        previous.copy_(action)
        tilt=np.stack([np.arccos(np.clip(1-2*(q[:,rq+4]**2+q[:,rq+5]**2),-1,1))for rq in env.ref.root_qadr],1)
        valid=(tilt.max(-1)<.8)&(q[:,env.ref.root_qadr+2].min(-1)>.45)&~env._numeric_fault
        observations.append(obs.cpu().numpy());labels.append(delta.clamp(-.98,.98).cpu().numpy())
        states.append(q);velocities.append(v);phases.append(phase);valids.append(valid)
    obs=np.asarray(observations);targets=np.asarray(labels);valid=np.asarray(valids)
    np.savez_compressed(a.output/'teacher_actual_rollouts.npz',observations=obs,normalized_motor_offsets=targets,
        qpos=np.asarray(states),qvel=np.asarray(velocities),phase_s=np.asarray(phases),valid=valid,
        original_sonic_weights_unchanged=np.asarray(True),source_reference_sha256=np.asarray(sha(reference)))
    x=torch.as_tensor(obs[valid],device=env.device);y=torch.as_tensor(targets[valid],device=env.device)
    perm=torch.randperm(len(x),device=env.device);split=int(.85*len(x));train,heldout=perm[:split],perm[split:]
    actor=ActorCritic(x.shape[-1],action_dim=56).to(env.device);actor.log_std.data.fill_(-2.3)
    actor.update_stats(x[train]);optimizer=torch.optim.Adam(actor.parameters(),lr=1e-3)
    losses=[]
    for epoch in range(a.epochs):
        rows=[]
        for batch in train[torch.randperm(len(train),device=env.device)].split(512):
            pred=torch.tanh(actor.actor(actor.normalize(x[batch])))
            loss=(pred-y[batch]).square().mean();optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(),1.);optimizer.step();rows.append(float(loss.detach()))
        with torch.no_grad():validation=float((torch.tanh(actor.actor(actor.normalize(x[heldout])))-y[heldout]).square().mean())
        losses.append({'epoch':epoch+1,'train_motor_offset_mse':sum(rows)/len(rows),'validation_motor_offset_mse':validation})
        if (epoch+1)%50==0:print(json.dumps(losses[-1]),flush=True)
    checkpoint=a.output/'bc_initial.pt'
    torch.save(dict(model=actor.state_dict(),optimizer=optimizer.state_dict(),iteration=0,
        obs_dim=x.shape[-1],action_dim=56,action_scale=a.action_scale,reference_path=str(reference),
        reference_sha256=sha(reference),model_path=str(model),model_sha256=sha(model),
        method='Independent privileged expert initialized by BC on actual stable SONIC data; not task success',
        privileged_observations=True,sonic_base=False),checkpoint)
    (a.output/'loss_curves.json').write_text(json.dumps(losses,indent=2)+'\n')
    report=dict(status='bc_initialization_complete_pending_independent_physics',checkpoint=str(checkpoint),
        checkpoint_sha256=sha(checkpoint),teacher='Unmodified frozen dual SONIC',teacher_worlds=env.n,
        teacher_trajectory_functional_success_claim=False,dataset_valid_samples=int(valid.sum()),
        dataset_total_samples=int(valid.size),training_samples=len(train),validation_samples=len(heldout),
        final_motor_target_RMSE_rad=float(validation**.5*a.action_scale),
        no_sonic_in_expert_inference=True,source_time_scale=1.,wall_seconds=time.monotonic()-began,
        limitations=['BC random frame holdout measures fitting only; actual closed-loop expert evaluation is required.',
                     'Teacher did not finish seating. RL must improve the interaction; expert is not yet a successful skill.'])
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True);env.close()


if __name__=='__main__':main()
