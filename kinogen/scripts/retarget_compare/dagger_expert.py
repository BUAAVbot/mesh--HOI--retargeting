"""DAgger stability initialization at student states; teacher removed at evaluation/RL."""
from __future__ import annotations
import argparse,json,time,hashlib
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('/data/KinoGen/outputs/sonic_expert_20261028'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=180)
    a=p.parse_args()
    import numpy as np,torch,mujoco as mj
    from two_sitting_native_ppo_env import NativeTracking
    from two_sitting_ppo_env import ActorCritic
    torch.set_num_threads(2);torch.manual_seed(174121)
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    source=a.root/'bc_stability_bootstrap/bc_initial.pt';ck=torch.load(source,map_location='cuda:0',weights_only=False)
    env=NativeTracking(Path(ck['model_path']),Path(ck['reference_path']),n=16,seed=174121,
        action_scale=3.,sonic_base=True,proxy_path=Path('/data/KinoGen/outputs/two_sitting_pipeline/assets_deploy_v3/vita_single_proxy.xml'),threads=8)
    actor=ActorCritic(ck['obs_dim'],56).to(env.device);actor.load_state_dict(ck['model'])
    optimizer=torch.optim.Adam(actor.parameters(),lr=5e-4)
    data=np.load(a.root/'bc_stability_bootstrap/teacher_actual_rollouts.npz',allow_pickle=False)
    x=torch.as_tensor(data['observations'][data['valid']],device=env.device)
    y=torch.as_tensor(data['normalized_motor_offsets'][data['valid']],device=env.device)
    log=[];began=time.monotonic()
    for round_id,beta in enumerate((.75,.5,.25,.1,0.,0.)):
        env.reset(torch.arange(env.n,device=env.device),force_start=True)
        previous=torch.zeros((env.n,56),device=env.device);newx=[];newy=[];resets=0
        for step in range(int(np.ceil(env.ref.horizon/.02-1e-8))):
            obs=env.ref.observations(env.q,env.v,env.xpos,env.phase,previous)
            target,_,_=env.ref.sample(env.phase)
            with torch.no_grad():
                teacher=env.base_controls().clone();label=((teacher[:,env.ref.body_aids_t]-target[:,env.ref.qa_t])/ck['action_scale']).clamp(-.98,.98)
                pred=torch.tanh(actor.actor(actor.normalize(obs)))
                student_target=target[:,env.ref.qa_t]+ck['action_scale']*pred
                mixed=beta*teacher[:,env.ref.body_aids_t]+(1-beta)*student_target
                residual=torch.atanh(((mixed-teacher[:,env.ref.body_aids_t])/env.action_scale).clamp(-.98,.98))
                newx.append(obs.clone());newy.append(label.clone())
                _,_,done,info=env.step(residual)
                previous.copy_(torch.atanh(((mixed-target[:,env.ref.qa_t])/ck['action_scale']).clamp(-.98,.98)))
                bad=(info['root_angle']>1.)|torch.as_tensor(env._numeric_fault,device=env.device)
                if bad.any():
                    ids=bad.nonzero().flatten();env.reset(ids,force_start=True);previous[ids]=0;resets+=len(ids)
        x=torch.cat([x,*newx]);y=torch.cat([y,*newy]);actor.update_stats(x)
        for epoch in range(a.epochs):
            losses=[]
            for ids in torch.randperm(len(x),device=env.device).split(1024):
                loss=(torch.tanh(actor.actor(actor.normalize(x[ids])))-y[ids]).square().mean()
                optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(actor.parameters(),1.);optimizer.step();losses.append(float(loss.detach()))
        ck.update(model=actor.state_dict(),optimizer=optimizer.state_dict(),iteration=0,sonic_base=False,
                  method='DAgger-initialized independent expert; no SONIC at inference; task not yet accepted')
        torch.save(ck,a.output/f'round_{round_id+1}.pt')
        row={'round':round_id+1,'teacher_action_blend':beta,'training_resets':resets,'aggregate_samples':len(x),
             'motor_offset_mse':sum(losses)/len(losses),'wall_seconds':time.monotonic()-began}
        log.append(row);print(json.dumps(row),flush=True)
    checkpoint=a.output/'dagger_initial.pt';torch.save(ck,checkpoint)
    np.savez_compressed(a.output/'aggregate_training.npz',observations=x.cpu().numpy(),normalized_motor_offsets=y.cpu().numpy())
    report={'status':'dagger_initialization_finished_pending_physical_validation','checkpoint':str(checkpoint),
        'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'rounds':log,
        'teacher_only_during_data_collection':True,'expert_sonic_base':False,
        'source':'Own stability bootstrap, additional engineering beyond SUGAR; no SUGAR reproduction claim',
        'warning':'Training resets and supervised fitting do not count as successful task episodes.'}
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');env.close()


if __name__=='__main__':main()
