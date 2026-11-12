"""Offline direct-expert initialization from an audited previous physical success."""
from __future__ import annotations
import argparse,hashlib,json,os,time
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/data/KinoGen/outputs/sonic_expert_20261028'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=1000)
    a=p.parse_args()
    import numpy as np,torch,mujoco as mj
    from direct_proprio import DirectProprioEnv,augment,CONTRACT
    from two_sitting_ppo_env import ActorCritic
    torch.set_num_threads(2);torch.manual_seed(174130)
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    reference=a.root/'references/phc_time_scale1.5.npz';model=a.root/'calibration/two_vita_expert.mjb'
    initial=a.root/'prior_physical_state_pool/candidate156_initial_state.npz'
    actual=a.root/'prior_physical_state_pool/candidate156_actual_states_controls.npz'
    if not actual.exists():actual=a.root/'prior_physical_state_pool/actual_states_controls.npz'
    os.environ['DIRECT_INITIAL_STATE_PATH']=str(initial)
    env=DirectProprioEnv(model,reference,n=1,seed=174130,action_scale=1.5,sonic_base=False,threads=1)
    z=dict(np.load(actual,allow_pickle=False));times=z.get('time',z.get('time_s'))
    sample=np.arange(0,min(float(times[-1]),env.ref.horizon)-.01,.02)
    rows=np.clip(np.searchsorted(times,sample,side='left'),0,len(times)-1)
    previous=torch.zeros((1,56),device=env.device);scratch=mj.MjData(env.model)
    x=[];y=[];saturation=[]
    for row in rows:
        scratch.qpos[:]=z['qpos'][row];scratch.qvel[:]=z['qvel'][row]
        mj.mj_kinematics(env.model,scratch);mj.mj_comPos(env.model,scratch);mj.mj_comVel(env.model,scratch)
        tensor=lambda value:torch.as_tensor(value,device=env.device,dtype=torch.float32)[None]
        q,v,xp,cv=(tensor(value)for value in(scratch.qpos.copy(),scratch.qvel.copy(),scratch.xpos.copy(),scratch.cvel.copy()))
        phase=tensor(np.asarray(float(times[row]))).flatten();target,_,_=env.ref.sample(phase)
        obs=env.ref.observations(q,v,xp,phase,previous)
        obs=torch.cat([obs,augment(env.model,env.ref,q,v,phase,cv)],-1)
        command=z['ctrl'][min(int(row)+3,len(z['ctrl'])-1),env.ref.body_aids]
        delta=(tensor(command)-target[:,env.ref.qa_t])/1.5
        saturation.append(float((delta.abs()>.98).float().mean()))
        label=delta.clamp(-.98,.98);x.append(obs[0]);y.append(label[0]);previous=torch.atanh(label)
    x=torch.stack(x);y=torch.stack(y)
    np.savez_compressed(a.output/'offline_positive_observations.npz',observations=x.cpu().numpy(),
        normalized_motor_offsets=y.cpu().numpy(),actual_time_s=times[rows],actual_state_indices=rows)
    actor=ActorCritic(x.shape[-1],56).to(env.device)
    old=torch.load(a.root/'dagger_stability_bootstrap/round_6.pt',map_location=env.device,weights_only=False)
    state=actor.state_dict()
    for key,value in old['model'].items():
        if key in ('actor.0.weight','critic.0.weight'):
            state[key].zero_();state[key][:,:value.shape[1]]=value
        elif key not in ('obs_mean','obs_var','obs_count'):state[key]=value
    actor.load_state_dict(state);actor.log_std.data.fill_(float(np.log(.1)))
    actor.update_stats(x);optimizer=torch.optim.Adam(actor.parameters(),lr=1e-3)
    log=[];began=time.monotonic()
    for epoch in range(a.epochs):
        losses=[]
        for ids in torch.randperm(len(x),device=env.device).split(128):
            pred=torch.tanh(actor.actor(actor.normalize(x[ids])))
            loss=(pred-y[ids]).square().mean();optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(),1.);optimizer.step();losses.append(float(loss.detach()))
        log.append({'epoch':epoch+1,'motor_offset_mse':sum(losses)/len(losses)})
        if (epoch+1)%200==0:print(json.dumps(log[-1]),flush=True)
    path=a.output/'positive_bc_initial.pt'
    torch.save(dict(model=actor.state_dict(),optimizer=optimizer.state_dict(),iteration=0,
        obs_dim=x.shape[-1],action_dim=56,action_scale=1.5,reference_path=str(reference),
        reference_sha256=sha(reference),model_path=str(model),model_sha256=sha(model),
        method='Direct privileged proprio expert initialized from declared prior physical success; not a new PHC success',
        sonic_base=False,privileged_observations=True,expert_observation_contract='direct_proprio_v1',
        fixed_initial_state_path=str(initial),fixed_initial_state_sha256=sha(initial)),path)
    report=dict(status='positive_physical_bc_fitted_pending_closed_loop',checkpoint=str(path),checkpoint_sha256=sha(path),
        old_success_data=str(actual),old_success_data_sha256=sha(actual),samples=len(x),
        reference=str(reference),reference_sha256=sha(reference),physical_initial_state=str(initial),physical_initial_sha256=sha(initial),
        actual_execution_speed_preserved=True,reference_time_scale=1.5,
        observation_contract=CONTRACT,extra_observation_channels=138,
        warm_start='DAgger MLP first layer expanded with old columns retained and new proprio columns initially zero',
        motor_target_training_RMSE_rad=1.5*log[-1]['motor_offset_mse']**.5,
        clipped_target_fraction=float(np.mean(saturation)),wall_seconds=time.monotonic()-began,
        limitations=['Prior archived candidate156 was previously physically successful. It is not a new PHC policy result.',
                     'Offline features use archived actual integrated states; fitting error is not a physical task score.',
                     'Direct expert never executes SONIC; standalone new closed-loop and control replay required.'])
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (a.output/'loss_curves.json').write_text(json.dumps(log,indent=2)+'\n');print(json.dumps(report),flush=True);env.close()


if __name__=='__main__':main()
