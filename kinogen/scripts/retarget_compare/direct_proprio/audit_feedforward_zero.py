"""Require full200Hz zero-feedback identity before learning a new feedback policy."""
from __future__ import annotations
import os,json,hashlib,time
from pathlib import Path


def main():
    import numpy as np,torch,mujoco as mj
    from physical_feedforward_env import PhysicalFeedforwardEnv,PRIOR
    root=Path('/data/KinoGen/outputs/sonic_expert_20261028')
    out=root/'audit/direct_feedforward_zero';out.mkdir(parents=True,exist_ok=True)
    initial=root/'prior_physical_state_pool/candidate156_initial_state.npz'
    os.environ['DIRECT_INITIAL_STATE_PATH']=str(initial);os.environ['FF_TRAIN_INITIAL_NOISE']='0'
    env=PhysicalFeedforwardEnv(root/'calibration/two_vita_expert.mjb',root/'references/phc_time_scale1.5.npz',
        n=1,seed=174140,action_scale=.15,sonic_base=False,threads=1)
    env.reset(torch.arange(1,device=env.device),force_start=True)
    z=dict(np.load(PRIOR,allow_pickle=False));spec=mj.mjtState(int(z['state_spec']))
    state=np.empty(mj.mj_stateSize(env.model,spec));mj.mj_getState(env.model,env.datas[0],state,spec)
    initial_error=float(abs(state-z['initial_state']).max())
    qerr=verr=uerr=0.;began=time.monotonic()
    for step in range(len(z['ctrl'])//4):
        env.step(torch.zeros((1,56),device=env.device))
        first=step*4
        uerr=max(uerr,float(abs(env.last_controls[0]-z['ctrl'][first:first+4]).max()))
        qerr=max(qerr,float(abs(env.last_qpos_steps[0]-z['qpos'][first+1:first+5]).max()))
        verr=max(verr,float(abs(env.last_qvel_steps[0]-z['qvel'][first+1:first+5]).max()))
    report=dict(status='zero_feedback_exact200Hz_replay_verified'if max(initial_error,qerr,verr,uerr)==0 else'FAILED',
        passed=max(initial_error,qerr,verr,uerr)==0,initial_integration_state_max_error=initial_error,
        qpos_max_absolute_error=qerr,qvel_max_absolute_error=verr,actual_control_max_absolute_error=uerr,
        physical_steps=len(z['ctrl']),duration_seconds=float(z['time'][-1]),
        source_clock_range_seconds=[float(z['source_time_s'][0]),float(z['source_time_s'][-1])],
        historical_command_delay_added=False,new_feedback_delay_physics_steps=3,
        extra_terminal_hold_added=False,prior=str(PRIOR),prior_sha256=hashlib.sha256(PRIOR.read_bytes()).hexdigest(),
        model=str(root/'calibration/two_vita_expert.mjb'),model_sha256=hashlib.sha256((root/'calibration/two_vita_expert.mjb').read_bytes()).hexdigest(),
        initial_state=str(initial),initial_sha256=hashlib.sha256(initial.read_bytes()).hexdigest(),
        wall_seconds=time.monotonic()-began,
        scope='This is an archived prior positive-control replay, not a newly trained expert task success.')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True);env.close()


if __name__=='__main__':main()
