"""Capture numerical integration state of a verified CEM plan, not an RL expert."""
from pathlib import Path
import hashlib,json
import numpy as np
import mujoco as mj

ROOT=Path('/data/KinoGen/outputs/sonic_expert_20261028')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    trial=ROOT/'terminal_refined_full_v4';out=ROOT/'cem_v4_physical_state_pool'
    source=out/'terminal_refined_v4_actual_time_physical_contact_valid_pool.npz'
    pool=dict(np.load(source,allow_pickle=False));z=dict(np.load(trial/'trace.npz',allow_pickle=False))
    model=mj.MjModel.from_binary_path(str(trial/'model.mjb'));data=mj.MjData(model)
    spec=mj.mjtState(int(z['state_spec']));ids=np.asarray(pool['state_index'],int)
    states=np.empty((len(z['qpos']),mj.mj_stateSize(model,spec)))
    mj.mj_setState(model,data,z['initial_state'],spec);mj.mj_forward(model,data)
    qerr=verr=0.
    for i in range(len(states)):
        mj.mj_getState(model,data,states[i],spec)
        qerr=max(qerr,float(np.max(abs(data.qpos-z['qpos'][i]))))
        verr=max(verr,float(np.max(abs(data.qvel-z['qvel'][i]))))
        if i<len(z['ctrl']):data.ctrl[:]=z['ctrl'][i];mj.mj_step(model,data)
    if qerr!=0 or verr!=0:raise ValueError(f'Source replay differs: {qerr}, {verr}')
    fullfile=out/'terminal_refined_v4_actual_time_full_integration_physical_pool.npz'
    pool.update(integration_state=states[ids],integration_state_spec=np.full(len(ids),int(spec),dtype=np.int64),
                has_integration_state=np.ones(len(ids),bool),feedback_target=np.zeros((len(ids),56)))
    np.savez_compressed(fullfile,**pool)
    allfile=out/'terminal_refined_v4_full_integration_200hz.npz'
    np.savez_compressed(allfile,states=states,state_spec=np.asarray(int(spec)),physics_index=np.arange(len(states)),
        time_s=z['time'],q=z['qpos'],v=z['qvel'],ctrl=z['ctrl'],source_trace_sha256=np.asarray(sha(trial/'trace.npz')))
    checks=[]
    samples=np.unique(ids[np.linspace(0,len(ids)-1,7,dtype=int)])
    for index in samples:
        row={'physics_index':int(index),'time_s':float(z['time'][index]),'window_steps':40}
        for mode in ('full_integration','qv_only','full_without_restore_after_forward'):
            d=mj.MjData(model)
            if mode.startswith('full'):
                mj.mj_setState(model,d,states[index],spec);mj.mj_forward(model,d)
                if mode=='full_integration':mj.mj_setState(model,d,states[index],spec)
            else:
                mj.mj_setState(model,d,z['initial_state'],spec)
                d.qpos[:]=z['qpos'][index];d.qvel[:]=z['qvel'][index];d.ctrl[:]=z['ctrl'][index];d.time=float(z['time'][index])
                mj.mj_forward(model,d)
            qe=ve=0.
            for step in range(int(index),int(index)+40):
                d.ctrl[:]=z['ctrl'][step];mj.mj_step(model,d)
                qe=max(qe,float(np.max(abs(d.qpos-z['qpos'][step+1]))))
                ve=max(ve,float(np.max(abs(d.qvel-z['qvel'][step+1]))))
            row[mode]={'max_qpos_absolute_error':qe,'max_qvel_absolute_error':ve,'bitwise_exact':qe==0 and ve==0}
        checks.append(row)
    receipt={'schema':'sonic_expert.full_integration_physical_pool.v1','source_is_cem_not_ppo':True,
      'source_trace':str(trial/'trace.npz'),'source_trace_sha256':sha(trial/'trace.npz'),'model_sha256':sha(trial/'model.mjb'),
      'qv_pool_source':str(source),'qv_pool_sha256':sha(source),'full_pool':str(fullfile),'full_pool_sha256':sha(fullfile),
      'states':len(ids),'state_size':states.shape[1],'state_spec':int(spec),'all_200hz_file':str(allfile),'all_200hz_sha256':sha(allfile),
      'source_replay_qpos_max_error':qerr,'source_replay_qvel_max_error':verr,'selected_indices_unchanged':True,
      'physical_validity_gates_unchanged':True,'q_v_ctrl_phase_unchanged':True,'feedback_history_reset_zero':True,
      'reset_contract':'mj_setState(full); mj_forward; mj_setState(full) again to restore numerical warmstart. No source velocity scaling.',
      'continuation_checks':checks,'audit_source_sha256':sha(__file__)}
    (out/'full_integration_export_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
