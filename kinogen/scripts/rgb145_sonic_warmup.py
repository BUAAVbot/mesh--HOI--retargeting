"""Closed-loop standing initialization, without repairing live robot qpos/qvel."""
from pathlib import Path
import json
import numpy as np
import mujoco as mj


def warmup(model,data,controller,seconds,output):
    if seconds<=0:return None
    keys=('qref','body_q','body_dq')
    if not all(hasattr(controller,k)for k in keys):raise ValueError('Warmup currently supports the original SONIC controller only')
    originals={k:getattr(controller,k)for k in keys}
    controller.qref=np.repeat(originals['qref'][:1],len(originals['qref']),axis=0)
    controller.body_q=np.repeat(originals['body_q'][:1],len(originals['body_q']),axis=0)
    controller.body_dq=np.zeros_like(originals['body_dq'])
    spec=int(mj.mjtState.mjSTATE_INTEGRATION);initial=np.empty(mj.mj_stateSize(model,spec));mj.mj_getState(model,data,initial,spec)
    times=[float(data.time)];q=[data.qpos.copy()];v=[data.qvel.copy()];ctrl=[];force=[]
    warnings=np.asarray([w.number for w in data.warning]);initial_root=data.qpos[:7].copy()
    try:
        for _ in range(int(np.ceil(seconds/model.opt.timestep))):
            packet=controller.action(data,float(data.time));data.ctrl[:]=packet
            if np.any(data.qfrc_applied)or np.any(data.xfrc_applied):raise ValueError('Warmup must use actuators only')
            mj.mj_step(model,data);times.append(float(data.time));q.append(data.qpos.copy());v.append(data.qvel.copy());ctrl.append(packet.copy());force.append(data.actuator_force.copy())
            if not np.isfinite(data.qpos).all()or not np.isfinite(data.qvel).all()or np.any(np.asarray([w.number for w in data.warning])>warnings):raise ValueError('Warmup physics failure')
    finally:
        for k,value in originals.items():setattr(controller,k,value)
    inference_count=controller.inference_count
    # Preserve proprioceptive/action history and the physically reached state.
    # Only the source-motion clock is restarted before the accepted rollout.
    data.time=0.;controller.next_inference_time=0.;controller.inference_count=0
    mj.mj_forward(model,data)
    output=Path(output);np.savez_compressed(output/'warmup_trace.npz',initial_state=initial,state_spec=spec,time=np.asarray(times),qpos=np.asarray(q),qvel=np.asarray(v),ctrl=np.asarray(ctrl),actuator_force=np.asarray(force))
    report={'mode':'closed_loop_constant_initial_pose_standing','duration_s':float(times[-1]-times[0]),'warmup_policy_inferences':inference_count,'live_qpos_qvel_repairs':0,'external_forces':False,'pose_snapshot_writes_after_initialization':0,'root_horizontal_shift_m':float(np.linalg.norm(data.qpos[:2]-initial_root[:2])),'final_root_qpos':data.qpos[:7].tolist(),'final_root_qvel':data.qvel[:6].tolist(),'controller_history_preserved_into_motion':True,'simulation_clock_reset_before_accepted_rollout':True,'meaning':'Initialization is a real closed-loop physical rollout. It is not part of the source action and does not establish task success.'}
    (output/'warmup_report.json').write_text(json.dumps(report,indent=2)+'\n');return report
