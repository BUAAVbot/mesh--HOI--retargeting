"""Convert the audited bounded motor torques into native PD position targets."""
import argparse
import json
from pathlib import Path

import mujoco as mj
import numpy as np

from common import configure_position_pd,load_canonical,physics_substeps,sha256,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    report=json.loads((args.audit/'report.json').read_text())
    assert report['reference_sha256']==sha256(args.reference) and report['model_sha256']==sha256(args.model)
    ref=load_canonical(args.reference);model=mj.MjModel.from_binary_path(str(args.model))
    configure_position_pd(model,mj)
    with np.load(args.audit/'dynamics_feasibility.npz',allow_pickle=False) as z:
        tau=z['actuator_torque_Nm'].copy();feasible=z['feasible'].copy()
    qadr=model.jnt_qposadr[model.actuator_trnid[:,0]];vadr=model.jnt_dofadr[model.actuator_trnid[:,0]]
    q,v=ref['qpos_ref'][:,qadr],ref['qvel_ref'][:,vadr]
    kp,kd=model.actuator_gainprm[:,0],-model.actuator_biasprm[:,2]
    assert tau.shape==q.shape==v.shape and np.isfinite(tau).all() and np.all(kp>0)
    target=q+(tau+kd*v)/kp
    np.testing.assert_allclose(kp*(target-q)-kd*v,tau,atol=1e-10)
    clipped=np.clip(target,model.actuator_ctrlrange[:,0],model.actuator_ctrlrange[:,1])
    data=mj.MjData(model)
    for frame in (0,len(q)//2,len(q)-1):
        data.qpos[:]=ref['qpos_ref'][frame];data.qvel[:]=ref['qvel_ref'][frame];data.ctrl[:]=clipped[frame]
        mj.mj_forward(model,data)
        predicted=np.clip(kp*(clipped[frame]-q[frame])-kd*v[frame],
            model.actuator_forcerange[:,0],model.actuator_forcerange[:,1])
        np.testing.assert_allclose(data.actuator_force,predicted,atol=1e-9)
    dt=float(ref['dt'].ravel()[0]);substeps=physics_substeps(dt,model.opt.timestep)
    steps=(len(q)-1)*substeps;times=np.arange(steps)*model.opt.timestep
    controls=np.column_stack([np.interp(times,np.arange(len(q))*dt,clipped[:,i]) for i in range(model.nu)])
    np.savez_compressed(args.output/'controls.npz',pd_target=controls,sim_dt=model.opt.timestep,
        frame_pd_target=clipped,unclipped_frame_pd_target=target,audited_motor_torque_Nm=tau,
        frame_audit_feasible=feasible)
    write_json(args.output/'manifest.json',dict(method='audited inverse dynamics PD feedforward',
        reference_sha256=sha256(args.reference),model_sha256=sha256(args.model),
        audit_npz_sha256=sha256(args.audit/'dynamics_feasibility.npz'),
        code_sha256=sha256(__file__),controls_sha256=sha256(args.output/'controls.npz'),
        clipped_frame_targets=int(np.count_nonzero(abs(clipped-target)>1e-12)),
        total_frame_targets=int(target.size),framewise_audit_fraction=float(feasible.mean()),
        no_root_force=True,control_semantics='joint PD position target; torque demand translated using frozen gains',
        status='unvalidated warm start; requires continuous replay'))


if __name__=='__main__':main()
