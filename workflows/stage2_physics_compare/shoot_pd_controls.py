"""Optimize leg/waist PD offsets by continuous native single shooting."""
import argparse
import json
from pathlib import Path
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import shutil

import mujoco as mj
from mujoco import rollout
import numpy as np
from scipy.optimize import least_squares

from common import classify_robot_static_bodies,configure_position_pd,load_canonical,physics_substeps,sha256,write_json
from shooting_residual import ShootingResidual,initialize_worker,evaluate_worker,orientation_error


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--seed',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--knots',type=int,default=3)
    parser.add_argument('--evaluations',type=int,default=40)
    parser.add_argument('--finite-difference-step',type=float,default=2e-4)
    parser.add_argument('--initial-state',type=Path)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,args.output/'optimizer_used.py')
    shutil.copy2(Path(__file__).with_name('shooting_residual.py'),args.output/'residual_used.py')
    ref=load_canonical(args.reference);model=mj.MjModel.from_binary_path(str(args.model))
    configure_position_pd(model,mj);substeps=physics_substeps(float(ref['dt'].ravel()[0]),model.opt.timestep)
    n=len(ref['qpos_ref']);steps=(n-1)*substeps
    with np.load(args.seed,allow_pickle=False) as z:base=z['pd_target'].copy()
    assert base.shape==(steps,model.nu) and np.isfinite(base).all()
    assert args.knots>=2 and args.evaluations>=1
    active=np.array([i for i,j in enumerate(model.actuator_trnid[:,0])
        if any(token in model.joint(j).name for token in ('hip_','knee_','ankle_','waist_'))])
    assert len(active)==14
    tk=np.linspace(0.,steps-1,args.knots);query=np.arange(steps)
    basis=np.column_stack([np.interp(query,tk,np.eye(args.knots)[:,k]) for k in range(args.knots)])
    np.testing.assert_allclose(basis.sum(axis=1),1.)
    initial=np.r_[0.,ref['qpos_ref'][0],ref['qvel_ref'][0]]
    initial_warmstart=np.zeros(model.nv)
    if args.initial_state:
        with np.load(args.initial_state,allow_pickle=False) as packet:
            assert str(packet['model_sha256'])==sha256(args.model)
            assert packet['qpos'].shape==(model.nq,) and packet['qvel'].shape==(model.nv,)
            assert packet['qacc_warmstart'].shape==(model.nv,)
            initial=np.r_[float(packet['time_s']),packet['qpos'],packet['qvel']]
            initial_warmstart=packet['qacc_warmstart'].copy()
    assert np.isfinite(initial).all() and np.isfinite(initial_warmstart).all() and model.na==0
    pool=[mj.MjData(model) for _ in range(8)];fk=mj.MjData(model)
    cache={};trace=[];started=time.monotonic()
    variables=args.knots*len(active);epsilon=args.finite_difference_step
    assert 0<epsilon<.1

    def controls(x):
        u=base.copy();u[:,active]+=basis@x.reshape(args.knots,len(active))
        return np.clip(u,model.actuator_ctrlrange[:,0],model.actuator_ctrlrange[:,1])

    def simulate(us):
        result,_=rollout.rollout(model,pool,np.tile(initial,(len(us),1)),control=np.asarray(us),
            initial_warmstart=np.tile(initial_warmstart,(len(us),1)),persistent_pool=True)
        assert result.shape==(len(us),steps,1+model.nq+model.nv) and np.isfinite(result).all()
        np.testing.assert_allclose(result[:,:,0],np.broadcast_to(initial[0]+np.arange(1,steps+1)*model.opt.timestep,(len(us),steps)),atol=1e-9,rtol=0)
        return result[:,:,1:]

    # Native batched shooting must match a separate stateful replay before optimization.
    check=simulate([base])[0];d=mj.MjData(model);d.time=initial[0];d.qpos[:]=initial[1:1+model.nq];d.qvel[:]=initial[1+model.nq:]
    mj.mj_forward(model,d);d.qacc_warmstart[:]=initial_warmstart;states=[]
    for u in base:
        d.ctrl[:]=u;mj.mj_step(model,d);states.append(np.r_[d.qpos.copy(),d.qvel.copy()])
    np.testing.assert_allclose(check,np.array(states),atol=1e-11,rtol=1e-12)

    residual=ShootingResidual(args.model,args.reference,initial,args.knots)
    workers=ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn'),
        initializer=initialize_worker,initargs=(args.model,args.reference,initial,args.knots))

    def evaluate(x):
        if 'x' not in cache or not np.array_equal(cache['x'],x):
            state=simulate([controls(x)])[0];r=residual((x,state))
            cache.update(x=x.copy(),residual=r,state=state)
            row=dict(evaluation=len(trace),wall_s=time.monotonic()-started,cost=float(r@r),
                root_rmse_m=float(np.sqrt(np.mean(np.sum((state[substeps-1::substeps,:3]-ref['qpos_ref'][1:,:3])**2,axis=1)))))
            trace.append(row);print(json.dumps(row),flush=True)
            write_json(args.output/'progress.json',dict(trace=trace))
            np.savez_compressed(args.output/'checkpoint.npz',pd_target=controls(x),offset_knots=x.reshape(args.knots,len(active)))
        return cache['residual']

    def jacobian(x):
        perturb=np.tile(x,(2*variables,1))
        perturb[np.arange(variables),np.arange(variables)]+=epsilon
        perturb[variables+np.arange(variables),np.arange(variables)]-=epsilon
        trajectories=simulate([controls(item) for item in perturb])
        values=list(workers.map(evaluate_worker,zip(perturb,trajectories),chunksize=2))
        return ((np.array(values[:variables])-np.array(values[variables:]))/(2*epsilon)).T

    result=least_squares(evaluate,np.zeros(variables),jac=jacobian,bounds=(-.6,.6),
        max_nfev=args.evaluations,ftol=1e-6,xtol=1e-6,gtol=1e-5,x_scale=.1,verbose=0)
    workers.shutdown()
    final=simulate([controls(result.x)])[0];np.savez_compressed(args.output/'controls.npz',pd_target=controls(result.x),
        sim_dt=model.opt.timestep,offset_knots=result.x.reshape(args.knots,len(active)),active_actuators=active)
    np.savez_compressed(args.output/'native_shooting_record.npz',states=np.vstack([initial[1:],final]))
    write_json(args.output/'result.json',dict(method='native single shooting + SciPy bounded least squares',
        reference_sha256=sha256(args.reference),model_sha256=sha256(args.model),seed_sha256=sha256(args.seed),
        code_sha256=sha256(args.output/'optimizer_used.py'),residual_code_sha256=sha256(args.output/'residual_used.py'),
        source_frames=n,physics_steps=steps,knots=args.knots,variables=variables,
        initial_packet_sha256=sha256(args.initial_state) if args.initial_state else None,
        finite_difference_step_rad=epsilon,initial_state_preserved=True,no_root_force=True,
        native_rollout_stateful_replay_equivalent=True,solver_status=int(result.status),message=result.message,
        function_evaluations=result.nfev,jacobian_evaluations=result.njev,wall_s=time.monotonic()-started,
        initial_objective=trace[0]['cost'],final_objective=float(result.fun@result.fun),
        validation='pending independent continuous replay'))


if __name__=='__main__':main()
