"""Run the upstream DSMS solver and G1 tracking objective on a Vita28 MJB."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import shutil

import mujoco as mj
import numpy as np

from common import configure_position_pd, load_canonical, physics_substeps, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dsms-root', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=150)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--node-dt', type=float, default=.02)
    parser.add_argument('--sim-dt', type=float, default=.005)
    parser.add_argument('--warmstart', type=Path)
    args = parser.parse_args()
    os.environ['TRAJOPT_ROOT_DIR'] = str(args.dsms_root)
    sys.path[:0] = [str(args.dsms_root), str(args.dsms_root/'examples/g1_gait')]
    from config import GaitTOConfig
    from g1_gait import G1GaitTO
    from src.dynamics import DynamicsConfig
    from src.multi_shooting import MultiShootingConfig
    from src.spline import SplineConfig

    source = load_canonical(args.reference)
    dt = float(source['dt'].ravel()[0])
    substeps=physics_substeps(dt,args.sim_dt)
    if args.iterations < 1 or args.threads < 1:
        raise ValueError('Expected positive solver limits')
    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__,args.output/'adapter_used.py')
    model = mj.MjModel.from_binary_path(str(args.model))
    configure_position_pd(model, mj)
    model.opt.timestep = args.sim_dt
    pd_model = args.output/'position_pd.mjb'
    mj.mj_saveModel(model, str(pd_model), None)
    root_body=next(mj.mj_id2name(model,mj.mjtObj.mjOBJ_BODY,i) for i in range(1,model.nbody)
                   if (mj.mj_id2name(model,mj.mjtObj.mjOBJ_BODY,i) or '').split('/')[-1]=='base_link')
    cfg = GaitTOConfig(name='vita28', traj_dir=(),
        w_base_pos=800., w_base_ori=800., w_base_linvel=1., w_base_angvel=1.,
        w_joint_pos=15., w_joint_vel=1., term_scale=2., Rtau=1e-6, Rrate=.01,
        ee_anchor=root_body, ee_track=())
    dyn_cfg = DynamicsConfig(model_path=str(pd_model), sim_dt=args.sim_dt,
        integrator=None, friction_cone=None, actuator_mode='position',
        # The frozen scene's contact transitions fail the gradient check at 1e-6.
        n_threads=args.threads, fd_eps=1e-7, fd_centered=True)
    frames = len(source['qpos_ref'])
    node_steps=physics_substeps(args.node_dt,args.sim_dt)
    if (frames-1)*substeps % node_steps:
        raise ValueError('Shooting intervals must exactly divide the physics horizon')
    nodes=(frames-1)*substeps//node_steps
    ms_cfg = MultiShootingConfig(N=nodes, node_dt=args.node_dt,
        spline=SplineConfig(M=frames, spline_type='linear'), keep_best_sol=False,
        ipopt_options={'linear_solver':'mumps', 'print_level':5,
            'max_iter':args.iterations, 'tol':1e-5, 'constr_viol_tol':1e-6,
            'acceptable_tol':1e-4, 'acceptable_constr_viol_tol':1e-5,
            'acceptable_iter':5, 'hessian_approximation':'limited-memory',
            'mu_strategy':'adaptive', 'max_cpu_time':1200.})
    ref = np.column_stack((source['qpos_ref'], source['qvel_ref']))
    problem = G1GaitTO(dyn_cfg, ms_cfg, cfg, ref, dt)
    problem.Qd[2] = 1500.
    problem.Qfd = cfg.term_scale*problem.Qd
    X0,U0 = problem.warm_start(problem.N_sim)
    if args.warmstart:
        with np.load(args.warmstart, allow_pickle=False) as data:
            X0, U0 = data['nodes'], data['pd_target']
        if X0.shape != (nodes+1, problem.nx) or U0.shape != (problem.N_sim, model.nu):
            raise ValueError('Warm start must use the same state/control grids')
    initial_z = problem.pack(problem.subsample_nodes(X0), problem.spline.fit(U0))
    # Check the adapter's objective gradient before the expensive solve.
    direction=np.random.default_rng(0).normal(size=initial_z.size)
    direction/=np.linalg.norm(direction)
    eps=1e-6
    numeric=(problem.objective(initial_z+eps*direction)-problem.objective(initial_z-eps*direction))/(2*eps)
    analytic=float(problem.gradient(initial_z)@direction)
    if not np.isclose(numeric,analytic,atol=2e-4,rtol=2e-3):
        raise AssertionError(('DSMS objective gradient',numeric,analytic))
    initial_defect=float(np.max(np.abs(problem.constraints(initial_z))))
    print(json.dumps(dict(initial_constraint_max_abs=initial_defect)),flush=True)
    started=time.perf_counter()
    X,U,info=problem.solve(X0,U0)
    elapsed=time.perf_counter()-started
    z=problem.pack(X,problem.spline.fit(U))
    defects=problem.constraints(z)
    np.savez_compressed(args.output/'controls.npz',pd_target=U,sim_dt=args.sim_dt)
    np.savez_compressed(args.output/'solution.npz',nodes=X,pd_target=U,
        stitched=problem.stitched_trajectory(X,U),constraint_residual=defects)
    status=info.get('status_msg',b'')
    if isinstance(status,bytes):status=status.decode()
    report=dict(method='Shooting for Contact / DSMS',upstream_commit='8b1bd4c5faaa4ac45a3dc1a3371e025f825e567f',
        upstream_adaptation='MJB loader only; upstream multiple-shooting and tracking objective',
        reference_sha256=sha256(args.reference),model_sha256=sha256(args.model),
        adapter_sha256=sha256(__file__),upstream_dynamics_sha256=sha256(args.dsms_root/'src/dynamics.py'),
        mujoco_version=mj.__version__,control_semantics='joint_position_target_rad',
        frames=frames,sim_dt=args.sim_dt,node_dt=args.node_dt,elapsed_s=elapsed,
        transition_fd_epsilon=dyn_cfg.fd_eps,
        solver_status=int(info['status']),solver_message=status,
        solver_converged=int(info['status']) in (0,1),
        constraint_max_abs=float(np.max(np.abs(defects))),
        initial_constraint_max_abs=initial_defect,
        objective_gradient_check=dict(numeric=numeric,analytic=analytic),
        continuous_replay_required=True,contact_parameters_unchanged=True,
        dependencies_note='Using existing MuJoCo 3.5.0 for binary-model parity; upstream environment requests >=3.8')
    write_json(args.output/'result.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
