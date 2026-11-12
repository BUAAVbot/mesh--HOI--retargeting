"""Freeze evaluation-only initial states for paired open-loop/learned-feedback tests."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/data/KinoGen/outputs/sonic_expert_20261028'))
    a=p.parse_args()
    import numpy as np,mujoco as mj
    seed=990013;count=32;root=a.root/'heldout_seed990013';root.mkdir(exist_ok=True)
    output=root/'initial_states.npz'
    if output.exists():raise FileExistsError('Heldout states are frozen; refusing to overwrite '+str(output))
    initial_path=a.root/'prior_physical_state_pool/candidate156_initial_state.npz'
    actual_path=a.root/'prior_physical_state_pool/candidate156_actual_states_controls.npz'
    model_path=a.root/'calibration/two_vita_expert.mjb'
    initial=dict(np.load(initial_path,allow_pickle=False));actual=dict(np.load(actual_path,allow_pickle=False))
    original=np.asarray(initial['initial_state'],dtype=np.float64);spec=mj.mjtState(int(initial['state_spec']))
    model=mj.MjModel.from_binary_path(str(model_path));scratch=mj.MjData(model)
    if len(original)!=mj.mj_stateSize(model,spec):raise ValueError('Integration state/model shape mismatch')
    mj.mj_setState(model,scratch,original,spec);original_q=scratch.qpos.copy();original_v=scratch.qvel.copy()
    roots=np.asarray(actual['robot_root_qadr'],int);names=list(map(str,actual['joint_names']))
    bodycols=np.asarray([i for i,name in enumerate(names)if not name.startswith(('lh_','rh_','neck_'))])
    qadr=np.asarray(actual['joint_qadr'])[:,bodycols].reshape(-1)
    joint_ids=np.asarray([mj.mj_name2id(model,mj.mjtObj.mjOBJ_JOINT,prefix+names[col])for prefix in ('p1_','p2_')for col in bodycols])
    limited=model.jnt_limited[joint_ids].astype(bool);lower,upper=model.jnt_range[joint_ids].T
    rng=np.random.default_rng(seed);states=[];qs=[];vs=[];root_delta=[];body_delta=[];requested_body=[]
    for case in range(count):
        mj.mj_resetData(model,scratch);mj.mj_setState(model,scratch,original,spec)
        xy=np.zeros((2,2));body=np.zeros(56)
        if case:
            xy=rng.uniform(-.01,.01,(2,2));body=rng.uniform(-.01,.01,56)
            for actor,rq in enumerate(roots):scratch.qpos[rq:rq+2]+=xy[actor]
            scratch.qpos[qadr]+=body
            values=scratch.qpos[qadr].copy();values[limited]=np.clip(values[limited],lower[limited],upper[limited]);scratch.qpos[qadr]=values
        state=original.copy()if case==0 else np.empty_like(original)
        if case:mj.mj_getState(model,scratch,state,spec)
        states.append(state);qs.append(scratch.qpos.copy());vs.append(scratch.qvel.copy())
        root_delta.append(np.asarray([scratch.qpos[rq:rq+2]-original_q[rq:rq+2]for rq in roots]))
        body_delta.append(scratch.qpos[qadr]-original_q[qadr]);requested_body.append(body)
    states,qs,vs,root_delta,body_delta,requested_body=map(np.asarray,(states,qs,vs,root_delta,body_delta,requested_body))
    if not np.array_equal(states[0],original)or not np.array_equal(vs,np.tile(original_v,(count,1))):raise RuntimeError('Original initial state/velocities changed')
    if np.max(abs(root_delta))>.01000000001 or np.max(abs(body_delta[1:]))>.01000000001:raise RuntimeError('Perturbation exceeded declared bound')
    legal=np.all((qs[1:,qadr][:,limited]>=lower[limited]-1e-12)&(qs[1:,qadr][:,limited]<=upper[limited]+1e-12))
    if not legal:raise RuntimeError('Perturbed body joints violate limits')
    np.savez_compressed(output,initial_states=states,states=states,state_spec=int(spec),case_ids=np.arange(count),
        qpos=qs,qvel=vs,root_qpos_addresses=roots,body_joint_qpos_addresses=qadr,
        root_xy_offset_m=root_delta,body_joint_offsets_rad=body_delta,requested_body_offsets_rad=requested_body,
        body_joint_names=np.asarray([prefix+names[col]for prefix in ('p1_','p2_')for col in bodycols]),
        seed=np.asarray(seed),evaluation_only=np.asarray(True),source_initial_sha256=np.asarray(sha(initial_path)),
        source_model_sha256=np.asarray(sha(model_path)))
    receipt={'schema':'sonic_expert.paired_heldout_initial_states.v1','status':'frozen_evaluation_only_do_not_train',
        'path':str(output),'sha256':sha(output),'cases':count,'seed':seed,'case0_bitwise_original_integration_state':True,
        'qvel_all_cases_bitwise_preserved':True,'all_perturbed_body_joints_legal':bool(legal),
        'root_xy_max_abs_offset_m':float(np.max(abs(root_delta))),'body_joint_max_abs_offset_rad':float(np.max(abs(body_delta))),
        'requested_vs_clipped_body_offset_max_difference_rad':float(np.max(abs(requested_body-body_delta))),
        'rng_draw_order':'For cases1..31: draw2x2 rootXY offsets, then56 ordered body-joint offsets; case0 draws no randomness.',
        'perturbation':'Only reset-time rootXY±0.01m and each28-body-joint±0.01rad per robot; clamp limited joints; all rootquaternions/qvel/other integration state retained.',
        'forward_recomputation_before_storage':False,'integration_state_size':int(states.shape[1]),'state_spec':int(spec),
        'source_initial_state':str(initial_path),'source_initial_sha256':sha(initial_path),
        'source_actual_data':str(actual_path),'source_actual_sha256':sha(actual_path),'model':str(model_path),'model_sha256':sha(model_path),
        'old_demonstration_source_label':'Old candidate156 functional actual initial state; not a new PHC expert success.',
        'training_use_prohibited':True,'source_sha256':sha(__file__)}
    (root/'initial_states.receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'path':str(output),'sha256':receipt['sha256'],'cases':count,'seed':seed,'max_root_offset':receipt['root_xy_max_abs_offset_m'],'max_body_offset':receipt['body_joint_max_abs_offset_rad']}),flush=True)


if __name__=='__main__':main()
