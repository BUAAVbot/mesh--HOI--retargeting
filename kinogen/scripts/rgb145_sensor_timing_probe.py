"""Compare partial-reset sensor timing, untouched GPU dynamics, and native MuJoCo.

No policy training occurs. All real steps use the same held actuator controls.
Only world 1 in one GPU batch is reset; world 0 must remain independent.
"""
from pathlib import Path
import argparse,dataclasses,hashlib,json,sys


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--sonic-base',action='store_true')
    p.add_argument('--prefix-control-steps',type=int,default=5)
    p.add_argument('--finger-driver-target',type=float)
    p.add_argument('--native-qpos-tolerance',type=float,default=.003)
    p.add_argument('--native-qvel-tolerance',type=float,default=.1)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    import numpy as np,torch,mujoco as mj,warp as wp
    from rgb145_ppo_env import WarpTracking,POLICY_SENSOR_CONTRACT
    torch.set_num_threads(2)
    # Distinct physical arrays/graphs; equal initialization and held controls.
    tested=WarpTracking(a.model,a.reference,2,145317,1.6,a.sonic_base)
    baseline=WarpTracking(a.model,a.reference,2,145317,1.6,a.sonic_base)
    repeat=WarpTracking(a.model,a.reference,2,145317,1.6,a.sonic_base)
    all_ids=torch.arange(2,device=tested.device)
    for env in (tested,baseline,repeat):env.reset(all_ids,force_start=True)
    held=tested.ref.controls(torch.zeros(2,device=tested.device),torch.zeros((2,28),device=tested.device),0.)
    if a.finger_driver_target is not None:
        for col,name in enumerate(tested.ref.names):
            aid=int(tested.ref.all_aids[col]);jid=int(tested.model.actuator_trnid[aid,0])
            if name.startswith(('lh_','rh_'))and np.any(tested.model.actuator_gear[aid])and tested.model.actuator_gainprm[aid,0]!=0:
                lo,hi=tested.model.jnt_range[jid];held[:,aid]=float(np.clip(a.finger_driver_target,lo+.01,hi-.01))
    for env in (tested,baseline,repeat):env.u.copy_(held)
    cpu=mj.MjData(tested.model)
    cpu.qpos[:]=tested.q[0].cpu().numpy();cpu.qvel[:]=tested.v[0].cpu().numpy();cpu.ctrl[:]=held[0].cpu().numpy()
    mj.mj_forward(tested.model,cpu)
    if hasattr(tested.d,'qacc_warmstart'):cpu.qacc_warmstart[:]=wp.to_torch(tested.d.qacc_warmstart)[0].cpu().numpy()
    def maxdiff(x,y):return float((x-y).abs().max())
    native_rows=[]
    follower_qa=tested.model.jnt_qposadr[tested.model.eq_obj1id]
    driver_qa=tested.model.jnt_qposadr[tested.model.eq_obj2id]
    qnames=['root_x','root_y','root_z','root_qw','root_qx','root_qy','root_qz']+tested.ref.names
    vnames=['root_vx','root_vy','root_vz','root_wx','root_wy','root_wz']+tested.ref.names
    hand_cols=np.asarray([k for k,n in enumerate(tested.ref.names)if n.startswith(('lh_','rh_'))])
    hand_qa=7+hand_cols;hand_va=6+hand_cols
    def grouped_errors(qerror,verror):
        return {'qpos_abs_max':float(qerror.max()),'qvel_abs_max':float(verror.max()),
            'qpos_worst_channel':qnames[int(qerror.argmax())],'qvel_worst_channel':vnames[int(verror.argmax())],
            'free_root_qpos_abs_max':float(qerror[:7].max()),'free_root_qvel_abs_max':float(verror[:6].max()),
            'body28_qpos_abs_max':float(qerror[tested.ref.qa].max()),'body28_qvel_abs_max':float(verror[tested.ref.va].max()),
            'hands_qpos_abs_max':float(qerror[hand_qa].max()),'hands_qvel_abs_max':float(verror[hand_va].max())}
    qcurve=[];vcurve=[];gpu_equalities=[];cpu_equalities=[]
    def equality_residual(q):
        follower=q[follower_qa]-tested.model.qpos0[follower_qa]
        driver=q[driver_qa]-tested.model.qpos0[driver_qa]
        return follower-np.sum(tested.model.eq_data[:,:5]*driver[:,None]**np.arange(5)[None],axis=1)
    def advance(env):
        env.u.copy_(held)
        if env.sonic is not None:env.sonic.record_applied(env.u)
        env.wp.capture_launch(env.graph);env.refresh_policy_sensors();env.phase+=env.dt;env.steps+=1
        # Populate the same SONIC observation cache used during training.
        env.observations()
    def native_error(stage):
        gpu_q=tested.q[0].cpu().numpy();gpu_v=tested.v[0].cpu().numpy()
        qerror=abs(gpu_q-cpu.qpos);verror=abs(gpu_v-cpu.qvel)
        qcurve.append(qerror);vcurve.append(verror)
        gpu_eq=equality_residual(gpu_q);cpu_eq=equality_residual(cpu.qpos)
        gpu_equalities.append(gpu_eq);cpu_equalities.append(cpu_eq)
        native_rows.append({'stage':stage,'time_s':float(cpu.time),**grouped_errors(qerror,verror),
            'o6_follower_qpos_difference_max_rad':float(qerror[follower_qa].max()),
            'o6_gpu_polynomial_residual_max_rad':float(abs(gpu_eq).max()),
            'o6_native_polynomial_residual_max_rad':float(abs(cpu_eq).max()),
            'xpos_abs_max':float(np.max(abs(tested.xpos[0].cpu().numpy()-cpu.xpos))),
            'xmat_abs_max':float(np.max(abs(tested.rotation[0].cpu().numpy().reshape(cpu.xmat.shape)-cpu.xmat))),
            'cvel_abs_max':float(np.max(abs(tested.cvel[0].cpu().numpy()-cpu.cvel)))})
    for k in range(a.prefix_control_steps):
        advance(tested);advance(baseline);advance(repeat)
        for _ in range(4):mj.mj_step(tested.model,cpu)
        native_error('prefix_'+str(k+1))
    wp.synchronize()
    pre_alignment_difference=grouped_errors((tested.q[0]-baseline.q[0]).abs().cpu().numpy(),(tested.v[0]-baseline.v[0]).abs().cpu().numpy())
    def data_arrays(obj,prefix='',seen=None):
        """Enumerate nested Data/Contact numeric arrays without rebinding them."""
        if isinstance(obj,wp.array):
            yield prefix,obj
            return
        if seen is None:seen=set()
        if id(obj)in seen:return
        seen.add(id(obj))
        if isinstance(obj,dict):members=obj
        elif isinstance(obj,(list,tuple)):members={str(i):v for i,v in enumerate(obj)}
        else:
            struct_members=getattr(getattr(obj,'_cls',None),'vars',{})
            if dataclasses.is_dataclass(obj):members={f.name:getattr(obj,f.name)for f in dataclasses.fields(obj)}
            elif struct_members:members={name:getattr(obj,name)for name in struct_members}
            elif type(obj).__module__.startswith('warp'):return
            else:
                try:members={k:v for k,v in vars(obj).items()if not k.startswith('_')}
                except TypeError:members={}
        for name,value in members.items():
            if name in ('_cls','_ctype','_struct','_type_'):continue
            if isinstance(value,(str,bytes,int,float,bool,type,type(None))):continue
            key=prefix+'.'+name if prefix else str(name)
            yield from data_arrays(value,key,seen)
    def align_full_state(dst,src):
        source=dict(data_arrays(src.d));destination=dict(data_arrays(dst.d))
        required={'qpos','qvel','qacc_warmstart','time','ctrl'}
        if not required.issubset(source):raise AssertionError({'missing_data_arrays':sorted(required-set(source)),'discovered':sorted(source)})
        if source.keys()!=destination.keys():raise AssertionError('Data layouts differ')
        for name,array in source.items():
            target=destination[name]
            if target.shape!=array.shape or target.dtype!=array.dtype:raise AssertionError('Data array ABI differs: '+name)
            if array.size:wp.copy(target,array)
        wp.synchronize()
        unequal=[name for name,array in source.items()if array.size and array.numpy().tobytes()!=destination[name].numpy().tobytes()]
        if unequal:raise AssertionError({'data_clone_not_bitwise_equal':unequal})
        for name in ('phase','previous','from_start','steps','xpos','rotation','cvel'):getattr(dst,name).copy_(getattr(src,name))
        if src.sonic is not None:
            for name in ('valid','phase','last','previous_imu','previous_gravity','cached'):getattr(dst.sonic,name).copy_(getattr(src.sonic,name))
            for name,value in src.sonic.history.items():dst.sonic.history[name].copy_(value)
            for name,value in src.sonic.tokens.items():dst.sonic.tokens[name].copy_(value)
            dst.sonic.inference_count=src.sonic.inference_count
        wp.synchronize()
        return {'array_count':len(source),'array_names':sorted(source),'bitwise_equal':not unequal,'method':'wp.copy into captured arrays; integration, contact and solver cache arrays all copied; policy caches copied separately'}
    # Start the causal comparison at identical complete physical/cache state.
    # A second no-reset branch estimates GPU FP32 variation without a reset.
    alignment={'tested':align_full_state(tested,baseline),'repeat':align_full_state(repeat,baseline)}
    before_q=tested.q[0].clone();before_v=tested.v[0].clone()
    physical_before={n:getattr(tested,'_physics_'+n)[0].clone()for n in ('xpos','rotation','cvel')}
    policy_before={n:getattr(tested,n)[0].clone()for n in physical_before}
    warmstart_before=wp.to_torch(tested.d.qacc_warmstart)[0].clone()if hasattr(tested.d,'qacc_warmstart')else None
    sonic_before=None
    if tested.sonic is not None:
        sonic_before={n:v[0].clone()for n,v in tested.sonic.history.items()}
        sonic_before.update(cached=tested.sonic.cached[0].clone(),previous_imu=tested.sonic.previous_imu[0].clone(),previous_gravity=tested.sonic.previous_gravity[0].clone(),last=tested.sonic.last[0].clone())
    tested.record_reset_diagnostics=True
    tested.reset(torch.tensor([1],device=tested.device),force_start=True)
    tested.observations();wp.synchronize()
    raw_change={n:maxdiff(getattr(tested,'_physics_'+n)[0],old)for n,old in physical_before.items()}
    policy_change={n:maxdiff(getattr(tested,n)[0],old)for n,old in policy_before.items()}
    reset_state_change={'qpos_abs_max':maxdiff(tested.q[0],before_q),'qvel_abs_max':maxdiff(tested.v[0],before_v)}
    warmstart_change=maxdiff(wp.to_torch(tested.d.qacc_warmstart)[0],warmstart_before)if warmstart_before is not None else None
    sonic_preserved=None
    if sonic_before is not None:
        values={n:v[0]for n,v in tested.sonic.history.items()}
        values.update(cached=tested.sonic.cached[0],previous_imu=tested.sonic.previous_imu[0],previous_gravity=tested.sonic.previous_gravity[0],last=tested.sonic.last[0])
        sonic_preserved=all(torch.equal(values[n],old)for n,old in sonic_before.items())
    before_trajectory_difference=grouped_errors((tested.q[0]-baseline.q[0]).abs().cpu().numpy(),(tested.v[0]-baseline.v[0]).abs().cpu().numpy())
    advance(tested);advance(baseline);advance(repeat)
    for _ in range(4):mj.mj_step(tested.model,cpu)
    wp.synchronize();native_error('after_partial_reset_next_control')
    after_trajectory_difference=grouped_errors((tested.q[0]-baseline.q[0]).abs().cpu().numpy(),(tested.v[0]-baseline.v[0]).abs().cpu().numpy())
    natural_qerror=(repeat.q[0]-baseline.q[0]).abs().cpu().numpy();natural_verror=(repeat.v[0]-baseline.v[0]).abs().cpu().numpy()
    natural_difference=grouped_errors(natural_qerror,natural_verror)
    reset_qerror=(tested.q[0]-baseline.q[0]).abs().cpu().numpy();reset_verror=(tested.v[0]-baseline.v[0]).abs().cpu().numpy()
    strict_all=after_trajectory_difference['qpos_abs_max']<=1e-6 and after_trajectory_difference['qvel_abs_max']<=1e-5
    within_observed_variance=after_trajectory_difference['hands_qpos_abs_max']<=natural_difference['hands_qpos_abs_max']+1e-6 and after_trajectory_difference['hands_qvel_abs_max']<=natural_difference['hands_qvel_abs_max']+1e-5
    auxiliary_assessment='passed_strict_all_channel_gate'if strict_all else'within_one_observed_no_reset_repeat_plus_declared_absolute_floor'if within_observed_variance else'inconclusive_above_one_observed_no_reset_repeat'
    # Reproduce the old sensor timing shift only after all physical comparison
    # samples have been recorded, on the disposable no-reset control batch.
    legacy_before={n:getattr(baseline,'_physics_'+n)[0].clone()for n in ('xpos','rotation','cvel')}
    baseline.mw.forward(baseline.m,baseline.d);wp.synchronize()
    legacy_change={n:maxdiff(getattr(baseline,'_physics_'+n)[0],old)for n,old in legacy_before.items()}
    checks={'partial_reset_did_not_write_world0_state':max(reset_state_change.values())==0.,
        'legacy_global_forward_timing_difference_reproduced':max(legacy_change.values())>1e-8,
        'unrelated_raw_physics_sensors_unchanged':max(raw_change.values())==0.,
        'world0_policy_sensors_unchanged':max(policy_change.values())==0.,
        'unrelated_solver_warmstart_unchanged':warmstart_change is not None and warmstart_change==0.,
        'unrelated_sonic_history_unchanged':sonic_preserved is not False,
        'gpu_full_data_and_cache_aligned_before_partial_reset':all(v['bitwise_equal']for v in alignment.values())and before_trajectory_difference['qpos_abs_max']==0. and before_trajectory_difference['qvel_abs_max']==0.,
        'world0_next_free_root_qpos_independent':after_trajectory_difference['free_root_qpos_abs_max']<=1e-6,
        'world0_next_free_root_qvel_independent':after_trajectory_difference['free_root_qvel_abs_max']<=1e-5,
        'world0_next_body28_qpos_independent':after_trajectory_difference['body28_qpos_abs_max']<=1e-6,
        'world0_next_body28_qvel_independent':after_trajectory_difference['body28_qvel_abs_max']<=1e-5,
        'native_qpos_within_declared_tolerance':max(r['qpos_abs_max']for r in native_rows)<=a.native_qpos_tolerance,
        'native_qvel_within_declared_tolerance':max(r['qvel_abs_max']for r in native_rows)<=a.native_qvel_tolerance}
    out={'schema':'rgb145.sensor_timing_probe.v2','status':'passed_sensor_and_body_checks'if all(checks.values())else'failed',
        'checks':checks,'mujoco_version':mj.__version__,'warp_version':wp.__version__,
        'model_sha256':sha(a.model),'reference_sha256':sha(a.reference),'probe_sha256':sha(__file__),
        'environment_source_sha256':sha(Path(__file__).with_name('rgb145_ppo_env.py')),
        'policy_sensor_contract':POLICY_SENSOR_CONTRACT,'sonic_base':a.sonic_base,'nworld':2,
        'prefix_control_steps':a.prefix_control_steps,'finger_driver_target':a.finger_driver_target,'policy_dt_s':tested.dt,'physics_dt_s':float(tested.model.opt.timestep),
        'held_ctrl':held[0].cpu().tolist(),'raw_world0_kinematics_changed_by_other_world_reset':raw_change,
        'legacy_global_forward_world0_sensor_shift':legacy_change,
        'cached_world0_policy_sensor_change':policy_change,'world0_state_change_during_other_reset':reset_state_change,
        'world0_warmstart_change_during_other_reset':warmstart_change,
        'solver_cache_diagnostics':getattr(tested,'last_reset_solver_cache_diagnostics',{}),
        'sonic_world0_history_preserved':sonic_preserved,'gpu_world0_before_reset_vs_no_reset_control':before_trajectory_difference,
        'gpu_world0_after_reset_vs_no_reset_control':after_trajectory_difference,'native_comparison':native_rows,
        'gpu_world0_before_full_state_alignment':pre_alignment_difference,'full_state_alignment':alignment,
        'gpu_no_reset_repeat_vs_no_reset_control':natural_difference,'auxiliary_fp32_variance_assessment':auxiliary_assessment,
        'all_channel_strict_diagnostic_gate':{'qpos_tolerance':1e-6,'qvel_tolerance':1e-5,'passed':strict_all},
        'variance_caveat':'One paired no-reset repeat quantifies observed numerical variation, not a confidence interval or proof of all-channel independence. Task acceptance thresholds are unchanged.',
        'native_tolerances':{'qpos_abs_max':a.native_qpos_tolerance,'qvel_abs_max':a.native_qvel_tolerance},
        'o6_equality_names':[mj.mj_id2name(tested.model,mj.mjtObj.mjOBJ_EQUALITY,e)for e in range(tested.model.neq)],
        'o6_equality_coefficients':tested.model.eq_data[:,:5].tolist(),
        'interpretation':'Probe only: no claim of full-task physics success. Native tolerances quantify float32 GPU versus float64 CPU transfer, not task acceptance.'}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    curve_path=a.output.with_suffix('.curves.npz')
    np.savez_compressed(curve_path,time_s=np.asarray([r['time_s']for r in native_rows]),qpos_abs_error=np.asarray(qcurve),qvel_abs_error=np.asarray(vcurve),qpos_channel_names=np.asarray(qnames),qvel_channel_names=np.asarray(vnames),gpu_o6_polynomial_residual_rad=np.asarray(gpu_equalities),native_o6_polynomial_residual_rad=np.asarray(cpu_equalities),paired_reset_qpos_abs_error=reset_qerror,paired_reset_qvel_abs_error=reset_verror,paired_no_reset_qpos_abs_error=natural_qerror,paired_no_reset_qvel_abs_error=natural_verror)
    out['error_curves']={'path':str(curve_path),'sha256':sha(curve_path)}
    a.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out),flush=True)
    return 0 if all(checks.values())else 2


if __name__=='__main__':sys.exit(main())
