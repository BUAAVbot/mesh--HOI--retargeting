"""Audit an old verified physical demonstration and prepare separately labeled reset seeds.

No old artifact is changed. Reusing a valid physical state does not establish a
new PHC expert success, and actual velocities are never rescaled to a new clock.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):value.update(chunk)
    return value.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/data/KinoGen/outputs/sonic_expert_20261028'))
    a=p.parse_args()
    import mujoco as mj,numpy as np
    from two_sitting_physics import independent_replay,interp
    old_dir=Path('/data/KinoGen/outputs/two_sitting_pipeline/physics_selected450_candidate156')
    old_model=old_dir/'model.mjb';new_model=a.root/'calibration/two_vita_expert.mjb'
    old_trace=old_dir/'trace.npz';old_report_path=old_dir/'report.json'
    reference=a.root/'references/phc.npz'
    out=a.root/'prior_physical_state_pool';out.mkdir(exist_ok=True)
    old=mj.MjModel.from_binary_path(str(old_model));new=mj.MjModel.from_binary_path(str(new_model))
    diffs=[];arrays_checked=0;scalars_checked=0
    for name in dir(old):
        if name.startswith('_'):continue
        x,y=getattr(old,name),getattr(new,name)
        if isinstance(x,np.ndarray)and isinstance(y,np.ndarray):
            arrays_checked+=1
            equal=(np.array_equal(x,y,equal_nan=True)if x.dtype.kind in 'fc'else np.array_equal(x,y))if x.shape==y.shape else False
            if not equal:
                diffs.append({'field':name,'old_shape':list(x.shape),'new_shape':list(y.shape),
                    'max_abs_difference':float(np.max(abs(x.astype(float)-y.astype(float))))if x.shape==y.shape and x.size and x.dtype.kind in 'fiu'else None})
        elif isinstance(x,(int,float,bool,bytes,str))and isinstance(y,type(x)):
            scalars_checked+=1
            if x!=y:diffs.append({'field':name,'old':str(x)[:200],'new':str(y)[:200]})
    option_diffs=[]
    for name in dir(old.opt):
        if name.startswith('_'):continue
        x,y=getattr(old.opt,name),getattr(new.opt,name)
        if isinstance(x,np.ndarray):
            if not np.array_equal(x,y):option_diffs.append(name)
        elif isinstance(x,(int,float,bool))and x!=y:option_diffs.append(name)
    dimensions={name:[getattr(old,name),getattr(new,name)]for name in ('nq','nv','nu','nbody','ngeom','nmesh','nmeshvert','neq')}
    order={kind.name:all(mj.mj_id2name(old,kind,i)==mj.mj_id2name(new,kind,i)for i in range(count))
           for kind,count in ((mj.mjtObj.mjOBJ_JOINT,old.njnt),(mj.mjtObj.mjOBJ_ACTUATOR,old.nu),
                              (mj.mjtObj.mjOBJ_BODY,old.nbody),(mj.mjtObj.mjOBJ_GEOM,old.ngeom))}
    report=json.loads(old_report_path.read_text());z=dict(np.load(old_trace,allow_pickle=False));ref=dict(np.load(reference,allow_pickle=False))
    model_match=not diffs and not option_diffs and all(order.values())
    receipt={'schema':'sonic_expert.prior_physical_state_pool.v1','old_model':str(old_model),'old_model_sha256':sha(old_model),
        'new_model':str(new_model),'new_model_sha256':sha(new_model),'model_arrays_checked':arrays_checked,
        'model_scalars_checked':scalars_checked,'model_parameter_differences':diffs,'physics_option_differences':option_diffs,
        'model_dimensions':dimensions,'names_and_order_identical':order,'model_parameters_identical':model_match,
        'old_trace':str(old_trace),'old_trace_sha256':sha(old_trace),'old_report':str(old_report_path),'old_report_sha256':sha(old_report_path),
        'current_reference':str(reference),'current_reference_sha256':sha(reference),
        'old_functional_pass':report['functional_task_success'],'old_strict_source_pass':report['success'],
        'old_independent_replay':report['independent_replay'],
        'source_label':'Previously selected candidate156 from old custom-IK + SONIC/PPO experiment; NOT a new PHC expert result.',
        'coordinate_contract':'Same named dual-Vita118/116/104 Z-up metre world and wxyz roots; no alignment transform, recenter, rescale or q/v modification.',
        'new_phc_expert_success_claim':False,'velocity_rescaled':False}
    if not model_match:
        receipt['status']='blocked_physical_model_parameters_differ'
        (out/'report.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'status':receipt['status'],'diffs':diffs}));return
    spec=mj.mjtState(int(z['state_spec']))
    cross=independent_replay(new_model,z['initial_state'],spec,z['ctrl'],z['qpos'],z['qvel'])
    receipt['old_controls_replayed_in_current_model']=cross
    if not cross['passed']:raise RuntimeError('Cross-model replay failed despite matching fields')
    time=z['time'];source=z['source_time_s'];q=z['qpos'];v=z['qvel'];control=z['ctrl']
    assert q.shape==(len(time),118)and v.shape==(len(time),116)and control.shape==(len(time)-1,104)
    assert np.all(np.diff(time)>0)and np.all(np.diff(source)>=-1e-12)
    assert list(map(str,z['joint_names']))==list(map(str,ref['dof_names']))
    desired=np.asarray([interp(ref,t)for t in source])
    roots=np.asarray(z['robot_root_qadr'],int);velroots=np.asarray(z['robot_root_vadr'],int)
    qa=np.asarray(z['joint_qadr'],int);bodycols=np.asarray([i for i,n in enumerate(z['joint_names'])if not str(n).startswith(('lh_','rh_','neck_'))])
    body_aids=np.asarray(z['actuator_ids'],int)[:,bodycols].reshape(-1)
    rootxy=np.stack([np.linalg.norm(q[:,rq:rq+2]-desired[:,actor,:2],axis=-1)for actor,rq in enumerate(roots)],1)
    rootz=np.stack([abs(q[:,rq+2]-desired[:,actor,2])for actor,rq in enumerate(roots)],1)
    orientation=np.stack([2*np.arccos(np.clip(abs(np.sum(q[:,rq+3:rq+7]*desired[:,actor,3:7],axis=-1)),0,1))for actor,rq in enumerate(roots)],1)
    joints=np.stack([np.sqrt(np.mean((q[:,qa[actor,bodycols]]-desired[:,actor,7+bodycols])**2,axis=-1))for actor in range(2)],1)
    heights=q[:,roots+2];tilt=np.stack([np.arccos(np.clip(1-2*(q[:,rq+4]**2+q[:,rq+5]**2),-1,1))for rq in roots],1)
    linear=np.stack([np.linalg.norm(v[:,rv:rv+3],axis=-1)for rv in velroots],1)
    angular=np.stack([np.linalg.norm(v[:,rv+3:rv+6],axis=-1)for rv in velroots],1)
    pen=np.zeros(len(control));np.maximum.at(pen,z['contact_interval_index'],np.maximum(-z['contact_signed_distance_m'],0))
    seatforce=z['seat_support_force_N'];seatcontact=z['seat_contact']
    weight=np.asarray(report['robot_weight_N']);future_steps=int(round(.2/float(new.opt.timestep)))
    candidates=np.arange(0,len(time)-future_steps,4,dtype=int)
    sourceidx=np.clip(np.searchsorted(ref['source_time_s'],source[candidates],side='right')-1,0,len(ref['time_s'])-1)
    expected=ref['seat_support_expected'][sourceidx].astype(bool)
    quality=[];eligible=[]
    for candidate,index in enumerate(candidates):
        end=index+future_steps
        stable=bool(np.isfinite(q[index:end+1]).all()and np.isfinite(v[index:end+1]).all()and
                    heights[index:end+1].min()>.45 and tilt[index:end+1].max()<.65 and pen[index:end].max()<.025)
        force=seatforce[index:end].mean(0);fraction=seatcontact[index:end].mean(0)
        seat_ok=bool(np.all(~expected[candidate]|((force>=.1*weight)&(fraction>=.5))))
        path_ok=bool(rootxy[index].max()<=.45 and rootz[index].max()<=.20 and orientation[index].max()<=.65 and joints[index].max()<=.6)
        quiet=bool(np.all(~expected[candidate]|((linear[index:end+1].max(0)<.20)&(angular[index:end+1].max(0)<.4))))
        passed=stable and seat_ok and path_ok and quiet
        quality.append({'state_index':int(index),'old_execution_time_s':float(time[index]),'source_phase_s':float(source[index]),
            'future_stable':stable,'source_expected_seat':expected[candidate].tolist(),'future_seat_match':seat_ok,
            'future_seated_motion_quiet':quiet,'phc_path_gate':path_ok,'eligible':passed,
            'root_xy_max_m':float(rootxy[index].max()),'root_z_max_m':float(rootz[index].max()),
            'root_orientation_max_rad':float(orientation[index].max()),'body_joint_rms_max_rad':float(joints[index].max()),
            'future_penetration_max_m':float(pen[index:end].max()),'future_seat_force_mean_N':force.tolist(),
            'future_seat_contact_fraction':fraction.tolist()})
        eligible.append(passed)
    selected=candidates[np.asarray(eligible)]
    # Zero new-expert history is deliberate. These are physical-state seeds, not transplanted recurrent policy states.
    values={'q':q[selected].copy(),'v':v[selected].copy(),'ctrl':control[np.minimum(selected,len(control)-1)].copy(),
        'phase':source[selected].copy(),'previous':np.zeros((len(selected),56)),
        'body_target':control[np.maximum(selected-1,0)][:,body_aids].copy(),
        'captured_iteration':np.full(len(selected),-1,dtype=np.int64),
        'root_xy_m':rootxy[selected].max(-1),'penetration_m':np.asarray([quality[i]['future_penetration_max_m']for i,b in enumerate(eligible)if b]),
        'old_execution_time_s':time[selected].copy(),'source_phase_s':source[selected].copy(),
        'old_state_index':selected.copy(),'is_old_physical_prior_seed':np.ones(len(selected),bool),
        'is_new_phc_expert_success':np.zeros(len(selected),bool)}
    files=[]
    for scale,name in ((1.,'source1x'),(1.5,'source1p5x'),(2.,'source2x')):
        values['phase']=source[selected]*scale
        output=out/f'candidate156_{name}_state_pool.npz';np.savez_compressed(output,**values)
        files.append({'path':str(output),'sha256':sha(output),'states':len(selected),'phase_time_scale':scale,
            'actual_q_v_ctrl_unchanged':True,'velocity_scaled':False})
    (out/'per_candidate_quality.json').write_text(json.dumps(quality,indent=2)+'\n')
    receipt.update(status='old_physical_prior_seeds_ready_not_new_expert_success',
        old_duration_s=float(time[-1]),old_source_duration_s=float(source[-1]),old_execution_reference_duration_s=report['execution_reference_duration_seconds'],
        all_states=len(time),candidates_50hz=len(candidates),eligible_seeds=len(selected),
        clock_policy='For current source1x phase=old source_time_s; source1.5x/source2x phase=old source_time_s*scale. Actual old q/v/controls unchanged. Old terminal hold maps to reference-end phase.',
        policy_history_policy='previous action is explicitly reset to zero; physical delayed body_target uses the preceding actual applied motor packet. SONIC history must be reset by the consumer. No claim that current expert reproduces old controller history.',
        future_validation={'seconds':.2,'native_physics_steps':future_steps,'cross_model_replay_exact':True,
            'no_fall_root_height_min_m':.45,'max_global_tilt_rad':.65,'max_penetration_m':.025,
            'expected_seat_mean_load_min_body_weight_fraction':.10,'expected_seat_contact_fraction_min':.5,
            'seated_linear_speed_max_m_s':.20,'seated_angular_speed_max_rad_s':.4},
        current_phc_gates={'root_xy_max_m':.45,'root_z_max_m':.20,'root_orientation_max_rad':.65,'body_joint_rms_max_rad':.6},
        path_difference_over_all_old_states={'root_xy_rms_m':float(np.sqrt(np.mean(rootxy**2))),'root_xy_max_m':rootxy.max(0).tolist(),
            'root_z_max_m':rootz.max(0).tolist(),'root_orientation_max_rad':orientation.max(0).tolist(),
            'body_joint_rms_median_rad':np.median(joints,axis=0).tolist()},
        rejection_counts={'future_stability':sum(not item['future_stable']for item in quality),
            'phc_path_gate':sum(not item['phc_path_gate']for item in quality),
            'seat_load_or_contact':sum(not item['future_seat_match']for item in quality),
            'seated_motion_not_quiet':sum(not item['future_seated_motion_quiet']for item in quality)},
        phase_coverage_source_s=[float(source[selected].min()),float(source[selected].max())]if len(selected)else[],
        files=files,audit_source_sha256=sha(__file__))
    (out/'report.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'status':receipt['status'],'model_parameters_identical':model_match,
        'cross_model_replay':cross,'eligible_seeds':len(selected),'phase_coverage':receipt['phase_coverage_source_s'],
        'phc_xy_rms_m':receipt['path_difference_over_all_old_states']['root_xy_rms_m'],
        'rejection_counts':receipt['rejection_counts']}),flush=True)


if __name__=='__main__':main()
