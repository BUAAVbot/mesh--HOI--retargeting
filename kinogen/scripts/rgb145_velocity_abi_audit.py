"""Compare legacy centered/interpolated and queried50Hz forward velocities.

No rollout, model modification, or trajectory resampling is performed.
"""
from pathlib import Path
import argparse,hashlib,json


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    import numpy as np
    from rgb145_sonic_controller import reference_joint_velocity
    z=np.load(a.reference,allow_pickle=False);names=list(map(str,z['dof_names']));cols=np.asarray([i for i,n in enumerate(names)if not n.startswith(('lh_','rh_','neck_'))])
    body_names=[names[i]for i in cols];q=np.asarray(z['qpos'][:,7+cols],float);times=np.asarray(z['time_s'],float)
    if len(body_names)!=28:raise ValueError('Expected28 body joints')
    query=np.arange(times[0],times[-1]+1e-9,.02)
    terminal_appended=bool(abs(query[-1]-times[-1])>1e-9)
    if terminal_appended:query=np.r_[query,times[-1]]
    future=np.minimum(query[:,None]+np.arange(10)[None]*.1,times[-1])
    old=reference_joint_velocity(times,q,future,'legacy_centered')
    new=reference_joint_velocity(times,q,future,'training_forward_50hz')
    diff=new-old;absolute=abs(diff);active=(abs(new)+abs(old))>1e-6
    warm=reference_joint_velocity(times,np.broadcast_to(q[0],q.shape),future,'training_forward_50hz')
    if np.max(abs(warm))>1e-12:raise AssertionError('Constant body_q warmup must naturally produce zero reference velocity')
    end_velocity=(q[-1]-np.asarray([np.interp(max(times[-1]-.02,times[0]),times,q[:,j])for j in range(28)]))/.02
    end_error=float(np.max(abs(new[-1]-end_velocity[None])))
    if end_error>1e-12:raise AssertionError('Explicit final backward-difference rule does not match')
    worst=np.unravel_index(int(np.argmax(absolute)),absolute.shape)
    query_rows=[]
    for i,t in enumerate(query):query_rows.append({'execution_time_s':float(t),'source_time_s':float(np.interp(t,times,z['source_time_s'])),'future_velocity_abs_difference_max_rad_s':float(absolute[i].max()),'future_velocity_abs_difference_p95_rad_s':float(np.quantile(absolute[i],.95))})
    a.output.parent.mkdir(parents=True,exist_ok=True);arrays=a.output.with_suffix('.arrays.npz')
    np.savez_compressed(arrays,execution_query_time_s=query,future_query_time_s=future,body_joint_names=np.asarray(body_names),legacy_centered_dq_rad_s=old,training_forward50hz_dq_rad_s=new,difference_rad_s=diff,source_frame_ids=z['frame_ids'],source_time_s=z['source_time_s'],source_reference_execution_time_s=times)
    training_source=Path('/data/yunhao.xia/GR00T-WholeBodyControl/gear_sonic/utils/motion_lib/torch_humanoid_batch.py')
    result={'schema':'rgb145.reference_velocity_abi_audit.v1','mode':'query_velocity_math_audit_not_physics','reference':{'path':str(a.reference.resolve()),'sha256':sha(a.reference)},'generator_sha256':sha(__file__),'native_wrapper_sha256':sha(Path(__file__).with_name('rgb145_sonic_controller.py')),'batch_wrapper_sha256':sha(Path(__file__).with_name('rgb145_batch_sonic.py')),'training_motion_source':{'path':str(training_source),'sha256':sha(training_source)}if training_source.exists()else None,'source_rule':'Training resamples FK poses to50Hz, then computes(q[t+1]-q[t])/.02. The source loader appends dof_vel[:, -2:-1].','adapter_rule':'Every future query: (q_interp(min(t+.02,end))-q_interp(t))/.02. At end explicitly use(q_end-q_interp(max(end-.02,start)))/.02. No resampled qf/body_q arrays are created.','legacy_rule':'np.gradient(q,original execution times), then linearly interpolate that cached derivative.','query_dt_s':.02,'query_count':len(query),'terminal_off_grid_query_appended':terminal_appended,'future_offsets_s':(np.arange(10)*.1).tolist(),'body_joints':body_names,'absolute_difference_max_rad_s':float(absolute.max()),'absolute_difference_p95_rad_s':float(np.quantile(absolute,.95)),'active_entries_absolute_difference_p95_rad_s':float(np.quantile(absolute[active],.95))if active.any()else 0.,'difference_rmse_rad_s':float(np.sqrt(np.mean(diff*diff))),'per_joint_absolute_difference_max_rad_s':dict(zip(body_names,np.max(absolute,axis=(0,1)).tolist())),'worst':{'execution_query_time_s':float(query[worst[0]]),'source_time_s':float(np.interp(query[worst[0]],times,z['source_time_s'])),'future_offset_s':float(worst[1]*.1),'future_query_time_s':float(future[worst[0],worst[1]]),'joint_name':body_names[worst[2]],'legacy_rad_s':float(old[worst]),'forward50hz_rad_s':float(new[worst])},'constant_pose_velocity_abs_max_rad_s':float(np.max(abs(warm))),'explicit_end_rule_abs_error_rad_s':end_error,'arrays':{'path':str(arrays.resolve()),'sha256':sha(arrays)},'not_causal_controller_failure_proof':True,'queries':query_rows}
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k!='queries'}),flush=True)


if __name__=='__main__':main()
