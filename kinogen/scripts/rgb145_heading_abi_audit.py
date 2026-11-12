"""Quantify legacy Euler heading versus training Z-twist future6D inputs.

This is a nominal reference/input ABI audit. It does not simulate motion or
establish that heading mismatch caused a particular tracking failure.
"""
from pathlib import Path
import argparse,hashlib,json


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    import numpy as np
    from rgb145_sonic_controller import sampled_heading_matrix,quaternion_matrix
    z=np.load(a.reference,allow_pickle=False);q=np.asarray(z['qpos'][:,3:7],float).copy();times=np.asarray(z['time_s'],float)
    for k in range(1,len(q)):
        if np.dot(q[k],q[k-1])<0:q[k]*=-1
    q/=np.linalg.norm(q,axis=1)[:,None]
    diffs=[];old_inputs=[];new_inputs=[];matrix_errors=[];native_training_errors=[];rows=[]
    def yaw_matrix(theta):
        c,s=np.cos(theta),np.sin(theta);return np.array([[c,-s,0],[s,c,0],[0,0,1.]])
    for i,quat in enumerate(q):
        r=quaternion_matrix(quat)
        old=sampled_heading_matrix(r,'euler_yaw');new=sampled_heading_matrix(r,'training_z_twist')
        # Literal training get_heading_q: zero x/y and normalize remaining w/z.
        training_quat=np.array([quat[0],0.,0.,quat[3]]);training_quat/=max(np.linalg.norm(training_quat),1e-12)
        training=quaternion_matrix(training_quat)
        native_training_errors.append(float(np.max(abs(new-training))))
        # Independent check of the float32 matrix formula used by BatchSonic.
        r32=r.astype(np.float32);theta=np.arctan2(r32[1,0]-r32[0,1],r32[0,0]+r32[1,1])
        batch_matrix=yaw_matrix(float(theta));matrix_errors.append(float(np.max(abs(batch_matrix-training))))
        old_yaw=np.arctan2(old[1,0],old[0,0]);new_yaw=np.arctan2(new[1,0],new[0,0])
        difference=float(np.arctan2(np.sin(new_yaw-old_yaw),np.cos(new_yaw-old_yaw)));diffs.append(difference)
        future_times=np.minimum(times[i]+np.arange(10)*.1,times[-1])
        future=np.stack([np.interp(future_times,times,q[:,j])for j in range(4)],axis=-1)
        future/=np.linalg.norm(future,axis=1)[:,None]
        old6=np.asarray([(old.T@quaternion_matrix(f))[:,:2].reshape(6)for f in future])
        new6=np.asarray([(new.T@quaternion_matrix(f))[:,:2].reshape(6)for f in future])
        old_inputs.append(old6);new_inputs.append(new6);err=old6-new6
        rows.append({'frame_id':int(z['frame_ids'][i]),'source_time_s':float(z['source_time_s'][i]),'execution_time_s':float(times[i]),'twist_minus_euler_rad':difference,'twist_minus_euler_degrees':float(np.degrees(difference)),'future6d_element_abs_max':float(np.max(abs(err))),'future6d_rmse':float(np.sqrt(np.mean(err*err))),'future6d_frame_l2_max':float(np.max(np.linalg.norm(err,axis=1)))})
    difference=np.asarray(diffs);old_inputs=np.asarray(old_inputs);new_inputs=np.asarray(new_inputs)
    a.output.parent.mkdir(parents=True,exist_ok=True);array_path=a.output.with_suffix('.arrays.npz')
    np.savez_compressed(array_path,frame_ids=z['frame_ids'],source_time_s=z['source_time_s'],execution_time_s=times,twist_minus_euler_rad=difference,legacy_future6d=old_inputs,training_future6d=new_inputs,future6d_difference=old_inputs-new_inputs,batch_matrix_vs_training_abs_error=np.asarray(matrix_errors),native_matrix_vs_training_abs_error=np.asarray(native_training_errors))
    worst=int(np.argmax(abs(difference)))
    result={'schema':'rgb145.heading_abi_audit.v1','mode':'nominal_reference_input_audit_not_physics','reference':{'path':str(a.reference.resolve()),'sha256':sha(a.reference)},'generator_sha256':sha(__file__),'native_wrapper_sha256':sha(Path(__file__).with_name('rgb145_sonic_controller.py')),'batch_wrapper_sha256':sha(Path(__file__).with_name('rgb145_batch_sonic.py')),'frames':len(q),'nominal_actual_orientation_assumption':'Robot sampled orientation is assumed equal to each reference root orientation; this is not a measured live-tracking error.','training_heading':'normalize((w,0,0,z)); quaternion sign does not change resulting rotation','legacy_heading':'atan2(R10,R00), projected-X/Euler yaw','batch_formula':'atan2(R10-R01,R00+R11) is the full Z-twist angle; quaternion half-angle is obtained by dividing it by2','native_training_matrix_abs_error_max':max(native_training_errors),'batch_float32_training_matrix_abs_error_max':max(matrix_errors),'math_checks_passed':bool(max(native_training_errors)<1e-10 and max(matrix_errors)<2e-6),'heading_difference_abs_max_degrees':float(np.max(abs(np.degrees(difference)))),'heading_difference_abs_median_degrees':float(np.median(abs(np.degrees(difference)))),'heading_difference_abs_p95_degrees':float(np.quantile(abs(np.degrees(difference)),.95)),'worst_heading_frame':rows[worst],'future6d_element_abs_error_max':float(np.max(abs(old_inputs-new_inputs))),'future6d_overall_rmse':float(np.sqrt(np.mean((old_inputs-new_inputs)**2))),'arrays':{'path':str(array_path.resolve()),'sha256':sha(array_path)},'limitations':['Does not establish the cause of a controller failure; compare native tracking with the same reference and model before/after heading-only change.','Heading is undefined at the upside-down Z-twist singularity; regular humanoid poses are assumed.'],'rows':rows}
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k!='rows'}),flush=True)
    if not result['math_checks_passed']:raise SystemExit(2)


if __name__=='__main__':main()
