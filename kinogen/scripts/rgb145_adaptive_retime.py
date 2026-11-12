"""Preserve all source poses while slowing sharp transitions under discrete limits."""
from pathlib import Path
import argparse,hashlib,json


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--minimum-scale',type=float,default=2.);p.add_argument('--root-speed',type=float,default=.75);p.add_argument('--root-acceleration',type=float,default=2.5)
    p.add_argument('--angular-speed',type=float,default=1.5);p.add_argument('--angular-acceleration',type=float,default=6.)
    p.add_argument('--joint-speed',type=float,default=3.);p.add_argument('--joint-acceleration',type=float,default=20.);a=p.parse_args()
    import numpy as np
    from scipy.spatial.transform import Rotation
    from scipy.ndimage import maximum_filter1d
    if a.output.exists():raise FileExistsError(a.output)
    z=dict(np.load(a.input,allow_pickle=False));q=z['qpos'];source=z['source_time_s'].astype(float)
    R=Rotation.from_quat(np.roll(q[:,3:7],-1,axis=1));angular=(R[1:]*R[:-1].inv()).as_rotvec()
    cols=np.array([7+i for i,n in enumerate(z['dof_names'])if not str(n).startswith(('lh_','rh_','neck_'))])
    delta=[np.diff(q[:,:3],axis=0),angular,np.diff(q[:,cols],axis=0)]
    speeds=[a.root_speed,a.angular_speed,a.joint_speed];accels=[a.root_acceleration,a.angular_acceleration,a.joint_acceleration]
    dt=np.diff(source)*a.minimum_scale
    for k,change in enumerate(delta):
        length=np.max(abs(change),axis=1)if k==2 else np.linalg.norm(change,axis=1)
        dt=np.maximum(dt,length/speeds[k])
    iterations=0
    for iterations in range(200):
        factor=np.ones(len(dt))
        for k,change in enumerate(delta):
            velocity=change/dt[:,None];acc=np.diff(velocity,axis=0)/(.5*(dt[:-1]+dt[1:]))[:,None]
            peak=np.max(abs(acc),axis=1)if k==2 else np.linalg.norm(acc,axis=1)
            ratio=np.sqrt(np.maximum(1.,peak/accels[k]))
            factor[:-1]=np.maximum(factor[:-1],ratio);factor[1:]=np.maximum(factor[1:],ratio)
        if factor.max()<=1.00001:break
        # Widen a local slowdown rather than create another adjacent speed cliff.
        factor=maximum_filter1d(factor,size=3,mode='nearest');dt*=factor*1.00001
    else:raise RuntimeError('Time scaling did not converge')
    execution=np.r_[0.,np.cumsum(dt)];z['time_s']=execution
    z['source_to_execution_time_map']=np.c_[source,execution]
    peaks=[]
    for k,change in enumerate(delta):
        velocity=change/dt[:,None];acc=np.diff(velocity,axis=0)/(.5*(dt[:-1]+dt[1:]))[:,None]
        peaks.append({'speed':float(np.max(abs(velocity)))if k==2 else float(np.max(np.linalg.norm(velocity,axis=1))),'acceleration':float(np.max(abs(acc)))if k==2 else float(np.max(np.linalg.norm(acc,axis=1)))})
    np.savez_compressed(a.output,**z)
    out={'mode':'adaptive_execution_clock_only','frames':len(q),'source_last_pts_s':float(source[-1]),'execution_last_pts_s':float(execution[-1]),'minimum_scale':a.minimum_scale,'poses_unchanged':True,'frame_ids_unchanged':True,'raw_source_time_unchanged':True,'input_sha256':sha(a.input),'output_sha256':sha(a.output),'limits':{'root_speed_m_s':a.root_speed,'root_acceleration_m_s2':a.root_acceleration,'angular_speed_rad_s':a.angular_speed,'angular_acceleration_rad_s2':a.angular_acceleration,'body_joint_speed_rad_s':a.joint_speed,'body_joint_acceleration_rad_s2':a.joint_acceleration},'discrete_peaks':dict(zip(('root','angular','body_joints'),peaks)),'iterations':iterations+1,'not_a_physics_success_claim':True,'interpolation':'Limits describe segment-average velocities and centered finite-difference accelerations; actual simulation is independently validated.'}
    a.output.with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)


if __name__=='__main__':main()
