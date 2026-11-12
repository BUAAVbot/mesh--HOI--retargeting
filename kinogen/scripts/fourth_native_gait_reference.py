"""Plan a native Vita030 leg-template alternative; no physical-success claim.

The primary output changes only 12 leg joint references. Base pose and all
upper-body/finger references remain exact copies until the source-derived
table boundary and full upper-body IK are finalized. Recommended base height
and heading are separate arrays and must be re-targeted before simulation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    root=Path(__file__).resolve().parents[1]
    p.add_argument('--reference',type=Path,default=root/'outputs/fourth_full_sequence/reference_v8.npz')
    p.add_argument('--template',type=Path,default=root/'data/vita_demo_20261026/motions/030_0806__0806pm__scene6_7__000_action_000.npz')
    p.add_argument('--output',type=Path,default=root/'outputs/fourth_full_sequence/reference_native030_leg_plan.npz')
    p.add_argument('--positive-prefix-seconds',type=float,default=10.)
    p.add_argument('--max-cadence-scale',type=float,default=1.5)
    p.add_argument('--standing-reference',type=Path,default=root/'outputs/fourth_full_sequence/sonic_newft_default_stand_reference.npz')
    p.add_argument('--source-root-intent',action='store_true',help='Use interpolated original human root XY for speed/heading intent, not provisional robot IK root')
    a=p.parse_args()
    if os.name!='posix':p.error('Prepare references on the server; local is preview only')
    import numpy as np
    from scipy.ndimage import gaussian_filter1d
    from scipy.spatial.transform import Rotation
    from scipy.interpolate import CubicSpline
    from scipy.signal import find_peaks
    z=np.load(a.reference,allow_pickle=False);t=np.load(a.template,allow_pickle=False)
    times=z['time_s'];n=len(times);fps=float(t['fps']);dt=1/fps
    names=[str(s)for s in z['dof_names']];tnames=[str(s)for s in t['dof_names']]
    legnames=[name for name in names if name.startswith(('left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle'))]
    if len(legnames)!=12:raise ValueError('Expected the exact 12-leg DOF mapping')
    j=np.asarray([names.index(name)for name in legnames]);tj=np.asarray([tnames.index(name)for name in legnames])
    count=min(len(t['root_pos']),int(a.positive_prefix_seconds*fps)+1)
    q=gaussian_filter1d(np.asarray(t['dof_pos'][:count,tj],float),.7,axis=0,mode='nearest')
    pos=np.asarray(t['root_pos'][:count],float);tt=np.arange(count)/fps
    vel=gaussian_filter1d(np.gradient(pos,tt,axis=0),2.,axis=0,mode='nearest')
    speed=np.linalg.norm(vel[:,:2],axis=1)
    rot=Rotation.from_quat(t['root_rot_xyzw'][:count]).as_matrix()
    heading=np.unwrap(np.arctan2(rot[:,1,0],rot[:,0,0]))
    forward=np.column_stack([np.cos(heading),np.sin(heading)])
    left=np.column_stack([-np.sin(heading),np.cos(heading)])
    fwd=np.sum(vel[:,:2]*forward,axis=1);lateral=np.sum(vel[:,:2]*left,axis=1)
    lk=legnames.index('left_knee_joint');rk=legnames.index('right_knee_joint')
    signal=q[:,lk]-q[:,rk]
    peaks=find_peaks(signal,prominence=.03,distance=max(1,int(.35*fps)))[0]
    candidates=[]
    # Search a cycle by pose closure, movement, and direction consistency.
    # Every candidate lies in the native motion's independently tested prefix.
    for start in range(int(.5*fps),count-int(.6*fps)):
        for length in range(int(.6*fps),int(1.8*fps)+1):
            stop=start+length
            if stop>=count:continue
            span=q[start:stop+1]
            amplitude=max(float(np.ptp(span[:,lk])),float(np.ptp(span[:,rk])))
            path=float(np.sum(np.linalg.norm(np.diff(pos[start:stop+1,:2],axis=0),axis=1)))
            if amplitude<.12 or path<.10 or np.mean(speed[start:stop+1])<.15:continue
            closure=float(np.sqrt(np.mean((q[start]-q[stop])**2)))
            yaw_change=float(abs(heading[stop]-heading[start]))
            velocity_cv=float(np.std(speed[start:stop+1])/max(np.mean(speed[start:stop+1]),1e-6))
            endpoint_dq=float(np.sqrt(np.mean(((q[start+1]-q[start])-(q[stop]-q[stop-1]))**2)))
            # Adjacent same-side knee peaks are a mild preference; pose/velocity
            # continuity remains the primary criterion.
            near_peak=min(abs(peaks-start),default=30)+min(abs(peaks-stop),default=30)
            score=closure+1.5*endpoint_dq+.08*yaw_change+.04*velocity_cv+.0003*near_peak
            candidates.append({'start':start,'stop':stop,'period_s':length/fps,'score':score,'closure_rms_rad':closure,'endpoint_dq_difference_rad':endpoint_dq,'heading_change_rad':yaw_change,'path_m':path,'knee_amplitude_rad':amplitude,'speed_mean_m_s':float(np.mean(speed[start:stop+1]))})
    if not candidates:raise ValueError('No moving periodic leg candidate in the validated native prefix')
    candidates.sort(key=lambda v:v['score']);chosen=candidates[0];start,stop=chosen['start'],chosen['stop']
    period=chosen['period_s'];u=np.linspace(0,1,stop-start+1)
    cycle=q[start:stop+1].copy();cycle[0]=cycle[-1]=.5*(cycle[0]+cycle[-1])
    spline=CubicSpline(u,cycle,bc_type='periodic',axis=0)
    hcycle=pos[start:stop+1,2].copy();hcycle[0]=hcycle[-1]=.5*(hcycle[0]+hcycle[-1])
    hspline=CubicSpline(u,hcycle,bc_type='periodic')
    # Stand blend uses a real native-pose sample from the positive prefix.
    standing_frame=int(np.argmin(speed[:min(count,int(2*fps))]))
    standing_leg=np.asarray(t['dof_pos'][standing_frame,tj],float)
    standing_height=float(t['root_pos'][standing_frame,2])
    standing_provenance={'kind':'lowest_speed_native_sample','frame':standing_frame,'sample_speed_m_s':float(speed[standing_frame])}
    if a.standing_reference.exists():
        stand=np.load(a.standing_reference,allow_pickle=False);snames=[str(s)for s in stand['dof_names']]
        standing_leg=stand['qpos'][0,7+np.asarray([snames.index(name)for name in legnames])].astype(float)
        standing_height=float(stand['qpos'][0,2])-float(z['physics_floor_z_m'])
        standing_provenance={'kind':'independently_positive_default_stand_reference','path':str(a.standing_reference),'sha256':sha(a.standing_reference),'root_height_m':standing_height}
    phase_grid=np.linspace(0,1,200,endpoint=False)
    initial_phase=float(phase_grid[np.argmin(np.mean((spline(phase_grid)-standing_leg[None,:])**2,axis=1))])

    rootpos=z['qpos'][:,:3];intent_root=rootpos.copy()
    if a.source_root_intent:
        original=z['raw_source_root_position_m'];valid=np.flatnonzero(np.isfinite(original).all(axis=1)&z['human_observed_mask'])
        for dim in range(2):intent_root[:,dim]=np.interp(times,times[valid],original[valid,dim])
        intent_root[:,:2]=gaussian_filter1d(intent_root[:,:2],4.,axis=0,mode='nearest')
    root_velocity=gaussian_filter1d(np.gradient(intent_root,times,axis=0),4.,axis=0,mode='nearest')
    desired_speed=np.linalg.norm(root_velocity[:,:2],axis=1)
    x=np.clip((desired_speed-.03)/(.15-.03),0,1);blend=x*x*(3-2*x)
    blend=np.clip(gaussian_filter1d(blend,4.,mode='nearest'),0,1)
    stride=chosen['path_m']
    rate=np.clip(desired_speed/max(stride,1e-6),0,a.max_cadence_scale/period)
    rate=np.where(blend>.03,np.maximum(rate,.35/period),0.)
    rate=gaussian_filter1d(rate,4.,mode='nearest')
    phase=np.zeros(n);phase[0]=initial_phase
    for frame in range(1,n):phase[frame]=(phase[frame-1]+.5*(rate[frame-1]+rate[frame])*(times[frame]-times[frame-1]))%1.
    raw_template=spline(phase);legq=(1-blend[:,None])*standing_leg[None,:]+blend[:,None]*raw_template
    legq=gaussian_filter1d(legq,.7,axis=0,mode='nearest')
    before_training_limits=legq.copy()
    for k,name in enumerate(legnames):
        if 'ankle_pitch'in name:legq[:,k]=np.clip(legq[:,k],-.751,.414)
    training_limit_correction=legq-before_training_limits
    unbounded=legq.copy();caps=np.asarray([6. if 'knee'in name else 4. for name in legnames])
    for frame in range(1,n):
        legq[frame]=np.clip(legq[frame],legq[frame-1]-caps*(times[frame]-times[frame-1]),legq[frame-1]+caps*(times[frame]-times[frame-1]))
    rootrot=Rotation.from_quat(z['qpos'][:,[4,5,6,3]]).as_matrix()
    source_yaw=np.unwrap(np.arctan2(rootrot[:,1,0],rootrot[:,0,0]))
    body_fwd=np.column_stack([np.cos(source_yaw),np.sin(source_yaw)])
    body_left=np.column_stack([-np.sin(source_yaw),np.cos(source_yaw)])
    desired_body_velocity=np.column_stack([np.sum(root_velocity[:,:2]*body_fwd,axis=1),np.sum(root_velocity[:,:2]*body_left,axis=1)])
    native_body_velocity=np.asarray([np.mean(fwd[start:stop+1]),np.mean(lateral[start:stop+1])])
    native_direction=native_body_velocity/max(np.linalg.norm(native_body_velocity),1e-9)
    alignment=np.sum(desired_body_velocity*native_direction[None,:],axis=1)/np.maximum(desired_speed,1e-9)
    mismatch=(desired_speed>.10)&(alignment<np.cos(np.deg2rad(30.)))
    speed_mismatch=desired_speed>stride*a.max_cadence_scale/period+.02
    velocity_heading=np.arctan2(root_velocity[:,1],root_velocity[:,0])-np.arctan2(native_direction[1],native_direction[0])
    delta=(velocity_heading-source_yaw+np.pi)%(2*np.pi)-np.pi
    recommended_yaw=source_yaw+blend*delta
    recommended_yaw=gaussian_filter1d(np.unwrap(recommended_yaw),6.,mode='nearest')
    for frame in range(1,n):recommended_yaw[frame]=np.clip(recommended_yaw[frame],recommended_yaw[frame-1]-1.2*(times[frame]-times[frame-1]),recommended_yaw[frame-1]+1.2*(times[frame]-times[frame-1]))
    floor=float(z['physics_floor_z_m']);recommended_z=floor+(1-blend)*standing_height+blend*hspline(phase)
    qpos=z['qpos'].copy();original_legs=qpos[:,7+j].copy();qpos[:,7+j]=legq
    payload={key:z[key]for key in z.files};payload.update(qpos=qpos,
      source_reference_leg_qpos_rad=original_legs,native_template_leg_qpos_rad=raw_template,
      native_gait_replacement_leg_qpos_rad=legq,native_gait_leg_names=np.asarray(legnames),
      native_gait_phase=phase,native_gait_rate_hz=rate,native_gait_walking_blend=blend,
      native_gait_source_speed_m_s=desired_speed,native_gait_source_body_velocity_m_s=desired_body_velocity,
      native_gait_intent_root_position_m=intent_root,native_gait_intent_from_original_source=np.asarray(a.source_root_intent),
      native_gait_template_body_velocity_m_s=native_body_velocity,native_gait_direction_alignment=alignment,
      native_gait_heading_mismatch_mask=mismatch,native_gait_recommended_root_z_m=recommended_z,
      native_gait_speed_saturation_mismatch_mask=speed_mismatch,
      native_gait_recommended_root_yaw_rad=recommended_yaw,
      native_gait_velocity_limit_correction_rad=legq-unbounded,
      native_gait_training_joint_limit_correction_rad=training_limit_correction,
      native_gait_template_sha256=np.asarray(sha(a.template)),native_gait_parent_reference_sha256=np.asarray(sha(a.reference)),
      native_gait_cycle_source_frame_ids=np.asarray([start,stop]),native_gait_cycle_period_s=np.asarray(period),
      native_gait_standing_leg_qpos_rad=standing_leg,native_gait_standing_provenance=np.asarray(json.dumps(standing_provenance)),
      native_gait_standing_source_frame=np.asarray(-1 if standing_provenance['kind']=='independently_positive_default_stand_reference'else standing_frame),native_gait_physical_ready=np.asarray(False),
      native_gait_plan_note=np.asarray('Pure 12-leg alternative; base pose and upperbody/finger joints preserved. Recommended native height/yaw not applied. Final source-derived table/base path and upperbody wrist/palm/waist IK must be reconciled before physical rollout. Cropped/periodized/blended motion requires new physics validation.'))
    stale_cache=[key for key in list(payload)if '_fk_'in key or key in ['foot_sole_clearance_m','ik_error_m']]
    for key in stale_cache:payload['parent_cached_'+key]=payload.pop(key)
    a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,**payload)
    nonlegs=np.asarray([i for i in range(52)if i not in j])
    report={'schema':'fourth_native_gait_leg_plan.v1','output':str(a.output),'sha256':sha(a.output),'frames':n,'physics_run':False,'physical_ready':False,
      'source_script_sha256':sha(Path(__file__)),
      'parent_reference':str(a.reference),'parent_reference_sha256':sha(a.reference),'template':str(a.template),'template_sha256':sha(a.template),
      'native_positive_prefix_used_s':a.positive_prefix_seconds,'selected_cycle':chosen,'top_cycle_candidates':candidates[:10],
      'template_source_leg_names':legnames,'fallback_low_speed_native_frame':standing_frame,'fallback_low_speed_native_speed_m_s':float(speed[standing_frame]),
      'standing_blend_provenance':standing_provenance,
      'native_cycle_body_velocity_m_s':native_body_velocity.tolist(),'native_cycle_root_height_range_m':[float(hcycle.min()),float(hcycle.max())],
      'root_pose_preserved':bool(np.array_equal(qpos[:,:7],z['qpos'][:,:7])),
      'parent_fk_arrays_relabelled':stale_cache,
      'upperbody_and_fingers_preserved':bool(np.array_equal(qpos[:,7+nonlegs],z['qpos'][:,7+nonlegs])),
      'moving_frames':int(np.sum(desired_speed>.10)),'heading_mismatch_moving_frames':int(mismatch.sum()),
      'root_intent_from_original_source':a.source_root_intent,
      'speed_saturation_mismatch_frames':int(speed_mismatch.sum()),'max_template_reference_speed_m_s':stride*a.max_cadence_scale/period,
      'desired_speed_max_m_s':float(desired_speed.max()),'cadence_rate_max_hz':float(rate.max()),'cadence_scale_cap':a.max_cadence_scale,
      'replacement_leg_velocity_peak_rad_s':np.max(np.abs(np.diff(legq,axis=0)/np.diff(times)[:,None]),axis=0).tolist(),
      'training_startup_ankle_pitch_limits_rad':[-.751,.414],
      'training_limit_correction_max_rad':float(np.max(np.abs(training_limit_correction))),
      'leg_replacement_rms_rad':float(np.sqrt(np.mean((legq-original_legs)**2))),
      'root_height_retarget_required_max_m':float(np.max(np.abs(recommended_z-rootpos[:,2]))),
      'source_root_xy_deviation_m_max':float(np.nanmax(np.linalg.norm(rootpos[:,:2]-z['raw_source_root_position_m'][:,:2],axis=1))),
      'notes':['A full native030 prefix passed earlier dynamics; this cropped/periodized template has NOT been physics-validated.',
        'Source velocity and heading need not align with this mostly single-direction native gait; mismatch is explicit.',
        'Root XY follows the input provisional reference; the source-derived true table boundary is pending.',
        'Primary qpos preserves base and upperbody exactly. Recommended height/yaw require subsequent upperbody contact IK; do not directly claim a physically consistent full-scene trajectory.']}
    a.output.with_suffix('.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
