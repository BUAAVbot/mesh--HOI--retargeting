#!/usr/bin/env python3
"""CPU known-rigid-cuboid sequence fit, static-calibrated cameras and censored occlusion."""
from pathlib import Path
import os
import json
import time
ROOT=Path(__file__).resolve().parents[1]


def image_residual(p, corners, P, observations, directions, split='train', precise=False):
    import numpy as np
    import cv2
    from reconstruct_box import project_pose, point_polygon_distances, sample_map, resample_polygon
    xy,_,_=project_pose(p,corners,P)
    result=[]
    for v,o in enumerate(observations):
        support=(xy[v]@directions.T).max(axis=0)
        residual=support-o['supports']
        # If a visible mask extremum is clipped by a hand, the observed support
        # is a lower bound, NOT a noisy measurement of the full box edge.
        censored=o['support_weights']<.8
        residual[censored]=np.minimum(residual[censored],0)
        sel=(np.arange(len(directions))%4==0) if split=='holdout' else (np.arange(len(directions))%4!=0)
        if split=='all':sel=np.ones(len(directions),bool)
        result.extend(residual[sel]/3)
        if precise:
            indices=cv2.convexHull(xy[v].astype(np.float32),returnPoints=False).reshape(-1)
            hull=xy[v,indices]  # Keep original float64 coordinates; topology only uses OpenCV.
        else:
            hull=cv2.convexHull(xy[v].astype(np.float32)).reshape(-1,2).astype(float)
        distances=point_polygon_distances(o['contour'],hull)
        # Remove artificial boundaries induced by hand/watermark cuts entirely.
        w=(o['contour_weights']>.8).astype(float)
        sel=(np.arange(len(distances))%4==0) if split=='holdout' else (np.arange(len(distances))%4!=0)
        if split=='all':sel=np.ones(len(distances),bool)
        result.extend(distances[sel]*w[sel]*.35/3)
        predicted=resample_polygon(hull,96)
        pred_distance=sample_map(o['distance'].astype(float,copy=False) if precise else o['distance'],predicted)
        excluded=o.get('excluded_float',o['excluded']) if precise else o['excluded']
        pred_weight=np.clip(1-sample_map(excluded.astype(float,copy=False),predicted,outside=1),0,1)
        sel=(np.arange(96)%4==0) if split=='holdout' else (np.arange(96)%4!=0)
        if split=='all':sel=np.ones(96,bool)
        result.extend(pred_distance[sel]*pred_weight[sel]*.35/3)
    return np.asarray(result)


def main():
    os.environ.setdefault('OMP_NUM_THREADS','4')
    import numpy as np
    import cv2
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    from reconstruct_box import box_corners, observation, view_metrics, independent_view_centres, digest, make_preview, project_pose
    cv2.setNumThreads(4)
    started=time.time()
    source=ROOT/'outputs/user_reconstruction';out=ROOT/'outputs/user_reconstruction_v2';out.mkdir(parents=True,exist_ok=True)
    camera=out/'camera_refinement.npz';cam=np.load(camera);P=cam['P_refined'];g=np.load(source/'geometry.npz')
    old=np.load(source/'box_trajectory.npz');seg=np.load(source/'segmentation.npz')
    if digest(source/'segmentation.npz')!='9a7698abb2297e25c3710fd954dd847e4c9c3017d592f8d9048c8ac31b1c0902':raise RuntimeError('Require final v3 masks')
    x0=np.column_stack((old['center_xyz_m'],old['rotation_vector']));n=len(x0);timestamps=old['timestamps'];size=old['box_size_xyz_m'];corners=box_corners(size)
    a=np.arange(64)*2*np.pi/64;directions=np.column_stack((np.cos(a),np.sin(a)))
    observations=[]
    masks,observation_valid,robots=seg['box'],seg['observation_valid'],seg['robot']
    for f in range(n):
        observations.append([observation(masks[f,v],observation_valid[f,v],robots[f,v],directions) for v in range(2)])
    if any(o is None for pair in observations for o in pair):raise RuntimeError('Missing actual observed silhouette')
    # Censored likelihood initial per-frame solve, followed by a single sequence
    # least-squares solve with a block sparse Jacobian and acceleration prior.
    initial=[]
    for f in range(n):
        fun=lambda p:image_residual(p,corners,P,observations[f],directions)
        fit=least_squares(fun,x0[f],loss='soft_l1',f_scale=2,max_nfev=65,diff_step=1e-3,x_scale=[.1,.1,.1,.3,.3,.3],bounds=(x0[f]-[.2,.2,.2,.35,.35,.35],x0[f]+[.2,.2,.2,.35,.35,.35]))
        initial.append(fit.x)
        if f%30==0:print('box v2 initial',f,float(fit.cost),flush=True)
    initial=np.array(initial);block=len(image_residual(initial[0],corners,P,observations[0],directions))
    acceleration_scale=np.array([.006,.006,.006,.025,.025,.025])
    def residual(flat):
        poses=flat.reshape(n,6)
        data=np.concatenate([image_residual(poses[f],corners,P,observations[f],directions) for f in range(n)])
        acc=(poses[2:]-2*poses[1:-1]+poses[:-2])/acceleration_scale
        return np.r_[data,acc.reshape(-1)]
    # Absolute central differences avoid OpenCV float32 hull quantization when
    # scipy's relative step becomes tiny at near-zero rotation coordinates.
    def jacobian(flat):
        from scipy.sparse import block_diag,diags,kron,vstack
        poses=flat.reshape(n,6);blocks=[];steps=np.array([1e-4]*3+[1e-3]*3)
        for f in range(n):
            fun=lambda p:image_residual(p,corners,P,observations[f],directions)
            blocks.append(np.column_stack([(fun(poses[f]+np.eye(6)[j]*steps[j])-fun(poses[f]-np.eye(6)[j]*steps[j]))/(2*steps[j]) for j in range(6)]))
        D=diags([np.ones(n-2),-2*np.ones(n-2),np.ones(n-2)],[0,1,2],shape=(n-2,n))
        return vstack((block_diag(blocks,format='csr'),kron(D,diags(1/acceleration_scale))),format='csr')
    fit=least_squares(residual,initial.reshape(-1),jac=jacobian,loss='soft_l1',f_scale=2,max_nfev=45,ftol=2e-5,xtol=2e-5,gtol=1e-4,x_scale=np.tile([.1,.1,.1,.3,.3,.3],n),verbose=1)
    poses=fit.x.reshape(n,6)
    rows=[];metrics=[];legacy_valid=[];image_valid=[];independent=[];diagnostic=[];covariances=[]
    for f,p in enumerate(poses):
        obs=observations[f];vm=view_metrics(p,corners,P,obs);metrics.append(vm)
        # Unmodified original size-depth consistency diagnostic and thresholds.
        centres,disagreement=independent_view_centres(p,corners,P,obs,directions);independent.append(centres)
        step=np.array([1e-4]*3+[1e-3]*3)
        fn=lambda z:image_residual(z,corners,P,obs,directions,'all')
        J=np.column_stack([(fn(p+np.eye(6)[j]*step[j])-fn(p-np.eye(6)[j]*step[j]))/(2*step[j]) for j in range(6)])
        data=fn(p);cov=np.linalg.pinv(J.T@J)*max(float(np.mean(data*data)),.25);covariances.append(cov)
        s=np.linalg.svd(J*np.array([.1,.1,.1,.3,.3,.3]),compute_uv=False);condition=float(s[0]/max(s[-1],1e-12))
        # Re-fit each single view under the corrected censored observation model,
        # and profile cost of the joint centre with orientation held fixed.
        mono=[];mono_centres=[];mono_cost=[]
        for v in range(2):
            def single(c):
                z=p.copy();z[:3]=c
                return image_residual(z,corners,P[v:v+1],[obs[v]],directions,'all')
            ff=least_squares(single,p[:3],loss='soft_l1',f_scale=2,max_nfev=65,bounds=(p[:3]-.4,p[:3]+.4),diff_step=1e-3)
            jj=np.column_stack([(single(ff.x+np.eye(3)[j]*1e-4)-single(ff.x-np.eye(3)[j]*1e-4))/(2e-4) for j in range(3)])
            ss=np.linalg.svd(jj,compute_uv=False)
            cc=np.linalg.pinv(jj.T@jj)*max(float(np.mean(single(ff.x)**2)),.25)
            mono_centres.append(ff.x)
            local_cost=float(np.mean(single(ff.x)**2));joint_cost=float(np.mean(single(p[:3])**2));mono_cost.append(joint_cost-local_cost)
            mono.append({'jacobian_condition':float(ss[0]/max(ss[-1],1e-12)),'max_local_translation_std_m':float(np.sqrt(np.maximum(np.diag(cc),0)).max()),'joint_minus_best_mean_normalized_squared_residual':joint_cost-local_cost,'best_residual_rms_px_equivalent':float(3*np.sqrt(local_cost)),'joint_residual_rms_px_equivalent':float(3*np.sqrt(joint_cost))})
        censored_disagreement=float(np.linalg.norm(np.array(mono_centres)[0]-np.array(mono_centres)[1]))
        img=all(r['observable_iou']>=.55 and r['chamfer_mean_px']<=6 and r['chamfer_p95_px']<=16 for r in vm)
        strict=img and disagreement<=.1 and condition<=1e6 and np.sqrt(np.maximum(np.diag(cov)[:3],0)).max()<=.05
        image_valid.append(img);legacy_valid.append(strict)
        holdold=image_residual(x0[f],corners,g['P_video'],obs,directions,'holdout')
        holdnew=image_residual(p,corners,P,obs,directions,'holdout')
        # Classification is explanatory. It NEVER promotes legacy strict validity.
        if disagreement<=.1:classification='original_independent_centres_consistent'
        elif any(m['jacobian_condition']>100 or m['max_local_translation_std_m']>.05 for m in mono):classification='single_view_weakly_observable_conflict_unresolved'
        elif max(mono_cost)<1:classification='single_view_profile_flat_at_image_noise_scale_conflict_unresolved'
        else:classification='cross_view_image_model_conflict_not_explained_by_flat_single_view_profile'
        diagnostic.append(classification)
        rows.append({'frame':f,'timestamp':float(timestamps[f]),'center_xyz_m':p[:3].tolist(),'quaternion_xyzw':Rotation.from_rotvec(p[3:]).as_quat().tolist(),'image_fit_valid':bool(img),'strict_geometry_valid_original_thresholds':bool(strict),'independent_view_center_disagreement_m':disagreement,'censored_independent_view_center_disagreement_m':censored_disagreement,'single_view_diagnostics':mono,'classification':classification,'view_metrics':vm,'data_jacobian_condition':condition,'translation_std_local_proxy_m':np.sqrt(np.maximum(np.diag(cov)[:3],0)).tolist(),'holdout_v1_rms_normalized':float(np.sqrt(np.mean(holdold**2))),'holdout_v2_rms_normalized':float(np.sqrt(np.mean(holdnew**2)))})
        if f%30==0:print('box v2 diagnostics',f,strict,disagreement,classification,flush=True)
    q=Rotation.from_rotvec(poses[:,3:]).as_quat();T=np.tile(np.eye(4),(n,1,1));T[:,:3,:3]=Rotation.from_rotvec(poses[:,3:]).as_matrix();T[:,:3,3]=poses[:,:3]
    valid=np.asarray(legacy_valid);img=np.asarray(image_valid);validonly=poses[:,:3].copy();validonly[~valid]=np.nan
    npz=out/'box_trajectory.npz';np.savez_compressed(npz,timestamps=timestamps,center_xyz_m=poses[:,:3],quaternion_xyzw=q,rotation_vector=poses[:,3:],box_world_transform=T,half_extents_m=size/2,box_size_xyz_m=size,valid=valid,image_fit_valid=img,measurement_backed=np.ones(n,bool),center_valid_only_xyz_m=validonly,covariance_local_proxy=np.array(covariances),independent_view_centers_m=np.array(independent),independent_view_center_disagreement_m=np.array([r['independent_view_center_disagreement_m'] for r in rows]),projection_video=P,world_corners_m=np.array([project_pose(p,corners,P)[2] for p in poses]),camera_sha256=np.array(digest(camera)),source_segmentation_sha256=np.array(digest(source/'segmentation.npz')))
    trajectory={'coordinate_system':'world XYZ metres Zup, conditional supplied-scene metric scale','known_size_xyz_m':size.tolist(),'physical_metric_ground_truth':False,'frames':rows}
    (out/'box_trajectory.json').write_text(json.dumps(trajectory,indent=2)+'\n')
    report={'status':'conditional_rigid_sequence_optimized_unresolved_geometry_preserved','frames':n,'v1_strict_valid_frames':int(old['valid'].sum()),'v2_strict_valid_frames_same_thresholds':int(valid.sum()),'v2_image_fit_valid_frames':int(img.sum()),'invalid_frames':np.flatnonzero(~valid).tolist(),'known_box_size_xyz_m':size.tolist(),'camera_sha256':digest(camera),'box_sha256':digest(npz),'camera_optimized_using_dynamic_objects':False,'dimensions_optimized':False,'physical_metric_ground_truth':False,'lift_was_prescribed':False,'candidate_lift_max_relative_first5_m':float(poses[:,2].max()-np.median(poses[:5,2])),'candidate_final_lift_relative_first5_m':float(poses[-1,2]-np.median(poses[:5,2])),'method':'Fixed static-only refined camera; known rigid cuboid; hand-clipped support observations treated as one-sided lower bounds, artificial occlusion cut boundaries excluded; 3/4 angular contour observations fit, 1/4 held out; joint121-frame 6D sparse least-squares with robust image residual and translational/rotational acceleration prior. No height target, body contact target or arbitrary lift prescribed.','sequence_acceleration_scale_m_rad':acceleration_scale.tolist(),'joint_solver':{'success':bool(fit.success),'nfev':fit.nfev,'message':str(fit.message),'cost':float(fit.cost)},'classification_counts':{k:diagnostic.count(k) for k in sorted(set(diagnostic))},'v1_failed57_classifications':{k:sum(diagnostic[i]==k for i in range(n) if not old['valid'][i]) for k in sorted(set(diagnostic))},'holdout_policy':'Every fourth contour sample and support direction omitted from v2 fitting. Correlated contour samples make this an image generalization diagnostic, not fully independent semantic corner labels. v1 used all contour samples; comparison is conservative for v2.','holdout_v1_median_rms_normalized':float(np.median([r['holdout_v1_rms_normalized'] for r in rows])),'holdout_v2_median_rms_normalized':float(np.median([r['holdout_v2_rms_normalized'] for r in rows])),'median_chamfer_px_per_view':[float(np.median([r['view_metrics'][v]['chamfer_mean_px'] for r in rows])) for v in range(2)],'median_iou_per_view':[float(np.median([r['view_metrics'][v]['observable_iou'] for r in rows])) for v in range(2)],'median_legacy_independent_centres_disagreement_m':float(np.median([r['independent_view_center_disagreement_m'] for r in rows])),'median_censored_independent_centres_disagreement_m':float(np.median([r['censored_independent_view_center_disagreement_m'] for r in rows])),'acceleration_rms_v1_m_per_frame2':float(np.sqrt(np.mean(np.diff(x0[:,:3],n=2,axis=0)**2))),'acceleration_rms_v2_m_per_frame2':float(np.sqrt(np.mean(np.diff(poses[:,:3],n=2,axis=0)**2))),'validity_note':'Original .1m cross-view centre, .55IoU,6px mean/16pxp95 edge,1e6condition,.05m localstd gates retained. v2 does not use multistart, so no new multistart claim: original v1 ambiguity span is preserved as a separate input and checked below. Image fit and metric geometry remain distinct; all centres are candidates under conditional cameras.','uncertainty_note':'Local linearization excludes camera/model uncertainty and correlations; not a statistical confidence interval. Profile classifications do not prove either real rigidity or nonrigidity; semantic matched corners unavailable.','elapsed_seconds':time.time()-started,'per_frame':rows}
    # Preserve original multistart gate without claiming a new search passed.
    badspread=np.max(old['multistart_parameter_span'][:,:3],axis=1)>.04
    if np.any(valid&badspread):raise RuntimeError('Original multistart gate was violated')
    preview=ROOT/'outputs/previews/user_box_optimization_v2.mp4'
    make_preview(ROOT/'data/video_generation/candidate_scene/generated.mp4',preview,poses,valid,metrics,corners,P,timestamps)
    report['preview_sha256']=digest(preview)
    report['joint_solver']['first_order_optimality']=float(fit.optimality)
    report['joint_solver']['termination_note']='Step-size termination of a nonsmooth contour objective does not establish a stationary or global optimum. Objective reduction and held-out image errors are reported separately.'
    (ROOT/'reports/user-box-optimization.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='per_frame'},indent=2),flush=True)


def diagnose_full_pose():
    """Allow monocular orientation freedom; retain strict checks and output arrays."""
    import numpy as np
    import cv2
    from scipy.optimize import least_squares
    from reconstruct_box import observation,box_corners,digest,data_residual
    cv2.setNumThreads(4)
    out=ROOT/'outputs/user_reconstruction_v2';source=ROOT/'outputs/user_reconstruction'
    pz=np.load(out/'box_trajectory.npz');camera=np.load(out/'camera_refinement.npz');P=camera['P_refined']
    seg=np.load(source/'segmentation.npz');masks,robots,valid=seg['box'],seg['robot'],seg['observation_valid']
    poses=np.column_stack((pz['center_xyz_m'],pz['rotation_vector']));corners=box_corners(pz['box_size_xyz_m']);old=np.load(source/'box_trajectory.npz')
    angles=np.arange(64)*2*np.pi/64;directions=np.column_stack((np.cos(angles),np.sin(angles)))
    rp=ROOT/'reports/user-box-optimization.json';report=json.loads(rp.read_text());diagnostics=[]
    for f,p in enumerate(poses):
        obs=[observation(masks[f,v],valid[f,v],robots[f,v],directions) for v in range(2)]
        mono=[]
        for v in range(2):
            fun=lambda q:image_residual(q,corners,P[v:v+1],[obs[v]],directions,'all')
            starts=[p.copy(),p+np.array([0,0,0,0,0,.2]),p-np.array([0,0,0,0,0,.2])]
            fits=[least_squares(fun,s,bounds=(p-[.4,.4,.4,.5,.5,.5],p+[.4,.4,.4,.5,.5,.5]),loss='soft_l1',f_scale=2,max_nfev=70,diff_step=1e-3,x_scale=[.1,.1,.1,.3,.3,.3]) for s in starts]
            best=min(fits,key=lambda z:z.cost);step=np.array([1e-4]*3+[1e-3]*3)
            J=np.column_stack([(fun(best.x+np.eye(6)[j]*step[j])-fun(best.x-np.eye(6)[j]*step[j]))/(2*step[j]) for j in range(6)])
            singular=np.linalg.svd(J*np.array([.1,.1,.1,.3,.3,.3]),compute_uv=False)
            cov=np.linalg.pinv(J.T@J)*max(float(np.mean(fun(best.x)**2)),.25)
            # With joint 3D centre fixed, optimize orientation: a fairer profile
            # than demanding that two differently biased mono point estimates agree.
            def orientation(r):return fun(np.r_[p[:3],r])
            orient=least_squares(orientation,p[3:],bounds=(p[3:]-.5,p[3:]+.5),loss='soft_l1',f_scale=2,max_nfev=60,diff_step=1e-3)
            mono.append({'best_pose_xyz_rotvec':best.x.tolist(),'normalized_6d_jacobian_condition':float(singular[0]/max(singular[-1],1e-12)),'translation_std_local_proxy_m':np.sqrt(np.maximum(np.diag(cov)[:3],0)).tolist(),'best_rms_px_equivalent':float(3*np.sqrt(np.mean(fun(best.x)**2))),'fixed_joint_center_orientation_profile_rms_px_equivalent':float(3*np.sqrt(np.mean(orientation(orient.x)**2))),'orientation_profile_cost_increase':float(orient.cost-best.cost),'multistart_center_span_m':np.ptp(np.array([z.x[:3] for z in fits]),axis=0).tolist()})
        # Re-evaluate original v1 local data Jacobian, not the changed likelihood.
        fun=lambda q:data_residual(q,corners,P,obs,directions)
        step=np.array([1e-4]*3+[1e-3]*3)
        J=np.column_stack([(fun(p+np.eye(6)[j]*step[j])-fun(p-np.eye(6)[j]*step[j]))/(2*step[j]) for j in range(6)])
        ss=np.linalg.svd(J*np.array([.1,.1,.1,.3,.3,.3]),compute_uv=False)
        cov=np.linalg.pinv(J.T@J,rcond=1e-10)*max(float(np.mean(fun(p)**2)),.25)
        d={'frame':f,'v1_was_invalid':not bool(old['valid'][f]),'monocular_full6d':mono,'original_residual_joint_condition':float(ss[0]/max(ss[-1],1e-12)),'original_residual_joint_max_translation_std_m':float(np.sqrt(np.maximum(np.diag(cov)[:3],0)).max())}
        diagnostics.append(d)
        if f%30==0:print('6d single-view profile',f,flush=True)
    report['full6d_single_view_profiles']=diagnostics
    report['full6d_diagnostic_note']='Monocular orientation is now free with three yaw starts; reported cost profiles and ambiguity are diagnostics only. Continuous noise-scaled residuals are reported rather than a new pass threshold. No monocular best point estimate is treated as geometric truth. Original residual local Jacobian is separately recomputed for comparison.'
    failed=[d for d in diagnostics if d['v1_was_invalid']]
    report['v1_failed57_full6d_summary']={
        'median_max_single_view_condition':float(np.median([max(m['normalized_6d_jacobian_condition'] for m in d['monocular_full6d']) for d in failed])),
        'median_max_single_view_local_translation_std_m':float(np.median([max(max(m['translation_std_local_proxy_m']) for m in d['monocular_full6d']) for d in failed])),
        'median_max_best_single_view_rms_px_equivalent':float(np.median([max(m['best_rms_px_equivalent'] for m in d['monocular_full6d']) for d in failed])),
        'median_max_fixed_center_orientation_profile_rms_px_equivalent':float(np.median([max(m['fixed_joint_center_orientation_profile_rms_px_equivalent'] for m in d['monocular_full6d']) for d in failed]))}
    report['original_v1_local_condition_gate_pass_frames']=sum(d['original_residual_joint_condition']<=1e6 for d in diagnostics)
    report['original_v1_local_std_gate_pass_frames']=sum(d['original_residual_joint_max_translation_std_m']<=.05 for d in diagnostics)
    rp.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['v1_failed57_full6d_summary'],indent=2),flush=True)
    summarize_profiles()


def summarize_profiles():
    import numpy as np
    rp=ROOT/'reports/user-box-optimization.json';r=json.loads(rp.read_text())
    failed=[z for z in r['full6d_single_view_profiles'] if z['v1_was_invalid']]
    distances=np.array([np.linalg.norm(np.array(z['monocular_full6d'][0]['best_pose_xyz_rotvec'][:3])-np.array(z['monocular_full6d'][1]['best_pose_xyz_rotvec'][:3])) for z in failed])
    r['v1_failed57_full6d_summary'].update({'median_monocular_free_orientation_center_disagreement_m':float(np.median(distances)),'free_orientation_same_original_0_1m_center_gate_pass':int((distances<=.1).sum()),'free_orientation_still_over_original_center_gate':int((distances>.1).sum()),'strict_validity_promoted_from_this_diagnostic':False})
    s=r['v1_failed57_full6d_summary']
    r['failure_interpretation']=f"Original {len(failed)} failures were independent-view centre disagreement, not joint-Jacobian singularity. Allowing monocular rotation freedom puts {s['free_orientation_same_original_0_1m_center_gate_pass']} pairs within the original .1m bound, while {s['free_orientation_still_over_original_center_gate']} still exceed it. Median mono full6D normalized condition {s['median_max_single_view_condition']:.2f} and local translation proxy {s['median_max_single_view_local_translation_std_m']:.5f}m do not support numerical singularity as the sole explanation. Best-mono and fixed-joint-centre orientation-profile image residuals are {s['median_max_best_single_view_rms_px_equivalent']:.3f}px vs {s['median_max_fixed_center_orientation_profile_rms_px_equivalent']:.3f}px; small image/model/occlusion errors can bias size-depth estimates. These data do not establish real physical nonrigidity or unique metric pose. No diagnostic upgrades strict validity; semantic corner correspondences/real calibrated video would be needed to resolve ambiguity."
    r['v2_selection_note']=f"v2 is the smoother image-consistent candidate under independently improved static cameras; strict metric pass count is {r['v2_strict_valid_frames_same_thresholds']} vs v1 {r['v1_strict_valid_frames']}. All v1 artifacts remain intact. Both candidates and their flags must remain available."
    rp.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r['v1_failed57_full6d_summary'],indent=2),flush=True)


def candidate_v3():
    """Independent numerical refinement; never replace the frozen v2 contract."""
    import numpy as np
    import cv2
    from scipy.optimize import least_squares
    from scipy.sparse import block_diag,diags,kron,vstack
    from scipy.spatial.transform import Rotation
    from reconstruct_box import observation,box_corners,view_metrics,independent_view_centres,data_residual,project_pose,make_preview,digest
    cv2.setNumThreads(4);started=time.time()
    out=ROOT/'outputs/user_reconstruction_v2';source=ROOT/'outputs/user_reconstruction'
    frozen_path=out/'box_trajectory.npz';frozen_sha=digest(frozen_path)
    frozen=np.load(frozen_path);old=np.load(source/'box_trajectory.npz');cam=np.load(out/'camera_refinement.npz');P=cam['P_refined']
    seg=np.load(source/'segmentation.npz');masks,robots,valid=seg['box'],seg['robot'],seg['observation_valid']
    x0=np.column_stack((old['center_xyz_m'],old['rotation_vector']));xf=np.column_stack((frozen['center_xyz_m'],frozen['rotation_vector']));n=len(x0)
    size=old['box_size_xyz_m'];corners=box_corners(size);timestamps=old['timestamps'];a=np.arange(64)*2*np.pi/64;directions=np.column_stack((np.cos(a),np.sin(a)))
    observations=[]
    for f in range(n):
        pair=[observation(masks[f,v],valid[f,v],robots[f,v],directions) for v in range(2)]
        for o in pair:
            o['distance']=o['distance'].astype(float)
            o['excluded_float']=o['excluded'].astype(float)
        observations.append(pair)
    steps=np.array([1e-5]*3+[1e-4]*3)
    def fun(f,p,split='train'):return image_residual(p,corners,P,observations[f],directions,split,precise=True)
    def jac(f,p):return np.column_stack([(fun(f,p+np.eye(6)[j]*steps[j])-fun(f,p-np.eye(6)[j]*steps[j]))/(2*steps[j]) for j in range(6)])
    def evaluate(f,p):
        vm=view_metrics(p,corners,P,observations[f]);centres,dis=independent_view_centres(p,corners,P,observations[f],directions)
        fn=lambda q:data_residual(q,corners,P,observations[f],directions)
        step=np.array([1e-4]*3+[1e-3]*3);J=np.column_stack([(fn(p+np.eye(6)[j]*step[j])-fn(p-np.eye(6)[j]*step[j]))/(2*step[j]) for j in range(6)])
        s=np.linalg.svd(J*np.array([.1,.1,.1,.3,.3,.3]),compute_uv=False);condition=float(s[0]/max(s[-1],1e-12));cov=np.linalg.pinv(J.T@J,rcond=1e-10)*max(float(np.mean(fn(p)**2)),.25)
        img=all(v['observable_iou']>=.55 and v['chamfer_mean_px']<=6 and v['chamfer_p95_px']<=16 for v in vm)
        strict=img and dis<=.1 and condition<=1e6 and np.sqrt(np.maximum(np.diag(cov)[:3],0)).max()<=.05
        validation=float(np.sqrt(np.mean(fun(f,p,'holdout')**2)))
        return {'strict':bool(strict),'image':bool(img),'view_metrics':vm,'independent_view_center_disagreement_m':dis,'validation_rms_normalized':validation,'original_data_condition':condition,'max_local_translation_std_m':float(np.sqrt(np.maximum(np.diag(cov)[:3],0)).max())},cov,centres
    chosen=[];selection=[];spreads=[]
    for f in range(n):
        starts=[x0[f],xf[f],x0[f]+[0,0,0,0,0,.2],x0[f]-[0,0,0,0,0,.2]]
        solutions=[]
        for start in starts:
            fit=least_squares(lambda p:fun(f,p),start,jac=lambda p:jac(f,p),bounds=(x0[f]-[.25,.25,.25,.6,.6,.6],x0[f]+[.25,.25,.25,.6,.6,.6]),loss='soft_l1',f_scale=2,max_nfev=100,x_scale=[.1,.1,.1,.3,.3,.3],ftol=1e-7,xtol=1e-8,gtol=1e-3)
            solutions.append((fit.x,{'cost':float(fit.cost),'optimality':float(fit.optimality),'nfev':fit.nfev,'message':str(fit.message)}))
        # Include measured unfitted candidates; selection is validation, never
        # relabel interpolation. All candidates use actual two-view masks.
        solutions.extend([(xf[f],{'source':'frozen_v2'}),(x0[f],{'source':'v1_pose_reprojected_new_camera'})])
        evaluated=[(p,info,*evaluate(f,p)) for p,info in solutions]
        reference=next(z for z in evaluated if z[1].get('source')=='frozen_v2')[2]['validation_rms_normalized']
        safe=[z for z in evaluated if z[2]['validation_rms_normalized']<=reference+1e-10]
        passing=[z for z in safe if z[2]['strict']]
        pick=min(passing or safe,key=lambda z:z[2]['validation_rms_normalized'])
        bestcost=min(z[1].get('cost',float('inf')) for z in evaluated)
        plausible=[z[0] for z in evaluated if z[1].get('cost',float('inf'))<=bestcost*1.1+.5]
        span=np.ptp(plausible,axis=0) if len(plausible)>1 else np.zeros(6)
        # Original multistart ambiguity criterion remains active.
        pick[2]['strict']=bool(pick[2]['strict'] and max(span[:3])<=.04)
        chosen.append(pick[0]);spreads.append(span);selection.append({'frame':f,'selected_solver':pick[1],'selected':pick[2],'candidate_count':len(evaluated),'strict_candidates_no_worse_validation':len(passing),'multistart_span':span.tolist()})
        if f%20==0:print('v3 multistart',f,pick[2]['strict'],pick[2]['validation_rms_normalized'],flush=True)
    chosen=np.array(chosen);accscale=np.array([.006,.006,.006,.025,.025,.025])
    D=diags([np.ones(n-2),-2*np.ones(n-2),np.ones(n-2)],[0,1,2],shape=(n-2,n));A=kron(D,diags(1/accscale))
    def sequence(flat):
        poses=flat.reshape(n,6)
        return np.r_[np.concatenate([fun(f,poses[f]) for f in range(n)]),((poses[2:]-2*poses[1:-1]+poses[:-2])/accscale).reshape(-1)]
    def sequence_jac(flat):
        poses=flat.reshape(n,6)
        return vstack((block_diag([jac(f,poses[f]) for f in range(n)],format='csr'),A),format='csr')
    fit=least_squares(sequence,chosen.reshape(-1),jac=sequence_jac,loss='soft_l1',f_scale=2,max_nfev=70,x_scale=np.tile([.1,.1,.1,.3,.3,.3],n),ftol=1e-7,xtol=1e-8,gtol=1e-3,verbose=1)
    smooth=fit.x.reshape(n,6);poses=chosen.copy();records=[];metrics=[];covs=[];centres=[];flags=[];images=[];fallbacks=[]
    for f in range(n):
        before,_,_=evaluate(f,chosen[f]);after,cov,c=evaluate(f,smooth[f])
        accept=after['validation_rms_normalized']<=before['validation_rms_normalized']+1e-10 and (not before['strict'] or after['strict'])
        if accept:poses[f]=smooth[f];row=after
        else:row,cov,c=evaluate(f,poses[f])
        row['strict']=bool(row['strict'] and max(spreads[f][:3])<=.04)
        records.append({'frame':f,'timestamp':float(timestamps[f]),'center_xyz_m':poses[f,:3].tolist(),'quaternion_xyzw':Rotation.from_rotvec(poses[f,3:]).as_quat().tolist(),'image_metrics':row,'selection':selection[f],'temporal_candidate_accepted':bool(accept),'temporal_validation_rms':after['validation_rms_normalized']});metrics.append(row['view_metrics']);covs.append(cov);centres.append(c);flags.append(row['strict']);images.append(row['image']);fallbacks.append(not accept)
    flags=np.array(flags);T=np.tile(np.eye(4),(n,1,1));T[:,:3,:3]=Rotation.from_rotvec(poses[:,3:]).as_matrix();T[:,:3,3]=poses[:,:3];q=Rotation.from_rotvec(poses[:,3:]).as_quat();vonly=poses[:,:3].copy();vonly[~flags]=np.nan
    path=out/'box_trajectory_candidate_v3.npz'
    np.savez_compressed(path,timestamps=timestamps,center_xyz_m=poses[:,:3],quaternion_xyzw=q,rotation_vector=poses[:,3:],box_world_transform=T,half_extents_m=size/2,box_size_xyz_m=size,valid=flags,image_fit_valid=np.array(images),measurement_backed=np.ones(n,bool),center_valid_only_xyz_m=vonly,covariance_local_proxy=np.array(covs),multistart_parameter_span=np.array(spreads),independent_view_centers_m=np.array(centres),independent_view_center_disagreement_m=np.array([r['image_metrics']['independent_view_center_disagreement_m'] for r in records]),projection_video=P,camera_sha256=np.array(digest(out/'camera_refinement.npz')),world_corners_m=np.array([project_pose(p,corners,P)[2] for p in poses]),temporal_fallback=np.array(fallbacks))
    (out/'box_trajectory_candidate_v3.json').write_text(json.dumps({'physical_metric_ground_truth':False,'frames':records},indent=2)+'\n')
    preview=ROOT/'outputs/previews/user_box_optimization_candidate_v3.mp4';make_preview(ROOT/'data/video_generation/candidate_scene/generated.mp4',preview,poses,flags,metrics,corners,P,timestamps)
    report={'status':'separate_refinement_candidate_v3','frames':n,'v1_strict_valid':int(old['valid'].sum()),'frozen_v2_strict_valid':int(frozen['valid'].sum()),'candidate_v3_strict_valid_same_thresholds':int(flags.sum()),'candidate_v3_image_fit_valid':int(sum(images)),'frozen_v2_unchanged':digest(frozen_path)==frozen_sha,'frozen_v2_sha256':frozen_sha,'candidate_sha256':digest(path),'camera_sha256':digest(out/'camera_refinement.npz'),'source_segmentation_sha256':digest(source/'segmentation.npz'),'numeric_correction':'OpenCV hull used only for vertex indices; coordinates and distance sampling remain float64. Explicit absolute central differences 1e-5m/1e-4rad with parameter scales. No criterion thresholds changed.','selection_policy':'Four actual image-fit starts plus frozen v2 and reprojected v1 poses; original strict gates preferred ONLY when validation error does not exceed frozen v2. Validation subset used for model selection, so no longer an unbiased held-out test. Temporal joint solve accepted framewise only if validation does not increase and original strict pass is preserved; failed temporal candidates explicitly fall back to measured optimized pose, never interpolation.','thresholds':{'center_disagreement_m':.1,'iou_min':.55,'chamfer_mean_max_px':6,'chamfer_p95_max_px':16,'data_condition_max':1e6,'local_std_max_m':.05,'multistart_translation_span_max_m':.04},'physical_metric_ground_truth':False,'sequence_solver':{'nfev':fit.nfev,'success':bool(fit.success),'optimality':float(fit.optimality),'cost':float(fit.cost),'termination':str(fit.message)},'temporal_fallback_frames':np.flatnonzero(fallbacks).tolist(),'validation_median_rms_normalized':float(np.median([r['image_metrics']['validation_rms_normalized'] for r in records])),'acceleration_rms_m_per_frame2':float(np.sqrt(np.mean(np.diff(poses[:,:3],n=2,axis=0)**2))),'elapsed_seconds':time.time()-started,'preview_sha256':digest(preview),'per_frame':records}
    (ROOT/'reports/user-box-optimization-v3.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='per_frame'},indent=2),flush=True)
    audit_candidate_v3()


def audit_candidate_v3():
    """Recompute candidate-vs-frozen validation and explain every failed gate."""
    import numpy as np
    import cv2
    from reconstruct_box import observation,box_corners,digest,data_residual,view_metrics
    cv2.setNumThreads(4)
    out=ROOT/'outputs/user_reconstruction_v2';source=ROOT/'outputs/user_reconstruction'
    frozen_path=out/'box_trajectory.npz';candidate_path=out/'box_trajectory_candidate_v3.npz'
    old=np.load(frozen_path);new=np.load(candidate_path);camera=np.load(out/'camera_refinement.npz')
    seg=np.load(source/'segmentation.npz');masks,robots,valid=seg['box'],seg['robot'],seg['observation_valid'];corners=box_corners(new['box_size_xyz_m'])
    a=np.arange(64)*2*np.pi/64;directions=np.column_stack((np.cos(a),np.sin(a)))
    rp=ROOT/'reports/user-box-optimization-v3.json';r=json.loads(rp.read_text());comparisons=[];reasons_count={}
    for f,row in enumerate(r['per_frame']):
        row['timestamp']=float(new['timestamps'][f]);row['center_xyz_m']=new['center_xyz_m'][f].tolist();row['quaternion_xyzw']=new['quaternion_xyzw'][f].tolist()
        pair=[observation(masks[f,v],valid[f,v],robots[f,v],directions) for v in range(2)]
        for o in pair:o['distance']=o['distance'].astype(float);o['excluded_float']=o['excluded'].astype(float)
        p=np.r_[new['center_xyz_m'][f],new['rotation_vector'][f]];base=np.r_[old['center_xyz_m'][f],old['rotation_vector'][f]]
        rms=lambda q:float(np.sqrt(np.mean(image_residual(q,corners,camera['P_refined'],pair,directions,'holdout',precise=True)**2)))
        baseline,candidate=rms(base),rms(p)
        comparisons.append({'frame':f,'frozen_v2_validation_rms':baseline,'candidate_v3_validation_rms':candidate,'delta':candidate-baseline})
        d=row['image_metrics'];reasons=[]
        if d['independent_view_center_disagreement_m']>.1:reasons.append('independent_view_centres_exceed_original_0_1m')
        if d['original_data_condition']>1e6:reasons.append('original_data_condition_exceeds_1e6')
        if d['max_local_translation_std_m']>.05:reasons.append('original_local_translation_std_exceeds_0_05m')
        if max(row['selection']['multistart_span'][:3])>.04:reasons.append('multistart_translation_ambiguity_exceeds_original_0_04m')
        for v,m in enumerate(d['view_metrics']):
            if m['observable_iou']<.55:reasons.append(f'view{v}_iou_below_original_0_55')
            if m['chamfer_mean_px']>6 or m['chamfer_p95_px']>16:reasons.append(f'view{v}_edge_error_exceeds_original_gate')
        row['original_gate_failure_reasons']=reasons
        if reasons:
            mono_centres=new['independent_view_centers_m'][f]
            profile=[];cross=[]
            for v in range(2):
                mono_pose=p.copy();mono_pose[:3]=mono_centres[v]
                local=data_residual(mono_pose,corners,camera['P_refined'][v:v+1],[pair[v]],directions)
                joint=data_residual(p,corners,camera['P_refined'][v:v+1],[pair[v]],directions)
                profile.append({'view':v,'local_mono_translation_xyz_m':mono_centres[v].tolist(),'local_mono_rms_px_equivalent':float(3*np.sqrt(np.mean(local**2))),'joint_rms_px_equivalent':float(3*np.sqrt(np.mean(joint**2))),'caution':'Orientation fixed to candidate; local optimizer estimate and weighted pixel-equivalent residual, not metric truth or semantic corner triangulation.'})
                cross.append([m['chamfer_mean_px'] for m in view_metrics(mono_pose,corners,camera['P_refined'],pair)])
            row['actual_single_view_translation_profiles']=profile
            row['mono_center_crossprojection_chamfer_mean_px_matrix']=cross
            row['conflict_interpretation']='Both visible silhouettes admit a good shared-box image fit, yet separately optimized size-depth point estimates differ beyond the original bound. Crossprojection errors show why those mono point estimates cannot simply be combined. This is conditional model/image disagreement, not evidence of unique physical nonrigidity.'
        for reason in reasons:reasons_count[reason]=reasons_count.get(reason,0)+1
        if bool(new['valid'][f])!=(not bool(reasons)):raise RuntimeError('Candidate strict flag does not match original gate audit')
    regress=[x['frame'] for x in comparisons if x['delta']>1e-8]
    if regress:raise RuntimeError(f'Candidate regressed actual image validation frames {regress}')
    r['independent_validation_recomputation']={'all_frames_no_worse_than_frozen_v2':not regress,'max_rms_delta':max(x['delta'] for x in comparisons),'median_frozen_rms':float(np.median([x['frozen_v2_validation_rms'] for x in comparisons])),'median_candidate_rms':float(np.median([x['candidate_v3_validation_rms'] for x in comparisons])),'per_frame':comparisons,'not_an_unbiased_test':'The validation pixels were explicitly used for candidate selection; this recomputation verifies the non-regression rule only.'}
    r['original_gate_failure_counts']=reasons_count
    r['immutable_contract_checks']={'frozen_v2_sha_unchanged':digest(frozen_path)==r['frozen_v2_sha256'],'same_fixed_size':bool(np.array_equal(new['box_size_xyz_m'],old['box_size_xyz_m'])),'same_static_refined_camera':bool(np.array_equal(new['projection_video'],camera['P_refined'])),'all_poses_finite':bool(np.isfinite(new['center_xyz_m']).all() and np.isfinite(new['quaternion_xyzw']).all()),'all121_have_real_observations':bool(new['measurement_backed'].all())}
    selected=[x['selection']['selected_solver'] for x in r['per_frame'] if 'optimality' in x['selection']['selected_solver']]
    r['selected_frame_solver_optimality']={'fitted_selected_frames':len(selected),'median':float(np.median([x['optimality'] for x in selected])) if selected else None,'max':max((x['optimality'] for x in selected),default=None),'note':'Termination codes are recorded per frame. Piecewise image objectives may stop by step size; this is not proof of global optimality.'}
    (out/'box_trajectory_candidate_v3.json').write_text(json.dumps({'coordinate_system':'world XYZ metres Zup, OpenCV cameras; metric scale conditional','known_size_xyz_m':new['box_size_xyz_m'].tolist(),'physical_metric_ground_truth':False,'frames':r['per_frame']},indent=2)+'\n')
    rp.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({'audit':'passed','gate_failure_counts':reasons_count,'immutable_contract_checks':r['immutable_contract_checks'],'frame_solver_optimality':r['selected_frame_solver_optimality']},indent=2),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--diagnose-6d',action='store_true',help='Diagnose existing frozen trajectory; do not refit it')
    mode.add_argument('--summarize-profiles',action='store_true',help='Only summarize existing completed 6D profiles')
    mode.add_argument('--fit-only',action='store_true',help='Run the sequence fit without the extra monocular profiles')
    mode.add_argument('--candidate-v3',action='store_true',help='Separate double-precision multistart candidate; leave v2 frozen')
    mode.add_argument('--audit-candidate-v3',action='store_true',help='Audit existing v3 against frozen v2 without changing arrays')
    args=parser.parse_args()
    if args.audit_candidate_v3:audit_candidate_v3()
    elif args.candidate_v3:candidate_v3()
    elif args.diagnose_6d:diagnose_full_pose()
    elif args.summarize_profiles:summarize_profiles()
    else:
        main()
        if not args.fit_only:diagnose_full_pose()
