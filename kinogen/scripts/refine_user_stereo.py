#!/usr/bin/env python3
"""Static-only constrained camera refinement with spatial holdout validation."""
from pathlib import Path
import json
import os
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    import cv2
    import numpy as np
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    from reconstruct_box import digest
    cv2.setNumThreads(4)
    start = time.time()
    source = ROOT/'outputs/user_reconstruction'
    output = ROOT/'outputs/user_reconstruction_v2'
    output.mkdir(parents=True, exist_ok=True)
    g = np.load(source/'geometry.npz')
    seg = np.load(source/'segmentation.npz')
    authored = cv2.imread(str(ROOT/'data/video_generation/candidate_scene/initial_two_view.png'))
    cap = cv2.VideoCapture(str(ROOT/'data/video_generation/candidate_scene/generated.mp4'))
    ok, first = cap.read(); cap.release()
    if not ok: raise RuntimeError('Missing video first frame')
    K, R, t = g['K'].copy(), g['R'].copy(), g['t'].copy()
    H, baseline = g['image_homography_ref_to_video'].copy(), g['P_video'].copy()
    refined_R, refined_t = R.copy(), t.copy()
    records, points_world, points_observed, holds, accepted = [], [], [], [], []
    for v in range(2):
        ref = cv2.cvtColor(authored[:,v*640:(v+1)*640], cv2.COLOR_BGR2GRAY)
        cur = cv2.cvtColor(first[:,v*640:(v+1)*640], cv2.COLOR_BGR2GRAY)
        valid = g['reference_static_mask'][v].astype(np.uint8)
        valid &= cv2.erode((ref<230).astype(np.uint8), np.ones((7,7),np.uint8))
        valid[:145] = 0
        candidates = cv2.goodFeaturesToTrack(ref, maxCorners=650, qualityLevel=.006, minDistance=9, mask=valid*255)
        if candidates is None: raise RuntimeError('No static observations')
        uv = candidates[:,0].astype(float)
        init = cv2.perspectiveTransform(candidates, H[v]).astype(np.float32)
        flow, status, error = cv2.calcOpticalFlowPyrLK(ref,cur,candidates,init.copy(),winSize=(19,19),maxLevel=4,criteria=(3,50,.005),flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
        back, bs, _ = cv2.calcOpticalFlowPyrLK(cur,ref,flow,None,winSize=(19,19),maxLevel=4)
        measured = flow[:,0].astype(float)
        xy = np.floor(uv).astype(int); dest = np.rint(measured).astype(int).clip([0,0],[639,719])
        z = g['reference_depth_z_m'][v,xy[:,1],xy[:,0]]
        dynamic = cv2.dilate((seg['robot'][0,v]|seg['box'][0,v]).astype(np.uint8),np.ones((17,17),np.uint8))>0
        keep = (status[:,0]>0)&(bs[:,0]>0)&(np.linalg.norm(back[:,0]-candidates[:,0],axis=1)<.7)&(error[:,0]<18)&np.isfinite(z)&(z>.1)&(z<12)
        keep &= ~dynamic[dest[:,1],dest[:,0]]&seg['observation_valid'][0,v,dest[:,1],dest[:,0]]
        # This fixed 4px correspondence filter is applied before splitting or optimization.
        keep &= (np.linalg.norm(measured-init[:,0],axis=1)<4)&(measured[:,0]>4)&(measured[:,0]<636)&(measured[:,1]>145)&(measured[:,1]<716)
        uv, measured, z = uv[keep], measured[keep], z[keep]
        # Independently detected descriptor correspondences can survive LK failure
        # on narrow rendered table/grid edges. Keep source pixels separated.
        sift=cv2.SIFT_create(nfeatures=3500,contrastThreshold=.015)
        aa,da=sift.detectAndCompute(ref,valid*255);bb,db=sift.detectAndCompute(cur,None)
        if da is not None and db is not None:
            pairs=cv2.BFMatcher().knnMatch(da,db,k=2)
            for m,n in pairs:
                if m.distance>=.70*n.distance: continue
                a=np.array(aa[m.queryIdx].pt);b=np.array(bb[m.trainIdx].pt)
                x,y=np.floor(a).astype(int);dx,dy=np.rint(b).astype(int)
                if not (4<=x<636 and 145<=y<716 and 4<=dx<636 and 145<=dy<716):continue
                zz=g['reference_depth_z_m'][v,y,x]
                predicted=H[v]@np.r_[a,1];predicted=predicted[:2]/predicted[2]
                if not np.isfinite(zz) or not .1<zz<12 or np.linalg.norm(predicted-b)>=4:continue
                if dynamic[dy,dx] or not seg['observation_valid'][0,v,dy,dx]:continue
                if len(uv) and np.linalg.norm(uv-a,axis=1).min()<9:continue
                uv=np.vstack((uv,a));measured=np.vstack((measured,b));z=np.r_[z,zz]
        # Ray samples use the same half-pixel convention as the mesh depth raster.
        ray_uv = np.floor(uv)+.5
        cam = np.column_stack((ray_uv,np.ones(len(uv))))@np.linalg.inv(K[v]).T*z[:,None]
        world = (cam-t[v])@R[v]
        # Spatial cells, not shuffled adjacent duplicate pixels, are held out.
        cells = np.floor(uv/[75,65]).astype(int)
        hold = ((cells[:,0]*3+cells[:,1]*5)%4)==0
        print('static candidates',v,len(world),int(hold.sum()),int((~hold).sum()),flush=True)
        if hold.sum()<8 or (~hold).sum()<16:
            points_world.append(world);points_observed.append(measured);holds.append(hold);accepted.append(False)
            records.append({'view':v,'observations':len(world),'training_points':int((~hold).sum()),'spatial_holdout_points':int(hold.sum()),'accepted':False,'reason':'insufficient_independent_spatial_holdout_or_training_points_original_8_16_gate','selected_camera':'unchanged_v1'})
            continue
        def matrices(p):
            rr = Rotation.from_rotvec(p[:3]).as_matrix()@R[v]
            tt = t[v]+p[3:]
            return rr,tt
        def project(p):
            rr,tt=matrices(p)
            q=(world@rr.T+tt)@K[v].T@H[v].T
            return q[:,:2]/q[:,2:]
        def residual(p):
            return np.r_[(project(p)-measured)[~hold].reshape(-1),p[:3]/.004,p[3:]/.012]
        fit=least_squares(residual,np.zeros(6),bounds=(np.r_[[ -.012]*3,[-.04]*3],np.r_[[.012]*3,[.04]*3]),loss='soft_l1',f_scale=1,max_nfev=100,x_scale=np.r_[[.004]*3,[.012]*3],diff_step=1e-3)
        before=np.linalg.norm(project(np.zeros(6))-measured,axis=1)
        after=np.linalg.norm(project(fit.x)-measured,axis=1)
        # Candidate must improve independent spatial holdout RMS without worsening p95.
        rms=lambda x:float(np.sqrt(np.mean(x*x)))
        allow=bool(rms(after[hold])<rms(before[hold])*.98 and np.quantile(after[hold],.95)<=np.quantile(before[hold],.95)+.05)
        if allow: refined_R[v],refined_t[v]=matrices(fit.x)
        normalized=fit.jac[:2*(~hold).sum()]*np.r_[[.004]*3,[.012]*3]
        singular=np.linalg.svd(normalized,compute_uv=False)
        points_world.append(world);points_observed.append(measured);holds.append(hold);accepted.append(allow)
        records.append({'view':v,'observations':len(world),'training_points':int((~hold).sum()),'spatial_holdout_points':int(hold.sum()),'world_z_range_m':[float(world[:,2].min()),float(world[:,2].max())], 'baseline_train_rms_px':rms(before[~hold]),'candidate_train_rms_px':rms(after[~hold]),'baseline_holdout_rms_px':rms(before[hold]),'candidate_holdout_rms_px':rms(after[hold]),'baseline_holdout_p95_px':float(np.quantile(before[hold],.95)),'candidate_holdout_p95_px':float(np.quantile(after[hold],.95)),'accepted':allow,'candidate_delta_rotation_rad':fit.x[:3].tolist(),'candidate_delta_t_camera_m':fit.x[3:].tolist(),'data_only_normalized_singular_values':singular.tolist(),'data_only_condition':float(singular[0]/max(singular[-1],1e-12))})
    P=np.array([H[v]@K[v]@np.column_stack((refined_R[v],refined_t[v])) for v in range(2)])
    maxpoints=max(map(len,points_world));worlds=np.full((2,maxpoints,3),np.nan);observed=np.full((2,maxpoints,2),np.nan);holdouts=np.zeros((2,maxpoints),bool);pointvalid=np.zeros_like(holdouts)
    for v in range(2):
        n=len(points_world[v]);worlds[v,:n]=points_world[v];observed[v,:n]=points_observed[v];holdouts[v,:n]=holds[v];pointvalid[v,:n]=True
    path=output/'camera_refinement.npz'
    np.savez_compressed(path,P_refined=P,P_video=P,P_baseline=baseline,K=K,R=refined_R,t=refined_t,H=H,image_homography_ref_to_video=H,camera_K=K,camera_R_wc=refined_R,camera_t_wc=refined_t,calibration_valid=g['calibration_valid'],refinement_accepted=np.array(accepted),static_points_world=worlds,static_observed_pixels_xy=observed,static_holdout=holdouts,static_point_valid=pointvalid,frame_camera_valid=g['camera_drift_all_frames_valid'],physical_calibration_proven=np.array(False),source_geometry_sha256=np.array(digest(source/'geometry.npz')))
    report={'status':'static_only_refinement_evaluated','method':'CPU reference-to-video LK with forward/backward filtering, mesh ray-depth anchors on rendered static table/floor, fixed K/H, bounded 6DOF per-camera correction and priors, spatial-cell holdout acceptance. No dynamic object/body points fitted.','warning':'Reference H was previously estimated using SIFT on this first frame. New LK holdout checks incremental physical pose correction only; it is not an independently acquired physical calibration. Source mesh/AI image errors remain.','conventions':'world XYZ metres Zup; OpenCV world-to-camera R/t; video P=H K [R|t]; fixed cameras all121 frames','physical_metric_ground_truth':False,'views':records,'known_foot_points_used_for_camera_fit':False,'dynamic_box_points_used_for_camera_fit':False,'output_sha256':digest(path),'source_geometry_sha256':digest(source/'geometry.npz'),'elapsed_seconds':time.time()-start}
    rp=ROOT/'reports/user-stereo-refinement.json';rp.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
