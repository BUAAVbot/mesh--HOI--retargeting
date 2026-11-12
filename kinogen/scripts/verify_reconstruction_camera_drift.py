#!/usr/bin/env python3
"""Check observed static-background motion in every video frame using CPU LK/RANSAC."""
from pathlib import Path
import hashlib
import json
import os
ROOT=Path(__file__).resolve().parents[1]


def main():
    os.environ.setdefault('OMP_NUM_THREADS','6')
    import cv2
    import numpy as np
    cv2.setNumThreads(4)
    out=ROOT/'outputs/user_reconstruction';geometry=out/'geometry.npz'
    with np.load(geometry) as d:arrays={k:d[k] for k in d.files}
    with np.load(out/'segmentation.npz') as d:foreground=d['robot'][0]|d['box'][0]
    with np.load(out/'input.npz') as d:invalid=d['watermark_invalid']
    authored=cv2.imread(str(ROOT/'data/video_generation/candidate_scene/initial_two_view.png'))
    rendered_valid=[]
    for view in range(2):
        gray=cv2.cvtColor(authored[:,640*view:640*(view+1)],cv2.COLOR_BGR2GRAY)
        rendered=cv2.erode((gray<230).astype(np.uint8),np.ones((5,5),np.uint8))>0
        uv=arrays['static_pixels_xy'][view];x=np.floor(uv[:,0]).astype(int).clip(0,639);y=np.floor(uv[:,1]).astype(int).clip(0,719)
        rendered_valid.append(rendered[y,x])
    arrays['static_rendered_surface_valid']=np.asarray(rendered_valid)
    arrays['static_valid'] &= arrays['static_rendered_surface_valid']
    cap=cv2.VideoCapture(str(ROOT/'data/video_generation/candidate_scene/generated.mp4'))
    frames=[]
    while True:
        ok,f=cap.read()
        if not ok:break
        frames.append(f)
    cap.release()
    matrices=np.tile(np.eye(3),(len(frames),2,1,1));valid=np.zeros((len(frames),2),dtype=bool)
    shifts=np.full((len(frames),2),np.nan);fit_errors=np.full_like(shifts,np.nan);counts=np.zeros((len(frames),2),dtype=np.int32)
    records=[]
    for view in range(2):
        reference=cv2.cvtColor(frames[0][:,640*view:640*(view+1)],cv2.COLOR_BGR2GRAY)
        mask=cv2.warpPerspective(arrays['reference_static_mask'][view].astype(np.uint8),arrays['image_homography_ref_to_video'][view],(640,720),flags=cv2.INTER_NEAREST)>0
        mask &= ~cv2.dilate((foreground[view]|invalid[0,view]).astype(np.uint8),np.ones((21,21),np.uint8)).astype(bool)
        mask[:140]=False
        points=cv2.goodFeaturesToTrack(reference,maxCorners=500,qualityLevel=.01,minDistance=8,mask=(mask*255).astype(np.uint8))
        if points is None or len(points)<20:raise RuntimeError('Too few static features')
        for index,frame in enumerate(frames):
            target=cv2.cvtColor(frame[:,640*view:640*(view+1)],cv2.COLOR_BGR2GRAY)
            tracked,status,error=cv2.calcOpticalFlowPyrLK(reference,target,points,None,winSize=(21,21),maxLevel=4)
            p=tracked[:,0];x=np.rint(p[:,0]).astype(int);y=np.rint(p[:,1]).astype(int)
            inside=(x>=1)&(x<639)&(y>=1)&(y<719);x=x.clip(0,639);y=y.clip(0,719)
            accept=(status[:,0]>0)&inside&~invalid[index,view,y,x]
            a,b=points[:,0][accept],p[accept]
            if len(a)<12:continue
            affine,keep=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=2.0,maxIters=4000,confidence=.999)
            if affine is None:continue
            keep=keep[:,0]>0;count=int(keep.sum())
            if count<12:continue
            matrices[index,view,:2]=affine
            motion=np.linalg.norm(b[keep]-a[keep],axis=1)
            predicted=np.column_stack((a[keep],np.ones(count)))@affine.T
            residual=np.linalg.norm(predicted-b[keep],axis=1)
            shifts[index,view]=np.median(motion);fit_errors[index,view]=np.median(residual);counts[index,view]=count
            valid[index,view]=count>=50 and count/max(len(a),1)>.5 and fit_errors[index,view]<2.0
        records.append({'view':view,'tested_frames':len(frames),'valid_fits':int(valid[:,view].sum()),'initial_static_features':len(points),'median_static_motion_px':float(np.nanmedian(shifts[:,view])),'max_frame_median_static_motion_px':float(np.nanmax(shifts[:,view])),'max_frame_median_ransac_error_px':float(np.nanmax(fit_errors[:,view])),'minimum_ransac_inliers':int(counts[:,view].min()),'fixed_camera_image_consistency_passed':bool(valid[:,view].all() and np.nanmax(shifts[:,view])<3.0)})
    arrays.update(camera_affines_first_to_all_frames=matrices,camera_drift_all_frames_valid=valid,camera_drift_all_frames_median_motion_px=shifts,camera_drift_all_frames_median_fit_error_px=fit_errors,camera_drift_all_frames_inliers=counts)
    tmp=geometry.with_name('geometry.drift.pending.npz');np.savez_compressed(tmp,**arrays);tmp.replace(geometry)
    reportpath=ROOT/'reports/user-geometry.json';report=json.loads(reportpath.read_text())
    report['all_frames_camera_drift']={'method':'Static first-frame corners; direct first-to-each-frame CPU LK, per-frame input watermark exclusion, RANSAC 2D similarity. No robot or box initial features.','frames_checked':len(frames),'physical_calibration_proven':False,'views':records,'note':'Small static image motion supports fixed-camera consistency in the generated video; it does not establish metric camera truth or rigid object consistency.'}
    report['static_rendered_surface_filter']={'policy':'Require authored gray<230 eroded by 5x5 to reject near-white blank/raster-clipped floor background and thin grid-only pixels. Ray-plane existence alone is insufficient.','retained_per_view':arrays['static_valid'].sum(axis=1).tolist(),'candidate_per_view':arrays['static_valid'].shape[1],'field':'static_rendered_surface_valid and static_valid'}
    report['geometry_sha256']=hashlib.sha256(geometry.read_bytes()).hexdigest()
    report['array_fields']={k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in arrays.items()}
    tmp=reportpath.with_suffix('.drift.pending.json');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(reportpath)
    print(json.dumps(report['all_frames_camera_drift'],indent=2),flush=True)


if __name__=='__main__':main()
