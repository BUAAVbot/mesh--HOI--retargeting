#!/usr/bin/env python3
"""CPU-only co-visible physical-point identities for real two-view tracking.

Extends geometry.npz atomically. Independent per-view point indices are never
treated as stereo correspondences. All shared IDs refer to one mesh point.
"""
from pathlib import Path
import hashlib
import json
import os

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    os.environ.setdefault('OMP_NUM_THREADS','6')
    import numpy as np
    import cv2
    cv2.setNumThreads(4)
    out=ROOT/'outputs/user_reconstruction';path=out/'geometry.npz'
    with np.load(path) as data: arrays={k:data[k] for k in data.files}
    scene=ROOT/'data/video_generation/candidate_scene'
    reference=cv2.imread(str(scene/'initial_two_view.png'))
    cap=cv2.VideoCapture(str(scene/'generated.mp4'));ok,frame=cap.read();cap.release()
    if not ok:raise RuntimeError('Missing first video frame')
    world=arrays['robot_points_world'].reshape(-1,3)
    links=arrays['robot_points_link_index'].reshape(-1)
    names=arrays['link_names'];body=np.asarray([not any(token in str(names[i]) for token in ('shoulder','elbow','wrist','hand')) for i in links])
    source_ids=np.arange(len(world),dtype=np.int64)
    world=world[body];links=links[body];source_ids=source_ids[body]
    # Avoid near duplicate samples from the two independent reference views.
    _,unique=np.unique(np.round(world/.003).astype(np.int64),axis=0,return_index=True)
    unique=np.sort(unique);world=world[unique];links=links[unique];source_ids=source_ids[unique]
    K=arrays['K'];R=arrays['R'];t=arrays['t'];H=arrays['image_homography_ref_to_video'];P=arrays['P_video']
    ref_uv=[];video_uv=[];valid=[];fb_errors=[];pose_shifts=[];view_reports=[]
    segmentation=out/'segmentation.npz';input_path=out/'input.npz'
    if not segmentation.exists() or not input_path.exists():raise RuntimeError('Require observed SAM masks and input watermark validity')
    with np.load(segmentation) as d: sam=d['robot'][0];sam_observation=d['observation_valid'][0]
    with np.load(input_path) as d: wm=d['watermark_invalid'][0]
    for view in range(2):
        xc=world@R[view].T+t[view]
        uv=xc@K[view].T;uv=uv[:,:2]/uv[:,2:]
        ix=np.floor(uv[:,0]).astype(int);iy=np.floor(uv[:,1]).astype(int)
        inside=(ix>=2)&(ix<638)&(iy>=2)&(iy<718)&(xc[:,2]>0)
        ix=ix.clip(0,639);iy=iy.clip(0,719)
        ray_z=arrays['reference_depth_z_m'][view,iy,ix]
        visible=inside&arrays['reference_robot_mask'][view,iy,ix]&(np.abs(ray_z-xc[:,2])<.008)
        homogeneous=np.column_stack((uv,np.ones(len(uv))))@H[view].T
        mapped=(homogeneous[:,:2]/homogeneous[:,2:]).astype(np.float32)
        gray=cv2.cvtColor(reference[:,view*640:(view+1)*640],cv2.COLOR_BGR2GRAY)
        target=cv2.cvtColor(frame[:,view*640:(view+1)*640],cv2.COLOR_BGR2GRAY)
        tracked,s,e=cv2.calcOpticalFlowPyrLK(gray,target,uv.astype(np.float32).reshape(-1,1,2),mapped.reshape(-1,1,2).copy(),winSize=(19,19),maxLevel=4,flags=cv2.OPTFLOW_USE_INITIAL_FLOW,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.01))
        back,bs,_=cv2.calcOpticalFlowPyrLK(target,gray,tracked,None,winSize=(19,19),maxLevel=4)
        fb=np.linalg.norm(back[:,0]-uv,axis=1);shift=np.linalg.norm(tracked[:,0]-mapped,axis=1)
        query=tracked[:,0];x=np.rint(query[:,0]).astype(int);y=np.rint(query[:,1]).astype(int)
        in_image=(x>=2)&(x<638)&(y>=2)&(y<718);x=x.clip(0,639);y=y.clip(0,719)
        eroded=cv2.erode((sam[view]&~wm[view]&sam_observation[view]).astype(np.uint8),np.ones((3,3),np.uint8))>0
        samvalid=in_image&eroded[y,x]
        accept=visible&samvalid&(s[:,0]>0)&(bs[:,0]>0)&(fb<1.5)&(e[:,0]<25)&(shift<3.5)&bool(arrays['calibration_valid'][view])
        ref_uv.append(uv);video_uv.append(query);valid.append(accept);fb_errors.append(fb);pose_shifts.append(shift)
        view_reports.append({'view':view,'reference_co_visibility_candidates':int(visible.sum()),'sam_frame0_valid_candidates':int(samvalid.sum()),'stable_lk_and_visibility_candidates':int(accept.sum())})
    ref_uv=np.asarray(ref_uv);video_uv=np.asarray(video_uv);valid=np.asarray(valid)
    center=-R[0].T@t[0];homogeneous_center=np.r_[center,1.]
    epipole=P[1]@homogeneous_center
    ex=np.asarray([[0,-epipole[2],epipole[1]],[epipole[2],0,-epipole[0]],[-epipole[1],epipole[0],0]])
    F=ex@P[1]@np.linalg.pinv(P[0]);F/=np.linalg.norm(F)
    x0=np.column_stack((video_uv[0],np.ones(len(world))));x1=np.column_stack((video_uv[1],np.ones(len(world))))
    l1=x0@F.T;l0=x1@F
    algebraic=(x1*l1).sum(axis=1)
    sampson=np.abs(algebraic)/np.sqrt((l1[:,:2]**2).sum(axis=1)+(l0[:,:2]**2).sum(axis=1))
    keep=valid.all(axis=0)&(sampson<2.)
    possible=np.flatnonzero(keep)
    # Farthest-point image sampling keeps distinct locations with common IDs.
    chosen=[]
    if len(possible):
        coords=np.column_stack((video_uv[0,possible],video_uv[1,possible]))
        index=int(np.argmin(np.linalg.norm(coords-coords.mean(axis=0),axis=1)))
        nearest=np.full(len(possible),np.inf)
        for _ in range(min(256,len(possible))):
            chosen.append(int(possible[index]));nearest=np.minimum(nearest,((coords-coords[index])**2).sum(axis=1));nearest[np.asarray([int(np.where(possible==i)[0][0]) for i in chosen])]=-1
            index=int(np.argmax(nearest))
    chosen=np.asarray(chosen,dtype=np.int64);count=len(chosen)
    arrays.update(shared_robot_points_world=world[chosen].astype(np.float32),shared_robot_points_reference_pixels_xy=ref_uv[:,chosen].astype(np.float32),shared_robot_points_video_pixels_xy=video_uv[:,chosen].astype(np.float32),shared_robot_points_initial_valid=valid[:,chosen],shared_link_index=links[chosen],shared_point_source_ids=source_ids[chosen],shared_robot_points_initial_fb_error_px=np.asarray(fb_errors)[:,chosen],shared_robot_points_initial_pose_shift_px=np.asarray(pose_shifts)[:,chosen],shared_robot_points_sampson_error_px=sampson[chosen],stereo_fundamental_matrix_video=F)
    temporary=path.with_name('geometry.shared.pending.npz');np.savez_compressed(temporary,**arrays);temporary.replace(path)
    reportpath=ROOT/'reports/user-geometry.json';report=json.loads(reportpath.read_text())
    record={'status':'shared_point_initialization_prepared' if count else 'blocked_no_trustworthy_shared_points','shared_count':count,'body_candidate_world_points':len(world),'after_dual_visibility_lk_sam_and_epipolar_filter':int(keep.sum()),'view_counts':view_reports,'physical_identity_policy':'One reference mesh world point is projected and tracked into both views; never join independent per-view point indices','reference_depth_visibility_tolerance_m':.008,'initial_lk_fb_tolerance_px':1.5,'initial_pose_shift_relative_to_background_H_tolerance_px':3.5,'initial_sampson_tolerance_px':2.,'exclude_links_containing':['shoulder','elbow','wrist','hand'],'segmentation_sha256':digest(segmentation),'input_npz_sha256':digest(input_path),'watermark_observation':'SAM robot frame0 intersected with observation_valid frame0 and NOT input watermark_invalid frame0','generated_video_metric_truth':False,'note':'These are conditional common material-point initializations, not measured 3D truth. Root must re-test epipolar/reprojection/positiveZ throughout tracking.'}
    report['shared_points']=record;report['geometry_sha256']=digest(path)
    report['array_fields']={k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in arrays.items()}
    tmp=reportpath.with_suffix('.pending.json');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(reportpath)
    for view in range(2):
        image=frame[:,view*640:(view+1)*640].copy()
        for point in video_uv[view,chosen]:cv2.circle(image,tuple(np.rint(point).astype(int)),2,(60,245,80),-1)
        cv2.imwrite(str(out/f'shared_geometry_points_view{view}.jpg'),image)
    print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':main()
