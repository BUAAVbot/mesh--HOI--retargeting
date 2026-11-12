"""Real TAPNext++ tracks and conditional stereo reconstruction of shared points."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/user_reconstruction"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "6":
        raise RuntimeError("Use the coordinated idle physical GPU6 only")
    import cv2
    import numpy as np
    import torch
    from smoke_vision import save_video
    cv2.setNumThreads(4)
    sys.path.insert(0, str(ROOT / "dependencies/tapnet"))
    from tapnet.tapnextpp.votsp2026.model import TAPNextPP
    started = time.monotonic()
    data = np.load(OUT / "input.npz")
    geo = np.load(OUT / "geometry.npz")
    segmentation = np.load(OUT / "segmentation.npz")
    rgb, invalid = data["rgb"], data["watermark_invalid"]
    T, V, H, W, _ = rgb.shape
    M = len(geo["shared_robot_points_world"])
    robot_masks = segmentation["robot"]
    masks = np.asarray([[cv2.dilate(m.astype(np.uint8), np.ones((5,5),np.uint8))>0 for m in row] for row in robot_masks])
    tracks = np.full((T,V,650,2), np.nan, np.float32)
    visible = np.zeros((T,V,650),bool)
    accepted = np.zeros_like(visible)
    cycle = np.full((T,V,650),np.nan,np.float32)
    shared = np.full((T,V,M,2),np.nan,np.float32)
    shared_valid = np.zeros((T,V,M),bool)
    shared_cycle = np.full((T,V,M),np.nan,np.float32)
    checkpoint = ROOT / "weights/tapnextpp/tapnextpp_ckpt.pt"
    model = TAPNextPP.from_checkpoint(checkpoint, device="cuda:0",input_resolution=256)
    counts=[]

    def observed(points, shown, frame, view):
        ok = np.asarray(shown,dtype=bool) & np.isfinite(points).all(axis=1)
        ok &= (points[:,0]>=0)&(points[:,0]<W)&(points[:,1]>=0)&(points[:,1]<H)
        xy = np.round(np.nan_to_num(points)).astype(np.int32)
        xy[:,0]=np.clip(xy[:,0],0,W-1);xy[:,1]=np.clip(xy[:,1],0,H-1)
        ok &= masks[frame,view,xy[:,1],xy[:,0]] & ~invalid[frame,view,xy[:,1],xy[:,0]]
        return ok

    with torch.inference_mode():
        for view in range(2):
            q = geo["robot_points_video_initial_xy"][view].astype(np.float32)
            source_valid = geo["robot_points_video_query_valid"][view].astype(bool)
            source_valid &= observed(q,np.ones(650,bool),0,view)
            indices = np.flatnonzero(source_valid)
            sq = geo["shared_robot_points_video_pixels_xy"][view].astype(np.float32)
            initial_shared = geo["shared_robot_points_initial_valid"][view] & observed(sq,np.ones(M,bool),0,view)
            si = np.flatnonzero(initial_shared)
            queries = np.concatenate((q[indices],sq[si]),axis=0)
            if not len(queries):
                raise RuntimeError(f"No valid robot queries in view{view}")
            state=None; positions=[];flags=[]
            for frame in range(T):
                pts,vis,state = model.track_frame(rgb[frame,view,...,::-1].copy(),queries if frame==0 else None,state=state)
                positions.append(pts.copy());flags.append(observed(pts,vis,frame,view))
                tracks[frame,view,indices]=pts[:len(indices)]
                visible[frame,view,indices]=np.asarray(vis[:len(indices)],dtype=bool)
                accepted[frame,view,indices]=flags[-1][:len(indices)]
                shared[frame,view,si]=pts[len(indices):]
                shared_valid[frame,view,si]=flags[-1][len(indices):]
                if frame%30==0:print(f'TAP view{view} frame{frame}/{T} queries={len(queries)}',flush=True)
            positions=np.stack(positions);flags=np.stack(flags)
            last_indices=np.flatnonzero(flags[-1])
            reverse_error=np.full((T,len(queries)),np.nan,np.float32)
            reverse_visible=np.zeros((T,len(queries)),bool)
            state=None
            if len(last_indices):
                for frame in range(T-1,-1,-1):
                    pts,vis,state=model.track_frame(rgb[frame,view,...,::-1].copy(),positions[-1,last_indices] if frame==T-1 else None,state=state)
                    reverse_error[frame,last_indices]=np.linalg.norm(positions[frame,last_indices]-pts,axis=1)
                    reverse_visible[frame,last_indices]=observed(pts,vis,frame,view)
            cycle[:,view,indices]=reverse_error[:,:len(indices)]
            shared_cycle[:,view,si]=reverse_error[:,len(indices):]
            # Only shared points satisfying independently run forward/backward
            # tracking are allowed into stereo reconstruction.
            shared_valid[:,view,si] &= reverse_visible[:,len(indices):] & (reverse_error[:,len(indices):]<=5.)
            counts.append({'view':view,'independent_queries':len(indices),'shared_queries':len(si),'reverse_queries':len(last_indices)})
            del state
            torch.cuda.empty_cache()
    del model
    torch.cuda.empty_cache()
    xyz=np.full((T,M,3),np.nan,np.float32)
    stereo_valid=np.zeros((T,M),bool)
    reprojection=np.full((T,2,M),np.nan,np.float32)
    epipolar=np.full((T,M),np.nan,np.float32)
    F=geo['stereo_fundamental_matrix_video']
    P=geo['P_video']
    for frame in range(T):
        ii=np.flatnonzero(shared_valid[frame,0]&shared_valid[frame,1])
        if not len(ii):continue
        a,b=shared[frame,0,ii],shared[frame,1,ii]
        ah=np.column_stack((a,np.ones(len(ii))));bh=np.column_stack((b,np.ones(len(ii))))
        Fa=ah@F.T;Ftb=bh@F
        sampson=np.abs(np.sum(bh*Fa,axis=1))/np.sqrt(Fa[:,0]**2+Fa[:,1]**2+Ftb[:,0]**2+Ftb[:,1]**2+1e-12)
        points=cv2.triangulatePoints(P[0],P[1],a.T,b.T).T
        points=points[:,:3]/points[:,3:4]
        good=np.isfinite(points).all(axis=1)&(sampson<=2.)
        epipolar[frame,ii]=sampson
        for view,uv in ((0,a),(1,b)):
            projected=np.column_stack((points,np.ones(len(points))))@P[view].T
            projected=projected[:,:2]/projected[:,2:3]
            err=np.linalg.norm(projected-uv,axis=1)
            reprojection[frame,view,ii]=err
            z=(points@geo['camera_R_wc'][view].T+geo['camera_t_wc'][view])[:,2]
            good &= (err<=3.)&(z>.1)
        xyz[frame,ii]=points;stereo_valid[frame,ii]=good
    np.savez_compressed(OUT/'tracking.npz',positions_xy=tracks,model_visible=visible,observation_valid=accepted,
        forward_backward_error_px=cycle,shared_positions_xy=shared,shared_observation_valid=shared_valid,
        shared_forward_backward_error_px=shared_cycle,query_reference_world=geo['robot_points_world'],
        query_world_anchor_valid=geo['robot_points_initial_alignment_valid'],timestamps=data['timestamps'],
        video_sha256=data['video_sha256'])
    np.savez_compressed(OUT/'sparse_stereo.npz',points_world=xyz,valid=stereo_valid,reprojection_error_px=reprojection,
        epipolar_sampson_px=epipolar,timestamps=data['timestamps'],reference_points_world=geo['shared_robot_points_world'],
        shared_link_index=geo['shared_link_index'],note=np.asarray('Conditional shared foot-point stereo; not full-body reconstruction'))
    preview=[]
    for frame in range(T):
        views=[]
        for view in range(2):
            im=rgb[frame,view,...,::-1].copy()
            for j in np.flatnonzero(accepted[frame,view])[::2]:
                xy=np.round(tracks[frame,view,j]).astype(int)
                cv2.circle(im,tuple(xy),2,(230,180,40),-1)
            for j in np.flatnonzero(shared_valid[frame,view]):
                xy=np.round(shared[frame,view,j]).astype(int)
                cv2.circle(im,tuple(xy),3,(50,240,60),-1)
            im[invalid[frame,view]]=(im[invalid[frame,view]].astype(np.float32)*.4).astype(np.uint8)
            cv2.putText(im,f'View {view} | frame {frame+1}/{T}',(10,24),0,.6,(255,255,255),1)
            views.append(im)
        preview.append(np.concatenate(views,axis=1))
    video=save_video(ROOT/'outputs/previews/user_tracking.mp4',preview,24)
    err=reprojection[np.broadcast_to(stereo_valid[:,None,:],reprojection.shape)]
    report={'created_at_utc':datetime.now(timezone.utc).isoformat(),'status':'real_tracking_and_conditional_sparse_stereo_completed',
        'input_video_sha256':str(data['video_sha256']),'input_npz_sha256':digest(OUT/'input.npz'),
        'geometry_sha256':digest(OUT/'geometry.npz'),'segmentation_sha256':digest(OUT/'segmentation.npz'),
        'checkpoint_sha256':digest(checkpoint),'frames':T,'views':2,'query_counts':counts,
        'accepted_2d_observations':int(accepted.sum()),'shared_point_count':M,
        'valid_stereo_observations':int(stereo_valid.sum()),'total_possible_stereo_observations':int(stereo_valid.size),
        'median_valid_reprojection_px':float(np.median(err)) if len(err) else None,
        'stereo_has_observations_all_frames':bool(stereo_valid.any(axis=1).all()),
        'shared_point_scope':'Strictly initialized common foot/ankle surface points; no arm or whole-body 3D claim',
        'calibration':'Conditional physical cameras with measured reference-to-video image similarity',
        'watermark_observations_excluded':True,'preview':video,'elapsed_seconds':round(time.monotonic()-started,2),
        'training':False,'full_kinogen_pipeline':False}
    (ROOT/'reports/user-tracking.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
