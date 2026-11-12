#!/usr/bin/env python3
"""CPU candidate metric calibration of real VDA predictions; never metric truth.

Requires final input, SAM and depth observation masks to agree exactly. Fits
static rendered table/ground ray observations with spatial-group holdout, then
exports conditional depth candidates and separate same-frame fused point clouds.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/user_reconstruction')
    parser.add_argument('--preview-dir',type=Path,default=ROOT/'outputs/previews')
    parser.add_argument('--cloud-step',type=int,default=4)
    parser.add_argument('--holdout-fraction',type=float,default=.25)
    args=parser.parse_args()
    os.environ.setdefault('OMP_NUM_THREADS','6')
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import numpy as np
    import cv2
    from scipy.optimize import least_squares
    cv2.setNumThreads(4)
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True);args.preview_dir.mkdir(parents=True,exist_ok=True)
    paths={name:out/(name+'.npz') for name in ('geometry','input','segmentation','depth')}
    with np.load(paths['geometry']) as d:g={k:d[k] for k in d.files}
    with np.load(paths['input']) as d:
        invalid=d['watermark_invalid'];timestamps=d['timestamps'];fps=float(d['fps']);video_sha=str(d['video_sha256'])
    with np.load(paths['segmentation']) as d:
        robot=d['robot'];box=d['box'];sam_valid=d['observation_valid'];sam_video_sha=str(d['source_video_sha256'])
    with np.load(paths['depth']) as d:
        relative=d['relative_depth'];depth_valid=d['observation_valid'];depth_video_sha=str(d['source_video_sha256']);checkpoint_sha=str(d['checkpoint_sha256'])
    if not (sam_video_sha==depth_video_sha==video_sha):raise RuntimeError('Input/SAM/VDA video source mismatch')
    if not (relative.shape==invalid.shape==robot.shape==box.shape==depth_valid.shape==sam_valid.shape):raise RuntimeError('Input array shape mismatch')
    if not (np.array_equal(depth_valid,~invalid) and np.array_equal(sam_valid,~invalid)):
        raise RuntimeError('Stale watermark validity: wait for final input/SAM/depth observation_valid refresh')
    if not g['calibration_valid'].all():raise RuntimeError('Reference camera compatibility is not accepted; cannot silently metric-calibrate')
    frame_count,views,height,width=relative.shape
    if (frame_count,views,height,width)!=(121,2,720,640):raise RuntimeError('This script expects the inspected two-view video')
    reference=cv2.imread(str(ROOT/'data/video_generation/candidate_scene/initial_two_view.png'))
    affines=g.get('camera_affines_first_to_all_frames',np.tile(np.eye(3),(frame_count,2,1,1)))
    candidate=np.full(relative.shape,np.nan,dtype=np.float32)
    candidate_valid=np.zeros(relative.shape,dtype=bool)
    selected_models=[];fit_reports=[];static_observed=[];static_estimated=[];static_validity=[];static_holdouts=[];scene_support=[]
    rng=np.random.default_rng(20261021)

    def warp(uv,H):
        p=np.column_stack((uv,np.ones(len(uv))))@H.T
        return p[:,:2]/p[:,2:]
    def model_predict(kind,coef,d):
        mapped=coef[0]*d+coef[1]
        if kind=='inverse_z':
            return np.divide(1.,mapped,out=np.full_like(mapped,np.nan,dtype=np.float64),where=mapped>1e-6)
        return mapped
    def metrics(pred,target,valid):
        accept=valid&np.isfinite(pred)&(pred>.2)&(pred<20.)
        error=np.abs(pred[accept]-target[accept])
        return {'observations':int(valid.sum()),'positive_finite_predictions':int(accept.sum()),'valid_fraction':float(accept.sum()/max(valid.sum(),1)),
                'median_abs_error_m':float(np.median(error)) if len(error) else None,'p90_abs_error_m':float(np.percentile(error,90)) if len(error) else None,
                'rmse_m':float(np.sqrt(np.mean(error**2))) if len(error) else None,'bias_m':float(np.median(pred[accept]-target[accept])) if len(error) else None}

    for view in range(views):
        static_uv=g['static_video_pixels_xy'][view];known_z=g['static_camera_depth_z_m'][view]
        static_base=g['static_valid'][view].astype(bool).copy()
        if 'static_rendered_surface_valid' in g:static_base &= g['static_rendered_surface_valid'][view]
        known_z_valid=np.isfinite(known_z)&(known_z>.2)&(known_z<20.)
        static_base &= known_z_valid
        samples=np.full((frame_count,len(static_uv)),np.nan,dtype=np.float64);observation=np.zeros(samples.shape,dtype=bool)
        for frame in range(frame_count):
            uv=warp(static_uv,affines[frame,view]);x=np.floor(uv[:,0]).astype(int);y=np.floor(uv[:,1]).astype(int)
            inside=(x>=1)&(x<width-2)&(y>=1)&(y<height-2);x=x.clip(0,width-1);y=y.clip(0,height-1)
            # Coordinates use image-corner origin, hence -0.5 for cv2 array-index remapping.
            value=cv2.remap(relative[frame,view],(uv[:,0]-.5).astype(np.float32).reshape(1,-1),(uv[:,1]-.5).astype(np.float32).reshape(1,-1),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=float('nan'))[0]
            foreground=cv2.dilate((robot[frame,view]|box[frame,view]).astype(np.uint8),np.ones((7,7),np.uint8))>0
            good=static_base&inside&~invalid[frame,view,y,x]&depth_valid[frame,view,y,x]&sam_valid[frame,view,y,x]&~foreground[y,x]&np.isfinite(value)
            samples[frame]=value;observation[frame]=good
        groups=(np.floor(g['static_pixels_xy'][view,:,0]/64).astype(int)+32*np.floor(g['static_pixels_xy'][view,:,1]/64).astype(int)+1024*g['static_surface_kind'][view])
        point_holdout=np.zeros(len(static_uv),dtype=bool);held_groups=[]
        for kind in (2,3):
            available=np.unique(groups[static_base&(g['static_surface_kind'][view]==kind)])
            if len(available)<2:continue
            rng.shuffle(available);selected=available[:max(1,int(round(len(available)*args.holdout_fraction)))]
            held_groups.extend(selected.tolist());point_holdout|=np.isin(groups,selected)
        target=np.broadcast_to(known_z,samples.shape)
        training=observation&~point_holdout[None];heldout=observation&point_holdout[None]
        if training.sum()<500 or heldout.sum()<200:raise RuntimeError(f'Insufficient spatially held-out observations in view {view}')
        train_x=samples[training];train_z=target[training]
        mu=float(np.median(train_x));sigma=max(float(np.std(train_x)),1e-8);normalized=(train_x-mu)/sigma
        alternatives=[]
        for kind in ('linear_z','inverse_z'):
            target_linear=train_z if kind=='linear_z' else 1/train_z
            initial=np.linalg.lstsq(np.column_stack((normalized,np.ones(len(normalized)))),target_linear,rcond=None)[0]
            def residual(params):
                z=params[0]*normalized+params[1]
                if kind=='inverse_z':z=1/np.maximum(z,1e-3)
                return z-train_z
            fit=least_squares(residual,initial,loss='soft_l1',f_scale=.05,max_nfev=100)
            coef=np.asarray([fit.x[0]/sigma,fit.x[1]-fit.x[0]*mu/sigma])
            prediction=model_predict(kind,coef,samples)
            train_metrics=metrics(prediction,target,training);test_metrics=metrics(prediction,target,heldout)
            per_frame=[]
            for frame in range(frame_count):
                r=metrics(prediction[frame],target[frame],heldout[frame]);per_frame.append({'frame':frame,**r})
            temporal_std=[]
            for point in np.flatnonzero(point_holdout&static_base):
                keep=observation[:,point]&np.isfinite(prediction[:,point])&(prediction[:,point]>.2)&(prediction[:,point]<20)
                if keep.sum()>=frame_count//2:temporal_std.append(float(np.std(prediction[keep,point])))
            temporal=float(np.median(temporal_std)) if temporal_std else None
            accepted=bool(fit.success and test_metrics['valid_fraction']>.98 and test_metrics['median_abs_error_m'] is not None and test_metrics['median_abs_error_m']<=.12 and test_metrics['p90_abs_error_m']<=.30)
            med=test_metrics['median_abs_error_m'];p90=test_metrics['p90_abs_error_m']
            score=(med if med is not None else 1e6)+.25*(p90 if p90 is not None else 1e6)+.1*(temporal or 0.)+(0 if test_metrics['valid_fraction']>.98 else 100.)
            alternatives.append({'model':kind,'equation':'Z_m=a*d+b' if kind=='linear_z' else '1/Z_m=a*d+b','coefficients_a_b':coef.tolist(),'optimizer_success':bool(fit.success),'train':train_metrics,'spatial_group_heldout':test_metrics,'heldout_static_point_median_temporal_std_m':temporal,'per_frame_heldout':per_frame,'static_candidate_fit_accepted':accepted,'selection_score':float(score)})
        selected=min(alternatives,key=lambda item:item['selection_score']);selected_models.append(selected)
        mapped=model_predict(selected['model'],np.asarray(selected['coefficients_a_b']),relative[:,view].astype(np.float64))
        good=depth_valid[:,view]&~invalid[:,view]&np.isfinite(mapped)&(mapped>.2)&(mapped<20)
        candidate[:,view]=np.where(good,mapped,np.nan).astype(np.float32);candidate_valid[:,view]=good
        static_prediction=model_predict(selected['model'],np.asarray(selected['coefficients_a_b']),samples)
        per_surface={}
        for kind,name in ((2,'table'),(3,'ground')):
            per_surface[name]=metrics(static_prediction,target,heldout&(g['static_surface_kind'][view][None]==kind))
        train_range=np.percentile(train_x,[1,99]);dynamic_report={}
        for name,mask in (('robot',robot[:,view]),('box',box[:,view])):
            observed=mask&depth_valid[:,view]&~invalid[:,view]
            raw=relative[:,view][observed]
            predictions=candidate[:,view][observed];finite=np.isfinite(predictions)
            dynamic_report[name]={'observed_pixels':int(observed.sum()),'candidate_positive_finite_fraction':float(finite.mean()) if len(finite) else 0.,'outside_static_training_relative_p01_p99_fraction':float(((raw<train_range[0])|(raw>train_range[1])).mean()) if len(raw) else None,'candidate_depth_p01_p99_m':np.percentile(predictions[finite],[1,99]).tolist() if finite.any() else None,'metric_accuracy_validated':False}
        fit_reports.append({'view':view,'static_ray_candidates':int(static_base.sum()),'valid_observations_all_frames':int(observation.sum()),'static_training_relative_p01_p99':train_range.tolist(),'dynamic_extrapolation_diagnostics':dynamic_report,'spatial_group_size_px':64,'heldout_groups':held_groups,'training_points':int(((~point_holdout)&static_base).sum()),'heldout_points':int((point_holdout&static_base).sum()),'no_same_point_time_leakage':True,'selected_model':selected['model'],'selected_static_fit_accepted':selected['static_candidate_fit_accepted'],'heldout_per_surface':per_surface,'alternatives':alternatives})
        static_observed.append(samples);static_estimated.append(static_prediction);static_validity.append(observation);static_holdouts.append(point_holdout)
        ref_gray=cv2.cvtColor(reference[:,view*width:(view+1)*width],cv2.COLOR_BGR2GRAY)
        rendered=cv2.erode((ref_gray<230).astype(np.uint8),np.ones((5,5),np.uint8))>0
        source_static=g['reference_static_mask'][view]&rendered
        support=np.zeros((frame_count,height,width),dtype=bool)
        for frame in range(frame_count):
            image_transform=affines[frame,view]@g['image_homography_ref_to_video'][view]
            warped=cv2.warpPerspective(source_static.astype(np.uint8),image_transform,(width,height),flags=cv2.INTER_NEAREST)>0
            support[frame]=(warped|robot[frame,view]|box[frame,view])&good[frame]&~invalid[frame,view]
        scene_support.append(support)
        print(f'view {view}: selected {selected["model"]}, heldout median={selected["spatial_group_heldout"]["median_abs_error_m"]} m p90={selected["spatial_group_heldout"]["p90_abs_error_m"]} m accepted={selected["static_candidate_fit_accepted"]}',flush=True)
    scene_support=np.stack(scene_support,axis=1)
    metric_path=out/'metric_depth.npz';temporary=out/'metric_depth.pending.npz'
    np.savez_compressed(temporary,candidate_depth_z_m=candidate,candidate_valid=candidate_valid,scene_surface_observation_valid=scene_support,
                        static_fit_accepted=np.asarray([model['static_candidate_fit_accepted'] for model in selected_models]),strict_metric_calibration_valid=np.zeros(2,dtype=bool),
                        mapping_model=np.asarray([m['model'] for m in selected_models]),mapping_coefficients_a_b=np.asarray([m['coefficients_a_b'] for m in selected_models]),
                        static_relative_samples=np.asarray(static_observed),static_candidate_z_m=np.asarray(static_estimated),static_observation_valid=np.asarray(static_validity),static_holdout_points=np.asarray(static_holdouts),
                        timestamps=timestamps,units=np.asarray('metres_candidate_camera_Z_conditional_static_scene_calibration'),source_video_sha256=np.asarray(video_sha),vda_checkpoint_sha256=np.asarray(checkpoint_sha),geometry_sha256=np.asarray(sha(paths['geometry'])),camera_K=g['K'],camera_R_wc=g['R'],camera_t_wc=g['t'],camera_P_video=g['P_video'],image_homography_ref_to_video=g['image_homography_ref_to_video'])
    temporary.replace(metric_path)
    cloud_records=[];clouds=[]
    cap=cv2.VideoCapture(str(ROOT/'data/video_generation/candidate_scene/generated.mp4'))
    for frame in (0,60,120):
        cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,color=cap.read()
        if not ok:raise RuntimeError('Cannot read cloud color frame')
        points=[];colors=[];view_ids=[]
        for view in range(2):
            yy,xx=np.mgrid[0:height:args.cloud_step,0:width:args.cloud_step]
            keep=scene_support[frame,view,yy,xx]
            uv=np.column_stack((xx[keep]+.5,yy[keep]+.5));z=candidate[frame,view,yy[keep],xx[keep]]
            Htotal=affines[frame,view]@g['image_homography_ref_to_video'][view]
            reference_uv=warp(uv,np.linalg.inv(Htotal))
            camera_ray=np.column_stack((reference_uv,np.ones(len(reference_uv))))@np.linalg.inv(g['K'][view]).T
            camera_xyz=camera_ray*z[:,None];world=(camera_xyz-g['t'][view])@g['R'][view]
            rgb=color[yy[keep],xx[keep]+view*width,::-1]
            points.append(world);colors.append(rgb);view_ids.append(np.full(len(world),view,dtype=np.uint8))
        points=np.concatenate(points).astype(np.float32);colors=np.concatenate(colors).astype(np.uint8);view_ids=np.concatenate(view_ids)
        cloud=out/f'cloud_frame_{frame:03d}.ply'
        header=('ply\nformat binary_little_endian 1.0\ncomment CANDIDATE metres world Z-up; conditional static scene depth mapping, not metric ground truth\n'
                f'comment frame {frame} at {float(timestamps[frame]):.6f} seconds; two simultaneous views fused only\nelement vertex {len(points)}\n'
                'property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar view_id\nend_header\n')
        packed=np.empty(len(points),dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1'),('view_id','u1')])
        for axis,name in enumerate(('x','y','z')):packed[name]=points[:,axis]
        for axis,name in enumerate(('red','green','blue')):packed[name]=colors[:,axis]
        packed['view_id']=view_ids
        with cloud.open('wb') as f:f.write(header.encode('ascii'));f.write(packed.tobytes())
        clouds.append((frame,points,colors))
        cloud_records.append({'frame':frame,'time_seconds':float(timestamps[frame]),'path':str(cloud),'points':len(points),'sha256':sha(cloud),'metre_units_are_conditional':True,'simultaneous_views_only':True})
    cap.release()
    from render_user_metric_cloud_preview import render_preview
    cloud_preview=args.preview_dir/'user_metric_pointcloud.jpg'
    preview_crop=render_preview(clouds,cloud_preview)
    import av
    preview=args.preview_dir/'user_metric_depth.mp4'
    with av.open(str(preview),'w') as container:
        stream=container.add_stream('libx264',rate=int(round(fps)));stream.width=1280;stream.height=720;stream.pix_fmt='yuv420p';stream.options={'crf':'23','preset':'medium'}
        for frame in range(frame_count):
            tiles=[]
            for view in range(2):
                scaled=(np.clip((np.nan_to_num(candidate[frame,view],nan=8.)-.5)/7.5,0,1)*255).astype(np.uint8)
                tile=cv2.applyColorMap(scaled,cv2.COLORMAP_TURBO);tile[~scene_support[frame,view]]=[28,28,28];tile[invalid[frame,view]]=[80,80,80]
                cv2.putText(tile,f'View {view} CANDIDATE Z: .5-8m',(10,24),cv2.FONT_HERSHEY_SIMPLEX,.60,(255,255,255),2,cv2.LINE_AA)
                cv2.putText(tile,f'Static calibration only | frame {frame:03d}',(10,49),cv2.FONT_HERSHEY_SIMPLEX,.52,(255,255,255),1,cv2.LINE_AA)
                tiles.append(tile)
            vf=av.VideoFrame.from_ndarray(np.concatenate(tiles,axis=1),format='bgr24')
            for packet in stream.encode(vf):container.mux(packet)
        for packet in stream.encode():container.mux(packet)
    with av.open(str(preview)) as c:decoded=sum(1 for _ in c.decode(video=0))
    if decoded!=frame_count:raise RuntimeError('Metric-depth preview frame count mismatch')
    report={'created_at_utc':datetime.now(timezone.utc).isoformat(),'status':'candidate_static_metric_calibration_accepted' if all(m['static_candidate_fit_accepted'] for m in selected_models) else 'candidate_static_metric_calibration_failed_quality',
            'strict_metric_reconstruction_valid':False,'training_performed':False,'gpu_used':False,'frames':frame_count,'views':views,'source_video_sha256':video_sha,'inputs':{name:{'path':str(path),'sha256':sha(path)} for name,path in paths.items()},
            'validation':'Each static point and every frame of its time trajectory remain in one spatial 64px tile group. Entire heldout groups never used in parameter fitting; all 121 frames evaluated.',
            'mapping_scope':'One fixed robust affine Z or inverse-Z mapping fitted independently per view over the whole sequence; no per-frame refit that could hide temporal instability.',
            'acceptance_thresholds':{'heldout_median_abs_error_m':.12,'heldout_p90_abs_error_m':.30,'positive_finite_prediction_fraction':.98},'fits':fit_reports,
            'validity_policy':'Final input invalid must equal complement of SAM and depth observation_valid. Static fit uses geometry.static_valid and static_rendered_surface_valid, dilated per-frame SAM robot/box exclusion and watermark exclusion.',
            'camera_pointcloud_policy':'P_video=H@P retained. For frame t, pixel -> inv(A_t@H) -> inv(K) -> camera ray with Z=1 -> candidate camera Z -> (Xcam-t)@R. World metres, Z up. Each PLY fuses exactly two views at one time; no whole-sequence static accumulation.',
            'dynamic_object_metric_validated':False,'physical_calibration_proven':False,'clouds':cloud_records,'pointcloud_preview_crop':preview_crop,
            'output':{'path':str(metric_path),'sha256':sha(metric_path),'dtype':'float32','candidate_shape':list(candidate.shape),'candidate_valid_fraction':float(candidate_valid.mean()),'supported_scene_fraction':float(scene_support.mean())},
            'previews':[{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in (preview,cloud_preview)],
            'limitations':['Static known geometry is an authored candidate scene, not calibrated sensor capture.','Static table/ground fit does not establish metric depth accuracy on generated robot, hands or box.','Reference image contains large-floor clipping appearance artifacts; near-white blank and thin grid-only regions excluded.','Per-view independent mappings may still disagree on corresponding dynamic surfaces.','Spatial heldout accuracy is conditional on generated video preserving those static surfaces.','No dense cross-view correspondence or physical surface fusion was used to certify the clouds.']}
    report_path=ROOT/'reports/user-metric-depth.json';report_path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':report['status'],'strict_metric_reconstruction_valid':False,'output':report['output'],'clouds':cloud_records,'previews':report['previews']},indent=2),flush=True)


if __name__=='__main__':main()
