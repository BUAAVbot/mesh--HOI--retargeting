#!/usr/bin/env python3
"""CPU image-driven 29-DoF G1 + floating-base fit, with explicit uncertainty.

Image evidence: visually reviewed shoulder/elbow/wrist keyframes, measured
bidirectional LK propagation, observed SAM contours, and reliable TAP feet.
Reference q0 is only initialization/weak regularization, never a measurement.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb')as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scale',type=int,default=3,help='CPU silhouette raycast downsampling; 3 = 240x214 approximately')
    parser.add_argument('--outer-iterations',type=int,default=6)
    parser.add_argument('--optimizer-iterations',type=int,default=32)
    parser.add_argument('--max-frames',type=int,default=121)
    parser.add_argument('--resume-existing',action='store_true',help='Refit every frame in time order from saved body poses, still using each actual image')
    parser.add_argument('--temporal-strength',type=float,default=1.,help='Multiplier for the explicit neighbouring-pose prior; not image evidence')
    parser.add_argument('--arm-prior',type=float,default=.35,help='Image-landmark IK pose regularizer')
    parser.add_argument('--refine-frames',default='',help='With resume, comma-separated inclusive frame ranges, e.g. 59:120; other frames only reevaluated')
    args=parser.parse_args();os.environ['CUDA_VISIBLE_DEVICES']='';os.environ.setdefault('OMP_NUM_THREADS','6')
    import numpy as np
    import cv2
    import open3d as o3d
    import trimesh
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation
    from scipy.optimize import minimize,least_squares
    from user_body_kinematics import G1Kinematics,project_points
    cv2.setNumThreads(4)
    out=ROOT/'outputs/user_reconstruction_v2';out.mkdir(parents=True,exist_ok=True)
    report_path=ROOT/'reports/user-body-optimization.json'
    source=ROOT/'outputs/user_reconstruction'
    model=G1Kinematics(ROOT);state0=model.initial_state.copy();T0=model.fk(state0)
    camera_path=out/'camera_refinement.npz'
    if not camera_path.exists():raise RuntimeError('Wait for frozen image-validated v2 cameras')
    with np.load(camera_path)as d:P=d['P_video'];camera_valid=d['frame_camera_valid']
    with np.load(source/'input.npz')as d:rgb=d['rgb'];invalid=d['watermark_invalid'];timestamps=d['timestamps'];video_sha=str(d['video_sha256'])
    with np.load(source/'segmentation.npz')as d:robot=d['robot'];box=d['box'];seg_valid=d['observation_valid']
    if not np.array_equal(seg_valid,~invalid):raise RuntimeError('Stale segmentation validity')
    with np.load(source/'geometry.npz')as d:g={k:d[k]for k in d.files}
    with np.load(source/'tracking.npz')as d:tracking={k:d[k]for k in d.files}
    frames=min(args.max_frames,len(rgb));grays=[[cv2.cvtColor(rgb[f,v],cv2.COLOR_RGB2GRAY)for v in range(2)]for f in range(frames)]
    # Approximate visual landmark annotations on REAL generated-video images.
    # Shoulder roll housing centre, elbow hinge centre, palm centre respectively.
    # Near-side arm only; box-occluded far wrists/elbows are intentionally absent.
    manual={
        0:{0:[[228.,244.],[202.,291.],[213.,314.],[228.,328.]],1:[[404.,245.],[433.,290.],[419.,314.],[404.,328.]]},
        60:{0:[[240.,255.],[239.,311.],[273.,316.],[299.,323.]],1:[[398.,257.],[403.,312.],[372.,317.],[343.,326.]]},
        120:{0:[[232.,243.],[240.,287.],[273.,286.],[299.,291.]],1:[[404.,244.],[401.,286.],[372.,285.],[342.,292.]]},
    }
    point_names=['right_shoulder','right_elbow','right_wrist','right_palm_roi_only','left_shoulder','left_elbow','left_wrist','left_palm_roi_only']
    landmark_links=np.asarray([model.link_index[n]for n in ('right_shoulder_roll_link','right_elbow_link','right_wrist_yaw_link','right_wrist_yaw_link','left_shoulder_roll_link','left_elbow_link','left_wrist_yaw_link','left_wrist_yaw_link')])
    landmark_local=np.zeros((8,3))
    observations=np.full((frames,2,8,2),np.nan);obs_valid=np.zeros((frames,2,8),dtype=bool);obs_spread=np.full((frames,2,8),np.inf)
    provenance={'method':'Approximate visual review of actual unmodified video keyframes by assistant; not q0 projection, SAM nearest pixel, ground truth, or a trained robot keypoint detector.','manual_uncertainty_px':8.,'point_definitions':point_names,'world_model_landmark_local_xyz_m':landmark_local.tolist(),'manual_keyframes':manual,'hidden_arm_observations_fabricated':False}
    for view in range(2):
        candidates=[[]for _ in range(frames)]
        for anchor,views in manual.items():
            if anchor>=frames:continue
            seed=np.asarray(views[view],dtype=np.float32)-.5
            candidates[anchor].append((seed+.5,np.ones(4,dtype=bool),0))
            for direction in (-1,1):
                current=seed.copy();alive=np.ones(4,dtype=bool)
                for target_frame in range(anchor+direction,frames if direction>0 else -1,direction):
                    previous=target_frame-direction
                    nextp,status,err=cv2.calcOpticalFlowPyrLK(grays[previous][view],grays[target_frame][view],current[:,None],None,winSize=(25,25),maxLevel=4)
                    back,backstatus,_=cv2.calcOpticalFlowPyrLK(grays[target_frame][view],grays[previous][view],nextp,None,winSize=(25,25),maxLevel=4)
                    fb=np.linalg.norm(back[:,0]-current,axis=1);current=nextp[:,0]
                    x=np.rint(current[:,0]).astype(int).clip(0,639);y=np.rint(current[:,1]).astype(int).clip(0,719)
                    alive &= (status[:,0]>0)&(backstatus[:,0]>0)&(fb<2.)&(err[:,0]<28)&robot[target_frame,view,y,x]&~invalid[target_frame,view,y,x]
                    candidates[target_frame].append((current.copy()+.5,alive.copy(),abs(target_frame-anchor)))
        for frame in range(frames):
            for local_index in range(4):
                available=[(points[local_index],distance)for points,valid,distance in candidates[frame]if valid[local_index]]
                if not available:continue
                available.sort(key=lambda item:item[1]);point=available[0][0]
                spread=max(float(np.linalg.norm(p-point))for p,_ in available)
                direct=frame in manual
                accepted=direct or spread<8.
                index=local_index+4*view
                if accepted:observations[frame,view,index]=point;obs_valid[frame,view,index]=True;obs_spread[frame,view,index]=spread
    (out/'body_landmark_observations.json').write_text(json.dumps({**provenance,'accepted_per_view':obs_valid.sum(axis=(0,2)).tolist()},indent=2)+'\n')
    np.savez_compressed(out/'body_landmark_observations.npz',positions_xy=observations,valid=obs_valid,propagation_disagreement_px=obs_spread,point_names=np.asarray(point_names),model_link_names=np.asarray([model.link_names[i]for i in landmark_links]),local_xyz_m=landmark_local)

    def outside_hand_domain(frame,view,pixels):
        # Uniform body-only domain: real image wrist/palm evidence, never a
        # projected reference hand. Missing palm tracks must not silently put
        # generated fingers back into a hand-free body silhouette objective.
        keep=np.ones(len(pixels),dtype=bool)
        for index,radius in ((2+4*view,55.),(3+4*view,35.)):
            if obs_valid[frame,view,index]:keep &= np.linalg.norm(pixels-observations[frame,view,index],axis=1)>radius
        return keep

    # Shared physical foot points have true common IDs; arm q0/LK points are not used.
    foot_links=g['shared_link_index'].astype(int);foot_world=g['shared_robot_points_world'];foot_local=np.einsum('nij,nj->ni',np.transpose(T0[foot_links,:3,:3],(0,2,1)),foot_world-T0[foot_links,:3,3])
    foot_xy=tracking['shared_positions_xy'];foot_valid=tracking['shared_observation_valid']
    if foot_xy.shape[:2]!=(len(rgb),2):raise RuntimeError('Unexpected shared TAP shape')
    foot_train=np.arange(len(foot_links))%5!=0
    # Proxy raycast scene, with authored table occlusion. Original meshes unchanged.
    static_meshes=[]
    for size,center in [([.70,.85,.045],[.48,0.,.7575])]+[([.035,.035,.735],[x,y,.3675])for x in (.19,.77)for y in (-.355,.355)]:
        mesh=trimesh.creation.box(extents=size);mesh.apply_translation(center);static_meshes.append(mesh)
    static_mesh=trimesh.util.concatenate(static_meshes);static_vertices=o3d.core.Tensor(np.asarray(static_mesh.vertices,dtype=np.float32));static_faces=o3d.core.Tensor(np.asarray(static_mesh.faces,dtype=np.uint32))
    h,w=720//args.scale,640//args.scale;scale_x,scale_y=640/w,720/h
    xx,yy=np.meshgrid((np.arange(w)+.5)*scale_x,(np.arange(h)+.5)*scale_y)
    rays=[]
    for view in range(2):
        M=P[view,:,:3];center=-np.linalg.solve(M,P[view,:,3]);directions=np.stack((xx,yy,np.ones_like(xx)),axis=-1)@np.linalg.inv(M).T;directions/=np.linalg.norm(directions,axis=-1,keepdims=True)
        rays.append(o3d.core.Tensor(np.concatenate((np.broadcast_to(center,directions.shape),directions),axis=-1).astype(np.float32)))
    body_faces=np.asarray(['hand'not in model.link_names[i]for i in model.proxy_face_links])
    render_faces=model.proxy_faces[body_faces];render_face_links=model.proxy_face_links[body_faces]
    def render(state,frame):
        scene=o3d.t.geometry.RaycastingScene(nthreads=6);scene.add_triangles(o3d.core.Tensor(model.proxy_world(state).astype(np.float32)),o3d.core.Tensor(render_faces));scene.add_triangles(static_vertices,static_faces)
        results=[]
        for view in range(2):
            answer=scene.cast_rays(rays[view],nthreads=6);hit=answer['geometry_ids'].numpy()==0;primitive=answer['primitive_ids'].numpy().astype(np.int64)
            suppress=cv2.resize(box[frame,view].astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)>0
            hit &= ~suppress
            boundary=hit&~cv2.erode(hit.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
            py,px=np.nonzero(boundary)
            keep=outside_hand_domain(frame,view,np.column_stack(((px+.5)*scale_x,(py+.5)*scale_y)));py,px=py[keep],px[keep]
            if len(px)>480:
                take=np.linspace(0,len(px)-1,480).astype(int);py,px=py[take],px[take]
            tri=primitive[py,px];uv=answer['primitive_uvs'].numpy()[py,px]
            local=(model.proxy_vertices[render_faces[tri,0]]*(1-uv.sum(axis=1))[:,None]+model.proxy_vertices[render_faces[tri,1]]*uv[:,0,None]+model.proxy_vertices[render_faces[tri,2]]*uv[:,1,None])
            results.append({'mask':hit,'xy':np.column_stack(((px+.5)*scale_x,(py+.5)*scale_y)),'links':render_face_links[tri],'local':local})
        return results

    observed=[]
    for frame in range(frames):
        views=[]
        for view in range(2):
            mask=robot[frame,view]&~invalid[frame,view]
            contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
            pixels=np.concatenate([c[:,0]for c in contours if len(c)>4]).astype(float)+.5
            # Leave finger topology entirely to the separate hand model fit.
            pixels=pixels[outside_hand_domain(frame,view,pixels)]
            if len(pixels)>650:pixels=pixels[np.linspace(0,len(pixels)-1,650).astype(int)]
            groups=(pixels[:,0]//16).astype(int)+64*(pixels[:,1]//16).astype(int);held=groups%5==0
            domain=outside_hand_domain(frame,view,np.column_stack((xx.ravel(),yy.ravel()))).reshape(h,w)
            views.append({'all':pixels,'train':pixels[~held],'heldout':pixels[held],'mask':cv2.resize(mask.astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)>0,'body_domain':domain})
        observed.append(views)
    lower=np.r_[state0[:3]-[.10,.10,.065],[-.28,-.28,-.28],model.body_limits[:,0]];upper=np.r_[state0[:3]+[.10,.10,.065],[.28,.28,.28],model.body_limits[:,1]]
    q_prior=np.r_[np.full(3,30.),np.full(3,2.),np.full(29,.8)];temporal_weight=np.r_[np.full(3,90.),np.full(3,8.),np.full(29,2.)]*args.temporal_strength
    states=np.full((frames,35),np.nan);transforms=np.full((frames,len(model.link_names),4,4),np.nan);records=[];valid_frames=np.zeros(frames,dtype=bool);observability=np.zeros((frames,29),dtype=bool);uncertainty=np.full((frames,29),np.inf);started=time.monotonic()
    schedule=[f for f in (0,60,120)if f<frames]+[f for f in range(frames)if f not in (0,60,120)]
    resume_metadata=None
    if args.resume_existing:
        import shutil
        old_path=out/'body_trajectory.npz';old_report=json.loads(report_path.read_text())
        old_sha=digest(old_path);archive=out/('body_before_refit_'+old_sha[:12]+'.npz')
        if not archive.exists():shutil.copy2(old_path,archive)
        archive_report=out/('body_before_refit_'+old_sha[:12]+'.json')
        if not archive_report.exists():shutil.copy2(report_path,archive_report)
        with np.load(old_path)as old:
            if old['joint_names'].tolist()!=model.body_joint_names:raise RuntimeError('Resume body joint order changed')
            states[:]=np.column_stack((old['base_xyz_m'],Rotation.from_quat(old['base_quaternion_xyzw']).as_rotvec(),old['joint_positions_rad']))[:frames]
            observability[:]=old['joint_image_supported'][:frames];uncertainty[:]=old['joint_approx_uncertainty_rad'][:frames]
        transforms[:]=np.stack([model.fk(s)for s in states])
        schedule=list(range(frames))
        if args.refine_frames:
            selected=set()
            for item in args.refine_frames.split(','):
                start,end=map(int,item.split(':'))if ':'in item else (int(item),int(item))
                selected.update(range(start,end+1))
            schedule=sorted(f for f in selected if 0<=f<frames)
        records=[r for r in old_report['frames_detail']if r['frame']not in schedule and r['frame']<frames]
        resume_metadata={'source_sha256':old_sha,'archive':str(archive),'source_quality_valid_frames':old_report['image_quality_valid_frames'],'reason':'Repair independently initialized keyframe branch discontinuities using chronological actual-image optimization; consistently exclude unmodelled hands even if palm LK becomes invalid.','temporal_strength':args.temporal_strength,'arm_landmark_prior':args.arm_prior,'interpolation_used':False,'heldout_caveat':'Failure-frame QA motivated refinement; these heldout metrics are diagnostic, not an untouched confirmatory test.'}
        resume_metadata.update(refined_frame_indices=schedule,reevaluated_only_frame_indices=[r['frame']for r in records])

    def correspondence(state,frame):
        predicted=render(state,frame);data=[]
        for view in range(2):
            pred=predicted[view];obs=observed[frame][view]['train']
            if len(pred['xy'])<10 or len(obs)<10:continue
            dt,ids=cKDTree(obs).query(pred['xy']);data.append((view,pred['links'],pred['local'],obs[ids],np.full(len(ids),.65)))
            dt,ids=cKDTree(pred['xy']).query(obs);data.append((view,pred['links'][ids],pred['local'][ids],obs,np.full(len(ids),.65)))
            good=obs_valid[frame,view].copy();good[[3,7]]=False
            landmark_weight=np.asarray([3.,5.,8.,0.,3.,5.,8.,0.])
            if good.any():data.append((view,landmark_links[good],landmark_local[good],observations[frame,view,good],landmark_weight[good]))
            good=foot_valid[frame,view]&foot_train
            if good.any():data.append((view,foot_links[good],foot_local[good],foot_xy[frame,view,good],np.full(int(good.sum()),1.3)))
        return predicted,data

    def objective(state,data,previous):
        loss=0.;gradient=np.zeros(35)
        for view,links,local,target,weight in data:
            world,jac=model.points(state,links,local,True);pixels,projection_jac=project_points(P[view],world,True);J=np.einsum('nij,njk->nik',projection_jac,jac)
            residual=(pixels-target)*weight[:,None];root=np.sqrt(1+(residual/4.)**2)
            loss+=float((32*(root-1)).sum());dr=2*residual/root*weight[:,None];gradient+=np.einsum('ni,nik->k',dr,J)
        residual=(state-state0)*q_prior;loss+=float(residual@residual);gradient+=2*residual*q_prior
        if previous is not None:
            residual=(state-previous)*temporal_weight;loss+=float(residual@residual);gradient+=2*residual*temporal_weight
        return loss,gradient

    def evaluate(state,frame,predicted):
        results=[]
        for view in range(2):
            pred=predicted[view];obs=observed[frame][view];mask=pred['mask']&obs['body_domain'];truth=obs['mask']&obs['body_domain'];union=(mask|truth).sum();iou=float((mask&truth).sum()/max(union,1))
            held=obs['heldout'];distance=cKDTree(pred['xy']).query(held)[0]if len(pred['xy']) and len(held)else np.asarray([float('inf')])
            landmarks=model.points(state,landmark_links,landmark_local);uv=project_points(P[view],landmarks);good=obs_valid[frame,view].copy();good[[3,7]]=False;lm=np.linalg.norm(uv[good]-observations[frame,view,good],axis=1)
            all_foot=project_points(P[view],model.points(state,foot_links,foot_local));heldfoot=foot_valid[frame,view]&~foot_train;footerr=np.linalg.norm(all_foot[heldfoot]-foot_xy[frame,view,heldfoot],axis=1)
            results.append({'view':view,'mask_iou':iou,'heldout_contour_count':len(held),'heldout_contour_median_px':float(np.median(distance)),'heldout_contour_p90_px':float(np.percentile(distance,90)),'landmark_count':int(good.sum()),'landmark_median_px':float(np.median(lm))if len(lm)else None,'landmark_p90_px':float(np.percentile(lm,90))if len(lm)else None,'landmark_errors':{point_names[k]:float(e)for k,e in zip(np.flatnonzero(good),lm)},'heldout_feet_count':int(heldfoot.sum()),'heldout_feet_median_px':float(np.median(footerr))if len(footerr)else None})
        return results

    baseline_metrics={f:evaluate(state0,f,render(state0,f))for f in (0,60,120)if f<frames}
    for record in records:
        f=record['frame'];record['metrics']=evaluate(states[f],f,render(states[f],f));record['body_domain_reevaluation_only']=True
        quality=all(m['mask_iou']>=.55 and m['heldout_contour_median_px']<=6 and m['heldout_contour_p90_px']<=15 and (m['landmark_median_px']is None or (m['landmark_median_px']<=8 and m['landmark_p90_px']<=16))for m in record['metrics'])
        valid_frames[f]=quality and record['joint_limits_satisfied']and bool(camera_valid[f].all());record['image_candidate_valid']=bool(valid_frames[f])
    for number,frame in enumerate(schedule):
        completed=np.flatnonzero(np.isfinite(states[:,0]));nearest=int(completed[np.argmin(np.abs(completed-frame))])if len(completed)else None
        previous=states[frame-1]if frame>0 and np.isfinite(states[frame-1]).all()else None
        initial=previous.copy()if args.resume_existing and previous is not None else states[nearest].copy()if nearest is not None else state0.copy()
        state=np.clip(initial,lower+1e-8,upper-1e-8);improvements=[];optimizer_ok=False
        # Genuine image-landmark IK prevents a good whole-mask score from
        # leaving the newly generated arms near the old reference pose.
        arm_variables=np.asarray([6+j for j,name in enumerate(model.body_joint_names)if any(token in name for token in ('shoulder','elbow','wrist','waist'))])
        def arm_landmark_residual(angles):
            trial=state.copy();trial[arm_variables]=angles;world=model.points(trial,landmark_links,landmark_local);residual=[]
            for view in range(2):
                good=obs_valid[frame,view].copy();good[[3,7]]=False
                if good.any():residual.extend(((project_points(P[view],world)[good]-observations[frame,view,good])*np.asarray([1.,1.3,1.8,0.,1.,1.3,1.8,0.])[good,None]).ravel())
            residual.extend((angles-initial[arm_variables])*args.arm_prior)
            return np.asarray(residual)
        arm_fit=least_squares(arm_landmark_residual,state[arm_variables],bounds=(lower[arm_variables],upper[arm_variables]),loss='soft_l1',f_scale=5.,max_nfev=100)
        state[arm_variables]=arm_fit.x
        for outer in range(args.outer_iterations+(3 if frame in (0,60,120)else 0)):
            predicted,data=correspondence(state,frame)
            if not data:break
            before=objective(state,data,previous)[0]
            fit=minimize(lambda x:objective(x,data,previous),state,jac=True,method='L-BFGS-B',bounds=list(zip(lower,upper)),options={'maxiter':args.optimizer_iterations,'ftol':1e-7,'gtol':1e-4,'maxls':25})
            state=fit.x;optimizer_ok |= bool(fit.success);improvements.append({'outer':outer,'before_fixed_correspondence_loss':before,'after_fixed_correspondence_loss':float(fit.fun),'iterations':int(fit.nit),'optimizer_success':bool(fit.success)})
        states[frame]=state;transforms[frame]=model.fk(state);predicted,data=correspondence(state,frame);metrics_by_view=evaluate(state,frame,predicted)
        # Approximate local image information, not unique hardware joint observability.
        info=np.zeros((35,35));view_support=np.zeros((2,29),dtype=int)
        for view,links,local,target,weight in data:
            world,jac=model.points(state,links,local,True);_,pj=project_points(P[view],world,True);J=np.einsum('nij,njk->nik',pj,jac)*weight[:,None,None];flat=J.reshape(-1,35);info+=flat.T@flat
            view_support[view]+=model.ancestor[links].sum(axis=0)
        covariance=np.linalg.pinv(info+np.diag(q_prior**2)+1e-6*np.eye(35));std=np.sqrt(np.maximum(np.diag(covariance)[6:],0))*4
        uncertainty[frame]=std;observability[frame]=(view_support.min(axis=0)>=8)&(std<.35)
        # Wrist orientation is not independently measured by a palm point.
        for j,name in enumerate(model.body_joint_names):
            if 'wrist_'in name:observability[frame,j]=False
        limits_ok=bool(((state[6:]>=model.body_limits[:,0]-1e-8)&(state[6:]<=model.body_limits[:,1]+1e-8)).all())
        quality=all(m['mask_iou']>=.55 and m['heldout_contour_median_px']<=6 and m['heldout_contour_p90_px']<=15 and (m['landmark_median_px'] is None or (m['landmark_median_px']<=8 and m['landmark_p90_px']<=16))for m in metrics_by_view)
        valid_frames[frame]=quality and limits_ok and bool(camera_valid[frame].all())
        records.append({'frame':frame,'time_seconds':float(timestamps[frame]),'image_candidate_valid':bool(valid_frames[frame]),'joint_limits_satisfied':limits_ok,'optimizer_converged_in_some_outer_step':optimizer_ok,'metrics':metrics_by_view,'outer_iterations':improvements,'two_view_silhouette_supported_body_joints':int(observability[frame].sum()),'wrist_orientation_is_prior_dominated':True})
        if number%4==0 or frame in (0,60,120):
            progress={'status':'running_image_body_optimization','completed_frames':frames-len(schedule)+number+1,'total_frames':frames,'refined_frames_completed':number+1,'total_scheduled_frames':len(schedule),'last_frame':frame,'elapsed_seconds':time.monotonic()-started,'last_metrics':metrics_by_view,'camera_sha256':digest(camera_path)}
            (out/'body_progress.json').write_text(json.dumps(progress,indent=2)+'\n');print(json.dumps(progress),flush=True)
        if not args.resume_existing and number==min(2,frames-1):
            done=np.flatnonzero(np.isfinite(states[:,0]));np.savez_compressed(out/'body_partial_keyframes.npz',frame_indices=done,base_xyz_m=states[done,:3],base_quaternion_xyzw=Rotation.from_rotvec(states[done,3:6]).as_quat(),joint_names=np.asarray(model.body_joint_names),joint_positions_rad=states[done,6:],link_names=np.asarray(model.link_names),link_world_transforms=transforms[done],candidate_valid=valid_frames[done])
    records.sort(key=lambda x:x['frame']);output=out/'body_trajectory.npz';base_quat=Rotation.from_rotvec(states[:,3:6]).as_quat()
    body_source_urdf_sha=model.urdf_sha256;fit_proxy_triangles=len(render_faces);retarget=None
    target_scene=ROOT/'data/video_generation/candidate_scene/scene_rohand.json'
    if target_scene.exists():
        target_model=G1Kinematics(ROOT,scene_path=target_scene)
        if target_model.body_joint_names!=model.body_joint_names:raise RuntimeError('Five-finger mount changed body joint order')
        target_transforms=np.stack([target_model.fk(s)for s in states])
        core=[name for name in model.link_names if 'hand'not in name]
        error=max(float(np.max(np.abs(transforms[:,model.link_index[name]]-target_transforms[:,target_model.link_index[name]])))for name in core)
        if error>1e-8:raise RuntimeError('Five-finger retarget changed the fitted body chain')
        retarget={'scene':str(target_scene),'urdf':str(target_model.urdf_path),'urdf_sha256':target_model.urdf_sha256,'body_link_transform_max_error':error,'finger_motion_estimated_by_body':False,'hand_link_transforms_are_default_mount_prior_only':True}
        model=target_model;transforms=target_transforms
    np.savez_compressed(output,base_xyz_m=states[:,:3],base_quaternion_xyzw=base_quat,joint_names=np.asarray(model.body_joint_names),joint_positions_rad=states[:,6:],link_names=np.asarray(model.link_names),link_world_transforms=transforms,
                        joint_image_supported=observability,joint_approx_uncertainty_rad=uncertainty,candidate_valid=valid_frames,joint_limits_rad=model.body_limits,timestamps=timestamps[:frames],camera_P_video=P,camera_sha256=np.asarray(digest(camera_path)),source_video_sha256=np.asarray(video_sha),finger_joint_positions_included=np.asarray(False))
    import av
    preview=ROOT/'outputs/previews/user_body_optimization.mp4'
    bone_edges=[(p,c)for name,p,c,fixed,axis,kind in model.joints if 'hand'not in model.link_names[p]and'hand'not in model.link_names[c]]
    with av.open(str(preview),'w')as container:
        stream=container.add_stream('libx264',rate=24);stream.width=1280;stream.height=720;stream.pix_fmt='yuv420p';stream.options={'crf':'21','preset':'medium'}
        for frame in range(frames):
            tiles=[]
            for view in range(2):
                image=cv2.cvtColor(rgb[frame,view],cv2.COLOR_RGB2BGR);points=project_points(P[view],transforms[frame,:,:3,3])
                for p,c in bone_edges:
                    a,b=tuple(np.rint(points[p]).astype(int)),tuple(np.rint(points[c]).astype(int));cv2.line(image,a,b,(255,180,20),2,cv2.LINE_AA)
                for k in np.flatnonzero(obs_valid[frame,view]):cv2.circle(image,tuple(np.rint(observations[frame,view,k]).astype(int)),5,(40,240,60),2)
                cv2.putText(image,f'BODY29 candidate f{frame:03d} | green=image landmarks',(8,24),cv2.FONT_HERSHEY_SIMPLEX,.55,(245,245,245),2,cv2.LINE_AA)
                cv2.putText(image,'Unseen joint axes remain prior-dominated; no 3D truth claim',(8,48),cv2.FONT_HERSHEY_SIMPLEX,.44,(245,245,245),1,cv2.LINE_AA);tiles.append(image)
            vf=av.VideoFrame.from_ndarray(np.concatenate(tiles,axis=1),format='bgr24')
            for packet in stream.encode(vf):container.mux(packet)
        for packet in stream.encode():container.mux(packet)
    report={'status':'body_image_candidate_optimization_completed','full_kinogen_pipeline':False,'strict_full_body_3d_ground_truth_recovered':False,'frames':frames,'state_parameters':35,'body_joints':model.body_joint_names,'finger_joints_delegated':model.finger_joint_names,'hands':'Target model is G1+five-finger ROHand. Entire hand meshes excluded from body fitting; actual visible wrist origins used instead of assuming stock palm length.','method':'Bounded floating-base+29-joint URDF FK; CPU simplified-mesh silhouette projective ICP with analytic FK/projection Jacobians, visually annotated real-video shoulder/elbow/palm landmarks propagated by measured bidirectional LK, and heldout-ID reliable shared TAP feet. No q0-to-nearest-SAM pseudo-measurements.','manual_observation_provenance':provenance,'heldout_policy':'Every fifth 16px spatial contour block withheld from silhouette target correspondence fitting; every fifth physical shared foot ID withheld across all times. Manual landmark propagation is a fit term, not independent heldout proof.','prior_policy':'Reference q0 only weak regularization and initialization; nearest optimized frame only warm-start, every output frame optimized against its own actual image evidence. No angle interpolation or hidden frame fill. No Kimodo prior used in this run.','uncertainty_policy':'Local data Jacobian approximation is conditional, correlations and generative topology mismatch remain. Wrist orientation always marked prior-dominated; joint_image_supported is not a uniquely observed hardware angle.','raster_proxy_scale':args.scale,'proxy_triangles':len(model.proxy_faces),'all_joint_limits_satisfied':all(r['joint_limits_satisfied']for r in records),'image_quality_valid_frames':int(valid_frames.sum()),'image_quality_failed_frames':np.flatnonzero(~valid_frames).tolist(),'quality_thresholds':{'per_view_mask_iou_min':.55,'heldout_contour_median_max_px':6,'heldout_contour_p90_max_px':15,'real_image_landmark_median_max_px':8,'real_image_landmark_p90_max_px':16},'baseline_q0_keyframes':baseline_metrics,'frames_detail':records,'inputs':{str(p.relative_to(ROOT)):digest(p)for p in (camera_path,source/'input.npz',source/'segmentation.npz',source/'tracking.npz',source/'geometry.npz')},'body_fit_source_urdf_sha256':body_source_urdf_sha,'urdf_sha256':model.urdf_sha256,'five_finger_retarget':retarget,'output':{'path':str(output),'sha256':digest(output)},'preview':{'path':str(preview),'sha256':digest(preview)},'elapsed_seconds':time.monotonic()-started,'gpu_used':False,'training_performed':False}
    step=np.abs(np.diff(states[:,6:],axis=0));largest=np.unravel_index(step.argmax(),step.shape)if step.size else (0,0)
    report['method']=report['method'].replace('shoulder/elbow/palm landmarks','shoulder/elbow/wrist landmarks')
    report['proxy_triangles_used_for_body_fitting']=fit_proxy_triangles
    report['hand_exclusion_policy']='Body-only image domain excludes radius55px about a genuinely observed wrist and radius35px about a genuinely observed palm ROI. Both predicted boundary and image contour use the same domain. Invalid observations are not filled; palm points are not fitted as wrist measurements.'
    report['chronological_image_refit']=resume_metadata
    report['temporal_diagnostics']={'max_adjacent_joint_step_rad':float(step.max())if step.size else 0.,'largest_step_frame':int(largest[0]+1),'largest_step_joint':model.body_joint_names[largest[1]],'median_adjacent_joint_step_rad':float(np.median(step))if step.size else 0.,'max_adjacent_base_step_m':float(np.linalg.norm(np.diff(states[:,:3],axis=0),axis=1).max())if frames>1 else 0.,'physical_dynamic_feasibility_proven':False}
    (out/'body_trajectory.json').write_text(json.dumps({k:v for k,v in report.items()if k!='frames_detail'},indent=2)+'\n');report_path.write_text(json.dumps(report,indent=2)+'\n')
    (out/'body_progress.json').write_text(json.dumps({'status':report['status'],'completed_frames':frames,'total_frames':frames,'image_quality_valid_frames':int(valid_frames.sum()),'output_sha256':report['output']['sha256'],'elapsed_seconds':report['elapsed_seconds']},indent=2)+'\n')
    print(json.dumps({k:report[k]for k in ('status','frames','all_joint_limits_satisfied','image_quality_valid_frames','image_quality_failed_frames','output','preview','elapsed_seconds')},indent=2),flush=True)


if __name__=='__main__':main()
