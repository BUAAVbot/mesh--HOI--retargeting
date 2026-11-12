#!/usr/bin/env python3
"""CPU five-finger ROHand image/temporal/contact candidate fitting for all frames.

Twelve physical actuator coordinates expand through the unmodified official
linkage equations to fifty mixed-unit URDF joint states. Optional bounded wrist
joint refinement is exported explicitly; no free rigid-body drift is allowed.
Occluded fingers and weak or conflicting evidence retain uncertainty.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def clean_json(value):
    import math
    if isinstance(value,dict):return {str(k):clean_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean_json(v) for v in value]
    if hasattr(value,'tolist'):return clean_json(value.tolist())
    if isinstance(value,float) and not math.isfinite(value):return None
    return value


def sample(image,xy,cval=25):
    from scipy.ndimage import map_coordinates
    return map_coordinates(image,[xy[:,1],xy[:,0]],order=1,mode='constant',cval=cval)


def observation(image,robot,box,valid,palm,direction):
    import cv2
    import numpy as np
    if not np.isfinite(palm).all():return None
    x,y=palm;x0,x1=max(0,int(x)-40),min(640,int(x)+40);y0,y1=max(0,int(y)-32),min(720,int(y)+30)
    roi=np.zeros((720,640),bool);roi[y0+2:y1-2,x0+2:x1-2]=True
    if direction>0:roi[:,:x0+12]=False
    else:roi[:,x1-12:]=False
    roi &= valid
    foreground=robot&roi
    if foreground.sum()<50:return None
    gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    edges=cv2.Canny(gray,40,110)>0
    silhouette=cv2.morphologyEx(robot.astype(np.uint8),cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))>0
    border=cv2.dilate((~roi).astype(np.uint8),np.ones((5,5),np.uint8))>0
    # Box and watermark cuts must not become apparent finger edges.
    excluded=cv2.dilate(box.astype(np.uint8),np.ones((3,3),np.uint8))>0
    observed=(silhouette|(edges&robot))&roi&~border&~excluded
    yy,xx=np.nonzero(observed)
    if len(xx)<12:return None
    all_points=np.column_stack((xx,yy)).astype(float)
    heldout=((xx//6+yy//6)%5)==0
    heldout_points=all_points[heldout]
    training=observed.copy();training[yy[heldout],xx[heldout]]=False
    points=all_points[~heldout]
    if len(points)<8:return None
    points=points[np.linspace(0,len(points)-1,min(48,len(points))).astype(int)]
    distance=cv2.distanceTransform((~training).astype(np.uint8),cv2.DIST_L2,3)
    outside=cv2.distanceTransform((~robot).astype(np.uint8),cv2.DIST_L2,3)
    adjacent=foreground&(cv2.dilate(box.astype(np.uint8),np.ones((7,7),np.uint8))>0)
    return dict(distance=distance,outside=outside,points=points,heldout_points=heldout_points,roi=roi,valid=valid,robot=robot,box=box,
                bounds=np.asarray([x0,y0,x1,y1]),contact_hint=bool(adjacent.sum()>6),palm=np.asarray(palm))


def boundary_geometry(model,u,base,P):
    import numpy as np
    from hand_geometry import project,polygon_samples
    actual=model.lower+u[-6:]*(model.upper-model.lower)
    world_base=base(u[:-6]) if callable(base) else base
    geometry,tips,_=model.world_geometry(actual,world_base)
    boundaries=[]
    for link,vertices in geometry.items():
        uv,z=project(vertices,P)
        boundaries.append(polygon_samples(uv,12))
    return np.concatenate(boundaries),tips,geometry


def heldout_error(model,u,base,projections,observations):
    import numpy as np
    from scipy.spatial import cKDTree
    distances=[]
    for P,obs in zip(projections,observations):
        if obs is None or len(obs['heldout_points'])==0:continue
        points,_,_=boundary_geometry(model,u,base,P)
        distances.extend(cKDTree(points).query(obs['heldout_points'])[0])
    return float(np.mean(distances)) if distances else float('nan')


def image_residual(model,u,base,projections,observations):
    import numpy as np
    from scipy.spatial import cKDTree
    residual=[]
    for P,obs in zip(projections,observations):
        if obs is None:continue
        points,_,_=boundary_geometry(model,u,base,P)
        visible=sample(obs['valid'].astype(float),points,cval=0)>.5
        box_occluded=(sample(obs['box'].astype(float),points,cval=0)>.5)&(sample(obs['robot'].astype(float),points,cval=0)<.5)
        weights=visible&~box_occluded
        distances=np.clip(sample(obs['distance'],points),0,18)
        outside=np.clip(sample(obs['outside'],points),0,18)
        backward=cKDTree(points).query(obs['points'])[0]
        residual.extend(distances*weights/3)
        residual.extend(outside*weights*.4/3)
        residual.extend(np.clip(backward,0,18)*.3/3)
    return np.asarray(residual) if residual else np.zeros(1)


def fit(model,base,projections,observations,box_tf,box_size,box_strict,previous,before,frame,max_nfev,bounds=None,full_mesh_contact=False,mano_seed=None):
    import numpy as np
    from scipy.optimize import least_squares
    from hand_geometry import box_signed_distance
    has_data=any(o is not None for o in observations)
    hints=any(o is not None and o['contact_hint'] for o in observations)
    lower,upper=(np.zeros(len(previous)),np.ones(len(previous))) if bounds is None else bounds
    def bounded(x):return np.clip(x,lower+1e-8,upper-1e-8)
    def residual(u,contact=False):
        data=image_residual(model,u,base,projections,observations)
        temporal=np.r_[(u-previous)*.25,(u-2*previous+before)*.18]
        result=np.r_[data,temporal,(u[-6:]-.1)*.02]
        if mano_seed is not None:result=np.r_[result,(u[-6:]-mano_seed)*.06]
        if callable(base):
            result=np.r_[result,(u[:-6]-base.reference)*.4]
            if hasattr(base,'wrist_residual'):result=np.r_[result,base.wrist_residual(u[:-6])]
        if contact:
            world_base=base(u[:-6]) if callable(base) else base
            _,tips,transforms=model.world_geometry(model.lower+u[-6:]*(model.upper-model.lower),world_base)
            gap=box_signed_distance(tips,box_tf,box_size)
            # Only close, image-adjacent fingers receive a weak surface prior.
            attraction=np.minimum(np.abs(gap),.06)*(.08/.02) if hints else np.zeros(5)
            penetration=np.maximum(-gap-.004,0)*(.08/.01)
            result=np.r_[result,attraction,penetration]
            if full_mesh_contact:
                sdf=box_signed_distance(model.contact_surface(transforms),box_tf,box_size)
                # Independent per-link surface samples cover base/palm/proximal/
                # distal/linkages; this remains weaker than image evidence.
                result=np.r_[result,np.maximum(-sdf-.004,0)*(.35/.01)/np.sqrt(len(sdf)/50)]
        return result
    starts=[bounded(previous)]
    if mano_seed is not None:
        extra=previous.copy();extra[-6:]=mano_seed;starts.append(bounded(extra))
    if frame in (0,30,60,90,120):
        for amount in (.1,.55):
            alternative=previous.copy();alternative[-6:]=amount
            if callable(base):alternative[:-6]=np.clip(base.reference+(.15 if amount==.1 else -.15),1e-5,1-1e-5)
            starts.append(bounded(alternative))
    results=[]
    for start in starts:
        r=least_squares(residual,start,bounds=(lower,upper),loss='soft_l1',f_scale=1.,max_nfev=max_nfev,
                        diff_step=.01,ftol=3e-4,xtol=3e-4,gtol=3e-4)
        results.append(r)
    best=min(results,key=lambda r:r.cost);u=best.x
    raw=image_residual(model,u,base,projections,observations)
    raw_cost=float(np.mean(raw**2));prior_used=False;conflict=False
    if box_strict and hints and has_data:
        c=least_squares(lambda x:residual(x,True),bounded(u),bounds=(lower,upper),loss='soft_l1',max_nfev=max_nfev//2+1,diff_step=.01,
                        ftol=3e-4,xtol=3e-4,gtol=3e-4)
        changed=image_residual(model,c.x,base,projections,observations)
        if float(np.mean(changed**2))<=raw_cost*1.08+.03:u=c.x;prior_used=True
        else:conflict=True
    eps=.005;columns=[]
    for j in range(len(u)):
        plus=u.copy();minus=u.copy();plus[j]=min(1,u[j]+eps);minus[j]=max(0,u[j]-eps)
        columns.append((image_residual(model,plus,base,projections,observations)-image_residual(model,minus,base,projections,observations))/max(plus[j]-minus[j],1e-9))
    jac=np.column_stack(columns)
    eigenvalues,eigenvectors=np.linalg.eigh(jac.T@jac)
    s=np.sqrt(np.maximum(eigenvalues[::-1],0))
    variance=max(float(np.mean(image_residual(model,u,base,projections,observations)**2)),1.)
    covariance=(eigenvectors*(variance/np.maximum(eigenvalues,1.)))@eigenvectors.T
    std=np.sqrt(np.maximum(np.diag(covariance),0));span=np.ptp(np.stack([r.x for r in results if r.cost<=best.cost*1.1+1]),axis=0)
    std=np.minimum(1,np.maximum(std,span/2))
    sensitivity=np.sqrt(np.mean(jac**2,axis=0))*3
    observed=(std<.25)&(sensitivity>1.5)&has_data
    return u,dict(std_normalized=std,multistart_span=span,sensitivity_px=sensitivity,observed=observed,
                  data_rms_px=float(np.sqrt(np.mean(image_residual(model,u,base,projections,observations)**2))*3),
                  data_jacobian_singular_values=s,contact_prior_used=prior_used,contact_image_conflict=conflict,
                  optimization_success=bool(best.success),nfev=int(best.nfev),views_observed=sum(o is not None for o in observations))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--max-nfev',type=int,default=24)
    parser.add_argument('--body-file',type=Path)
    parser.add_argument('--source-hand-file',type=Path,help='Baseline hand comparison parent SHA; fitting remains driven by current images/body')
    parser.add_argument('--initial-hand-file',type=Path,help='Per-frame control baseline preserved exactly when missing data or a new candidate fails image/collision acceptance')
    parser.add_argument('--mano-initialization',action='store_true',help='Weak nonmetric MANO finger-rotation magnitude seed/prior; retain RGB and dense collision acceptance')
    parser.add_argument('--output-suffix',help='Independent output/report suffix, e.g. velocity_refined or contact')
    parser.add_argument('--wrist-velocity-limit',type=float,help='Yaw rad/s hard optimization bound, including safe next-source fallback')
    parser.add_argument('--full-mesh-contact',action='store_true',help='Use all hand visual-link surface samples for weak nonpenetration loss')
    parser.add_argument('--keyframes',nargs='+',type=int,help='Explicit sparse debug only; separate output, never a completed full sequence')
    parser.add_argument('--refine-wrist-yaw',action='store_true',help='Explicitly fit true bounded wrist_yaw joint; keeps measured wrist origin unchanged')
    parser.add_argument('--refine-wrist-all',action='store_true',help='Compare bounded roll/pitch/yaw+hand with true wrist pixel constraints; export all changes explicitly')
    args=parser.parse_args();os.environ.setdefault('OMP_NUM_THREADS','4')
    if args.output_suffix and not all(c.isalnum() or c=='_' for c in args.output_suffix):parser.error('Unsafe output suffix')
    if args.wrist_velocity_limit is not None and (args.wrist_velocity_limit<=0 or not args.refine_wrist_yaw or args.refine_wrist_all or args.keyframes):parser.error('Velocity bounds require positive limit and full yaw-only run')
    import cv2
    import imageio.v2 as imageio
    import numpy as np
    from hand_geometry import ROHandGeometry,box_signed_distance,project
    from user_rohand_coupling import ROHandCoupling
    cv2.setNumThreads(3)
    output=ROOT/'outputs/user_reconstruction_v2';source=ROOT/'outputs/user_reconstruction'
    body_path=(args.body_file or output/'body_trajectory.npz').resolve();camera_path=output/'camera_refinement.npz';box_path=output/'box_trajectory.npz'
    body=np.load(body_path,allow_pickle=False);camera=np.load(camera_path,allow_pickle=False);boxes=np.load(box_path,allow_pickle=False)
    data=np.load(source/'input.npz',allow_pickle=False);seg=np.load(source/'segmentation.npz',allow_pickle=False)
    landmarks=np.load(output/'body_landmark_observations.npz',allow_pickle=False)
    rgb=data['rgb'];timestamps=data['timestamps'];P=camera['P_refined'];link_names=body['link_names'].tolist()
    robot_masks=seg['robot'];box_masks=seg['box'];pixel_valid=seg['observation_valid']
    landmark_valid=landmarks['valid'];landmark_positions=landmarks['positions_xy']
    camera_valid=camera['frame_camera_valid'] if 'frame_camera_valid' in camera else np.zeros((121,2),bool)
    transforms=body['link_world_transforms'];coupling=ROHandCoupling(ROOT)
    models=[ROHandGeometry(ROOT,s) for s in ('left','right')]
    initial_hand=np.load(args.initial_hand_file,allow_pickle=False) if args.initial_hand_file else None
    initial_normalized=None
    if initial_hand is not None:
        if initial_hand['actuator_names'].tolist()!=list(coupling.actuator_names) or not np.array_equal(initial_hand['timestamps'],timestamps):raise ValueError('Initial hand contract mismatch')
        initial_normalized=(initial_hand['actuator_positions']-np.asarray(coupling.lower))/(np.asarray(coupling.upper)-np.asarray(coupling.lower))
        if initial_normalized.shape!=(121,12) or np.any(initial_normalized<0) or np.any(initial_normalized>1):raise ValueError('Initial hand controls out of bounds')
    mano_path=output/'mano_hand_candidate.npz';mano=np.load(mano_path,allow_pickle=False) if args.mano_initialization else None
    mano_seed_used=np.zeros((121,2),bool)
    def neural_seed(frame,hand,reference):
        if mano is None or not bool(mano['valid'][frame,hand]):return None
        rotations=mano['mano_hand_pose_rotation_matrices'][frame,hand]
        magnitudes=Rotation.from_matrix(rotations).magnitude().reshape(5,3).sum(axis=1)
        # MANO groups index,middle,little,ring,thumb. Angular magnitudes mix
        # flexion and splay; deliberately weak prior, not robot observations.
        target=magnitudes[[4,0,1,3,2]];seed=reference.copy();side=('left','right')[hand]
        for control,label in enumerate(('th','if','mf','rf','lf')):
            grid=np.linspace(0,1,41);values=[]
            for value in grid:
                actuators=models[hand].lower.copy();actuators[control]+=value*(models[hand].upper[control]-models[hand].lower[control])
                joints=coupling.expand_side(actuators,side)
                values.append(sum(abs(q) for name,q in joints.items() if name.split('_hand_ro_')[1] in (label+'_proximal_link',label+'_distal_link')))
            values=np.asarray(values);bend=np.abs(values-values[0]);seed[control]=grid[np.argmin(abs(bend-target[control]))]
        mano_seed_used[frame,hand]=True
        return seed
    frame_ids=np.asarray(args.keyframes if args.keyframes else list(range(121)),int)
    if len(timestamps)!=121 or (not args.keyframes and transforms.shape[0]!=121):raise ValueError('Full run must fit all original 121 frames')
    body_frames=body['frame_indices'] if 'frame_indices' in body else np.arange(len(transforms))
    tf_by_frame={int(frame):matrix for frame,matrix in zip(body_frames,transforms)}
    if not all(int(frame) in tf_by_frame for frame in frame_ids):raise ValueError('Body transforms absent for requested frames')
    scene=json.loads((ROOT/'data/video_generation/candidate_scene/scene_rohand.json').read_text())
    mounts=[]
    for side in ('left','right'):
        reference=scene['robot']['link_world_transforms']
        mounts.append(np.linalg.inv(np.asarray(reference[side+'_wrist_yaw_link']))@np.asarray(reference[side+'_hand_ro_base_link']))
    import xml.etree.ElementTree as ET
    from scipy.spatial.transform import Rotation
    urdf=ET.parse(ROOT/scene['robot']['urdf']).getroot()
    yaw_names=[s+'_wrist_yaw_joint' for s in ('left','right')]
    body_joint_names=body['joint_names'].tolist();body_q_by_frame={int(f):q for f,q in zip(body_frames,body['joint_positions_rad'])}
    yaw_definitions=[]
    for name in yaw_names:
        joint=next(j for j in urdf.findall('joint') if j.attrib['name']==name)
        axis=np.fromstring(joint.find('axis').attrib['xyz'],sep=' ');axis/=np.linalg.norm(axis)
        lim=joint.find('limit');yaw_definitions.append((axis,float(lim.attrib['lower']),float(lim.attrib['upper'])))
    class YawBase:
        def __init__(self,frame,side):
            self.wrist=tf_by_frame[frame][link_names.index(('left','right')[side]+'_wrist_yaw_link')]
            self.axis,self.low,self.high=yaw_definitions[side];self.range=self.high-self.low
            self.angle=float(body_q_by_frame[frame][body_joint_names.index(yaw_names[side])]);self.reference=(self.angle-self.low)/self.range
            self.mount=mounts[side]
        def __call__(self,u):
            u=float(np.asarray(u).ravel()[0])
            motion=np.eye(4);motion[:3,:3]=Rotation.from_rotvec(self.axis*(self.low+u*self.range-self.angle)).as_matrix()
            return self.wrist@motion@self.mount
        def wrist_point(self,u):return self.wrist[:3,3]
    class Wrist3Base:
        def __init__(self,frame,side):
            self.frame,self.side=frame,side;label=('left','right')[side]
            joints=[next(j for j in urdf.findall('joint') if j.attrib['name']==label+'_wrist_'+axis+'_joint') for axis in ('roll','pitch','yaw')]
            self.parent=tf_by_frame[frame][link_names.index(joints[0].find('parent').attrib['link'])]
            self.origins=[];self.axes=[];low=[];high=[]
            self.names=[j.attrib['name'] for j in joints]
            for joint in joints:
                el=joint.find('origin');tf=np.eye(4);tf[:3,3]=np.fromstring(el.attrib.get('xyz','0 0 0'),sep=' ')
                tf[:3,:3]=Rotation.from_euler('xyz',np.fromstring(el.attrib.get('rpy','0 0 0'),sep=' ')).as_matrix();self.origins.append(tf)
                axis=np.fromstring(joint.find('axis').attrib['xyz'],sep=' ');self.axes.append(axis/np.linalg.norm(axis))
                lim=joint.find('limit');low.append(float(lim.attrib['lower']));high.append(float(lim.attrib['upper']))
            self.low=np.asarray(low);self.range=np.asarray(high)-self.low
            self.angle=np.asarray([body_q_by_frame[frame][body_joint_names.index(n)] for n in self.names])
            self.reference=(self.angle-self.low)/self.range;self.mount=mounts[side]
        def wrist_transform(self,u):
            matrix=self.parent.copy()
            for q,origin,axis in zip(self.low+np.asarray(u)*self.range,self.origins,self.axes):
                rotation=np.eye(4);rotation[:3,:3]=Rotation.from_rotvec(axis*q).as_matrix();matrix=matrix@origin@rotation
            return matrix
        def __call__(self,u):return self.wrist_transform(u)@self.mount
        def wrist_point(self,u):return self.wrist_transform(u)[:3,3]
        def wrist_residual(self,u):
            target=6 if self.side==0 else 2;result=[]
            for view in range(2):
                if landmark_valid[self.frame,view,target]:
                    uv,_=project(self.wrist_point(u)[None],P[view]);result.extend((uv[0]-landmark_positions[self.frame,view,target])*3.)
            return np.asarray(result)
    def base_tf(frame,side):
        name=('left','right')[side]+'_hand_ro_base_link'
        matrices=tf_by_frame[int(frame)]
        if name in link_names:return matrices[link_names.index(name)]
        return matrices[link_names.index(('left','right')[side]+'_wrist_yaw_link')]@mounts[side]
    source_valid=body['candidate_valid'] if 'candidate_valid' in body else (body['valid'] if 'valid' in body else (body['image_fit_valid'] if 'image_fit_valid' in body else np.zeros(len(transforms),bool)))
    if source_valid.ndim>1:source_valid=source_valid.all(axis=1)
    body_valid=np.zeros(121,bool)
    for frame,v in zip(body_frames,source_valid):body_valid[int(frame)]=v
    normalized=np.zeros((121,12));uncertainty=np.ones((121,12));act_observed=np.zeros((121,12),bool)
    yaw_positions=np.full((121,2),np.nan);yaw_observed=np.zeros((121,2),bool);yaw_accepted=np.zeros((121,2),bool)
    all_wrist_positions=np.full((121,2,3),np.nan);wrist_world_shift=np.zeros((121,2))
    all_wrist_observed=np.zeros((121,2,3),bool)
    wrist_error_before=np.full((121,2),np.nan);wrist_error_after=np.full((121,2),np.nan)
    before_heldout=np.full((121,2),np.nan);after_heldout=np.full((121,2),np.nan)
    image_valid=np.zeros((121,2),bool);side_valid=np.zeros_like(image_valid);contact=np.full((121,2,5),np.nan)
    surface_min=np.full((121,2),np.nan);surface_fraction=np.full((121,2),np.nan)
    hand_update_accepted=np.zeros((121,2),bool);dense_penetration_before=np.full((121,2,3),np.nan);dense_penetration_after=np.full((121,2,3),np.nan)
    evidence=[];prior_used=np.zeros((121,2),bool);conflict=np.zeros_like(prior_used)
    start=time.monotonic()
    for frame_value in frame_ids:
        frame=int(frame_value)
        row=[];frame_observations=[]
        for hand,(side,view,direction,index) in enumerate((('left',1,-1,7),('right',0,1,3))):
            obs=[None,None]
            if bool(landmark_valid[frame,view,index]):
                obs[view]=observation(rgb[frame,view],robot_masks[frame,view],box_masks[frame,view],pixel_valid[frame,view],landmark_positions[frame,view,index],direction)
            model=models[hand];base=base_tf(frame,hand);sl=slice(hand*6,(hand+1)*6)
            previous=normalized[frame-1,sl] if frame else np.full(6,.05)
            if initial_normalized is not None:previous=initial_normalized[frame,sl].copy()
            neural=neural_seed(frame,hand,previous)
            before=normalized[max(frame-2,0),sl] if frame>1 else previous
            u,info=fit(model,base,P,obs,boxes['box_world_transform'][frame],boxes['box_size_xyz_m'],bool(boxes['valid'][frame]),previous,before,frame,args.max_nfev,full_mesh_contact=args.full_mesh_contact,mano_seed=neural)
            yaw_base=YawBase(frame,hand);yaw_positions[frame,hand]=yaw_base.angle
            wrist_names=[side+'_wrist_'+a+'_joint' for a in ('roll','pitch','yaw')]
            all_wrist_positions[frame,hand]=[body_q_by_frame[frame][body_joint_names.index(n)] for n in wrist_names]
            wrist_landmark=6 if side=='left' else 2
            if landmark_valid[frame,view,wrist_landmark]:
                wrist_uv,_=project(yaw_base.wrist[:3,3][None],P[view])
                wrist_error_before[frame,hand]=np.linalg.norm(wrist_uv[0]-landmark_positions[frame,view,wrist_landmark])
                wrist_error_after[frame,hand]=wrist_error_before[frame,hand]
            before_heldout[frame,hand]=heldout_error(model,u,base,P,obs);after_heldout[frame,hand]=before_heldout[frame,hand]
            chosen_base=base
            if args.refine_wrist_yaw or args.refine_wrist_all:
                refining_base=Wrist3Base(frame,hand) if args.refine_wrist_all else yaw_base
                previous_yaw=(yaw_positions[frame-1,hand]-yaw_base.low)/yaw_base.range if frame>0 and np.isfinite(yaw_positions[frame-1,hand]) else yaw_base.reference
                older_yaw=(yaw_positions[frame-2,hand]-yaw_base.low)/yaw_base.range if frame>1 and np.isfinite(yaw_positions[frame-2,hand]) else previous_yaw
                if args.refine_wrist_all:
                    previous_controls=(all_wrist_positions[frame-1,hand]-refining_base.low)/refining_base.range if frame>0 and np.isfinite(all_wrist_positions[frame-1,hand]).all() else refining_base.reference
                    older_controls=(all_wrist_positions[frame-2,hand]-refining_base.low)/refining_base.range if frame>1 and np.isfinite(all_wrist_positions[frame-2,hand]).all() else previous_controls
                else:previous_controls,older_controls=previous_yaw,older_yaw
                seed=np.r_[previous_controls,u];older=np.r_[older_controls,before]
                trial_bounds=None;velocity_locked=False
                if args.wrist_velocity_limit is not None:
                    lo,hi=yaw_base.low,yaw_base.high
                    if frame:
                        step=args.wrist_velocity_limit*(timestamps[frame]-timestamps[frame-1]);lo=max(lo,yaw_positions[frame-1,hand]-step);hi=min(hi,yaw_positions[frame-1,hand]+step)
                    if frame<120:
                        next_angle=body_q_by_frame[frame+1][body_joint_names.index(yaw_names[hand])]
                        step=args.wrist_velocity_limit*(timestamps[frame+1]-timestamps[frame]);lo=max(lo,next_angle-step);hi=min(hi,next_angle+step)
                    if lo>hi+1e-9:raise ValueError(f'No velocity-feasible yaw interval at frame {frame} {side}')
                    velocity_locked=hi-lo<1e-7
                    if not velocity_locked:trial_bounds=(np.r_[(lo-yaw_base.low)/yaw_base.range,np.zeros(6)],np.r_[(hi-yaw_base.low)/yaw_base.range,np.ones(6)])
                if velocity_locked:
                    # A saturated source interval may determine a unique yaw;
                    # retain that true body joint while still fitting controls.
                    trial=np.r_[yaw_base.reference,u];trial_info=dict(info)
                else:
                    trial,trial_info=fit(model,refining_base,P,obs,boxes['box_world_transform'][frame],boxes['box_size_xyz_m'],bool(boxes['valid'][frame]),seed,older,frame,args.max_nfev,bounds=trial_bounds,full_mesh_contact=args.full_mesh_contact,mano_seed=neural)
                trial_heldout=heldout_error(model,trial,refining_base,P,obs)
                trial_error=float('nan')
                if landmark_valid[frame,view,wrist_landmark]:
                    trial_uv,_=project(refining_base.wrist_point(trial[:-6])[None],P[view]);trial_error=float(np.linalg.norm(trial_uv[0]-landmark_positions[frame,view,wrist_landmark]))
                point_ok=not np.isfinite(wrist_error_before[frame,hand]) or trial_error<=wrist_error_before[frame,hand]+1.5
                accepted=bool(not velocity_locked and point_ok and np.isfinite(trial_heldout) and trial_info['data_rms_px']<info['data_rms_px']-.03 and trial_heldout<=before_heldout[frame,hand]+.25)
                if accepted:
                    changed_angles=refining_base.low+trial[:-6]*refining_base.range
                    if args.refine_wrist_all:all_wrist_positions[frame,hand]=changed_angles
                    else:all_wrist_positions[frame,hand,2]=float(changed_angles[0])
                    if args.refine_wrist_all:all_wrist_observed[frame,hand]=trial_info['observed'][:3]
                    else:all_wrist_observed[frame,hand,2]=trial_info['observed'][0]
                    yaw_positions[frame,hand]=all_wrist_positions[frame,hand,2];yaw_observed[frame,hand]=trial_info['observed'][-7]
                    yaw_accepted[frame,hand]=True;after_heldout[frame,hand]=trial_heldout;chosen_base=refining_base(trial[:-6]);u=trial[-6:]
                    wrist_error_after[frame,hand]=trial_error
                    wrist_world_shift[frame,hand]=np.linalg.norm(refining_base.wrist_point(trial[:-6])-yaw_base.wrist[:3,3])
                    for key in ('std_normalized','multistart_span','sensitivity_px','observed'):trial_info[key]=trial_info[key][-6:]
                    info=trial_info
            if initial_normalized is not None:
                # Accept only a genuine image-supported, collision-nonworsening
                # update relative to C + this frame's exact initial controls.
                reference=initial_normalized[frame,sl]
                reference_rms=float(np.sqrt(np.mean(image_residual(model,reference,base,P,obs)**2))*3)
                reference_heldout=heldout_error(model,reference,base,P,obs)
                before_heldout[frame,hand]=reference_heldout
                def dense_metrics(controls,world_base):
                    _,_,tf=model.world_geometry(model.lower+controls*(model.upper-model.lower),world_base)
                    sdf=box_signed_distance(model.contact_surface(tf,dense=True),boxes['box_world_transform'][frame],boxes['box_size_xyz_m'])
                    return np.asarray([np.maximum(-sdf,0).max(),np.mean(sdf<-.005),np.mean(np.maximum(-sdf-.005,0))])
                baseline_pen=dense_metrics(reference,base);candidate_pen=dense_metrics(u,chosen_base)
                dense_penetration_before[frame,hand]=baseline_pen
                collision_ok=bool(np.all(candidate_pen<=baseline_pen+np.asarray([1e-6,1e-7,1e-7])))
                update_ok=bool(info['views_observed'] and info['data_rms_px']<4 and after_heldout[frame,hand]<4.5 and info['data_rms_px']<=reference_rms+.03 and after_heldout[frame,hand]<=reference_heldout+.25 and collision_ok)
                hand_update_accepted[frame,hand]=update_ok
                if not update_ok:
                    u=reference.copy();chosen_base=base;yaw_positions[frame,hand]=yaw_base.angle;yaw_accepted[frame,hand]=False;yaw_observed[frame,hand]=False
                    all_wrist_positions[frame,hand]=[body_q_by_frame[frame][body_joint_names.index(n)] for n in wrist_names];all_wrist_observed[frame,hand]=False
                    wrist_error_after[frame,hand]=wrist_error_before[frame,hand];wrist_world_shift[frame,hand]=0
                    after_heldout[frame,hand]=reference_heldout
                    info.update(data_rms_px=reference_rms,std_normalized=np.ones(6),multistart_span=np.ones(6),sensitivity_px=np.zeros(6),observed=np.zeros(6,bool),contact_prior_used=False,contact_image_conflict=not collision_ok,optimization_success=False)
                    candidate_pen=baseline_pen
                dense_penetration_after[frame,hand]=candidate_pen
            normalized[frame,sl]=u;uncertainty[frame,sl]=info['std_normalized'];act_observed[frame,sl]=info['observed']
            _,tips,chosen_transforms=model.world_geometry(model.lower+u*(model.upper-model.lower),chosen_base)
            contact[frame,hand]=box_signed_distance(tips,boxes['box_world_transform'][frame],boxes['box_size_xyz_m'])
            surface_sdf=box_signed_distance(model.contact_surface(chosen_transforms),boxes['box_world_transform'][frame],boxes['box_size_xyz_m'])
            surface_min[frame,hand]=surface_sdf.min();surface_fraction[frame,hand]=np.mean(surface_sdf<-.005)
            prior_used[frame,hand]=info['contact_prior_used'];conflict[frame,hand]=info['contact_image_conflict']
            image_valid[frame,hand]=bool(info['views_observed'] and info['data_rms_px']<4. and np.isfinite(after_heldout[frame,hand]) and after_heldout[frame,hand]<4.5)
            side_valid[frame,hand]=bool(image_valid[frame,hand] and body_valid[frame] and camera_valid[frame,view] and np.all(info['observed']) and not info['contact_image_conflict'])
            if not image_valid[frame,hand]:act_observed[frame,sl]=False
            row.append({'frame':frame,'side':side,**{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in info.items()},
                        'image_fit_valid':bool(image_valid[frame,hand]),'strict_valid':bool(side_valid[frame,hand]),
                        'contact_tip_signed_distance_m':contact[frame,hand].tolist(),
                        'contact_geometry_strict_valid':bool(boxes['valid'][frame]),'wrist_yaw_refinement_accepted':bool(yaw_accepted[frame,hand]),
                        'wrist_yaw_position_rad':float(yaw_positions[frame,hand]),'wrist_origin_world_shift_m':float(wrist_world_shift[frame,hand]),
                        'wrist_keypoint_error_before_px':float(wrist_error_before[frame,hand]),'wrist_keypoint_error_after_px':float(wrist_error_after[frame,hand]),
                        'contact_geometry_uncertain':not bool(boxes['valid'][frame]),
                        'hand_surface_min_signed_distance_m':float(surface_min[frame,hand]),'hand_surface_fraction_beyond_5mm':float(surface_fraction[frame,hand]),
                        'new_hand_candidate_accepted':bool(hand_update_accepted[frame,hand]),'dense_collision_before':dense_penetration_before[frame,hand].tolist(),'dense_collision_after':dense_penetration_after[frame,hand].tolist(),
                        'heldout_edge_mean_before_px':float(before_heldout[frame,hand]),'heldout_edge_mean_after_px':float(after_heldout[frame,hand])})
            frame_observations.append(obs)
        evidence.extend(row)
        if frame%10==0 or frame==120:print(f"ROHand optimized frame {frame+1}/121; observed controls {int(act_observed[frame].sum())}/12",flush=True)
    yaw_speed=np.abs(np.diff(yaw_positions[frame_ids],axis=0)/np.diff(timestamps[frame_ids])[:,None]) if len(frame_ids)>1 else np.zeros((0,2))
    if args.wrist_velocity_limit is not None and np.any(yaw_speed>args.wrist_velocity_limit+1e-6):raise ValueError('Final true yaw sequence violates velocity limit; no output published')
    lower=np.asarray(coupling.lower);ranges=np.asarray(coupling.upper)-lower;actuators=lower+normalized*ranges
    if initial_hand is not None:
        # Preserve source values bit-for-bit, avoiding a normalize/denormalize
        # round-trip on every side whose proposed image update was rejected.
        for hand in range(2):
            rejected=~hand_update_accepted[:,hand];sl=slice(hand*6,(hand+1)*6)
            actuators[rejected,sl]=initial_hand['actuator_positions'][rejected,sl]
    expanded=coupling.expand_array(actuators[frame_ids])
    joint_observed=np.zeros((121,50),bool)
    for j,name in enumerate(coupling.joint_names):
        side=0 if name.startswith('left_') else 1;suffix=name.split('_hand_ro_')[1]
        if suffix.startswith('th_'):
            control=side*6+(5 if suffix=='th_root_link' else 0)
            joint_observed[:,j]=act_observed[:,control]
        else:
            finger={'if':1,'mf':2,'rf':3,'lf':4}[suffix[:2]];joint_observed[:,j]=act_observed[:,side*6+finger]
    suffix='_keyframes' if args.keyframes else ''
    if args.refine_wrist_yaw:suffix+='_wrist_refined'
    if args.refine_wrist_all:suffix+='_wrist_all'
    if args.output_suffix:suffix='_'+args.output_suffix
    path=output/('hand_trajectory'+suffix+'.npz')
    refined_body=np.stack([body_q_by_frame[int(f)].copy() for f in frame_ids])
    expanded_wrist_names=[side+'_wrist_'+axis+'_joint' for side in ('left','right') for axis in ('roll','pitch','yaw')]
    for hand,side in enumerate(('left','right')):
        for axis,name in enumerate(expanded_wrist_names[hand*3:hand*3+3]):refined_body[:,body_joint_names.index(name)]=all_wrist_positions[frame_ids,hand,axis]
    np.savez_compressed(path,actuator_names=np.asarray(coupling.actuator_names),actuator_positions=actuators[frame_ids],actuator_units=np.asarray(coupling.actuator_units),
                        actuator_std=(uncertainty*ranges)[frame_ids],actuator_observed=act_observed[frame_ids],actuator_uncertain=~act_observed[frame_ids],
                        joint_names=np.asarray(coupling.joint_names),joint_positions=expanded,joint_units=np.asarray(coupling.joint_units),
                        observed=joint_observed[frame_ids],uncertain=~joint_observed[frame_ids],valid=side_valid[frame_ids].all(axis=1),side_valid=side_valid[frame_ids],image_fit_valid=image_valid[frame_ids],
                        prior_only=~joint_observed[frame_ids],timestamps=timestamps[frame_ids],frame_indices=frame_ids,side_names=np.asarray(['left','right']),
                        contact_tip_signed_distance_m=contact[frame_ids],contact_prior_used=prior_used[frame_ids],contact_image_conflict=conflict[frame_ids],
                        contact_geometry_uncertain=~boxes['valid'][frame_ids],contact_tip_penetration_beyond_4mm_m=np.maximum(-contact[frame_ids]-.004,0),
                        contact_tip_link_names=np.asarray(list(models[0].tip_local)),
                        hand_surface_min_signed_distance_m=surface_min[frame_ids],hand_surface_fraction_beyond_5mm=surface_fraction[frame_ids],
                        contact_nonpenetration_sample_valid=(surface_min[frame_ids]>=-.005)&boxes['valid'][frame_ids,None],
                        wrist_velocity_limit_rad_s=np.asarray(args.wrist_velocity_limit if args.wrist_velocity_limit is not None else float('nan')),
                        new_hand_candidate_accepted=hand_update_accepted[frame_ids],dense_collision_metrics_before=dense_penetration_before[frame_ids],dense_collision_metrics_after=dense_penetration_after[frame_ids],
                        mano_prior_used=mano_seed_used[frame_ids],mano_prior_sha256=np.asarray(digest(mano_path) if mano is not None else ''),
                        body_wrist_origins_fixed=np.asarray(not args.refine_wrist_all),body_wrist_transforms_fixed=np.asarray(not (args.refine_wrist_yaw or args.refine_wrist_all)),
                        wrist_joint_names=np.asarray(expanded_wrist_names),wrist_joint_positions_rad=all_wrist_positions[frame_ids].reshape(len(frame_ids),6),
                        wrist_observed=all_wrist_observed[frame_ids].reshape(len(frame_ids),6),wrist_uncertain=~all_wrist_observed[frame_ids].reshape(len(frame_ids),6),
                        wrist_yaw_positions_rad=yaw_positions[frame_ids],
                        wrist_refinement_accepted=yaw_accepted[frame_ids],body_joint_names=np.asarray(body_joint_names),body_joint_positions_rad_refined=refined_body,
                        wrist_keypoint_error_before_px=wrist_error_before[frame_ids],wrist_keypoint_error_after_px=wrist_error_after[frame_ids],
                        all_wrist_joint_names=np.asarray(expanded_wrist_names),all_wrist_joint_positions_rad=all_wrist_positions[frame_ids].reshape(len(frame_ids),6),wrist_origin_world_shift_m=wrist_world_shift[frame_ids],
                        heldout_edge_mean_before_px=before_heldout[frame_ids],heldout_edge_mean_after_px=after_heldout[frame_ids],
                        body_sha256=np.asarray(digest(body_path)),source_body_sha256=np.asarray(digest(body_path)),camera_sha256=np.asarray(digest(camera_path)),
                        urdf_sha256=np.asarray(digest(ROOT/scene['robot']['urdf'])),coupling_math_sha256=np.asarray(digest(coupling.math_path)),
                        source_hand_sha256=np.asarray(digest(args.initial_hand_file or args.source_hand_file) if (args.initial_hand_file or args.source_hand_file) else ''),
                        box_sha256=np.asarray(digest(box_path)),source_video_sha256=data['video_sha256'])
    report={'status':'sparse_keyframe_debug_only' if args.keyframes else 'full_sequence_five_finger_candidates_with_uncertainty','frames':len(frame_ids),'frame_indices':frame_ids.tolist(),'independent_actuators':12,'expanded_mixed_unit_joints':50,
            'physical_hardware_validated':False,'official_linkage_used':True,'coupling_domain':coupling.domain_review(),
            'body_wrist_origins_fixed':not args.refine_wrist_all,'explicit_bounded_wrist_yaw_refinement':args.refine_wrist_yaw,'explicit_bounded_wrist_three_axes_refinement':args.refine_wrist_all,'wrist_refinements_accepted':int(yaw_accepted.sum()),
            'heldout_policy':'Every fifth spatial 6px edge block withheld; semantic mask remains shared. Refinement accepted only with training improvement and heldout no worse by more than .25px.',
            'quality_thresholds':{'weighted_training_proxy_rms_px_max':4.,'independent_spatial_heldout_edge_mean_px_max':4.5},
            'data_rms_px_note':'Weighted simplified-link training proxy only; not a full-mesh overlay error or evidence all fingers are observed',
            'contact_is_weak_prior_not_measurement':True,'image_observations':'Actual RGB/SAM robot silhouettes and Canny edges in independently tracked palm ROIs; far/occluded hand view is not fabricated',
            'full_hand_visual_surface_nonpenetration':args.full_mesh_contact,'contact_sampling':'Up to 256 deterministic vertices/triangle-centers for every visual link; independent full-vertex audit still required',
            'wrist_velocity_limit_rad_s':args.wrist_velocity_limit,'wrist_velocity_bound_policy':'Actual image optimization within intersection of previous final yaw and next source yaw +/- limit*dt; rejected frames retain source angles',
            'max_wrist_yaw_speed_rad_s':float(yaw_speed.max()) if yaw_speed.size else None,
            'initial_hand_fallback_policy':'Exact same-frame input controls and current body yaw retained on missing observations or rejected image/dense-collision update' if args.initial_hand_file else None,
            'new_hand_candidates_accepted':int(hand_update_accepted.sum()),'dense_collision_metrics':'All visual mesh vertices plus every triangle center: max penetration m, fraction >5mm, mean penetration beyond5mm m. Every metric must not worsen relative to current body + initial hand.',
            'mano_prior_used_as_measurement':False,'strict_side_valid_counts':side_valid.sum(0).tolist(),'image_fit_valid_side_counts':image_valid.sum(0).tolist(),
            'mano_initialization':args.mano_initialization,'mano_seed_count':int(mano_seed_used.sum()),'mano_prior_description':'Local rotation angular magnitude per finger retargeted through official linkage LUT; mixes flexion/splay, nonmetric weak prior, thumbroot retained',
            'actuator_observed_counts':act_observed.sum(0).tolist(),'unobserved_policy':'Finite regularized candidates are retained for rendering, but uncertain/prior_only remain true; local Jacobian is not a proof of 3D identifiability',
            'contact_conflicts':int(conflict.sum()),'elapsed_seconds':time.monotonic()-start,
            'files':{str(p.relative_to(ROOT)):digest(p) for p in (path,body_path,camera_path,box_path)},'frames_and_sides':evidence}
    if args.source_hand_file:report['source_hand_comparison_parent']={'path':str(args.source_hand_file),'sha256':digest(args.source_hand_file),'used_as_measurement':False}
    if args.initial_hand_file:report['initial_hand_parent']={'path':str(args.initial_hand_file),'sha256':digest(args.initial_hand_file),'used_as_measurement':False,'rejected_controls_retained_bit_exact':True}
    report=clean_json(report)
    report_file=ROOT/('reports/user-hand-optimization'+(suffix if args.keyframes or args.output_suffix else '')+'.json')
    (output/('hand_trajectory'+suffix+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    report_file.write_text(json.dumps(report,indent=2)+'\n')
    preview_path=ROOT/('outputs/previews/user_rohand_optimization'+suffix+'.mp4')
    with imageio.get_writer(str(preview_path),fps=24,codec='libx264',macro_block_size=1,ffmpeg_params=['-crf','20']) as writer:
        for frame in frame_ids:
            image=np.concatenate([cv2.cvtColor(rgb[frame,v],cv2.COLOR_RGB2BGR) for v in range(2)],axis=1)
            insets=[]
            for hand,(side,view,idx) in enumerate((('left',1,7),('right',0,3))):
                model=models[hand];base=base_tf(frame,hand);sl=slice(hand*6,(hand+1)*6)
                if yaw_accepted[frame,hand]:
                    if args.refine_wrist_all:
                        yb=Wrist3Base(int(frame),hand);base=yb((all_wrist_positions[frame,hand]-yb.low)/yb.range)
                    else:
                        yb=YawBase(int(frame),hand);base=yb((yaw_positions[frame,hand]-yb.low)/yb.range)
                points,_,_=boundary_geometry(model,normalized[frame,sl],base,P[view]);colour=(60,220,70) if image_valid[frame,hand] else (20,140,255)
                for point in points:
                    uv=np.clip(np.round(point),-10000,10000).astype(int);cv2.circle(image,(int(uv[0])+view*640,int(uv[1])),1,colour,-1)
                palm=landmark_positions[frame,view,idx]
                if not np.isfinite(palm).all():palm=np.asarray([320,320])
                x,y=np.round(palm).astype(int);x0,x1=max(0,x-55),min(640,x+55);y0,y1=max(0,y-45),min(720,y+45)
                crop=image[y0:y1,view*640+x0:view*640+x1];crop=cv2.resize(crop,(640,300))
                cv2.putText(crop,f'{side} ROHand: {int(act_observed[frame,sl].sum())}/6 controls image-constrained',(8,24),cv2.FONT_HERSHEY_SIMPLEX,.62,colour,1)
                insets.append(crop)
            canvas=np.concatenate([image,np.concatenate(insets,axis=1)],axis=0)
            wrist_label='explicit bounded wrist refinement' if args.refine_wrist_yaw or args.refine_wrist_all else 'wrists fixed'
            cv2.putText(canvas,f'frame {frame:03d}: 5-finger CANDIDATE | uncertainty retained | {wrist_label}',(10,22),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,255),1)
            writer.append_data(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB))
    report['preview']={'path':str(preview_path.relative_to(ROOT)),'sha256':digest(preview_path),'frames':len(frame_ids)}
    report_file.write_text(json.dumps(report,indent=2)+'\n')
    (output/('hand_trajectory'+suffix+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('frames_and_sides','coupling_domain')}),flush=True)


if __name__=='__main__':main()
