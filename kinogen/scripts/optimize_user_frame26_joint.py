#!/usr/bin/env python3
"""Image-constrained, CPU G1 sole/floor feasibility and trajectory fitting.

The authored floor stays z=0. Only the robot floating base, legs and arms may
change; waist, camera, box, meshes, mounts and coupled hand controls are fixed.
Foot tracking material points are 2-D evidence, never sole-height measurements.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def clean(x):
    import math
    if isinstance(x,dict):return {str(k):clean(v)for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v)for v in x]
    if hasattr(x,'tolist'):return clean(x.tolist())
    if isinstance(x,float)and not math.isfinite(x):return None
    return x


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all-frames',action='store_true')
    parser.add_argument('--frames',default='0,27,54,60,120')
    parser.add_argument('--outer',type=int,default=3)
    parser.add_argument('--maxiter',type=int,default=90)
    parser.add_argument('--starts',type=int,default=2)
    parser.add_argument('--retry-from',type=Path,default=ROOT/'outputs/user_reconstruction_v3/ground_trajectory_retry.npz')
    parser.add_argument('--stage-hand',type=Path,required=True)
    parser.add_argument('--prehand-body',type=Path,required=True)
    parser.add_argument('--budget-seconds',type=float,default=180.)
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--retry-report',type=Path,default=ROOT/'reports/user-ground-optimization-retry.json')
    parser.add_argument('--landmark-scale',type=float,default=1.5)
    parser.add_argument('--allow-waist',action='store_true')
    parser.add_argument('--foot-contour-weight',type=float,default=1.5)
    parser.add_argument('--hand-edge-weight',type=float,default=.65)
    args=parser.parse_args();args.frames='26';args.all_frames=False;args.allow_waist=True;args.outer=min(args.outer,4);args.maxiter=min(args.maxiter,160);args.starts=min(args.starts,2);args.budget_seconds=min(args.budget_seconds,180.);args.landmark_scale=2.;os.environ['CUDA_VISIBLE_DEVICES']='';os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
    import numpy as np
    import cv2
    import open3d as o3d
    import trimesh
    import xml.etree.ElementTree as ET
    from scipy.optimize import minimize
    from scipy.spatial import cKDTree,ConvexHull
    from scipy.spatial.transform import Rotation
    from user_body_kinematics import G1Kinematics,project_points
    from user_rohand_coupling import ROHandCoupling
    from optimize_user_hands import observation
    from hand_geometry import polygon_samples,ROHandGeometry
    from user_hand_observations_v3 import make_observation
    from optimize_user_hands import image_residual as legacy_image_residual,heldout_error as hand_heldout_error
    cv2.setNumThreads(2);started=time.monotonic();v2=ROOT/'outputs/user_reconstruction_v2';src=ROOT/'outputs/user_reconstruction';out=ROOT/'outputs/user_reconstruction_v3';out.mkdir(parents=True,exist_ok=True)
    paths={'body':v2/'body_trajectory_contact.npz','hand':v2/'hand_trajectory_contact.npz','camera':v2/'camera_refinement.npz','box':v2/'box_trajectory.npz','landmarks':v2/'body_landmark_observations.npz','geometry':src/'geometry.npz','tracking':src/'tracking.npz','input':src/'input.npz','segmentation':src/'segmentation.npz','scene':ROOT/'data/video_generation/candidate_scene/scene_rohand.json','reference_scene':ROOT/'data/video_generation/candidate_scene/scene.json'}
    paths['body']=args.prehand_body.resolve();paths['hand']=args.stage_hand.resolve();paths['hand_observation']=ROOT/'scripts/user_hand_observations_v3.py';paths['hand_geometry']=ROOT/'scripts/hand_geometry.py';paths['coupling_math']=ROOT/'dependencies/rohand-urdf/scripts/FingerMathURDF.py';paths['retry_from']=args.retry_from.resolve();paths['retry_report']=args.retry_report.resolve()
    hashes={key:sha(path)for key,path in paths.items()}
    def load(p):
        with np.load(p)as d:return {k:d[k]for k in d.files}
    C,H,cam,box,L,G,T,V,S=[load(paths[k])for k in ('body','hand','camera','box','landmarks','geometry','tracking','input','segmentation')]
    if str(H['observation_module_sha256'])!=hashes['hand_observation']:raise RuntimeError('Hand image observation module changed after stageH')
    if not np.array_equal(H['timestamps'],C['timestamps']):raise RuntimeError('StageB/H timestamps differ')
    if str(H['source_body_sha256'])!=hashes['body']:raise RuntimeError('Final hand contact is not bound to final C')
    if str(H['camera_sha256'])!=hashes['camera']or str(H['box_sha256'])!=hashes['box']:raise RuntimeError('Frozen contact camera/box binding mismatch')
    if not np.array_equal(S['observation_valid'],~V['watermark_invalid']):raise RuntimeError('Stale final SAM validity')
    kin=G1Kinematics(ROOT,scene_path=paths['scene']);coupling=ROHandCoupling(ROOT);P=cam['P_video'];times=C['timestamps'];frames=len(times);schedule=list(range(frames))if args.all_frames else [int(f)for f in args.frames.split(',')]
    if args.all_frames:
        raise RuntimeError('Retry supports explicit failed frames only; never re-fit the full sequence')
        feasibility=ROOT/'reports/user-ground-optimization-keyframes.json'
        if not feasibility.exists()or not json.loads(feasibility.read_text())['all_tested_ground_image_feasible']:raise RuntimeError('Full trajectory requires passing actual-image keyframe feasibility first')
    initial=load(paths['retry_from']);parent_report=json.loads(paths['retry_report'].read_text())
    if list(initial['frame_indices'])!=list(range(frames)):raise RuntimeError('Retry parent must be complete121 frames')
    if parent_report['output']['sha256']!=hashes['retry_from']:raise RuntimeError('Parent ground report SHA mismatch')
    if any(f not in parent_report['rejected_frames']for f in schedule):raise RuntimeError('Only actually failed frames may be re-fit')
    state0=np.column_stack((H['base_xyz_m'],Rotation.from_quat(H['base_quaternion_xyzw']).as_rotvec(),H['body_joint_positions_rad_refined']));states=state0.copy();waist=np.asarray([kin.body_index[n]for n in ('waist_yaw_joint','waist_roll_joint','waist_pitch_joint')]);variables=np.asarray([j for j in range(35)if j not in (waist+6)])
    states=state0.copy()
    if args.allow_waist:variables=np.arange(35)
    limits=kin.body_limits;xml=ET.parse(kin.urdf_path).getroot();speeds={j.attrib['name']:float(j.find('limit').attrib['velocity'])for j in xml.findall('joint')if j.attrib['name']in kin.body_index};velocity=np.asarray([speeds[n]for n in kin.body_joint_names]);wrist_ids=[kin.link_index[s+'_wrist_yaw_link']for s in ('left','right')]
    reference=json.loads(paths['reference_scene'].read_text());reference_tf=reference['robot']['link_world_transforms'];oldnames=G['link_names'].tolist();foot_names=[oldnames[int(i)]for i in G['shared_link_index']];foot_links=np.asarray([kin.link_index[n]for n in foot_names]);foot_local=[]
    for world,name in zip(G['shared_robot_points_world'],foot_names):
        tf=np.asarray(reference_tf[name]);foot_local.append((world-tf[:3,3])@tf[:3,:3])
    foot_local=np.asarray(foot_local);foot_side=np.asarray([0 if n.startswith('left')else 1 for n in foot_names]);foot_train=np.arange(len(foot_links))%5!=0;foot_xy=T['shared_positions_xy'];foot_valid=T['shared_observation_valid'];obs=L['positions_xy'];obs_valid=L['valid']
    landmark_links=np.asarray([kin.link_index[n]for n in ('right_shoulder_roll_link','right_elbow_link','right_wrist_yaw_link','right_wrist_yaw_link','left_shoulder_roll_link','left_elbow_link','left_wrist_yaw_link','left_wrist_yaw_link')]);landmark_local=np.zeros((8,3))
    raw_visuals=[];foot_parts=[[],[]];hand_parts=[[],[]];mesh_records=[]
    for visual in kin.visuals:
        name=visual['link_name'];link=visual['link_index'];vertices=visual['vertices_local'];unique=np.unique(np.round(vertices,10),axis=0);raw_visuals.append((link,unique));centres=vertices[visual['faces']].mean(1)
        if '_ankle_roll_link'not in name and '_hand_ro_'not in name:continue
        full=np.concatenate((unique,centres));count=1024 if '_ankle_roll_link'in name else 128;sample=full[np.linspace(0,len(full)-1,min(count,len(full))).astype(int)];side=0 if name.startswith('left')else 1
        if '_ankle_roll_link'in name:
            tf=np.asarray(reference_tf[name]);q0z=full@tf[2,:3]+tf[2,3];sole=full[q0z<=q0z.min()+.00075]
            if len(sole)<10:raise RuntimeError('Authored sole region is ambiguous')
            foot_parts[side].append({'link':link,'full':full,'sample':sample,'sole':sole})
        else:hand_parts[side].append({'name':name,'link':link,'full':full,'sample':sample})
        mesh_records.append({'link':name,'mesh_sha256':visual['mesh_sha256'],'unique_vertices':len(unique),'triangle_centres':len(centres),'optimization_samples':len(sample)})
    table=kin.scene['table'];static=[]
    for size,center in [(table['top_size_xyz_m'],table['top_center_xyz_m'])]+[([.035,.035,.735],[x,y,.3675])for x in (.19,.77)for y in (-.355,.355)]:
        mesh=trimesh.creation.box(extents=size);mesh.apply_translation(center);static.append(mesh)
    tablemesh=trimesh.util.concatenate(static);table_v=o3d.core.Tensor(np.asarray(tablemesh.vertices,dtype=np.float32));table_f=o3d.core.Tensor(np.asarray(tablemesh.faces,dtype=np.uint32));hh,ww=240,213;sx,sy=640/ww,720/hh;xx,yy=np.meshgrid((np.arange(ww)+.5)*sx,(np.arange(hh)+.5)*sy);rays=[]
    for view in range(2):
        M=P[view,:,:3];camera=-np.linalg.solve(M,P[view,:,3]);directions=np.stack((xx,yy,np.ones_like(xx)),axis=-1)@np.linalg.inv(M).T;directions/=np.linalg.norm(directions,axis=-1,keepdims=True);rays.append(o3d.core.Tensor(np.concatenate((np.broadcast_to(camera,directions.shape),directions),axis=-1).astype(np.float32)))
    chosen_faces=np.asarray(['_hand_ro_'not in kin.link_names[i]for i in kin.proxy_face_links]);faces=kin.proxy_faces[chosen_faces];face_links=kin.proxy_face_links[chosen_faces]
    def domain(frame,view,pixels):
        keep=np.ones(len(pixels),bool)
        for k,radius in ((2+4*view,55.),(3+4*view,35.)):
            if obs_valid[frame,view,k]:keep &= np.linalg.norm(pixels-obs[frame,view,k],axis=1)>radius
        return keep
    def targets(frame):
        result=[]
        for view in range(2):
            mask=S['robot'][frame,view]&~V['watermark_invalid'][frame,view];contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE);pixels=np.concatenate([c[:,0]for c in contours if len(c)>4]).astype(float)+.5;pixels=pixels[domain(frame,view,pixels)]
            roi=[]
            for side in range(2):
                good=foot_valid[frame,view]&(foot_side==side)&foot_train;points=foot_xy[frame,view,good]
                if len(points)<3:raise RuntimeError('Insufficient genuine foot observations for local ROI')
                low=points.min(0)-[24.,24.];high=points.max(0)+[24.,24.];roi.append(np.r_[low,high])
            def inside(p,rect):return (p[:,0]>=rect[0])&(p[:,0]<=rect[2])&(p[:,1]>=rect[1])&(p[:,1]<=rect[3])
            foot_masks=[inside(pixels,r)for r in roi];union=foot_masks[0]|foot_masks[1];g16=(pixels[:,0]//16).astype(int)+64*(pixels[:,1]//16).astype(int);g8=(pixels[:,0]//8).astype(int)+128*(pixels[:,1]//8).astype(int);body_held=g16%5==0;foot_held=union&(g8%5==0);training=~(body_held|foot_held)
            train=pixels[training]
            if len(train)>650:train=train[np.linspace(0,len(train)-1,650).astype(int)]
            dom=domain(frame,view,np.column_stack((xx.ravel(),yy.ravel()))).reshape(hh,ww);result.append({'train':train,'heldout':pixels[body_held],'foot_heldout':[pixels[foot_masks[s]&foot_held]for s in range(2)],'roi':roi,'inside':inside,'mask':(cv2.resize(mask.astype(np.uint8),(ww,hh),interpolation=cv2.INTER_NEAREST)>0)&dom,'domain':dom})
        return result
    def render(state,frame):
        scene=o3d.t.geometry.RaycastingScene(nthreads=4);scene.add_triangles(o3d.core.Tensor(kin.proxy_world(state).astype(np.float32)),o3d.core.Tensor(faces));scene.add_triangles(table_v,table_f);result=[]
        for view in range(2):
            cast=scene.cast_rays(rays[view],nthreads=4);hit=cast['geometry_ids'].numpy()==0;hit &= ~(cv2.resize(S['box'][frame,view].astype(np.uint8),(ww,hh),interpolation=cv2.INTER_NEAREST)>0);boundary=hit&~cv2.erode(hit.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool);py,px=np.nonzero(boundary);pixels=np.column_stack(((px+.5)*sx,(py+.5)*sy));keep=domain(frame,view,pixels);py,px,pixels=py[keep],px[keep],pixels[keep]
            if len(px)>650:take=np.linspace(0,len(px)-1,650).astype(int);py,px,pixels=py[take],px[take],pixels[take]
            triangle=cast['primitive_ids'].numpy()[py,px].astype(int);uv=cast['primitive_uvs'].numpy()[py,px];local=kin.proxy_vertices[faces[triangle,0]]*(1-uv.sum(1))[:,None]+kin.proxy_vertices[faces[triangle,1]]*uv[:,0,None]+kin.proxy_vertices[faces[triangle,2]]*uv[:,1,None];result.append({'mask':hit,'xy':pixels,'links':face_links[triangle],'local':local})
        return result
    def box_sdf(world,frame,jac=False):
        tf=box['box_world_transform'][frame];local=(world-tf[:3,3])@tf[:3,:3];q=np.abs(local)-box['box_size_xyz_m']/2;outside=np.maximum(q,0);length=np.linalg.norm(outside,axis=1);distance=length+np.minimum(q.max(1),0)
        if not jac:return distance
        g=np.zeros_like(local);ext=length>1e-12;g[ext]=outside[ext]/length[ext,None]*np.sign(local[ext]);ix=np.flatnonzero(~ext);axis=q.argmax(1);g[ix,axis[ix]]=np.sign(local[ix,axis[ix]]);return distance,g@tf[:3,:3].T
    def geometry(frame,actuators=None):
        finger=coupling.expand_one(H['actuator_positions'][frame]if actuators is None else actuators);tf=kin.fk(state0[frame],finger);hands=[]
        for side in range(2):
            wrist=tf[wrist_ids[side]];full=[];sample=[];outlines=[]
            for part in hand_parts[side]:
                link=part['link'];R=wrist[:3,:3].T@tf[link,:3,:3];t=wrist[:3,:3].T@(tf[link,:3,3]-wrist[:3,3]);full.append(part['full']@R.T+t);sample.append(part['sample']@R.T+t)
                if 'slider'not in part['name']and 'connecting'not in part['name']:outlines.append(part['sample']@R.T+t)
            hands.append({'full':np.concatenate(full),'sample':np.concatenate(sample),'outlines':outlines})
        return hands,finger
    def world_hand(state,side,points):
        tf=kin.fk(state)[wrist_ids[side]];return points@tf[:3,:3].T+tf[:3,3]
    def geometry_metrics(state,frame,hands):
        tf=kin.fk(state);feet=[]
        for side in range(2):
            z=[];sole=[]
            for part in foot_parts[side]:z.extend(part['full']@tf[part['link'],2,:3]+tf[part['link'],2,3]);sole.extend(part['sole']@tf[part['link'],2,:3]+tf[part['link'],2,3])
            z=np.asarray(z);sole=np.asarray(sole);feet.append({'raw_full_foot_min_z_m':float(z.min()),'raw_full_surface_samples':len(z),'sole_min_z_m':float(sole.min()),'sole_max_z_m':float(sole.max()),'sole_samples':len(sole),'raw_full_foot_fraction_below_5mm':float((z<-.005).mean())})
        hand=[]
        for side in range(2):
            d=box_sdf(world_hand(state,side,hands[side]['full']),frame);hand.append({'max_penetration_m':float(max(0,-d.min())),'min_sdf_m':float(d.min()),'fraction_deeper_than_5mm':float((d<-.005).mean())})
        robotmin=min(float((vertices@tf[link,2,:3]+tf[link,2,3]).min())for link,vertices in raw_visuals)
        return {'feet':feet,'hands':hand,'raw_all_robot_min_z_m':robotmin}
    def image_metrics(state,frame,target,hand_obs,hands):
        pred=render(state,frame);foot_world=kin.points(state,foot_links,foot_local);landmarks=kin.points(state,landmark_links,landmark_local);result=[]
        for view in range(2):
            xy=pred[view]['xy'];distance=cKDTree(xy).query(target[view]['heldout'])[0];mask=pred[view]['mask']&target[view]['domain'];truth=target[view]['mask'];good=obs_valid[frame,view].copy();good[[3,7]]=False;err=np.linalg.norm(project_points(P[view],landmarks)[good]-obs[frame,view,good],axis=1);wi=2+4*view;foot_uv=project_points(P[view],foot_world);held=foot_valid[frame,view]&~foot_train;tap=np.linalg.norm(foot_uv[held]-foot_xy[frame,view,held],axis=1);local=[]
            for side in range(2):
                region=target[view]['inside'](xy,target[view]['roi'][side]);model=xy[region];pixels=target[view]['foot_heldout'][side];d=cKDTree(model).query(pixels)[0]if len(model)and len(pixels)else np.asarray([np.inf]);local.append({'side':side,'heldout_pixels':len(pixels),'median_px':float(np.median(d)),'p90_px':float(np.percentile(d,90))})
            side=1-view;hand_error=None
            if hand_obs[side]is not None and len(hand_obs[side]['heldout_points']):
                boundaries=np.concatenate([polygon_samples(project_points(P[view],world_hand(state,side,p)),12)for p in hands[side]['outlines']]);hand_error=float(cKDTree(boundaries).query(hand_obs[side]['heldout_points'])[0].mean())
            named=[{'name':str(L['point_names'][k])if 'point_names'in L else ['right_shoulder','right_elbow','right_wrist','right_palm_roi_only','left_shoulder','left_elbow','left_wrist','left_palm_roi_only'][k],'pixel_observed':obs[frame,view,k].tolist(),'pixel_predicted':project_points(P[view],landmarks[[k]])[0].tolist(),'error_px':float(e)}for k,e in zip(np.flatnonzero(good),err)]
            result.append({'landmark_training':named,'view':view,'body_mask_iou':float((mask&truth).sum()/max((mask|truth).sum(),1)),'body_heldout_median_px':float(np.median(distance)),'body_heldout_p90_px':float(np.percentile(distance,90)),'actual_landmark_train_median_px':float(np.median(err))if len(err)else None,'actual_landmark_train_p90_px':float(np.percentile(err,90))if len(err)else None,'actual_wrist_train_error_px':float(np.linalg.norm(project_points(P[view],landmarks[[wi]])[0]-obs[frame,view,wi]))if obs_valid[frame,view,wi]else None,'tap_physical_id_heldout_count':int(held.sum()),'tap_heldout_median_px':float(np.median(tap))if len(tap)else np.inf,'tap_heldout_p90_px':float(np.percentile(tap,90))if len(tap)else np.inf,'local_foot_contour':local,'hand_heldout_mean_px':hand_error})
        return result,pred
    def acceptance(after,before,geo,oldgeo):
        failures=[]
        for s in range(2):
            if geo['feet'][s]['raw_full_foot_min_z_m']<-.002:failures.append(f'foot{s}_floor_penetration')
            if geo['feet'][s]['sole_min_z_m']>.008:failures.append(f'foot{s}_new_hover')
            if geo['hands'][s]['max_penetration_m']>max(.004,oldgeo['hands'][s]['max_penetration_m'])+.0005:failures.append(f'hand{s}_collision_worsened')
        if geo['raw_all_robot_min_z_m']<-.005:failures.append('whole_robot_floor')
        for a,b in zip(after,before):
            v=a['view']
            for k,absolute,extra in [('body_heldout_median_px',6,2),('body_heldout_p90_px',15,3),('tap_heldout_median_px',3,1.5),('tap_heldout_p90_px',6,2.5),('actual_landmark_train_median_px',8,2),('actual_landmark_train_p90_px',16,3),('actual_wrist_train_error_px',8,3)]:
                if a[k]is not None and (a[k]>absolute or (b[k]is not None and a[k]>b[k]+extra)):failures.append(f'view{v}_{k}')
            if a['body_mask_iou']<.55 or a['body_mask_iou']<b['body_mask_iou']-.04:failures.append(f'view{v}_body_iou')
            for fa,fb in zip(a['local_foot_contour'],b['local_foot_contour']):
                if fa['median_px']>4 or fa['median_px']>fb['median_px']+1.5 or fa['p90_px']>8 or fa['p90_px']>fb['p90_px']+2.5:failures.append(f'view{v}_foot{fa["side"]}_local_contour')
            if a['hand_heldout_mean_px']is not None and b['hand_heldout_mean_px']is not None and a['hand_heldout_mean_px']>b['hand_heldout_mean_px']+2:failures.append(f'view{v}_hand_heldout')
        return failures
    # This is one bounded experiment, only after the final full-sequence hand
    # parent is frozen. All states outside frame26 remain exact parent arrays.
    frame=26;deadline=time.monotonic()+args.budget_seconds
    if len(H['frame_indices'])!=121 or not np.array_equal(H['frame_indices'],np.arange(121)):
        raise RuntimeError('A complete frozen stageH is required')
    a_low=np.asarray(coupling.lower);a_range=np.asarray(coupling.upper)-a_low
    actuator0=H['actuator_positions'].copy();u0=(actuator0[frame]-a_low)/a_range
    models=[ROHandGeometry(ROOT,side)for side in ('left','right')]
    hand_base_ids=[kin.link_index[side+'_hand_ro_base_link']for side in ('left','right')]
    target=targets(frame);old_obs=[];new_obs=[]
    for side in range(2):
        view=1-side;pi=3+4*view;wi=pi-1
        palm=obs[frame,view,pi]if obs_valid[frame,view,pi]else np.full(2,np.nan)
        wrist=obs[frame,view,wi]if obs_valid[frame,view,wi]else np.full(2,np.nan)
        old_obs.append(observation(V['rgb'][frame,view],S['robot'][frame,view],S['box'][frame,view],~V['watermark_invalid'][frame,view],palm,1 if view==0 else -1))
        measured=make_observation(V['rgb'][frame,view],S['robot'][frame,view],S['box'][frame,view],~V['watermark_invalid'][frame,view],palm,wrist)
        if measured is not None:measured['points']=measured['training_all_points']
        new_obs.append(measured)
    def hand_metrics(state,normalized):
        transform=kin.fk(state,coupling.expand_one(a_low+normalized*a_range));result=[]
        for side in range(2):
            observations=[None,None];observations[1-side]=new_obs[side];sl=slice(side*6,side*6+6);base=transform[hand_base_ids[side]]
            rms=float(np.sqrt(np.mean(legacy_image_residual(models[side],normalized[sl],base,P,observations)**2))*3)
            held=float(hand_heldout_error(models[side],normalized[sl],base,P,observations))
            result.append({'side':side,'new_gray_training_proxy_rms_px':rms,'new_gray_heldout_proxy12_px':held,'strict_image_fit_valid':bool(new_obs[side]is not None and rms<4. and held<4.5),'training_points':len(new_obs[side]['points'])if new_obs[side]is not None else 0,'heldout_points':len(new_obs[side]['heldout_points'])if new_obs[side]is not None else 0})
        return result
    hands0,_=geometry(frame,actuator0[frame]);before_img,_=image_metrics(state0[frame],frame,target,old_obs,hands0);before_geo=geometry_metrics(state0[frame],frame,hands0);before_new=hand_metrics(state0[frame],u0)
    reference_state=state0[frame];low35=np.r_[reference_state[:3]-[.035,.035,.01],reference_state[3:6]-.12,limits[:,0]];high35=np.r_[reference_state[:3]+[.035,.035,.05],reference_state[3:6]+.12,limits[:,1]]
    actuator_velocity=[]
    mounted_joints={j.attrib['name']:j for j in xml.findall('joint')}
    for name in coupling.actuator_names:
        supplied=float(mounted_joints[name].find('limit').attrib['velocity'])
        # The public ROHand URDF supplies zero effort/velocity placeholders.
        # Only a positive value is an enforceable velocity specification.
        actuator_velocity.append(supplied if supplied>0 else np.nan)
    actuator_velocity=np.asarray(actuator_velocity);alow=a_low.copy();ahigh=a_low+a_range
    for neighbour in (25,27):
        dt=abs(times[frame]-times[neighbour]);step=velocity*dt;low35[6:]=np.maximum(low35[6:],state0[neighbour,6:]-step);high35[6:]=np.minimum(high35[6:],state0[neighbour,6:]+step)
        actuator_step=np.where(np.isfinite(actuator_velocity),actuator_velocity*dt,np.inf)
        alow=np.maximum(alow,actuator0[neighbour]-actuator_step);ahigh=np.minimum(ahigh,actuator0[neighbour]+actuator_step)
    lower=np.r_[low35,(alow-a_low)/a_range];upper=np.r_[high35,(ahigh-a_low)/a_range]
    if np.any(lower>upper+1e-10):raise RuntimeError('Frozen final neighbours have no velocity-feasible intersection')
    reference=np.r_[reference_state,u0];seeds=[reference.copy(),reference.copy()];seeds[1][2]+=max(0,-min(f['raw_full_foot_min_z_m']for f in before_geo['feet']))+.001
    trials=[];geometry_jacobian_check=None;optimization_started=time.monotonic()
    def assess(value):
        state=value[:35];u=value[35:];hands,_=geometry(frame,a_low+u*a_range);im,_=image_metrics(state,frame,target,old_obs,hands);geo=geometry_metrics(state,frame,hands);new=hand_metrics(state,u);fail=acceptance(im,before_img,geo,before_geo)
        for side,(a,b)in enumerate(zip(new,before_new)):
            if new_obs[side]is not None:
                if a['new_gray_training_proxy_rms_px']>b['new_gray_training_proxy_rms_px']+.03:fail.append(f'hand{side}_new_gray_training_worse')
                if a['new_gray_heldout_proxy12_px']>b['new_gray_heldout_proxy12_px']+.25:fail.append(f'hand{side}_new_gray_heldout_worse')
            if geo['hands'][side]['max_penetration_m']>before_geo['hands'][side]['max_penetration_m']+1e-6:fail.append(f'hand{side}_dense_collision_worse')
            oldgap=before_geo['hands'][side]['min_sdf_m']
            if oldgap<=.004 and geo['hands'][side]['min_sdf_m']>.004:fail.append(f'hand{side}_lost_near_surface')
        return im,geo,new,fail
    for seed_number,seed in enumerate(seeds[:args.starts]):
        value=np.clip(seed,lower,upper);iterations=[]
        for outer in range(args.outer):
            if time.monotonic()>deadline:break
            state=value[:35];controls=a_low+value[35:]*a_range;hands,_=geometry(frame,controls);predicted=render(state,frame);links=[];points=[];image_groups=[];hand_maps=[];offset=0
            def add(ids,local):
                nonlocal offset
                sl=slice(offset,offset+len(local));offset+=len(local);links.append(np.asarray(ids,int));points.append(np.asarray(local));return sl
            def add_hand(side,original_ids,original_local):
                sl=add(np.full(len(original_local),wrist_ids[side]),np.zeros_like(original_local));hand_maps.append((sl,side,np.asarray(original_ids,int),np.asarray(original_local)));return sl
            wrist_budget_groups=[]
            for view in range(2):
                pred=predicted[view];pixels=target[view]['train'];in_foot=np.zeros(len(pixels),bool)
                for roi in target[view]['roi']:in_foot|=target[view]['inside'](pixels,roi)
                weight=np.where(in_foot,1.5,.65);_,idx=cKDTree(pixels).query(pred['xy']);image_groups.append((view,add(pred['links'],pred['local']),pixels[idx],weight[idx]));_,idx=cKDTree(pred['xy']).query(pixels);image_groups.append((view,add(pred['links'][idx],pred['local'][idx]),pixels,weight))
                good=foot_valid[frame,view]&foot_train;image_groups.append((view,add(foot_links[good],foot_local[good]),foot_xy[frame,view,good],3.));good=obs_valid[frame,view].copy();good[[3,7]]=False;image_groups.append((view,add(landmark_links[good],landmark_local[good]),obs[frame,view,good],np.asarray([6.,10.,12.,0.,6.,10.,12.,0.])[good]))
                wi=2+4*view
                if obs_valid[frame,view,wi]:
                    budget=min(8.,before_img[view]['actual_wrist_train_error_px']+3.)
                    wrist_budget_groups.append((view,add(landmark_links[[wi]],landmark_local[[wi]]),obs[frame,view,wi],max(0.,budget-.1)))
            floor_groups=[];tf=kin.fk(state)
            for side in range(2):
                for part in foot_parts[side]:
                    z=part['full']@tf[part['link'],2,:3]+tf[part['link'],2,3];deep=np.argpartition(z,63)[:64];local=np.concatenate((part['sample'],part['full'][deep]));floor_groups.append(add(np.full(len(local),part['link']),local))
            collision_groups=[];tf_actual=kin.fk(state,coupling.expand_one(controls))
            for side in range(2):
                all_ids=[];all_local=[];sparse_ids=[];sparse_local=[];boundary_ids=[];boundary_local=[]
                for part in hand_parts[side]:
                    lid=part['link'];local=part['full'];all_ids.extend(np.full(len(local),lid));all_local.extend(local);take=np.linspace(0,len(local)-1,min(32,len(local))).astype(int);sparse_ids.extend(np.full(len(take),lid));sparse_local.extend(local[take])
                    if 'slider'in part['name']or 'connecting'in part['name']:continue
                    local=part['sample'];world=local@tf_actual[lid,:3,:3].T+tf_actual[lid,:3,3];uv=project_points(P[1-side],world)
                    try:take=ConvexHull(uv).vertices
                    except Exception:continue
                    boundary_ids.extend(np.full(len(take),lid));boundary_local.extend(local[take])
                all_ids=np.asarray(all_ids,int);all_local=np.asarray(all_local);world=np.empty_like(all_local)
                for lid in np.unique(all_ids):sel=all_ids==lid;world[sel]=all_local[sel]@tf_actual[lid,:3,:3].T+tf_actual[lid,:3,3]
                sdf=box_sdf(world,frame);deep=np.argpartition(sdf,255)[:256];ids=np.r_[np.asarray(sparse_ids,int),all_ids[deep]];local=np.concatenate((np.asarray(sparse_local),all_local[deep]));collision_groups.append((side,add_hand(side,ids,local)))
                if new_obs[side]is not None and len(boundary_ids)>3:
                    ids=np.asarray(boundary_ids,int);local=np.asarray(boundary_local);world=np.empty_like(local)
                    for lid in np.unique(ids):sel=ids==lid;world[sel]=local[sel]@tf_actual[lid,:3,:3].T+tf_actual[lid,:3,3]
                    uv=project_points(P[1-side],world);training=new_obs[side]['training_all_points'];_,match=cKDTree(uv).query(training);image_groups.append((1-side,add_hand(side,ids[match],local[match]),training,.8))
            ids=np.concatenate(links);fixed_local=np.concatenate(points)
            def dynamic_local(u):
                local=fixed_local.copy();tf=kin.fk(reference_state,coupling.expand_one(a_low+u*a_range))
                for sl,side,original_ids,original_local in hand_maps:
                    wrist=tf[wrist_ids[side]];out=local[sl]
                    for lid in np.unique(original_ids):
                        sel=original_ids==lid;relative_R=wrist[:3,:3].T@tf[lid,:3,:3];relative_t=wrist[:3,:3].T@(tf[lid,:3,3]-wrist[:3,3]);out[sel]=original_local[sel]@relative_R.T+relative_t
                return local
            prior=np.r_[np.full(3,150.),np.full(3,12.),np.full(29,4.),np.full(12,.2)];prior[waist+6]=20.
            def objective(x):
                state=x[:35];u=x[35:];local=dynamic_local(u);world,Jbody=kin.points(state,ids,local,True);J=np.zeros((len(local),3,47));J[:,:,:35]=Jbody;wrist_tf=kin.fk(state)
                for control in range(12):
                    plus=u.copy();minus=u.copy();plus[control]=min(1.,u[control]+1e-5);minus[control]=max(0.,u[control]-1e-5);derivative=(dynamic_local(plus)-dynamic_local(minus))/(plus[control]-minus[control])
                    for sl,side,_,_ in hand_maps:J[sl,:,35+control]=derivative[sl]@wrist_tf[wrist_ids[side],:3,:3].T
                loss=0.;gradient=np.zeros(47)
                for view,sl,pixels,weight in image_groups:
                    xy,pj=project_points(P[view],world[sl],True);jac=np.einsum('nij,njk->nik',pj,J[sl]);weight=np.asarray(weight);weight=weight[:,None]if weight.ndim else weight;res=(xy-pixels)*weight;root=np.sqrt(1+(res/4.)**2);loss+=float((32*(root-1)).sum());gradient+=np.einsum('ni,nik->k',2*res/root*weight,jac)
                for view,sl,pixel,budget in wrist_budget_groups:
                    xy,pj=project_points(P[view],world[sl],True);jac=np.einsum('nij,njk->nik',pj,J[sl])[0];error=xy[0]-pixel;radius=float(np.linalg.norm(error));hinge=max(0.,radius-budget);loss+=2000.*hinge*hinge
                    if radius>1e-12:gradient+=4000.*hinge*(error/radius)@jac
                for sl in floor_groups:
                    z=world[sl,2];pen=np.maximum(.0005-z,0)/.002;loss+=float(3000*np.mean(pen**2)+8000*pen.max()**2);der=-6000*pen/(.002*len(pen));der[int(pen.argmax())]-=16000*pen.max()/.002;gradient+=der@J[sl,2,:];near=int(z.argmin());gap=max(float(z[near])-.003,0)/.003;loss+=1500*gap*gap;gradient+=3000*gap/.003*J[sl,2,:][near]
                for side,sl in collision_groups:
                    sdf,g=box_sdf(world[sl],frame,True);pen=np.maximum(.0005-sdf,0)/.003;loss+=float(1500*np.mean(pen**2)+3000*pen.max()**2);der=-3000*pen/(.003*len(pen));der[int(pen.argmax())]-=6000*pen.max()/.003;gradient+=np.einsum('n,ni,nik->k',der,g,J[sl])
                regular=(x-reference)*prior;loss+=float(regular@regular);gradient+=2*regular*prior
                return loss,gradient
            if geometry_jacobian_check is None:
                # Independent directional finite difference tests the assembled
                # body+official-coupling gradient, including metre/radian scales.
                check=np.clip(value,lower+1e-7,upper-1e-7);direction=np.random.default_rng(26).normal(size=47);direction/=np.linalg.norm(direction);eps=1e-7;cost,gradient=objective(check);numeric=(objective(check+eps*direction)[0]-objective(check-eps*direction)[0])/(2*eps);analytic=float(gradient@direction);geometry_jacobian_check={'analytic_directional':analytic,'numeric_directional':numeric,'relative_error':abs(analytic-numeric)/max(1,abs(numeric))}
                if geometry_jacobian_check['relative_error']>3e-3:raise RuntimeError('Joint body/finger gradient check failed')
                if args.check_only:print(json.dumps(geometry_jacobian_check),flush=True);return
            def callback(x):
                if time.monotonic()>deadline:raise StopIteration
            fit=minimize(objective,value,jac=True,method='L-BFGS-B',bounds=list(zip(lower,upper)),callback=callback,options={'maxiter':args.maxiter,'ftol':1e-8,'gtol':1e-4,'maxls':25});value=fit.x;im,geo,new,fail=assess(value);iterations.append({'outer':outer,'nit':int(fit.nit),'objective':float(fit.fun),'failures':fail,'elapsed_seconds':time.monotonic()-optimization_started});print(json.dumps(clean({'seed':seed_number,'outer':outer,'failures':fail,'wrist_px':[v['actual_wrist_train_error_px']for v in im],'foot_min_z_m':[f['raw_full_foot_min_z_m']for f in geo['feet']],'elapsed_seconds':time.monotonic()-optimization_started})),flush=True)
            if not fail:break
        im,geo,new,fail=assess(value);trials.append({'value':value.copy(),'images':im,'geometry':geo,'new_hand_metrics':new,'failures':fail,'iterations':iterations})
        if not fail or time.monotonic()>deadline:break
    passing=[t for t in trials if not t['failures']];best=min(passing,key=lambda t:sum(v['tap_heldout_median_px']for v in t['images']))if passing else min(trials,key=lambda t:len(t['failures']))
    selected=bool(passing);chosen=best['value']if selected else reference;states[frame]=chosen[:35];actuators=actuator0.copy();actuators[frame]=a_low+chosen[35:]*a_range
    final_img,final_geo,final_new,final_fail=assess(chosen);q=states[:,6:];speed=np.abs(np.diff(q,axis=0))/np.diff(times)[:,None]
    if np.any(speed>velocity+1e-6)or np.any(q<limits[:,0]-1e-8)or np.any(q>limits[:,1]+1e-8):raise RuntimeError('Final original body q/velocity bounds failed')
    if not np.array_equal(q[np.arange(frames)!=26],H['body_joint_positions_rad_refined'][np.arange(frames)!=26]):raise RuntimeError('Nonframe26 body changed')
    if not np.array_equal(actuators[np.arange(frames)!=26],actuator0[np.arange(frames)!=26]):raise RuntimeError('Nonframe26 hand controls changed')
    base_xyz=H['base_xyz_m'].copy();base_quat=H['base_quaternion_xyzw'].copy()
    if selected:base_xyz[26]=states[26,:3];base_quat[26]=Rotation.from_rotvec(states[26,3:6]).as_quat()
    full_finger=coupling.expand_array(actuators);transforms=np.stack([kin.forward(base_xyz[f],base_quat[f],q[f],dict(zip(coupling.joint_names,full_finger[f])))for f in range(frames)])
    body={k:v.copy()for k,v in C.items()};body.update(base_xyz_m=base_xyz,base_quaternion_xyzw=base_quat,joint_positions_rad=q,link_world_transforms=transforms,link_names=np.asarray(kin.link_names),source_body_sha256=np.asarray(hashes['body']),source_hand_sha256=np.asarray(hashes['hand']),source_stage_hand_sha256=np.asarray(hashes['hand']),finger_actuator_positions=actuators,finger_actuator_names=np.asarray(coupling.actuator_names),finger_actuator_units=np.asarray(coupling.actuator_units),finger_joint_positions=full_finger,finger_joint_names=np.asarray(coupling.joint_names),finger_joint_units=np.asarray(coupling.joint_units),finger_joint_positions_included=np.asarray(True),frame26_joint_trial_accepted=np.asarray(selected),source_ground_parent_sha256=np.asarray(hashes['retry_from']))
    changed=np.abs(q-C['joint_positions_rad'])>1e-7
    if 'joint_image_supported'in body:body['joint_image_supported'][changed]=False
    if 'joint_approx_uncertainty_rad'in body:body['joint_approx_uncertainty_rad'][changed]=np.nan
    if selected:
        for key in ('accepted','ground_image_valid','ground_nonpenetration_valid','candidate_valid'):
            if key in body:body[key][26]=True
        for key,values in [('raw_foot_min_z_m',[f['raw_full_foot_min_z_m']for f in final_geo['feet']]),('sole_min_z_m',[f['sole_min_z_m']for f in final_geo['feet']]),('hand_max_penetration_m',[h['max_penetration_m']for h in final_geo['hands']])]:
            if key in body:body[key][26]=values
    body_path=out/'body_frame26_joint.npz';np.savez_compressed(body_path,**body)
    # Build a source-bound hand artifact. Remove every inherited pose-dependent
    # tip/surface/wrist/edge diagnostic instead of retaining stale parent values.
    stale_tokens=('contact_tip','hand_surface','contact_nonpenetration','wrist_keypoint_error','wrist_origin_world_shift','dense_collision','heldout_edge','legacy_observation','v3_proxy16')
    hand={k:v.copy()for k,v in H.items()if not any(token in k for token in stale_tokens)}
    hand.update(actuator_positions=actuators,joint_positions=full_finger,body_joint_positions_rad_refined=q,base_xyz_m=base_xyz,base_quaternion_xyzw=base_quat,source_hand_sha256=np.asarray(hashes['hand']),source_body_sha256=np.asarray(sha(body_path)),body_sha256=np.asarray(sha(body_path)),frame26_joint_trial_accepted=np.asarray(selected),source_stage_hand_sha256=np.asarray(hashes['hand']))
    if selected:
        for key in ('actuator_observed','observed','wrist_observed'):
            if key in hand:hand[key][26]=False
        for key in ('actuator_uncertain','uncertain','prior_only','wrist_uncertain'):
            if key in hand:hand[key][26]=True
        if 'actuator_std'in hand:hand['actuator_std'][26]=np.nan
        if 'image_fit_valid'in hand:hand['image_fit_valid'][26]=[m['strict_image_fit_valid']for m in final_new]
        for key in ('contact_prior_used','contact_image_conflict','mano_prior_used'):
            if key in hand:hand[key][26]=False
        if 'new_hand_candidate_selected'in hand:hand['new_hand_candidate_selected'][26]=True
        if 'new_hand_candidate_accepted'in hand:hand['new_hand_candidate_accepted'][26]=[m['strict_image_fit_valid']for m in final_new]
        if 'refinement_selected_conditional'in hand:hand['refinement_selected_conditional'][26]=[not m['strict_image_fit_valid']for m in final_new]
        for key in ('side_valid','valid'):
            if key in hand:hand[key][26]=False
    wrist_names=hand.get('wrist_joint_names',np.asarray([])).tolist()
    if wrist_names:
        wq=q[:,[kin.body_index[n]for n in wrist_names]];hand['wrist_joint_positions_rad']=wq;hand['all_wrist_joint_positions_rad']=wq
    if 'wrist_yaw_positions_rad'in hand:hand['wrist_yaw_positions_rad']=q[:,[kin.body_index[s+'_wrist_yaw_joint']for s in ('left','right')]]
    hand_path=out/'hand_frame26_joint.npz';np.savez_compressed(hand_path,**hand)
    report={'status':'single_finite_frame26_joint_trial_complete','accepted':selected,'frame':26,'inputs':{str(p.relative_to(ROOT)):hashes[k]for k,p in paths.items()},'outputs':{'body':{'path':str(body_path.relative_to(ROOT)),'sha256':sha(body_path)},'hand':{'path':str(hand_path.relative_to(ROOT)),'sha256':sha(hand_path)}},'script_sha256':sha(Path(__file__)),'body_q_base_and_hand_controls_source':'Actual frozen stageH refined body/base/actuators; source ground only supplies reference geometry and evidence policy.','variables':47,'settings':{'starts_max':args.starts,'outers_max':args.outer,'iterations_max':args.maxiter,'budget_seconds':args.budget_seconds},'gradient_check':geometry_jacobian_check,'before':{'images':before_img,'geometry':before_geo,'new_hand_metrics':before_new},'after':{'images':final_img,'geometry':final_geo,'new_hand_metrics':final_new,'failures':final_fail},'searched_trials':[{**{k:v for k,v in t.items()if k!='value'},'value47':t['value'].tolist()}for t in trials],'other120_body_q_base_and_hand_controls_exact_parent':True,'all_original_body_q_and_velocity_satisfied':True,'original_actuator_urdf_velocity_bounds_used':dict(zip(coupling.actuator_names,actuator_velocity.tolist())),'saved89tf_uses_declared_actual50mixedstates':True,'stale_hand_diagnostics_removed_matching_tokens':stale_tokens,'f25_f26_f27':{'base_xyz_m':base_xyz[25:28],'base_step_m':np.linalg.norm(np.diff(base_xyz[25:28],axis=0),axis=1),'base_speed_m_s':np.linalg.norm(np.diff(base_xyz[25:28],axis=0),axis=1)/np.diff(times[25:28]),'q_max_speed_rad_s':np.max(speed[25:27],axis=1)},'base_hardware_velocity_validated':False,'floor_z_m':0.,'floor_moved':False,'conditional_calibration_and_box':True,'physical_hardware_validated':False,'strict_reconstruction_valid':False,'elapsed_seconds':time.monotonic()-optimization_started}
    report_path=ROOT/'reports/user-frame26-joint.json';report_path.write_text(json.dumps(clean(report),indent=2)+'\n');print(json.dumps(clean({k:report[k]for k in ('status','accepted','outputs','f25_f26_f27','elapsed_seconds')}),indent=2),flush=True)


if __name__=='__main__':main()
