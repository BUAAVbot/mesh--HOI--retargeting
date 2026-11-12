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
    args=parser.parse_args();os.environ['CUDA_VISIBLE_DEVICES']='';os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
    import numpy as np
    import cv2
    import open3d as o3d
    import trimesh
    import xml.etree.ElementTree as ET
    from scipy.optimize import minimize
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation
    from user_body_kinematics import G1Kinematics,project_points
    from user_rohand_coupling import ROHandCoupling
    from optimize_user_hands import observation
    from hand_geometry import polygon_samples
    cv2.setNumThreads(2);started=time.monotonic();v2=ROOT/'outputs/user_reconstruction_v2';src=ROOT/'outputs/user_reconstruction';out=ROOT/'outputs/user_reconstruction_v3';out.mkdir(parents=True,exist_ok=True)
    paths={'body':v2/'body_trajectory_contact.npz','hand':v2/'hand_trajectory_contact.npz','camera':v2/'camera_refinement.npz','box':v2/'box_trajectory.npz','landmarks':v2/'body_landmark_observations.npz','geometry':src/'geometry.npz','tracking':src/'tracking.npz','input':src/'input.npz','segmentation':src/'segmentation.npz','scene':ROOT/'data/video_generation/candidate_scene/scene_rohand.json','reference_scene':ROOT/'data/video_generation/candidate_scene/scene.json'}
    hashes={key:sha(path)for key,path in paths.items()}
    def load(p):
        with np.load(p)as d:return {k:d[k]for k in d.files}
    C,H,cam,box,L,G,T,V,S=[load(paths[k])for k in ('body','hand','camera','box','landmarks','geometry','tracking','input','segmentation')]
    if str(H['source_body_sha256'])!=hashes['body']:raise RuntimeError('Final hand contact is not bound to final C')
    if str(H['camera_sha256'])!=hashes['camera']or str(H['box_sha256'])!=hashes['box']:raise RuntimeError('Frozen contact camera/box binding mismatch')
    if not np.array_equal(S['observation_valid'],~V['watermark_invalid']):raise RuntimeError('Stale final SAM validity')
    kin=G1Kinematics(ROOT,scene_path=paths['scene']);coupling=ROHandCoupling(ROOT);P=cam['P_video'];times=C['timestamps'];frames=len(times);schedule=list(range(frames))if args.all_frames else [int(f)for f in args.frames.split(',')]
    if args.all_frames:
        feasibility=ROOT/'reports/user-ground-optimization-keyframes.json'
        if not feasibility.exists()or not json.loads(feasibility.read_text())['all_tested_ground_image_feasible']:raise RuntimeError('Full trajectory requires passing actual-image keyframe feasibility first')
    state0=np.column_stack((C['base_xyz_m'],Rotation.from_quat(C['base_quaternion_xyzw']).as_rotvec(),H['body_joint_positions_rad_refined']));states=state0.copy();waist=np.asarray([kin.body_index[n]for n in ('waist_yaw_joint','waist_roll_joint','waist_pitch_joint')]);variables=np.asarray([j for j in range(35)if j not in (waist+6)])
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
    def geometry(frame):
        finger=coupling.expand_one(H['actuator_positions'][frame]);tf=kin.fk(state0[frame],finger);hands=[]
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
            result.append({'view':view,'body_mask_iou':float((mask&truth).sum()/max((mask|truth).sum(),1)),'body_heldout_median_px':float(np.median(distance)),'body_heldout_p90_px':float(np.percentile(distance,90)),'actual_landmark_train_median_px':float(np.median(err))if len(err)else None,'actual_landmark_train_p90_px':float(np.percentile(err,90))if len(err)else None,'actual_wrist_train_error_px':float(np.linalg.norm(project_points(P[view],landmarks[[wi]])[0]-obs[frame,view,wi]))if obs_valid[frame,view,wi]else None,'tap_physical_id_heldout_count':int(held.sum()),'tap_heldout_median_px':float(np.median(tap))if len(tap)else np.inf,'tap_heldout_p90_px':float(np.percentile(tap,90))if len(tap)else np.inf,'local_foot_contour':local,'hand_heldout_mean_px':hand_error})
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
    records=[];accepted=np.zeros(frames,bool);ground_valid=np.zeros(frames,bool);image_valid=np.zeros(frames,bool);transforms=np.zeros((frames,len(kin.link_names),4,4));allsole=np.full((frames,2),np.nan);allraw=allsole.copy();allhand=allsole.copy()
    for frame in schedule:
        hands,finger=geometry(frame);target=targets(frame);hand_obs=[]
        for side in range(2):
            view=1-side;pi=3+4*view;palm=obs[frame,view,pi]if obs_valid[frame,view,pi]else np.asarray([np.nan,np.nan]);hand_obs.append(observation(V['rgb'][frame,view],S['robot'][frame,view],S['box'][frame,view],~V['watermark_invalid'][frame,view],palm,1 if view==0 else -1))
        before_img,_=image_metrics(state0[frame],frame,target,hand_obs,hands);before_geo=geometry_metrics(state0[frame],frame,hands);reference_state=state0[frame];lower=np.r_[reference_state[:3]-[.035,.035,.01],reference_state[3:6]-.12,limits[:,0]];upper=np.r_[reference_state[:3]+[.035,.035,.05],reference_state[3:6]+.12,limits[:,1]]
        for other in (frame-1,frame+1):
            if 0<=other<frames:
                neighbour=states[other]if args.all_frames and other<frame else state0[other];step=velocity*abs(times[frame]-times[other]);lower[6:]=np.maximum(lower[6:],neighbour[6:]-step);upper[6:]=np.minimum(upper[6:],neighbour[6:]+step)
        if (lower[variables]>upper[variables]+1e-10).any():raise RuntimeError('Frozen neighbour velocity bounds inconsistent')
        seeds=[reference_state.copy()];lift=reference_state.copy();lift[2]+=max(0,-min(f['raw_full_foot_min_z_m']for f in before_geo['feet']))+.001;seeds.append(lift);trials=[]
        for seed in seeds[:args.starts]:
            state=seed.copy();state[variables]=np.clip(state[variables],lower[variables],upper[variables]);iterations=[]
            for outer in range(args.outer):
                predicted=render(state,frame);image_data=[]
                for view in range(2):
                    pred=predicted[view];pixels=target[view]['train'];_,idx=cKDTree(pixels).query(pred['xy']);image_data.append((view,pred['links'],pred['local'],pixels[idx],.65));_,idx=cKDTree(pred['xy']).query(pixels);image_data.append((view,pred['links'][idx],pred['local'][idx],pixels,.65));good=foot_valid[frame,view]&foot_train;image_data.append((view,foot_links[good],foot_local[good],foot_xy[frame,view,good],3.));good=obs_valid[frame,view].copy();good[[3,7]]=False;image_data.append((view,landmark_links[good],landmark_local[good],obs[frame,view,good],np.asarray([3.,5.,8.,0.,3.,5.,8.,0.])[good]))
                links=[];local=[];image_groups=[];offset=0
                def add(ids,points):
                    nonlocal offset
                    sl=slice(offset,offset+len(points));links.append(np.asarray(ids,dtype=int));local.append(points);offset+=len(points);return sl
                for view,ids,points,pixels,weight in image_data:image_groups.append((view,add(ids,points),pixels,weight))
                floor_groups=[];tf=kin.fk(state)
                for side in range(2):
                    for part in foot_parts[side]:
                        z=part['full']@tf[part['link'],2,:3]+tf[part['link'],2,3];deep=np.argpartition(z,min(63,len(z)-1))[:64];points=np.concatenate((part['sample'],part['full'][deep]));floor_groups.append(add(np.full(len(points),part['link']),points))
                collision_groups=[]
                for side in range(2):
                    d=box_sdf(world_hand(state,side,hands[side]['full']),frame);deep=np.argpartition(d,255)[:256];points=np.concatenate((hands[side]['sample'],hands[side]['full'][deep]));collision_groups.append((side,add(np.full(len(points),wrist_ids[side]),points)))
                ids=np.concatenate(links);points=np.concatenate(local);prior=np.r_[np.full(3,150.),np.full(3,12.),np.full(29,4.)][variables]
                def objective(value):
                    candidate=state.copy();candidate[variables]=value;world,J=kin.points(candidate,ids,points,True);J=J[:,:,variables];loss=0.;gradient=np.zeros(len(variables))
                    for view,sl,pixels,weight in image_groups:
                        xy,pj=project_points(P[view],world[sl],True);jac=np.einsum('nij,njk->nik',pj,J[sl]);weight=np.asarray(weight);weight=weight[:,None]if weight.ndim else weight;res=(xy-pixels)*weight;root=np.sqrt(1+(res/4.)**2);loss+=float((32*(root-1)).sum());gradient+=np.einsum('ni,nik->k',2*res/root*weight,jac)
                    for sl in floor_groups:
                        z=world[sl,2];pen=np.maximum(.0005-z,0)/.002;loss+=float(3000*np.mean(pen**2)+8000*pen.max()**2);der=-6000*pen/(.002*len(pen));worst=int(pen.argmax());der[worst]-=16000*pen[worst]/.002;gradient+=der@J[sl,2,:];near=int(z.argmin());gap=max(float(z[near])-.003,0)/.003;loss+=1500*gap*gap;gradient+=3000*gap/.003*J[sl,2,:][near]
                    for side,sl in collision_groups:
                        sdf,g=box_sdf(world[sl],frame,True);allowed=max(0,before_geo['hands'][side]['max_penetration_m']-.0005);pen=np.maximum(.0005-allowed-sdf,0)/.003;loss+=float(1500*np.mean(pen**2)+3000*pen.max()**2);der=-3000*pen/(.003*len(pen));worst=int(pen.argmax());der[worst]-=6000*pen[worst]/.003;gradient+=np.einsum('n,ni,nik->k',der,g,J[sl])
                    regular=(value-reference_state[variables])*prior;loss+=float(regular@regular);gradient+=2*regular*prior
                    return loss,gradient
                fit=minimize(objective,state[variables],jac=True,method='L-BFGS-B',bounds=list(zip(lower[variables],upper[variables])),options={'maxiter':args.maxiter,'ftol':1e-8,'gtol':1e-4,'maxls':30});state[variables]=fit.x;iterations.append({'outer':outer,'cost':float(fit.fun),'iterations':int(fit.nit),'success':bool(fit.success)})
                after_img,_=image_metrics(state,frame,target,hand_obs,hands);after_geo=geometry_metrics(state,frame,hands);reasons=acceptance(after_img,before_img,after_geo,before_geo)
                if not reasons:break
            trials.append({'state':state.copy(),'images':after_img,'geometry':after_geo,'failures':reasons,'iterations':iterations})
            if not reasons:break
        passes=[t for t in trials if not t['failures']];best=min(passes,key=lambda t:sum(i['tap_heldout_median_px']for i in t['images']))if passes else min(trials,key=lambda t:sum(max(0,-f['raw_full_foot_min_z_m'])for f in t['geometry']['feet'])+len(t['failures'])*.001)
        if passes:states[frame]=best['state'];accepted[frame]=True
        final_img,_=image_metrics(states[frame],frame,target,hand_obs,hands);final_geo=geometry_metrics(states[frame],frame,hands);transforms[frame]=kin.fk(states[frame],finger);ground_valid[frame]=all(f['raw_full_foot_min_z_m']>=-.002 for f in final_geo['feet'])and final_geo['raw_all_robot_min_z_m']>=-.005;image_valid[frame]=not [x for x in acceptance(final_img,before_img,final_geo,before_geo)if 'floor'not in x and 'hover'not in x and 'collision'not in x]
        for side in range(2):allsole[frame,side]=final_geo['feet'][side]['sole_min_z_m'];allraw[frame,side]=final_geo['feet'][side]['raw_full_foot_min_z_m'];allhand[frame,side]=final_geo['hands'][side]['max_penetration_m']
        record={'frame':frame,'accepted':bool(accepted[frame]),'before_geometry':before_geo,'after_geometry':final_geo,'before_images':before_img,'after_images':final_img,'best_searched_candidate':{k:v for k,v in best.items()if k!='state'},'trial_count':len(trials),'box_geometry_conditional':not bool(box['valid'][frame])};records.append(record);progress={'frame':frame,'completed':len(records),'scheduled':len(schedule),'accepted':bool(accepted[frame]),'before_foot_min_z_m':[f['raw_full_foot_min_z_m']for f in before_geo['feet']],'after_foot_min_z_m':allraw[frame].tolist(),'best_candidate_failures':best['failures'],'best_candidate_tap_heldout_median_px':[v['tap_heldout_median_px']for v in best['images']],'best_candidate_local_foot_heldout_median_px':[[f['median_px']for f in v['local_foot_contour']]for v in best['images']],'elapsed_seconds':time.monotonic()-started};print(json.dumps(clean(progress)),flush=True);(out/'ground_progress.json').write_text(json.dumps(clean(progress),indent=2)+'\n');(out/'ground_partial_records.json').write_text(json.dumps(clean(records),indent=2)+'\n')
        if len(records)%8==0 or len(records)==len(schedule):
            done=np.asarray([r['frame']for r in records]);np.savez_compressed(out/'ground_partial.npz',frame_indices=done,base_xyz_m=states[done,:3],base_quaternion_xyzw=Rotation.from_rotvec(states[done,3:6]).as_quat(),joint_names=np.asarray(kin.body_joint_names),joint_positions_rad=states[done,6:],link_names=np.asarray(kin.link_names),link_world_transforms=transforms[done],timestamps=times[done],accepted=accepted[done],ground_nonpenetration_valid=ground_valid[done],source_body_sha256=np.asarray(hashes['body']),source_hand_sha256=np.asarray(hashes['hand']),provisional=np.asarray(True))
    selected=np.asarray(schedule);suffix=''if args.all_frames else '-keyframes';file=out/('ground_trajectory.npz'if args.all_frames else 'ground_keyframes.npz');rp=ROOT/f'reports/user-ground-optimization{suffix}.json';q=states[:,6:];changed=np.abs(q[selected]-state0[selected,6:])>1e-7;support=C['joint_image_supported'][selected].copy();support[changed]=False;uncert=C['joint_approx_uncertainty_rad'][selected].copy();uncert[changed]=np.nan
    if not np.array_equal(q[:,waist],state0[:,6+waist]):raise RuntimeError('Ground stage changed frozen waist')
    speed=np.abs(np.diff(q,axis=0))/np.diff(times)[:,None];velocity_ok=bool((speed<=velocity[None]+1e-6).all());limits_ok=bool(((q>=limits[:,0]-1e-8)&(q<=limits[:,1]+1e-8)).all())
    if not velocity_ok or not limits_ok:raise RuntimeError('Original G1 hard angular bounds failed')
    np.savez_compressed(file,base_xyz_m=states[selected,:3],base_quaternion_xyzw=Rotation.from_rotvec(states[selected,3:6]).as_quat(),joint_names=np.asarray(kin.body_joint_names),joint_positions_rad=q[selected],link_names=np.asarray(kin.link_names),link_world_transforms=transforms[selected],timestamps=times[selected],frame_indices=selected,joint_limits_rad=limits,joint_velocity_limits_rad_s=velocity,joint_image_supported=support,joint_approx_uncertainty_rad=uncert,source_joint_approx_uncertainty_rad=C['joint_approx_uncertainty_rad'][selected],joint_ground_stage_changed=changed,candidate_valid=image_valid[selected]&ground_valid[selected],ground_image_valid=image_valid[selected],ground_nonpenetration_valid=ground_valid[selected],accepted=accepted[selected],sole_min_z_m=allsole[selected],raw_foot_min_z_m=allraw[selected],hand_max_penetration_m=allhand[selected],camera_P_video=P,camera_sha256=np.asarray(hashes['camera']),source_video_sha256=C['source_video_sha256'],source_body_sha256=np.asarray(hashes['body']),source_hand_sha256=np.asarray(hashes['hand']),source_box_sha256=np.asarray(hashes['box']),finger_actuator_positions=H['actuator_positions'][selected],finger_actuator_names=H['actuator_names'],finger_actuator_units=H['actuator_units'],finger_joint_positions=H['joint_positions'][selected],finger_joint_names=H['joint_names'],finger_joint_units=H['joint_units'],finger_joint_positions_included=np.asarray(True),parent_hand_image_fit_valid=H['image_fit_valid'][selected],parent_hand_side_valid=H['side_valid'][selected])
    report={'status':'ground_actual_image_feasibility_completed'if not args.all_frames else 'ground_actual_image_trajectory_completed','inputs':{str(p.relative_to(ROOT)):hashes[k]for k,p in paths.items()},'output':{'path':str(file),'sha256':sha(file)},'frames':len(schedule),'frame_indices':schedule,'all_tested_ground_image_feasible':bool(accepted[selected].all()),'accepted_frames':int(accepted[selected].sum()),'rejected_frames':selected[~accepted[selected]].tolist(),'all_original_joint_limits_satisfied':limits_ok,'all_original_joint_velocity_limits_satisfied':velocity_ok,'fixed_waist_unchanged':True,'fixed_finger_controls_unchanged':True,'floor_z_m':0.,'floor_moved':False,'calibration_is_conditional':True,'strict_metric_or_physical_reconstruction_valid':False,'source_pose_policy':'Final Hcontact.body_joint_positions_rad_refined + C floating base and exactly fixed Hcontact actuator12→joint50. Source parameters are initial priors, not invented image keypoints.','sole_definition':'Actual ankle_roll visual full unique vertices and all triangle centres. Sole-only subset is its authored q0 bottom0.75mm band. Tracking foot/ankle surface points are used only as 2D image observations, never sole heights.','heldout_policy':'Physical TAP point IDs every fifth held out globally. Spatial body16px and local-foot8px holdout tiles are both excluded from all contour training. RGB shoulder/elbow/wrist observations are training only, with manual approximation uncertainty8px. Hand ROI heldout uses the existing real RGB edge split. QA selected outcomes are diagnostic, not an untouched confirmatory test.','variables':[int(i)for i in variables],'source_base_changed_max_m':float(np.linalg.norm(states[selected,:3]-state0[selected,:3],axis=1).max()),'source_base_rotation_changed_max_rad':float(np.linalg.norm(states[selected,3:6]-state0[selected,3:6],axis=1).max()),'hard_angular_velocity_limits_rad_s':dict(zip(kin.body_joint_names,velocity.tolist())),'base_velocity_is_not_hardware_validated':True,'mesh_sources':mesh_records,'records':records,'thresholds':{'optimizer_ground_allowance_m':.002,'independent_raw_all_robot_allowance_m':.005,'sole_new_hover_max_m':.008,'tap_heldout_median_max_px':3,'tap_heldout_p90_max_px':6,'tap_heldout_median_degradation_max_px':1.5,'tap_heldout_p90_degradation_max_px':2.5,'local_foot_heldout_median_max_px':4,'local_foot_heldout_p90_max_px':8,'local_foot_median_degradation_max_px':1.5,'local_foot_p90_degradation_max_px':2.5,'hand_penetration_nonworsening_allowance_m':.0005,'hand_heldout_degradation_max_px':2},'elapsed_seconds':time.monotonic()-started,'gpu_used':False,'training':False};rp.write_text(json.dumps(clean(report),indent=2)+'\n');print(json.dumps(clean({k:report[k]for k in ('status','frames','accepted_frames','rejected_frames','all_original_joint_limits_satisfied','all_original_joint_velocity_limits_satisfied','source_base_changed_max_m','output','elapsed_seconds')}),indent=2),flush=True)


if __name__=='__main__':main()
