#!/usr/bin/env python3
"""CPU, image-constrained G1 arm branch search with fixed coupled ROHand meshes.

Only the fourteen existing shoulder/elbow/wrist angles may change. Parents,
mounts, box scale, base, legs, waist and hand actuators are immutable inputs.
Full original hand vertices and triangle centres audit each candidate; the
optimizer uses every link plus a refreshed deepest-point active set.
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
    with Path(path).open('rb')as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def cleaned(value):
    import math
    if isinstance(value,dict):return {str(k):cleaned(v)for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [cleaned(v)for v in value]
    if hasattr(value,'tolist'):return cleaned(value.tolist())
    if isinstance(value,float)and not math.isfinite(value):return None
    return value


def finalize_metadata():
    """Attach explicit unknown uncertainty and training-only landmark evidence.

    This is idempotent and leaves every fitted state and link transform intact.
    It also supports an already-running contact job compiled before this audit
    metadata was added, without interrupting or repeating any optimized frame.
    """
    import numpy as np
    from user_body_kinematics import G1Kinematics,project_points
    from scipy.spatial.transform import Rotation
    out=ROOT/'outputs/user_reconstruction_v2';path=out/'body_trajectory_contact.npz';rp=ROOT/'reports/user-arm-contact.json'
    report=json.loads(rp.read_text())
    def load(p):
        with np.load(p)as n:return {k:n[k]for k in n.files}
    data=load(path);D=load(out/'body_trajectory.npz');H=load(Path(report['parent_roles']['H']['path']));L=load(out/'body_landmark_observations.npz');C=load(out/'camera_refinement.npz')
    if sha(Path(report['parent_roles']['H']['path']))!=str(data['source_hand_sha256']):raise RuntimeError('Parent H changed before metadata finalization')
    kin=G1Kinematics(ROOT,scene_path=ROOT/'data/video_generation/candidate_scene/scene_rohand.json');indices=[np.asarray([kin.body_index[f'{side}_{name}_joint']for name in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw')])for side in ('left','right')]
    changed=np.zeros(data['joint_positions_rad'].shape,bool)
    for row in range(len(changed)):
        for side in range(2):
            if data['contact_accepted'][row,side]:changed[row,indices[side]]=True
    data['joint_contact_branch_changed']=changed
    if 'source_joint_approx_uncertainty_rad'not in data:data['source_joint_approx_uncertainty_rad']=data['joint_approx_uncertainty_rad'].copy()
    data['joint_image_supported'][changed]=False;data['joint_approx_uncertainty_rad'][changed]=np.nan
    for name,key in [('parent_hand_image_fit_valid','image_fit_valid'),('parent_hand_side_valid','side_valid'),('parent_hand_observed','observed'),('parent_hand_uncertain','uncertain')]:data[name]=H[key].copy()
    for name,key in [('finger_actuator_names','actuator_names'),('finger_actuator_units','actuator_units'),('finger_joint_names','joint_names')]:data[name]=H[key].copy()
    point_links=np.asarray([kin.link_index[n]for n in ('right_shoulder_roll_link','right_elbow_link','right_wrist_yaw_link','right_wrist_yaw_link','left_shoulder_roll_link','left_elbow_link','left_wrist_yaw_link','left_wrist_yaw_link')]);zeros=np.zeros((8,3));training={str(name):[]for k,name in enumerate(L['point_names'])if k not in (3,7)}
    states=np.column_stack((data['base_xyz_m'],Rotation.from_quat(data['base_quaternion_xyzw']).as_rotvec(),data['joint_positions_rad']))
    for record in report['records']:
        f=record['frame'];world=kin.points(states[f],point_links,zeros);per_frame={}
        for view in range(2):
            good=L['valid'][f,view].copy();good[[3,7]]=False;uv=project_points(C['P_video'][view],world)
            for k in np.flatnonzero(good):
                error=float(np.linalg.norm(uv[k]-L['positions_xy'][f,view,k]));name=str(L['point_names'][k]);training[name].append(error);per_frame[name]=error
        record['actual_RGB_landmark_training_errors_px']=per_frame;record['independent_heldout_landmark_count']=0
    report['landmark_training_summary']={name:{'count':len(e),'median_px':float(np.median(e))if e else None,'p90_px':float(np.percentile(e,90))if e else None}for name,e in training.items()}
    report['landmark_evidence_policy']={'source':'Actual unmodified RGB approximate manual annotations at frames0/60/120 plus accepted measured bidirectional LK; never a reference 3D projection as a pseudo-target.','manual_annotation_approximate_uncertainty_px':8.,'training_landmark_count':sum(len(e)for e in training.values()),'independent_heldout_landmark_count':0,'heldout_landmark_error_px':None,'small_training_error_is_not_independent_validation':True,'independent_image_domains':'Spatial heldout body contours and hand ROI edges; candidate selection makes these diagnostic holdout, not untouched confirmatory validation.'}
    report['contact_branch_uncertainty_policy']='Accepted arm branch angles retain no independent 3D observability claim: all seven changed-arm support flags are false and their previous local-Jacobian uncertainty is unknown NaN. Original proxy std is retained separately. Parent hand observed/uncertain/image flags are preserved without promotion.'
    report['unknown_uncertainty_joint_entries']=int(np.isnan(data['joint_approx_uncertainty_rad']).sum());report['metadata_finalized_without_pose_changes']=True
    report['failed_trial_example_policy']='For failed multistart searches, best_trial_images denotes the collision-first example selected by minimum penetration with search-order tie breaking; it is not a global minimum image-error claim. All finite searched candidates must pass every unchanged criterion before adoption.'
    report['per_view_heldout_summary']=[{'view':v,'before_body_contour_median_px':float(np.median([r['before_image'][v]['heldout_body_contour_median_px']for r in report['records']])),'after_body_contour_median_px':float(np.median([r['after_image'][v]['heldout_body_contour_median_px']for r in report['records']])),'worst_body_contour_p90_px':max(r['after_image'][v]['heldout_body_contour_p90_px']for r in report['records']),'min_body_iou':min(r['after_image'][v]['body_mask_iou']for r in report['records'])}for v in range(2)]
    report['script_sha256']=sha(Path(__file__));report['metadata_finalization_command']='OMP_NUM_THREADS=6 CUDA_VISIBLE_DEVICES= .venvs/vision/bin/python scripts/optimize_user_arm_contact.py --finalize-metadata'
    np.savez_compressed(path,**data);report['output']['sha256']=sha(path);rp.write_text(json.dumps(cleaned(report),indent=2)+'\n')
    print(json.dumps(cleaned({k:report[k]for k in ('output','accepted_sides','collision_valid_sides','body_image_valid_frames','max_penetration_before_m','max_penetration_after_m','unresolved_collision_frame_indices','landmark_training_summary','per_view_heldout_summary','unknown_uncertainty_joint_entries')}),indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--hand-file',type=Path,default=ROOT/'outputs/user_reconstruction_v2/hand_trajectory_velocity_refined.npz')
    ap.add_argument('--keyframes-only',action='store_true')
    ap.add_argument('--maxiter',type=int,default=70)
    ap.add_argument('--outer',type=int,default=3)
    ap.add_argument('--starts',type=int,default=3)
    ap.add_argument('--samples-per-link',type=int,default=256)
    ap.add_argument('--body-file',type=Path,help='Explicit contact body baseline for a separate diagnostic; immutable D still supplies fixed base/legs/waist')
    ap.add_argument('--left-near-contact-diagnostic',action='store_true',help='Separate keyframes-only left-arm surface-approach feasibility test, never overwrite C')
    ap.add_argument('--repair-frames',default='',help='Only refit listed failure frames from an immutable archived --body-file')
    ap.add_argument('--source-contact-report',type=Path,help='Matching immutable archived report for --repair-frames')
    ap.add_argument('--hand-image-weight',type=float,default=0.,help='Actual hand ROI training-edge correspondence weight; spatial heldout remains excluded')
    ap.add_argument('--finalize-metadata',action='store_true',help='Audit a finished full sequence without changing any fitted pose')
    args=ap.parse_args();os.environ['CUDA_VISIBLE_DEVICES']='';os.environ.setdefault('OMP_NUM_THREADS','6')
    if args.finalize_metadata:finalize_metadata();return
    if args.left_near_contact_diagnostic:args.keyframes_only=True
    for name in ('hand_file','body_file','source_contact_report'):
        value=getattr(args,name)
        if value is not None and not value.is_absolute():setattr(args,name,ROOT/value)
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
    cv2.setNumThreads(4)
    started=time.monotonic();out=ROOT/'outputs/user_reconstruction_v2';source=ROOT/'outputs/user_reconstruction'
    body_path=out/'body_trajectory.npz';hand_path=args.hand_file;box_path=out/'box_trajectory.npz';camera_path=out/'camera_refinement.npz'
    scene_path=ROOT/'data/video_generation/candidate_scene/scene_rohand.json'
    bindings={key:{'path':str(path),'sha256':sha(path)}for key,path in [('D',body_path),('H',hand_path),('box',box_path),('camera',camera_path),('input',source/'input.npz'),('segmentation',source/'segmentation.npz'),('landmarks',out/'body_landmark_observations.npz'),('scene',scene_path)]}
    def load(path):
        with np.load(path)as d:return {k:d[k]for k in d.files}
    D=load(body_path);H=load(hand_path);B=load(box_path);C=load(camera_path);L=load(out/'body_landmark_observations.npz')
    baseline_body=None
    if args.body_file:
        baseline_body=load(args.body_file);bindings['baseline_contact_body']={'path':str(args.body_file),'sha256':sha(args.body_file)}
        if not np.array_equal(baseline_body['base_xyz_m'],D['base_xyz_m'])or not np.array_equal(baseline_body['base_quaternion_xyzw'],D['base_quaternion_xyzw']):raise RuntimeError('Contact diagnostic baseline moved original base')
        if str(baseline_body['source_hand_sha256'])!=bindings['H']['sha256']:raise RuntimeError('Contact diagnostic body and hand parents differ')
    inherited_report=None
    if args.repair_frames:
        if baseline_body is None or args.source_contact_report is None or args.keyframes_only:raise RuntimeError('Repair requires immutable whole-contact body and matching report')
        inherited_report=json.loads(args.source_contact_report.read_text())
        if inherited_report['output']['sha256']!=bindings['baseline_contact_body']['sha256']:raise RuntimeError('Repair body/report mismatch')
        bindings['archived_contact_report']={'path':str(args.source_contact_report),'sha256':sha(args.source_contact_report)}
    if str(H['source_body_sha256'])!=bindings['D']['sha256']:raise RuntimeError('H is not bound to D')
    if str(H['camera_sha256'])!=bindings['camera']['sha256']:raise RuntimeError('H camera mismatch')
    if str(H['box_sha256'])!=bindings['box']['sha256']:raise RuntimeError('H box mismatch')
    V=load(source/'input.npz');S=load(source/'segmentation.npz');rgb=V['rgb'];robot=S['robot'];boxes=S['box'];valid=~V['watermark_invalid']
    if not np.array_equal(S['observation_valid'],valid):raise RuntimeError('Stale SAM validity')
    timestamps=D['timestamps'];P=C['P_video'];frames=len(timestamps);schedule=[0,60,120]if args.keyframes_only else list(range(frames))
    if args.repair_frames:schedule=sorted(set(int(f)for f in args.repair_frames.split(',')))
    kin=G1Kinematics(ROOT,scene_path=scene_path);coupling=ROHandCoupling(ROOT)
    if D['joint_names'].tolist()!=kin.body_joint_names:raise RuntimeError('Body joint ordering mismatch')
    qsource=(baseline_body['joint_positions_rad']if baseline_body is not None else H['body_joint_positions_rad_refined']).copy();states=np.column_stack((D['base_xyz_m'],Rotation.from_quat(D['base_quaternion_xyzw']).as_rotvec(),qsource));original_states=states.copy()
    arm_indices=[np.asarray([kin.body_index[f'{side}_{name}_joint']for name in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw')])for side in ('left','right')]
    motion_indices=np.concatenate(arm_indices);fixed_indices=np.asarray([j for j in range(29)if j not in motion_indices])
    if not np.array_equal(qsource[:,fixed_indices],D['joint_positions_rad'][:,fixed_indices]):raise RuntimeError('H changed non-arm body')
    xml=ET.parse(kin.urdf_path).getroot();speed={j.attrib['name']:float(j.find('limit').attrib['velocity'])for j in xml.findall('joint')if j.attrib['name']in kin.body_index}
    velocity=np.asarray([speed[n]for n in kin.body_joint_names]);limits=kin.body_limits
    wrist_links=[kin.link_index[s+'_wrist_yaw_link']for s in ('left','right')]
    point_links=np.asarray([kin.link_index[n]for n in ('right_shoulder_roll_link','right_elbow_link','right_wrist_yaw_link','right_wrist_yaw_link','left_shoulder_roll_link','left_elbow_link','left_wrist_yaw_link','left_wrist_yaw_link')])
    zero_points=np.zeros((8,3));observed=L['positions_xy'];observed_valid=L['valid']
    # All public hand visual meshes including palm, sliders and linkages.
    hand_visuals=[[],[]];mesh_sources=[]
    for visual in kin.visuals:
        if '_hand_ro_'not in visual['link_name']:continue
        side=0 if visual['link_name'].startswith('left_')else 1
        v=visual['vertices_local'];f=visual['faces'];unique=np.unique(np.round(v,10),axis=0);centres=v[f].mean(axis=1)
        full=np.concatenate((unique,centres));n=min(args.samples_per_link,len(full));sample=full[np.linspace(0,len(full)-1,n).astype(int)]
        hand_visuals[side].append((visual['link_name'],visual['link_index'],full,sample))
        mesh_sources.append({'link':visual['link_name'],'sha256':visual['mesh_sha256'],'all_unique_vertices':len(unique),'all_triangle_centres':len(centres),'initial_optimization_samples':n})
    static=[]
    table=kin.scene['table']
    for size,center in [(table['top_size_xyz_m'],table['top_center_xyz_m'])]+[([.035,.035,.735],[x,y,.3675])for x in (.19,.77)for y in (-.355,.355)]:
        m=trimesh.creation.box(extents=size);m.apply_translation(center);static.append(m)
    table_mesh=trimesh.util.concatenate(static)
    table_v=o3d.core.Tensor(np.asarray(table_mesh.vertices,dtype=np.float32));table_f=o3d.core.Tensor(np.asarray(table_mesh.faces,dtype=np.uint32))
    h,w=240,213;sx,sy=640/w,720/h;xx,yy=np.meshgrid((np.arange(w)+.5)*sx,(np.arange(h)+.5)*sy);rays=[]
    for view in range(2):
        M=P[view,:,:3];center=-np.linalg.solve(M,P[view,:,3]);direction=np.stack((xx,yy,np.ones_like(xx)),axis=-1)@np.linalg.inv(M).T;direction/=np.linalg.norm(direction,axis=-1,keepdims=True)
        rays.append(o3d.core.Tensor(np.concatenate((np.broadcast_to(center,direction.shape),direction),axis=-1).astype(np.float32)))
    body_faces=np.asarray(['_hand_ro_'not in kin.link_names[i]for i in kin.proxy_face_links]);rf=kin.proxy_faces[body_faces];rl=kin.proxy_face_links[body_faces]
    def hand_domain(frame,view,points):
        good=np.ones(len(points),bool)
        for k,radius in ((2+4*view,55.),(3+4*view,35.)):
            if observed_valid[frame,view,k]:good &= np.linalg.norm(points-observed[frame,view,k],axis=1)>radius
        return good
    def image_targets(frame):
        result=[]
        for view in range(2):
            mask=robot[frame,view]&valid[frame,view];contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
            pixels=np.concatenate([c[:,0]for c in contours if len(c)>4]).astype(float)+.5;pixels=pixels[hand_domain(frame,view,pixels)]
            if len(pixels)>650:pixels=pixels[np.linspace(0,len(pixels)-1,650).astype(int)]
            group=(pixels[:,0]//16).astype(int)+64*(pixels[:,1]//16).astype(int);held=group%5==0
            domain=hand_domain(frame,view,np.column_stack((xx.ravel(),yy.ravel()))).reshape(h,w)
            result.append({'train':pixels[~held],'heldout':pixels[held],'mask':(cv2.resize(mask.astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)>0)&domain,'domain':domain})
        return result
    def render(state,frame):
        scene=o3d.t.geometry.RaycastingScene(nthreads=6);scene.add_triangles(o3d.core.Tensor(kin.proxy_world(state).astype(np.float32)),o3d.core.Tensor(rf));scene.add_triangles(table_v,table_f)
        answer=[]
        for view in range(2):
            cast=scene.cast_rays(rays[view],nthreads=6);hit=cast['geometry_ids'].numpy()==0;hit &= ~(cv2.resize(boxes[frame,view].astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)>0)
            boundary=hit&~cv2.erode(hit.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool);py,px=np.nonzero(boundary);keep=hand_domain(frame,view,np.column_stack(((px+.5)*sx,(py+.5)*sy)));py,px=py[keep],px[keep]
            if len(px)>500:take=np.linspace(0,len(px)-1,500).astype(int);py,px=py[take],px[take]
            tri=cast['primitive_ids'].numpy()[py,px].astype(int);uv=cast['primitive_uvs'].numpy()[py,px]
            local=kin.proxy_vertices[rf[tri,0]]*(1-uv.sum(1))[:,None]+kin.proxy_vertices[rf[tri,1]]*uv[:,0,None]+kin.proxy_vertices[rf[tri,2]]*uv[:,1,None]
            answer.append({'mask':hit,'xy':np.column_stack(((px+.5)*sx,(py+.5)*sy)),'local':local,'links':rl[tri]})
        return answer
    def sdf(points,frame,gradient=False):
        tf=B['box_world_transform'][frame];local=(points-tf[:3,3])@tf[:3,:3];q=np.abs(local)-B['box_size_xyz_m']/2;outside=np.maximum(q,0);length=np.linalg.norm(outside,axis=1);value=length+np.minimum(q.max(1),0)
        if not gradient:return value
        g=np.zeros_like(local);ext=length>1e-12;g[ext]=outside[ext]/length[ext,None]*np.sign(local[ext]);axis=q.argmax(1);inside=~ext;g[np.flatnonzero(inside),axis[inside]]=np.sign(local[np.flatnonzero(inside),axis[inside]])
        return value,g@tf[:3,:3].T
    def frame_geometry(frame):
        finger=coupling.expand_one(H['actuator_positions'][frame]);T=kin.fk(original_states[frame],finger);result=[]
        for side in range(2):
            wrist=T[wrist_links[side]];full=[];samples=[];outlines=[]
            for name,link,v,sample in hand_visuals[side]:
                local_R=wrist[:3,:3].T@T[link,:3,:3];local_t=wrist[:3,:3].T@(T[link,:3,3]-wrist[:3,3]);full.append(v@local_R.T+local_t);samples.append(sample@local_R.T+local_t)
                if 'slider'not in name and 'connecting'not in name:outlines.append(sample@local_R.T+local_t)
            result.append({'full':np.concatenate(full),'sample':np.concatenate(samples),'outlines':outlines})
        return result,finger
    def transform_hand(state,side,points,jac=False):
        T,axes,centres=kin.fk(state,return_axes=True);wrist=T[wrist_links[side]];world=points@wrist[:3,:3].T+wrist[:3,3]
        if not jac:return world
        js=arm_indices[side];J=np.cross(axes[js][None],world[:,None]-centres[js][None]).transpose(0,2,1)
        return world,J
    def collision(state,frame,geometry):
        result=[]
        for side in range(2):
            s=sdf(transform_hand(state,side,geometry[side]['full']),frame);result.append({'full_surface_samples':len(s),'max_penetration_m':float(max(0,-s.min())),'fraction_deeper_than_5mm':float((s<-.005).mean()),'min_sdf_m':float(s.min())})
        return result
    def image_metrics(state,frame,targets,hand_observations,geometry):
        predicted=render(state,frame);landmarks=kin.points(state,point_links,zero_points);results=[]
        for view in range(2):
            pred=predicted[view];truth=targets[view];mask=pred['mask']&truth['domain'];intersection=(mask&truth['mask']).sum();union=(mask|truth['mask']).sum();distance=cKDTree(pred['xy']).query(truth['heldout'])[0]if len(pred['xy'])else np.asarray([np.inf])
            good=observed_valid[frame,view].copy();good[[3,7]]=False;err=np.linalg.norm(project_points(P[view],landmarks)[good]-observed[frame,view,good],axis=1);wi=2+4*view
            wrist_error=float(np.linalg.norm(project_points(P[view],landmarks[[wi]])[0]-observed[frame,view,wi]))if observed_valid[frame,view,wi]else None
            side=1-view;obs=hand_observations[side];hand_error=None
            if obs is not None and len(obs['heldout_points']):
                boundaries=[]
                for points in geometry[side]['outlines']:boundaries.append(polygon_samples(project_points(P[view],transform_hand(state,side,points)),12))
                hand_error=float(cKDTree(np.concatenate(boundaries)).query(obs['heldout_points'])[0].mean())
            results.append({'view':view,'body_mask_iou':float(intersection/max(union,1)),'heldout_body_contour_median_px':float(np.median(distance)),'heldout_body_contour_p90_px':float(np.percentile(distance,90)),'actual_landmark_median_px':float(np.median(err))if len(err)else None,'actual_landmark_p90_px':float(np.percentile(err,90))if len(err)else None,'wrist_keypoint_error_px':wrist_error,'hand_heldout_edge_mean_px':hand_error})
        return results,predicted
    def image_ok(after,before):
        for a,b in zip(after,before):
            if a['body_mask_iou']<.55 or a['body_mask_iou']<b['body_mask_iou']-.04:return False
            if a['heldout_body_contour_median_px']>6 or a['heldout_body_contour_median_px']>b['heldout_body_contour_median_px']+2:return False
            if a['heldout_body_contour_p90_px']>15 or a['heldout_body_contour_p90_px']>b['heldout_body_contour_p90_px']+3:return False
            for k,absolute,extra in [('actual_landmark_median_px',8,2),('actual_landmark_p90_px',16,3),('wrist_keypoint_error_px',8,3)]:
                if a[k]is not None and (a[k]>absolute or (b[k]is not None and a[k]>b[k]+extra)):return False
            if a['hand_heldout_edge_mean_px']is not None and b['hand_heldout_edge_mean_px']is not None and a['hand_heldout_edge_mean_px']>b['hand_heldout_edge_mean_px']+2:return False
        return True
    records=[];accepted=np.zeros((frames,2),bool);image_valid=np.zeros((frames,2),bool);collision_valid=np.zeros((frames,2),bool);all_transforms=np.empty((frames,len(kin.link_names),4,4));before_pen=np.full((frames,2),np.nan);after_pen=before_pen.copy()
    if inherited_report is not None:
        records=[r for r in inherited_report['records']if r['frame']not in schedule];accepted=baseline_body['contact_accepted'].copy();image_valid=baseline_body['image_fit_valid'].copy();collision_valid=baseline_body['contact_collision_valid'].copy();all_transforms=baseline_body['link_world_transforms'].copy();before_pen=baseline_body['hand_box_max_penetration_before_m'].copy();after_pen=baseline_body['hand_box_max_penetration_after_m'].copy()
    for frame in schedule:
        frame_start=time.monotonic();geometry,finger=frame_geometry(frame);targets=image_targets(frame);hand_obs=[]
        for side in range(2):
            view=1-side;pi=3+4*view;palm=observed[frame,view,pi]if observed_valid[frame,view,pi]else np.asarray([np.nan,np.nan])
            hand_obs.append(observation(rgb[frame,view],robot[frame,view],boxes[frame,view],valid[frame,view],palm,1 if view==0 else -1))
        original=original_states[frame];baseline_images,_=image_metrics(original,frame,targets,hand_obs,geometry);baseline_collision=collision(original,frame,geometry);side_records=[]
        for side in range(2):
            if args.left_near_contact_diagnostic and side!=0:
                side_records.append({'side':side,'accepted':False,'reason':'left_only_diagnostic_preserves_right_contact_body'});continue
            if not args.keyframes_only and baseline_collision[side]['max_penetration_m']<=.004:
                side_records.append({'side':side,'accepted':False,'reason':'source_already_nonpenetrating_no_refit','baseline_min_sdf_m':baseline_collision[side]['min_sdf_m']});continue
            js=arm_indices[side];variables=6+js;lower=limits[js,0].copy();upper=limits[js,1].copy()
            if frame>0:
                previous=states[frame-1,6+js]if not args.keyframes_only else original_states[frame-1,6+js];delta=velocity[js]*(timestamps[frame]-timestamps[frame-1]);lower=np.maximum(lower,previous-delta+1e-9);upper=np.minimum(upper,previous+delta-1e-9)
            if frame+1<frames:
                following=original_states[frame+1,6+js];delta=velocity[js]*(timestamps[frame+1]-timestamps[frame]);lower=np.maximum(lower,following-delta+1e-9);upper=np.minimum(upper,following+delta-1e-9)
            if (lower>=upper).any():
                side_records.append({'side':side,'accepted':False,'reason':'input_neighbour_velocity_bounds_infeasible'});continue
            reference=states[frame].copy();current=reference.copy();sample=geometry[side]['sample'].copy();trials=[]
            starts=[np.clip(reference[variables],lower+1e-10,upper-1e-10)]
            for sign in (-1,1):
                seed=starts[0].copy();seed[[0,2,3]]+=sign*np.asarray([.2,.35,.2]);starts.append(np.clip(seed,lower+1e-10,upper-1e-10))
            for seed in starts[:args.starts]:
                trial=current.copy();trial[variables]=seed;iterations=[]
                for outer in range(args.outer):
                    predicted=render(trial,frame);data=[]
                    for view in range(2):
                        pred=predicted[view];obs=targets[view]['train'];d,idx=cKDTree(obs).query(pred['xy']);data.append((view,pred['links'],pred['local'],obs[idx],.65))
                        d,idx=cKDTree(pred['xy']).query(obs);data.append((view,pred['links'][idx],pred['local'][idx],obs,.65))
                    hand_observation=hand_obs[side]
                    if args.hand_image_weight>0 and hand_observation is not None:
                        view=1-side;local_boundary=[];wrist=kin.fk(trial)[wrist_links[side]]
                        for local_outline in geometry[side]['outlines']:
                            world_outline=local_outline@wrist[:3,:3].T+wrist[:3,3];hom=np.column_stack((world_outline,np.ones(len(world_outline))))@P[view].T;uv=hom[:,:2]/hom[:,2,None];hull=cv2.convexHull(uv.astype(np.float32),returnPoints=False).ravel()
                            if len(hull)<3:continue
                            xy=uv[hull];edge=np.roll(xy,-1,axis=0)-xy;length=np.linalg.norm(edge,axis=1);cum=np.r_[0,length.cumsum()];values=np.arange(16)/16*cum[-1];index=np.clip(np.searchsorted(cum,values,side='right')-1,0,len(length)-1);alpha=(values-cum[index])/np.maximum(length[index],1e-9);a=hull[index];b=hull[(index+1)%len(hull)];za=hom[a,2];zb=hom[b,2];fraction=alpha*za/np.maximum((1-alpha)*zb+alpha*za,1e-9);local_boundary.append(local_outline[a]*(1-fraction[:,None])+local_outline[b]*fraction[:,None])
                        local_boundary=np.concatenate(local_boundary);world_boundary=local_boundary@wrist[:3,:3].T+wrist[:3,3];uv=project_points(P[view],world_boundary);train=hand_observation['points'];_,idx=cKDTree(train).query(uv);wl=np.full(len(local_boundary),wrist_links[side],int);data.append((view,wl,local_boundary,train[idx],args.hand_image_weight))
                        _,idx=cKDTree(uv).query(train);data.append((view,np.full(len(train),wrist_links[side],int),local_boundary[idx],train,args.hand_image_weight*.5))
                    full_world=transform_hand(trial,side,geometry[side]['full']);full_sdf=sdf(full_world,frame);deep=np.argpartition(full_sdf,min(511,len(full_sdf)-1))[:min(512,len(full_sdf))];points=np.concatenate((sample,geometry[side]['full'][deep]))
                    def objective(angles):
                        state=trial.copy();state[variables]=angles;loss=0.;gradient=np.zeros(7)
                        for view,links,local,target,weight in data:
                            world,jac=kin.points(state,links,local,True);xy,pj=project_points(P[view],world,True);J=np.einsum('nij,njk->nik',pj,jac[:,:,variables]);res=(xy-target)*weight;root=np.sqrt(1+(res/4.)**2);loss+=float((32*(root-1)).sum());gradient+=np.einsum('ni,nik->k',2*res/root*weight,J)
                        world,jac=kin.points(state,point_links,zero_points,True)
                        for view in range(2):
                            good=observed_valid[frame,view].copy();good[[3,7]]=False;xy,pj=project_points(P[view],world,True);J=np.einsum('nij,njk->nik',pj,jac[:,:,variables]);weights=np.asarray([3.,5.,8.,0.,3.,5.,8.,0.]);res=(xy[good]-observed[frame,view,good])*weights[good,None];root=np.sqrt(1+(res/4.)**2);loss+=float((32*(root-1)).sum());gradient+=np.einsum('ni,nik->k',2*res/root*weights[good,None],J[good])
                        world,J=transform_hand(state,side,points,True);distance,g=sdf(world,frame,True);pen=np.maximum(.001-distance,0)/.003;loss+=float(2000*np.mean(pen**2)+4000*np.max(pen)**2);derivative=-4000*pen/(.003*len(pen));worst=int(pen.argmax());derivative[worst]-=8000*pen[worst]/.003;gradient+=np.einsum('n,ni,nik->k',derivative,g,J)
                        if args.left_near_contact_diagnostic:
                            nearest=int(distance.argmin());gap=max(float(distance[nearest])-.002,0)/.005;loss+=4000*gap*gap;gradient+=8000*gap/.005*(g[nearest]@J[nearest])
                        prior=(angles-reference[variables])*16;loss+=float(prior@prior);gradient+=2*prior*16
                        return loss,gradient
                    fit=minimize(objective,trial[variables],jac=True,method='L-BFGS-B',bounds=list(zip(lower,upper)),options={'maxiter':args.maxiter,'ftol':1e-9,'gtol':1e-4,'maxls':30});trial[variables]=fit.x;iterations.append({'outer':outer,'cost':float(fit.fun),'iterations':int(fit.nit),'success':bool(fit.success)})
                images,_=image_metrics(trial,frame,targets,hand_obs,geometry);collisions=collision(trial,frame,geometry);passes_collision=collisions[side]['max_penetration_m']<=.004 and (not args.left_near_contact_diagnostic or collisions[side]['min_sdf_m']<=.004);passes_image=image_ok(images,baseline_images);trials.append((trial.copy(),images,collisions,passes_collision and passes_image,iterations))
                if passes_collision and passes_image:break
            feasible=[t for t in trials if t[3]]
            chosen=min(feasible,key=lambda t:sum(m['heldout_body_contour_median_px']for m in t[1])+float(np.linalg.norm(t[0][variables]-reference[variables]))*.1)if feasible else min(trials,key=lambda t:t[2][side]['max_penetration_m']+(max(t[2][side]['min_sdf_m']-.004,0)if args.left_near_contact_diagnostic else 0))
            if feasible:states[frame,variables]=chosen[0][variables];accepted[frame,side]=True
            side_records.append({'side':side,'accepted':bool(accepted[frame,side]),'reason':'collision_and_image_passed'if feasible else 'image_geometry_conflict_or_collision_infeasible','best_trial_images':chosen[1],'best_trial_collision':chosen[2][side],'trials':[{'collision':t[2][side],'image_acceptance':image_ok(t[1],baseline_images),'accepted':t[3],'iterations':t[4]}for t in trials]})
        final_images,_=image_metrics(states[frame],frame,targets,hand_obs,geometry);final_collision=collision(states[frame],frame,geometry)
        for view,m in enumerate(final_images):image_valid[frame,view]=m['body_mask_iou']>=.55 and m['heldout_body_contour_median_px']<=6 and m['heldout_body_contour_p90_px']<=15 and (m['actual_landmark_median_px']is None or (m['actual_landmark_median_px']<=8 and m['actual_landmark_p90_px']<=16))
        for side in range(2):before_pen[frame,side]=baseline_collision[side]['max_penetration_m'];after_pen[frame,side]=final_collision[side]['max_penetration_m'];collision_valid[frame,side]=final_collision[side]['max_penetration_m']<=.004
        all_transforms[frame]=kin.fk(states[frame],finger);record={'frame':frame,'box_strict_valid':bool(B['valid'][frame]),'collision_geometry_is_conditional':not bool(B['valid'][frame]),'before_image':baseline_images,'after_image':final_images,'before_collision':baseline_collision,'after_collision':final_collision,'side_optimization':side_records,'elapsed_seconds':time.monotonic()-frame_start};records.append(record)
        progress={'status':'running_actual_arm_contact_fit','frame':frame,'completed':len(records)-(frames-len(schedule)if inherited_report is not None else 0),'scheduled':len(schedule),'accepted_sides':accepted[frame].tolist(),'before_penetration_m':before_pen[frame].tolist(),'after_penetration_m':after_pen[frame].tolist(),'elapsed_seconds':time.monotonic()-started};print(json.dumps(cleaned(progress)),flush=True);(out/'arm_contact_progress.json').write_text(json.dumps(cleaned(progress),indent=2)+'\n')
    records.sort(key=lambda r:r['frame']);selected=np.asarray(schedule if args.keyframes_only else list(range(frames)));suffix='_keyframes'if args.keyframes_only else '';mode='left_near_contact'if args.left_near_contact_diagnostic else 'contact';output=out/f'body_trajectory_{mode}{suffix}.npz';report_path=ROOT/f'reports/user-arm-{mode.replace("_","-")}{suffix}.json'
    q=states[:,6:];dt=np.diff(timestamps);actual_speed=np.abs(np.diff(q,axis=0))/dt[:,None];speed_valid=bool((actual_speed<=velocity[None]+1e-6).all())
    if not np.array_equal(q[:,fixed_indices],D['joint_positions_rad'][:,fixed_indices]):raise RuntimeError('Contact optimization changed fixed body joints')
    if not np.array_equal(states[:,:3],D['base_xyz_m']):raise RuntimeError('Contact changed base')
    if not args.keyframes_only and not speed_valid:raise RuntimeError('Contact candidate violated original URDF velocity limits')
    payload={k:v for k,v in D.items()};payload.update(base_xyz_m=D['base_xyz_m'][selected],base_quaternion_xyzw=D['base_quaternion_xyzw'][selected],joint_positions_rad=q[selected],link_world_transforms=all_transforms[selected],timestamps=timestamps[selected],candidate_valid=image_valid[selected].all(1)&C['frame_camera_valid'][selected].all(1),contact_accepted=accepted[selected],contact_collision_valid=collision_valid[selected],image_fit_valid=image_valid[selected],collision_geometry_conditional=~B['valid'][selected],source_body_sha256=np.asarray(bindings['D']['sha256']),source_hand_sha256=np.asarray(bindings['H']['sha256']),source_box_sha256=np.asarray(bindings['box']['sha256']),source_camera_sha256=np.asarray(bindings['camera']['sha256']),finger_actuator_positions=H['actuator_positions'][selected],finger_joint_positions=H['joint_positions'][selected],finger_joint_units=H['joint_units'],finger_joint_positions_included=np.asarray(True),frame_indices=selected,hand_box_max_penetration_before_m=before_pen[selected],hand_box_max_penetration_after_m=after_pen[selected])
    for k in ('joint_image_supported','joint_approx_uncertainty_rad'):payload[k]=(baseline_body[k]if baseline_body is not None else D[k])[selected]
    if baseline_body is not None:payload['source_contact_body_sha256']=np.asarray(bindings['baseline_contact_body']['sha256'])
    changed=np.zeros((len(selected),29),bool)
    for row,frame in enumerate(selected):
        for side in range(2):
            if accepted[frame,side]:changed[row,arm_indices[side]]=True
    payload['joint_contact_branch_changed']=changed
    payload['source_joint_approx_uncertainty_rad']=payload['joint_approx_uncertainty_rad'].copy()
    payload['joint_image_supported'][changed]=False
    payload['joint_approx_uncertainty_rad'][changed]=np.nan
    payload['parent_hand_image_fit_valid']=H['image_fit_valid'][selected]
    payload['parent_hand_side_valid']=H['side_valid'][selected]
    payload['parent_hand_observed']=H['observed'][selected]
    payload['parent_hand_uncertain']=H['uncertain'][selected]
    np.savez_compressed(output,**payload)
    report={'status':'keyframe_contact_feasibility_evaluated'if args.keyframes_only else 'actual_arm_contact_fit_completed','frames':len(schedule),'frame_indices':schedule,'inputs':{str(Path(v['path']).relative_to(ROOT)):v['sha256']for v in bindings.values()},'parent_roles':bindings,'output':{'path':str(output),'sha256':sha(output)},'only_variables':[kin.body_joint_names[j]for j in motion_indices],'fixed_base_legs_waist_unchanged':True,'fixed_actuators_unchanged':True,'all_original_joint_limits_satisfied':bool(((q>=limits[:,0]-1e-8)&(q<=limits[:,1]+1e-8)).all()),'all_original_arm_velocity_limits_satisfied':bool((actual_speed[:,motion_indices]<=velocity[None,motion_indices]+1e-6).all()),'full_body_source_velocity_limits_satisfied':speed_valid,'accepted_sides':accepted[selected].sum(0).tolist(),'collision_valid_sides':collision_valid[selected].sum(0).tolist(),'body_image_valid_frames':int(image_valid[selected].all(1).sum()),'max_penetration_before_m':float(np.nanmax(before_pen)),'max_penetration_after_m':float(np.nanmax(after_pen)),'mesh_sources':mesh_sources,'collision_policy':'Every hand visual is included, including palm/base, sliders, links and phalanges. Every unique original visual vertex and every original triangle centre is audited; per-link samples plus refreshed full-mesh deepest points guide optimization. This sampled-surface check is not a continuous triangle-intersection proof.','image_policy':'Actual source image landmarks and spatial heldout body contours are recomputed. Hand ROI heldout edges diagnose the fixed controls. Selection is conditional on image quality; QA-selected heldout is not an untouched confirmatory test.','acceptance_thresholds':{'max_full_mesh_surface_penetration_m':.004,'body_iou_min':.55,'body_iou_decrease_max':.04,'heldout_body_median_max_px':6,'heldout_body_median_degradation_max_px':2,'heldout_body_p90_max_px':15,'heldout_body_p90_degradation_max_px':3,'wrist_error_max_px':8,'wrist_error_degradation_max_px':3,'hand_heldout_mean_degradation_max_px':2},'strict_metric_contact_recovered':False,'full_kinogen_pipeline':False,'gpu_used':False,'training':False,'records':records,'elapsed_seconds':time.monotonic()-started}
    report['source_baseline_body_policy']='Actual baseline is H.body_joint_positions_rad_refined with its own immutable 12 actuator controls, never D angles mixed with H controls. D supplies unchanged base/legs/waist and observation uncertainty.'if baseline_body is None else 'Separate near-contact diagnostic starts from the frozen contact C body and its bound immutable H controls; original C is never overwritten.'
    if inherited_report is not None:
        report['frames']=frames;report['frame_indices']=list(range(frames));report['repair_processed_frame_indices']=schedule;report['hand_roi_training_edge_weight']=args.hand_image_weight;report['preserved_prior_frame_indices']=[r['frame']for r in inherited_report['records']if r['frame']not in schedule];report['source_baseline_body_policy']='A bounded failure-only second search starts from an immutable archived C with its bound H controls; untouched frames preserve their actual states and previously measured image/collision results.';report['prior_contact_elapsed_seconds']=inherited_report['elapsed_seconds'];report['this_pass_elapsed_seconds']=report['elapsed_seconds'];report['elapsed_seconds']+=inherited_report['elapsed_seconds']
    if args.left_near_contact_diagnostic:
        report['left_near_contact_diagnostic_only']=True
        report['acceptance_thresholds']['max_left_hand_surface_gap_m']=.004
        report['near_contact_policy']='A closest surface sample within4mm plus nonpenetration and unchanged image/velocity acceptance is only a surface-approach feasibility candidate, not evidence of force closure or a physical grasp.'
    report['original_joint_velocity_limits_rad_s']=dict(zip(kin.body_joint_names,velocity.tolist()))
    report['maximum_actual_joint_velocity_rad_s']=dict(zip(kin.body_joint_names,actual_speed.max(0).tolist()))
    report['already_clear_sides_preserved']=sum(s['reason']=='source_already_nonpenetrating_no_refit'for r in records for s in r['side_optimization'])
    report['unresolved_collision_frame_indices']=[r['frame']for r in records if any(c['max_penetration_m']>.004 for c in r['after_collision'])]
    training={str(name):[]for k,name in enumerate(L['point_names'])if k not in (3,7)}
    for record in records:
        f=record['frame'];world=kin.points(states[f],point_links,zero_points);per_frame={}
        for view in range(2):
            good=observed_valid[f,view].copy();good[[3,7]]=False;uv=project_points(P[view],world)
            for k in np.flatnonzero(good):
                error=float(np.linalg.norm(uv[k]-observed[f,view,k]));name=str(L['point_names'][k]);training[name].append(error);per_frame[name]=error
        record['actual_RGB_landmark_training_errors_px']=per_frame
        record['independent_heldout_landmark_count']=0
    report['landmark_training_summary']={name:{'count':len(e),'median_px':float(np.median(e))if e else None,'p90_px':float(np.percentile(e,90))if e else None}for name,e in training.items()}
    report['landmark_evidence_policy']={'source':'Actual unmodified RGB approximate manual annotations at frames0/60/120 plus accepted measured bidirectional LK; never a reference 3D projection as a pseudo-target.','manual_annotation_approximate_uncertainty_px':8.,'training_landmark_count':sum(len(e)for e in training.values()),'independent_heldout_landmark_count':0,'heldout_landmark_error_px':None,'small_training_error_is_not_independent_validation':True,'independent_image_domains':'Spatial heldout body contours and hand ROI edges; candidate selection makes these diagnostic holdout, not untouched confirmatory validation.'}
    report['contact_branch_uncertainty_policy']='Accepted arm branch angles retain no claim of independent 3D observability: all seven changed-arm support flags are false and their old local-Jacobian uncertainty is unknown NaN. Original proxy std is retained separately. Parent hand observed/uncertain/image flags are preserved without promotion.'
    report_path.write_text(json.dumps(cleaned(report),indent=2)+'\n');print(json.dumps(cleaned({k:v for k,v in report.items()if k not in ('records','mesh_sources')}),indent=2),flush=True)


if __name__=='__main__':main()
