"""Build a provenance-labelled 923-frame, six-object Vita52 reference.

Only reference construction writes kinematic states. Physical rollout is a
separate program. Missing observations are explicitly filled and labelled.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ranges(mask):
    import numpy as np
    ids = np.flatnonzero(mask)
    if not len(ids): return []
    groups = np.split(ids, np.flatnonzero(np.diff(ids) != 1)+1)
    return [[int(v[0]), int(v[-1])+1] for v in groups]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    root=Path(__file__).resolve().parents[1]
    p.add_argument('--data',type=Path,default=root/'data/fourth_sim_handoff')
    p.add_argument('--output',type=Path,default=root/'outputs/fourth_full_sequence/reference.npz')
    p.add_argument('--iterations',type=int,default=24)
    p.add_argument('--safe-root-x',type=float,default=.16)
    p.add_argument('--collision-model',type=Path,default=root/'outputs/fourth_full_sequence/assets/vita_six_objects_newft_isaac52_selfcontacts.xml')
    p.add_argument('--source-smoothing-sigma-frames',type=float,default=1.)
    p.add_argument('--contact-distance-m',type=float,default=.04)
    p.add_argument('--stance-foot-weight',type=float,default=3.)
    p.add_argument('--swing-foot-weight',type=float,default=3.)
    p.add_argument('--com-weight',type=float,default=6.)
    p.add_argument('--foot-orientation-weight',type=float,default=.55)
    p.add_argument('--collision-only-root',action='store_true',help='Use true scene contacts rather than a global root-X half-plane')
    p.add_argument('--native-base-height',action='store_true')
    p.add_argument('--source-height-scale',type=float,default=.25)
    p.add_argument('--standing-reference',type=Path,default=root/'outputs/fourth_full_sequence/sonic_newft_default_stand_reference.npz')
    p.add_argument('--ik-self-collision',action='store_true',help='Enable virtual robot self geometry queries only during reference IK')
    a=p.parse_args()
    if os.name != 'posix':p.error('Build references on the SSH server; local host is preview only')
    import numpy as np
    from scipy.ndimage import gaussian_filter1d,median_filter
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation, Slerp
    from fourth_vita_retarget import read_segments, segment_vertices, compile_vita

    began=time.monotonic();a.output.parent.mkdir(parents=True,exist_ok=True)
    data=a.data
    tl=np.load(data/'timeline.npz',allow_pickle=False)
    h=np.load(data/'human/mesh_sequence_sim.npz',allow_pickle=False)
    hp=np.load(data/'human/smplx_params_sim.npz',allow_pickle=False)
    hr=np.load(data/'human/root_poses_sim.npz',allow_pickle=False)
    ob=np.load(data/'objects/poses_sim.npz',allow_pickle=False)
    times=np.asarray(tl['pts_seconds']);n=len(times)
    if n != 923:raise ValueError('This adapter expects the complete 923-frame handoff')
    times=times-times[0]
    human_mask=np.asarray(h['valid']).astype(bool).copy()
    human_mask[len(h['vertices_sim']):]=False
    hand_mask=np.asarray(h['hand_valid_full']).astype(bool)&human_mask[:,None]
    object_mask=np.asarray(ob['valid']).astype(bool)&np.isfinite(ob['position_m']).all(axis=2)&np.isfinite(ob['quaternion_wxyz']).all(axis=2)
    object_pose=np.concatenate([ob['position_m'],ob['quaternion_wxyz']],axis=2).copy()
    object_conf=np.zeros((n,6)); object_gap=np.zeros((n,6),np.int32)
    for oi in range(6):
        valid=np.flatnonzero(object_mask[:,oi])
        if not len(valid):raise ValueError('Object has no valid pose: '+str(oi+1))
        for j in range(3):object_pose[:,oi,j]=np.interp(times,times[valid],object_pose[valid,oi,j])
        q=ob['quaternion_wxyz'][valid,oi][:,[1,2,3,0]]
        filled=Slerp(times[valid],Rotation.from_quat(q))(np.clip(times,times[valid[0]],times[valid[-1]])).as_quat()[:,[3,0,1,2]]
        object_pose[:,oi,3:]=filled
        # Preserve valid source poses byte-for-byte; interpolation only fills holes.
        object_pose[valid,oi,:3]=ob['position_m'][valid,oi]
        object_pose[valid,oi,3:]=ob['quaternion_wxyz'][valid,oi]
        source_quality=np.asarray([{1:.85,2:.30,3:.20,4:.20,5:.45}.get(int(s),.10)for s in ob['status'][valid,oi]])
        object_conf[valid,oi]=source_quality
        for start,stop in ranges(~object_mask[:,oi]):
            object_gap[start:stop,oi]=stop-start
            dist=np.min(np.abs(times[start:stop,None]-times[valid][None,:]),axis=1)
            object_conf[start:stop,oi]=.45*np.exp(-dist/.5)

    segments=read_segments(np);faces=h['faces'];verts=h['vertices_sim']
    ids={
        'left_palm':segment_vertices(faces,segments,range(25,40),np),
        'right_palm':segment_vertices(faces,segments,range(40,55),np),
        'left_wrist':segment_vertices(faces,segments,[20],np),
        'right_wrist':segment_vertices(faces,segments,[21],np),
        'left_elbow':segment_vertices(faces,segments,[18],np),
        'right_elbow':segment_vertices(faces,segments,[19],np),
        'left_shoulder':segment_vertices(faces,segments,[16],np),
        'right_shoulder':segment_vertices(faces,segments,[17],np),
        'left_foot':segment_vertices(faces,segments,[7,10],np),
        'right_foot':segment_vertices(faces,segments,[8,11],np),
    }
    valid_h=np.flatnonzero(human_mask)
    centroids={}
    for key,index in ids.items():
        observed=np.median(verts[:,index],axis=1)
        target=np.column_stack([np.interp(times,times[valid_h],observed[valid_h,j]) for j in range(3)])
        centroids[key]=gaussian_filter1d(target,a.source_smoothing_sigma_frames,axis=0,mode='nearest')
        # Trailing missing human frames use exact last observed target, no extrapolation.
        centroids[key][valid_h[-1]+1:]=centroids[key][valid_h[-1]]
    source_root=np.asarray(hr['position_m']).copy()
    pelvis=np.column_stack([np.interp(times,times[valid_h],source_root[valid_h,j])for j in range(3)])
    pelvis=gaussian_filter1d(pelvis,a.source_smoothing_sigma_frames,axis=0,mode='nearest')
    lateral=centroids['left_shoulder']-centroids['right_shoulder'];lateral[:,2]=0
    lateral/=np.maximum(np.linalg.norm(lateral,axis=1)[:,None],1e-9)
    forward=np.cross(lateral,np.array([0.,0.,1.]))
    yaw=gaussian_filter1d(np.unwrap(np.arctan2(forward[:,1],forward[:,0])),a.source_smoothing_sigma_frames,mode='nearest')
    yaw[valid_h[-1]+1:]=yaw[valid_h[-1]]
    raw_shoulder_yaw=yaw.copy();yaw=gaussian_filter1d(yaw,4.,mode='nearest')
    for frame in range(1,n):yaw[frame]=np.clip(yaw[frame],yaw[frame-1]-1.2*(times[frame]-times[frame-1]),yaw[frame-1]+1.2*(times[frame]-times[frame-1]))
    human_conf=human_mask.astype(float)
    # Palm frames use wrist and proximal index/pinky geometry, independent of
    # fingertip curl. Frames are proxies derived from the supplied mesh.
    palm_frames=np.empty((n,2,3,3));raw_palm_frames=np.full((n,2,3,3),np.nan)
    anatomical_palm=np.empty((n,2,3));anatomical_wrist=np.empty((n,2,3))
    def palm_frame(wrist,index,pinky):
        f=.5*(index+pinky)-wrist;f/=max(np.linalg.norm(f),1e-9)
        l=index-pinky;l-=f*(f@l);l/=max(np.linalg.norm(l),1e-9)
        normal=np.cross(f,l);normal/=max(np.linalg.norm(normal),1e-9)
        return np.column_stack([f,l,normal])
    for side in range(2):
        ii=segment_vertices(faces,segments,[25+15*side],np);pi=segment_vertices(faces,segments,[31+15*side],np)
        wi=ids[['left_wrist','right_wrist'][side]]
        for frame in valid_h:
            patch=verts[frame,wi];index=np.median(verts[frame,ii],axis=0);pinky=np.median(verts[frame,pi],axis=0)
            initial_frame=palm_frame(np.median(patch,axis=0),index,pinky)
            projection=patch@initial_frame[:,0]
            wrist=np.median(patch[projection<=np.quantile(projection,.20)],axis=0)
            palm=np.median(patch[(projection>=np.quantile(projection,.25))&(projection<=np.quantile(projection,.85))],axis=0)
            anatomical_wrist[frame,side]=wrist;anatomical_palm[frame,side]=palm
            raw_palm_frames[frame,side]=palm_frame(wrist,index,pinky)
        quat=Rotation.from_matrix(raw_palm_frames[valid_h,side]).as_quat()
        for j in range(1,len(quat)):
            if quat[j]@quat[j-1]<0:quat[j]*=-1
        quat=gaussian_filter1d(quat,a.source_smoothing_sigma_frames,axis=0,mode='nearest');quat/=np.linalg.norm(quat,axis=1)[:,None]
        filled=Slerp(times[valid_h],Rotation.from_quat(quat))(np.clip(times,times[valid_h[0]],times[valid_h[-1]])).as_matrix()
        palm_frames[:,side]=filled
        for source in [anatomical_wrist,anatomical_palm]:
            for dim in range(3):source[:,side,dim]=np.interp(times,times[valid_h],source[valid_h,side,dim])
            source[:,side]=gaussian_filter1d(source[:,side],a.source_smoothing_sigma_frames,axis=0,mode='nearest')

    tips=np.empty((n,2,5,3));distances=np.full((n,2,6),np.inf)
    # Standard SMPL-X anatomical segmentation: index/middle/pinky/ring/thumb.
    tip_segments=[27,30,33,36,39]
    for side in range(2):
        for fi,segment in enumerate(tip_segments):
            ix=segment_vertices(faces,segments,[segment+15*side],np)
            observed=np.median(verts[:,ix],axis=1)
            for j in range(3):tips[:,side,fi,j]=np.interp(times,times[valid_h],observed[valid_h,j])
    object_trees=[];object_vertices=[]
    for oi in range(6):
        om=np.load(data/f'objects/object_{oi+1:02d}/mesh.npz',allow_pickle=False)['vertices']
        samples=om[::max(1,len(om)//3000)];object_trees.append(cKDTree(samples));object_vertices.append(samples)
    for frame in range(n):
        hf=min(frame,int(valid_h[-1]))
        for oi,tree in enumerate(object_trees):
            pose=object_pose[frame,oi]
            rot=Rotation.from_quat(pose[[4,5,6,3]]).as_matrix()
            for side,key in enumerate(['left_palm','right_palm']):
                local=(verts[hf,ids[key]]-pose[:3])@rot
                distances[frame,side,oi]=float(tree.query(local,workers=1)[0].min())
    candidate=distances<=a.contact_distance_m
    observed_contact=candidate&hand_mask[:,:,None]&object_mask[:,None,:]
    contact_conf=candidate.astype(float)*human_conf[:,None,None]*object_conf[:,None,:]

    mj,model,names,qa,va,lo,hi,speed,bodies,urdf=compile_vita(np)
    for name in ['left_ankle_pitch_joint','right_ankle_pitch_joint']:
        index=names.index(name);lo[index]=max(lo[index],-.751);hi[index]=min(hi[index],.414)
    if a.collision_model.exists():
        model=mj.MjModel.from_xml_path(str(a.collision_model))
        jids=np.asarray([mj.mj_name2id(model,mj.mjtObj.mjOBJ_JOINT,name)for name in names])
        qa=model.jnt_qposadr[jids];va=model.jnt_dofadr[jids]
        bodies={name:mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,name)for name in bodies}
        object_qadr=[int(model.jnt_qposadr[mj.mj_name2id(model,mj.mjtObj.mjOBJ_JOINT,f'target_object_{i:02d}_free')])for i in range(1,7)]
    else:object_qadr=[]
    if a.ik_self_collision:
        for gid in range(model.ngeom):
            bn=mj.mj_id2name(model,mj.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[gid]))or'world'
            if bn!='world'and not bn.startswith('target_object_')and model.geom_contype[gid]>0:model.geom_conaffinity[gid]|=int(model.geom_contype[gid])
    d=mj.MjData(model)
    ancestors=[]
    for bid in range(model.nbody):
        path={};cur=bid;distance=0
        while cur>0:
            path[cur]=distance;cur=int(model.body_parentid[cur]);distance+=1
        path[0]=distance;ancestors.append(path)
    def adjacent_structure(b1,b2):
        common=set(ancestors[b1]).intersection(ancestors[b2])
        return min(ancestors[b1][b]+ancestors[b2][b]for b in common)<=3
    mj.mj_forward(model,d);robot_palm_local=[];robot_palm_site_local=[]
    for prefix in ['lh','rh']:
        hb=mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,prefix+'_hand_base_link')
        ib=mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,prefix+'_index_proximal')
        pb=mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,prefix+'_pinky_proximal')
        baseR=d.xmat[hb].reshape(3,3)
        frame=palm_frame(d.xpos[hb],d.xpos[ib],d.xpos[pb])
        robot_palm_local.append(baseR.T@frame)
        gid=mj.mj_name2id(model,mj.mjtObj.mjOBJ_GEOM,'collision_'+prefix+'_palm_mesh')
        if gid<0:raise ValueError('Native palm collision geometry missing: '+prefix)
        center=d.geom_xpos[gid]+d.geom_xmat[gid].reshape(3,3)@model.geom_aabb[gid,:3]
        robot_palm_site_local.append(baseR.T@(center-d.xpos[hb]))
    for name in ['left_wrist_pitch_link','right_wrist_pitch_link']:bodies[name]=mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,name)
    desired_hand_rot=np.asarray([[palm_frames[frame,side]@robot_palm_local[side].T for side in range(2)]for frame in range(n)])
    baseline_object_pose=object_pose.copy()
    if object_qadr:
        # Only missing poses may change in this collision/scene-prior fill.
        # The original valid observations remain exact immutable constraints.
        previous_correction=np.zeros((6,3))
        body_to_object={mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,f'target_object_{i:02d}'):i-1 for i in range(1,7)}
        for frame in range(n):
            missing=~object_mask[frame]
            correction=.75*previous_correction;correction[~missing]=0
            object_pose[frame,:,:3]+=correction
            d.qpos[:3]=[20,20,3]
            for _ in range(16):
                for oi,adr in enumerate(object_qadr):d.qpos[adr:adr+7]=object_pose[frame,oi]
                mj.mj_forward(model,d);A=[];e=[]
                for ci in range(d.ncon):
                    c=d.contact[ci]
                    if c.dist>=-.0005:continue
                    b1,b2=int(model.geom_bodyid[c.geom1]),int(model.geom_bodyid[c.geom2])
                    oi1=body_to_object.get(b1);oi2=body_to_object.get(b2)
                    if oi1 is None and oi2 is None:continue
                    if any(b>0 and b not in body_to_object for b in [b1,b2]):continue
                    movable1=oi1 is not None and missing[oi1];movable2=oi2 is not None and missing[oi2]
                    if not(movable1 or movable2):continue
                    normal=np.asarray(c.frame).reshape(3,3)[0];row=np.zeros((6,3))
                    if movable1:row[oi1]-=normal
                    if movable2:row[oi2]+=normal
                    A.append(row.ravel());e.append(.0005-float(c.dist))
                if not A:break
                J=np.asarray(A);step=np.linalg.solve(J.T@J+.05*np.eye(18),J.T@np.asarray(e)).reshape(6,3)
                step[~missing]=0;length=np.linalg.norm(step,axis=1)
                step*=np.minimum(1.,.015/np.maximum(length,1e-9))[:,None]
                object_pose[frame,:,:3]+=step
                delta=object_pose[frame,:,:3]-baseline_object_pose[frame,:,:3]
                dl=np.linalg.norm(delta,axis=1);delta*=np.minimum(1.,.15/np.maximum(dl,1e-9))[:,None]
                object_pose[frame,:,:3]=baseline_object_pose[frame,:,:3]+delta
            previous_correction=object_pose[frame,:,:3]-baseline_object_pose[frame,:,:3]
        # Re-evaluate geometry only where the fill changed; raw observed
        # contact candidates continue to refer to unmodified source poses.
        for frame,oi in np.argwhere(~object_mask):
            hf=min(frame,int(valid_h[-1]));pose=object_pose[frame,oi]
            rot=Rotation.from_quat(pose[[4,5,6,3]]).as_matrix()
            for side,key in enumerate(['left_palm','right_palm']):
                local=(verts[hf,ids[key]]-pose[:3])@rot
                distances[frame,side,oi]=float(object_trees[oi].query(local,workers=1)[0].min())
        candidate=distances<=a.contact_distance_m
        observed_contact=candidate&hand_mask[:,:,None]&object_mask[:,None,:]
        contact_conf=candidate.astype(float)*human_conf[:,None,None]*object_conf[:,None,:]
    selected=[i for i,name in enumerate(names)if name.startswith(('left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle','waist_','left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist'))]
    sq=qa[selected];sv=va[selected]
    # Root height participates in IK to accommodate robot/human leg length.
    solve_q=np.r_[0,1,2,sq];solve_v=np.r_[0,1,2,sv]
    floor_z=-1.790970733158933
    foot_geoms={side:[g for g in range(model.ngeom)if int(model.geom_bodyid[g])==bodies[side+'_ankle_roll_link']and model.geom_contype[g]>0]for side in ('left','right')}
    floor_id=mj.mj_name2id(model,mj.mjtObj.mjOBJ_GEOM,'assumed_floor')
    def foot_lowest(side):
        if floor_id>=0:
            best=None
            for g in foot_geoms[side]:
                witness=np.zeros(6);distance=float(mj.mj_geomDistance(model,d,g,floor_id,3.,witness))
                if best is None or distance<best[0]:best=(distance,witness[:3].copy())
            return floor_z+best[0],best[1]
        gid=min(foot_geoms[side],key=lambda g:d.geom_xpos[g,2]-model.geom_size[g,0])
        point=d.geom_xpos[gid].copy();point[2]-=model.geom_size[gid,0]
        return point[2],point
    def soles():return np.asarray([foot_lowest(side)[0]for side in ('left','right')])
    prev=np.zeros(52)
    for joint,value in {'left_hip_pitch_joint':-.10,'right_hip_pitch_joint':-.10,'left_knee_joint':.20,'right_knee_joint':.20,'left_ankle_pitch_joint':-.10,'right_ankle_pitch_joint':-.10,'left_elbow_joint':.4,'right_elbow_joint':.4}.items():
        if joint in names:prev[names.index(joint)]=value
    d.qpos[:3]=pelvis[0];d.qpos[3:7]=[np.cos(yaw[0]/2),0,0,np.sin(yaw[0]/2)];d.qpos[qa]=prev
    mj.mj_forward(model,d)
    root_z_offset=floor_z+.002-min(soles())
    ankle_offset=np.array([d.xpos[bodies[s+'_ankle_roll_link'],2]-soles()[i]for i,s in enumerate(('left','right'))])
    feet=np.stack([centroids[s+'_foot']for s in ('left','right')],axis=1)
    footbase=np.percentile(feet[valid_h,:,2],10,axis=0)
    foot_targets=feet.copy()
    raw_foot_targets=feet.copy();raw_foot_targets[:,:,2]=floor_z+.002+ankle_offset[None,:]+np.maximum(feet[:,:,2]-footbase[None,:],0)
    root_goals=pelvis.copy();root_goals[:,2]+=root_z_offset
    native_stand_height=None
    if a.native_base_height:
        stand=np.load(a.standing_reference,allow_pickle=False)
        native_stand_height=float(stand['qpos'][0,2])-floor_z
        source_origin=float(np.median(pelvis[:30,2]))
        root_goals[:,2]=floor_z+native_stand_height+np.clip(a.source_height_scale*(pelvis[:,2]-source_origin),-.06,.04)
    if not a.collision_only_root:root_goals[:,0]=np.maximum(root_goals[:,0],a.safe_root_x)
    root_goals=gaussian_filter1d(root_goals,4.,axis=0,mode='nearest')
    for frame in range(1,n):
        dt=times[frame]-times[frame-1];delta=root_goals[frame,:2]-root_goals[frame-1,:2];length=np.linalg.norm(delta)
        root_goals[frame,:2]=root_goals[frame-1,:2]+delta*min(1.,.65*dt/max(length,1e-9))
        root_goals[frame,2]=np.clip(root_goals[frame,2],root_goals[frame-1,2]-.4*dt,root_goals[frame-1,2]+.4*dt)
    # Translate planned support with the table-safe root adaptation. Keeping
    # source feet behind while moving pelvis sideways produces impossible COM.
    foot_targets[:,:,:2]+=(root_goals[:,:2]-pelvis[:,:2])[:,None,:]
    source_foot_speed=np.linalg.norm(np.gradient(feet[:,:,:2],times,axis=0),axis=2)
    relative_lift=feet[:,:,2]-np.min(feet[:,:,2],axis=1)[:,None]
    planned_stance=(source_foot_speed<.20)&(relative_lift<.06)
    planned_stance=median_filter(planned_stance.astype(np.uint8),size=(7,1),mode='nearest').astype(bool)
    for side in range(2):
        for _ in range(3):
            for value in [True,False]:
                for start,stop in ranges(planned_stance[:,side]==value):
                    if stop-start<6:planned_stance[start:stop,side]=not value
    for start,stop in ranges(~np.any(planned_stance,axis=1)):
        previous=np.flatnonzero(planned_stance[start-1])if start else []
        support=int(previous[0])if len(previous)else int(np.argmin(np.mean(source_foot_speed[start:stop],axis=0)))
        planned_stance[start:stop,support]=True
    foot_targets[:,:,2]=floor_z+.002+ankle_offset[None,:]+np.where(planned_stance,0.,relative_lift)
    # Plant-foot anchors are fixed over each inferred stance interval; source
    # targets and the full correction are retained in the artifact.
    for side in range(2):
        for start,stop in ranges(planned_stance[:,side]):
            foot_targets[start:stop,side,:2]=np.median(foot_targets[start:stop,side,:2],axis=0)
        for start,stop in ranges(~planned_stance[:,side]):
            if start==0 or stop>=n:continue
            duration=times[stop]-times[start-1]
            u=(times[start:stop]-times[start-1])/duration;blend=6*u**5-15*u**4+10*u**3
            before=foot_targets[start-1,side,:2].copy();after=foot_targets[stop,side,:2].copy()
            foot_targets[start:stop,side,:2]=before[None,:]+blend[:,None]*(after-before)[None,:]
            lift=.05 if np.linalg.norm(after-before)>.03 else .025
            foot_targets[start:stop,side,2]=floor_z+.002+ankle_offset[side]+lift*np.sin(np.pi*u)
    pre_non_crossing_foot_targets=foot_targets.copy()
    for frame in range(n):
        lateral_axis=np.array([-np.sin(yaw[frame]),np.cos(yaw[frame]),0.])
        signed_width=(foot_targets[frame,0]-foot_targets[frame,1])@lateral_axis
        if signed_width<.20:
            shift=.5*(.20-signed_width)*lateral_axis
            foot_targets[frame,0]+=shift;foot_targets[frame,1]-=shift
    planned_com_target=np.asarray([foot_targets[f,planned_stance[f],:2].mean(axis=0)+np.asarray([.02*np.cos(yaw[f]),.02*np.sin(yaw[f])])for f in range(n)])
    planned_com_target=gaussian_filter1d(planned_com_target,8.,axis=0,mode='nearest')
    # Map SMPL-X local finger flexion magnitudes to 11 native finger joints/hand.
    finger_targets=np.zeros((n,52))
    finger_order={'index':0,'middle':1,'pinky':2,'ring':3,'thumb':4}
    for side,prefix in [('left','lh_'),('right','rh_')]:
        poses=np.asarray(hp[side+'_hand_pose']).reshape(-1,5,3,3)
        mags=np.linalg.norm(poses,axis=-1)
        for ji,name in enumerate(names):
            if not name.startswith(prefix):continue
            finger=next((k for k in finger_order if prefix+k in name),None)
            if finger is None:continue
            fi=finger_order[finger]
            values=mags[:,fi,0] if 'mcp' in name or 'cmc_yaw' in name else .5*(mags[:,fi,1]+mags[:,fi,2])
            if 'cmc_pitch' in name:values=.3*mags[:,fi,0]
            finger_targets[:,ji]=np.clip(np.interp(times,times[:len(values)],values),lo[ji],hi[ji])
    finger_targets=gaussian_filter1d(finger_targets,2.,axis=0,mode='nearest')
    practical_speed=speed.copy()
    for j,name in enumerate(names):
        cap=6. if 'knee'in name else 4. if any(k in name for k in ['hip','ankle','elbow','wrist'])else 3. if 'shoulder'in name else 2. if 'waist'in name else 1.
        practical_speed[j]=min(practical_speed[j],cap)
    output=[];fk_hands=[];fk_palms=[];fk_feet=[];errors=[];root_delta=[];sole_clear=[];adapted_hands=[];com_targets=[];com_fk=[]
    robot_root_bid=mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,'base_link')
    raw_full_hand_centroid=np.stack([centroids['left_palm'],centroids['right_palm']],axis=1)
    human_hand=anatomical_palm
    for frame in range(n):
        rt=root_goals[frame].copy()
        # The full source root path is retained except collision-safe tabletop projection.
        if not a.collision_only_root and 1.15 < rt[1] < 2.46 and rt[0] < a.safe_root_x:rt[0]=a.safe_root_x
        d.qpos[:3]=rt;d.qpos[3:7]=[np.cos(yaw[frame]/2),0,0,np.sin(yaw[frame]/2)]
        d.qpos[qa]=prev
        dt=times[frame]-times[frame-1]if frame else 1.
        frame_lo=np.maximum(lo,prev-practical_speed*dt)if frame else lo
        frame_hi=np.minimum(hi,prev+practical_speed*dt)if frame else hi
        for oi,adr in enumerate(object_qadr):d.qpos[adr:adr+7]=object_pose[frame,oi]
        for j,name in enumerate(names):
            if name.startswith(('lh_','rh_')):d.qpos[qa[j]]=np.clip(finger_targets[frame,j],frame_lo[j],frame_hi[j])
        root_rot=Rotation.from_euler('z',yaw[frame]).as_matrix()
        com_target=planned_com_target[frame]
        com_targets.append(com_target)
        hand_target=human_hand[frame].copy()
        for side in range(2):
            local=root_rot.T@(hand_target[side]-rt)
            if abs(local[0])<.30 and -.4<local[2]<.25:
                local[1]=max(local[1],.34)if side==0 else min(local[1],-.34)
                hand_target[side]=rt+root_rot@local
        adapted_hands.append(hand_target)
        goals=[('lh_hand_base_link',hand_target[0],1.5),('rh_hand_base_link',hand_target[1],1.5),
               ('left_elbow_link',centroids['left_elbow'][frame],.2),('right_elbow_link',centroids['right_elbow'][frame],.2),
               ('left_ankle_roll_link',foot_targets[frame,0],a.stance_foot_weight if planned_stance[frame,0]else a.swing_foot_weight),('right_ankle_roll_link',foot_targets[frame,1],a.stance_foot_weight if planned_stance[frame,1]else a.swing_foot_weight),
               ('left_wrist_pitch_link',anatomical_wrist[frame,0],.5),('right_wrist_pitch_link',anatomical_wrist[frame,1],.5)]
        desired_foot_rot=Rotation.from_euler('z',yaw[frame]).as_matrix()
        for iteration in range(a.iterations):
            mj.mj_forward(model,d);jac=[];err=[]
            for bname,target,weight in goals:
                jp=np.zeros((3,model.nv));jr=np.zeros_like(jp);bid=bodies[bname]
                point=d.xpos[bid].copy()
                if bname in ['lh_hand_base_link','rh_hand_base_link']:
                    side=0 if bname.startswith('lh_')else 1
                    point+=d.xmat[bid].reshape(3,3)@robot_palm_site_local[side]
                mj.mj_jac(model,d,jp,jr,point,bid)
                jac.append(np.sqrt(weight)*jp[:,solve_v]);err.append(np.sqrt(weight)*(target-point))
            for side,bname in enumerate(['lh_hand_base_link','rh_hand_base_link']):
                bid=bodies[bname];jp=np.zeros((3,model.nv));jr=np.zeros_like(jp)
                mj.mj_jacBody(model,d,jp,jr,bid)
                angle_error=Rotation.from_matrix(desired_hand_rot[frame,side]@d.xmat[bid].reshape(3,3).T).as_rotvec()
                jac.append(np.sqrt(.08)*jr[:,solve_v]);err.append(np.sqrt(.08)*angle_error)
            for side in ('left','right'):
                bid=bodies[side+'_ankle_roll_link']
                jp=np.zeros((3,model.nv));jr=np.zeros_like(jp)
                mj.mj_jacBody(model,d,jp,jr,bid)
                rot_error=Rotation.from_matrix(desired_foot_rot@d.xmat[bid].reshape(3,3).T).as_rotvec()
                jac.append(np.sqrt(a.foot_orientation_weight)*jr[:,solve_v]);err.append(np.sqrt(a.foot_orientation_weight)*rot_error)
                low_z,point=foot_lowest(side)
                if point[2]<floor_z+.002:
                    mj.mj_jac(model,d,jp,jr,point,bid)
                    jac.append(np.sqrt(8.)*jp[2:3,solve_v]);err.append(np.asarray([np.sqrt(8.)*(floor_z+.002-point[2])]))
            root_jac=np.zeros((3,len(solve_v)));root_jac[:,:3]=.3*np.eye(3)
            jac.append(root_jac);err.append(.3*(rt-d.qpos[:3]))
            com_jac=np.zeros((3,model.nv));mj.mj_jacSubtreeCom(model,d,com_jac,robot_root_bid)
            jac.append(np.sqrt(a.com_weight)*com_jac[:2,solve_v]);err.append(np.sqrt(a.com_weight)*(com_target-d.subtree_com[robot_root_bid,:2]))
            for side,prefix in enumerate(['left','right']):
                if not np.any(observed_contact[frame,side]):continue
                oi=int(np.argmin(np.where(observed_contact[frame,side],distances[frame,side],np.inf)))
                pose=object_pose[frame,oi];normal=rt-pose[:3];normal[2]=0
                normal/=max(np.linalg.norm(normal),1e-9)
                R=Rotation.from_quat(pose[[4,5,6,3]]).as_matrix()
                support=float(np.max((object_vertices[oi]@R.T)@normal))
                bid=mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,prefix+'_wrist_pitch_link')
                gap=float((d.xpos[bid]-pose[:3])@normal)-support-.055
                if gap<0:
                    jp=np.zeros((3,model.nv));jr=np.zeros_like(jp);mj.mj_jacBody(model,d,jp,jr,bid)
                    jac.append(np.sqrt(8.)*(normal@jp)[None,solve_v]);err.append(np.asarray([-np.sqrt(8.)*gap]))
            # Resolve active self/scene/object penetrations using point
            # Jacobians. Objects remain source reference inputs, not unknowns.
            for ci in range(d.ncon):
                c=d.contact[ci]
                b1,b2=int(model.geom_bodyid[c.geom1]),int(model.geom_bodyid[c.geom2])
                n1=mj.mj_id2name(model,mj.mjtObj.mjOBJ_BODY,b1)or'world'
                n2=mj.mj_id2name(model,mj.mjtObj.mjOBJ_BODY,b2)or'world'
                robot1=b1>0 and not n1.startswith('target_object_')
                robot2=b2>0 and not n2.startswith('target_object_')
                if not(robot1 or robot2)or c.dist>=.002:continue
                # The original training plant filters self contact. Coarse
                # proxies overlap around nearby joints by construction; do
                # not distort pose attempting to separate a hip from pelvis.
                if robot1 and robot2:
                    if adjacent_structure(b1,b2):continue
                    if not any(name.startswith(('lh_','rh_','left_wrist','right_wrist'))for name in [n1,n2]):continue
                j1=np.zeros((3,model.nv));j2=np.zeros_like(j1);jr=np.zeros_like(j1)
                if robot1:mj.mj_jac(model,d,j1,jr,c.pos,b1)
                if robot2:mj.mj_jac(model,d,j2,jr,c.pos,b2)
                normal=np.asarray(c.frame).reshape(3,3)[0]
                jac.append(np.sqrt(12.)*(normal@(j2-j1))[:,None].T[:,solve_v])
                err.append(np.asarray([np.sqrt(12.)*(.002-float(c.dist))]))
            J=np.vstack(jac);e=np.concatenate(err)
            delta=np.linalg.solve(J.T@J+.02*np.eye(len(solve_v)),J.T@e)
            low_root=rt-np.asarray([.12,.12,.05 if a.native_base_height else .20]);high_root=rt+np.asarray([.12,.12,.05 if a.native_base_height else .35])
            if not a.collision_only_root and 1.15<rt[1]<2.46:low_root[0]=max(low_root[0],a.safe_root_x)
            if frame:
                previous_root=output[-1][:3]
                low_root=np.maximum(low_root,previous_root-np.asarray([.8,.8,.5])*dt)
                high_root=np.minimum(high_root,previous_root+np.asarray([.8,.8,.5])*dt)
                low_root=np.minimum(low_root,high_root)
            d.qpos[solve_q]=np.clip(d.qpos[solve_q]+np.clip(delta,-.12,.12),np.r_[low_root,frame_lo[selected]],np.r_[high_root,frame_hi[selected]])
            if np.max(np.abs(e))<.006:break
        current=d.qpos[qa].copy()
        if frame:
            dt=times[frame]-times[frame-1];current=np.clip(current,prev-practical_speed*dt,prev+practical_speed*dt)
        d.qpos[qa]=current;prev=current;mj.mj_forward(model,d)
        # Hard clearance projection is a reference repair only. Its exact root
        # offset is saved; active simulation never uses this operation.
        clearance=soles()-floor_z
        d.qpos[2]+=.002-min(clearance)
        mj.mj_forward(model,d)
        root_delta.append(d.qpos[:3]-pelvis[frame])
        com_fk.append(d.subtree_com[robot_root_bid].copy())
        output.append(np.r_[d.qpos[:7],d.qpos[qa]])
        fk_hands.append([d.xpos[bodies[s]].copy()for s in ('lh_hand_base_link','rh_hand_base_link')])
        fk_feet.append([d.xpos[bodies[s+'_ankle_roll_link']].copy()for s in ('left','right')])
        current_errors=[];palm_positions=[]
        for bn,target,w in goals:
            point=d.xpos[bodies[bn]].copy()
            if bn in ['lh_hand_base_link','rh_hand_base_link']:
                side=0 if bn.startswith('lh_')else 1;point+=d.xmat[bodies[bn]].reshape(3,3)@robot_palm_site_local[side];palm_positions.append(point.copy())
            current_errors.append(float(np.linalg.norm(point-target)))
        errors.append(current_errors);fk_palms.append(palm_positions)
        sole_clear.append(soles()-floor_z)
        if frame%100==0:print(json.dumps({'stage':'full_body_ik','frame':frame,'frames':n,'wall_s':time.monotonic()-began}),flush=True)
    qpos=np.asarray(output)
    # Missing trailing human observations use an explicit last-observation hold.
    qpos[valid_h[-1]+1:]=qpos[valid_h[-1]]
    last=int(valid_h[-1])
    if last+1<n:
        d.qpos[:7]=qpos[last,:7];d.qpos[qa]=qpos[last,7:];mj.mj_forward(model,d)
        for frame in range(last+1,n):
            fk_hands[frame]=np.asarray(fk_hands[last]).copy();fk_palms[frame]=np.asarray(fk_palms[last]).copy()
            fk_feet[frame]=np.asarray(fk_feet[last]).copy();com_fk[frame]=np.asarray(com_fk[last]).copy()
            sole_clear[frame]=np.asarray(sole_clear[last]).copy();root_delta[frame]=qpos[last,:3]-pelvis[frame]
            actual=np.vstack([fk_palms[last],d.xpos[bodies['left_elbow_link']],d.xpos[bodies['right_elbow_link']],fk_feet[last],d.xpos[bodies['left_wrist_pitch_link']],d.xpos[bodies['right_wrist_pitch_link']]])
            targets=np.vstack([adapted_hands[frame],centroids['left_elbow'][frame],centroids['right_elbow'][frame],foot_targets[frame],anatomical_wrist[frame]])
            errors[frame]=np.linalg.norm(actual-targets,axis=1).tolist()
    payload=dict(qpos=qpos,dof_names=np.asarray(names),time_s=times,source_frame_ids=np.arange(n,dtype=np.int32),frame_ids=np.arange(n,dtype=np.int32),
        object_ids=np.arange(1,7),object_target_pose=object_pose,object_observed_mask=object_mask,object_imputed_mask=~object_mask,
        object_status=ob['status'],object_confidence=object_conf,object_gap_frames=object_gap,
        object_confidence_method=np.asarray('Heuristic provenance score, not calibrated probability: status1=.85, prior-assisted5=.45, other valid statuses=.2-.3; missing gap score decays from .45'),
        baseline_filled_object_pose=baseline_object_pose,object_imputation_collision_correction_xyz_m=object_pose[:,:,:3]-baseline_object_pose[:,:,:3],
        human_observed_mask=human_mask,human_imputed_mask=~human_mask,human_confidence=human_conf,hand_observed_mask=hand_mask,
        human_hand_target_world=human_hand,human_wrist_target_world=anatomical_wrist,raw_source_hand_fullmesh_centroid_world=raw_full_hand_centroid,
        robot_palm_target_world=np.asarray(adapted_hands),robot_palm_contact_site_local=np.asarray(robot_palm_site_local),
        robot_hand_target_world=np.asarray(adapted_hands)-np.einsum('nsij,sj->nsi',desired_hand_rot,np.asarray(robot_palm_site_local)),
        robot_hand_safety_adaptation_m=np.asarray(adapted_hands)-human_hand,
        hand_anchor_semantics=np.asarray('Source wrist proxy=proximal20% of SMPL-X wrist/palm segment; source palm proxy=central25-85% of that segment, excludes distal fingers. Native wrist_pitch matches wrist; actual palm mesh AABB center site matches palm. robot_hand_target_world is corresponding hand-base target, robot_palm_target_world is contact-site target.'),
        raw_source_palm_frame_rotation_world=raw_palm_frames,source_palm_frame_rotation_world_filled=palm_frames,
        robot_hand_orientation_target_world_wxyz=Rotation.from_matrix(desired_hand_rot.reshape(-1,3,3)).as_quat().reshape(n,2,4)[:,:,[3,0,1,2]],
        robot_palm_anatomical_frame_in_hand_base=np.asarray(robot_palm_local),
        human_elbow_target_world=np.stack([centroids[s+'_elbow']for s in ('left','right')],axis=1),
        human_fingertip_target_world=tips,human_foot_target_world=feet,robot_foot_target_world=foot_targets,
        robot_foot_non_crossing_adaptation_m=foot_targets-pre_non_crossing_foot_targets,
        robot_foot_total_support_adaptation_m=foot_targets-raw_foot_targets,robot_foot_source_based_baseline_world=raw_foot_targets,
        planned_stance_mask=planned_stance,source_foot_speed_m_s=source_foot_speed,
        practical_joint_velocity_limit_rad_s=practical_speed,raw_source_shoulder_yaw_rad=raw_shoulder_yaw,
        robot_com_target_world_xy_m=np.asarray(com_targets),robot_com_fk_world_m=np.asarray(com_fk),
        support_adaptation_description=np.asarray('Table-safe root XY shift also applied to feet; source-foot speed/relative lift infer stance, median filtering and >=.2s phases; stance XY anchored, swing quintic XY and sinusoidal5cm clearance; COM-center lowpass8frames, practical joint/root velocity bounds; one foot always grounded. Explicit dynamic-feasibility adaptations, not raw observations.'),
        robot_hand_fk_world=np.asarray(fk_hands),robot_palm_fk_world=np.asarray(fk_palms),robot_foot_fk_world=np.asarray(fk_feet),ik_error_m=np.asarray(errors),
        foot_sole_clearance_m=np.asarray(sole_clear),source_hand_object_min_distance_m=distances,
        contact_candidate=candidate,contact_candidate_observed=observed_contact,contact_candidate_confidence=contact_conf,
        raw_source_object_position_m=ob['position_m'],raw_source_object_quaternion_wxyz=ob['quaternion_wxyz'],raw_source_object_valid=ob['valid'],
        raw_source_root_position_m=source_root,raw_source_root_quaternion_wxyz=hr['quaternion_wxyz'],
        root_alignment_offset_xyz_m=np.asarray(root_delta),source_root_yaw_rad=yaw,physics_floor_z_m=np.asarray(floor_z),
        root_table_safety_projection=np.asarray(not a.collision_only_root),root_scene_collision_ik_enabled=np.asarray(True),finger_mapping_approximate=np.asarray(True),
        root_safety_constraint_mode=np.asarray('actual geometry contacts only'if a.collision_only_root else'global X half-plane plus actual geometry'),
        native_base_height_prior_m=np.asarray(native_stand_height if native_stand_height is not None else np.nan),
        source_height_scale=np.asarray(a.source_height_scale if a.native_base_height else 1.),
        ik_virtual_self_collision_enabled=np.asarray(a.ik_self_collision),
        missing_human_fill_method=np.asarray('Last observed full-body pose/targets held for trailing 85 frames; not observed motion'),
        missing_object_fill_method=np.asarray('Translation linear interpolation + rotation SLERP within observed span; nearest endpoint hold outside; confidence decays with gap'),
        object_imputation_collision_prior=np.asarray('Only unobserved translations may move by <=15cm to resolve object/object or support-plane penetration; observed poses unchanged; baseline fill and deltas retained'),
        target_semantics=np.asarray('Anatomical SMPL-X mesh segment targets with temporal smoothing and table-safe root adaptation; candidate contacts are geometric estimates, not force ground truth'))
    stance_foot_speed=np.linalg.norm(np.gradient(np.asarray(fk_feet)[:,:,:2],times,axis=0),axis=2)*planned_stance
    payload['robot_stance_foot_speed_m_s']=stance_foot_speed
    payload['robot_com_support_center_error_m']=np.linalg.norm(np.asarray(com_fk)[:,:2]-np.asarray(com_targets),axis=1)
    payload['root_deviation_from_raw_source_xyz_m']=qpos[:,:3]-source_root
    for key in hp.files:payload['raw_source_smplx_'+key]=hp[key]
    np.savez_compressed(a.output,**payload)
    report={'schema':'fourth_full_reference.v1','output':str(a.output),'sha256':sha(a.output),'frames':n,'duration_s':float(times[-1]),
        'human_observed_frames':int(human_mask.sum()),'human_imputed_ranges':ranges(~human_mask),
        'object_observed_frames':object_mask.sum(axis=0).tolist(),'object_imputed_ranges':{str(i+1):ranges(~object_mask[:,i])for i in range(6)},
        'candidate_contacts_per_hand_object':candidate.sum(axis=0).tolist(),'observed_candidate_contacts_per_hand_object':observed_contact.sum(axis=0).tolist(),
        'source_object_valid_values_preserved':bool(np.array_equal(object_pose[object_mask],np.concatenate([ob['position_m'],ob['quaternion_wxyz']],axis=2)[object_mask])),
        'all_filled_targets_finite':bool(np.isfinite(qpos).all()and np.isfinite(object_pose).all()),'qpos_shape':list(qpos.shape),'object_pose_shape':list(object_pose.shape),
        'ik_hand_error_median_m':np.median(errors,axis=0)[:2].tolist(),'ik_hand_error_max_m':np.max(errors,axis=0)[:2].tolist(),
        'ik_foot_error_median_m':np.median(errors,axis=0)[4:6].tolist(),'ik_foot_error_max_m':np.max(errors,axis=0)[4:6].tolist(),
        'ik_weights':{'stance_foot':a.stance_foot_weight,'swing_foot':a.swing_foot_weight,'com':a.com_weight,'foot_orientation':a.foot_orientation_weight},
        'training_startup_ankle_pitch_limits_rad':[-.751,.414],
        'foot_clearance_min_m':np.min(sole_clear,axis=0).tolist(),'root_source_motion_range_xyz_m':np.ptp(pelvis,axis=0).tolist(),
        'root_safe_x_m':a.safe_root_x,'root_alignment_offset_max_m':float(np.max(np.linalg.norm(root_delta,axis=1))),
        'source_root_xy_error_max_m':float(np.nanmax(np.linalg.norm(qpos[:,:2]-source_root[:,:2],axis=1))),
        'root_height_above_floor_range_m':[float(np.min(qpos[:,2]-floor_z)),float(np.max(qpos[:,2]-floor_z))],
        'planned_stance_fk_speed_peak_m_s':float(np.max(stance_foot_speed)),
        'root_global_x_constraint_enabled':not a.collision_only_root,'native_stand_height_prior_m':native_stand_height,
        'source_pelvis_height_delta_scale':a.source_height_scale if a.native_base_height else 1.,
        'ik_virtual_self_filter_override':a.ik_self_collision,
        'collision_model':str(a.collision_model),'collision_model_sha256':sha(a.collision_model)if a.collision_model.exists()else None,
        'object_imputation_collision_correction_max_m':float(np.max(np.linalg.norm(object_pose[:,:,:3]-baseline_object_pose[:,:,:3],axis=2))),
        'object_imputation_collision_corrected_frames':np.sum(np.linalg.norm(object_pose[:,:,:3]-baseline_object_pose[:,:,:3],axis=2)>1e-8,axis=0).tolist(),
        'hand_orientation_method':'Wrist/index-proximal/pinky-proximal anatomical frame mapped to corresponding native robot points; weak orientation IK; fingertip curl not used for palm frame',
        'ik_structural_self_contact_policy':'Ignore native body/body self pairs and proxy overlaps within kinematic graph distance <=3; optimize nonadjacent hand/wrist versus robot self contacts. Dynamic SONIC uses the source training self-contact filter.',
        'finger_mapping':'SMPL-X local axis-angle magnitudes to native 11 flexion joints/hand; approximate, not anatomical axis calibration',
        'source_sha256':{str(p.relative_to(data)):sha(p)for p in [data/'timeline.npz',data/'human/smplx_params_sim.npz',data/'human/root_poses_sim.npz',data/'objects/poses_sim.npz']},
        'source_script_sha256':sha(Path(__file__)),'elapsed_wall_s':time.monotonic()-began,'physics_success_claimed':False,
        'limitations':['Last 85 human frames contain no source observation and use an explicit hold.',
          'Invalid object poses are imputed; original masks, status and raw NaNs are retained.',
          'Full-body IK and finger flexion mapping do not establish dynamic stability or physical contact.']}
    a.output.with_suffix('.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
