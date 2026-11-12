"""Fuse fixed native leg cycles with source-space wrist/palm upper-body IK.

This constructs a kinematic reference only. It cannot establish that SONIC
tracks the planned root path or that any object is physically manipulated.
"""
from __future__ import annotations
import argparse,hashlib,json,os,time
from pathlib import Path


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    root=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,default=root/'outputs/fourth_full_sequence/reference_v10.npz')
    p.add_argument('--leg-plan',type=Path,required=True)
    p.add_argument('--model',type=Path,default=root/'outputs/fourth_full_sequence/assets/vita_six_objects_newft_source_table_v1.xml')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--iterations',type=int,default=40)
    p.add_argument('--table-clearance-guard',action='store_true')
    p.add_argument('--optimize-fingers',action='store_true')
    p.add_argument('--verify-kinematics',action='store_true',help='Compare position-only evaluation against mj_forward, with no integration')
    a=p.parse_args()
    if os.name!='posix':p.error('Server reference construction only')
    import numpy as np
    import mujoco as mj
    from scipy.spatial.transform import Rotation
    from scipy.spatial import ConvexHull
    from scipy.ndimage import gaussian_filter1d
    # NpzFile decompresses on every key access. Cache once before the inner
    # per-finger/per-iteration loop instead of rereading whole qpos arrays.
    with np.load(a.reference,allow_pickle=False)as archive:z={key:archive[key]for key in archive.files}
    with np.load(a.leg_plan,allow_pickle=False)as archive:g={key:archive[key]for key in archive.files}
    m=mj.MjModel.from_xml_path(str(a.model));d=mj.MjData(m)
    names=[str(v)for v in z['dof_names']];jids=np.asarray([mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,v)for v in names]);qa=m.jnt_qposadr[jids];va=m.jnt_dofadr[jids]
    lo=m.jnt_range[jids,0].copy();hi=m.jnt_range[jids,1].copy()
    for name in ['left_ankle_pitch_joint','right_ankle_pitch_joint']:
        k=names.index(name);lo[k]=max(lo[k],-.751);hi[k]=min(hi[k],.414)
    upper=np.asarray([k for k,v in enumerate(names)if v.startswith(('waist_','left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist'))or a.optimize_fingers and v.startswith(('lh_','rh_'))])
    legnames=[str(v)for v in g['native_gait_leg_names']];legs=np.asarray([names.index(v)for v in legnames]);legq=g['native_gait_replacement_leg_qpos_rad']
    sq=np.r_[0,1,qa[upper]];sv=np.r_[0,1,va[upper]];times=z['time_s'];n=len(times);floor=float(z['physics_floor_z_m'])
    bodies={s:mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,s)for s in ['base_link','lh_hand_base_link','rh_hand_base_link','left_wrist_pitch_link','right_wrist_pitch_link','left_elbow_link','right_elbow_link','left_ankle_roll_link','right_ankle_roll_link']}
    footbids=[bodies[s+'_ankle_roll_link']for s in ['left','right']];floorid=mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,'assumed_floor')
    footgeoms=[[i for i in range(m.ngeom)if int(m.geom_bodyid[i])==b and m.geom_contype[i]>0]for b in footbids]
    object_q=[int(m.jnt_qposadr[mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,f'target_object_{i:02d}_free')])for i in range(1,7)]
    for gid in range(m.ngeom):
        bn=mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[gid]))or'world'
        if bn!='world'and not bn.startswith('target_object_')and m.geom_contype[gid]>0:m.geom_conaffinity[gid]|=int(m.geom_contype[gid])
    def kinematic_forward(target=d):
        mj.mj_kinematics(m,target);mj.mj_comPos(m,target);mj.mj_collision(m,target)
    if a.verify_kinematics:
        comparisons=[]
        for frame in [0,min(192,n-1),min(600,n-1)]:
            slow=mj.MjData(m);fast=mj.MjData(m)
            slow.qpos[:7]=z['qpos'][frame,:7];slow.qpos[qa]=z['qpos'][frame,7:]
            for oi,adr in enumerate(object_q):slow.qpos[adr:adr+7]=z['object_target_pose'][frame,oi]
            fast.qpos[:]=slow.qpos;mj.mj_forward(m,slow);kinematic_forward(fast)
            errors={key:float(np.max(np.abs(getattr(slow,key)-getattr(fast,key))))for key in ['xpos','xmat','geom_xpos','geom_xmat','subtree_com','cdof']}
            j1=np.zeros((3,m.nv));j2=np.zeros_like(j1);jr=np.zeros_like(j1)
            mj.mj_jacBody(m,slow,j1,jr,bodies['lh_hand_base_link']);mj.mj_jacBody(m,fast,j2,jr,bodies['lh_hand_base_link']);errors['palm_jacobian']=float(np.max(np.abs(j1-j2)))
            if slow.ncon!=fast.ncon:raise ValueError('Position-only contact count changed')
            contact_error=max([abs(float(slow.contact[i].dist-fast.contact[i].dist))for i in range(slow.ncon)]or[0.]);errors['contact_distance']=contact_error
            if max(errors.values())>1e-9:raise ValueError('Position-only geometric evaluation mismatch: '+str(errors))
            comparisons.append({'frame':frame,'contacts':int(fast.ncon),'max_abs_errors':errors})
        result={'status':'position_only_evaluation_matches_mj_forward','model_sha256':sha(a.model),'no_time_integration':True,'comparisons':comparisons}
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.with_suffix('.kinematic_evaluation_verification.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True);return
    ancestors=[]
    for bid in range(m.nbody):
        path={};cur=bid;depth=0
        while cur>0:path[cur]=depth;cur=int(m.body_parentid[cur]);depth+=1
        path[0]=depth;ancestors.append(path)
    def close(b1,b2):return min(ancestors[b1][b]+ancestors[b2][b]for b in set(ancestors[b1]).intersection(ancestors[b2]))<=3
    def clearances():return np.asarray([min(float(mj.mj_geomDistance(m,d,gid,floorid,3.,np.zeros(6)))for gid in gs)for gs in footgeoms])
    intent=g['native_gait_intent_root_position_m'][:,:2];yaw=g['native_gait_recommended_root_yaw_rad'];height=g['native_gait_recommended_root_z_m']
    handR=Rotation.from_quat(z['robot_hand_orientation_target_world_wxyz'][:,:,[1,2,3,0]].reshape(-1,4)).as_matrix().reshape(n,2,3,3)
    sites=z['robot_palm_contact_site_local'];palm_target=z['robot_palm_target_world'].copy();wrist_target=z['human_wrist_target_world'].copy();elbow_target=z['human_elbow_target_world']
    control_site_ids=[mj.mj_name2id(m,mj.mjtObj.mjOBJ_SITE,name)for name in ['lh_palm_control_site','rh_palm_control_site']]
    if min(control_site_ids)>=0 and not np.allclose(m.site_pos[control_site_ids],sites,atol=1e-6,rtol=0):
        raise ValueError('Explicit model palm control sites differ from source reference anchor; an explicit anchor conversion is required.')
    unprojected_palm=palm_target.copy();unprojected_wrist=wrist_target.copy();table_guard=np.zeros((n,2,3))
    table_guard_plane=None
    if a.table_clearance_guard:
        tableid=mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,'table_support')
        mid=int(m.geom_dataid[tableid]);adr=int(m.mesh_vertadr[mid]);num=int(m.mesh_vertnum[mid]);kinematic_forward()
        world=m.mesh_vert[adr:adr+num]@d.geom_xmat[tableid].reshape(3,3).T+d.geom_xpos[tableid]
        hull3=ConvexHull(world);top=hull3.equations[np.argmax(hull3.equations[:,2])];normal=top[:3];offset=float(top[3]);hull2=ConvexHull(world[:,:2]);xyplanes=hull2.equations
        table_guard_plane=top.tolist()
        hand_geoms=[]
        for side,prefix in enumerate(['lh_','rh_']):
            wrist_prefix=['left_wrist','right_wrist'][side]
            hand_geoms.append([gid for gid in range(m.ngeom)if m.geom_contype[gid]>0 and(mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[gid]))or'').startswith((prefix,wrist_prefix))])
        for frame in range(n):
            d.qpos[:7]=z['qpos'][frame,:7];d.qpos[qa]=z['qpos'][frame,7:];kinematic_forward()
            for side,name in enumerate(['lh_hand_base_link','rh_hand_base_link']):
                if np.max(xyplanes[:,:2]@palm_target[frame,side,:2]+xyplanes[:,2])>.06:continue
                bid=bodies[name];Rhand=d.xmat[bid].reshape(3,3);desiredR=handR[frame,side];lowest=1e9
                for gid in hand_geoms[side]:
                    bbox=m.geom_aabb[gid];Rgeom=d.geom_xmat[gid].reshape(3,3)
                    c=d.geom_xpos[gid]+Rgeom@bbox[:3]
                    relative=Rhand.T@(c-d.xpos[bid]);newR=desiredR@Rhand.T@Rgeom
                    lowest=min(lowest,float(normal@(desiredR@relative)-np.sum(np.abs(newR.T@normal)*bbox[3:])))
                origin=palm_target[frame,side]-desiredR@sites[side]
                dz=max(0.,(.003-(float(normal@origin)+offset+lowest))/normal[2]);table_guard[frame,side,2]=dz
        for side in range(2):
            dz=table_guard[:,side,2];lead=np.asarray([np.max(dz[f:min(n,f+9)])for f in range(n)])
            table_guard[:,side,2]=np.maximum(dz,gaussian_filter1d(lead,2.,mode='nearest'))
        palm_target+=table_guard;wrist_target+=table_guard
    object_vertices=[]
    for oi in range(6):
        vertices=np.load(root/f'data/fourth_sim_handoff/objects/object_{oi+1:02d}/mesh.npz',allow_pickle=False)['vertices']
        object_vertices.append(vertices[::max(1,len(vertices)//1500)])
    prev=z['qpos'][0,7:].copy();qs=[];palmfk=[];handfk=[];feetfk=[];comfk=[];comtarget=[];stance=[];clears=[];errors=[];rootheightrepair=[]
    speeds=np.asarray([1. if names[k].startswith(('lh_','rh_'))else 2. if 'waist'in names[k]else 3. if 'shoulder'in names[k]else 4. for k in upper])
    began=time.monotonic()
    for frame in range(n):
        dt=times[frame]-times[frame-1]if frame else 1.
        d.qpos[:3]=np.r_[intent[frame],height[frame]];d.qpos[3:7]=[np.cos(yaw[frame]/2),0,0,np.sin(yaw[frame]/2)]
        current=z['qpos'][frame,7:].copy();current[upper]=prev[upper];current[legs]=legq[frame];d.qpos[qa]=current
        for oi,adr in enumerate(object_q):d.qpos[adr:adr+7]=z['object_target_pose'][frame,oi]
        kinematic_forward();repair=.002-float(clearances().min());d.qpos[2]+=repair;kinematic_forward();rootheightrepair.append(repair)
        support=clearances()<.027
        if not support.any():support[np.argmin(clearances())]=True
        frame_lo=np.maximum(lo[upper],prev[upper]-speeds*dt)if frame else lo[upper]
        frame_hi=np.minimum(hi[upper],prev[upper]+speeds*dt)if frame else hi[upper]
        for iteration in range(a.iterations):
            kinematic_forward();A=[];e=[]
            for side,prefix in enumerate(['left','right']):
                hand=bodies[['lh_hand_base_link','rh_hand_base_link'][side]]
                for bid,point,target,w in [(hand,d.xpos[hand]+d.xmat[hand].reshape(3,3)@sites[side],palm_target[frame,side],1.5),
                    (bodies[prefix+'_wrist_pitch_link'],d.xpos[bodies[prefix+'_wrist_pitch_link']],wrist_target[frame,side],.5),
                    (bodies[prefix+'_elbow_link'],d.xpos[bodies[prefix+'_elbow_link']],elbow_target[frame,side],.15)]:
                    jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mj.mj_jac(m,d,jp,jr,point,bid);A.append(np.sqrt(w)*jp[:,sv]);e.append(np.sqrt(w)*(target-point))
                jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mj.mj_jacBody(m,d,jp,jr,hand)
                A.append(np.sqrt(.08)*jr[:,sv]);e.append(np.sqrt(.08)*Rotation.from_matrix(handR[frame,side]@d.xmat[hand].reshape(3,3).T).as_rotvec())
                if np.any(z['contact_candidate_observed'][frame,side]):
                    oi=int(np.argmin(np.where(z['contact_candidate_observed'][frame,side],z['source_hand_object_min_distance_m'][frame,side],np.inf)))
                    pose=z['object_target_pose'][frame,oi];normal=np.r_[intent[frame]-pose[:2],0.];normal/=max(np.linalg.norm(normal),1e-9)
                    R=Rotation.from_quat(pose[[4,5,6,3]]).as_matrix();extent=float(np.max((object_vertices[oi]@R.T)@normal))
                    wrist=bodies[prefix+'_wrist_pitch_link'];gap=float((d.xpos[wrist]-pose[:3])@normal)-extent-.055
                    if gap<0:
                        jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mj.mj_jacBody(m,d,jp,jr,wrist)
                        A.append(np.sqrt(8.)*(normal@jp)[None,sv]);e.append(np.asarray([-np.sqrt(8.)*gap]))
            support_center=d.xpos[np.asarray(footbids)[support],:2].mean(axis=0)
            jc=np.zeros((3,m.nv));mj.mj_jacSubtreeCom(m,d,jc,bodies['base_link']);js=np.zeros_like(jc)
            for fb in np.asarray(footbids)[support]:
                jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mj.mj_jacBody(m,d,jp,jr,int(fb));js+=jp/support.sum()
            cw=2.+4.*(1-float(g['native_gait_walking_blend'][frame]))
            A.append(np.sqrt(cw)*(jc-js)[:2,sv]);e.append(np.sqrt(cw)*(support_center-d.subtree_com[bodies['base_link'],:2]))
            for ci in range(d.ncon):
                c=d.contact[ci]
                if c.dist>=.002:continue
                b1,b2=int(m.geom_bodyid[c.geom1]),int(m.geom_bodyid[c.geom2]);bn=[mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,b)or'world'for b in [b1,b2]]
                robot=[b>0 and not name.startswith('target_object_')for b,name in zip([b1,b2],bn)]
                if not any(robot):continue
                if all(robot)and(close(b1,b2)or not any(name.startswith(('lh_','rh_','left_wrist','right_wrist'))for name in bn)):continue
                j1=np.zeros((3,m.nv));j2=np.zeros_like(j1);jr=np.zeros_like(j1)
                if robot[0]:mj.mj_jac(m,d,j1,jr,c.pos,b1)
                if robot[1]:mj.mj_jac(m,d,j2,jr,c.pos,b2)
                normal=np.asarray(c.frame).reshape(3,3)[0];A.append(np.sqrt(12.)*(normal@(j2-j1))[None,sv]);e.append(np.asarray([np.sqrt(12.)*(.002-float(c.dist))]))
            reg=np.zeros((2,len(sv)));reg[:,:2]=.8*np.eye(2);A.append(reg);e.append(.8*(intent[frame]-d.qpos[:2]))
            if a.optimize_fingers:
                finger_columns=[kk for kk,k in enumerate(upper)if names[k].startswith(('lh_','rh_'))]
                if finger_columns:
                    Jf=np.zeros((len(finger_columns),len(sv)));ef=[]
                    for row,kk in enumerate(finger_columns):
                        k=upper[kk];Jf[row,2+kk]=.15;ef.append(.15*(z['qpos'][frame,7+k]-d.qpos[qa[k]]))
                    A.append(Jf);e.append(np.asarray(ef))
            J=np.vstack(A);err=np.concatenate(e);step=np.linalg.solve(J.T@J+.025*np.eye(len(sv)),J.T@err)
            low_xy=intent[frame]-.10;high_xy=intent[frame]+.10
            if frame:
                low_xy=np.maximum(low_xy,qs[-1][:2]-.8*dt);high_xy=np.minimum(high_xy,qs[-1][:2]+.8*dt);low_xy=np.minimum(low_xy,high_xy)
            d.qpos[sq]=np.clip(d.qpos[sq]+np.clip(step,-.10,.10),np.r_[low_xy,frame_lo],np.r_[high_xy,frame_hi])
            if np.max(np.abs(err))<.005:break
        kinematic_forward();prev=d.qpos[qa].copy();qs.append(np.r_[d.qpos[:7],prev])
        hf=np.asarray([d.xpos[bodies[name]].copy()for name in ['lh_hand_base_link','rh_hand_base_link']]);pf=np.asarray([hf[s]+d.xmat[bodies[name]].reshape(3,3)@sites[s]for s,name in enumerate(['lh_hand_base_link','rh_hand_base_link'])])
        handfk.append(hf);palmfk.append(pf);feetfk.append(d.xpos[footbids].copy());comfk.append(d.subtree_com[bodies['base_link']].copy());comtarget.append(d.xpos[np.asarray(footbids)[support],:2].mean(axis=0));stance.append(support);clears.append(clearances());errors.append(np.linalg.norm(pf-palm_target[frame],axis=1))
        if frame%150==0:
            print(json.dumps({'stage':'native_leg_upperbody_ik','frame':frame,'frames':n,'wall_s':time.monotonic()-began}),flush=True)
            if frame>0:
                prefix=a.output.with_suffix('.prefix.npz');partial=prefix.with_suffix('.partial')
                with partial.open('wb')as stream:
                    np.savez_compressed(stream,qpos_prefix=np.asarray(qs),source_frame_ids_prefix=np.arange(frame+1),
                        palm_fk_prefix=np.asarray(palmfk),foot_fk_prefix=np.asarray(feetfk),com_fk_prefix=np.asarray(comfk),
                        expected_frames=np.asarray(n),model_sha256=np.asarray(sha(a.model)),complete=np.asarray(False))
                partial.replace(prefix)
    qs=np.asarray(qs);payload={k:z[k]for k in z}
    for k in list(payload):
        if '_fk_'in k or k in ['ik_error_m','foot_sole_clearance_m','robot_stance_foot_speed_m_s','robot_foot_target_world','robot_com_target_world_xy_m']or k.startswith('robot_foot_')and'adaptation'in k:payload['parent_cached_'+k]=payload.pop(k)
    for k in g:
        if k.startswith('native_gait_'):payload[k]=g[k]
    payload.update(qpos=qs,robot_hand_fk_world=np.asarray(handfk),robot_palm_fk_world=np.asarray(palmfk),robot_foot_fk_world=np.asarray(feetfk),robot_com_fk_world_m=np.asarray(comfk),robot_com_target_world_xy_m=np.asarray(comtarget),robot_foot_target_world=np.asarray(feetfk),foot_sole_clearance_m=np.asarray(clears),planned_stance_mask=np.asarray(stance),native_gait_root_height_clearance_repair_m=np.asarray(rootheightrepair),root_deviation_from_raw_source_xyz_m=qs[:,:3]-z['raw_source_root_position_m'],root_alignment_offset_xyz_m=qs[:,:3]-g['native_gait_intent_root_position_m'],native_gait_physical_ready=np.asarray(False),missing_human_fill_method=np.asarray('Last observed human anatomical targets held; robot legs smoothly settle through native standing blend. These last85 frames are imputed, not observed human motion.'),native_gait_fusion_note=np.asarray('12 template legs held exactly; root uses original-source intent XY plus <=10cm geometric correction, velocity-heading and native height; 16 waist/arm DOFs refit to source wrist/palm targets. Physical tracking/path fidelity not established.'))
    payload['robot_stance_foot_speed_m_s']=np.linalg.norm(np.gradient(np.asarray(feetfk)[:,:,:2],times,axis=0),axis=2)*np.asarray(stance)
    payload['parent_unprojected_robot_palm_target_world']=unprojected_palm
    payload['robot_palm_target_world']=palm_target
    payload['robot_hand_target_world']=palm_target-np.einsum('nsij,sj->nsi',handR,sites)
    payload['robot_wrist_target_world']=wrist_target
    payload['robot_table_clearance_target_projection_xyz_m']=table_guard
    payload['robot_table_clearance_projection_is_synthetic']=np.asarray(a.table_clearance_guard)
    payload['robot_finger_collision_adaptation_rad']=qs[:,7+np.asarray([i for i,name in enumerate(names)if name.startswith(('lh_','rh_'))])]-z['qpos'][:,7+np.asarray([i for i,name in enumerate(names)if name.startswith(('lh_','rh_'))])]
    for key in ['collision_model_path','collision_model_sha256']:
        if key in payload:payload['parent_'+key]=payload[key]
    payload['collision_model_path']=np.asarray(str(a.model));payload['collision_model_sha256']=np.asarray(sha(a.model))
    payload['robot_finger_collision_ik_enabled']=np.asarray(a.optimize_fingers)
    payload['robot_explicit_palm_control_sites_verified']=np.asarray(min(control_site_ids)>=0)
    a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,**payload)
    report={'schema':'fourth_native_gait_upperbody_reference.v1','output':str(a.output),'sha256':sha(a.output),'frames':n,'physical_success_claimed':False,'model':str(a.model),'model_sha256':sha(a.model),'source_reference':str(a.reference),'source_reference_sha256':sha(a.reference),'leg_plan':str(a.leg_plan),'leg_plan_sha256':sha(a.leg_plan),'native_legs_preserved_exactly':bool(np.array_equal(qs[:,7+legs],legq)),'root_height_above_floor_range_m':[float(np.min(qs[:,2]-floor)),float(np.max(qs[:,2]-floor))],'native_height_clearance_repair_max_abs_m':float(np.max(np.abs(rootheightrepair))),'source_root_xy_error_max_m':float(np.nanmax(np.linalg.norm(qs[:,:2]-z['raw_source_root_position_m'][:,:2],axis=1))),'palm_target_error_median_m':np.median(errors,axis=0).tolist(),'palm_target_error_max_m':np.max(errors,axis=0).tolist(),'foot_clearance_min_m':np.min(clears,axis=0).tolist(),'source_script_sha256':sha(Path(__file__)),'limitations':['Source root direction/velocity differ from the validated native cycle; heading/rate adaptation is explicit and requires new policy validation.','The source-derived table is a measured-point/support-prior collision completion, not uniquely measured full boundaries.','No physics simulation performed by this generator.']}
    report['table_clearance_guard_enabled']=a.table_clearance_guard;report['table_guard_plane']=table_guard_plane
    report['table_clearance_target_projection_max_m']=float(np.max(np.linalg.norm(table_guard,axis=2)))
    report['finger_collision_ik_enabled']=a.optimize_fingers
    a.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
