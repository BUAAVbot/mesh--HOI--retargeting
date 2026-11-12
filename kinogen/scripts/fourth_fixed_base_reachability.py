"""Private geometric fixed-stand reachability analysis; no physics integration."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--stand-reference',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--inspect',action='store_true')
    parser.add_argument('--x-values',nargs='+',type=float,default=[-.046,-.15,-.10,-.20])
    parser.add_argument('--y-values',nargs='+',type=float,default=[1.9,2.05,2.0])
    parser.add_argument('--yaw-offsets',nargs='+',type=float,default=[0.,-.25,.25])
    parser.add_argument('--iterations',type=int,default=70)
    parser.add_argument('--waist-yaw-limit',type=float,default=.6)
    parser.add_argument('--target-mode',choices=['near-rim','source-palms'],default='near-rim')
    parser.add_argument('--stack-event',choices=['last-observed','first-final-arrival'],default='last-observed')
    parser.add_argument('--object-ids',nargs='+',type=int,default=[1,2,3,4,5,6])
    args=parser.parse_args()
    if os.name!='posix':parser.error('Server-only private FK analysis')
    import numpy as np
    import mujoco as mj
    from scipy.spatial import ConvexHull
    from scipy.spatial.transform import Rotation
    with np.load(args.reference,allow_pickle=False)as archive:
        source={key:archive[key]for key in archive.files}
    with np.load(args.stand_reference,allow_pickle=False)as archive:
        stand={key:archive[key]for key in archive.files}
    model=mj.MjModel.from_xml_path(str(args.model))
    data=mj.MjData(model)
    names=[str(v)for v in source['dof_names']]
    joints=np.asarray([mj.mj_name2id(model,mj.mjtObj.mjOBJ_JOINT,name)for name in names])
    qadr=model.jnt_qposadr[joints]
    vadr=model.jnt_dofadr[joints]
    stand_names=[str(v)for v in stand['dof_names']]
    stand_q=np.asarray([stand['qpos'][0,7+stand_names.index(name)]for name in names])
    original_stand_q=stand_q.copy();couplings=[]
    for equality in range(model.neq):
        if model.eq_type[equality]!=mj.mjtEq.mjEQ_JOINT:continue
        dependent,master=int(model.eq_obj1id[equality]),int(model.eq_obj2id[equality])
        if dependent not in joints or master not in joints:continue
        di=int(np.flatnonzero(joints==dependent)[0]);mi=int(np.flatnonzero(joints==master)[0])
        coefficients=model.eq_data[equality,:5]
        master_value=stand_q[mi]-model.qpos0[qadr[mi]]
        stand_q[di]=model.qpos0[qadr[di]]+sum(float(c)*master_value**power for power,c in enumerate(coefficients))
        couplings.append({'dependent':names[di],'master':names[mi],'polynomial':coefficients.tolist()})
    object_qadr=[int(model.jnt_qposadr[mj.mj_name2id(model,mj.mjtObj.mjOBJ_JOINT,f'target_object_{i:02d}_free')])for i in range(1,7)]
    object_bids=[mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,f'target_object_{i:02d}')for i in range(1,7)]
    data.qpos[qadr]=stand_q
    data.qpos[:7]=stand['qpos'][0,:7]
    for oi,adr in enumerate(object_qadr):data.qpos[adr:adr+7]=source['object_target_pose'][0,oi]
    mj.mj_kinematics(model,data);mj.mj_comPos(model,data)
    info=[]
    for oi in range(6):
        valid=np.flatnonzero(source['object_observed_mask'][:,oi])
        sides=[]
        for side in range(2):
            frames=np.flatnonzero(source['contact_candidate_observed'][:,side,oi])
            sides.append({'side':['left','right'][side],'frames':len(frames),
                'first_frame':int(frames[0])if len(frames)else None,'last_frame':int(frames[-1])if len(frames)else None,
                'first_palm':source['human_hand_target_world'][frames[0],side].tolist()if len(frames)else None,
                'last_palm':source['human_hand_target_world'][frames[-1],side].tolist()if len(frames)else None})
        geoms=np.flatnonzero((model.geom_bodyid==object_bids[oi])&(model.geom_contype>0))
        vertices=[]
        for geom in geoms:
            mesh=int(model.geom_dataid[geom])
            if model.geom_type[geom]!=mj.mjtGeom.mjGEOM_MESH:continue
            adr=int(model.mesh_vertadr[mesh]);count=int(model.mesh_vertnum[mesh])
            world=model.mesh_vert[adr:adr+count]@data.geom_xmat[geom].reshape(3,3).T+data.geom_xpos[geom]
            vertices.append((world-data.xpos[object_bids[oi]])@data.xmat[object_bids[oi]].reshape(3,3))
        vertices=np.concatenate(vertices)
        final=source['object_target_pose'][valid[-1],oi]
        near=source['object_observed_mask'][:,oi]&(np.linalg.norm(source['object_target_pose'][:,oi,:2]-final[:2],axis=1)<.04)&(np.abs(source['object_target_pose'][:,oi,2]-final[2])<.03)
        groups=np.flatnonzero(np.convolve(near.astype(int),np.ones(5,dtype=int),'valid')==5)
        arrival=int(groups[0])if len(groups)else int(np.flatnonzero(near)[0])
        info.append({'object_id':oi+1,'first_observed_frame':int(valid[0]),'last_observed_frame':int(valid[-1]),'first_final_arrival_frame':arrival,
            'initial_pose_wxyz':source['object_target_pose'][0,oi].tolist(),
            'final_observed_pose_wxyz':source['object_target_pose'][valid[-1],oi].tolist(),
            'proxy_local_min':vertices.min(axis=0).tolist(),'proxy_local_max':vertices.max(axis=0).tolist(),'source_hands':sides})
    report={'schema':'fourth_fixed_base_reachability.inspect.v1','model':str(args.model),'model_sha256':sha(args.model),
        'stand_root':stand['qpos'][0,:7].tolist(),'source_objects':info,'dynamics_run':False,
        'static_finger_mimic_expansion':couplings,'static_finger_expansion_max_delta_rad':float(np.max(np.abs(stand_q-original_stand_q)))}
    shoulder=mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,'left_shoulder_pitch_link')
    jp=np.zeros((3,model.nv));jr=np.zeros_like(jp);mj.mj_jacBody(model,data,jp,jr,shoulder)
    base=mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,'base_link');forward_axis=data.xmat[base].reshape(3,3)[:,0]
    report['waist_contract']={name:{'joint_local_axis':model.jnt_axis[joints[k]].tolist(),'urdf_joint_range_rad':model.jnt_range[joints[k]].tolist(),
        'shoulder_forward_derivative_m_per_rad':float(forward_axis@jp[:,vadr[k]])}
        for k,name in enumerate(names)if name.startswith('waist_')}
    closure_sign=[]
    for side,prefix in enumerate(['lh_','rh_']):
        index=next(k for k,name in enumerate(names)if name.startswith(prefix)and'index'in name and('mcp'in name or'proximal'in name))
        hand=mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,prefix+'hand_base_link')
        frame=data.xmat[hand].reshape(3,3)@source['robot_palm_anatomical_frame_in_hand_base'][side]
        value=float(frame[:,2]@np.cross(data.xaxis[joints[index]],frame[:,0]))
        if abs(value)<.1:raise ValueError('Cannot determine palm normal from positive MCP flexion')
        closure_sign.append(float(np.sign(value)))
    report['anatomical_palm_normal_positive_mcp_curl_sign']=closure_sign
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.inspect:
        args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True);return
    upper=np.asarray([i for i,name in enumerate(names)if name.startswith(('waist_','left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist'))])
    if len(upper)!=16:raise ValueError('Expected two waist and fourteen arm joints')
    lo=model.jnt_range[joints[upper],0].copy();hi=model.jnt_range[joints[upper],1].copy()
    for col,index in enumerate(upper):
        if names[index].startswith('waist_'):
            limit=(-args.waist_yaw_limit,args.waist_yaw_limit)if'yaw'in names[index]else(-.15,.45)
            lo[col]=max(lo[col],limit[0]);hi[col]=min(hi[col],limit[1])
    palm_bids=[mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,name)for name in ['lh_hand_base_link','rh_hand_base_link']]
    wrist_bids=[mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,name)for name in ['left_wrist_pitch_link','right_wrist_pitch_link']]
    foot_bids=[mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,name+'_ankle_roll_link')for name in ['left','right']]
    base_bid=mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,'base_link')
    table_gid=mj.mj_name2id(model,mj.mjtObj.mjOBJ_GEOM,'table_support')
    floor_gid=mj.mj_name2id(model,mj.mjtObj.mjOBJ_GEOM,'assumed_floor')
    sites=source['robot_palm_contact_site_local']
    ancestors=[]
    body_names=[mj.mj_id2name(model,mj.mjtObj.mjOBJ_BODY,i)or'world'for i in range(model.nbody)]
    for bid in range(model.nbody):
        path={};cur=bid;depth=0
        while cur>0:path[cur]=depth;cur=int(model.body_parentid[cur]);depth+=1
        path[0]=depth;ancestors.append(path)
    def close(b1,b2):
        return min(ancestors[b1][b]+ancestors[b2][b]for b in set(ancestors[b1]).intersection(ancestors[b2]))<=3
    def hand_side(body):
        name=body_names[body]
        return 0 if name.startswith('lh_')else 1 if name.startswith('rh_')else -1
    for gid in range(model.ngeom):
        body=int(model.geom_bodyid[gid]);name=body_names[body]
        if body>0 and not name.startswith('target_object_')and model.geom_contype[gid]>0:
            model.geom_conaffinity[gid]|=int(model.geom_contype[gid])
    def forward():
        mj.mj_kinematics(model,data);mj.mj_comPos(model,data);mj.mj_collision(model,data)
    def contacts():
        result=[]
        for ci in range(data.ncon):
            contact=data.contact[ci]
            b1,b2=int(model.geom_bodyid[contact.geom1]),int(model.geom_bodyid[contact.geom2])
            bn=[body_names[b]for b in [b1,b2]]
            robot=[b>0 and not name.startswith('target_object_')for b,name in zip([b1,b2],bn)]
            if not any(robot):continue
            if all(robot):
                if close(b1,b2)or not any(name.startswith(('lh_','rh_','left_wrist','right_wrist'))for name in bn):continue
                if hand_side(b1)>=0 and hand_side(b1)==hand_side(b2):continue
            result.append((contact,b1,b2,robot))
        return result
    signs=np.asarray([[x,y,z]for x in [-1,1]for y in [-1,1]for z in [-1,1]])
    def geom_corners(gid):
        typ=model.geom_type[gid];size=model.geom_size[gid]
        if typ==mj.mjtGeom.mjGEOM_MESH:
            mesh=int(model.geom_dataid[gid]);adr=int(model.mesh_vertadr[mesh]);count=int(model.mesh_vertnum[mesh])
            local=model.mesh_vert[adr:adr+count]
            lower,upper_box=local.min(axis=0),local.max(axis=0)
            local=(lower+upper_box)/2+signs*(upper_box-lower)/2
        else:
            if typ==mj.mjtGeom.mjGEOM_SPHERE:extent=np.repeat(size[0],3)
            elif typ==mj.mjtGeom.mjGEOM_CAPSULE:extent=np.asarray([size[0],size[0],size[0]+size[1]])
            elif typ==mj.mjtGeom.mjGEOM_CYLINDER:extent=np.asarray([size[0],size[0],size[1]])
            else:extent=size[:3]
            local=signs*extent
        return local@data.geom_xmat[gid].reshape(3,3).T+data.geom_xpos[gid]
    def support_polygon():
        vertices=[];clearances=[]
        for bid in foot_bids:
            geoms=np.flatnonzero((model.geom_bodyid==bid)&(model.geom_contype>0))
            foot_clear=[]
            for gid in geoms:
                clearance=float(mj.mj_geomDistance(model,data,int(gid),floor_gid,3.,np.zeros(6)))
                foot_clear.append(clearance)
                if clearance<.025:vertices.append(geom_corners(int(gid))[:,:2])
            clearances.append(min(foot_clear))
        return ConvexHull(np.concatenate(vertices)).equations,np.asarray(clearances)
    tasks=[]
    for row in info:
        if row['object_id']not in args.object_ids:continue
        oi=row['object_id']-1;preferred=0 if row['source_hands'][0]['frames']else 1
        for phase,frame in [('pickup',0),('stack',row['last_observed_frame'])]:
            if args.target_mode=='source-palms':frame=row['source_hands'][preferred]['first_frame'if phase=='pickup'else'last_frame']
            elif phase=='stack'and args.stack_event=='first-final-arrival':frame=row['first_final_arrival_frame']
            tasks.append({'id':f'{phase}_{oi+1:02d}','phase':phase,'object_index':oi,'frame':frame,'side':preferred})
    def task_goal(task,xy):
        if args.target_mode=='source-palms':
            frame=task['frame'];side=task['side'];goal=source['human_hand_target_world'][frame,side].copy()
            desired=Rotation.from_quat(source['robot_hand_orientation_target_world_wxyz'][frame,side,[1,2,3,0]]).as_matrix()
            return goal,np.repeat(goal[2],2),desired
        oi=task['object_index'];pose=source['object_target_pose'][task['frame'],oi]
        rotation=Rotation.from_quat(pose[[4,5,6,3]]).as_matrix()
        radial=np.r_[xy-pose[:2],0.];radial/=max(np.linalg.norm(radial),1e-9)
        local=rotation.T@radial;local[2]=0.;local/=max(np.linalg.norm(local),1e-9)
        row=info[oi];radius=max(abs(float(v))for v in row['proxy_local_min'][:2]+row['proxy_local_max'][:2])
        rim=pose[:3]+rotation@np.r_[radius*local[:2],row['proxy_local_max'][2]]
        goal=rim+.035*radial
        band=np.asarray([rim[2]+.03,rim[2]+.11]);goal[2]=band.mean()
        finger=np.asarray([0.,0.,-1.]);normal=-radial*closure_sign[task['side']];lateral=np.cross(normal,finger)
        desired=np.column_stack([finger,lateral,normal])@source['robot_palm_anatomical_frame_in_hand_base'][task['side']].T
        return goal,band,desired
    def solve(task,root,seed):
        frame=task['frame'];side=task['side'];hand=palm_bids[side]
        data.qpos[:]=model.qpos0;data.qpos[:7]=root;data.qpos[qadr]=stand_q
        data.qpos[qadr[upper]]=np.clip(seed,lo,hi)
        for oi,adr in enumerate(object_qadr):data.qpos[adr:adr+7]=source['object_target_pose'][frame,oi]
        forward();equations,foot_clear=support_polygon();goal,band,desired=task_goal(task,root[:2])
        for iteration in range(args.iterations):
            forward();point=data.xpos[hand]+data.xmat[hand].reshape(3,3)@sites[side]
            target=goal.copy();target[2]=np.clip(point[2],band[0],band[1])
            jp=np.zeros((3,model.nv));jr=np.zeros_like(jp);mj.mj_jac(model,data,jp,jr,point,hand)
            rows=[jp[:,vadr[upper]]/.02];errors=[(target-point)/.02]
            orientation=Rotation.from_matrix(desired@data.xmat[hand].reshape(3,3).T).as_rotvec()
            rows.append(jr[:,vadr[upper]]*.7);errors.append(orientation*.7)
            com=data.subtree_com[base_bid,:2];outside=equations[:,:2]@com+equations[:,2]+.005
            jc=np.zeros((3,model.nv));mj.mj_jacSubtreeCom(model,data,jc,base_bid)
            for equation,violation in zip(equations,outside):
                if violation>0:
                    rows.append((equation[:2]@jc[:2,vadr[upper]])[None,:]/.005)
                    errors.append(np.asarray([-violation/.005]))
            for contact,b1,b2,robot in contacts():
                if contact.dist>=.002:continue
                j1=np.zeros((3,model.nv));j2=np.zeros_like(j1);jr=np.zeros_like(j1)
                if robot[0]:mj.mj_jac(model,data,j1,jr,contact.pos,b1)
                if robot[1]:mj.mj_jac(model,data,j2,jr,contact.pos,b2)
                row=np.asarray(contact.frame).reshape(3,3)[0]@(j2-j1)
                rows.append(row[None,vadr[upper]]/.005);errors.append(np.asarray([(.002-float(contact.dist))/.005]))
            rows.append(.03*np.eye(len(upper)));errors.append(.03*(stand_q[upper]-data.qpos[qadr[upper]]))
            jac=np.vstack(rows);error=np.concatenate(errors)
            delta=np.linalg.solve(jac.T@jac+.01*np.eye(len(upper)),jac.T@error)
            delta*=min(1.,.18/max(np.max(np.abs(delta)),1e-12))
            data.qpos[qadr[upper]]=np.clip(data.qpos[qadr[upper]]+delta,lo,hi)
            if iteration>12 and np.max(np.abs(delta))<1e-5:break
        forward();point=data.xpos[hand]+data.xmat[hand].reshape(3,3)@sites[side]
        target=goal.copy();target[2]=np.clip(point[2],band[0],band[1]);error=float(np.linalg.norm(point-target))
        pairs=[];depth=0.;core_table=0.;hand_table=0.
        for contact,b1,b2,robot in contacts():
            penetration=max(0.,-float(contact.dist));depth=max(depth,penetration)
            if table_gid in [contact.geom1,contact.geom2]:
                if any(body_names[b].startswith(('lh_','rh_','left_wrist','right_wrist'))for b in [b1,b2]):hand_table=max(hand_table,penetration)
                else:core_table=max(core_table,penetration)
            if penetration>.001:pairs.append({'bodies':[body_names[b1],body_names[b2]],'penetration_m':penetration})
        outside=float(max(0.,np.max(equations[:,:2]@data.subtree_com[base_bid,:2]+equations[:,2]+.005)))
        orientation_error=float(np.linalg.norm(Rotation.from_matrix(desired@data.xmat[hand].reshape(3,3).T).as_rotvec()))
        result=dict(task,side=['left','right'][side],target_palm_world=goal.tolist(),height_band_m=band.tolist(),palm_world=point.tolist(),
            position_error_m=error,orientation_error_rad=orientation_error,com_support_violation_m=outside,
            max_actionable_penetration_m=depth,max_core_table_penetration_m=core_table,max_hand_table_penetration_m=hand_table,
            foot_floor_clearance_m=foot_clear.tolist(),bad_pairs=sorted(pairs,key=lambda p:-p['penetration_m'])[:4],
            reachable_safe=bool(error<=.02 and outside<=.001 and depth<=.001 and foot_clear.min()>=-.001),
            other_objects_observed=source['object_observed_mask'][frame].tolist())
        return result,np.r_[data.qpos[:7],data.qpos[qadr]].copy()
    candidates=[];best_poses=None;best_score=None;began=time.monotonic()
    floor=float(source['physics_floor_z_m']);height=.8743374236734526
    report.update(schema='fourth_fixed_base_reachability.v1',stand_reference=str(args.stand_reference),stand_reference_sha256=sha(args.stand_reference),
        source_script_sha256=sha(__file__),object_ids=args.object_ids,
        base_height_m=height,root_is_fixed_in_each_candidate=True,legs_are_native_default_and_fixed=True,
        torso_bounds_rad={names[k]:[float(lo[col]),float(hi[col])]for col,k in enumerate(upper)if names[k].startswith('waist_')},
        task_semantics=('Exact reconstructed source palm positions at first/last source observed contact-candidate frames; these source estimates are not ground-truth contacts.'if args.target_mode=='source-palms'else'Six initial pickups and six '+args.stack_event+' stack poses; near-side rim radial offset35mm and palm height rim+30..110mm. Permissive open-hand reach envelope; no force closure or precise rim orientation proof.'),
        target_mode=args.target_mode,stack_event=args.stack_event,
        support_semantics='COM inside outer hull of native active foot geometry bounds within25mm floor, with5mm inward margin; no ZMP/force proof.',
        collision_semantics='Actual segmented hand/table/object/core contacts; native structural and neighboring robot proxy pairs filtered. Source object poses stay unchanged; other-object imputation labelled.',
        grid={'x':args.x_values,'y':args.y_values,'yaw_offset_from_minus_pi':args.yaw_offsets},candidates=candidates)
    for x in args.x_values:
        for y in args.y_values:
            for offset in args.yaw_offsets:
                yaw=-np.pi+offset;root=np.asarray([x,y,floor+height,np.cos(yaw/2),0.,0.,np.sin(yaw/2)])
                rows=[];poses=[]
                for task in tasks:
                    seeds=[stand_q[upper],source['qpos'][task['frame'],7+upper]];choice=None;choice_pose=None
                    for seed in seeds:
                        result,pose=solve(task,root,seed)
                        score=(not result['reachable_safe'],result['position_error_m']+10*result['max_actionable_penetration_m']+5*result['com_support_violation_m'])
                        if choice is None or score<choice[0]:choice=(score,result);choice_pose=pose
                        if result['reachable_safe']:break
                    rows.append(choice[1]);poses.append(choice_pose)
                passed=sum(row['reachable_safe']for row in rows)
                candidate={'root_xy_yaw':[x,y,float(yaw)],'passed':passed,'total':len(tasks),
                    'worst_position_error_m':max(row['position_error_m']for row in rows),'tasks':rows}
                candidates.append(candidate)
                score=(-passed,sum(row['position_error_m']+10*row['max_actionable_penetration_m']+5*row['com_support_violation_m']for row in rows))
                if best_score is None or score<best_score:best_score=score;report['best']=candidate;best_poses=np.asarray(poses)
                args.output.write_text(json.dumps(report,indent=2)+'\n')
                np.savez_compressed(args.output.with_suffix('.npz'),qpos=best_poses,dof_names=np.asarray(names),task_ids=np.asarray([task['id']for task in tasks]),
                    root_xy_yaw=np.asarray(report['best']['root_xy_yaw']),physical_ready=np.asarray(False))
                print(json.dumps({'candidate':len(candidates),'root_xy_yaw':candidate['root_xy_yaw'],'passed':passed,'total':len(tasks),'worst_error_m':candidate['worst_position_error_m'],'wall_s':time.monotonic()-began}),flush=True)
    report['wall_s']=time.monotonic()-began;report['physical_success_claimed']=False
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'best_root':report['best']['root_xy_yaw'],'passed':report['best']['passed'],'total':len(tasks),'output':str(args.output),'wall_s':report['wall_s']}),flush=True)


if __name__=='__main__':main()
