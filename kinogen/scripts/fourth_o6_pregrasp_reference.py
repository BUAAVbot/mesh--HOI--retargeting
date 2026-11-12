"""Remap a validated O6 object-local pregrasp to raw initial object poses.

The 923-frame nominal hold is a first-lift control seed, not six-object motion
reproduction. Native legs and waist stay fixed; no physics is integrated here.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--stand-reference',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if os.name!='posix':parser.error('Server-only reference preparation')
    if args.output.exists():raise FileExistsError(args.output)
    import numpy as np
    import mujoco as mj
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    from fourth_arm_object_feedback import ArmObjectFeedback
    from fourth_convex_separation import ConvexSeparationOracle
    source=dict(np.load(args.reference,allow_pickle=False));stand=dict(np.load(args.stand_reference,allow_pickle=False))
    receipt=json.loads(args.candidate.read_text());candidate=receipt['best'];template=candidate['template']
    if sha(args.model)!=receipt['model_sha256']:raise ValueError('Candidate/model SHA mismatch')
    model=mj.MjModel.from_xml_path(str(args.model.resolve()));data=mj.MjData(model)
    helper=ArmObjectFeedback(model,source);oracle=ConvexSeparationOracle(model)
    names=list(map(str,source['dof_names']));joints=np.asarray([mj.mj_name2id(model,mj.mjtObj.mjOBJ_JOINT,name)for name in names]);qa=model.jnt_qposadr[joints]
    stand_names=list(map(str,stand['dof_names']));stand_q=np.asarray([stand['qpos'][0,7+stand_names.index(name)]for name in names])
    original_candidate=np.asarray(candidate['qpos'],dtype=float);data.qpos[:]=original_candidate
    legs=np.asarray([i for i,name in enumerate(names)if name.startswith(('left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle'))])
    waist=np.asarray([i for i,name in enumerate(names)if name.startswith('waist_')])
    if len(legs)!=12 or len(waist)!=2:raise ValueError('Unexpected Vita standing ABI')
    data.qpos[qa[legs]]=stand_q[legs];data.qpos[qa[waist]]=0.
    relations=[]
    for ei in range(model.neq):
        if model.eq_type[ei]!=mj.mjtEq.mjEQ_JOINT:continue
        slave,master=int(model.eq_obj1id[ei]),int(model.eq_obj2id[ei]);coeff=model.eq_data[ei,:5].copy()
        if slave not in joints or master not in joints:continue
        relations.append((slave,master,coeff))
    if len(relations)!=10:raise ValueError('Expected ten O6 couplings')
    def project():
        for slave,master,coeff in relations:
            value=data.qpos[model.jnt_qposadr[master]]-model.qpos0[model.jnt_qposadr[master]]
            data.qpos[model.jnt_qposadr[slave]]=model.qpos0[model.jnt_qposadr[slave]]+sum(float(c)*value**power for power,c in enumerate(coeff))
    project()
    for oi,adr in enumerate(helper.object_qadr):data.qpos[adr:adr+7]=source['object_target_pose'][0,oi]
    pose=source['object_target_pose'][0,0];object_R=Rotation.from_quat(pose[[4,5,6,3]]).as_matrix()
    target_palm=pose[:3]+object_R@np.asarray(template['palm_offset_in_object_m'])
    target_R=object_R@Rotation.from_quat(np.asarray(template['hand_rotation_in_object_wxyz'])[[1,2,3,0]]).as_matrix()
    chain=helper.chains[1];armqa=chain['qa'];start=data.qpos[armqa].copy()
    def forward():mj.mj_kinematics(model,data);mj.mj_comPos(model,data)
    def residual(q):
        data.qpos[armqa]=q;project();forward()
        orientation=Rotation.from_matrix(data.xmat[helper.hand_bids[1]].reshape(3,3)@target_R.T).as_rotvec()
        return np.r_[(helper.palm(data,1)-target_palm)/.001,orientation/.01,1e-5*(q-start)]
    fit=least_squares(residual,start,bounds=(chain['lower']+1e-6,chain['upper']-1e-6),max_nfev=150,ftol=1e-12,xtol=1e-12,gtol=1e-12)
    residual(fit.x);palm_error=float(np.linalg.norm(helper.palm(data,1)-target_palm))
    rotation_error=float(np.linalg.norm(Rotation.from_matrix(data.xmat[helper.hand_bids[1]].reshape(3,3)@target_R.T).as_rotvec()))
    if palm_error>1e-7 or rotation_error>1e-6:raise ValueError(f'Raw-object remap IK inaccurate: {palm_error},{rotation_error}')
    numeric=mj.mj_name2id(model,mj.mjtObj.mjOBJ_NUMERIC,'source_table_plane_abcd');adr=int(model.numeric_adr[numeric]);plane=model.numeric_data[adr:adr+4].copy();plane/=np.linalg.norm(plane[:3])
    def plane_clearance(gid):
        rotation=data.geom_xmat[gid].reshape(3,3);size=model.geom_size[gid];typ=model.geom_type[gid]
        if typ==mj.mjtGeom.mjGEOM_MESH:
            points=oracle._shape(gid)['vertices']@rotation.T+data.geom_xpos[gid]
            return float(np.min(points@plane[:3]+plane[3]))
        centre=float(data.geom_xpos[gid]@plane[:3]+plane[3]);projection=float(rotation[:,2]@plane[:3])
        if typ==mj.mjtGeom.mjGEOM_CAPSULE:return centre-size[0]-size[1]*abs(projection)
        if typ==mj.mjtGeom.mjGEOM_CYLINDER:return centre-size[1]*abs(projection)-size[0]*np.sqrt(max(0.,1-projection**2))
        if typ==mj.mjtGeom.mjGEOM_SPHERE:return centre-size[0]
        if typ==mj.mjtGeom.mjGEOM_BOX:return centre-float(np.abs(rotation.T@plane[:3])@size)
        raise ValueError('Unsupported hand primitive')
    clearances=[{'geom':model.geom(gid).name,'plane_gap_m':plane_clearance(gid)}for gid in helper.hand_geoms[1]]
    def bounds(gid):
        rotation=data.geom_xmat[gid].reshape(3,3);centre=data.geom_xpos[gid]+rotation@model.geom_aabb[gid,:3];extent=np.abs(rotation)@model.geom_aabb[gid,3:]
        return centre-extent,centre+extent
    gaps=[]
    for oi,object_geoms in enumerate(helper.object_geoms):
        minimum=(np.inf,None,None,None)
        for hand in helper.hand_geoms[1]:
            lower,upper=bounds(hand)
            for obj in object_geoms:
                ol,oh=bounds(obj);lower_bound=float(np.linalg.norm(np.maximum(np.maximum(lower-oh,ol-upper),0.)))
                if lower_bound>.01:gap=lower_bound;method='AABB conservative lower bound'
                elif model.geom_type[hand]==mj.mjtGeom.mjGEOM_MESH:gap=float(oracle.query(data,hand,obj)[0]);method='convex SAT conservative signed gap'
                else:
                    gap=float(mj.mj_geomDistance(model,data,hand,obj,2.,np.zeros(6)));method='native primitive distance'
                if gap<minimum[0]:minimum=(gap,model.geom(hand).name,model.geom(obj).name,method)
        gaps.append({'object_id':oi+1,'minimum_hand_clearance_lower_bound_m':minimum[0],'hand_geom':minimum[1],'object_geom':minimum[2],'method':minimum[3]})
    if min(row['plane_gap_m']for row in clearances)<.002 or min(row['minimum_hand_clearance_lower_bound_m']for row in gaps)<.00035:
        raise ValueError('Raw-pose pregrasp lost required table/object positive gap: '+json.dumps({'plane':clearances,'objects':gaps}))
    mj.mj_collision(model,data);robot_scene=[]
    for contact in data.contact[:data.ncon]:
        b=[int(model.geom_bodyid[contact.geom1]),int(model.geom_bodyid[contact.geom2])];bn=[model.body(body).name or'world'for body in b]
        robot=[body>0 and not name.startswith('target_object_')for body,name in zip(b,bn)]
        if any(robot)and not all(robot)and contact.dist<-.001:
            robot_scene.append({'bodies':bn,'distance_m':float(contact.dist)})
    if robot_scene:raise ValueError('Unsafe robot scene penetration '+json.dumps(robot_scene))
    q=np.r_[data.qpos[:7],data.qpos[qa]];n=len(source['time_s']);qpos=np.repeat(q[None,:],n,axis=0)
    payload=dict(source)
    for key in list(payload):
        if key.startswith('robot_')or key in ['planned_stance_mask','ik_error_m','foot_sole_clearance_m','support_adaptation_description']:
            payload['pregrasp_source_'+key]=payload.pop(key)
    payload['pregrasp_source_qpos']=source['qpos'];payload['qpos']=qpos
    payload['pregrasp_nominal_hold_first_lift_only']=np.asarray(True)
    payload['pregrasp_phase_policy']=np.asarray('Initial pose is raw-object-remapped positive-gap candidate; all923 nominal body poses hold. Actual actuator feedback performs first lift. No source upper-body phases or six-object task success claimed.')
    payload['pregrasp_candidate_qpos_model_order']=original_candidate;payload['pregrasp_raw_initial_qpos_model_order']=data.qpos.copy()
    payload.setdefault('source_qpos',source['qpos'])
    payload['pregrasp_joint_adaptation_from_input_rad']=qpos[:,7:]-source['qpos'][:,7:]
    payload['collision_model_path']=np.asarray(str(args.model));payload['collision_model_sha256']=np.asarray(sha(args.model))
    payload['o6_mimic_manifold_projection']=np.asarray(True);payload['o6_independent_command_count']=np.asarray(12);payload['o6_physical_finger_joint_count']=np.asarray(22)
    payload['pregrasp_projected_joint_velocity_rad_s']=np.zeros((n,52));payload['physical_ready']=np.asarray(False)
    payload['planned_stance_mask']=np.ones((n,2),dtype=bool)
    payload['robot_hand_fk_world']=np.repeat(data.xpos[helper.hand_bids][None,:,:],n,axis=0)
    payload['robot_palm_fk_world']=np.repeat(np.asarray([helper.palm(data,side)for side in range(2)])[None,:,:],n,axis=0)
    payload['robot_palm_target_world']=payload['robot_palm_fk_world'].copy();payload['robot_hand_target_world']=payload['robot_hand_fk_world'].copy()
    payload['robot_palm_contact_site_local']=model.site_pos[helper.palm_sites].copy()
    payload['robot_palm_anatomical_frame_in_hand_base']=source['robot_palm_anatomical_frame_in_hand_base']
    orientation=Rotation.from_matrix(data.xmat[helper.hand_bids].reshape(2,3,3)).as_quat()[:,[3,0,1,2]]
    payload['robot_hand_orientation_target_world_wxyz']=np.repeat(orientation[None,:,:],n,axis=0)
    wristb=[mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,side+'_wrist_pitch_link')for side in ['left','right']]
    payload['robot_wrist_target_world']=np.repeat(data.xpos[wristb][None,:,:],n,axis=0)
    payload['robot_hand_safety_adaptation_m']=payload['robot_palm_target_world']-source['human_hand_target_world']
    footb=[mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,side+'_ankle_roll_link')for side in ['left','right']]
    payload['robot_foot_fk_world']=np.repeat(data.xpos[footb][None,:,:],n,axis=0);payload['robot_foot_target_world']=payload['robot_foot_fk_world'].copy()
    payload['robot_com_fk_world_m']=np.repeat(data.subtree_com[mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,'base_link')][None,:],n,axis=0)
    payload['robot_com_target_world_xy_m']=payload['robot_com_fk_world_m'][:,:2].copy()
    floorid=mj.mj_name2id(model,mj.mjtObj.mjOBJ_GEOM,'assumed_floor')
    sole=[min(float(mj.mj_geomDistance(model,data,int(g),floorid,3.,np.zeros(6)))for g in np.flatnonzero((model.geom_bodyid==bid)&(model.geom_contype>0)))for bid in footb]
    payload['foot_sole_clearance_m']=np.repeat(np.asarray(sole)[None,:],n,axis=0)
    payload['support_adaptation_description']=np.asarray('Fixed actual native default12leg posture and exact waist0; no root/foot movement. Positive-gap raw-initial pregrasp nominal hold, not source gait reproduction.')
    for key in ['root_alignment_offset_xyz_m','source_height_scale']:
        if key in source:payload['pregrasp_source_'+key]=source[key]
    payload['root_alignment_offset_xyz_m']=qpos[:,:3]-source['raw_source_root_position_m']
    payload['root_deviation_from_raw_source_xyz_m']=payload['root_alignment_offset_xyz_m'].copy()
    payload['source_height_scale']=np.asarray(0.)
    payload['robot_stance_foot_speed_m_s']=np.zeros((n,2))
    for key in ['robot_dq','qvel']:
        if key in source:payload['pregrasp_source_'+key]=source[key];payload[key]=np.zeros_like(source[key])
    protected=[key for key in source if key.startswith(('raw_source_','human_','object_','contact_candidate'))or key in ['time_s','source_frame_ids','frame_ids']or key.endswith('_mask')and key!='planned_stance_mask'and not key.startswith('robot_')]
    def equal(a,b):return np.array_equal(a,b,equal_nan=True)if a.dtype.kind in 'fc'else np.array_equal(a,b)
    if not all(equal(payload[key],source[key])for key in protected):raise ValueError('Source provenance changed')
    args.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(args.output,**payload)
    report={'schema':'fourth_o6_raw_object_pregrasp_nominal_reference.v1','output':str(args.output),'sha256':sha(args.output),'frames':n,'duration_s':float(source['time_s'][-1]-source['time_s'][0]),
        'role':'923-timeline fixed-stand nominal seed for first real actuator-only object1 lift; no six-object replay/physics claim',
        'model':str(args.model),'model_sha256':sha(args.model),'candidate':str(args.candidate),'candidate_sha256':sha(args.candidate),
        'source_reference':str(args.reference),'source_reference_sha256':sha(args.reference),'source_script_sha256':sha(__file__),
        'root_xyz_wxyz':q[:7].tolist(),'native_legs_unchanged':bool(np.array_equal(q[7+legs],stand_q[legs])),'waist_zero_exact':bool(np.max(abs(q[7+waist]))==0),
        'raw_initial_object_pose_used':True,'raw_template_palm_position_error_m':palm_error,'raw_template_rotation_error_rad':rotation_error,
        'arm_ik_max_delta_rad':float(np.max(abs(fit.x-start))),'finger_commands_unchanged':True,'mimic_equalities':len(relations),'protected_source_arrays_unchanged':len(protected),
        'hand_table_clearances':clearances,'all_six_hand_object_clearances':gaps,'robot_scene_penetrations_gt1mm':robot_scene,
        'source_objects_preserved_exactly':bool(equal(payload['object_target_pose'],source['object_target_pose'])),'dynamics_run':False,'physical_grasp_success_claimed':False,'oracle':oracle.report()}
    args.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
