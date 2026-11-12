"""Prepare an unchanged common-body vendor motion as a dynamic ABI positive control.

This is a diagnostic vendor-motion reference, not the rgb145 source reconstruction.
Run on the server. Only initial forward kinematics is evaluated; no physics steps.
"""
from pathlib import Path
import argparse,hashlib,json,shutil,sys


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--template',type=Path,required=True,help='Existing v12 single-Vita reference; only first rootXY/yaw and auxiliary24 are used')
    p.add_argument('--motion',type=Path,default=Path('data/vita_demo_20261026/motions/012_0806__0806pm__scene3_3__000_action_000.npz'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if sys.platform!='linux':p.error('Prepare this FK-based reference on the SSH server; no local simulation')
    for path in (a.output,a.output.with_suffix('.json'),a.output.with_suffix('.generator.py'),a.output.with_suffix('.vendor_original.npz')):
        if path.exists():raise FileExistsError(path)
    import numpy as np,mujoco as mj
    original=dict(np.load(a.motion,allow_pickle=False));template=dict(np.load(a.template,allow_pickle=False))
    required={'root_pos','root_rot_xyzw','dof_pos','dof_names','fps'}
    if not required.issubset(original):raise ValueError('Missing vendor fields '+str(required-set(original)))
    position=np.asarray(original['root_pos'],float);rotation_xyzw=np.asarray(original['root_rot_xyzw'],float);dof=np.asarray(original['dof_pos'])
    source_names=list(map(str,original['dof_names']));fps=float(np.asarray(original['fps']).item());frames=len(position)
    if frames!=438 or fps!=30. or position.shape!=(frames,3)or rotation_xyzw.shape!=(frames,4)or dof.shape!=(frames,len(source_names)):raise ValueError('Expected the complete438frame30Hz vendor motion')
    if len(set(source_names))!=len(source_names)or not all(np.isfinite(v).all()for v in (position,rotation_xyzw,dof)):raise ValueError('Duplicate source joint or nonfinite source data')
    m=mj.MjModel.from_binary_path(str(a.model.resolve()))if a.model.suffix=='.mjb'else mj.MjModel.from_xml_path(str(a.model.resolve()))
    if(m.nq,m.nv,m.nu)!=(59,58,52):raise ValueError('Expected current single-Vita59/58/52 ABI')
    jids=np.asarray([j for j in range(m.njnt)if m.jnt_type[j]==mj.mjtJoint.mjJNT_HINGE]);jids=jids[np.argsort(m.jnt_qposadr[jids])]
    qa=m.jnt_qposadr[jids];names=[mj.mj_id2name(m,mj.mjtObj.mjOBJ_JOINT,int(j))for j in jids]
    if len(names)!=52 or not np.array_equal(qa,np.arange(7,59)):raise ValueError('Expected one free root atqadr0, followed by52 named hinge coordinates')
    body=np.asarray([k for k,n in enumerate(names)if not n.startswith(('lh_','rh_','neck_'))]);aux=np.asarray([k for k in range(52)if k not in body])
    if len(body)!=28 or len(aux)!=24:raise ValueError('Common body28 and auxiliary24 contract mismatch')
    tnames=list(map(str,template['dof_names']));tq=np.asarray(template['qpos'],float)
    if len(tnames)!=52 or tq.ndim!=2 or tq.shape[1]!=59:raise ValueError('Invalid current template')
    if not set(names).issubset(tnames)or not all(names[k]in source_names for k in body):raise ValueError('Joint names cannot be mapped exactly')
    output=np.tile(m.qpos0,(frames,1))
    for k,n in enumerate(names):output[:,qa[k]]=dof[:,source_names.index(n)]if k in body else tq[0,7+tnames.index(n)]
    # Convert the supplied xyzw representation; retain its original arrays verbatim below.
    source_q=rotation_xyzw[:,[3,0,1,2]].copy();norm=np.linalg.norm(source_q,axis=1)
    if np.max(abs(norm-1.))>1e-5:raise ValueError('Vendor quaternion is not a unit orientation')
    source_q/=norm[:,None]
    def heading(q):
        mat=np.empty(9);mj.mju_quat2Mat(mat,q/np.linalg.norm(q));R=mat.reshape(3,3);return float(np.arctan2(R[1,0],R[0,0]))
    source_heading=heading(source_q[0]);target_heading=heading(tq[0,3:7]);delta_yaw=float(np.arctan2(np.sin(target_heading-source_heading),np.cos(target_heading-source_heading)))
    c,s=np.cos(delta_yaw),np.sin(delta_yaw);Rz=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
    offset_xy=tq[0,:2]-Rz[:2,:2]@position[0,:2];output[:,:3]=position@Rz.T;output[:,:2]+=offset_xy
    zquat=np.array([np.cos(delta_yaw/2),0.,0.,np.sin(delta_yaw/2)]);sign_changes=0
    for i in range(frames):
        mj.mju_mulQuat(output[i,3:7],zquat,source_q[i]);output[i,3:7]/=np.linalg.norm(output[i,3:7])
        if i and np.dot(output[i,3:7],output[i-1,3:7])<0:output[i,3:7]*=-1;sign_changes+=1
    ground=mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,'ground_support')
    if ground<0 or m.geom_type[ground]!=mj.mjtGeom.mjGEOM_PLANE or m.geom_bodyid[ground]!=0:raise ValueError('Expected unchanged static ground_support plane in effective model')
    feet=[g for g in range(m.ngeom)if(mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,g)or'').startswith(('collision_foot_l_','collision_foot_r_'))and m.geom_contype[g]]
    if not feet:raise ValueError('No actual foot collision geoms')
    d=mj.MjData(m);d.qpos[:]=output[0];mj.mj_kinematics(m,d)
    normal=d.geom_xmat[ground].reshape(3,3)[:,2].copy();normal*=1 if normal[2]>=0 else-1;point=d.geom_xpos[ground].copy()
    if normal[2]<.9:raise ValueError('Ground normal is unsuitable for a vertical-only initial adjustment')
    def gaps():
        rows=[]
        for g in feet:
            typ=m.geom_type[g];distance=float((d.geom_xpos[g]-point)@normal);radius=float(m.geom_size[g,0])
            if typ==mj.mjtGeom.mjGEOM_CAPSULE:distance-=radius+float(m.geom_size[g,1])*abs(float(d.geom_xmat[g].reshape(3,3)[:,2]@normal))
            elif typ==mj.mjtGeom.mjGEOM_SPHERE:distance-=radius
            else:raise ValueError('Positive control requires the actual canonical capsule/sphere foot collider')
            rows.append({'geom':mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,g),'signed_ground_distance_m':distance})
        return rows
    before=gaps();dz=float((.001-min(v['signed_ground_distance_m']for v in before))/normal[2]);output[:,2]+=dz
    d.qpos[:]=output[0];mj.mj_kinematics(m,d);after=gaps()
    if abs(min(v['signed_ground_distance_m']for v in after)-.001)>1e-9:raise AssertionError('Initial actual sole is not1mm from ground')
    source_cols=np.asarray([source_names.index(names[k])for k in body])
    if not np.array_equal(output[:,qa[body]],dof[:,source_cols].astype(float)):raise AssertionError('Vendor body28 was changed')
    if not np.array_equal(output[:,qa[aux]],np.broadcast_to(output[0,qa[aux]],(frames,len(aux)))):raise AssertionError('Auxiliary template pose was changed over time')
    mapped_source=dof[:,source_cols].astype(float);ranges=m.jnt_range[jids[body]];violation=np.maximum(np.maximum(ranges[:,0]-mapped_source,mapped_source-ranges[:,1]),0.)
    a.output.parent.mkdir(parents=True,exist_ok=True);time=np.arange(frames)/fps
    original_fields={'vendor_original_'+k:v for k,v in original.items()}
    np.savez_compressed(a.output,qpos=output,dof_names=np.asarray(names),time_s=time,source_time_s=time.copy(),frame_ids=np.arange(frames),diagnostic_not_rgb145_source=np.asarray(True),reference_kind=np.asarray('vendor_common_body28_dynamic_ABI_positive_control'),vendor_to_template_rotation_z=Rz,vendor_to_template_translation_xyz=np.r_[offset_xy,dz],vendor_common_body_joint_names=np.asarray([names[k]for k in body]),auxiliary_template_joint_names=np.asarray([names[k]for k in aux]),**original_fields)
    preserved=a.output.with_suffix('.vendor_original.npz');shutil.copyfile(a.motion,preserved)
    report={'schema':'rgb145.vendor_dynamic_ABI_positive_control.v1','diagnostic_not_rgb145_source':True,'task_success_claim':False,'physics_validated':False,'physics_steps':0,'model_parameter_mutations':0,'frames':frames,'fps':fps,'source_last_pts_s':float(time[-1]),'display_duration_s':frames/fps,'source_frames_dropped':0,'motion_amplitude_scale':1.,'joint_clipping_applied':False,'body28_exactly_preserved_by_name':True,'body_joint_mapping':[{'name':names[k],'source_column':int(source_names.index(names[k])),'model_qadr':int(qa[k])}for k in body],'source_motion_model':'Vendor Vita02a Linkerhand articulation; only common body28 is transferred','current_auxiliary_pose':'All24 current O6 finger/neck coordinates fixed to the v12 template first frame; vendor hand/neck trajectories are not transferred','auxiliary_reference_rad':{names[k]:float(output[0,qa[k]])for k in aux},'contact_plan':'Intentionally omitted; this diagnostic checks dynamic tracking ABI, not the full rgb145 task','transform':{'constant_world_Z_rotation_rad':delta_yaw,'constant_world_Z_rotation_matrix':Rz.tolist(),'constant_root_translation_xyz_m':np.r_[offset_xy,dz].tolist(),'initial_template_root_xy_m':tq[0,:2].tolist(),'initial_vendor_heading_rad':source_heading,'initial_template_heading_rad':target_heading,'initial_actual_sole_clearance_target_m':.001,'initial_foot_ground_distances_before':before,'initial_foot_ground_distances_after':after},'quaternion_representation':{'input':'xyzw','output':'wxyz','unit_normalization_max_abs_correction':float(np.max(abs(norm-1.))),'hemisphere_sign_changes':sign_changes},'body_reference_joint_range_violation_rad_max_without_clipping':float(violation.max()),'input':{'motion_path':str(a.motion.resolve()),'motion_sha256':sha(a.motion),'effective_model_path':str(a.model.resolve()),'effective_model_sha256':sha(a.model),'v12_template_path':str(a.template.resolve()),'v12_template_sha256':sha(a.template)},'reference_sha256':sha(a.output),'verbatim_vendor_copy_sha256':sha(preserved),'source_pickle_sha256_provenance_only':original.get('source_pickle_sha256',np.asarray([])).tolist(),'generator_sha256':sha(__file__)}
    a.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');a.output.with_suffix('.generator.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
