"""Export audited RGB145 HOI data; failed/partial runs require --diagnostic.

Run with the trial's MuJoCo version on the simulation server. Body poses use
forward kinematics only; this exporter never steps or repairs the simulation.
"""
from pathlib import Path
import argparse,hashlib,json,shutil,sys

_HASH_CACHE={}


def sha(path):
    path=Path(path);stat=path.stat();key=(str(path.resolve()),stat.st_size,stat.st_mtime_ns)
    if key in _HASH_CACHE:return _HASH_CACHE[key]
    h=hashlib.sha256()
    with path.open('rb')as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    _HASH_CACHE[key]=h.hexdigest();return _HASH_CACHE[key]


def receipt(path):
    p=Path(path).resolve();return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}


def write_json(path,obj):Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trial',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--reference',type=Path)
    p.add_argument('--diagnostic',action='store_true');a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    trial=a.trial.resolve();source=a.source.resolve()
    source_manifest=json.loads((source/'handoff_manifest.json').read_text(encoding='utf-8'))
    if source_manifest.get('objects'):raise ValueError('Source contains dynamic objects; this export contract must not invent or omit their trajectories')
    report_path=trial/'report.json';replay_path=trial/'independent_replay.json'
    report=json.loads(report_path.read_text(encoding='utf-8'));replay=json.loads(replay_path.read_text(encoding='utf-8'))
    task_passed=report.get('simulation_passed')is True;replay_passed=replay.get('replay_passed')is True
    if task_passed and(not report.get('checks')or not all(v is True for v in report['checks'].values())):raise ValueError('simulation_passed disagrees with task checks')
    if replay_passed and(not replay.get('checks')or not all(v is True for v in replay['checks'].values())):raise ValueError('replay_passed disagrees with replay checks')
    accepted=task_passed and replay_passed
    if not accepted and not a.diagnostic:raise RuntimeError('Formal HOI export requires simulation_passed AND replay_passed. Use --diagnostic for explicitly failed/partial data.')
    import numpy as np,mujoco as mj
    model_path=trial/'model.mjb';trace_path=trial/'trace.npz';contacts_path=trial/'contacts.npz'
    for path,expected in ((model_path,report['model']['sha256']),(trace_path,report['trace']['sha256'])):
        if sha(path)!=expected:raise ValueError('Trial SHA mismatch: '+str(path))
    if replay.get('model_sha256')!=sha(model_path)or replay.get('trace_sha256')!=sha(trace_path):raise ValueError('Replay does not bind the current frozen model/trace')
    if mj.__version__!=report['mujoco_version']or mj.__version__!=replay.get('mujoco_version'):raise ValueError('Run exporter with the original trial/replay MuJoCo version')
    ref_receipt=report['input']['reference'];candidates=[a.reference]if a.reference else[Path(ref_receipt['path']),trial/Path(ref_receipt['path']).name,trial.parent/Path(ref_receipt['path']).name]
    reference=next((v for v in candidates if v is not None and v.is_file()and sha(v)==ref_receipt['sha256']),None)
    if reference is None:raise FileNotFoundError('Exact reference NPZ unavailable; supply --reference with matching SHA256')
    z=np.load(trace_path,allow_pickle=False);co=np.load(contacts_path,allow_pickle=False);ref=np.load(reference,allow_pickle=False)
    tl=np.load(source/'timeline.npz',allow_pickle=False);params=np.load(source/'human/person_01/smplx_params_world.npz',allow_pickle=False)
    camera=np.load(source/'cameras_world.npz',allow_pickle=False)
    ids=np.asarray(tl['frame_ids']);pts=np.asarray(tl['pts_seconds'],float)
    if len(ids)!=350 or not np.array_equal(ids,z['source_frame_ids'])or not np.array_equal(ids,ref['frame_ids']):raise ValueError('Original 350 frame IDs must match trace and reference exactly')
    for name,pack in (('SMPL-X',params),('camera',camera)):
        if not np.array_equal(ids,pack['frame_ids'])or not np.allclose(pts,pack['pts_seconds'],atol=1e-7,rtol=0):raise ValueError(name+' source clock mismatch')
    execution=np.asarray(z['reference_execution_time_s'],float);source_time=np.asarray(z['reference_source_time_s'],float)
    if execution.shape!=(350,)or not np.all(np.diff(execution)>0)or not np.allclose(source_time,pts-pts[0],atol=1e-7,rtol=0):raise ValueError('Source/execution clock mapping mismatch')
    if not np.array_equal(execution,ref['time_s']):raise ValueError('Frozen reference execution clock differs from trace')
    required=('source_joints_world_zup','source_root_world_from_pelvis_zup','source_to_sim_U')
    if any(k not in ref for k in required):raise ValueError('Reference lacks preserved source human/Z-up coordinate fields')
    U=np.asarray(ref['source_to_sim_U'],float)
    if not np.array_equal(U,np.array([[1,0,0],[0,0,-1],[0,1,0]])):raise ValueError('Unexpected source-to-simulation rotation')
    m=mj.MjModel.from_binary_path(str(model_path));d=mj.MjData(m)
    if(m.nq,m.nv,m.nu,m.neq)!=(59,58,52,10)or np.any(m.eq_type!=int(mj.mjtEq.mjEQ_JOINT)):raise ValueError('Expected one Vita 59/58/52 with ten typed joint couplings')
    if m.jnt_type[0]!=int(mj.mjtJoint.mjJNT_FREE)or np.any(m.jnt_type[1:]!=int(mj.mjtJoint.mjJNT_HINGE))or not np.all(np.isin(m.actuator_gear[:,0],[0.,1.]))or np.any(m.actuator_gear[:,1:]):raise ValueError('Rotational actuator torque/unit contract mismatch')
    if abs(m.opt.timestep-.005)>1e-12:raise ValueError('This 200Hz export requires a 5ms physics timestep')
    if report.get('dynamic_objects')!=0:raise ValueError('This source/export schema requires zero independent dynamic objects')
    times=np.asarray(z['time'],float);steps=len(z['ctrl'])
    if steps<1 or len(times)!=steps+1 or np.any(np.diff(times)<=0)or z['qpos'].shape!=(len(times),m.nq)or z['qvel'].shape!=(len(times),m.nv):raise ValueError('Malformed recorded state/interval timeline')
    interval=np.asarray(co['interval_index'],int)
    if np.any(interval<0)or np.any(interval>=steps)or np.any(np.diff(interval)<0):raise ValueError('Invalid raw contact interval indices')
    if not np.allclose(co['solver_time_s'],times[interval],atol=1e-8,rtol=0):raise ValueError('Contacts must use recorded interval-start time, never post-step time')
    if z['eq_generalized_force'].shape!=(steps,10,m.nv):raise ValueError('Ten equality generalized reactions are required at every real physics interval')
    # Nearest recorded state only within coverage. Never clamp later source
    # frames to the last available physics state.
    covered=(execution>=times[0]-1e-9)&(execution<=times[-1]+1e-9)
    right=np.searchsorted(times,execution).clip(0,len(times)-1);left=np.maximum(right-1,0)
    nearest=np.where(abs(times[left]-execution)<=abs(times[right]-execution),left,right)
    valid=covered&np.isfinite(z['qpos'][nearest]).all(1)&np.isfinite(z['qvel'][nearest]).all(1)
    for key in ('ctrl','actuator_force','eq_generalized_force','qfrc_equality','qfrc_constraint'):
        values=np.asarray(z[key])[np.minimum(nearest,steps-1)]
        valid&=np.isfinite(values.reshape(350,-1)).all(1)
    if accepted and not valid.all()and not a.diagnostic:raise ValueError('Formal sequence lacks actual recorded physics for all 350 source samples')
    formal=accepted and valid.all()and not a.diagnostic
    label='FORMAL_AUDITED_HOI'if formal else'DIAGNOSTIC_FAILED_OR_UNACCEPTED_HOI'
    a.output.mkdir(parents=True);write_json(a.output/'EXPORT_INCOMPLETE.json',{'status':'incomplete_do_not_deliver'})
    records=[]
    def copy_input(path,relative,origin):
        dest=a.output/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
        r=receipt(dest);r.update(source=receipt(path),relative_path=str(Path(relative).as_posix()),origin=origin);records.append(r)
    for path,relative,origin in ((model_path,'assets/model.mjb','imported_model_and_assumed_physics'),(trace_path,'raw_physics/trace.npz','simulation'),(contacts_path,'raw_physics/contacts.npz','simulation'),(report_path,'validation/report.json','task_quality_audit'),(replay_path,'validation/independent_replay.json','independent_dynamics_replay'),(reference,'reference_motion.npz','kinematic_retargeting_and_inferred_contacts')):
        copy_input(path,relative,origin)
    source_files=('timeline.npz','cameras_world.npz','coordinates.npz','handoff_manifest.json','README_FOR_SIM.zh-CN.txt','SHA256SUMS.txt','human/person_01/smplx_params_world.npz','human/person_01/root_poses_world.npz','scene/pi3_mesh.npz')
    for rel in source_files:copy_input(source/rel,Path('source')/rel,'reconstruction'if rel.endswith('.npz')and rel!='timeline.npz'else'source_metadata')
    for path in (trial/'declared_physics_contract_v1.json',trial/'run_config.json',trial.parent/'support_geometry.json',trial.parent/'assets/assets_report.json'):
        if path.exists():copy_input(path,Path('assumptions')/path.name,'declared_contract_or_assumption')
    for path in sorted(trial.glob('source_*')):
        if path.is_file():copy_input(path,Path('provenance')/path.name,'frozen_runtime_source')
    copy_input(Path(__file__),'provenance/rgb145_export_hoi.py','exporter_source')
    state_ids=np.where(valid,nearest,-1);force_ids=np.where(valid,np.minimum(nearest,steps-1),-1)
    def states(key):
        src=np.asarray(z[key]);out=np.full((350,)+src.shape[1:],np.nan);out[valid]=src[nearest[valid]];return out
    def intervals(key):
        src=np.asarray(z[key]);out=np.full((350,)+src.shape[1:],np.nan);out[valid]=src[force_ids[valid]];return out
    state_time=np.full(350,np.nan);state_time[valid]=times[nearest[valid]]
    force_time=np.full(350,np.nan);force_end=np.full(350,np.nan)
    force_time[valid]=times[force_ids[valid]];force_end[valid]=times[force_ids[valid]+1]
    body_pose=np.full((350,m.nbody,7),np.nan)
    actual_q=states('qpos');actual_v=states('qvel')
    for f in np.flatnonzero(valid):
        d.qpos[:]=actual_q[f];d.qvel[:]=actual_v[f];mj.mj_kinematics(m,d)
        body_pose[f,:,:3]=d.xpos;body_pose[f,:,3:]=d.xquat
    body_names=np.asarray([mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,b)or'world'for b in range(m.nbody)])
    generalized_units=np.asarray(['N']*3+['Nm']*(m.nv-3))
    np.savez_compressed(a.output/'robot_source_frames.npz',frame_ids=ids,source_pts_seconds_original=pts,source_time_s=source_time,reference_execution_time_s=execution,recorded_state_valid=valid,sequence_physics_passed=np.full(350,formal,bool),actual_state_index=state_ids,actual_state_time_s=state_time,state_time_error_s=state_time-execution,actual_interval_index=force_ids,force_solver_time_s=force_time,force_interval_end_s=force_end,force_solver_time_error_s=force_time-execution,qpos=actual_q,qvel=actual_v,ctrl=intervals('ctrl'),actuator_force_Nm=intervals('actuator_force'),body_pose_world_xyz_wxyz=body_pose,body_names=body_names,joint_names=z['joint_names'],joint_qadr=z['joint_qadr'],joint_vadr=z['joint_vadr'],actuator_ids=z['actuator_ids'],eq_generalized_force=intervals('eq_generalized_force'),qfrc_equality=intervals('qfrc_equality'),qfrc_constraint=intervals('qfrc_constraint'),generalized_force_units=generalized_units,status=np.asarray(label))
    human={'frame_ids':ids,'source_pts_seconds_original':pts,'joints_world_zup':ref['source_joints_world_zup'],'world_from_pelvis_zup':ref['source_root_world_from_pelvis_zup'],'source_to_sim_U':U,'human_valid':tl['human_valid'],'smplx_parameter_valid':params['valid'],'hand_valid':params['hand_valid']}
    np.savez_compressed(a.output/'human_source_zup.npz',**human)
    cam_zup=np.asarray(camera['world_from_camera']).copy();cam_zup[:,:3,:]=np.einsum('ij,tjk->tik',U,cam_zup[:,:3,:])
    np.savez_compressed(a.output/'camera_source_zup.npz',frame_ids=ids,source_pts_seconds_original=pts,K=camera['K'],world_from_camera_yup_original=camera['world_from_camera'],world_from_camera_zup=cam_zup,pnp_valid=camera['pnp_valid'],source_to_sim_U=U)
    force_norm=np.linalg.norm(co['force_world_on_geom2_N'],axis=1)
    edge_kind=np.where(np.any(co['body_ids']==0,axis=1),'robot_scene_contact','robot_self_contact')
    graph={k:co[k]for k in co.files}
    graph.update(edge_kind=edge_kind,active_force_edge=force_norm>1e-8,physics_interval_time_s=times[:-1],physics_interval_end_s=times[1:],physics_interval_source_time_s=np.interp(times[:-1],execution,source_time),status=np.asarray(label))
    np.savez_compressed(a.output/'contact_graph_200hz.npz',**graph)
    eqnames=np.asarray([mj.mj_id2name(m,mj.mjtObj.mjOBJ_EQUALITY,e)for e in range(m.neq)])
    object1=np.asarray([mj.mj_id2name(m,mj.mjtObj.mjOBJ_JOINT,int(j))for j in m.eq_obj1id]);object2=np.asarray([mj.mj_id2name(m,mj.mjtObj.mjOBJ_JOINT,int(j))for j in m.eq_obj2id])
    np.savez_compressed(a.output/'joint_coupling_reactions_200hz.npz',physics_interval_time_s=times[:-1],equality_names=eqnames,equality_type_id=m.eq_type,equality_kind=np.full(10,'internal_polynomial_joint_coupling'),follower_joint_names=object1,driver_joint_names=object2,eq_active=z['eq_active'],eq_data=z['eq_data'],eq_generalized_force=z['eq_generalized_force'],qfrc_equality=z['qfrc_equality'],generalized_force_units=generalized_units,status=np.asarray(label))
    parameter_fields=('body_parentid','body_pos','body_quat','body_mass','body_inertia','body_ipos','body_iquat','geom_type','geom_bodyid','geom_size','geom_pos','geom_quat','geom_friction','geom_contype','geom_conaffinity','geom_solref','geom_solimp','jnt_type','jnt_pos','jnt_axis','jnt_range','jnt_limited','dof_armature','dof_frictionloss','dof_damping','actuator_trnid','actuator_gear','actuator_gainprm','actuator_biasprm','actuator_forcerange','eq_type','eq_obj1id','eq_obj2id','eq_data')
    np.savez_compressed(a.output/'assets/physical_parameters.npz',**{k:np.asarray(getattr(m,k))for k in parameter_fields},gravity_world_m_s2=m.opt.gravity,physics_dt_s=np.asarray(m.opt.timestep),body_names=body_names)
    node_metadata={'body_nodes':[{'body_id':b,'name':str(body_names[b]),'kind':'static_world'if b==0 else'robot_body'}for b in range(m.nbody)],'geom_names':list(map(str,co['geom_names'])),'dynamic_objects':0,'object_6d_trajectories':'not_applicable_source_contains_no_independent_dynamic_objects','force_convention':'contact force/torque expressed in Z-up world, applied to geom2; geom1 receives the negative wrench','equality_reactions':'Ten internal joint-coupling generalized reactions, not external contact forces; first3 generalized components N, remaining Nm'}
    write_json(a.output/'contact_graph_nodes.json',node_metadata)
    (a.output/'README.txt').write_text(label+'\n'+('正式导出：全程物理验收与独立动力学回放均通过。\n'if formal else'诊断数据：未作为通过验收的完整物理结果。严禁用于声称任务成功。\n')+'保留原始350帧与原时间轴；未覆盖机器人帧为NaN、valid=false，不复制末帧。\n力/力矩均为仿真输出；源人体、场景和相机为重建估计；动力学参数未经现场测量标定。\n源视频与完整人体顶点仅保留外部输入收据；源SMPL-X参数、人体关节/根位姿、相机和场景mesh已包含。\n冻结model.mjb和physical_parameters.npz为实际仿真参数依据；上游assets_report可能描述运行前资产构建。\n详见manifest.json、validation/和assumptions/。\n',encoding='utf-8')
    files=[{**receipt(path),'relative_path':path.relative_to(a.output).as_posix()}for path in sorted(a.output.rglob('*'))if path.is_file()and path.name!='EXPORT_INCOMPLETE.json']
    manifest={'schema':'rgb145.hoi_motion_physics.v1','status':label,'formal_accepted':formal,'diagnostic_requested':a.diagnostic,'gates':{'simulation_passed':task_passed,'replay_passed':replay_passed,'all350_recorded_states_available':bool(valid.all())},'frames':350,'recorded_valid_frames':int(valid.sum()),'uncovered_or_nonfinite_frame_ids':ids[~valid].tolist(),'source_last_pts_s':float(pts[-1]),'source_display_duration_s':float(pts[-1]-pts[0]+np.median(np.diff(pts))),'recorded_execution_duration_s':float(times[-1]),'reference_execution_last_pts_s':float(execution[-1]),'source_to_sim_U':U.tolist(),'coordinate_conventions':{'source':'right-handed Y-up; camera OpenCV x-right y-down z-forward','simulation_and_exported_world_poses':'right-handed Z-up; p_zup=U@p_yup','body_pose_layout':'world xyz metres followed by quaternion wxyz','original_smplx_parameters':'Copied unchanged in source Y-up. transl is decoder translation, NOT pelvis position. Do not apply source coordinates.npz transform again.'},'units':{'length':'approximate metres from reconstruction scale prior','time':'seconds; original source and retimed execution clocks both retained','joint_angle':'radians','joint_velocity':'radians/second; free-root linear entries metres/second','actuator_force':'Nm for this rotational-joint motor model','contact_force':'N, simulated','contact_torque':'Nm, simulated','mass':'kg, imported/assumed','inertia':'kg*m^2, imported/assumed'},'temporal_alignment':{'state':'nearest actually recorded state within coverage, no interpolation or last-frame extension','signed_error':'actual_state_time_s minus reference_execution_time_s','forces':'recorded solver interval beginning at selected state, except terminal state uses the preceding interval; independent timestamps and errors are stored','invalid_frames':'All robot numeric outputs including FK/forces are NaN; indices -1; recorded_state_valid false. Source human/camera observations remain preserved.','raw_physics':'All real intervals and contact edges preserved without source-frame downsampling','physics_dt_s':float(m.opt.timestep),'state_time_error_abs_max_s':float(np.nanmax(abs(state_time-execution)))if valid.any()else None},'provenance_classes':{'measured_or_recorded':'Original RGB video receipt only; no measured force/torque or physical calibration claimed','reconstructed':'Source SMPL-X parameters, joints/root, scene mesh and cameras','retargeted_or_inferred':'Robot reference and contact plan; hand reconstruction is invalid according to source hand_valid','simulated':'Actual robot states, joint actuator outputs, raw contact wrenches and internal coupling reactions','assumed_or_imported':'Frozen robot inertias/collision shapes, scene collision proxies/friction and actuator parameters; see assets/physical_parameters.npz and assumption receipts'},'input_receipts':{'report':receipt(report_path),'replay':receipt(replay_path),'model':receipt(model_path),'trace':receipt(trace_path),'contacts':receipt(contacts_path),'reference':receipt(reference),'source_video':receipt(source/'input.mp4'),'source_decoded_human_mesh':receipt(source/'human/person_01/smplx_meshes_world.npz'),'exporter':receipt(__file__)},'copied_input_receipts':records,'files':files,'mujoco_version':mj.__version__,'body_fk':'Server MuJoCo mj_kinematics on recorded qpos only; no mj_step, state repair, policy inference or new force calculation','physical_assumptions_and_limits':report.get('limitations',[]),'dynamic_objects':0,'object_trajectory_status':'not_applicable','ego_view':'Not generated by this exporter; any rendered head camera is synthetic and requires a separately declared extrinsic assumption.'}
    write_json(a.output/'manifest.json',manifest)
    (a.output/'EXPORT_INCOMPLETE.json').unlink()
    print(json.dumps({'status':label,'output':str(a.output.resolve()),'formal_accepted':formal,'valid_robot_source_frames':int(valid.sum()),'frames':350,'manifest_sha256':sha(a.output/'manifest.json')},ensure_ascii=False),flush=True)
    return 0


if __name__=='__main__':sys.exit(main())
