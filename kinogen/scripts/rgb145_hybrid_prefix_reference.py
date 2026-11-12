"""Blend the known prefix; independently audit hybrid FK, limits and static knee load."""
from pathlib import Path
import argparse,hashlib,json,sys


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prefix',type=Path,required=True,help='Unretimed reference_v2.npz')
    p.add_argument('--suffix',type=Path,required=True,help='Unretimed reference_v12_robot_turn.npz')
    p.add_argument('--model',type=Path,required=True,help='Unchanged effective single-Vita model')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--blend-start-frame',type=int,default=90,help='Last frame copied from v2 root/body, except relaxed wrist/auxiliary coordinates')
    p.add_argument('--blend-end-frame',type=int,default=109,help='First frame copied from v12 root/body')
    p.add_argument('--knee-load-target-ratio',type=float,default=.8,help='Diagnostic raw actuator load ratio threshold; does not alter the hybrid poses or contact plan')
    a=p.parse_args()
    if sys.platform!='linux':p.error('Run this FK/collision audit on the SSH server; no local simulation')
    for path in(a.output,a.output.with_suffix('.json'),a.output.with_suffix('.generator.py'),a.output.with_suffix('.knee_load.py')):
        if path.exists():raise FileExistsError(path)
    import numpy as np,mujoco as mj
    from rgb145_knee_load import KneeLoadProxy
    if a.knee_load_target_ratio<0 or not np.isfinite(a.knee_load_target_ratio):raise ValueError('Invalid diagnostic knee-load ratio threshold')
    first=dict(np.load(a.prefix,allow_pickle=False));second=dict(np.load(a.suffix,allow_pickle=False))
    qa=np.asarray(first['qpos'],float);qb=np.asarray(second['qpos'],float)
    if qa.shape!=(350,59)or qb.shape!=qa.shape:raise ValueError('Both parents must contain all350 source poses')
    ids=np.asarray(first['frame_ids']);time=np.asarray(first['time_s']);source=np.asarray(first.get('source_time_s',time))
    if not np.array_equal(ids,second['frame_ids'])or not np.array_equal(time,second['time_s'])or not np.array_equal(source,second.get('source_time_s',second['time_s'])):raise ValueError('Parent source IDs/timelines differ; use the unretimed parents')
    if not np.array_equal(time,source)or np.any(np.diff(time)<=0):raise ValueError('This candidate preserves the original source clock; retime downstream')
    if not 0<=a.blend_start_frame<a.blend_end_frame<350:raise ValueError('Invalid fusion window')
    names=list(map(str,first['dof_names']));other_names=list(map(str,second['dof_names']))
    if len(names)!=52 or len(set(names))!=52 or set(names)!=set(other_names):raise ValueError('Parent52joint names differ')
    qb=qb.copy();qb[:,7:]=np.asarray(second['qpos'])[:,7+np.asarray([other_names.index(n)for n in names])]
    m=mj.MjModel.from_binary_path(str(a.model.resolve()))if a.model.suffix=='.mjb'else mj.MjModel.from_xml_path(str(a.model.resolve()))
    if(m.nq,m.nv,m.nu)!=(59,58,52):raise ValueError('Expected current59/58/52 plant')
    jids=np.asarray([mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,n)for n in names]);qadr=m.jnt_qposadr[jids]
    if np.any(jids<0)or not np.array_equal(qadr,np.arange(7,59)):raise ValueError('Expected the current canonical root+52 coordinate ABI')
    u=np.clip((np.arange(350)-a.blend_start_frame)/(a.blend_end_frame-a.blend_start_frame),0,1);weight=u**3*(10-15*u+6*u*u)
    out=qa*(1-weight[:,None])+qb*weight[:,None]
    def slerp(aq,bq,t):
        aq=aq/np.linalg.norm(aq);bq=bq/np.linalg.norm(bq);dot=float(np.dot(aq,bq))
        if dot<0:bq=-bq;dot=-dot
        if dot>.9995:q=(1-t)*aq+t*bq
        else:
            angle=np.arccos(np.clip(dot,-1,1));q=(np.sin((1-t)*angle)*aq+np.sin(t*angle)*bq)/np.sin(angle)
        return q/np.linalg.norm(q)
    for i,w in enumerate(weight):out[i,3:7]=slerp(qa[i,3:7],qb[i,3:7],float(w))
    sign_changes=0
    for i in range(1,350):
        if np.dot(out[i-1,3:7],out[i,3:7])<0:out[i,3:7]*=-1;sign_changes+=1
    aux=[i for i,n in enumerate(names)if n.startswith(('lh_','rh_','neck_'))];wrists=[i for i,n in enumerate(names)if '_wrist_'in n]
    body=[i for i in range(52)if i not in aux];common_body=[i for i in body if i not in wrists]
    if(len(aux),len(wrists),len(common_body))!=(24,6,22):raise ValueError('Expected auxiliary24, wrist6 and remaining body22')
    out[:,7+np.asarray(aux)]=qb[0,7+np.asarray(aux)]
    wrist_targets=dict(zip(map(str,second.get('wrist_reference_joint_names',np.asarray([]))),map(float,second.get('wrist_reference_joint_targets_rad',np.asarray([])))))
    wrist_policy='Explicit v12 relaxed wrist targets'if all(names[i]in wrist_targets for i in wrists)else'Fallback v12 first-frame wrist pose'
    for i in wrists:out[:,7+i]=wrist_targets.get(names[i],float(qb[0,7+i]))
    prefix_mask=weight==0;suffix_mask=weight==1;blend_mask=~(prefix_mask|suffix_mask)
    if not np.array_equal(out[prefix_mask][:,7+np.asarray(common_body)],qa[prefix_mask][:,7+np.asarray(common_body)])or not np.array_equal(out[prefix_mask,:3],qa[prefix_mask,:3]):raise AssertionError('Known prefix root translation/body22 was changed')
    if not np.array_equal(out[suffix_mask][:,7+np.asarray(common_body)],qb[suffix_mask][:,7+np.asarray(common_body)])or not np.array_equal(out[suffix_mask,:3],qb[suffix_mask,:3]):raise AssertionError('Suffix root translation/body22 was changed')
    def plan(parent):
        on=np.asarray(parent.get('foot_support_expected',np.zeros((350,2),bool)),bool);surface=np.asarray(parent.get('foot_support_geom_name',np.full((350,2),''))).astype('<U40');confidence=np.asarray(parent.get('foot_support_confidence',np.zeros((350,2))),float)
        if on.shape!=(350,2)or surface.shape!=on.shape or confidence.shape!=on.shape:raise ValueError('Invalid parent support plan')
        return on,surface,confidence
    pa,na,ca=plan(first);pb,nb,cb=plan(second);expected=pa.copy();surface=na.copy();confidence=ca.copy()
    expected[suffix_mask]=pb[suffix_mask];surface[suffix_mask]=nb[suffix_mask];confidence[suffix_mask]=cb[suffix_mask]
    expected[blend_mask]=(pa&pb&(na==nb))[blend_mask];surface[blend_mask]=np.where(na==nb,na,'')[blend_mask];confidence[blend_mask]=np.where(expected,np.minimum(ca,cb),0.)[blend_mask]
    ankle_ids=np.asarray([mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,s+'_ankle_roll_link')for s in('left','right')]);base=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'base_link')
    if np.any(ankle_ids<0)or base<0:raise ValueError('Missing Vita FK bodies')
    d=mj.MjData(m);ankles=[];com=[];penetration=[];rows=[];ranges=m.jnt_range[jids].copy();limited=m.jnt_limited[jids].astype(bool);load_proxy=KneeLoadProxy(m);load_rows=[]
    def physical_violation(q):
        v=np.maximum(np.maximum(ranges[:,0]-q[:,qadr],q[:,qadr]-ranges[:,1]),0.);v[:,~limited]=0.;return v
    limit=physical_violation(out)
    def limit_summary(q,mask,columns=None):
        columns=np.arange(52)if columns is None else np.asarray(columns);v=physical_violation(q);selected=v[mask][:,columns];frame_indices=np.flatnonzero(mask);bad=[]
        for col in columns:
            values=v[mask,col];i=int(np.argmax(values))
            if values[i]>.05:
                frame=int(frame_indices[i]);bad.append({'joint':names[col],'max_violation_rad':float(values[i]),'worst_frame':int(ids[frame]),'reference_rad':float(q[frame,qadr[col]]),'physical_lower_rad':float(ranges[col,0]),'physical_upper_rad':float(ranges[col,1]),'frames_exceeding_0_05rad':int(np.sum(values>.05))})
        return {'max_violation_rad':float(selected.max()),'passed_0_05rad':bool(selected.max()<=.05),'frames_exceeding_0_05rad':frame_indices[np.any(selected>.05,axis=1)].tolist(),'offending_joints':bad}
    parent_limit_audit={'parent_v2_prefix_frames0_to90_raw_all52':limit_summary(qa,np.arange(350)<=90),'parent_v2_prefix_frames0_to90_retained_body22':limit_summary(qa,np.arange(350)<=90,common_body),'actual_hybrid_prefix_all52':limit_summary(out,prefix_mask),'parent_v12_full_all52':limit_summary(qb,np.ones(350,bool))}
    for i in range(350):
        d.qpos[:]=out[i];d.qvel[:]=0.;mj.mj_fwdPosition(m,d)
        ankles.append(d.xpos[ankle_ids].copy());com.append(d.subtree_com[base].copy())
        _,_,load=load_proxy.single_stance_cost(d,expected[i],seated=bool(source[i]>=9.),ratio_threshold=a.knee_load_target_ratio)
        load_info=None if load is None else{k:(v.tolist()if isinstance(v,np.ndarray)else v)for k,v in load.items()if not k.startswith('jacobian_')}
        if load_info is not None:
            gear=load_proxy.knees[load_info['side']]['gear'];load_info['gear']=float(gear);load_info['original_joint_torque_range_Nm']=np.sort(np.asarray(load_info['original_actuator_force_range'])*gear).tolist()
        load_rows.append(load_info)
        worst=max(d.contact[:d.ncon],key=lambda c:-float(c.dist),default=None);depth=max(0.,-float(worst.dist))if worst is not None else 0.;penetration.append(depth)
        rows.append({'frame':int(ids[i]),'source_time_s':float(source[i]),'v12_weight':float(weight[i]),'max_static_penetration_m':depth,'worst_geom_pair':[mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,int(g))for g in(worst.geom1,worst.geom2)]if worst is not None else[],'joint_limit_violation_rad_max':float(limit[i].max()),'hybrid_stance_knee_load_proxy':load_info})
    penetration=np.asarray(penetration);ankles=np.asarray(ankles);com=np.asarray(com)
    if not np.isfinite(out).all()or not np.isfinite(ankles).all()or not np.isfinite(com).all():raise ValueError('Nonfinite hybrid geometry')
    raw={k:v for k,v in first.items()if k.startswith('source_')}
    for k,v in second.items():
        if k.startswith('source_')and k not in raw:raw[k]=v
    raw.pop('source_time_s',None)
    # Preserve all suffix optimization diagnostics only under an explicit parent namespace.
    # In particular, root-bound activity, ankle target errors and knee loads cannot
    # describe the substituted prefix or blend and are never published as hybrid facts.
    parent_suffix_fields={'parent_suffix_'+k:v for k,v in second.items()if not k.startswith('source_')}
    def load_summary(mask):
        valid=[(i,v)for i,v in enumerate(load_rows)if mask[i]and v is not None]
        return {'eligible_frames':len(valid),'effort_ratio_max':max([v['effort_ratio']for _,v in valid]or[0.]),'joint_torque_abs_Nm_max':max([abs(v['joint_torque_Nm'])for _,v in valid]or[0.]),'frames_exceeding_actual_motor_capacity':[int(ids[i])for i,v in valid if v['effort_ratio']>1.]}
    load_summary_all=load_summary(np.ones(350,bool))
    knee_helper=Path(__file__).with_name('rgb145_knee_load.py')
    # Each parent's original arrays remain separate, avoiding stale FK/goals from either parent.
    a.output.parent.mkdir(parents=True,exist_ok=True)
    collision_passed=bool(penetration.max()<=.02);joint_limit_passed=bool(limit.max()<=.05);passed=collision_passed and joint_limit_passed
    physical_limit_metadata={'model_path':str(a.model.resolve()),'model_sha256':sha(a.model),'joint_names':names,'limited':limited.tolist(),'lower_rad':ranges[:,0].tolist(),'upper_rad':ranges[:,1].tolist(),'allowed_violation_rad':.05,'model_limits_modified':False,'joint_values_clipped':False,'precheck_rule':'Validate reference coordinates against the same effective runtime model; collision clearance alone does not pass static geometry'}
    np.savez_compressed(a.output,qpos=out,dof_names=np.asarray(names),time_s=time.copy(),source_time_s=source.copy(),frame_ids=ids.copy(),foot_support_expected=expected,foot_support_geom_name=surface,foot_support_confidence=confidence,foot_actual_ankle_world=ankles,foot_target_ankle_world=ankles.copy(),reference_com_world=com,root_target_world=out[:,:3].copy(),hybrid_reference_version=np.asarray('v17c_hybrid_recomputed_diagnostics'),parent_prefix_sha256=np.asarray(sha(a.prefix)),parent_suffix_sha256=np.asarray(sha(a.suffix)),hybrid_suffix_weight=weight,stance_knee_load_diagnostic_only=np.asarray(True),stance_knee_load_valid=np.asarray([v is not None for v in load_rows]),stance_knee_load_side=np.asarray([v['side']if v is not None else''for v in load_rows]),stance_knee_joint_torque_Nm=np.asarray([v['joint_torque_Nm']if v is not None else 0. for v in load_rows]),stance_knee_actuator_force=np.asarray([v['actuator_force']if v is not None else 0. for v in load_rows]),stance_knee_effort_ratio=np.asarray([v['effort_ratio']if v is not None else 0. for v in load_rows]),stance_knee_original_actuator_force_range=np.asarray([v['original_actuator_force_range']if v is not None else[0.,0.]for v in load_rows]),stance_knee_original_joint_torque_range_Nm=np.asarray([v['original_joint_torque_range_Nm']if v is not None else[0.,0.]for v in load_rows]),stance_knee_load_target_ratio=np.asarray(a.knee_load_target_ratio),stance_knee_load_cost_weight=np.asarray(0.),knee_load_helper_sha256=np.asarray(sha(knee_helper)),hybrid_v12_weight=weight,parent_v2_sha256=np.asarray(sha(a.prefix)),parent_v12_sha256=np.asarray(sha(a.suffix)),parent_v2_qpos=qa,parent_v12_qpos=qb,static_penetration_m=penetration,static_joint_limit_violation_rad=limit,static_collision_passed=np.asarray(collision_passed),static_physical_joint_limits_passed=np.asarray(joint_limit_passed),static_geometry_passed=np.asarray(passed),physical_joint_limit_model_sha256=np.asarray(sha(a.model)),physical_joint_names=np.asarray(names),physical_joint_limited=limited,physical_joint_lower_rad=ranges[:,0],physical_joint_upper_rad=ranges[:,1],physical_joint_violation_tolerance_rad=np.asarray(.05),wrist_reference_joint_names=np.asarray([names[i]for i in wrists]),wrist_reference_joint_targets_rad=out[0,7+np.asarray(wrists)],**raw,**parent_suffix_fields)
    report={'schema':'rgb145.hybrid_prefix_reference.v1','version':'v17c_hybrid_recomputed_diagnostics','status':'static_geometry_passed_pending_dynamic_validation'if passed else'static_geometry_failed_review_physical_constraints','physics_passed':False,'physics_steps':0,'active_simulation_pose_writes':0,'model_parameter_mutations':0,'root_height_repair_applied':False,'frames':350,'source_frame_ids_unchanged':True,'source_time_unchanged':True,'blend':{'last_exact_v2_root_body22_frame':a.blend_start_frame,'first_exact_v12_root_body22_frame':a.blend_end_frame,'start_source_time_s':float(source[a.blend_start_frame]),'end_source_time_s':float(source[a.blend_end_frame]),'weight':'u^3*(10-15u+6u^2)','root_orientation':'shortest-path SLERP; quaternion hemisphere continuity preserves rotations','quaternion_hemisphere_sign_changes':sign_changes,'auxiliary24':'Fixed to v12 first frame across the entire sequence','wrist6':wrist_policy,'relaxed_wrist_targets_rad':{names[i]:float(out[0,7+i])for i in wrists},'support_plan':'v2 in prefix; v12 in suffix; same-surface expected intersection and minimum confidence in blend'},'max_static_penetration_m':float(penetration.max()),'max_static_penetration_by_region_m':{label:float(penetration[mask].max())if mask.any()else None for label,mask in [('prefix',prefix_mask),('blend',blend_mask),('suffix',suffix_mask)]},'frames_over20mm':np.flatnonzero(penetration>.02).tolist(),'joint_limit_violation_rad_max':float(limit.max()),'static_collision_threshold_m':.02,'static_collision_threshold_passed':collision_passed,'physical_joint_limit_threshold_rad':.05,'physical_joint_limit_threshold_passed':joint_limit_passed,'failed_static_checks':([name for name,ok in [('collision_clearance',collision_passed),('physical_joint_limits',joint_limit_passed)]if not ok]),'physical_joint_limit_metadata':physical_limit_metadata,'parent_physical_limit_audit':parent_limit_audit,'fk':'Actual model ankle frames and robot subtree COM recomputed for every hybrid pose; no parent FK arrays reused','source_raw_fields_copied_without_modification':sorted(raw),'input':{'prefix_path':str(a.prefix.resolve()),'prefix_sha256':sha(a.prefix),'suffix_path':str(a.suffix.resolve()),'suffix_sha256':sha(a.suffix),'effective_model_path':str(a.model.resolve()),'effective_model_sha256':sha(a.model)},'reference_sha256':sha(a.output),'generator_sha256':sha(__file__),'limitations':['Static collision clearance does not establish dynamic feasibility.','Known prefix is preserved for root and22 non-wrist body joints; all6 wrists and24 auxiliary joints intentionally use the v12 relaxed pose.','If a blend exceeds20mm penetration, inspect the affected frames or choose another fusion window; this generator does not lift the root to hide it.'],'rows':rows}
    report['hybrid_knee_load_proxy']={'recomputed_from_hybrid_qpos_and_hybrid_support_plan':True,'inherited_suffix_metrics_used':False,'source_time_eligibility':'Exactly one supporting foot and source time<9s; zero/invalid during seated or double-support phases','static_friction_assistance_used':False,'target_effort_ratio':a.knee_load_target_ratio,'hybrid_optimization_cost_applied':False,'all_frames':load_summary_all,'by_region':{label:load_summary(mask)for label,mask in [('prefix',prefix_mask),('blend',blend_mask),('suffix',suffix_mask)]},'dynamic_feasibility_claimed':False,'helper_sha256':sha(knee_helper)}
    report['metadata_provenance']={'parent_suffix_namespace_fields':sorted(parent_suffix_fields),'parent_suffix_fields_are_not_hybrid_metrics':True,'hybrid_pose_dependent_fields_recomputed':['foot_actual_ankle_world','foot_target_ankle_world (FK-derived tracking target)','reference_com_world','static_penetration_m','static_joint_limit_violation_rad','stance_knee_*'],'stale_suffix_optimizer_fields_not_reused':['ankle_height_error_m','root_z_at_lower_bound','root_z_lower_bound_m','swing_projection_*','original stance_knee_*'],'source_raw_fields_preserved':sorted(raw),'legacy_parent_role_aliases':{'v2':'prefix input','v12':'suffix input; compatibility labels only, actual parent path/SHA is authoritative'},'hybrid_qpos_and_contact_generation_rules_changed':False}
    a.output.with_suffix('.knee_load.py').write_bytes(knee_helper.read_bytes())
    a.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');a.output.with_suffix('.generator.py').write_bytes(Path(__file__).read_bytes());print(json.dumps({k:v for k,v in report.items()if k!='rows'}),flush=True)
    return 0 if passed else 3


if __name__=='__main__':raise SystemExit(main())
