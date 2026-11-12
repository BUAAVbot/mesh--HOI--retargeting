"""Independently replay a frozen full-sequence trial and audit six free objects.

Replay validity, complete source coverage and task quality are separate results.
This audit can inspect a one-object diagnostic without calling it a six-object
or 923-frame success. It never changes the frozen trial or its quality report.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trial-dir',type=Path,required=True)
    p.add_argument('--output',type=Path)
    p.add_argument('--require-six-objects',action='store_true')
    p.add_argument('--require-full-coverage',action='store_true')
    a=p.parse_args()
    import numpy as np
    import mujoco as mj
    trial=a.trial_dir.resolve();r=read(trial/'report.json')
    model_path=Path(r['model']['path']);reference_path=Path(r['reference']['path'])
    trace_path=trial/'physical_trace.npz';contact_path=trial/'physical_contacts.npz'
    snapshot=trial/'source_fourth_full_physics.py'
    checksums=[]
    frozen_sources=[]
    sonic=r.get('sonic',{})
    if sonic.get('controller_source'):
        frozen_sources.append(sonic['controller_source'])
    if sonic.get('runtime_config_source'):
        frozen_sources.append(sonic['runtime_config_source'])
    frozen_sources.extend(sonic.get('helper_sources',[]))
    for path,receipt in [(model_path,r['model']),(reference_path,r['reference']),
                         (trace_path,r['artifacts']['trace']),(contact_path,r['artifacts']['contacts']),
                         (snapshot,r['script'])]+[(Path(item['path']),item)for item in frozen_sources]:
        actual=sha(path)
        if actual!=receipt['sha256']:raise ValueError('Frozen file SHA mismatch: '+str(path))
        checksums.append({'path':str(path),'sha256':actual})
    m=(mj.MjModel.from_binary_path(str(model_path)) if model_path.suffix.lower()=='.mjb'
       else mj.MjModel.from_xml_path(str(model_path)))
    d=mj.MjData(m)
    z=np.load(trace_path,allow_pickle=False)
    with np.load(contact_path,allow_pickle=False) as compact:
        c={key:compact[key] for key in compact.files}
    ref=np.load(reference_path,allow_pickle=False)
    q,v,t,u=z['qpos'],z['qvel'],z['time'],z['ctrl']
    steps=len(u);dt=float(m.opt.timestep)
    if q.shape!=(steps+1,m.nq)or v.shape!=(steps+1,m.nv)or t.shape!=(steps+1,)or u.shape!=(steps,m.nu):
        raise ValueError('Frozen state/control dimensions disagree with model')
    if not steps or not all(np.isfinite(x).all()for x in [q,v,t,u])or np.any(np.diff(t)<=0):
        raise ValueError('Empty, nonfinite or nonmonotonic physical trace')
    dt_error=float(np.max(np.abs(np.diff(t)-dt)))
    if dt_error>1e-8:raise ValueError('Trace timeline differs from model timestep')
    external={key:float(np.max(np.abs(z[key])))for key in ['qfrc_applied','xfrc_applied']}
    if any(value!=0 for value in external.values()):raise ValueError('External force exists in frozen trace')
    hand_mimic=[]
    for eq in range(m.neq):
        if m.eq_type[eq]!=mj.mjtEq.mjEQ_JOINT:continue
        child,parent=int(m.eq_obj1id[eq]),int(m.eq_obj2id[eq])
        name=mj.mj_id2name(m,mj.mjtObj.mjOBJ_JOINT,child)or''
        if parent<0 or not name.startswith(('lh_','rh_')):continue
        cq,pq=int(m.jnt_qposadr[child]),int(m.jnt_qposadr[parent])
        coefficient=m.eq_data[eq,:5]
        residual=q[:,cq]-m.qpos0[cq]-np.polynomial.polynomial.polyval(q[:,pq]-m.qpos0[pq],coefficient)
        aids=np.where(m.actuator_trnid[:,0]==child)[0]
        disabled=bool(np.all(m.actuator_gainprm[aids]==0)and np.all(m.actuator_biasprm[aids]==0)and np.all(m.actuator_gear[aids]==0))
        force_max=float(np.max(np.abs(z['actuator_force'][:,aids])))if len(aids)else 0.
        hand_mimic.append({'child_joint':name,'parent_joint':mj.mj_id2name(m,mj.mjtObj.mjOBJ_JOINT,parent),
            'polynomial':coefficient.tolist(),'constraint_active':bool(m.eq_active0[eq]),
            'actual_constraint_error_max_rad':float(np.max(np.abs(residual))),
            'actual_constraint_error_p95_rad':float(np.percentile(np.abs(residual),95)),
            'passive_actuator_ids':aids.tolist(),'passive_gain_bias_gear_all_zero':disabled,
            'passive_actuator_force_max_abs_Nm':force_max})
    mimic_contract=all(item['constraint_active']and item['passive_gain_bias_gear_all_zero']and
                       item['passive_actuator_force_max_abs_Nm']==0 for item in hand_mimic)
    ids=np.asarray(z['object_ids'],int).reshape(-1)
    oqa=np.asarray(z['object_qadr'],int).reshape(-1);ova=np.asarray(z['object_vadr'],int).reshape(-1)
    bodies=[];objects=[]
    for i,oid in enumerate(ids):
        name=f'target_object_{oid:02d}'
        body=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,name)
        joint=mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,name+'_free')
        if min(body,joint)<0:raise ValueError('Missing target object '+name)
        free=bool(m.jnt_type[joint]==mj.mjtJoint.mjJNT_FREE)
        mocap=bool(m.body_mocapid[body]>=0)
        actuated=bool(any(int(j)==joint for j in m.actuator_trnid[:,0]))
        if int(m.jnt_qposadr[joint])!=int(oqa[i])or int(m.jnt_dofadr[joint])!=int(ova[i]):
            raise ValueError('Trace object address differs from model')
        bodies.append(body)
        objects.append({'object_id':int(oid),'body':name,'body_id':body,'joint_id':joint,
                        'qpos_address':int(oqa[i]),'qvel_address':int(ova[i]),
                        'dynamic_freejoint':free,'mocap':mocap,'actuated':actuated})
    integration=int(np.asarray(z['state_spec']).reshape(-1)[0])
    mj.mj_setState(m,d,z['initial_state'],integration);mj.mj_forward(m,d)
    qerr=float(np.max(np.abs(d.qpos-q[0])));verr=float(np.max(np.abs(d.qvel-v[0])))
    ci=np.asarray(c['interval_index'],int)
    if np.any(np.diff(ci)<0)or(len(ci)and(ci[0]<0 or ci[-1]>=steps)):
        raise ValueError('Invalid compact contact interval indices')
    body_names=[mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,b)or'world'for b in range(m.nbody)]
    hand=np.array([name.startswith(('lh_','rh_'))for name in body_names])
    geom_names=[mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,g)or str(g)for g in range(m.ngeom)]
    object_lookup={int(body):i for i,body in enumerate(bodies)}
    robot_geoms=[g for g in range(m.ngeom)if int(m.geom_bodyid[g])>0 and int(m.geom_bodyid[g])not in object_lookup and(m.geom_contype[g]or m.geom_conaffinity[g])]
    obj_geoms=[g for g in range(m.ngeom)if int(m.geom_bodyid[g])in object_lookup and(m.geom_contype[g]or m.geom_conaffinity[g])]
    world_geoms=[g for g in range(m.ngeom)if int(m.geom_bodyid[g])==0 and(m.geom_contype[g]or m.geom_conaffinity[g])]
    def allowed(g1,g2):
        return bool((int(m.geom_contype[g1])&int(m.geom_conaffinity[g2]))or(int(m.geom_contype[g2])&int(m.geom_conaffinity[g1])))
    collision_config={'model_disableflags':int(m.opt.disableflags),'gravity_m_s2':m.opt.gravity.tolist(),
        'robot_self_contact_bitmask_allowed_geom_pairs':int(sum(allowed(g,h)for i,g in enumerate(robot_geoms)for h in robot_geoms[i+1:])),
        'robot_object_all_pairs_bitmask_enabled':bool(all(allowed(g,h)for g in robot_geoms for h in obj_geoms)),
        'robot_world_all_pairs_bitmask_enabled':bool(all(allowed(g,h)for g in robot_geoms for h in world_geoms)),
        'object_world_all_pairs_bitmask_enabled':bool(all(allowed(g,h)for g in obj_geoms for h in world_geoms)),
        'object_object_all_pairs_bitmask_enabled':bool(all(allowed(g,h)for i,g in enumerate(obj_geoms)for h in obj_geoms[i+1:]if m.geom_bodyid[g]!=m.geom_bodyid[h])),
        'note':'Bitmask allowance is distinct from parent-child filtering and explicit contact exclusions.'}
    force_error=0.;position_error=0.;distance_error=0.;contact_mismatch=0;pair_mismatch=0
    force=np.zeros((steps,len(ids)));table=np.zeros((steps,len(ids)),bool)
    maxpen=0.;objectpair_counts={}
    for step,ctrl in enumerate(u):
        d.ctrl[:]=ctrl;d.qfrc_applied[:]=0;d.xfrc_applied[:]=0
        mj.mj_step(m,d)
        qerr=max(qerr,float(np.max(np.abs(d.qpos-q[step+1]))))
        verr=max(verr,float(np.max(np.abs(d.qvel-v[step+1]))))
        left,right=np.searchsorted(ci,[step,step+1])
        contact_mismatch+=abs((right-left)-d.ncon)
        for j in range(d.ncon):
            contact=d.contact[j];g=np.asarray([contact.geom1,contact.geom2],int)
            b=m.geom_bodyid[g];wrench=np.zeros(6);mj.mj_contactForce(m,d,j,wrench)
            frame=np.asarray(contact.frame).reshape(3,3)
            fw=frame.T@wrench[:3];tw=frame.T@wrench[3:]
            maxpen=max(maxpen,-float(contact.dist))
            index=left+j
            if index<right:
                pair_mismatch+=int(not np.array_equal(g,c['geom_ids'][index])or not np.array_equal(b,c['body_ids'][index]))
                force_error=max(force_error,float(np.max(np.abs(fw-c['force_world_on_geom2_N'][index]))),
                                float(np.max(np.abs(tw-c['torque_world_on_geom2_Nm'][index]))))
                position_error=max(position_error,float(np.max(np.abs(contact.pos-c['position_world_m'][index]))))
                distance_error=max(distance_error,abs(float(contact.dist)-float(c['signed_distance_m'][index])))
            for bb in b:
                if int(bb) in object_lookup:
                    oi=object_lookup[int(bb)]
                    if hand[b[0]]or hand[b[1]]:force[step,oi]+=max(float(wrench[0]),0)
                    if any(geom_names[int(gg)].startswith('table_support')for gg in g):table[step,oi]=True
            if any(int(bb)in object_lookup for bb in b):
                pair='|'.join(sorted(geom_names[int(gg)]for gg in g))
                objectpair_counts[pair]=objectpair_counts.get(pair,0)+1
    source_time=np.asarray(ref['time_s'],float);source_time-=source_time[0]
    object_ref=np.asarray(ref['object_target_pose'],float)
    if object_ref.ndim==2:object_ref=object_ref[:,None,:]
    observed=np.asarray(ref.get('object_observed_mask',np.ones((len(source_time),len(ids)),bool)),bool)
    if object_ref.shape!=(len(source_time),len(ids),7)or observed.shape!=(len(source_time),len(ids)):
        raise ValueError('Reference object/observed dimensions mismatch')
    coverage=source_time<=t[-1]+1e-8
    nearest=np.minimum(np.searchsorted(t,source_time),len(t)-1)
    desired_pose=np.stack([q[nearest,int(k):int(k)+7]for k in oqa],axis=1)
    observed_errors=np.linalg.norm(desired_pose[:,:,:3]-object_ref[:,:,:3],axis=2)
    for i,item in enumerate(objects):
        selected=coverage&observed[:,i];err=observed_errors[selected,i]
        item.update(source_observed_frames_total=int(observed[:,i].sum()),
                    source_observed_frames_physically_covered=int(selected.sum()),
                    source_imputed_frames_total=int((~observed[:,i]).sum()),
                    covered_observed_position_error_max_m=float(err.max())if len(err)else None,
                    covered_observed_position_error_rms_m=float(np.sqrt(np.mean(err**2)))if len(err)else None,
                    actual_displacement_m=float(np.linalg.norm(q[-1,oqa[i]:oqa[i]+3]-q[0,oqa[i]:oqa[i]+3])),
                    final_linear_speed_m_s=float(np.linalg.norm(v[-1,ova[i]:ova[i]+3])),
                    hand_contact_duration_s=float((force[:,i]>.01).sum()*dt),
                    hand_normal_impulse_Ns=float(force[:,i].sum()*dt),
                    final_0p15s_table_support_fraction=float(table[-max(1,int(.15/dt)):,i].mean()))
    rootid=mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,'vita_demo_free_root');rq=int(m.jnt_qposadr[rootid])
    qref=ref['qpos'];rootdesired=np.column_stack([np.interp(t[1:],source_time,qref[:,j])for j in range(3)])
    rooterr=q[1:,rq:rq+3]-rootdesired
    rootup=1-2*(q[1:,rq+4]**2+q[1:,rq+5]**2)
    all923=bool(len(source_time)==923 and coverage.all())
    six=bool(len(ids)==6 and set(ids.tolist())==set(range(1,7)))
    replay=bool(qerr<1e-7 and verr<1e-6 and contact_mismatch==0 and pair_mismatch==0 and force_error<1e-6 and position_error<1e-7 and distance_error<1e-8
                and all(item['dynamic_freejoint']and not item['mocap']and not item['actuated']for item in objects))
    source_frame_ids=np.asarray(ref.get('frame_ids',np.arange(len(source_time))),int)
    result={'schema':'fourth.full_sequence_independent_audit.v1',
            'status':'deterministic_full_physics_trace_verified'if replay else'audit_failed',
            'replay_verified':replay,'audit_proves_full_task_quality':False,
            'frozen_files':checksums,'mujoco_version':mj.__version__,'physics_steps':steps,
            'actual_duration_s':float(t[-1]-t[0]),'physics_dt_s':dt,'dt_max_error_s':dt_error,
            'qpos_replay_max_abs_error':qerr,'qvel_replay_max_abs_error':verr,
            'contact_record_count_mismatch':int(contact_mismatch),'contact_pair_mismatch':int(pair_mismatch),
            'contact_force_torque_max_abs_error':force_error,'contact_position_max_abs_error_m':position_error,
            'contact_signed_distance_max_abs_error_m':distance_error,'external_forces_max_abs':external,
            'collision_configuration':collision_config,
            'robot_hand_mimic_constraints':hand_mimic,
            'robot_hand_mimic_passive_actuation_contract_passed':mimic_contract,
            'six_dynamic_objects':six,'all_923_source_frames_covered':all923,
            'source_reference_frames':len(source_time),'source_frames_physically_covered':int(coverage.sum()),
            'covered_frame_id_range':[int(source_frame_ids[coverage][0]),int(source_frame_ids[coverage][-1])],
            'source_duration_s':float(source_time[-1]),'objects':objects,
            'replayed_quality_diagnostics':{'max_penetration_m':maxpen,
                'root_horizontal_error_max_m':float(np.linalg.norm(rooterr[:,:2],axis=1).max()),
                'root_vertical_error_min_m':float(rooterr[:,2].min()),'root_up_z_min':float(rootup.min()),
                'joint_rms_median_rad':float(np.median(z['joint_rms_rad']))},
            'original_task_quality_report_passed':bool(r.get('simulation_passed')),
            'original_quality_checks':r.get('checks',{}),
            'all_full_sequence_requirements_passed':bool(replay and six and all923 and r.get('simulation_passed')and mimic_contract),
            'object_contact_pair_point_counts':objectpair_counts,
            'limitations':['Deterministic replay is distinct from task tracking quality and measured physical accuracy.',
                           'Missing observations remain imputed; only covered observed source frames enter per-object observed errors.']}
    out=a.output or trial/'independent_replay_audit.json';out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k]for k in ['status','physics_steps','actual_duration_s','qpos_replay_max_abs_error','qvel_replay_max_abs_error','contact_force_torque_max_abs_error','six_dynamic_objects','all_923_source_frames_covered','all_full_sequence_requirements_passed','replayed_quality_diagnostics']}),flush=True)
    if not replay or not mimic_contract or(a.require_six_objects and not six)or(a.require_full_coverage and not all923):raise SystemExit(2)


if __name__=='__main__':main()
