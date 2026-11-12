"""Continuous, unforced MuJoCo rollout of a full reference and six free objects.

Reference poses are assigned only once at reset. All active motion is produced
by joint actuators. Missing observations and imputed targets remain labelled.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def receipt(path):
    p = Path(path).resolve()
    return dict(path=str(p), sha256=sha(p), bytes=p.stat().st_size)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--controller', choices=['pd', 'sonic'], default='pd')
    p.add_argument('--sonic-config', type=Path)
    p.add_argument('--seconds', type=float)
    p.add_argument('--hold-seconds', type=float, default=0.)
    p.add_argument('--ankle-offset', type=float, default=-.15)
    p.add_argument('--hip-offset', type=float, default=.05)
    p.add_argument('--gravity-compensation', type=float, default=0.)
    p.add_argument('--hand-world-gain', type=float, default=0.)
    p.add_argument('--balance-gain', type=float, default=0.)
    p.add_argument('--control-hz', type=float, default=50.)
    p.add_argument('--initialize-reference-velocity',action='store_true')
    a = p.parse_args()
    import os
    if os.name != 'posix':
        p.error('Simulation is server-only; local machine previews and verifies outputs')
    import numpy as np
    import mujoco as mj
    if a.output_dir.exists():
        raise ValueError('Use a fresh trial directory')
    a.output_dir.mkdir(parents=True)
    script_snapshot=a.output_dir/'source_fourth_full_physics.py'
    script_snapshot.write_bytes(Path(__file__).read_bytes())
    source = {k:v for k,v in np.load(a.reference, allow_pickle=False).items()}
    qref = source['qpos'].astype(float)
    t = source['time_s'].astype(float)
    t = t-t[0]
    objref = source['object_target_pose'].astype(float)
    if objref.ndim == 2:
        objref = objref[:,None,:]
    object_ids = np.asarray(source.get('object_ids', [int(source.get('object_id', 5))]),dtype=int).reshape(-1)
    names = source['dof_names'].astype(str).tolist()
    nobj = len(object_ids)
    if qref.shape != (len(t),59) or objref.shape != (len(t),nobj,7) or len(names) != 52:
        raise ValueError('Invalid full-reference dimensions')
    if not np.isfinite(qref).all() or not np.isfinite(objref).all() or np.any(np.diff(t)<=0):
        raise ValueError('Reference must be finite and strictly timed')
    model = mj.MjModel.from_binary_path(str(a.model.resolve())) if a.model.suffix=='.mjb' else mj.MjModel.from_xml_path(str(a.model.resolve()))
    sonic=None
    helper_snapshots=[]
    effective_model=a.model
    if a.controller=='sonic':
        from fourth_sonic_controller import Controller
        config=json.loads(a.sonic_config.read_text()) if a.sonic_config else {}
        config_snapshot=a.output_dir/'source_sonic_config.json'
        config_snapshot.write_text(json.dumps(config,indent=2,sort_keys=True)+'\n',encoding='utf-8')
        sonic=Controller(model=model,reference=source,config=deepcopy(config))
        controller_snapshot=a.output_dir/'source_fourth_sonic_controller.py'
        controller_snapshot.write_bytes(Path(__file__).with_name('fourth_sonic_controller.py').read_bytes())
        if config.get('hand_object_feedback',False):
            helper_names=['fourth_arm_object_feedback.py','fourth_contact_feedback.py']
            guard=config.get('finger_feedback',{}).get('table_clearance_guard',{})
            if isinstance(guard,dict)and guard.get('enabled',False):
                helper_names.append('fourth_contact_feedback_tableguard.py')
            backend=config.get('finger_feedback',{}).get('geometry_backend')
            if backend in ('convex_sat','convex_sat_o6'):
                if 'fourth_contact_feedback_tableguard.py'not in helper_names:
                    helper_names.append('fourth_contact_feedback_tableguard.py')
                helper_names.extend(['fourth_contact_feedback_segmented.py',
                                     'fourth_convex_separation.py'])
            if backend=='convex_sat_o6':
                helper_names.append('fourth_contact_feedback_o6_coupled.py')
            for name in helper_names:
                target_snapshot=a.output_dir/('source_'+name)
                target_snapshot.write_bytes(Path(__file__).with_name(name).read_bytes())
                helper_snapshots.append(receipt(target_snapshot))
            finger_contract=config.get('finger_feedback',{}).get('finger_urdf')
            if finger_contract:
                contract_snapshot=a.output_dir/'source_finger_contract.urdf'
                contract_snapshot.write_bytes(Path(finger_contract).read_bytes())
                helper_snapshots.append(receipt(contract_snapshot))
        # A binary snapshot preserves all numeric model changes exactly, with
        # no XML decimal precision loss. It is the audited plant receipt.
        effective_model=a.output_dir/'effective_model.mjb'
        mj.mj_saveModel(model,str(effective_model),None)
        mj.mj_saveLastXML(str(a.output_dir/'effective_model.xml'),model)
    if model.opt.disableflags or model.opt.gravity[2] > -9.:
        raise ValueError('Real gravity and active contacts are required')
    def jid(n): return mj.mj_name2id(model,mj.mjtObj.mjOBJ_JOINT,n)
    def bid(n): return mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,n)
    joints=np.asarray([jid(n) for n in names]); aids=np.asarray([mj.mj_name2id(model,mj.mjtObj.mjOBJ_ACTUATOR,'motor_'+n) for n in names])
    if min(joints)<0 or min(aids)<0 or len(set(aids))!=52:
        raise ValueError('Incomplete actuator mapping')
    qa=model.jnt_qposadr[joints]; va=model.jnt_dofadr[joints]
    root=jid('vita_demo_free_root'); rootqa=int(model.jnt_qposadr[root]); rootbody=int(model.jnt_bodyid[root])
    objb=np.asarray([bid(f'target_object_{i:02d}')for i in object_ids])
    objj=np.asarray([jid(f'target_object_{i:02d}_free')for i in object_ids])
    if min(objb)<0 or min(objj)<0 or np.any(model.jnt_type[objj]!=mj.mjtJoint.mjJNT_FREE) or np.any(model.body_mocapid[objb]>=0):
        raise ValueError('All six target objects must be dynamic free bodies')
    if any(j in model.actuator_trnid[:,0] for j in objj):
        raise ValueError('Objects cannot have actuators')
    oqa=model.jnt_qposadr[objj]; ova=model.jnt_dofadr[objj]
    dat=mj.MjData(model); kin=mj.MjData(model)
    dat.qpos[rootqa:rootqa+7]=qref[0,:7]; dat.qpos[qa]=qref[0,7:]
    for j,k in enumerate(oqa): dat.qpos[k:k+7]=objref[0,j]
    if a.initialize_reference_velocity:
        nextq=dat.qpos.copy();nextq[rootqa:rootqa+7]=qref[1,:7];nextq[qa]=qref[1,7:]
        for j,k in enumerate(oqa):nextq[k:k+7]=objref[1,j]
        mj.mj_differentiatePos(model,dat.qvel,float(t[1]-t[0]),dat.qpos,nextq)
    mj.mj_forward(model,dat)
    spec=int(mj.mjtState.mjSTATE_INTEGRATION); initial=np.empty(mj.mj_stateSize(model,spec));mj.mj_getState(model,dat,initial,spec)
    dt=float(model.opt.timestep); end=float(t[-1]+a.hold_seconds)
    if a.seconds is not None: end=min(end,a.seconds)
    steps=int(np.ceil(end/dt)); control_stride=max(1,round(1/(a.control_hz*dt)))
    offset=np.zeros(52)
    for k,n in enumerate(names):
        if 'ankle_pitch' in n:offset[k]=a.ankle_offset
        if 'hip_pitch' in n:offset[k]=a.hip_offset
    arms=[]
    for side,hand in [('left','lh_hand_base_link'),('right','rh_hand_base_link')]:
        select=np.asarray([k for k,n in enumerate(names)if n.startswith((side+'_shoulder',side+'_elbow',side+'_wrist'))])
        arms.append((bid(hand),select))
    if sonic is not None:
        sonic.reset(dat)
    qseq=[dat.qpos.copy()];vseq=[dat.qvel.copy()];tseq=[0.];ctrls=[];torques=[];qforces=[];xforces=[]
    errors=[];rooterrors=[];upseq=[];pens=[];handforces=[];tablecontacts=[]
    cs=[];cg=[];cb=[];cp=[];cf=[];ct=[];cn=[];cd=[]
    body_names=[mj.mj_id2name(model,mj.mjtObj.mjOBJ_BODY,b)or'world'for b in range(model.nbody)]
    geom_names=[mj.mj_id2name(model,mj.mjtObj.mjOBJ_GEOM,g)or str(g)for g in range(model.ngeom)]
    handbody=np.asarray([n.startswith(('lh_','rh_'))for n in body_names])
    objlookup={int(b):j for j,b in enumerate(objb)}
    target=qref[0,7:].copy();controller_target=target.copy();start=time.monotonic();warn0=np.array([v.number for v in dat.warning]);reason='requested_duration'
    jac=np.zeros((3,model.nv));jacr=np.zeros_like(jac);failure=None
    for step in range(steps):
        now=float(dat.time)
        if sonic is not None:
            # Delay history advances at every physical step, while the
            # controller performs network inference at its metadata rate.
            controller_target=np.asarray(sonic.action(dat,now),dtype=float)
            if controller_target.shape!=(model.nu,):raise ValueError('SONIC must return native actuator ctrl vector')
        if step%control_stride==0:
            target=np.asarray([np.interp(now,t,qref[:,7+j])for j in range(52)])
            if sonic is None:
                command=target+offset
                if a.gravity_compensation:
                    command+=np.clip(a.gravity_compensation*dat.qfrc_bias[va]/model.actuator_gainprm[aids,0],-.25,.25)
                if a.balance_gain:
                    feet=[bid('left_ankle_roll_link'),bid('right_ankle_roll_link')]
                    support=(dat.xpos[feet[0]]+dat.xpos[feet[1]])/2
                    com=dat.subtree_com[rootbody]
                    rot=dat.xmat[rootbody].reshape(3,3)
                    err=rot[:2,:2].T@(com[:2]-support[:2])
                    velocity=rot[:2,:2].T@dat.qvel[:2]
                    correction=np.clip(a.balance_gain*(err+.18*velocity),-.25,.25)
                    for j,n in enumerate(names):
                        if 'ankle_pitch' in n: command[j]-=correction[0]
                        if 'ankle_roll' in n: command[j]+=correction[1]
                if a.hand_world_gain:
                    kin.qpos[:]=dat.qpos
                    kin.qpos[:7]=np.r_[[np.interp(now,t,qref[:,j])for j in range(3)],qref[min(np.searchsorted(t,now),len(t)-1),3:7]]
                    kin.qpos[qa]=target;mj.mj_forward(model,kin)
                    for hb,select in arms:
                        mj.mj_jacBody(model,dat,jac,jacr,hb)
                        j=jac[:,va[select]];e=kin.xpos[hb]-dat.xpos[hb]
                        correction=j.T@np.linalg.solve(j@j.T+.003*np.eye(3),e)
                        command[select]+=np.clip(a.hand_world_gain*correction,-.2,.2)
                controller_target=dat.ctrl.copy();controller_target[aids]=command
        dat.ctrl[:]=controller_target
        if np.any(dat.qfrc_applied)or np.any(dat.xfrc_applied):raise ValueError('Unexpected applied force')
        ctrls.append(dat.ctrl.copy());qforces.append(dat.qfrc_applied.copy());xforces.append(dat.xfrc_applied.copy())
        mj.mj_step(model,dat)
        warnings=np.array([v.number for v in dat.warning])
        if not np.isfinite(dat.qpos).all()or not np.isfinite(dat.qvel).all()or np.any(warnings>warn0)or abs(dat.time-(step+1)*dt)>1e-7:
            reason='solver_failure'
            failure={'time':np.asarray(dat.time),'expected_time':np.asarray((step+1)*dt),
                     'qpos':dat.qpos.copy(),'qvel':dat.qvel.copy(),'warning_counts':warnings}
            ctrls.pop();qforces.pop();xforces.pop()
            # Preserve a valid monotonic prefix. A simulator reset is never
            # appended as a new state of the purported continuous rollout.
            break
        qseq.append(dat.qpos.copy());vseq.append(dat.qvel.copy());tseq.append(float(dat.time));torques.append(dat.actuator_force.copy())
        errors.append(float(np.sqrt(np.mean((dat.qpos[qa]-target)**2))))
        desired_root=np.array([np.interp(dat.time,t,qref[:,j])for j in range(3)])
        rooterrors.append(dat.qpos[:3]-desired_root)
        upseq.append(float(dat.xmat[rootbody].reshape(3,3)[2,2]))
        penetration=0.;hf=np.zeros(nobj);tab=np.zeros(nobj,dtype=bool)
        for ci in range(dat.ncon):
            c=dat.contact[ci];g=[int(c.geom1),int(c.geom2)];b=[int(model.geom_bodyid[v])for v in g]
            wrench=np.zeros(6);mj.mj_contactForce(model,dat,ci,wrench);frame=np.asarray(c.frame).reshape(3,3)
            penetration=max(penetration,-float(c.dist))
            cs.append(step);cg.append(g);cb.append(b);cp.append(c.pos.copy());cf.append(frame.T@wrench[:3]);ct.append(frame.T@wrench[3:]);cn.append(max(wrench[0],0));cd.append(float(c.dist))
            for bb in b:
                if bb in objlookup:
                    oi=objlookup[bb]
                    if handbody[b[0]]or handbody[b[1]]:hf[oi]+=max(wrench[0],0)
                    if any(geom_names[gg].startswith('table_support')for gg in g):tab[oi]=True
        pens.append(penetration);handforces.append(hf);tablecontacts.append(tab)
    qseq=np.asarray(qseq);vseq=np.asarray(vseq);tseq=np.asarray(tseq);ctrls=np.asarray(ctrls);torques=np.asarray(torques)
    if failure is not None:np.savez_compressed(a.output_dir/'solver_failure_snapshot.npz',**failure)
    if not len(ctrls):raise RuntimeError('Solver failed at reset; see solver_failure_snapshot.npz')
    nearest=np.minimum(np.searchsorted(tseq,t),len(tseq)-1)
    frame_actual=np.stack([qseq[nearest,k:k+7]for k in oqa],axis=1)
    pose_errors=np.linalg.norm(frame_actual[:,:,:3]-objref[:,:,:3],axis=2)
    observed=np.asarray(source.get('object_observed_mask',np.ones((len(t),nobj),bool)),dtype=bool)
    covered=t<=tseq[-1]+1e-8
    stats=[]
    candidate=np.asarray(source.get('contact_candidate_observed',np.zeros((len(t),2,nobj),bool)),bool).any(axis=1)
    timed_index=np.clip(np.searchsorted(t,tseq[1:]),0,len(t)-1)
    for j,oid in enumerate(object_ids):
        e=pose_errors[observed[:,j]&covered,j]
        expected=candidate[:,j].copy()
        # Permit three source frames of temporal tolerance for noisy inferred
        # contact boundaries; do not count unrelated later object contacts.
        for shift in range(1,4):
            expected[shift:]|=candidate[:-shift,j];expected[:-shift]|=candidate[shift:,j]
        expected_step=expected[timed_index]
        force=np.asarray(handforces)[:,j]
        stats.append({'object_id':int(oid),'observed_frames':int(observed[:,j].sum()),'imputed_frames':int((~observed[:,j]).sum()),
            'observed_frames_physically_covered':int((observed[:,j]&covered).sum()),
            'observed_position_error_max_m':float(e.max())if len(e)else None,
            'observed_position_error_rms_m':float(np.sqrt(np.mean(e*e)))if len(e)else None,
            'all_filled_position_error_max_m':float(pose_errors[:,j].max()),
            'hand_contact_duration_s':float((np.asarray(handforces)[:,j]>.01).sum()*dt),
            'hand_normal_impulse_Ns':float(np.asarray(handforces)[:,j].sum()*dt),
            'source_contact_candidate_frames':int(candidate[:,j].sum()),
            'source_timed_hand_contact_duration_s':float(((force>.01)&expected_step).sum()*dt),
            'source_timed_hand_normal_impulse_Ns':float((force*expected_step).sum()*dt),
            'actual_displacement_m':float(np.linalg.norm(qseq[-1,oqa[j]:oqa[j]+3]-qseq[0,oqa[j]:oqa[j]+3])),
            'final_linear_speed_m_s':float(np.linalg.norm(vseq[-1,ova[j]:ova[j]+3]))})
    fullcovered=bool(len(t)==923 and tseq[-1]>=t[-1]-1e-8)
    checks={'completed_duration':reason=='requested_duration'and tseq[-1]>=end-1e-8,
            'all_923_source_frames_covered':fullcovered,'six_dynamic_objects':nobj==6,
            'joint_tracking':float(np.median(errors))<=.35,
            'no_fall':float(min(upseq))>=.5 and float(np.min(np.asarray(rooterrors)[:,2]))>=-.25,
            'root_horizontal_tracking':float(np.max(np.linalg.norm(np.asarray(rooterrors)[:,:2],axis=1)))<=.25,
            'penetration':float(max(pens))<=.02,
            'all_required_source_contacts':all(s['source_contact_candidate_frames']<3 or
                (s['source_timed_hand_contact_duration_s']>=.10 and s['source_timed_hand_normal_impulse_Ns']>=.5)for s in stats),
            'all_observed_object_trajectories':all(s['observed_position_error_max_m']is not None and s['observed_position_error_max_m']<=.12 for s in stats)}
    checks={key:bool(value)for key,value in checks.items()}
    trace=a.output_dir/'physical_trace.npz'
    np.savez_compressed(trace,time=tseq,qpos=qseq,qvel=vseq,ctrl=ctrls,actuator_force=torques,
        initial_state=initial,state_spec=np.asarray(spec),qfrc_applied=np.asarray(qforces),xfrc_applied=np.asarray(xforces),
        object_ids=object_ids,object_qadr=oqa,object_vadr=ova,object_body_ids=objb,joint_qadr=qa,joint_vadr=va,actuator_ids=aids,dof_names=np.asarray(names),
        joint_rms_rad=np.asarray(errors),root_position_error_m=np.asarray(rooterrors),max_penetration_m=np.asarray(pens),
        object_hand_normal_force_N=np.asarray(handforces),object_table_contact=np.asarray(tablecontacts),
        source_time_s=t,source_state_index=nearest,source_frame_physically_covered_mask=covered,
        source_frame_object_actual_pose=frame_actual,source_frame_object_position_error_m=pose_errors)
    contact=a.output_dir/'physical_contacts.npz'
    np.savez_compressed(contact,interval_index=np.asarray(cs,dtype=np.int64),geom_ids=np.asarray(cg,dtype=np.int32),body_ids=np.asarray(cb,dtype=np.int32),
        position_world_m=np.asarray(cp),force_world_on_geom2_N=np.asarray(cf),torque_world_on_geom2_Nm=np.asarray(ct),normal_force_N=np.asarray(cn),signed_distance_m=np.asarray(cd),
        body_names=np.asarray(body_names),geom_names=np.asarray(geom_names))
    report={'schema':'fourth.full_sequence_physics.v1','status':'passed_full_sequence_physics'if all(checks.values())else'failed_full_sequence_quality',
        'simulation_passed':all(checks.values()),'checks':checks,'controller':a.controller,'controller_config':vars(a).copy(),
        'model':receipt(effective_model),'input_model':receipt(a.model),'reference':receipt(a.reference),'script':receipt(script_snapshot),
        'source_frames':len(t),'source_duration_s':float(t[-1]),'actual_duration_s':float(tseq[-1]),'physics_steps':len(ctrls),'physics_dt_s':dt,
        'mujoco_version':mj.__version__,'numpy_version':np.__version__,
        'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'continuous_rollout':True,'active_state_writes':0,'active_external_force_writes':0,'object_mocap':False,'object_actuated':False,
        'object_metrics':stats,'joint_rms_median_rad':float(np.median(errors)),
        'root_horizontal_error_max_m':float(np.max(np.linalg.norm(np.asarray(rooterrors)[:,:2],axis=1))),
        'root_vertical_error_min_m':float(np.min(np.asarray(rooterrors)[:,2])),'root_up_vector_z_min':float(min(upseq)),
        'first_root_tilt_fall_time_s':float(tseq[1:][np.flatnonzero(np.asarray(upseq)<.5)[0]])if np.any(np.asarray(upseq)<.5)else None,
        'max_penetration_m':float(max(pens)),'end_reason':reason,'elapsed_wall_s':time.monotonic()-start,
        'quality_thresholds':{'observed_object_error_max_m':.12,'max_penetration_m':.02,'joint_rms_median_rad_max':.35,'root_horizontal_error_max_m':.25,
            'required_contact_candidate_frames_min':3,'source_timed_contact_duration_s_min':.10,'source_timed_hand_impulse_Ns_min':.5,'contact_timing_tolerance_source_frames':3},
        'artifacts':{'trace':receipt(trace),'contacts':receipt(contact)},
        'limitations':['Missing source observations are imputed and labelled, not measured ground truth.','Object physical parameters and collision proxies are assumptions.','Simulated forces are not sensor measurements.']}
    report['controller_config']={k:str(v)if isinstance(v,Path)else v for k,v in report['controller_config'].items()}
    if sonic is not None:
        report['sonic']=sonic.report()
        report['sonic']['controller_source']=receipt(controller_snapshot)
        report['sonic']['runtime_config_source']=receipt(config_snapshot)
        report['sonic']['helper_sources']=helper_snapshots
        if hasattr(sonic,'dump_history'):
            report['sonic']['contact_feedback_artifacts']=sonic.dump_history(a.output_dir)
    (a.output_dir/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k]for k in ['status','checks','actual_duration_s','object_metrics','max_penetration_m','root_horizontal_error_max_m']}),flush=True)
    return 0 if report['simulation_passed']else 3


if __name__=='__main__':raise SystemExit(main())
