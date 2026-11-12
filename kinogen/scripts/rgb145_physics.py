"""Single free-root Vita stair task using the actual saved 28-body SONIC actor.

After reset, only actuator ctrl is written. Hand/neck reference PD and O6
passive polynomial joint equalities extend the original body articulation.
No root helper, foot/arm IK, mocap, attachment, or external force is used.
"""
from __future__ import annotations
import argparse,hashlib,json,shutil,sys,time
from pathlib import Path
import numpy as np
import mujoco as mj
from rgb145_sonic_controller import Controller,freeze_effective_model

ROOT=Path(__file__).resolve().parents[1]

def receipt(path):
    path=Path(path).resolve();return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}

def write_json(path,payload):
    path=Path(path);temp=path.with_suffix(path.suffix+'.partial');temp.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n');temp.replace(path)

def ident(model,kind,name):
    i=mj.mj_name2id(model,kind,name)
    if i<0:raise ValueError('Missing named model element '+name)
    return i

def descendants(model,body):
    out=[]
    for b in range(model.nbody):
        parent=b
        while parent>0 and parent!=body:parent=int(model.body_parentid[parent])
        if parent==body:out.append(b)
    return set(out)

def coupling_reactions(model,data):
    result=np.zeros((model.neq,model.nv));weights=np.zeros(data.nefc)
    for eq in range(model.neq):
        weights[:]=0.;idx=np.flatnonzero((data.efc_type[:data.nefc]==int(mj.mjtConstraint.mjCNSTR_EQUALITY))&(data.efc_id[:data.nefc]==eq))
        weights[idx]=data.efc_force[idx];mj.mj_mulJacTVec(model,data,result[eq],weights)
    return result

def contacts(model,data,step,foot_bodies,support_ids):
    rows=[];support=np.zeros(2);by_geom=[{},{}];penetration=0.;seat_force=0.
    for i in range(data.ncon):
        c=data.contact[i];gs=[int(c.geom1),int(c.geom2)];bs=[int(model.geom_bodyid[g])for g in gs]
        local=np.zeros(6);mj.mj_contactForce(model,data,i,local);rotation=np.asarray(c.frame).reshape(3,3)
        force=rotation.T@local[:3];torque=rotation.T@local[3:];normal=rotation[0]
        rows.append((step,float(data.time-model.opt.timestep),gs,bs,np.asarray(c.pos).copy(),force.copy(),torque.copy(),float(local[0]),float(c.dist),normal.copy()))
        penetration=max(penetration,-float(c.dist))
        for k in range(2):
            body_name=mj.mj_id2name(model,mj.mjtObj.mjOBJ_BODY,bs[k])or''
            other_name=mj.mj_id2name(model,mj.mjtObj.mjOBJ_GEOM,gs[1-k])or''
            if other_name.startswith('seat_support_')and('hip_'in body_name or body_name=='base_link'):
                seat_force+=max(0.,float(force[2]*(1 if k==1 else-1)))
        if abs(normal[2])<.5:continue
        for side,foot in enumerate(foot_bodies):
            for k in range(2):
                if bs[k]in foot and gs[1-k]in support_ids:
                    vertical=max(0.,float(force[2]*(1 if k==1 else-1)));support[side]+=vertical
                    by_geom[side][gs[1-k]]=by_geom[side].get(gs[1-k],0.)+vertical
    return rows,support,by_geom,penetration,seat_force

def make_support_plan(reference,times,model):
    expected=np.asarray(reference.get('foot_support_expected',np.zeros((len(times),2),bool)),bool)
    names=np.asarray(reference.get('foot_support_geom_name',np.full((len(times),2),''))).astype(str)
    confidence=np.asarray(reference.get('foot_support_confidence',np.ones((len(times),2))),float)
    if expected.shape!=(len(times),2)or names.shape!=expected.shape or confidence.shape!=expected.shape:raise ValueError('Foot support plan must be T x 2')
    geom=np.full(expected.shape,-1,dtype=int);starts=np.zeros(expected.shape);ends=np.zeros(expected.shape)
    for side in range(2):
        changes=[0]+[i for i in range(1,len(times))if expected[i,side]!=expected[i-1,side]or names[i,side]!=names[i-1,side]]+[len(times)]
        for begin,end in zip(changes[:-1],changes[1:]):
            starts[begin:end,side]=times[begin];ends[begin:end,side]=times[end]if end<len(times)else times[-1]+np.median(np.diff(times))
        for i in range(len(times)):
            if expected[i,side]and names[i,side]:geom[i,side]=ident(model,mj.mjtObj.mjOBJ_GEOM,names[i,side])
    return {'available':'foot_support_expected'in reference and'foot_support_geom_name'in reference,'expected':expected,'names':names,'confidence':confidence,'geom':geom,'start':starts,'end':ends}

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--seconds',type=float);p.add_argument('--initial-velocity',choices=['reference','zero'],default='reference')
    p.add_argument('--diagnostic-reference',action='store_true');p.add_argument('--finger-driver-rotor-kg-m2',type=float,default=.00005)
    p.add_argument('--finger-follower-regularization-kg-m2',type=float,default=0.)
    p.add_argument('--task-policy',type=Path)
    p.add_argument('--wbc',action='store_true');p.add_argument('--wbc-config',type=Path)
    p.add_argument('--mpc',action='store_true');p.add_argument('--mpc-config',type=Path)
    p.add_argument('--warmup-seconds',type=float,default=0.)
    p.add_argument('--initial-root-xy-offset',type=float,nargs=2,default=[0.,0.],metavar=('X','Y'),help='Declared initial scene placement only; source reference stays unchanged and no runtime root correction is applied')
    p.add_argument('--final-seat-force-threshold-N',type=float,default=50.)
    p.add_argument('--max-root-xy-error-m',type=float,default=.35);p.add_argument('--max-root-z-error-m',type=float,default=.20);p.add_argument('--max-root-orientation-rad',type=float,default=.6)
    p.add_argument('--max-body-joint-rms-median-rad',type=float,default=.4);p.add_argument('--max-body-joint-rms-p95-rad',type=float,default=.6);p.add_argument('--max-penetration-m',type=float,default=.02)
    p.add_argument('--max-joint-limit-violation-rad',type=float,default=.05);p.add_argument('--min-stance-coverage',type=float,default=.8);p.add_argument('--support-timing-tolerance-s',type=float,default=.15);p.add_argument('--support-force-threshold-N',type=float,default=5.)
    a=p.parse_args()
    if int(a.wbc)+int(a.mpc)+int(a.task_policy is not None)>1:p.error('Select one controller')
    if a.warmup_seconds<0 or a.warmup_seconds and(a.wbc or a.mpc or a.task_policy):p.error('Warmup supports original SONIC only')
    if sys.platform!='linux':p.error('Physics and GPU inference run on SSH server only')
    if a.output.exists():raise FileExistsError(a.output)
    ref=dict(np.load(a.reference,allow_pickle=False));times=np.asarray(ref['time_s'],float);times=times-times[0];ref['time_s']=times
    source=np.asarray(ref.get('source_time_s',times),float);source=source-source[0];frames=np.asarray(ref.get('frame_ids',np.arange(len(times))))
    qref=np.asarray(ref['qpos'],float).copy();names=list(map(str,ref['dof_names']))
    if qref.shape!=(len(times),59)or len(names)!=52 or len(times)<2 or np.any(np.diff(times)<=0)or source.shape!=times.shape or np.any(np.diff(source)<=0):raise ValueError('Invalid single-Vita reference/timeline')
    # q and -q encode the same rotation. A windowed offline optimizer may
    # choose opposite signs at a join; linear interpolation must use one
    # hemisphere, just as the training Reference and WBC sampler do.
    quaternion_sign_changes=[]
    for i in range(1,len(qref)):
        if np.dot(qref[i-1,3:7],qref[i,3:7])<0:
            qref[i,3:7]*=-1.;quaternion_sign_changes.append(i)
    ref['qpos']=qref
    if not a.diagnostic_reference and len(times)!=350:raise ValueError('Task requires all 350 source frames')
    if not np.isfinite(qref).all():raise ValueError('Nonfinite reference')
    m=mj.MjModel.from_xml_path(str(a.model.resolve()))
    if(m.nq,m.nv,m.nu)!=(59,58,52)or abs(m.opt.timestep-.005)>1e-10 or m.opt.gravity[2]>-9. or m.opt.disableflags:raise ValueError('Require real free-root Vita 59/58/52, 5ms gravity/contact plant')
    root=ident(m,mj.mjtObj.mjOBJ_JOINT,'vita_demo_free_root');rq=int(m.jnt_qposadr[root]);rv=int(m.jnt_dofadr[root]);base=int(m.jnt_bodyid[root])
    if m.jnt_type[root]!=mj.mjtJoint.mjJNT_FREE or m.body_mocapid[base]>=0:raise ValueError('Robot root must be unactuated free joint')
    jids=np.asarray([ident(m,mj.mjtObj.mjOBJ_JOINT,n)for n in names]);qa=m.jnt_qposadr[jids];va=m.jnt_dofadr[jids];aids=np.asarray([ident(m,mj.mjtObj.mjOBJ_ACTUATOR,'motor_'+n)for n in names])
    if not np.array_equal(m.actuator_trnid[aids,0],jids):raise ValueError('Reference actuator map mismatch')
    if m.neq!=10 or np.any(m.eq_type!=int(mj.mjtEq.mjEQ_JOINT)):raise ValueError('Only ten passive O6 joint couplings are allowed; no root/scene welds')
    followers=np.asarray([int(eq)for eq in m.eq_obj1id]);passive=np.flatnonzero(np.isin(m.actuator_trnid[:,0],followers))
    if len(passive)!=10 or np.any(m.actuator_gear[passive])or np.any(m.actuator_gainprm[passive])or np.any(m.actuator_biasprm[passive]):raise ValueError('Passive follower gear/gain/bias must stay zero')
    sonic_config={'mode':'newft15225','device':a.device,'cpu_threads':2,'delay_steps':3,'imu_latency_s':.0025,'median_training_friction':True,'hand_object_feedback':False}
    if a.mpc:
        from rgb145_mpc_controller import MPCController
        c=MPCController(m,a.reference,a.model,a.device,json.loads(a.mpc_config.read_text())if a.mpc_config else {})
    elif a.wbc:
        from rgb145_wbc_controller import WBCController
        c=WBCController(m,ref,json.loads(a.wbc_config.read_text())if a.wbc_config else {})
    elif a.task_policy:
        from rgb145_task_controller import TaskPPOController
        c=TaskPPOController(m,a.reference,a.task_policy,a.model,a.device)
    else:c=Controller(m,ref,sonic_config)
    if c.arm_feedback is not None or len(c.aids)!=28:raise ValueError('Full original 28-body policy control required')
    powered=np.flatnonzero(np.any(m.actuator_gear!=0,axis=1)&(m.actuator_gainprm[:,0]!=0))
    if len(powered)!=42:raise ValueError('Expected 42 powered channels with ten zero-force ABI channels')
    finger=np.asarray([i for i,n in enumerate(names)if n.startswith(('lh_','rh_'))and aids[i]in powered])
    if len(finger)!=12:raise ValueError('Expected twelve independent O6 finger drivers')
    before=m.dof_armature[va[finger]].copy();m.dof_armature[va[finger]]+=a.finger_driver_rotor_kg_m2
    if a.finger_follower_regularization_kg_m2<0:raise ValueError('Regularization must be nonnegative')
    m.dof_armature[m.jnt_dofadr[followers]]+=a.finger_follower_regularization_kg_m2
    foot_bodies=[descendants(m,ident(m,mj.mjtObj.mjOBJ_BODY,side+'_ankle_roll_link'))for side in ['left','right']]
    geom_names=[mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,g)or str(g)for g in range(m.ngeom)]
    support_ids={g for g,n in enumerate(geom_names)if n.startswith(('ground_support','tier_support_','seat_support_'))and m.geom_contype[g]}
    if not support_ids:raise ValueError('No named static support collision geometry')
    plan=make_support_plan(ref,times,m);a.output.mkdir(parents=True)
    config={k:str(v)if isinstance(v,Path)else v for k,v in vars(a).items()};config['sonic_config']=sonic_config;write_json(a.output/'run_config.json',config)
    for name in ['rgb145_physics.py','rgb145_sonic_controller.py','rgb145_policy_tracking.py','rgb145_controller_provenance.json']:shutil.copyfile(ROOT/'scripts'/name,a.output/('source_'+name))
    if a.task_policy:
        for name in ['rgb145_task_controller.py','rgb145_ppo_env.py']:shutil.copyfile(ROOT/'scripts'/name,a.output/('source_'+name))
        if getattr(c,'base',None)is not None:shutil.copyfile(ROOT/'scripts/rgb145_batch_sonic.py',a.output/'source_rgb145_batch_sonic.py')
    if a.wbc:shutil.copyfile(ROOT/'scripts/rgb145_wbc_controller.py',a.output/'source_rgb145_wbc_controller.py')
    if a.mpc:
        for name in ['rgb145_mpc_controller.py','rgb145_batch_sonic.py','rgb145_ppo_env.py']:shutil.copyfile(ROOT/'scripts'/name,a.output/('source_'+name))
    declared_contract=ROOT/('outputs/rgb145_pipeline/physics_contract_mpc_v1.json'if a.mpc else'outputs/rgb145_pipeline/physics_contract_wbc_v1.json'if a.wbc else'outputs/rgb145_pipeline/physics_contract_taskppo_v1.json'if a.task_policy else'outputs/rgb145_pipeline/physics_contract_v1.json')
    if declared_contract.exists():shutil.copyfile(declared_contract,a.output/'declared_physics_contract_v1.json')
    freeze=freeze_effective_model(m,a.output/'model');d=mj.MjData(m);d.qpos[rq:rq+7]=qref[0,:7];d.qpos[qa]=qref[0,7:]
    if not np.isfinite(a.initial_root_xy_offset).all():raise ValueError('Initial placement must be finite')
    d.qpos[rq:rq+2]+=a.initial_root_xy_offset
    if a.initial_velocity=='reference':
        second=d.qpos.copy();second[rq:rq+7]=qref[1,:7];second[qa]=qref[1,7:];mj.mj_differentiatePos(m,d.qvel,times[1]-times[0],d.qpos,second)
    mj.mj_forward(m,d);c.reset(d);warmup_report=None
    if a.warmup_seconds:
        from rgb145_sonic_warmup import warmup
        shutil.copyfile(ROOT/'scripts/rgb145_sonic_warmup.py',a.output/'source_rgb145_sonic_warmup.py')
        warmup_report=warmup(m,d,c,a.warmup_seconds,a.output)
    spec=int(mj.mjtState.mjSTATE_INTEGRATION);initial=np.empty(mj.mj_stateSize(m,spec));mj.mj_getState(m,d,initial,spec)
    frame_dt=float(np.median(np.diff(times)));full_horizon=float(times[-1]+frame_dt);horizon=full_horizon if a.seconds is None else min(full_horizon,a.seconds);dt=float(m.opt.timestep)
    ts=[0.];qs=[d.qpos.copy()];vs=[d.qvel.copy()];ctrl=[];force=[];raw=[];target=[];delayed=[];infer=[];root_errors=[];rms=[];limit_errors=[];support=[];expected_force=[];support_core=[];rows=[];eq_active=[];eq_data=[];eq_force=[];constraint=[]
    warnings=np.asarray([w.number for w in d.warning]);depth=0.;reason='requested_duration';wall=time.monotonic();seat_forces=[];base_policy_targets=[]
    for step in range(int(np.ceil(horizon/dt))):
        t=float(d.time);old_count=c.inference_count;u=c.action(d,t)
        if not np.array_equal(u[c.aids],c.target_history[0])or np.any(u[passive]):raise ValueError('Body packet differs from original delayed policy or passive command nonzero')
        d.ctrl[:]=u
        if np.any(d.qfrc_applied)or np.any(d.xfrc_applied):raise ValueError('Forbidden external force')
        ctrl.append(u.copy());raw.append(c.last.copy());target.append(c.target.copy());delayed.append(c.target_history[0].copy());infer.append(c.inference_count>old_count);eq_active.append(d.eq_active.copy());eq_data.append(m.eq_data.copy())
        base_policy_targets.append(getattr(c,'last_base_target',np.zeros(28)).copy())
        before_t=d.time;mj.mj_step(m,d);ts.append(float(d.time));qs.append(d.qpos.copy());vs.append(d.qvel.copy());force.append(d.actuator_force.copy());constraint.append(d.qfrc_constraint.copy());eq_force.append(coupling_reactions(m,d))
        t=min(float(d.time),times[-1]);q=np.asarray([np.interp(t,times,qref[:,j])for j in range(59)]);q[3:7]/=np.linalg.norm(q[3:7]);actual=d.qpos[rq+3:rq+7];angle=float(2*np.arccos(np.clip(abs(np.dot(actual/np.linalg.norm(actual),q[3:7])),0,1)))
        root_errors.append([float(np.linalg.norm(d.qpos[rq:rq+2]-q[:2])),float(abs(d.qpos[rq+2]-q[2])),angle]);error=d.qpos[c.qa]-q[7+c.jref];rms.append(float(np.sqrt(np.mean(error**2))))
        lim=m.jnt_range[jids];violation=np.maximum(np.maximum(lim[:,0]-d.qpos[qa],d.qpos[qa]-lim[:,1]),0.);limit_errors.append(float(np.max(violation)))
        cs,sf,bg,pen,seat_load=contacts(m,d,step,foot_bodies,support_ids);rows.extend(cs);support.append(sf);depth=max(depth,pen);seat_forces.append(seat_load)
        idx=int(np.clip(np.searchsorted(times,t,side='right')-1,0,len(times)-1));expected_force.append([bg[k].get(int(plan['geom'][idx,k]),0.)for k in range(2)])
        support_core.append([bool(plan['expected'][idx,k]and plan['confidence'][idx,k]>=.5 and t>=plan['start'][idx,k]+a.support_timing_tolerance_s and t<=plan['end'][idx,k]-a.support_timing_tolerance_s)for k in range(2)])
        if not np.isfinite(d.qpos).all()or not np.isfinite(d.qvel).all()or abs(d.time-before_t-dt)>1e-9 or np.any(np.asarray([w.number for w in d.warning])>warnings):reason='solver_failure';break
        if angle>1.8:reason='root_orientation_safety_stop';break
    ts=np.asarray(ts);qs=np.asarray(qs);vs=np.asarray(vs);ctrl=np.asarray(ctrl);force=np.asarray(force);err=np.asarray(root_errors);rms=np.asarray(rms);support=np.asarray(support);expected_force=np.asarray(expected_force);support_core=np.asarray(support_core);eq_force=np.asarray(eq_force)
    trace=a.output/'trace.npz';np.savez_compressed(trace,time=ts,qpos=qs,qvel=vs,ctrl=ctrl,actuator_force=force,initial_state=initial,state_spec=np.asarray(spec),qfrc_applied=np.zeros((len(ctrl),m.nv)),xfrc_applied=np.zeros((len(ctrl),m.nbody,6)),
        joint_names=np.asarray(names),joint_qadr=qa,joint_vadr=va,actuator_ids=aids,robot_root_qadr=np.asarray([rq]),robot_root_vadr=np.asarray([rv]),policy_actuator_ids=c.aids,policy_joint_indices=c.jref,powered_actuator_ids=powered,passive_actuator_ids=passive,
        reference_qpos=qref,reference_time_s=times,reference_execution_time_s=times,reference_source_time_s=source,source_time_s=np.interp(ts,times,source),source_frame_ids=frames,
        root_horizontal_error_m=err[:,0],root_height_error_m=err[:,1],root_orientation_error_rad=err[:,2],body_joint_rms_rad=rms,joint_limit_violation_rad=np.asarray(limit_errors),foot_support_force_N=support,planned_support_force_N=expected_force,stance_core_mask=support_core,seat_support_force_N=np.asarray(seat_forces),
        foot_support_expected=plan['expected'],foot_support_geom_name=plan['names'],foot_support_confidence=plan['confidence'],eq_active=np.asarray(eq_active),eq_data=np.asarray(eq_data),eq_generalized_force=eq_force,qfrc_equality=np.sum(eq_force,axis=1),qfrc_constraint=np.asarray(constraint),equality_names=np.asarray([mj.mj_id2name(m,mj.mjtObj.mjOBJ_EQUALITY,i)for i in range(m.neq)]),equality_type=m.eq_type.copy(),equality_object1_id=m.eq_obj1id.copy(),equality_object2_id=m.eq_obj2id.copy())
    packet={'time_s':ts[:-1],'new_inference':np.asarray(infer),'controller_body_target':np.asarray(target),'delayed_body_target':np.asarray(delayed),'applied_body_target':ctrl[:,c.aids],'frozen_sonic_body_target':np.asarray(base_policy_targets),'frozen_sonic_present':np.asarray(getattr(c,'base',None)is not None)}
    if a.wbc:
        packet['qp_requested_body_actuator_force_Nm']=np.asarray(raw)
        write_json(a.output/'wbc_diagnostics.json',c.diagnostics)
    elif a.mpc:
        packet['normalized_effective_sonic_action']=np.asarray(raw)
        write_json(a.output/'mpc_diagnostics.json',c.diagnostics)
    else:packet.update(normalized_learned_action=np.asarray(raw),network_body_target=np.asarray(target))
    np.savez_compressed(a.output/'policy_targets.npz',**packet)
    def col(i,shape,dtype=float):return np.asarray([r[i]for r in rows],dtype=dtype).reshape(shape)
    np.savez_compressed(a.output/'contacts.npz',interval_index=col(0,(-1,),int),solver_time_s=col(1,(-1,)),geom_ids=col(2,(-1,2),int),body_ids=col(3,(-1,2),int),position_world_m=col(4,(-1,3)),force_world_on_geom2_N=col(5,(-1,3)),torque_world_on_geom2_Nm=col(6,(-1,3)),normal_force_N=col(7,(-1,)),signed_distance_m=col(8,(-1,)),normal_world=col(9,(-1,3)),geom_names=np.asarray(geom_names),body_names=np.asarray([mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,b)or'world'for b in range(m.nbody)]))
    stance_count=np.sum(support_core,axis=0);coverage=np.asarray([float(np.mean(expected_force[support_core[:,k],k]>=a.support_force_threshold_N))if stance_count[k]else 0. for k in range(2)])
    torque_margin=m.actuator_forcerange[:,1]+1e-7;torque_ok=bool(np.all(np.abs(force[:,powered])<=torque_margin[powered]))
    completed=reason=='requested_duration'and ts[-1]>=full_horizon-dt
    final_window=ts[1:]>=full_horizon-.2;final_feet=np.flatnonzero(plan['expected'][-1]& (plan['confidence'][-1]>=.5))
    final_support=bool(completed and plan['available']and len(final_feet)>0 and np.any(final_window)and all(np.mean(expected_force[final_window,k]>=a.support_force_threshold_N)>=.5 for k in final_feet))
    checks={'all_350_source_frames_and_last_display_interval':bool(completed and len(times)==350 and not a.diagnostic_reference),'finite_no_solver_warning':reason!='solver_failure','root_horizontal_tracking':bool(np.max(err[:,0])<=a.max_root_xy_error_m),'root_height_tracking':bool(np.max(err[:,1])<=a.max_root_z_error_m),'root_orientation_tracking':bool(np.max(err[:,2])<=a.max_root_orientation_rad),'body_joint_tracking_median':bool(np.median(rms)<=a.max_body_joint_rms_median_rad),'body_joint_tracking_p95':bool(np.quantile(rms,.95)<=a.max_body_joint_rms_p95_rad),'penetration':depth<=a.max_penetration_m,'joint_limits':max(limit_errors)<=a.max_joint_limit_violation_rad,'actuator_effort_limits':torque_ok,'passive_compatibility_zero_force':bool(np.max(np.abs(force[:,passive]))<1e-10),'full_28_body_original_policy_packet':bool(np.array_equal(ctrl[:,c.aids],np.asarray(delayed))),'planned_stance_correct_surface_support':bool(plan['available']and np.all(stance_count>0)and np.all(coverage>=a.min_stance_coverage))}
    checks['final_task_support_surface']=final_support
    final_seated=bool(completed and np.any(final_window)and np.mean(np.asarray(seat_forces)[final_window]>=a.final_seat_force_threshold_N)>=.5)
    checks['final_seated_load_support']=final_seated
    if a.task_policy:checks['all_28_body_task_policy_packet']=checks.pop('full_28_body_original_policy_packet')
    if a.wbc:checks['all_28_body_wbc_actuator_packet']=checks.pop('full_28_body_original_policy_packet')
    if a.mpc:checks['all_28_body_mpc_actuator_packet']=checks.pop('full_28_body_original_policy_packet')
    local={k:v for k,v in checks.items()if k not in ['all_350_source_frames_and_last_display_interval','final_task_support_surface','final_seated_load_support']};passed=all(checks.values());status='passed_full_stair_sequence'if passed else'passed_diagnostic_checks'if a.diagnostic_reference and all(local.values())and completed else'passed_prefix_checks'if all(local.values())and not completed else'failed_physical_quality'
    report={'schema':'rgb145.single_vita_sonic_physics.v1','status':status,'simulation_passed':passed,'checks':checks,'end_reason':reason,'duration_s':float(ts[-1]),'source_frames':len(times),'source_last_pts_s':float(source[-1]),'source_display_duration_s':float(source[-1]+np.median(np.diff(source))),'execution_reference_last_pts_s':float(times[-1]),'execution_horizon_s':full_horizon,'source_time_covered_s':float(np.interp(ts[-1],times,source)),'retimed':bool(not np.array_equal(times,source)),'physics_steps':len(ctrl),'physics_dt_s':dt,'mujoco_version':mj.__version__,
        'root_horizontal_error_m_max':float(np.max(err[:,0])),'root_height_error_m_max':float(np.max(err[:,1])),'root_orientation_error_rad_max':float(np.max(err[:,2])),'body_joint_rms_rad_median':float(np.median(rms)),'body_joint_rms_rad_p95':float(np.quantile(rms,.95)),'max_contact_penetration_m':depth,'max_joint_limit_violation_rad':max(limit_errors),'stance_core_steps_per_foot':stance_count.tolist(),'correct_surface_stance_coverage':coverage.tolist(),'torque_abs_peak_Nm':float(np.max(np.abs(force))),'final_seat_load_mean_N':float(np.mean(np.asarray(seat_forces)[final_window]))if np.any(final_window)else None,
        'controller':c.report(),'initial_velocity_mode':a.initial_velocity,'initial_scene_placement':{'root_xy_offset_m':a.initial_root_xy_offset,'reference_arrays_changed':False,'runtime_root_corrections':0,'applied_before_any_physics_step':True},'actuator_channels':{'physical':52,'powered':len(powered),'zero_force_passive':len(passive),'policy_body':28},'added_powered_finger_driver_rotor_kg_m2':a.finger_driver_rotor_kg_m2,'added_passive_follower_numerical_armature_kg_m2':a.finger_follower_regularization_kg_m2,'finger_driver_armature_before':before.tolist(),'active_state_writes':0,'external_force_writes':0,'root_welds':0,'dynamic_objects':0,'object_grasp_checks':'Not applicable: source has no independent dynamic object.',
        'input':{'model_xml':receipt(a.model),'reference':receipt(a.reference)},'model':receipt(a.output/'model.mjb'),'freeze':freeze,'trace':receipt(trace),'config':receipt(a.output/'run_config.json'),'declared_contract':receipt(a.output/'declared_physics_contract_v1.json')if declared_contract.exists()else None,'frozen_sources':[receipt(p)for p in sorted(a.output.glob('source_*'))],'quality_thresholds':{k:v for k,v in config.items()if k.startswith(('max_','min_','support_'))},'elapsed_wall_s':time.monotonic()-wall,
        'limitations':['Original saved policy is strictly loaded, but native MuJoCo actuator/contact dynamics are a transfer from Isaac.','The original policy has no world root XYZ target or terrain height-map observation; stance contacts and progress are independently checked.','World scale, collision proxies and physical parameters are estimates unless explicitly measured.','O6 coupling and auxiliary PD extend the 28-body training articulation; hand motion is not an observed contact measurement.']}
    report['closed_loop_warmup']=warmup_report
    report['reference_quaternion_interpolation']={'rule':'Continuous quaternion hemisphere before interpolation; q and -q have identical physical rotation','sign_changed_frame_indices':quaternion_sign_changes,'physical_reference_rotations_changed':False,'original_reference_file_modified':False}
    write_json(a.output/'report.json',report);print(json.dumps({k:report[k]for k in ['status','checks','duration_s','root_horizontal_error_m_max','root_orientation_error_rad_max','body_joint_rms_rad_median','max_contact_penetration_m','correct_surface_stance_coverage']}),flush=True)
    return 0 if passed else 3

if __name__=='__main__':raise SystemExit(main())
