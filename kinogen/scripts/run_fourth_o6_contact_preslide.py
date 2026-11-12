"""Isolated original-position, actuator-only wall push with live SONIC balance.

No object state assignments after initial reset, no applied forces or welds.
The hand follows a slow world-space path; contact load limits stop its advance.
"""
from pathlib import Path
import argparse,hashlib,json,shutil,time
import numpy as np
import mujoco as mj
from scipy.spatial.transform import Rotation
from fourth_sonic_controller import Controller

R=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--plan',type=Path,default=R/'reports/fourth_o6_contact_preslide_clearance_v2.json')
p.add_argument('--output',type=Path,default=R/'outputs/fourth_full_sequence/physics_o6_contact_preslide_v1')
p.add_argument('--seconds',type=float,default=16.);p.add_argument('--speed',type=float,default=.006)
p.add_argument('--stop-displacement',type=float,default=.0485)
p.add_argument('--raw-settle-approach',action='store_true');p.add_argument('--side-entry-standoff-m',type=float,default=0.);a=p.parse_args()
if a.output.exists():raise ValueError('Preserve existing trials; use a new output directory')
a.output.mkdir(parents=True);plan=json.loads(a.plan.read_text());qplan=np.asarray(plan['best']['qpos']);q0=qplan.copy();m=mj.MjModel.from_binary_path(plan['model']);bn=[m.body(b).name or'world'for b in range(m.nbody)];names=[m.joint(j).name or''for j in range(m.njnt)]
early_arm=[j for j,n in enumerate(names)if n.startswith(('right_shoulder','right_elbow','right_wrist'))];early_aq=m.jnt_qposadr[early_arm]
if a.raw_settle_approach:
    q0[59:]=np.load(Path(plan['background_trial'])/'physical_trace.npz')['qpos'][0,59:]
    q0[early_aq]=[0.,-.4226,0.,0.,0.,0.,0.]
joints=[j for j in range(m.njnt)if m.jnt_type[j]==mj.mjtJoint.mjJNT_HINGE];dofnames=[names[j]for j in joints];qa=m.jnt_qposadr[joints]
ref={'time_s':np.linspace(0,30.74,923),'dof_names':np.asarray(dofnames),'qpos':np.tile(q0[:59],(923,1)),
     'object_ids':np.arange(1,7),'object_target_pose':np.tile(q0[59:].reshape(1,6,7),(923,1,1)),
     'diagnostic_only':np.asarray(True)}
if a.raw_settle_approach:
    blend=np.clip((ref['time_s']-2.)/(4.5+(4. if a.side_entry_standoff_m>0 else 0)),0,1);blend=blend*blend*(3-2*blend)
    ref['qpos'][:,early_aq]=q0[early_aq]+blend[:,None]*(qplan[early_aq]-q0[early_aq])
config=json.loads((Path(plan['background_trial'])/'source_sonic_config.json').read_text());config['hand_object_feedback']=False
config.pop('arm_feedback',None);config.pop('finger_feedback',None)
sonic=Controller(m,ref,config);mj.mj_saveModel(m,str(a.output/'effective_model.mjb'),None)
np.savez_compressed(a.output/'source_reference.npz',**ref);(a.output/'source_config.json').write_text(json.dumps(config,indent=2)+'\n');shutil.copy2(a.plan,a.output/'source_plan.json')
for name in [Path(__file__).name,'fourth_sonic_controller.py','vita_policy_tracking.py']:
    src=Path(__file__).with_name(name)
    if src.exists():shutil.copy2(src,a.output/('source_'+name))
d=mj.MjData(m);d.qpos[:]=q0;d.qvel[:]=0;mj.mj_forward(m,d);sonic.reset(d)
kin=mj.MjData(m);arm=np.asarray([j for j,n in enumerate(names)if n.startswith(('right_shoulder','right_elbow','right_wrist'))]);aq=m.jnt_qposadr[arm];av=m.jnt_dofadr[arm]
aa=np.asarray([mj.mj_name2id(m,mj.mjtObj.mjOBJ_ACTUATOR,'motor_'+names[j])for j in arm]);si=np.asarray([sonic.names.index(names[j])for j in arm]);site=mj.mj_name2id(m,mj.mjtObj.mjOBJ_SITE,'rh_palm_control_site')
hand_masters={};projections={};hand_aids={}
for side in ['lh','rh']:
    js=[names.index(side+'_'+n)for n in ['thumb_cmc_yaw','thumb_cmc_pitch','index_mcp_pitch','middle_mcp_pitch','ring_mcp_pitch','pinky_mcp_pitch']];P=np.zeros((m.nv,6))
    for col,j in enumerate(js):P[int(m.jnt_dofadr[j]),col]=1
    for e in range(m.neq):
        if m.eq_type[e]==mj.mjtEq.mjEQ_JOINT and int(m.eq_obj2id[e])in js:P[int(m.jnt_dofadr[int(m.eq_obj1id[e])]),js.index(int(m.eq_obj2id[e]))]=m.eq_data[e,1]
    hand_masters[side]=js;projections[side]=P;hand_aids[side]=np.asarray([mj.mj_name2id(m,mj.mjtObj.mjOBJ_ACTUATOR,'motor_'+names[j])for j in js])
passive=np.flatnonzero(np.all(m.actuator_gear==0,axis=1));assert len(passive)==10 and np.all(m.actuator_gainprm[passive]==0)and np.all(m.actuator_biasprm[passive]==0)
objects=np.asarray([mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,f'target_object_{oid:02d}')for oid in range(1,7)]);obj=int(objects[0]);oj=int(m.body_jntadr[obj]);oq=int(m.jnt_qposadr[oj]);out=np.asarray(plan['outward_tangent_unit']);initial_obj=q0[oq:oq+3].copy();up=np.asarray(plan['table_plane'][:3])
basep=d.site_xpos[site].copy();baseR=d.site_xmat[site].reshape(3,3).copy();initial_palm=basep.copy();initial_R=baseR.copy();goalR=baseR.copy();lastgeom=q0[aq].copy();goal=basep.copy();command_distance=0.;phase='settle';phase_since=0.;withdraw_start=None;reason='duration';dt=float(m.opt.timestep)
plan_kin=mj.MjData(m);plan_kin.qpos[:]=qplan;mj.mj_forward(m,plan_kin);planned_palm=plan_kin.site_xpos[site].copy();planned_R=plan_kin.site_xmat[site].reshape(3,3).copy()
settled_poses=None;settled_at=None;settle_hand_object_impulse=0.;settled_prediction_error=None;raw_initial_obj=initial_obj.copy()
spec=int(mj.mjtState.mjSTATE_INTEGRATION);state0=np.empty(mj.mj_stateSize(m,spec));mj.mj_getState(m,d,state0,spec)
qseq=[d.qpos.copy()];vseq=[d.qvel.copy()];tseq=[0.];controls=[];torques=[];qforces=[];xforces=[];history=[];phases=[];contacts=[];warn0=np.asarray([v.number for v in d.warning]);maxpen=0.;maxmimic=0.;start=time.monotonic();force_previous=0.;failed=False
def contacts_now(step):
    force=0.;hand_table=0.;allcontacts=[]
    for c in d.contact[:d.ncon]:
        gs=[int(c.geom1),int(c.geom2)];bs=[int(m.geom_bodyid[g])for g in gs];F=np.zeros(6);mj.mj_contactForce(m,d,len(allcontacts),F)
        allcontacts.append((step,gs,bs,np.asarray(c.pos).copy(),np.asarray(c.frame).copy(),F.copy(),float(c.dist)))
        other=bs[1]if bs[0]==obj else bs[0]if bs[1]==obj else -1
        if other>=0 and bn[other].startswith('rh_index'):force+=max(float(F[0]),0.)
        if any(bn[b].startswith('rh_')for b in bs)and any((m.geom(g).name or'').startswith('table_support')for g in gs):hand_table+=max(float(F[0]),0.)
    return force,hand_table,allcontacts
for step in range(int(np.ceil(a.seconds/dt))):
    now=float(d.time);ctrl=np.asarray(sonic.action(d,now));displacement=d.qpos[oq:oq+3]-initial_obj;progress=float(displacement@out);norm=float(np.linalg.norm(displacement))
    if phase=='settle'and now>=(2. if a.raw_settle_approach else .5):
        if a.raw_settle_approach:
            settled_poses=d.qpos[59:].reshape(6,7).copy();settled_at=now;settled_prediction_error=float(np.max(np.linalg.norm(settled_poses[:,:3]-qplan[59:].reshape(6,7)[:,:3],axis=1)))
            initial_obj=d.qpos[oq:oq+3].copy()
            rawO=Rotation.from_quat(qplan[oq+3:oq+7][[1,2,3,0]]).as_matrix();actualO=Rotation.from_quat(d.qpos[oq+3:oq+7][[1,2,3,0]]).as_matrix();change=actualO@rawO.T
            basep=initial_obj+change@(planned_palm-qplan[oq:oq+3]);baseR=change@planned_R
            phase='approach_high'
        else:phase='push'
        phase_since=now
    if phase=='approach_high':
        alpha=float(np.clip((now-phase_since)/2.,0,1));smooth=alpha*alpha*(3-2*alpha)
        goal=initial_palm+(basep-out*a.side_entry_standoff_m+up*.12-initial_palm)*smooth;goalR=Rotation.from_rotvec(smooth*Rotation.from_matrix(baseR@initial_R.T).as_rotvec()).as_matrix()@initial_R
        if alpha>=1:phase='descend';phase_since=now
    if phase=='descend':
        alpha=float(np.clip((now-phase_since)/2.5,0,1));smooth=alpha*alpha*(3-2*alpha);goal=basep-out*a.side_entry_standoff_m+up*.12*(1-smooth);goalR=baseR.copy()
        if alpha>=1:phase='side_acquire'if a.side_entry_standoff_m>0 else'push';phase_since=now
    if phase=='side_acquire':
        alpha=float(np.clip((now-phase_since)/4.,0,1));smooth=alpha*alpha*(3-2*alpha);goal=basep-out*a.side_entry_standoff_m*(1-smooth);goalR=baseR.copy()
        if alpha>=1:phase='push';phase_since=now
    if phase=='push'and (norm>=a.stop_displacement or progress>=a.stop_displacement):phase='withdraw';phase_since=now;withdraw_start=goal.copy()
    if phase=='push':
        speed=a.speed if progress<.04 else min(a.speed,.002)
        command_distance+=dt*(speed*np.clip((2.5-force_previous)/.8,0,1)if force_previous<=2.5 else -.006*np.clip((force_previous-2.5)/.5,0,1))
        command_distance=float(np.clip(command_distance,0,.075));goal=basep+out*command_distance
    elif phase=='withdraw':goal=withdraw_start-out*.015*np.clip((now-phase_since)/.35,0,1)
    if step%4==0:
        kin.qpos[:]=d.qpos;kin.qvel[:]=0
        for iteration in range(10):
            mj.mj_kinematics(m,kin);mj.mj_comPos(m,kin);JP=np.zeros((3,m.nv));JR=np.zeros_like(JP);mj.mj_jacSite(m,kin,JP,JR,site)
            e=np.r_[goal-kin.site_xpos[site],.2*Rotation.from_matrix(goalR@kin.site_xmat[site].reshape(3,3).T).as_rotvec()];J=np.vstack([JP[:,av],.2*JR[:,av]])
            correction=J.T@np.linalg.solve(J@J.T+1e-5*np.eye(6),e);kin.qpos[aq]=np.clip(kin.qpos[aq]+np.clip(correction,-.06,.06),m.jnt_range[arm,0],m.jnt_range[arm,1])
            if np.linalg.norm(e)<2e-5:break
        lastgeom=lastgeom+np.clip(kin.qpos[aq]-lastgeom,-3.*.02,3.*.02)
    kin.qpos[:]=d.qpos;kin.qvel[:]=0;mj.mj_forward(m,kin)
    ctrl[aa]=lastgeom+np.clip(kin.qfrc_bias[av]/m.actuator_gainprm[aa,0],-.25,.25)
    for side,js in hand_masters.items():
        aids=hand_aids[side];ctrl[aids]=q0[m.jnt_qposadr[js]]+projections[side].T@kin.qfrc_bias/m.actuator_gainprm[aids,0]
    ctrl[passive]=0.;sonic.last[si]=(ctrl[aa]-sonic.default[si])/sonic.scale[si]
    d.ctrl[:]=ctrl
    if np.any(d.qfrc_applied)or np.any(d.xfrc_applied):raise ValueError('Unexpected external force')
    previous_time=float(d.time);mj.mj_step(m,d)
    if not np.isfinite(d.qpos).all()or abs(d.time-(previous_time+dt))>1e-9:reason='numerical_reset';failed=True;break
    force,hand_table,ct=contacts_now(step);contacts.extend(ct);force_previous=force
    if a.raw_settle_approach and previous_time<2.:
        for _,gs,bs,point,frame,F,distance in ct:
            if any(b in objects for b in bs)and any(bn[b].startswith(('rh_','lh_'))for b in bs):settle_hand_object_impulse+=max(float(F[0]),0)*dt
    errors=[]
    for e in range(m.neq):
        if m.eq_type[e]==mj.mjtEq.mjEQ_JOINT:
            child,parent=int(m.eq_obj1id[e]),int(m.eq_obj2id[e]);errors.append(float(d.qpos[int(m.jnt_qposadr[child])]-m.eq_data[e,1]*d.qpos[int(m.jnt_qposadr[parent])]))
    error=max(abs(v)for v in errors);maxmimic=max(maxmimic,error);penetration=max([max(-float(c.dist),0.)for c in d.contact[:d.ncon]]+[0.]);maxpen=max(maxpen,penetration);upright=float(d.xmat[1].reshape(3,3)[2,2]);rootdist=float(np.linalg.norm(d.qpos[:3]-q0[:3]))
    palm_error=goal-kin.site_xpos[site]
    history.append([d.time,progress,norm,force,hand_table,error,upright,rootdist,command_distance,np.linalg.norm(palm_error),*palm_error]);phases.append(phase);qseq.append(d.qpos.copy());vseq.append(d.qvel.copy());tseq.append(d.time);controls.append(ctrl.copy());torques.append(d.actuator_force.copy());qforces.append(d.qfrc_applied.copy());xforces.append(d.xfrc_applied.copy())
    if step%100==0:print(json.dumps({'t':round(d.time,3),'phase':phase,'displacement_m':norm,'progress_m':progress,'index_normal_N':force,'mimic_error_rad':error,'root_drift_m':rootdist,'palm_goal_minus_actual_xyz_m':palm_error.tolist()}),flush=True)
    if phase!='withdraw'and (error>.03 or hand_table>3. or upright<.8 or rootdist>.15):
        reason='mimic_protection'if error>.03 else'hand_table_load'if hand_table>3 else'body_stability';phase='withdraw';phase_since=float(d.time);withdraw_start=goal.copy();failed=True
    if norm>.05+1e-9:reason='object_exceeded_5cm';failed=True
    if phase=='withdraw'and d.time-phase_since>=1.:reason=reason if failed else'target_slide_then_withdraw';break
np.savez_compressed(a.output/'physical_trace.npz',time_s=np.asarray(tseq),qpos=np.asarray(qseq),qvel=np.asarray(vseq),ctrl=np.asarray(controls),actuator_force=np.asarray(torques),initial_state=state0,state_spec=np.asarray(spec),qfrc_applied=np.asarray(qforces),xfrc_applied=np.asarray(xforces))
if contacts:
    np.savez_compressed(a.output/'physical_contacts.npz',step=np.asarray([c[0]for c in contacts]),geom=np.asarray([c[1]for c in contacts]),body=np.asarray([c[2]for c in contacts]),point=np.asarray([c[3]for c in contacts]),frame=np.asarray([c[4]for c in contacts]),force=np.asarray([c[5]for c in contacts]),distance=np.asarray([c[6]for c in contacts]))
np.savez_compressed(a.output/'preslide_control_history.npz',values=np.asarray(history),phase=np.asarray(phases),columns=np.asarray(['time_s','object_progress_pre_step_m','object_displacement_pre_step_m','index_normal_N','hand_table_normal_N','max_mimic_error_rad','root_up','root_drift_m','planned_palm_translation_m','palm_goal_error_m','palm_goal_minus_actual_x_m','palm_goal_minus_actual_y_m','palm_goal_minus_actual_z_m']))
positions=np.asarray(qseq)[:,oq:oq+3];norms=np.linalg.norm(positions-initial_obj,axis=1);actual=positions[-1]-initial_obj
final_progress=float(actual@out);lateral=float(np.linalg.norm(actual-out*final_progress));minimum_up=min(row[6]for row in history);maximum_root=max(row[7]for row in history)
ok=not failed and final_progress>=.045 and lateral<=.005 and max(norms)<=.05 and maxmimic<=.03 and maxpen<=.02 and minimum_up>=.85 and maximum_root<=.1 and(settle_hand_object_impulse==0 if a.raw_settle_approach else True)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
report={'schema':'fourth.o6_original_object_contact_preslide_physics.v1','success':bool(ok),'termination_reason':reason,'duration_s':float(d.time),'elapsed_wall_s':time.monotonic()-start,
        'model':{'path':str(a.output/'effective_model.mjb'),'sha256':sha(a.output/'effective_model.mjb')},'reference':{'path':str(a.output/'source_reference.npz'),'sha256':sha(a.output/'source_reference.npz')},'plan':str(a.plan),'plan_sha256':sha(a.plan),'policy':sonic.report(),
        'control':'Live SONIC28body; RH7arm actuator-only world IK/gravity hold; both6command hands hold original preshape with gravity feedforward; all10passive ctrl0.',
        'initial_object_pose':q0[59:].reshape(6,7).tolist(),'final_object_pose':qseq[-1][59:].reshape(6,7).tolist(),'initial_robot_qpos':q0[:59].tolist(),
        'object1_actual_delta_m':actual.tolist(),'object1_final_displacement_m':float(norms[-1]),'object1_max_displacement_m':float(max(norms)),'max_mimic_error_rad':maxmimic,
        'object1_edge_direction_progress_m':final_progress,'object1_edge_lateral_displacement_m':lateral,'root_up_minimum':minimum_up,'root_displacement_maximum_m':maximum_root,
        'index_object_normal_force_maximum_N':max(row[3]for row in history),'hand_table_normal_force_maximum_N':max(row[4]for row in history),
        'max_penetration_m':maxpen,'external_forces_zero':bool(np.all(np.asarray(qforces)==0)and np.all(np.asarray(xforces)==0)),
        'all_6_objects_dynamic_free':bool(all(m.jnt_type[int(m.body_jntadr[b])]==mj.mjtJoint.mjJNT_FREE and m.body_mocapid[b]<0 for b in objects)),
        'passive_max_abs_actuator_force_Nm':float(np.max(np.abs(np.asarray(torques)[:,passive])))if torques else None,
        'rh_active_hand_max_abs_actuator_force_Nm':float(np.max(np.abs(np.asarray(torques)[:,hand_aids['rh']])))if torques else None,
        'initial_objects_not_preslid':True,'post_reset_state_overrides':0,'not_full_six_object_stacking_task':True,'table_and_body_model_modified':False,
        'raw_initialization_then_settle_approach':a.raw_settle_approach,'actual_settled_object_poses':None if settled_poses is None else settled_poses.tolist(),'settled_time_s':settled_at,
        'side_entry_standoff_m':a.side_entry_standoff_m,
        'settle_phase_all_six_hand_object_impulse_Ns':settle_hand_object_impulse,'settled_object_prediction_max_origin_error_m':settled_prediction_error,
        'displacement_reference_origin':initial_obj.tolist(),'displacement_reference_is_actual_two_second_settled_pose':a.raw_settle_approach,
        'maximum_object1_raw_origin_displacement_m':float(max(np.linalg.norm(positions-raw_initial_obj,axis=1)))}
(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
