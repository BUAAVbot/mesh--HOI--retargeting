"""Actuator-only thick-dish rim grasp. Original hand colliders are immutable.

The object engineering variant is explicit; objects start at source raw poses,
settle untouched, and are subsequently moved only by actual contacts.
"""
from pathlib import Path
import argparse,hashlib,json,shutil,time
import numpy as np
import mujoco as mj
from scipy.spatial.transform import Rotation
from fourth_sonic_controller import Controller
from fourth_arm_object_feedback import ArmObjectFeedback
from fourth_contact_feedback_o6_coupled import O6CoupledReferenceContactFeedback

R=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--model',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);p.add_argument('--gain',type=float,default=.0003)
p.add_argument('--seconds',type=float,default=20.);p.add_argument('--force-scale',type=float,default=1.10)
p.add_argument('--approach',choices=['top','side'],default='top')
p.add_argument('--closure-rate',type=float,default=.1)
p.add_argument('--hold-open-seed',action='store_true')
p.add_argument('--finger-hz',type=float,default=50.)
a=p.parse_args()
if a.output.exists():raise ValueError('Preserve existing trial')
a.output.mkdir(parents=True)
candidate=json.loads(a.candidate.read_text());best=candidate.get('best',candidate)
planned=np.asarray(best['qpos']);m=mj.MjModel.from_xml_path(str(a.model.resolve()))
template_supplied='template'in best
if template_supplied:template=best['template']
else:
    probe=mj.MjData(m);probe.qpos[:]=planned;mj.mj_forward(m,probe)
    ob=m.body('target_object_01').id;hb=m.body('rh_hand_base_link').id;si=m.site('rh_palm_control_site').id
    OR=probe.xmat[ob].reshape(3,3);HR=probe.xmat[hb].reshape(3,3)
    template={'palm_offset_in_object_m':(OR.T@(probe.site_xpos[si]-probe.xpos[ob])).tolist(),'hand_rotation_in_object_wxyz':Rotation.from_matrix(OR.T@HR).as_quat()[[3,0,1,2]].tolist(),
              'finger_joint_targets':{m.joint(j).name:float(planned[int(m.jnt_qposadr[j])])for j in range(m.njnt)if(m.joint(j).name or'').startswith('rh_')},'controlled_digits':['thumb','index','middle','ring','pinky'],'required_contact_digits':['thumb','index','middle']}
background=R/'outputs/fourth_full_sequence/physics_sonic15225_o6_support_absolute_hold_v1_5s'
original=dict(np.load(background/'physical_trace.npz'));q0=original['qpos'][0].copy()
# A declared, non-contact standing posture may be changed, but object reset
# remains byte-for-byte the raw source poses, never a planned grasp state.
q0[:7]=planned[:7]
for j in range(m.njnt):
    name=m.joint(j).name or ''
    if name.startswith('waist') or name.startswith('rh_'):
        q0[int(m.jnt_qposadr[j])]=planned[int(m.jnt_qposadr[j])]
if not a.hold_open_seed:
    for h in ['thumb_cmc_pitch','index_mcp_pitch','middle_mcp_pitch','ring_mcp_pitch','pinky_mcp_pitch']:
        j=m.joint('rh_'+h).id;adr=int(m.jnt_qposadr[j]);q0[adr]=max(float(m.jnt_range[j,0]),q0[adr]-(.04 if 'thumb'in h else .18))
if a.hold_open_seed:
    open16=[1.24,.255555556,.724714707,.817017986,.810186464,.775401704]
    for name,value in zip(['thumb_cmc_yaw','thumb_cmc_pitch','index_mcp_pitch','middle_mcp_pitch','ring_mcp_pitch','pinky_mcp_pitch'],open16):
        q0[int(m.jnt_qposadr[m.joint('rh_'+name).id])]=value
for e in range(m.neq):
    if m.eq_type[e]==mj.mjtEq.mjEQ_JOINT:
        child,parent=int(m.eq_obj1id[e]),int(m.eq_obj2id[e]);q0[int(m.jnt_qposadr[child])]=m.eq_data[e,1]*q0[int(m.jnt_qposadr[parent])]
# The recorded support-hold arm sits close to the original thin dish. Move
# the seven arm reset angles to a neutral, hands-away posture before settling.
right_arm=[j for j in range(m.njnt)if(m.joint(j).name or'').startswith(('right_shoulder','right_elbow','right_wrist'))]
q0[m.jnt_qposadr[right_arm]]=[0.,-.4226,0.,0.,0.,0.,0.]
background_report=json.loads((background/'report.json').read_text())
ref=dict(np.load(background_report['reference']['path']))
ref['qpos']=np.tile(q0[:59],(len(ref['time_s']),1));ref['contact_candidate_observed']=np.zeros((len(ref['time_s']),2,6),bool)
close_start=6. if a.approach=='top'else 9.
blend=np.clip((ref['time_s']-close_start)/2.5,0,1);blend=blend*blend*(3-2*blend)
if a.hold_open_seed:blend[:]=0.
for j in range(m.njnt):
    if (m.joint(j).name or'').startswith('rh_'):
        adr=int(m.jnt_qposadr[j]);ref['qpos'][:,adr]=q0[adr]+blend*(planned[adr]-q0[adr])
ref['contact_candidate_observed'][:,1,0]=True;ref['contact_candidate']=ref['contact_candidate_observed'].copy()
ref['contact_candidate_confidence']=ref['contact_candidate'].astype(float)
config=json.loads((background/'source_sonic_config.json').read_text());config['hand_object_feedback']=False
config.pop('arm_feedback',None);config.pop('finger_feedback',None)
sonic=Controller(m,ref,config)
arm=ArmObjectFeedback(m,ref,{'initial_geometric_seed':'actual_state','ik_iterations':18,'orientation_weight_m':.15,'arm_speed_rad_s':2.0})
fcfg={'geometry_backend':'convex_sat_o6','finger_urdf':str(R/'outputs/fourth_full_sequence/assets/o6_calibrated_linear_joint_contracts_v2.urdf'),
      'max_motor_torque_Nm':.4,'force_to_gap_gain_m_per_N':a.gain,'auto_gravity_normal_force':False,
      'force_target_per_digit_N':1.5,'max_force_per_digit_N':3.,'max_hand_normal_force_N':12.,
      'load_preload_normal_force_N':.1,'load_force_ramp_s':.5,'closure_rate_rad_s':a.closure_rate,'release_rate_rad_s':a.closure_rate,
      'table_clearance_guard':{'enabled':True,'minimum_clearance_m':.002,'guard_phases':['idle','release','fault']}}
finger=O6CoupledReferenceContactFeedback(m,ref,fcfg)
mj.mj_saveModel(m,str(a.output/'effective_model.mjb'),None)
np.savez_compressed(a.output/'source_reference.npz',**ref)
(a.output/'source_config.json').write_text(json.dumps(config,indent=2)+'\n')
(a.output/'finger_config.json').write_text(json.dumps(fcfg,indent=2)+'\n')
(a.output/'source_candidate.json').write_text(json.dumps(candidate,indent=2)+'\n')
for name in [Path(__file__).name,'fourth_sonic_controller.py','fourth_arm_object_feedback.py','fourth_contact_feedback_o6_coupled.py','fourth_contact_feedback_segmented.py','fourth_convex_separation.py','fourth_contact_feedback.py','fourth_contact_feedback_tableguard.py']:
    shutil.copy2(Path(__file__).with_name(name),a.output/('source_'+name))
d=mj.MjData(m);d.qpos[:]=q0;d.qvel[:]=0;mj.mj_forward(m,d);sonic.reset(d)
kin=mj.MjData(m);chain=arm.chains[1];qa=chain['qa'];va=chain['va'];aa=chain['aids'];si=np.asarray([sonic.names.index(n)for n in chain['names']])
initial_palm=arm.palm(d,1).copy();initial_R=d.xmat[arm.hand_bids[1]].reshape(3,3).copy()
obj=arm.object_bids[0];oq=arm.object_qadr[0];table=set(arm.table_geoms);names=[m.body(b).name or ''for b in range(m.nbody)]
passive=np.flatnonzero(np.all(m.actuator_gear==0,axis=1));spec=int(mj.mjtState.mjSTATE_INTEGRATION)
state0=np.empty(mj.mj_stateSize(m,spec));mj.mj_getState(m,d,state0,spec)
phase='settle';lift_started=None;ready_since=None;settled=None;holdp=None;holdR=None;lift_verified_since=None;aligned_since=None;closing_started=None;thumb_started=None
qseq=[d.qpos.copy()];vseq=[d.qvel.copy()];tseq=[0.];controls=[];forces=[];qexternal=[];xexternal=[];hist=[];phases=[];ct=[]
maxmimic=0.;maxpen=0.;stop='duration';settle_impulse=0.;start=time.monotonic();lift_verified=False
controlled=['thumb','index','middle','ring','pinky']
required=list(template.get('required_contact_digits',['thumb','index','middle']))
targets={k:float(v)*a.force_scale for k,v in template.get('digit_force_targets_N',{}).items()}
targets={'thumb':.20,'index':.25,'middle':.20,'ring':.80,'pinky':.10};required=['index','ring']
closed10=dict(zip(['thumb_cmc_yaw','thumb_cmc_pitch','index_mcp_pitch','middle_mcp_pitch','ring_mcp_pitch','pinky_mcp_pitch'],[1.24,.29481481481481486,.7335204420730096,.8266338343964567,.8196339641296827,.7844371667373781]))
if template_supplied:
    for name in closed10:
        closed10[name]=float(template['finger_joint_targets'].get('rh_'+name,closed10[name]))
previous_digit_forces={v:0. for v in controlled}
finger_stride=max(1,int(round(1/(a.finger_hz*m.opt.timestep))));held_finger_ctrl=None
for step in range(int(np.ceil(a.seconds/m.opt.timestep))):
    now=float(d.time);ctrl=sonic.action(d,now);pose=d.qpos[oq:oq+7];O=arm.rot(pose[3:])
    grasp=pose[:3]+O@np.asarray(template['palm_offset_in_object_m']);graspR=O@arm.rot(template['hand_rotation_in_object_wxyz'])
    outward=np.array([.35216658756772096,.9359373347617386,-.02508]);outward/=np.linalg.norm(outward)
    acquire_time=8. if a.approach=='top'else 12.
    standoff=np.zeros(3)if a.approach=='top'else outward*.08
    if now<2.:goal=initial_palm;goalR=initial_R
    else:
        if settled is None:settled=pose.copy()
        if now<5.:
            phase='approach_above';s=np.clip((now-2)/3,0,1);s=s*s*(3-2*s)
            goal=initial_palm+s*(grasp+standoff+np.array([0,0,.10])-initial_palm)
            goalR=Rotation.from_rotvec(s*Rotation.from_matrix(graspR@initial_R.T).as_rotvec()).as_matrix()@initial_R
        elif now<8.:
            phase='descend';s=np.clip((now-5)/3,0,1);s=s*s*(3-2*s);goal=grasp+standoff+np.array([0,0,.10*(1-s)]);goalR=graspR
        elif now<acquire_time:
            phase='side_insert';s=np.clip((now-8)/4,0,1);s=s*s*(3-2*s);goal=grasp+standoff*(1-s);goalR=graspR
        elif lift_started is None:phase='align'if a.hold_open_seed and closing_started is None else'acquire';goal=grasp;goalR=graspR
        else:
            phase='supported_tilt';angle=np.deg2rad(8.)*float(np.clip((now-lift_started)/5.,0,1));Q=Rotation.from_rotvec(tilt_axis*angle).as_matrix();goal=pivot_world+Q@(holdp-pivot_world);goalR=Q@holdR
    if step%4==0:
        geometric,error=arm.solve(d,1,goal,goalR,ctrl);arm.last_targets[1]=geometric
    kin.qpos[:]=d.qpos;kin.qvel[:]=0;mj.mj_forward(m,kin)
    ctrl[aa]=arm.last_targets[1]+np.clip(kin.qfrc_bias[va]/m.actuator_gainprm[aa,0],-.25,.25)
    live_controlled=controlled
    if a.hold_open_seed and closing_started is not None:
        if thumb_started is None and ((now-closing_started>=2. and sum(previous_digit_forces[v]>.05 for v in controlled if v!='thumb')>=2)or now-closing_started>=3.):thumb_started=now
        if thumb_started is None:live_controlled=[v for v in controlled if v!='thumb']
        for name,val in closed10.items():
            j=m.joint('rh_'+name).id;aid=m.actuator('motor_rh_'+name).id;start_at=thumb_started if'thumb'in name else closing_started
            s=0. if start_at is None else float(np.clip((now-start_at)/2.,0,1));s=s*s*(3-2*s)
            ctrl[aid]=q0[int(m.jnt_qposadr[j])]+s*(val-q0[int(m.jnt_qposadr[j])])
    plan={'active_object_ids_by_hand':[-1,1 if phase in ['acquire','supported_tilt']else -1],'phase_by_hand':['idle','grasp'if phase=='supported_tilt'else'acquire'if phase=='acquire'else'idle'],
          'controlled_digits_by_hand':[[],live_controlled],'required_contact_digits_by_hand':[[],[v for v in required if v in live_controlled]],'digit_force_targets_N_by_hand':[None,{v:x for v,x in targets.items()if v in live_controlled}or None]}
    if held_finger_ctrl is None or step%finger_stride==0:
        ctrl=finger.apply(d,ctrl,now,plan);held_finger_ctrl=ctrl[finger.active_finger_actuator_ids].copy()
    else:ctrl[finger.active_finger_actuator_ids]=held_finger_ctrl
    ctrl[passive]=0.;sonic.last[si]=(ctrl[aa]-sonic.default[si])/sonic.scale[si]
    d.ctrl[:]=ctrl
    if np.any(d.qfrc_applied)or np.any(d.xfrc_applied):raise ValueError('External force')
    mj.mj_step(m,d)
    digitforces={k:0. for k in ['thumb','index','middle','ring','pinky']};upforce=0.;tabforce=0.;normals=[]
    for ci,c in enumerate(d.contact[:d.ncon]):
        gs=np.asarray(c.geom,dtype=int);bs=m.geom_bodyid[gs];F=np.zeros(6);mj.mj_contactForce(m,d,ci,F)
        ct.append((step,gs.copy(),bs.copy(),c.pos.copy(),c.frame.copy(),F.copy(),float(c.dist)))
        if obj not in bs:continue
        if any(g in table for g in gs):tabforce+=max(float(F[0]),0.)
        obj2=bs[1]==obj;other=int(bs[0]if obj2 else bs[1]);name=names[other]
        if name.startswith('rh_'):
            digit=next((v for v in digitforces if v in name),None)
            if digit:
                digitforces[digit]+=max(float(F[0]),0.);normal=np.asarray(c.frame).reshape(3,3)[0]*(1 if obj2 else-1)
                upforce+=float(((1 if obj2 else-1)*np.asarray(c.frame).reshape(3,3).T@F[:3])[2]);normals.append((digit,normal,float(F[0])))
            if now<2:settle_impulse+=max(float(F[0]),0.)*m.opt.timestep
    opposition=min((float(n1@n2)for a1,n1,f1 in normals for a2,n2,f2 in normals if a1=='thumb'and a2!='thumb'and min(f1,f2)>.05),default=1.)
    previous_digit_forces=digitforces.copy()
    if phase=='align':
        aligned=np.linalg.norm(arm.palm(d,1)-goal)<.008 and np.linalg.norm(d.qvel[va])<.15
        if aligned:
            if aligned_since is None:aligned_since=float(d.time)
            if d.time-aligned_since>=.2:closing_started=float(d.time)
        else:aligned_since=None
    ready=all(digitforces[v]>.05 for v in required)and upforce>.30
    # Supported tilt requires real lower-rim support; do not require a free-lift wrench.
    if ready:
        if ready_since is None:ready_since=float(d.time)
    else:ready_since=None
    if phase=='acquire'and ready_since is not None and d.time-ready_since+1e-9>=.1:
        lift_started=float(d.time);holdp=arm.palm(d,1).copy();holdR=d.xmat[arm.hand_bids[1]].reshape(3,3).copy();base_oP=pose[:3].copy();base_oR=O.copy();up=O[:,2];rr=outward-up*(outward@up);rr/=np.linalg.norm(rr);tilt_axis=-np.cross(up,rr);bottom=m.geom('target_object_01_bottom_collision').id;pivot_world=base_oP-rr*m.geom_size[bottom,0]+up*(m.geom_pos[bottom,2]-m.geom_size[bottom,1]);near_local=base_oR.T@rr*.102455974+np.array([0.,0.,.02]);near_initial=base_oP+base_oR@near_local
    rise=0. if settled is None else float(d.qpos[oq+2]-settled[2])
    angle_actual=0. if lift_started is None else float(Rotation.from_matrix(O@base_oR.T).as_rotvec()@tilt_axis);near_rise=0. if lift_started is None else float((pose[:3]+O@near_local-near_initial)@up);proof=lift_started is not None and angle_actual>=np.deg2rad(6.)and near_rise>=.015 and tabforce>.20 and upforce>.20
    if proof:
        if lift_verified_since is None:lift_verified_since=float(d.time)
        if d.time-lift_verified_since+1e-9>=.08:lift_verified=True
    else:lift_verified_since=None
    mimic=max(abs(d.qpos[int(m.jnt_qposadr[int(m.eq_obj1id[e])])]-m.eq_data[e,1]*d.qpos[int(m.jnt_qposadr[int(m.eq_obj2id[e])])])for e in range(m.neq)if m.eq_type[e]==mj.mjtEq.mjEQ_JOINT)
    maxmimic=max(maxmimic,float(mimic));maxpen=max(maxpen,max([-c.dist for c in d.contact[:d.ncon]]+[0.]))
    upright=float(d.xmat[m.body('base_link').id].reshape(3,3)[2,2]);root=float(np.linalg.norm(d.qpos[:3]-q0[:3]));palmerror=float(np.linalg.norm(arm.palm(d,1)-goal))
    hist.append([d.time,rise,tabforce,upforce,opposition,mimic,upright,root,palmerror,*digitforces.values()]);phases.append(phase)
    qseq.append(d.qpos.copy());vseq.append(d.qvel.copy());tseq.append(d.time);controls.append(ctrl.copy());forces.append(d.actuator_force.copy());qexternal.append(d.qfrc_applied.copy());xexternal.append(d.xfrc_applied.copy())
    if step%100==0:print(json.dumps({'t':d.time,'phase':phase,'rise':rise,'tilt_deg':float(np.rad2deg(angle_actual)),'near_rise':near_rise,'table_N':tabforce,'hand_up_N':upforce,'digit_N':digitforces,'mimic':mimic,'palmerror':palmerror}),flush=True)
    if mimic>.03 or upright<.8 or root>.2 or not np.isfinite(d.qpos).all():stop='mimic_protection'if mimic>.03 else'body_or_numerical_protection';break
np.savez_compressed(a.output/'physical_trace.npz',time_s=np.asarray(tseq),qpos=np.asarray(qseq),qvel=np.asarray(vseq),ctrl=np.asarray(controls),actuator_force=np.asarray(forces),initial_state=state0,state_spec=np.asarray(spec),qfrc_applied=np.asarray(qexternal),xfrc_applied=np.asarray(xexternal))
np.savez_compressed(a.output/'physical_contacts.npz',step=np.asarray([c[0]for c in ct]),geom=np.asarray([c[1]for c in ct]),body=np.asarray([c[2]for c in ct]),point=np.asarray([c[3]for c in ct]),frame=np.asarray([c[4]for c in ct]),force=np.asarray([c[5]for c in ct]),distance=np.asarray([c[6]for c in ct]))
np.savez_compressed(a.output/'grasp_history.npz',values=np.asarray(hist),phase=np.asarray(phases),columns=np.asarray(['time_s','rise_m','table_normal_N','hand_upward_N','opposition_dot','mimic_error_rad','root_up','root_drift_m','palm_error_m','thumb_N','index_N','middle_N','ring_N','pinky_N']))
np.savez_compressed(a.output/'finger_history.npz',**finger.history_arrays())
report={'success':lift_verified,'scope':'Supported near-edge tilt pre-operation only; NOT a free-object lift or full923/six task','duration_s':float(d.time),'termination':stop,'elapsed_wall_s':time.monotonic()-start,'model':str(a.output/'effective_model.mjb'),'candidate':str(a.candidate),'gain':a.gain,'force_scale':a.force_scale,'closing_targets':closed10,'opening_targets':'Exact common-plane16mm seed when hold_open_seed enabled','template_curve_targets_used':template_supplied,'finger_feedback_hz':1/(finger_stride*m.opt.timestep),'physics_hz':1/m.opt.timestep,'initial_objects_raw':True,'post_reset_state_overrides':0,'external_forces_zero':True,'hand_colliders_modified':False,'settle_hand_object_impulse_Ns':settle_impulse,'max_mimic_error_rad':maxmimic,'max_penetration_m':maxpen,'maximum_object_rise_m':float(np.max(np.asarray(hist)[:,1])),'final_tilt_deg':float(np.rad2deg(angle_actual)),'final_near_edge_rise_m':near_rise,'supported_tilt_started_s':lift_started,'passive_max_abs_torque_Nm':float(np.max(abs(np.asarray(forces)[:,passive]))),'hand_active_max_abs_torque_Nm':float(np.max(abs(np.asarray(forces)[:,finger.active_finger_actuator_ids]))),'finger_report':finger.report()}
(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k!='finger_report'}),flush=True)
