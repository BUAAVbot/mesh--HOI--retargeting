"""Continuous actual raw-scene push, then under-rim grasp; no object writes.

The first stage executes the frozen previous push program and returns its live
MjData/SONIC history. The second stage modifies actuator targets only.
"""
from pathlib import Path
import argparse,contextlib,hashlib,json,runpy,shutil,sys,time
import numpy as np
import mujoco as mj
from scipy.spatial.transform import Rotation
from fourth_arm_object_feedback import ArmObjectFeedback
from fourth_contact_feedback_o6_coupled import O6CoupledReferenceContactFeedback
R=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=float,default=30.);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
a.output.mkdir(parents=True);candidate=json.loads(a.candidate.read_text());best=candidate['best'];template=best['template'];planned=np.asarray(best['qpos'])
(a.output/'source_candidate.json').write_text(json.dumps(candidate,indent=2)+'\n')
for name in [Path(__file__).name,'run_fourth_o6_preslide_extended_v4.py','fourth_sonic_controller.py','fourth_arm_object_feedback.py','fourth_contact_feedback_o6_coupled.py','fourth_contact_feedback_segmented.py','fourth_contact_feedback_tableguard.py','fourth_contact_feedback.py','fourth_convex_separation.py']:
    shutil.copy2(Path(__file__).with_name(name),a.output/('source_'+name))
argv=sys.argv.copy();sys.argv=[str(R/'scripts/run_fourth_o6_preslide_extended_v4.py'),'--plan',str(R/'outputs/fourth_full_sequence/physics_o6_contact_preslide_v3/source_plan.json'),'--output',str(a.output/'push'),'--seconds','40','--speed','.006','--stop-displacement','.0495','--raw-settle-approach','--side-entry-standoff-m','.04']
with(a.output/'push.log').open('w')as log,contextlib.redirect_stdout(log):g=runpy.run_path(str(R/'scripts/run_fourth_o6_preslide_extended_v4.py'),run_name='__main__')
sys.argv=argv;m=g['m'];d=g['d'];sonic=g['sonic'];q0=g['q0'];t0=float(d.time);prefix=np.load(R/'outputs/fourth_full_sequence/physics_o6_contact_preslide_v4_extended40/physical_trace.npz')
prefix_q_error=float(np.max(abs(np.asarray(g['qseq'])-prefix['qpos'])));prefix_v_error=float(np.max(abs(np.asarray(g['vseq'])-prefix['qvel'])))
if prefix_q_error>1e-9 or prefix_v_error>1e-9:raise ValueError('Push prefix differs; replan from this exact new prefix before grasp')
if np.max(abs(d.qpos[59:]-planned[59:]))>1e-9:raise ValueError('Candidate must use exact recorded final scene')
print(json.dumps({'phase':'continuous_push_done','time_s':t0,'prefix_q_error':prefix_q_error,'prefix_v_error':prefix_v_error}),flush=True)
# Retarget only the reference; actual state and policy history stay continuous.
oldtimes=sonic.times.copy();oldq=sonic.qref.copy();newtimes=t0+np.linspace(0,a.seconds,int(np.ceil(a.seconds/.02))+1)
blend=np.clip((newtimes-t0)/5.,0,1);blend=blend*blend*(3-2*blend)
desired=planned[:59].copy()
if oldq[-1,3:7]@desired[3:7]<0:desired[3:7]*=-1
newq=oldq[-1]+blend[:,None]*(desired-oldq[-1]);newq[:,3:7]/=np.linalg.norm(newq[:,3:7],axis=1)[:,None]
sonic.times=np.r_[oldtimes,newtimes];sonic.qref=np.r_[oldq,newq];sonic.body_q=sonic.qref[:,7+sonic.jref];sonic.body_dq=np.gradient(sonic.body_q,sonic.times,axis=0)
np.savez_compressed(a.output/'adapted_robot_reference.npz',time_s=sonic.times,qpos=sonic.qref,dof_names=np.asarray(sonic.reference_names))
ref=dict(g['ref']);ref['contact_candidate_observed']=np.zeros((len(ref['time_s']),2,6),bool);ref['contact_candidate_observed'][:,1,0]=True;ref['contact_candidate']=ref['contact_candidate_observed'];ref['contact_candidate_confidence']=ref['contact_candidate'].astype(float)
arm=ArmObjectFeedback(m,ref,{'initial_geometric_seed':'actual_state','ik_iterations':18,'orientation_weight_m':.15,'arm_speed_rad_s':1.5})
fcfg={'geometry_backend':'convex_sat_o6','finger_urdf':str(R/'outputs/fourth_full_sequence/assets/o6_calibrated_linear_joint_contracts_v2.urdf'),'max_motor_torque_Nm':.4,'force_to_gap_gain_m_per_N':.00015,'auto_gravity_normal_force':False,'force_target_per_digit_N':1.5,'max_force_per_digit_N':5.,'max_hand_normal_force_N':16.,'closure_rate_rad_s':.1,'release_rate_rad_s':.1,'load_preload_normal_force_N':.4,'load_force_ramp_s':.6,'table_clearance_guard':{'enabled':True,'minimum_clearance_m':.002,'guard_phases':['idle','release','fault','acquire','grasp']}}
finger=O6CoupledReferenceContactFeedback(m,ref,fcfg);(a.output/'finger_config.json').write_text(json.dumps(fcfg,indent=2)+'\n');mj.mj_saveModel(m,str(a.output/'effective_model.mjb'),None)
kin=mj.MjData(m);chain=arm.chains[1];qa=chain['qa'];va=chain['va'];aa=chain['aids'];si=np.asarray([sonic.names.index(n)for n in chain['names']]);passive=g['passive'];obj=arm.object_bids[0];oq=arm.object_qadr[0];table=set(arm.table_geoms);bn=[m.body(b).name or''for b in range(m.nbody)]
initial_palm=arm.palm(d,1).copy();initial_R=d.xmat[arm.hand_bids[1]].reshape(3,3).copy();settled=d.qpos[oq:oq+7].copy();qseq=g['qseq'];vseq=g['vseq'];tseq=g['tseq'];controls=g['controls'];torques=g['torques'];qforces=g['qforces'];xforces=g['xforces'];contacts=g['contacts'];step0=len(controls);history=[];phases=[];maxmimic=g['maxmimic'];maxpen=g['maxpen'];held_finger=None;aligned_since=None;closing_started=None;ready_since=None;lift_started=None;holdp=None;holdR=None;proof_since=None;verified=False;stop='duration';start=time.monotonic()
digits=['thumb','index','middle','ring','pinky'];required=['thumb','index','middle'];targets=template.get('digit_force_targets_N',{'thumb':2.514,'index':1.682,'middle':2.641,'ring':.1,'pinky':.1})
if 'wrench'in best and best['wrench'].get('feasible',False):
    targets={name:1.03*float(value)for name,value in zip(required,best['wrench']['normal_per_digit_N'])};targets.update(ring=.1,pinky=.1)
(a.output/'load_targets.json').write_text(json.dumps({'targets_N':targets,'LP_multiplier':1.03,'engineering_feedback_digit_cap_N':5.,'physical_motor_cap_Nm':.4},indent=2)+'\n')
for localstep in range(int(np.ceil(a.seconds/m.opt.timestep))):
    now=float(d.time);t=now-t0;ctrl=sonic.action(d,now);pose=d.qpos[oq:oq+7];OR=arm.rot(pose[3:]);grasp=pose[:3]+OR@np.asarray(template['palm_offset_in_object_m']);GR=OR@arm.rot(template['hand_rotation_in_object_wxyz']);out=np.array([.3521665876,.9359373348,-.02508]);out/=np.linalg.norm(out)
    if t<5:phase='posture';goal=initial_palm;rotation=initial_R
    elif t<8:
        phase='approach';s=np.clip((t-5)/3,0,1);s=s*s*(3-2*s);goal=initial_palm+s*(grasp+out*.08+np.array([0,0,.10])-initial_palm);rotation=Rotation.from_rotvec(s*Rotation.from_matrix(GR@initial_R.T).as_rotvec()).as_matrix()@initial_R
    elif t<11:
        phase='descend_outside';s=np.clip((t-8)/3,0,1);s=s*s*(3-2*s);goal=grasp+out*.08+np.array([0,0,.10*(1-s)]);rotation=GR
    elif t<15:
        phase='insert_under';s=np.clip((t-11)/4,0,1);s=s*s*(3-2*s);goal=grasp+out*.08*(1-s);rotation=GR
    elif lift_started is None:phase='align'if closing_started is None else'acquire';goal=grasp;rotation=GR
    else:phase='lift';goal=holdp+np.array([0,0,min(.08,.015*(now-lift_started))]);rotation=holdR
    if localstep%4==0:geometric,error=arm.solve(d,1,goal,rotation,ctrl);arm.last_targets[1]=geometric
    kin.qpos[:]=d.qpos;kin.qvel[:]=0;mj.mj_forward(m,kin);ctrl[aa]=arm.last_targets[1]+np.clip(kin.qfrc_bias[va]/m.actuator_gainprm[aa,0],-.25,.25)
    # Explicit coupled finger preshape follows a slow actuator-only ramp while
    # outside the table; selected contact feedback begins after actual alignment.
    s=np.clip((t-5)/5,0,1);s=s*s*(3-2*s)
    for name,target in template['finger_joint_targets'].items():
        j=m.joint(name).id;aid=m.actuator('motor_'+name).id
        if aid not in passive:ctrl[aid]=g['qseq'][step0][int(m.jnt_qposadr[j])]+s*(target-g['qseq'][step0][int(m.jnt_qposadr[j])])
    plan={'active_object_ids_by_hand':[-1,1 if phase in ['acquire','lift']else-1],'phase_by_hand':['idle','grasp'if phase=='lift'else'acquire'if phase=='acquire'else'idle'],'controlled_digits_by_hand':[[],digits],'required_contact_digits_by_hand':[[],required],'digit_force_targets_N_by_hand':[None,targets]}
    if localstep%4==0 or held_finger is None:ctrl=finger.apply(d,ctrl,now,plan);held_finger=ctrl[finger.active_finger_actuator_ids].copy()
    else:ctrl[finger.active_finger_actuator_ids]=held_finger
    ctrl[passive]=0;sonic.last[si]=(ctrl[aa]-sonic.default[si])/sonic.scale[si];d.ctrl[:]=ctrl
    if np.any(d.qfrc_applied)or np.any(d.xfrc_applied):raise ValueError('External force')
    mj.mj_step(m,d);Fdigit={v:0. for v in digits};upforce=0.;tableforce=0.;normals=[]
    for ci,c in enumerate(d.contact[:d.ncon]):
        gs=list(map(int,c.geom));bs=list(map(int,m.geom_bodyid[gs]));F=np.zeros(6);mj.mj_contactForce(m,d,ci,F);contacts.append((step0+localstep,gs,bs,c.pos.copy(),c.frame.copy(),F.copy(),float(c.dist)))
        if obj not in bs:continue
        if any(v in table for v in gs):tableforce+=max(float(F[0]),0.)
        obj2=bs[1]==obj;other=bs[0]if obj2 else bs[1];name=bn[other]
        if name.startswith('rh_'):
            digit=next((v for v in digits if v in name),None)
            if digit:
                Fdigit[digit]+=max(float(F[0]),0.);sign=1 if obj2 else-1;C=np.asarray(c.frame).reshape(3,3);upforce+=float((sign*C.T@F[:3])[2]);normals.append((digit,sign*C[0],float(F[0])))
    opposition=min((float(n1@n2)for v1,n1,f1 in normals for v2,n2,f2 in normals if v1=='thumb'and v2!='thumb'and min(f1,f2)>.05),default=1.)
    ready=opposition<=-.25 and all(Fdigit[v]>=max(.05,.9*float(targets.get(v,0.)))for v in required)and sum(Fdigit.values())>=1.3*m.body_mass[obj]*9.81
    if phase=='align':
        good=np.linalg.norm(arm.palm(d,1)-goal)<.008 and np.linalg.norm(d.qvel[va])<.15
        if good:
            if aligned_since is None:aligned_since=float(d.time)
            if d.time-aligned_since>=.2:closing_started=float(d.time)
        else:aligned_since=None
    if ready:
        if ready_since is None:ready_since=float(d.time)
    else:ready_since=None
    if phase=='acquire'and ready_since is not None and d.time-ready_since+1e-9>=.1:lift_started=float(d.time);holdp=arm.palm(d,1).copy();holdR=d.xmat[arm.hand_bids[1]].reshape(3,3).copy()
    rise=float(d.qpos[oq+2]-settled[2]);proof=ready and rise>=.012 and tableforce<.15 and upforce>=.7*m.body_mass[obj]*9.81
    if proof:
        if proof_since is None:proof_since=float(d.time)
        if d.time-proof_since+1e-9>=.08:verified=True
    else:proof_since=None
    mimic=max(abs(d.qpos[int(m.jnt_qposadr[int(m.eq_obj1id[e])])]-m.eq_data[e,1]*d.qpos[int(m.jnt_qposadr[int(m.eq_obj2id[e])])])for e in range(m.neq)if m.eq_type[e]==mj.mjtEq.mjEQ_JOINT);maxmimic=max(maxmimic,float(mimic));maxpen=max(maxpen,max([-c.dist for c in d.contact]+[0.]));upright=float(d.xmat[sonic.base].reshape(3,3)[2,2]);root=float(np.linalg.norm(d.qpos[:3]-q0[:3]));palmerror=float(np.linalg.norm(arm.palm(d,1)-goal))
    history.append([d.time,rise,tableforce,upforce,opposition,mimic,upright,root,palmerror,*Fdigit.values()]);phases.append(phase);qseq.append(d.qpos.copy());vseq.append(d.qvel.copy());tseq.append(d.time);controls.append(ctrl.copy());torques.append(d.actuator_force.copy());qforces.append(d.qfrc_applied.copy());xforces.append(d.xfrc_applied.copy())
    if localstep%100==0:print(json.dumps({'t':d.time,'phase':phase,'rise_m':rise,'table_N':tableforce,'up_N':upforce,'digits_N':Fdigit,'mimic':mimic,'palm_error_m':palmerror}),flush=True)
    if mimic>.03 or upright<.8 or root>.2 or not np.isfinite(d.qpos).all():stop='protection';break
np.savez_compressed(a.output/'physical_trace.npz',time_s=np.asarray(tseq),qpos=np.asarray(qseq),qvel=np.asarray(vseq),ctrl=np.asarray(controls),actuator_force=np.asarray(torques),initial_state=g['state0'],state_spec=np.asarray(g['spec']),qfrc_applied=np.asarray(qforces),xfrc_applied=np.asarray(xforces))
np.savez_compressed(a.output/'physical_contacts.npz',step=np.asarray([c[0]for c in contacts]),geom=np.asarray([c[1]for c in contacts]),body=np.asarray([c[2]for c in contacts]),point=np.asarray([c[3]for c in contacts]),frame=np.asarray([c[4]for c in contacts]),force=np.asarray([c[5]for c in contacts]),distance=np.asarray([c[6]for c in contacts]))
np.savez_compressed(a.output/'grasp_history.npz',values=np.asarray(history),phase=np.asarray(phases),columns=np.asarray(['time_s','rise_m','table_normal_N','hand_upward_N','opposition_dot','mimic_error_rad','root_up','root_drift_m','palm_error_m','thumb_N','index_N','middle_N','ring_N','pinky_N']))
np.savez_compressed(a.output/'finger_history.npz',**finger.history_arrays())
report={'scope':'Continuous raw-scene push then deep under-rim grasp; diagnostic object1 only, original thin dishes and original hands.','success':verified,'duration_s':float(d.time),'termination':stop,'prefix_q_error':prefix_q_error,'prefix_v_error':prefix_v_error,'post_reset_object_state_writes':0,'external_forces_zero':bool(np.all(np.asarray(qforces)==0)and np.all(np.asarray(xforces)==0)),'max_mimic_error_rad':maxmimic,'max_penetration_m':maxpen,'max_grasp_rise_m':float(np.max(np.asarray(history)[:,1])),'passive_max_abs_actuator_force':float(np.max(abs(np.asarray(torques)[:,passive]))),'hand_max_abs_actuator_force_Nm':float(np.max(abs(np.asarray(torques)[:,finger.active_finger_actuator_ids]))),'continuation_elapsed_wall_s':time.monotonic()-start,'candidate':str(a.candidate),'finger_report':finger.report()}
(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k!='finger_report'}),flush=True)
