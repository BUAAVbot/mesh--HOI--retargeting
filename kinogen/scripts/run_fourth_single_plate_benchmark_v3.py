"""Resume the continuous C4 prefix, then low-preload five-digit tactile align.

No physical state is assigned after raw reset in the nested push program.
Loaded lift requires actual five-digit contact and actual-point wrench LP.
"""
from pathlib import Path
import os
import argparse,contextlib,json,runpy,shutil,sys,time
import numpy as np
import mujoco as mj
from scipy.spatial.transform import Rotation
from fourth_arm_object_feedback import ArmObjectFeedback
from fourth_o6_load_hysteresis_cached_c10 import O6CoupledReferenceContactFeedback
from fourth_contact_wrench_balanced_c9 import contact_columns,wrench_lp
R=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=float,default=32.);p.add_argument('--orientation-weight',type=float,default=.12);p.add_argument('--closure-rate',type=float,default=.03);p.add_argument('--force-gain',type=float,default=.00005);p.add_argument('--waist-hold',action='store_true');p.add_argument('--setup',type=Path,required=True);p.add_argument('--setup-model',type=Path);a=p.parse_args()
a.body_control=os.environ.get('KINOGEN_BENCH_BODY_CONTROL','sonic')
if a.body_control not in ['sonic','pd','pd_all_gff','pd_groundaware']:raise ValueError('Unknown body control')
if a.output.exists():raise FileExistsError(a.output)
a.output.mkdir(parents=True);candidate=json.loads(a.candidate.read_text());best=candidate['best'];template=best['template'];(a.output/'source_candidate.json').write_text(json.dumps(candidate,indent=2)+'\n')
for name in [Path(__file__).name,'fourth_single_plate_benchmark_setup.py','fourth_sonic_controller.py','fourth_arm_object_feedback.py','fourth_contact_feedback_o6_coupled.py','fourth_contact_feedback_segmented.py','fourth_contact_feedback_tableguard.py','fourth_contact_feedback.py','fourth_convex_separation.py','fourth_contact_wrench.py','fourth_o6_load_hysteresis_c7.py','fourth_contact_wrench_balanced_c9.py']:
 shutil.copy2(Path(__file__).with_name(name),a.output/('source_'+name))
from fourth_single_plate_benchmark_groundaware import prepare
g=prepare(a.setup,a.output,a.setup_model,a.body_control)
m=g['m'];d=g['d'];sonic=g['sonic'];t0=float(d.time);qerr=verr=None
print(json.dumps({'phase':'controlled_reset_settled','time_s':t0}),flush=True)
oldtimes=sonic.times.copy();oldq=sonic.qref.copy();keep=oldtimes<t0-1e-9;nt=t0+np.arange(0,int(np.ceil(a.seconds/.02))+1)*.02;sonic.times=np.r_[oldtimes[keep],nt];sonic.qref=np.r_[oldq[keep],np.tile(oldq[-1],(len(nt),1))];sonic.body_q=sonic.qref[:,7+sonic.jref];sonic.body_dq=np.gradient(sonic.body_q,sonic.times,axis=0)
if np.any(np.diff(sonic.times)<=0):raise ValueError('Reference clock must remain monotonic')
ref=g['ref'];arm=ArmObjectFeedback(m,ref,{'initial_geometric_seed':'actual_state','ik_iterations':20,'orientation_weight_m':a.orientation_weight,'arm_speed_rad_s':.8});cfg=dict(g['fcfg']);cfg['load_preload_normal_force_N']=.2;cfg['closure_rate_rad_s']=a.closure_rate;cfg['force_to_gap_gain_m_per_N']=a.force_gain;cfg['release_rate_rad_s']=a.closure_rate;cfg['load_force_ramp_s']=.8;finger=O6CoupledReferenceContactFeedback(m,ref,cfg)
(a.output/'finger_config.json').write_text(json.dumps(cfg,indent=2)+'\n');mj.mj_saveModel(m,str(a.output/'effective_model.mjb'),None)
chain=arm.chains[1];qa=chain['qa'];va=chain['va'];aa=chain['aids'];si=np.asarray([sonic.names.index(v)for v in chain['names']]);kin=mj.MjData(m);obj=arm.object_bids[0];oq=arm.object_qadr[0];table=set(arm.table_geoms);passive=g['passive'];bn=[m.body(b).name or''for b in range(m.nbody)]
qstart=d.qpos.copy();pstart=arm.palm(d,1).copy();Rstart=d.xmat[arm.hand_bids[1]].reshape(3,3).copy();object_start=d.qpos[oq:oq+7].copy();digits=['thumb','index','middle','ring','pinky'];required=list(template.get('required_contact_digits',digits));targets={v:1.05*float(template.get('digit_force_targets_N',{}).get(v,1.5))for v in digits};P=np.zeros((m.nv,6));hand=finger.hands[1];P[hand['va']]=hand['coupling_matrix']
waist_joints=[j for j in range(m.njnt)if(m.joint(j).name or'').startswith('waist')];wqa=m.jnt_qposadr[waist_joints];wva=m.jnt_dofadr[waist_joints];waids=np.asarray([m.actuator('motor_'+m.joint(j).name).id for j in waist_joints]);wsi=np.asarray([sonic.names.index(m.joint(j).name)for j in waist_joints]);whold=d.qpos[wqa].copy()
left=arm.chains[0];lqa=left['qa'];lva=left['va'];laids=left['aids'];lsi=np.asarray([sonic.names.index(v)for v in left['names']]);lhold=d.qpos[lqa].copy()
if a.waist_hold:
 ids=np.flatnonzero(sonic.times>=t0-.02);sonic.qref[np.ix_(ids,wqa)]=whold[None,:];sonic.body_q[np.ix_(ids,wsi)]=whold[None,:];sonic.body_dq[np.ix_(ids,wsi)]=0.;sonic.qref[np.ix_(ids,lqa)]=lhold[None,:];sonic.body_q[np.ix_(ids,lsi)]=lhold[None,:];sonic.body_dq[np.ix_(ids,lsi)]=0.
qseq=g['qseq'];vseq=g['vseq'];tseq=g['tseq'];controls=g['controls'];torques=g['torques'];qforces=g['qforces'];xforces=g['xforces'];contacts=g['contacts'];step0=len(controls);history=[];phases=[];lplog=[];maxmimic=g['maxmimic'];maxpen=g['maxpen'];held=None;lp_ok=False;lastlp=-1.;load_started=None;ready_since=None;lift_started=None;holdp=None;holdR=None;proof_since=None;verified=False;locked_palm=None;lost_since=None;last_good_lp=-1e9;last_seen=np.full(5,-np.inf);longest_proof=0.;measured_support_checks=0;stop='duration';start=time.monotonic()
table_plane=np.array([.010271048830894367,.02292500590934548,.999684425036206,1.0058665495441206]);table_plane/=np.linalg.norm(table_plane[:3])
object_shapes=[]
for og in arm.object_geoms[0]:
 if m.geom_type[og]==mj.mjtGeom.mjGEOM_MESH:
  mid=int(m.geom_dataid[og]);adr=int(m.mesh_vertadr[mid]);num=int(m.mesh_vertnum[mid]);object_shapes.append((og,m.mesh_vert[adr:adr+num].copy()))
 else:object_shapes.append((og,None))
def lowest_object_clearance():
 values=[];normal=table_plane[:3]
 for og,vertices in object_shapes:
  G=d.geom_xmat[og].reshape(3,3);center=d.geom_xpos[og]
  if vertices is not None:values.append(float(np.min((vertices@G.T+center)@normal+table_plane[3])))
  else:
   projection=abs(float(G[:,2]@normal));size=m.geom_size[og]
   if m.geom_type[og]==mj.mjtGeom.mjGEOM_CYLINDER:values.append(float(center@normal+table_plane[3]-size[1]*projection-size[0]*np.sqrt(max(0,1-projection*projection))))
   else:raise ValueError('Unsupported dynamic object collider in clearance proof')
 return min(values)
for localstep in range(int(np.ceil(a.seconds/m.opt.timestep))):
 now=float(d.time);t=now-t0
 if arm.last_targets[1]is not None:
  ids=np.flatnonzero(sonic.times>=now-.02);sonic.body_q[np.ix_(ids,si)]=arm.last_targets[1][None,:];sonic.body_dq[np.ix_(ids,si)]=0.;sonic.qref[np.ix_(ids,qa)]=arm.last_targets[1][None,:]
 ctrl=sonic.action(d,now);pose=d.qpos[oq:oq+7];OR=arm.rot(pose[3:]);want=pose[:3]+OR@np.asarray(template['palm_offset_in_object_m']);wantR=OR@arm.rot(template['hand_rotation_in_object_wxyz']);s=np.clip(t/4.,0,1);s=s*s*(3-2*s)
 if t<2.:
  phase='descend_outside';out=np.array([.3521665876,.9359373348,-.02508]);out/=np.linalg.norm(out)
  alpha=np.clip(t/2.,0,1);alpha=alpha*alpha*(3-2*alpha)
  goal=pstart+alpha*(want+out*.08-pstart);rotation=wantR
 elif t<6.:
  phase='insert_under';out=np.array([.3521665876,.9359373348,-.02508]);out/=np.linalg.norm(out)
  alpha=np.clip((t-2.)/4.,0,1);alpha=alpha*alpha*(3-2*alpha)
  goal=want+out*.08*(1-alpha);rotation=wantR
 elif lift_started is None:
  phase='tactile_align'if not lp_ok else'load'
  if locked_palm is None:goal=pstart+s*(want-pstart);rotation=Rotation.from_rotvec(s*Rotation.from_matrix(wantR@Rstart.T).as_rotvec()).as_matrix()@Rstart
  else:goal=locked_palm[0];rotation=locked_palm[1]
 else:phase='lift';goal=holdp+np.array([0,0,min(.08,.012*(now-lift_started))]);rotation=holdR
 if localstep%4==0:geom,error=arm.solve(d,1,goal,rotation,ctrl);arm.last_targets[1]=geom
 kin.qpos[:]=d.qpos;kin.qvel[:]=0;mj.mj_forward(m,kin);ctrl[aa]=arm.last_targets[1]+np.clip(kin.qfrc_bias[va]/m.actuator_gainprm[aa,0],-.25,.25)
 if a.waist_hold:
  kin.qpos[wqa]=whold;kin.qpos[lqa]=lhold;mj.mj_forward(m,kin);ctrl[waids]=whold+kin.qfrc_bias[wva]/m.actuator_gainprm[waids,0];ctrl[laids]=lhold+kin.qfrc_bias[lva]/m.actuator_gainprm[laids,0];sonic.last[wsi]=(ctrl[waids]-sonic.default[wsi])/sonic.scale[wsi];sonic.last[lsi]=(ctrl[laids]-sonic.default[lsi])/sonic.scale[lsi]
 for name,value in template['finger_joint_targets'].items():
  j=m.joint(name).id;aid=m.actuator('motor_'+name).id
  if aid not in passive:ctrl[aid]=qstart[int(m.jnt_qposadr[j])]+s*(value-qstart[int(m.jnt_qposadr[j])])
 goals=targets if lp_ok else{v:min(.25,targets[v])for v in digits};plan={'active_object_ids_by_hand':[-1,1],'phase_by_hand':['idle','grasp'if lift_started is not None else'acquire'],'controlled_digits_by_hand':[[],digits],'required_contact_digits_by_hand':[[],required],'digit_force_targets_N_by_hand':[None,goals]}
 if t<6.:
  plan['active_object_ids_by_hand']=[-1,-1];plan['phase_by_hand']=['idle','idle']
 if localstep%4==0 or held is None:ctrl=finger.apply(d,ctrl,now,plan);held=ctrl[finger.active_finger_actuator_ids].copy()
 else:ctrl[finger.active_finger_actuator_ids]=held
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
 all_contact=all(Fdigit[v]>.05 for v in required)
 rawforces=np.asarray([Fdigit[v]for v in digits]);last_seen[rawforces>.03]=now
 if localstep%4==0:
  finger._kin.qpos[:]=d.qpos;finger._kin.qvel[:]=d.qvel;mj.mj_kinematics(m,finger._kin);mj.mj_comPos(m,finger._kin)
  cached_actual_gaps,_=finger._geometry(finger.hands[1],0)
 gaps=cached_actual_gaps
 recent=all(now-last_seen[digits.index(v)]<=.100000001 and gaps[digits.index(v)]<.002 for v in required)
 if t>=6. and now-lastlp>=.019999999:
  items=contact_columns(m,d,obj,tuple('rh_'+v for v in digits),P,max_gap_m=0.);kin.qpos[:]=d.qpos;kin.qvel[:]=0;mj.mj_forward(m,kin)
  lp=wrench_lp(items['columns'],items['normal_rows'],np.r_[m.body_mass[obj]*m.opt.gravity,np.zeros(3)],normal_cap_N=4.6,motor_torque_matrix=items['motor_torque_matrix'],motor_gravity_bias_Nm=P.T@kin.qfrc_bias,motor_cap_Nm=.35);lastlp=now
  if lp['feasible']:
   scaled_torque=P.T@kin.qfrc_bias-np.asarray(items['motor_torque_matrix'])@(1.08*np.asarray(lp['ray_weights']))
   lp['scaled_load_max_motor_Nm']=float(np.max(abs(scaled_torque)))
   if lp['scaled_load_max_motor_Nm']>.4:lp['feasible']=False;lp['scaled_load_rejected']=True
  if lp['feasible']:
   last_good_lp=now
   if locked_palm is None:locked_palm=(arm.palm(d,1).copy(),d.xmat[arm.hand_bids[1]].reshape(3,3).copy())
   desired={v:max(.05,1.08*float(x))for v,x in zip(digits,lp['normal_per_digit_N'])}
   if sum(desired.values())<=float(cfg['max_hand_normal_force_N']) and max(desired.values())<=float(cfg['max_force_per_digit_N']):
    targets=desired
  lplog.append({'time_s':now,'LP':lp,'patches':items['patches'],'actual_derived_targets_N':targets.copy()})
 lp_ok=bool(recent and now-last_good_lp<=.100000001)
 if recent:lost_since=None
 else:
  if lost_since is None:lost_since=now
  if now-lost_since>.15:locked_palm=None
 opposition=min((float(n1@n2)for v1,n1,f1 in normals for v2,n2,f2 in normals if v1=='thumb'and v2!='thumb'and min(f1,f2)>.05),default=1.)
 measured_ok=False
 if all_contact and sum(Fdigit.values())>=1.3*m.body_mass[obj]*9.81:
  measured_items=contact_columns(m,d,obj,tuple('rh_'+v for v in digits),P,max_gap_m=0.)
  kin.qpos[:]=d.qpos;kin.qvel[:]=0;mj.mj_forward(m,kin)
  measured_lp=wrench_lp(measured_items['columns'],measured_items['normal_rows'],np.r_[m.body_mass[obj]*m.opt.gravity,np.zeros(3)],normal_cap_N=.95*np.asarray([Fdigit[v]for v in digits]),motor_torque_matrix=measured_items['motor_torque_matrix'],motor_gravity_bias_Nm=P.T@kin.qfrc_bias,motor_cap_Nm=.4)
  measured_ok=bool(measured_lp['feasible']);measured_support_checks+=int(measured_ok)
 ready=lp_ok and all_contact and opposition<=-.25 and measured_ok
 if ready:
  if ready_since is None:ready_since=float(d.time)
 else:ready_since=None
 if lift_started is None and ready_since is not None and d.time-ready_since+1e-9>=.1:lift_started=float(d.time);holdp=arm.palm(d,1).copy();holdR=d.xmat[arm.hand_bids[1]].reshape(3,3).copy()
 rise=float(d.qpos[oq+2]-object_start[2]);clearance=lowest_object_clearance();proof=ready and clearance>=.03 and tableforce<.15 and upforce>=.7*m.body_mass[obj]*9.81
 if proof:
  if proof_since is None:proof_since=float(d.time)
  longest_proof=max(longest_proof,float(d.time-proof_since))
  if d.time-proof_since+1e-9>=2.:verified=True
 else:proof_since=None
 mimic=max(abs(d.qpos[int(m.jnt_qposadr[int(m.eq_obj1id[e])])]-m.eq_data[e,1]*d.qpos[int(m.jnt_qposadr[int(m.eq_obj2id[e])])])for e in range(m.neq)if m.eq_type[e]==mj.mjtEq.mjEQ_JOINT);maxmimic=max(maxmimic,float(mimic));maxpen=max(maxpen,max([-c.dist for c in d.contact]+[0.]));upright=float(d.xmat[sonic.base].reshape(3,3)[2,2]);root=float(np.linalg.norm(d.qpos[:3]-g['q0'][:3]));palmerror=float(np.linalg.norm(arm.palm(d,1)-goal))
 history.append([d.time,rise,tableforce,upforce,opposition,mimic,upright,root,palmerror,*Fdigit.values(),lp_ok,measured_ok,clearance]);phases.append(phase);qseq.append(d.qpos.copy());vseq.append(d.qvel.copy());tseq.append(d.time);controls.append(ctrl.copy());torques.append(d.actuator_force.copy());qforces.append(d.qfrc_applied.copy());xforces.append(d.xfrc_applied.copy())
 if localstep%100==0:print(json.dumps({'t':d.time,'phase':phase,'digit_N':Fdigit,'LP':lp_ok,'rise':rise,'up_N':upforce,'table_N':tableforce,'mimic':mimic,'palm_error_m':palmerror,'measured_budget_LP':measured_ok,'min_clearance_m':clearance}),flush=True)
 if mimic>.03 or upright<.8 or root>.2 or not np.isfinite(d.qpos).all():stop='mimic_protection'if mimic>.03 else'body_or_numerical_protection';break
np.savez_compressed(a.output/'physical_trace.npz',time_s=np.asarray(tseq),qpos=np.asarray(qseq),qvel=np.asarray(vseq),ctrl=np.asarray(controls),actuator_force=np.asarray(torques),initial_state=g['g']['state0'],state_spec=np.asarray(g['g']['spec']),qfrc_applied=np.asarray(qforces),xfrc_applied=np.asarray(xforces))
np.savez_compressed(a.output/'physical_contacts.npz',step=np.asarray([c[0]for c in contacts]),geom=np.asarray([c[1]for c in contacts]),body=np.asarray([c[2]for c in contacts]),point=np.asarray([c[3]for c in contacts]),frame=np.asarray([c[4]for c in contacts]),force=np.asarray([c[5]for c in contacts]),distance=np.asarray([c[6]for c in contacts]))
np.savez_compressed(a.output/'grasp_history.npz',values=np.asarray(history),phase=np.asarray(phases),columns=np.asarray(['time_s','rise_m','table_normal_N','hand_upward_N','opposition_dot','mimic_error_rad','root_up','root_drift_m','palm_error_m','thumb_N','index_N','middle_N','ring_N','pinky_N','actual_contact_wrench_feasible','measured_normal_budget_wrench_feasible','object_min_clearance_from_table_plane_m']))
np.savez_compressed(a.output/'finger_history.npz',**finger.history_arrays());(a.output/'actual_wrench_checks.json').write_text(json.dumps(lplog,indent=2)+'\n')
report={'scope':'Controlled table-edge reset benchmark: plate1 initial position explicitly shifted; free SONIC robot and original hand geometry; not source raw-scene reproduction.','success':verified,'duration_s':float(d.time),'termination':stop,'source_raw_prefix_not_run':True,'setup_report':'settle_report.json','post_reset_object_state_writes':0,'external_forces_zero':bool(np.all(np.asarray(qforces)==0)and np.all(np.asarray(xforces)==0)),'max_mimic_error_rad':maxmimic,'max_penetration_m':maxpen,'max_tactile_rise_m':float(np.max(np.asarray(history)[:,1])),'passive_max_abs_actuator_force':float(np.max(abs(np.asarray(torques)[:,passive]))),'hand_max_abs_actuator_force_Nm':float(np.max(abs(np.asarray(torques)[:,finger.active_finger_actuator_ids]))),'stable_3cm_support_max_duration_s':longest_proof,'load_contact_grace_s':.1,'actual_wrench_check_hz':50,'palm_loading_policy':'Lock actual world palm/R at first actual full-wrench feasible contact; release latch after150ms real contact loss','actual_LP_budget_Nm':.39,'load_targets_from_actual_LP':True,'measured_normal_budget_feasible_steps':measured_support_checks,'prelift_readiness':'Actual current contact LP feasible with each digit normal capped at95pct of measured normal force; raw required contact and1.3mg total retained','proof_height':'Lowest actual object collision geometry30mm above fitted tabletop plane, not object origin','actual_LP_feasible_checks':sum(v['LP']['feasible']for v in lplog),'target_forces_N':targets,'waist_hold':a.waist_hold,'left_arm_actual_joint_hold':a.waist_hold,'waist_and_left_arm_gFF':'Exact private static bias/gain, physical original force limits preserved; no arbitrary target-offset cap','waist_geometric_target':whold.tolist(),'IK_orientation_weight_m':a.orientation_weight,'force_to_gap_gain_m_per_N':a.force_gain,'closure_rate_rad_s':a.closure_rate,'elapsed_tactile_wall_s':time.monotonic()-start,'finger_report':finger.report()}
report.update(scope='Explicit table-edge reset benchmark; free robot and original hand geometry; body controller declared in settle_report.json, not source raw reproduction.',body_controller=a.body_control,learned_leg_policy_used=a.body_control=='sonic',actual_LP_budget_Nm=.35,load_ray_scale=1.08,scaled_motor_budget_rechecked=True)
(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k!='finger_report'}),flush=True)
