"""Independent actuator-only supported plate pre-tilt experiment.

Only joint controls change after reset. Objects initialize at untouched raw0,
settle physically, and move only through real MuJoCo contacts. This script does
not edit canonical SONIC/arm/finger helpers or claim an unsupported grasp.
"""
from pathlib import Path
import argparse,json,hashlib,time,os

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--attempt',type=int,default=3);p.add_argument('--sonic-config',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--seconds',type=float,default=8.);p.add_argument('--target-angle-deg',type=float,default=12.)
    p.add_argument('--angle-force-gain',type=float,default=.65);p.add_argument('--press-speed',type=float,default=.002);p.add_argument('--press-max',type=float,default=.012);p.add_argument('--force-filter-tau',type=float,default=0.);p.add_argument('--press-axis',choices=['vertical','object_wall_normal'],default='vertical');p.add_argument('--anchor',choices=['object_relative','fixed_far_pivot'],default='object_relative');p.add_argument('--arm-integral-gain',type=float,default=0.);a=p.parse_args()
    if os.name!='posix':p.error('Server-only physics')
    if a.output_dir.exists():raise FileExistsError(a.output_dir)
    import numpy as np
    import mujoco as mj
    from scipy.spatial.transform import Rotation
    from scipy.optimize import least_squares
    from fourth_arm_object_feedback import ArmObjectFeedback
    from fourth_sonic_controller import Controller
    source=dict(np.load(a.reference,allow_pickle=False));search=json.loads(a.candidate.read_text());candidate=next(row for row in search['attempts']if row['attempt']==a.attempt)
    if sha(a.model)!=search['model_sha256']:raise ValueError('Candidate/model mismatch')
    model=mj.MjModel.from_xml_path(str(a.model.resolve()));kin=mj.MjData(model);h=ArmObjectFeedback(model,source);chain=h.chains[1]
    names=list(map(str,source['dof_names']));joints=np.array([model.joint(name).id for name in names]);qa=model.jnt_qposadr[joints];aids=np.array([model.actuator('motor_'+name).id for name in names])
    finger_aids=np.array([model.actuator('motor_'+name).id for name in names if name.startswith(('lh_','rh_'))and np.any(model.actuator_gear[model.actuator('motor_'+name).id])])
    if len(finger_aids)!=12 or np.max(abs(model.actuator_forcerange[finger_aids]))>.40000001:raise ValueError('Actual O6 hard motor contract required')
    coupled=[(int(model.eq_obj1id[e]),int(model.eq_obj2id[e]),model.eq_data[e,:5].copy())for e in range(model.neq)if model.eq_type[e]==mj.mjtEq.mjEQ_JOINT]
    if len(coupled)!=10:raise ValueError('Ten O6 mimics required')
    def expand(data):
        for child,parent,coef in coupled:
            value=data.qpos[model.jnt_qposadr[parent]]-model.qpos0[model.jnt_qposadr[parent]]
            data.qpos[model.jnt_qposadr[child]]=model.qpos0[model.jnt_qposadr[child]]+sum(c*value**i for i,c in enumerate(coef))
    def forward(data):mj.mj_kinematics(model,data);mj.mj_comPos(model,data)
    kin.qpos[:]=candidate['qpos'];forward(kin);object_id=h.object_bids[0];oq=h.object_qadr[0]
    object_P=kin.xpos[object_id].copy();object_R=kin.xmat[object_id].reshape(3,3).copy();template_P=object_R.T@(h.palm(kin,1)-object_P);template_R=object_R.T@kin.xmat[h.hand_bids[1]].reshape(3,3)
    ni=model.numeric('source_table_plane_abcd').id;ad=int(model.numeric_adr[ni]);plane=model.numeric_data[ad:ad+4].copy();up=plane[:3]/np.linalg.norm(plane[:3])
    normal=np.asarray(candidate['actual_force_on_object_normal_world']);normal/=np.linalg.norm(normal);press_local=object_R.T@normal;outward=np.r_[-normal[:2],0.];outward/=np.linalg.norm(outward)
    park_P=h.palm(kin,1)+.030*outward+.010*up;park_R=kin.xmat[h.hand_bids[1]].reshape(3,3).copy();original=kin.qpos.copy()
    def park_residual(q):
        kin.qpos[chain['qa']]=q;forward(kin)
        return np.r_[30*(h.palm(kin,1)-park_P),Rotation.from_matrix(kin.xmat[h.hand_bids[1]].reshape(3,3)@park_R.T).as_rotvec()]
    fit=least_squares(park_residual,kin.qpos[chain['qa']],bounds=(chain['lower']+1e-6,chain['upper']-1e-6),max_nfev=120,ftol=1e-10,xtol=1e-10,gtol=1e-10)
    park_residual(fit.x)
    if np.linalg.norm(h.palm(kin,1)-park_P)>.001:raise ValueError('Park pose unreachable')
    park_q=kin.qpos.copy();qref=np.repeat(np.r_[park_q[:7],park_q[qa]][None,:],len(source['time_s']),axis=0);reference=dict(source);reference['pretilt_input_reference_qpos']=source['qpos'];reference['qpos']=qref
    reference['pretilt_nominal_park_only']=np.asarray(True);reference['pretilt_base_adaptation_xyz_wxyz']=park_q[:7].copy()
    config=json.loads(a.sonic_config.read_text());config['hand_object_feedback']=False
    # Constructor changes only the established native body deployment contract.
    sonic=Controller(model,reference,config)
    if np.max(abs(model.actuator_forcerange[finger_aids]))>.40000001:raise ValueError('SONIC changed hard hand cap')
    a.output_dir.mkdir(parents=True);(a.output_dir/'source_pretilt_trial.py').write_bytes(Path(__file__).read_bytes());(a.output_dir/'source_sonic_config.json').write_text(json.dumps(config,indent=2)+'\n')
    np.savez_compressed(a.output_dir/'reference_pretilt_nominal.npz',**reference);mj.mj_saveModel(model,str(a.output_dir/'plant.mjb'))
    data=mj.MjData(model);data.qpos[:7]=park_q[:7];data.qpos[qa]=park_q[qa]
    for oi,adr in enumerate(h.object_qadr):data.qpos[adr:adr+7]=source['object_target_pose'][0,oi]
    mj.mj_forward(model,data);sonic.reset(data)
    spec=int(mj.mjtState.mjSTATE_INTEGRATION);initial=np.empty(mj.mj_stateSize(model,spec));mj.mj_getState(model,data,initial,spec);np.save(a.output_dir/'initial_integration_state.npy',initial)
    body_names=[model.body(b).name or'world'for b in range(model.nbody)];dt=float(model.opt.timestep);warn=np.array([row.number for row in data.warning]);steps=int(np.ceil(a.seconds/dt))
    finger_j=[j for j in joints if model.joint(j).name.startswith(('lh_','rh_'))];finger_motor=[model.actuator('motor_'+model.joint(j).name).id for j in finger_j if np.any(model.actuator_gear[model.actuator('motor_'+model.joint(j).name).id])]
    fixed_controls=park_q[model.jnt_qposadr[[int(model.actuator_trnid[aid,0])for aid in finger_motor]]]
    passive=np.flatnonzero(np.all(model.actuator_gear==0,axis=1));kp=model.actuator_gainprm[chain['aids'],0]
    effective_arm_indices=np.array([sonic.names.index(name)for name in chain['names']]);arm_target=park_q[chain['qa']].copy();press=0.;next_update=0.;last_update=0.;baseline=None;axis=None;local_near=None;pivot_local=None;pivot_world=None;previous_angle=0.;index_up=0.;filtered_index_up=0.;table_up=0.;reason='requested_duration';phase='settle';contact_rows=[]
    ts=[0.];qs=[data.qpos.copy()];vs=[data.qvel.copy()];ctrls=[];taus=[];forces=[];xforces=[];metrics=[];began=time.monotonic()
    pose_error=0.;max_pen=0.;proof_time=0.;arm_integral=np.zeros(7)
    def arm_ik(P,R):
        kin.qpos[:]=data.qpos;kin.qpos[chain['qa']]=arm_target
        for _ in range(12):
            forward(kin);jp=np.zeros((3,model.nv));jr=np.zeros_like(jp);mj.mj_jac(model,kin,jp,jr,h.palm(kin,1),h.hand_bids[1]);J=np.vstack([jp[:,chain['va']],.12*jr[:,chain['va']]])
            e=np.r_[P-h.palm(kin,1),.12*Rotation.from_matrix(R@kin.xmat[h.hand_bids[1]].reshape(3,3).T).as_rotvec()]
            dq=J.T@np.linalg.solve(J@J.T+.0008*np.eye(6),e);kin.qpos[chain['qa']]=np.clip(kin.qpos[chain['qa']]+np.clip(dq,-.08,.08),chain['lower']+1e-5,chain['upper']-1e-5)
        forward(kin);return kin.qpos[chain['qa']].copy(),float(np.linalg.norm(P-h.palm(kin,1)))
    for step in range(steps):
        now=float(data.time);op=data.qpos[oq:oq+3].copy();oR=Rotation.from_quat(data.qpos[oq+3:oq+7][[1,2,3,0]]).as_matrix()
        if baseline is None and now>=2.-1e-9:
            baseline=(op.copy(),oR.copy());radial=oR.T@outward;radial[2]=0;radial/=np.linalg.norm(radial);local_near=radial*.1024559745+np.array([0.,0.,.02]);axis=-np.cross(up,oR@radial);axis/=np.linalg.norm(axis)
            bottom_geom=model.geom('target_object_01_bottom_collision').id;pivot_local=-radial*model.geom_size[bottom_geom,0];pivot_local[2]=model.geom_pos[bottom_geom,2]-model.geom_size[bottom_geom,1];pivot_world=op+oR@pivot_local
        angle=0. if baseline is None else float(Rotation.from_matrix(oR@baseline[1].T).as_rotvec()@axis)
        if now+1e-9>=next_update:
            elapsed=max(now-last_update,.005);last_update=now;next_update+=.02
            if now<2.:phase='settle';target_P=park_P;target_R=park_R
            else:
                target_R=oR@template_R;target_P=op+oR@template_P if a.anchor=='object_relative'else pivot_world+oR@(template_P-pivot_local)
                if now<3.5:
                    phase='approach';fraction=(now-2.)/1.5;s=fraction**3*(10-15*fraction+6*fraction**2);target_P+=(1-s)*(.030*outward+.010*up)
                else:
                    phase='partial_support_tilt';omega=(angle-previous_angle)/elapsed;desired=np.clip(1.00+a.angle_force_gain*(np.deg2rad(a.target_angle_deg)-angle)-.15*omega,.15,1.8)
                    velocity=np.clip(.002*(desired-filtered_index_up),-a.press_speed,a.press_speed);press=float(np.clip(press+elapsed*velocity,0,a.press_max))
                    if angle>np.deg2rad(a.target_angle_deg+3):press=max(0.,press-.001)
                    target_P+=press*(up if a.press_axis=='vertical'else oR@press_local)
                previous_angle=angle
            desired_arm,pose_error=arm_ik(target_P,target_R);arm_target+=np.clip(desired_arm-arm_target,-2.*elapsed,2.*elapsed)
        ctrl=sonic.action(data,now);ctrl[finger_motor]=fixed_controls
        kin.qpos[:]=data.qpos;kin.qvel[:]=0;kin.ctrl[:]=0;mj.mj_forward(model,kin)
        arm_integral=np.clip(arm_integral+a.arm_integral_gain*dt*(arm_target-data.qpos[chain['qa']]),-.08,.08)
        ctrl[chain['aids']]=arm_target+np.clip(kin.qfrc_bias[chain['va']]/kp,-.25,.25)+arm_integral
        sonic.last[effective_arm_indices]=(ctrl[chain['aids']]-sonic.default[effective_arm_indices])/sonic.scale[effective_arm_indices]
        ctrl[passive]=0;data.ctrl[:]=ctrl
        if np.any(data.qfrc_applied)or np.any(data.xfrc_applied):raise ValueError('External force forbidden')
        mj.mj_step(model,data)
        if not np.isfinite(data.qpos).all()or np.any(np.array([row.number for row in data.warning])>warn)or abs(data.time-(step+1)*dt)>1e-7:reason='solver_failure';break
        index_force=np.zeros(3);table_force=np.zeros(3);max_pen=0.;nindex=0
        for ci,c in enumerate(data.contact[:data.ncon]):
            g1,g2=int(c.geom1),int(c.geom2);b1,b2=int(model.geom_bodyid[g1]),int(model.geom_bodyid[g2]);max_pen=max(max_pen,-float(c.dist))
            if object_id not in [b1,b2]:continue
            other=b2 if b1==object_id else b1;sign=1 if b2==object_id else-1;w=np.zeros(6);mj.mj_contactForce(model,data,ci,w);f=sign*np.asarray(c.frame).reshape(3,3).T@w[:3]
            if body_names[other].startswith('rh_index'):index_force+=f;nindex+=1
            if any(model.geom(g).name.startswith('table_support')for g in [g1,g2]):table_force+=f
            contact_rows.append((step,g1,g2,*c.pos,*f,float(c.dist)))
        index_up=float(index_force@up);alpha=1. if a.force_filter_tau<=0 else dt/(a.force_filter_tau+dt);filtered_index_up+=(index_up-filtered_index_up)*alpha
        table_up=float(table_force@up);upbase=float(Rotation.from_quat(data.qpos[[4,5,6,3]]).as_matrix()[2,2])
        op=data.qpos[oq:oq+3];oR=Rotation.from_quat(data.qpos[oq+3:oq+7][[1,2,3,0]]).as_matrix();angle=0. if baseline is None else float(Rotation.from_matrix(oR@baseline[1].T).as_rotvec()@axis)
        near_rise=0. if baseline is None else float((op+oR@local_near-baseline[0]-baseline[1]@local_near)@up)
        comrise=0. if baseline is None else float((op+oR@model.body_ipos[object_id]-baseline[0]-baseline[1]@model.body_ipos[object_id])@up)
        proven=angle>=np.deg2rad(6)and near_rise>=.018 and table_up>.2 and index_up>.2 and upbase>.9
        proof_time=proof_time+dt if proven else 0.
        kin.qpos[:]=data.qpos;forward(kin);tracking_error=float(np.linalg.norm(h.palm(kin,1)-target_P))
        metrics.append([index_up,table_up,angle,near_rise,comrise,upbase,press,pose_error,float(nindex),max_pen,proof_time,filtered_index_up,tracking_error]);ts.append(float(data.time));qs.append(data.qpos.copy());vs.append(data.qvel.copy());ctrls.append(ctrl.copy());taus.append(data.actuator_force.copy());forces.append(data.qfrc_applied.copy());xforces.append(data.xfrc_applied.copy())
        if step%200==0:print(json.dumps({'time':float(data.time),'phase':phase,'index_up_N':index_up,'table_up_N':table_up,'tilt_deg':np.rad2deg(angle),'near_rise_mm':near_rise*1000,'press_mm':press*1000,'base_up':upbase,'FK_error_mm':pose_error*1000}),flush=True)
        if upbase<.85 or data.qpos[2]<park_q[2]-.18:reason='robot_fall';break
        if baseline is not None and np.linalg.norm(op[:2]-baseline[0][:2])>.03:reason='object_slid_over_3cm';break
        if np.max(abs(data.actuator_force[finger_aids]))>.400001:reason='hand_torque_cap_failure';break
    metric=np.asarray(metrics);qarray=np.asarray(qs);tarray=np.asarray(ts)
    np.savez_compressed(a.output_dir/'physical_trace.npz',time=tarray,qpos=qarray,qvel=np.asarray(vs),ctrl=np.asarray(ctrls),actuator_force=np.asarray(taus),qfrc_applied=np.asarray(forces),xfrc_applied=np.asarray(xforces),metrics=metric,
        metric_names=np.asarray(['index_up_N','table_up_N','tilt_rad','near_rim_rise_m','object_COM_rise_m','robot_up','upward_press_offset_m','palm_FK_error_m','index_contacts','max_penetration_m','consecutive_pretilt_proof_s','filtered_index_up_N','actual_palm_target_error_m']),object_ids=np.arange(1,7),object_contacts=np.asarray(contact_rows))
    if len(qs)>1:
        final=np.empty(mj.mj_stateSize(model,spec));mj.mj_getState(model,data,final,spec);np.save(a.output_dir/'final_actual_integration_state.npy',final)
    passed=bool(len(metric)and np.max(metric[:,10])>=.20)
    report={'schema':'fourth_actuator_partial_pretilt_trial.v1','termination_reason':reason,'physical_time_s':float(data.time),'wall_time_s':time.monotonic()-began,'actual_supported_pretilt_proven':passed,
        'unsupported_grasp_or_lift_success_claimed':False,'source_objects_only_assigned_once_at_reset':True,'external_forces_ever_nonzero':bool(np.any(forces)or np.any(xforces)),
        'model':str(a.model),'model_sha256':sha(a.model),'source_reference_sha256':sha(a.reference),'candidate':str(a.candidate),'candidate_attempt':a.attempt,'base_xyz_wxyz':park_q[:7].tolist(),
        'raw_initial_object_poses_preserved':True,'target_angle_deg':a.target_angle_deg,'max_actual_tilt_deg':float(np.max(metric[:,2])*180/np.pi)if len(metric)else None,
        'max_actual_near_rim_rise_m':float(np.max(metric[:,3]))if len(metric)else None,'max_actual_index_up_force_N':float(np.max(metric[:,0]))if len(metric)else None,
        'max_consecutive_supported_tilt_s':float(np.max(metric[:,10]))if len(metric)else 0.,'maximum_actual_hand_motor_torque_Nm':float(np.max(abs(np.asarray(taus)[:,finger_aids])))if len(taus)else None,
        'control_parameters':{'angle_force_gain_N_rad':a.angle_force_gain,'press_speed_m_s':a.press_speed,'press_max_m':a.press_max,'force_filter_tau_s':a.force_filter_tau,'press_axis':a.press_axis,'press_direction_object_local':press_local.tolist(),'initial_press_direction_world':normal.tolist(),'anchor':a.anchor,'far_pivot_object_local':None if pivot_local is None else pivot_local.tolist(),'far_pivot_world_m':None if pivot_world is None else pivot_world.tolist(),'arm_integral_gain_s_inv':a.arm_integral_gain,'arm_integral_bound_rad':.08,'arm_integral_final_rad':arm_integral.tolist()},
        'controller':'Frozen native SONIC balance + independent actual-object arm IK/up-force PI, position motors only','canonical_controller_sha256':sha(Path(__file__).with_name('fourth_sonic_controller.py')),
        'reference_scope':'Nominal parked native stand, not reconstructed six-object action','future_handoff_note':'Any later insertion must continue actual actuator-generated state/history; a private tilted initializer is not a successful pre-tilt.'}
    (a.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
