"""Short diagnostic for the new physical WBC; never a full-task acceptance."""
import argparse,hashlib,json,sys,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=float,default=2.)
    p.add_argument('--start-source-s',type=float,default=0.);p.add_argument('--hold-reference',action='store_true')
    p.add_argument('--initial-velocity',choices=['zero','reference'],default='reference')
    p.add_argument('--all-dof-acceleration-bounds',action='store_true',help='Diagnostic reproduction of the original artificial all-DOF acceleration box')
    p.add_argument('--planned-contacts-only',action='store_true',help='Diagnostic reproduction of source-label contact deletion')
    p.add_argument('--wbc-config',type=Path)
    p.add_argument('--implicit-damping-compensation',action='store_true')
    a=p.parse_args()
    if sys.platform!='linux':p.error('Run physical simulation on the authorized server only')
    if a.output.exists():raise FileExistsError(a.output)
    import numpy as np,mujoco as mj
    from rgb145_wbc_controller import WBCController,implicit_velocity_damping_diagonal
    from rgb145_physics import contacts,descendants
    z=dict(np.load(a.reference,allow_pickle=False));m=mj.MjModel.from_xml_path(str(a.model.resolve()));d=mj.MjData(m)
    cfg=json.loads(a.wbc_config.read_text())if a.wbc_config else {}
    cfg['hold_reference']=a.hold_reference
    for k in ('all_dof_acceleration_bounds','planned_contacts_only','implicit_damping_compensation'):
        if getattr(a,k):cfg[k]=True
        else:cfg.setdefault(k,False)
    c=WBCController(m,z,cfg);start=float(np.interp(a.start_source_s,z['source_time_s'],z['time_s']))
    q,v,_,_=c.sample(start);d.qpos[:]=q;d.qvel[:]=v if a.initial_velocity=='reference'else 0.
    mj.mj_forward(m,d);c.reset(d);state_spec=int(mj.mjtState.mjSTATE_INTEGRATION)
    initial=np.zeros(mj.mj_stateSize(m,state_spec));mj.mj_getState(m,d,initial,state_spec)
    foot=[descendants(m,int(b))for b in c.feet];support=c.support
    a.output.mkdir(parents=True);mj.mj_saveModel(m,str(a.output/'model.mjb'))
    t=[0.];qs=[d.qpos.copy()];vs=[d.qvel.copy()];ctrl=[];force=[];errors=[];supportforce=[];seat=[];depth=0.;predicted_acc=[];actual_acc=[];predicted_contact=[];requested_tau=[]
    warnings=np.array([w.number for w in d.warning]);began=time.monotonic();reason='requested_duration'
    integrator_predictions={k:[]for k in ('continuous','joint_damping_only','all_affine_damping','unsaturated_affine_damping')}
    for step in range(int(np.ceil(a.seconds/m.opt.timestep))):
        phase=start if a.hold_reference else start+d.time
        u=c.action(d,phase);d.ctrl[:]=u
        if np.any(d.qfrc_applied)or np.any(d.xfrc_applied):raise AssertionError('Forbidden external force')
        ctrl.append(u.copy());predicted_acc.append(c.last_qp_ddq.copy());predicted_contact.append(c.last_qp_contact_force_world.copy());requested_tau.append(c.last.copy())
        previous_velocity=d.qvel.copy();mj.mj_step(m,d);actual_acc.append((d.qvel-previous_velocity)/m.opt.timestep)
        # Read only the real step's actual generalized force and original mass.
        # This isolates integration from the QP's approximate contact predictions.
        true_mass=np.zeros((m.nv,m.nv))
        try:mj.mj_fullM(m,true_mass,d.qM)
        except TypeError:mj.mj_fullM(m,d,true_mass)
        real_rhs=d.qfrc_smooth+d.qfrc_constraint
        integrator_predictions['continuous'].append(np.linalg.solve(true_mass,real_rhs))
        for name,mode in (('joint_damping_only','none'),('all_affine_damping','all'),('unsaturated_affine_damping','unsaturated')):
            damping,_=implicit_velocity_damping_diagonal(m,d,c.powered,mode)
            integrator_predictions[name].append(np.linalg.solve(true_mass+m.opt.timestep*np.diag(damping),real_rhs))
        t.append(float(d.time));qs.append(d.qpos.copy());vs.append(d.qvel.copy());force.append(d.actuator_force.copy())
        target,_,_,_=c.sample(start if a.hold_reference else start+d.time)
        angle=float(2*np.arccos(np.clip(abs(np.dot(d.qpos[3:7],target[3:7])),0,1)))
        errors.append([float(np.linalg.norm(d.qpos[:2]-target[:2])),abs(float(d.qpos[2]-target[2])),angle])
        _,sf,_,pen,se=contacts(m,d,step,foot,support);supportforce.append(sf);seat.append(se);depth=max(depth,pen)
        if np.any(np.array([w.number for w in d.warning])>warnings)or not np.isfinite(d.qpos).all():reason='solver_failure';break
        if angle>1.8:reason='orientation_safety_stop';break
    np.savez_compressed(a.output/'trace.npz',time=np.asarray(t),qpos=np.asarray(qs),qvel=np.asarray(vs),ctrl=np.asarray(ctrl),actuator_force=np.asarray(force),initial_state=initial,state_spec=np.asarray(state_spec),root_errors=np.asarray(errors),foot_support_force_N=np.asarray(supportforce),seat_support_force_N=np.asarray(seat),qp_predicted_ddq=np.asarray(predicted_acc),actual_finite_difference_ddq=np.asarray(actual_acc),qp_predicted_net_scene_force_world_N=np.asarray(predicted_contact),qp_requested_body_actuator_force=np.asarray(requested_tau),**{'actual_rhs_predicted_ddq_'+k:np.asarray(v)for k,v in integrator_predictions.items()})
    report={'mode':'experimental_wbc_phase_probe_not_full_acceptance','simulation_passed':False,'start_source_time_s':a.start_source_s,'execution_start_s':start,'hold_reference':a.hold_reference,'duration_s':t[-1],'end_reason':reason,'max_root_errors_xy_z_angle':np.max(errors,axis=0).tolist(),'max_penetration_m':depth,'controller':c.report(),'wall_s':time.monotonic()-began,'initial_state_reset_only':True,'external_force_writes':0,'model_sha256':hashlib.sha256(a.model.read_bytes()).hexdigest(),'reference_sha256':hashlib.sha256(a.reference.read_bytes()).hexdigest()}
    measured=np.asarray(actual_acc);report['actual_rhs_integrator_comparison']={}
    for name,values in integrator_predictions.items():
        diff=np.asarray(values)-measured
        report['actual_rhs_integrator_comparison'][name]={
            'root6_rmse':float(np.sqrt(np.mean(diff[:,:6]**2))),
            'body28_rmse':float(np.sqrt(np.mean(diff[:,c.va]**2))),
            'all_dof_rmse':float(np.sqrt(np.mean(diff**2))),
            'max_absolute_error':float(np.max(abs(diff))),
            'first_100ms_body28_rmse':float(np.sqrt(np.mean(diff[:min(len(diff),20),c.va]**2)))}
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');(a.output/'qp_diagnostics.json').write_text(json.dumps(c.diagnostics,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
