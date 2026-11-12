"""Short fixed-control numerical constraint comparison, not a grasp rollout.

Reconstruct the true integration state by replaying the raw-reset trace up to
59.5s. Branch only hand-joint equality solimp, keeping collision/actuation fixed.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import mujoco as mj
p=argparse.ArgumentParser();p.add_argument('--trial',type=Path,required=True);p.add_argument('--start',type=float,default=59.5);p.add_argument('--output',type=Path,required=True);a=p.parse_args();z=dict(np.load(a.trial/'physical_trace.npz'));m=mj.MjModel.from_binary_path(str((a.trial/'effective_model.mjb').resolve()));d=mj.MjData(m);spec=int(z['state_spec']);mj.mj_setState(m,d,z['initial_state'],spec);mj.mj_forward(m,d);t=z['time_s'];k=int(np.argmin(abs(t-a.start)));prefix_q=prefix_v=prefix_force=0.
for step in range(k):
    d.ctrl[:]=z['ctrl'][step];mj.mj_step(m,d);prefix_q=max(prefix_q,float(np.max(abs(d.qpos-z['qpos'][step+1]))));prefix_v=max(prefix_v,float(np.max(abs(d.qvel-z['qvel'][step+1]))));prefix_force=max(prefix_force,float(np.max(abs(d.actuator_force-z['actuator_force'][step]))))
state=np.empty(mj.mj_stateSize(m,spec));mj.mj_getState(m,d,state,spec);eqids=[e for e in range(m.neq)if m.eq_type[e]==mj.mjtEq.mjEQ_JOINT and(m.joint(int(m.eq_obj1id[e])).name or'').startswith(('rh_','lh_'))];obj=m.body('target_object_01').id;oq=int(m.jnt_qposadr[int(m.body_jntadr[obj])]);aids=[i for i in range(m.nu)if(m.actuator(i).name or'').startswith(('motor_rh_','motor_lh_'))and np.any(m.actuator_gear[i]!=0)];passive=np.flatnonzero(np.all(m.actuator_gear==0,axis=1));branches=[];saved={}
for label,stiff in [('original',False),('stiffer_solimp',True)]:
    mm=mj.MjModel.from_binary_path(str((a.trial/'effective_model.mjb').resolve()))
    if stiff:mm.eq_solimp[eqids,0]=.999;mm.eq_solimp[eqids,1]=.9999
    invariant=['body_mass','body_inertia','body_pos','body_quat','jnt_range','geom_type','geom_bodyid','geom_pos','geom_quat','geom_size','geom_dataid','mesh_vert','mesh_face','eq_data','eq_solref','actuator_gainprm','actuator_biasprm','actuator_gear','actuator_forcerange','actuator_forcelimited','dof_damping','dof_armature']
    assert all(np.array_equal(getattr(mm,f),getattr(m,f))for f in invariant)
    dd=mj.MjData(mm);mj.mj_setState(mm,dd,state,spec);qerr=verr=ferr=0.;qseq=[dd.qpos.copy()];vseq=[dd.qvel.copy()];fseq=[];metric=[];peak_eqforce=0.
    for step in range(k,len(z['ctrl'])):
        dd.ctrl[:]=z['ctrl'][step];mj.mj_step(mm,dd);errs=[]
        for e in eqids:
            child,parent=int(mm.eq_obj1id[e]),int(mm.eq_obj2id[e]);cq,pq=int(mm.jnt_qposadr[child]),int(mm.jnt_qposadr[parent]);errs.append(float(dd.qpos[cq]-mm.qpos0[cq]-np.polynomial.polynomial.polyval(dd.qpos[pq]-mm.qpos0[pq],mm.eq_data[e,:5])))
        totalhand=np.zeros(3);thumbN=0.;thumbF=np.zeros(3);maxpen=0.;normalpeak=0.
        for ci,c in enumerate(dd.contact[:dd.ncon]):
            g1,g2=int(c.geom1),int(c.geom2);b1,b2=int(mm.geom_bodyid[g1]),int(mm.geom_bodyid[g2]);maxpen=max(maxpen,float(max(0,-c.dist)))
            if obj not in(b1,b2):continue
            other=b2 if b1==obj else b1;name=mm.body(other).name or''
            if not name.startswith(('rh_','lh_')):continue
            F=np.empty(6);mj.mj_contactForce(mm,dd,ci,F);vec=(1 if b2==obj else-1)*(np.asarray(c.frame).reshape(3,3).T@F[:3]);totalhand+=vec;normalpeak=max(normalpeak,float(F[0]))
            if name.startswith('rh_thumb'):thumbN+=float(F[0]);thumbF+=vec
        if dd.nefc:
            equal=np.asarray(dd.efc_type)==mj.mjtConstraint.mjCNSTR_EQUALITY
            if np.any(equal):peak_eqforce=max(peak_eqforce,float(np.max(abs(dd.efc_force[equal]))))
        qerr=max(qerr,float(np.max(abs(dd.qpos-z['qpos'][step+1]))));verr=max(verr,float(np.max(abs(dd.qvel-z['qvel'][step+1]))));ferr=max(ferr,float(np.max(abs(dd.actuator_force-z['actuator_force'][step]))));metric.append([dd.time,*errs,thumbN,*thumbF,*totalhand,normalpeak,maxpen]);qseq.append(dd.qpos.copy());vseq.append(dd.qvel.copy());fseq.append(dd.actuator_force.copy())
    values=np.asarray(metric);forces=np.asarray(fseq);errors=values[:,1:1+len(eqids)];first=np.flatnonzero(np.max(abs(errors),axis=1)>.03);saved[label+'_qpos']=np.asarray(qseq);saved[label+'_qvel']=np.asarray(vseq);saved[label+'_actuator_force']=forces;saved[label+'_metrics']=values
    branches.append({'label':label,'eq_solimp':mm.eq_solimp[eqids].tolist(),'eq_solref_unchanged':True,'other_model_arrays_preserved':invariant,'mimic_max_rad':float(np.max(abs(errors))),'thumb_ip_error_max_rad':float(np.max(abs(errors[:,eqids.index(next(e for e in eqids if mm.joint(int(mm.eq_obj1id[e])).name=='rh_thumb_ip'))]))),'first_mimic_over_30mrad_time_s':float(values[first[0],0])if len(first)else None,'active_hand_torque_max_abs_Nm':float(np.max(abs(forces[:,aids]))),'passive_force_max_abs_Nm':float(np.max(abs(forces[:,passive]))),'thumb_normal_peak_N':float(np.max(values[:,1+len(eqids)])),'actual_contact_normal_peak_N':float(values[:,-2].max()),'thumb_up_force_peak_N':float(values[:,1+len(eqids)+3].max()),'all_contact_penetration_peak_m':float(values[:,-1].max()),'joint_equality_constraint_force_peak_Nm':peak_eqforce,'original_trace_qpos_error_max':qerr,'original_trace_qvel_error_max':verr,'original_trace_actuator_force_error_max':ferr,'object1_final_translation_from_branch_start_m':(dd.qpos[oq:oq+3]-z['qpos'][k,oq:oq+3]).tolist()})
saved['initial_integration_state']=state;saved['state_spec']=np.asarray(spec);saved['ctrl']=z['ctrl'][k:];np.savez_compressed(a.output.with_suffix('.npz'),**saved);out={'schema':'fourth.C_short_solimp_numerical_counterfactual.v1','scope':'Fixed recorded control replay only, no feedback adaptation, no policy rerun and no successful grasp/lift claim. Only10soft joint equalities stiffened privately; real robot geometry and physical motor caps unchanged. Stop at original failure time, no expansion beyond requested1.535s comparison.','trial':str(a.trial),'start_time_s':float(t[k]),'end_time_s':float(t[-1]),'duration_s':float(t[-1]-t[k]),'prefix_reconstruction_max_errors':{'qpos':prefix_q,'qvel':prefix_v,'actuator_force':prefix_force},'initial_integration_state_sha256':hashlib.sha256(state.tobytes()).hexdigest(),'branches':branches,'numeric_trace':str(a.output.with_suffix('.npz'))};a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
