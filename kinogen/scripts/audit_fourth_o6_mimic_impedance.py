"""Frozen-control counterfactual regression of O6 mechanical joint impedance.

Only finger equality solimp changes. Initial state and ctrl are identical to
the frozen trial; no actual-state projections, controller re-execution, force
injection, changed fault threshold, or manipulation-success claim is made.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import mujoco as mj

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_model(p):return mj.MjModel.from_binary_path(str(p))if p.suffix=='.mjb'else mj.MjModel.from_xml_path(str(p))

def run(model,z,label,solimp=None,max_steps=None):
    model=__import__('copy').copy(model)
    preserved=['body_mass','body_inertia','body_pos','body_quat','dof_armature','dof_damping','jnt_range',
               'actuator_gainprm','actuator_biasprm','actuator_gear','actuator_forcerange','actuator_forcelimited',
               'geom_pos','geom_quat','geom_size','geom_contype','geom_conaffinity','eq_data','eq_solref']
    before={key:getattr(model,key).copy()for key in preserved}
    equations=[]
    for e in range(model.neq):
        if model.eq_type[e]!=mj.mjtEq.mjEQ_JOINT:continue
        child,parent=int(model.eq_obj1id[e]),int(model.eq_obj2id[e]);name=model.joint(child).name or''
        if name.startswith(('lh_','rh_')):
            if solimp is not None:model.eq_solimp[e]=solimp
            equations.append((e,child,parent))
    if any(not np.array_equal(getattr(model,key),value)for key,value in before.items()):raise ValueError('Changed non-impedance parameters')
    d=mj.MjData(model);mj.mj_setState(model,d,z['initial_state'],int(np.asarray(z['state_spec']).reshape(-1)[0]));mj.mj_forward(model,d)
    qerr=float(np.max(abs(d.qpos-z['qpos'][0])));verr=float(np.max(abs(d.qvel-z['qvel'][0])))
    residual=[];vel_residual=[];qhistory=[d.qpos.copy()];root=[];events=[];solver=[];maxpen=0.;maxacc=0.;maxpassive=0.;maxactive=0.;reset=None
    passive=np.flatnonzero(np.all(model.actuator_gear==0,axis=1))
    active=[i for i in range(model.nu)if(model.actuator(i).name or'').startswith(('motor_lh_','motor_rh_'))and i not in passive]
    names=[model.body(i).name or'world'for i in range(model.nbody)]
    for step,ctrl in enumerate(z['ctrl'][:max_steps]):
        d.ctrl[:]=ctrl
        mj.mj_step(model,d)
        expected=(step+1)*float(model.opt.timestep)
        if abs(d.time-(float(z['time'][0])+expected))>1e-8 or not np.isfinite(d.qpos).all()or not np.isfinite(d.qvel).all():
            reset={'step':step,'expected_time_s':expected,'actual_time_s':float(d.time)};break
        qhistory.append(d.qpos.copy());root.append(d.qpos[:7].copy())
        qerr=max(qerr,float(np.max(abs(d.qpos-z['qpos'][step+1]))));verr=max(verr,float(np.max(abs(d.qvel-z['qvel'][step+1]))))
        rr=[];vv=[]
        for e,child,parent in equations:
            cq,pq=int(model.jnt_qposadr[child]),int(model.jnt_qposadr[parent]);cv,pv=int(model.jnt_dofadr[child]),int(model.jnt_dofadr[parent])
            coefficients=model.eq_data[e,:5];x=d.qpos[pq]-model.qpos0[pq]
            rr.append(float(d.qpos[cq]-model.qpos0[cq]-np.polynomial.polynomial.polyval(x,coefficients)))
            vv.append(float(d.qvel[cv]-np.polynomial.polynomial.polyval(x,np.polynomial.polynomial.polyder(coefficients))*d.qvel[pv]))
        residual.append(rr);vel_residual.append(vv)
        rows=[i for i in range(d.nefc)if d.efc_type[i]==mj.mjtConstraint.mjCNSTR_EQUALITY]
        maxacc=max(maxacc,float(np.max(abs(d.qacc))));solver.append(int(np.max(d.solver_niter)))
        maxpassive=max(maxpassive,float(np.max(abs(d.actuator_force[passive]))));maxactive=max(maxactive,float(np.max(abs(d.actuator_force[active]))))
        for c in d.contact[:d.ncon]:maxpen=max(maxpen,-float(c.dist))
        # Keep causal snapshots around initial thumb error/fault, not just peaks.
        thumb_index=next(i for i,(_,child,_)in enumerate(equations)if model.joint(child).name=='rh_thumb_ip')
        important=step in [0,19,31,39,41,51,52,53,59,79,99]or(abs(rr[thumb_index])>.03 and not any(e['reason']=='first_thumb_residual_above_fault_limit'for e in events))
        if important:
            e,child,parent=equations[thumb_index];cq,pq=int(model.jnt_qposadr[child]),int(model.jnt_qposadr[parent]);cv,pv=int(model.jnt_dofadr[child]),int(model.jnt_dofadr[parent])
            equation_rows=[row for row in rows if int(d.efc_id[row])==e]
            contact=[]
            for contact_index,c in enumerate(d.contact[:d.ncon]):
                gs=[int(c.geom1),int(c.geom2)];bs=model.geom_bodyid[gs]
                if any(names[int(b)].startswith('rh_thumb')for b in bs):
                    wrench=np.zeros(6);mj.mj_contactForce(model,d,contact_index,wrench)
                    contact.append({'geoms':[model.geom(g).name for g in gs],'bodies':[names[int(b)]for b in bs],
                        'distance_m':float(c.dist),'actual_normal_force_N':float(wrench[0])})
            parent_aid=mj.mj_name2id(model,mj.mjtObj.mjOBJ_ACTUATOR,'motor_rh_thumb_cmc_pitch')
            events.append({'time_s':float(d.time),'reason':'first_thumb_residual_above_fault_limit'if abs(rr[thumb_index])>.03 and not any(ev['reason']=='first_thumb_residual_above_fault_limit'for ev in events)else'startup_sample',
                'thumb_mimic_residual_rad':rr[thumb_index],'parent_child_qpos_rad':[float(d.qpos[pq]),float(d.qpos[cq])],
                'parent_child_qvel_rad_s':[float(d.qvel[pv]),float(d.qvel[cv])],
                'parent_ctrl_rad':float(ctrl[parent_aid]),'parent_actual_motor_torque_Nm':float(d.actuator_force[parent_aid]),
                'parent_child_qfrc_bias_Nm':[float(d.qfrc_bias[pv]),float(d.qfrc_bias[cv])],
                'parent_child_armature_kgm2':[float(model.dof_armature[pv]),float(model.dof_armature[cv])],
                'equality_solver_rows':[{'constraint_pos_rad':float(d.efc_pos[row]),'constraint_velocity_rad_s':float(d.efc_vel[row]),'constraint_force_Nm':float(d.efc_force[row]),'regularisation_R':float(d.efc_R[row])}for row in equation_rows],
                'actual_thumb_contacts':contact})
    rr=np.asarray(residual);vv=np.asarray(vel_residual);roots=np.asarray(root);actual_steps=len(residual)
    mimic=[]
    for col,(e,child,parent)in enumerate(equations):
        k=int(np.argmax(abs(rr[:,col])))
        mimic.append({'child':model.joint(child).name,'parent':model.joint(parent).name,'solref':model.eq_solref[e].tolist(),'solimp':model.eq_solimp[e].tolist(),
                      'max_abs_error_rad':float(np.max(abs(rr[:,col]))),'p95_abs_error_rad':float(np.percentile(abs(rr[:,col]),95)),
                      'peak_time_s':float((k+1)*model.opt.timestep),'max_abs_velocity_error_rad_s':float(np.max(abs(vv[:,col]))),
                      'samples_above_unchanged_fault_limit':int(np.count_nonzero(abs(rr[:,col])>.03))})
    root0=z['qpos'][0,:7];up=1-2*(roots[:,4]**2+roots[:,5]**2)
    warning={str(w):int(d.warning[int(w)].number)for w in [mj.mjtWarning.mjWARN_BADQPOS,mj.mjtWarning.mjWARN_BADQVEL,mj.mjtWarning.mjWARN_BADQACC]}
    result={'label':label,'counterfactual_only':solimp is not None,'actual_replayed_steps':actual_steps,'duration_s':actual_steps*float(model.opt.timestep),
            'non_impedance_model_arrays_exactly_preserved':preserved,
            'qpos_difference_to_frozen_trace_max_abs':qerr,'qvel_difference_to_frozen_trace_max_abs':verr,'mimic_constraints':mimic,
            'maximum_mimic_error_rad':float(np.max(abs(rr))),'all_mimic_errors_below_unchanged_fault_limit':bool(np.max(abs(rr))<.03),
            'max_contact_penetration_m':maxpen,'root_xy_displacement_max_m':float(np.max(np.linalg.norm(roots[:,:2]-root0[:2],axis=1))),
            'root_up_z_min':float(np.min(up)),'max_abs_qacc':maxacc,'solver_iterations_max':int(max(solver)),'solver_iterations_mean':float(np.mean(solver)),
            'passive_actuator_force_max_abs_Nm':maxpassive,'active_hand_actuator_force_max_abs_Nm':maxactive,
            'reset_or_nonfinite':reset,'warnings':warning,'startup_evidence':events}
    return result,np.asarray(qhistory),rr


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trial',type=Path,default=ROOT/'outputs/fourth_full_sequence/physics_sonic15225_o6_obj1_liftproof_v1_6s')
    p.add_argument('--output',type=Path,default=ROOT/'reports/fourth_o6_mimic_impedance_counterfactual.json')
    p.add_argument('--duration-s',type=float);a=p.parse_args()
    r=json.loads((a.trial/'report.json').read_text());model_path=Path(r['model']['path']);trace=a.trial/'physical_trace.npz'
    if sha(model_path)!=r['model']['sha256']or sha(trace)!=r['artifacts']['trace']['sha256']:raise ValueError('Frozen input receipt mismatch')
    z=dict(np.load(trace,allow_pickle=False));model=load_model(model_path)
    if np.any(z['qfrc_applied']!=0)or np.any(z['xfrc_applied']!=0):raise ValueError('Source trial contains external forces')
    max_steps=None if a.duration_s is None else min(len(z['ctrl']),int(round(a.duration_s/model.opt.timestep)))
    baseline,bq,br=run(model,z,'original_solimp',max_steps=max_steps)
    if baseline['qpos_difference_to_frozen_trace_max_abs']>1e-7 or baseline['qvel_difference_to_frozen_trace_max_abs']>1e-6:raise ValueError('Original-model exact replay failed')
    harder,hq,hr=run(model,z,'mechanical_mimic_solimp_0p999_0p9999',[.999,.9999,.001,.5,2],max_steps=max_steps)
    npz=a.output.with_suffix('.npz');np.savez_compressed(npz,time_s=z['time'][:len(bq)],baseline_qpos=bq,counterfactual_qpos=hq,baseline_mimic_residual_rad=br,counterfactual_mimic_residual_rad=hr)
    report={'schema':'fourth.o6_mechanical_mimic_impedance_counterfactual.v1','trial':str(a.trial),'frozen_model':str(model_path),'frozen_model_sha256':sha(model_path),'frozen_trace_sha256':sha(trace),
            'only_model_parameter_changed':'Finger equality solimp d0=.999,dwidth=.9999; all 10 finger equalities. Solref and all other model fields retained.',
            'timestep_s':float(model.opt.timestep),'original_model_disableflags':int(model.opt.disableflags),
            'post_initialisation_state_projection_count':0,'applied_external_force_injection_count':0,'controller_reexecution':False,
            'original':baseline,'counterfactual':harder,'arrays':str(npz),'arrays_sha256':sha(npz),
            'physical_grasp_or_lift_success_claimed':False,'scope':'True forward dynamics with frozen identical actuator controls; numerical transmission regression only. New closed-loop trial still required.'}
    a.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'original':{k:v for k,v in baseline.items()if k not in ['startup_evidence','mimic_constraints']},'counterfactual':{k:v for k,v in harder.items()if k not in ['startup_evidence','mimic_constraints']},'counterfactual_mimic':harder['mimic_constraints']}),flush=True)

if __name__=='__main__':main()
