"""Diagnose auxiliary-joint numerical failure and add declared rotor inertia.

Only neck/finger generalized rotor inertia is changed. Body link inertias,
28 learned joint parameters, actuator gains/limits and geometry are unchanged.
"""
from pathlib import Path
import argparse
import hashlib
import json
import xml.etree.ElementTree as ET

import numpy as np
import mujoco as mj


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(path):
    return mj.MjModel.from_binary_path(str(path))if path.suffix.lower()=='.mjb'else mj.MjModel.from_xml_path(str(path))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trial-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--make-variant',action='store_true')
    p.add_argument('--finger-armature',type=float,default=1e-4)
    p.add_argument('--neck-armature',type=float,default=1e-3)
    a=p.parse_args()
    r=json.loads((a.trial_dir/'report.json').read_text())
    m=load(Path(r['model']['path']))
    z=np.load(a.trial_dir/'physical_trace.npz',allow_pickle=False)
    t,q,v,u=z['time'],z['qpos'],z['qvel'],z['ctrl']
    bad=np.flatnonzero(np.abs(np.diff(t)-m.opt.timestep)>1e-8)
    regular_end=int(bad[0]+1)if len(bad)else len(t)
    tt=t[:regular_end];qq=q[:regular_end];vv=v[:regular_end]
    rows=[]
    aux=[];body=[]
    for j in range(m.njnt):
        name=mj.mj_id2name(m,mj.mjtObj.mjOBJ_JOINT,j)or''
        if m.jnt_type[j]!=mj.mjtJoint.mjJNT_HINGE:continue
        (aux if name.startswith(('lh_','rh_','neck_'))else body).append(j)
    for j in aux:
        name=mj.mj_id2name(m,mj.mjtObj.mjOBJ_JOINT,j)
        dof=int(m.jnt_dofadr[j]);qa=int(m.jnt_qposadr[j]);b=int(m.jnt_bodyid[j])
        aid=mj.mj_name2id(m,mj.mjtObj.mjOBJ_ACTUATOR,'motor_'+name)
        peak=int(np.argmax(np.abs(vv[:,dof])))
        acc=np.diff(vv[:,dof])/np.diff(tt)
        apeak=int(np.argmax(np.abs(acc)))if len(acc)else 0
        rows.append({'joint':name,'dof':dof,'qpos_address':qa,
            'body':mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,b),'body_mass_kg':float(m.body_mass[b]),
            'body_principal_inertia_kg_m2':m.body_inertia[b].tolist(),
            'armature_kg_m2':float(m.dof_armature[dof]),'joint_viscous_damping':float(m.dof_damping[dof]),
            'range_rad':m.jnt_range[j].tolist(),'actuator_kp':float(m.actuator_gainprm[aid,0]),
            'actuator_kd':float(-m.actuator_biasprm[aid,2]),'actuator_force_limits_Nm':m.actuator_forcerange[aid].tolist(),
            'actual_qpos_min_max_rad':[float(qq[:,qa].min()),float(qq[:,qa].max())],
            'max_abs_qvel_rad_s':float(abs(vv[peak,dof])),'peak_qvel_time_s':float(tt[peak]),
            'max_abs_finite_difference_qacc_rad_s2':float(abs(acc[apeak]))if len(acc)else 0.,
            'peak_qacc_interval_start_s':float(tt[apeak]),
            'last_pre_reset_qpos_rad':float(qq[-1,qa]),'last_pre_reset_qvel_rad_s':float(vv[-1,dof])})
    d57=int(m.dof_jntid[57]);d57name=mj.mj_id2name(m,mj.mjtObj.mjOBJ_JOINT,d57)
    root_quat=qq[:,3:7];up=1-2*(root_quat[:,1]**2+root_quat[:,2]**2)
    fall=np.flatnonzero(up<.5)
    result={'schema':'fourth.auxiliary_rotor_inertia_diagnosis.v1',
        'trial':str(a.trial_dir),'frozen_model_sha256':sha(Path(r['model']['path'])),
        'trace_sha256':sha(a.trial_dir/'physical_trace.npz'),
        'reported_warning_dof_57_joint':d57name,'last_monotonic_physics_time_s':float(tt[-1]),
        'nonmonotonic_reset_intervals':bad.tolist(),'last_recorded_time_s':float(t[-1]),
        'first_root_up_below_0p5_time_s':float(tt[fall[0]])if len(fall)else None,
        'root_up_min_before_reset':float(up.min()),'auxiliary_joint_count':len(aux),'learned_body_joint_count':len(body),
        'auxiliary_joint_diagnostics':rows,
        'interpretation':['Reported QACC DOF is identified from the frozen effective model, not guessed from URDF order.',
                          'Finite-difference accelerations are derived from recorded qvel; instantaneous solver qacc was not recorded.',
                          'Rotor-inertia regularization targets auxiliary numerical conditioning; it does not prove body tracking or interaction success.']}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.make_variant:
        if len(aux)!=24 or len(body)!=28:raise ValueError('Unexpected auxiliary/learned joint sets')
        base=Path(r['input_model']['path']);base_model=load(base);xml=ET.parse(base).getroot()
        if a.finger_armature<=0 or a.neck_armature<=0:raise ValueError('Positive rotor inertia required')
        applied=[]
        for j in aux:
            name=mj.mj_id2name(m,mj.mjtObj.mjOBJ_JOINT,j)
            element=xml.find(f".//joint[@name='{name}']")
            if element is None:raise ValueError('Auxiliary hinge missing from input XML')
            value=a.neck_armature if name.startswith('neck_')else a.finger_armature
            element.set('armature',format(value,'.17g'));applied.append({'joint':name,'armature_kg_m2':value})
        variant=a.output.with_suffix('.xml')
        if variant.exists():raise ValueError('Use a new output variant path')
        variant.write_text(ET.tostring(xml,encoding='unicode'),encoding='utf-8')
        changed=load(variant)
        learned_dofs=[]
        for j in body:
            name=mj.mj_id2name(m,mj.mjtObj.mjOBJ_JOINT,j)
            jj=mj.mj_name2id(base_model,mj.mjtObj.mjOBJ_JOINT,name)
            learned_dofs.append(int(base_model.jnt_dofadr[jj]))
        for key in ['dof_armature','dof_damping','dof_frictionloss']:
            if not np.array_equal(getattr(base_model,key)[learned_dofs],getattr(changed,key)[learned_dofs]):
                raise ValueError('Learned body dynamics changed: '+key)
        preserved=['body_mass','body_inertia','body_ipos','body_iquat','body_pos','body_quat','geom_type','geom_size','geom_pos','geom_quat',
                   'geom_contype','geom_conaffinity','jnt_axis','jnt_range','actuator_gainprm','actuator_biasprm','actuator_forcerange']
        if any(not np.array_equal(getattr(base_model,key),getattr(changed,key))for key in preserved):
            raise ValueError('Geometry, mass or actuator bank changed')
        result['variant']={'path':str(variant),'sha256':sha(variant),'base_model':str(base),'base_sha256':sha(base),
            'applied':applied,'learned_28_joint_dynamics_unchanged':True,'other_physical_arrays_unchanged':preserved,
            'damping_and_gains_modified':False,
            'engineering_assumption':{'finger_joint_reflected_rotor_inertia_kg_m2':a.finger_armature,
                'neck_joint_reflected_rotor_inertia_kg_m2':a.neck_armature,
                'finger_example_motor_rotor_inertia_kg_m2':1e-8,'finger_example_gear_ratio':100,
                'neck_example_motor_rotor_inertia_kg_m2':1e-7,'neck_example_gear_ratio':100,
                'relation':'J_reflected = J_motor * gear_ratio^2; these illustrative values are assumptions, not measured actuator data.',
                'reason':'Restored articulated auxiliary joints had no rotor armature despite tiny link rotational inertias; added reflected rotor inertia improves conditioning without changing link mass or body policy parameters.'}}
    report=a.output.with_suffix('.json')
    report.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'report':str(report),'dof57':d57name,'last_monotonic_time_s':float(tt[-1]),
        'first_root_up_below_0p5_time_s':result['first_root_up_below_0p5_time_s'],
        'worst_aux':sorted(rows,key=lambda row:row['max_abs_qvel_rad_s'],reverse=True)[:3],
        'variant':result.get('variant',{}).get('path')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
