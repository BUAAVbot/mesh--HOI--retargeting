"""Create two explicitly diagnostic 350-sample static standing references.

A holds the retargeted first pose exactly. B uses the saved SONIC training
defaults, source initial yaw, and a capsule/ground-based initial root height.
Neither sequence represents the source action. No physics steps are run.
"""
from pathlib import Path
import argparse,hashlib,json,shutil


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--duration',type=float,default=8.);p.add_argument('--ground-clearance-m',type=float,default=.001)
    a=p.parse_args()
    if a.output.exists()or a.duration<=0:raise ValueError('Choose a fresh output directory and positive duration')
    import numpy as np,mujoco as mj
    from rgb145_sonic_controller import Controller
    ref=np.load(a.reference,allow_pickle=False)
    if ref['qpos'].shape!=(350,59)or ref['frame_ids'].shape!=(350,):raise ValueError('Expected all350 source samples')
    def load_model():return mj.MjModel.from_binary_path(str(a.model.resolve()))if a.model.suffix=='.mjb'else mj.MjModel.from_xml_path(str(a.model.resolve()))
    model=load_model();data=mj.MjData(model)
    if(model.nq,model.nv,model.nu,model.neq)!=(59,58,52,10):raise ValueError('Vita/O6 ABI mismatch')
    fields=('body_mass','body_inertia','body_pos','body_quat','geom_pos','geom_quat','geom_size','geom_friction','geom_contype','geom_conaffinity','jnt_range','jnt_limited','dof_armature','dof_damping','dof_frictionloss','actuator_gainprm','actuator_biasprm','actuator_gear','actuator_forcerange','eq_data','eq_active0')
    before={k:getattr(model,k).copy()for k in fields}
    # Controller constructors configure actuator parameters. Load the original
    # defaults on a disposable clone, never on the authoritative FK model.
    clone=load_model()
    controller=Controller(clone,dict(ref),{'mode':'newft15225','device':a.device,'cpu_threads':2,'delay_steps':3,'imu_latency_s':.0025,'hand_object_feedback':False})
    names=list(map(str,ref['dof_names']));qA=np.asarray(ref['qpos'][0],float).copy();qB=qA.copy()
    for name,value in zip(controller.names,controller.default):qB[7+names.index(name)]=float(value)
    rotation=np.empty(9);mj.mju_quat2Mat(rotation,qA[3:7]);rotation=rotation.reshape(3,3)
    yaw=float(np.arctan2(rotation[1,0],rotation[0,0]));qB[3:7]=[np.cos(yaw/2),0.,0.,np.sin(yaw/2)]
    ground=mj.mj_name2id(model,mj.mjtObj.mjOBJ_GEOM,'ground_support')
    if ground<0 or model.geom_type[ground]!=int(mj.mjtGeom.mjGEOM_PLANE):raise ValueError('Need the actual named ground_support plane')
    foot_geoms=[]
    for side in ('l','r'):
        geoms=[g for g in range(model.ngeom)if(mj.mj_id2name(model,mj.mjtObj.mjOBJ_GEOM,g)or'').startswith('collision_foot_'+side+'_')and model.geom_contype[g]]
        if not geoms or any(model.geom_type[g]!=int(mj.mjtGeom.mjGEOM_CAPSULE)for g in geoms):raise ValueError('Need actual foot capsule collision geometry')
        foot_geoms.append(geoms)
    ankles=[mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,side+'_ankle_roll_link')for side in ('left','right')]
    def inspect_pose(q):
        data.qpos[:]=q;mj.mj_kinematics(model,data)
        normal=data.geom_xmat[ground].reshape(3,3)[:,2].copy();point=data.geom_xpos[ground].copy()
        if normal[2]<=1e-6:raise ValueError('Ground plane normal must point upward')
        minimum=[]
        for geoms in foot_geoms:
            gaps=[]
            for g in geoms:
                axis=data.geom_xmat[g].reshape(3,3)[:,2]
                support=data.geom_xpos[g]-np.sign(normal@axis)*model.geom_size[g,1]*axis-model.geom_size[g,0]*normal
                gaps.append(float(normal@(support-point)))
            minimum.append(min(gaps))
        return np.asarray(minimum),normal,data.xpos[ankles].copy()
    gaps_before,normal,_=inspect_pose(qB)
    delta_z=float((a.ground_clearance_m-np.min(gaps_before))/normal[2]);qB[2]+=delta_z
    gaps_after,_,_=inspect_pose(qB)
    if abs(float(np.min(gaps_after))-a.ground_clearance_m)>1e-9:raise AssertionError('Foot/plane support-height calculation failed')
    changed=[k for k,v in before.items()if not np.array_equal(v,getattr(model,k))]
    if changed:raise AssertionError('Authoritative model parameters were modified: '+','.join(changed))
    execution=np.arange(350,dtype=float)*(a.duration/350.)
    source_time=np.asarray(ref['source_time_s']).copy()
    preserved=[k for k in ref.files if k.startswith('source_')and k!='source_to_execution_time_s']
    for k in ('frame_ids','dof_names','wrist_reference_joint_names','wrist_reference_joint_targets_rad'):
        if k in ref and k not in preserved:preserved.append(k)
    a.output.mkdir(parents=True)
    outputs=[]
    for tag,q in (('A_retarget_q0_static',qA),('B_training_default_static',qB)):
        payload={k:ref[k].copy()for k in preserved};gaps,_,feet=inspect_pose(q)
        payload.update(qpos=np.repeat(q[None],350,axis=0),time_s=execution,source_time_s=source_time,source_to_execution_time_s=np.c_[source_time,execution],foot_support_expected=np.ones((350,2),bool),foot_support_geom_name=np.full((350,2),'ground_support'),foot_support_confidence=np.ones((350,2)),foot_target_ankle_world=np.repeat(feet[None],350,axis=0),foot_actual_ankle_world=np.repeat(feet[None],350,axis=0),root_target_world=np.repeat(q[None,:3],350,axis=0),diagnostic_not_source_motion=np.asarray(True),diagnostic_variant=np.asarray(tag),diagnostic_duration_s=np.asarray(a.duration),diagnostic_parent_reference_sha256=np.asarray(sha(a.reference)))
        output=a.output/(tag+'.npz');np.savez_compressed(output,**payload)
        row={'variant':tag,'path':str(output.resolve()),'sha256':sha(output),'frames':350,'duration_s':a.duration,'execution_last_pts_s':float(execution[-1]),'source_frame_ids_unchanged':bool(np.array_equal(payload['frame_ids'],ref['frame_ids'])),'source_time_unchanged':bool(np.array_equal(payload['source_time_s'],ref['source_time_s'])),'preserved_source_field_names':preserved,'diagnostic_not_source_motion':True,'initial_qpos':q.tolist(),'initial_minimum_foot_plane_gap_m':gaps.tolist(),'physics_passed':False,'meaning':'Static reference generation only; no physical tracking test has been run'}
        output.with_suffix('.json').write_text(json.dumps(row,indent=2)+'\n');outputs.append(row)
    shutil.copy2(a.reference,a.output/'parent_reference.npz');shutil.copy2(__file__,a.output/'generator.py')
    receipt={'schema':'rgb145.diagnostic_static_references.v1','diagnostic_not_source_motion':True,'purpose':'Separate retarget initial-pose trackability from saved-policy/training-default standing transfer; neither output reproduces the source motion','model_input':{'path':str(a.model.resolve()),'sha256':sha(a.model)},'parent_reference':{'path':str(a.reference.resolve()),'sha256':sha(a.reference)},'source':{'generator_sha256':sha(__file__),'controller_source_sha256':sha(Path(__file__).with_name('rgb145_sonic_controller.py')),'saved_controller':controller.report()},'authoritative_model_physical_parameters_unchanged':True,'verified_model_fields':list(fields),'controller_default_loaded_on_disposable_clone':True,'initial_yaw_from_robot_source_reference_rad':yaw,'B_root_z_adjustment_m':delta_z,'B_minimum_foot_plane_gaps_before_m':gaps_before.tolist(),'B_minimum_foot_plane_gaps_after_m':gaps_after.tolist(),'only_B_root_translation_change':'Z only; XY unchanged','ground_clearance_target_m':a.ground_clearance_m,'mujoco_version':mj.__version__,'physics_steps_run':0,'outputs':outputs}
    (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
