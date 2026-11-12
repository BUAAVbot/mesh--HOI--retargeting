"""Run official HOI-Retarget window NLP on one exo person with HDMI Vita28.

Pinocchio's CasADi bindings are unavailable in the existing environments. This
explicit backend adapter preserves the official window objective/constraints
and uses the same compiled MuJoCo geometry/joint transforms. It is not a claim
that stock HOI-Retarget ships a Vita robot or measured sitting contact labels.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import types

from exo_world_frame import estimate


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--stage1-reference',type=Path,required=True,
                    help='Independent UMR or OmniRetarget HDMI28 initialization, never GMR')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--frames',type=int,default=0)
    ap.add_argument('--window-size',type=int,default=60)
    a=ap.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    sys.path.append('/data/grail-envs/retarget/lib/python3.10/site-packages')
    import numpy as np
    import mujoco as mj
    import casadi as ca
    from scipy.spatial.transform import Rotation
    with np.load(a.stage1_reference,allow_pickle=False) as init:
        qref=init['qpos']
        names=init['robot_joint_names'].tolist()
        times=init['time_s'] if 'time_s' in init else np.arange(len(qref),dtype=float)/float(init['fps'])
    with np.load(a.input,allow_pickle=False) as motion:
        source_joints=motion['joints'][:,:22]
    if a.frames>0:
        qref=qref[:a.frames];times=times[:a.frames];source_joints=source_joints[:a.frames]
    source_joints=source_joints[:len(qref)]
    R,offset,frame_audit=estimate(a.input,a.input.parents[2]/'scene'/'mesh.npz')
    source_joints=source_joints@R.T
    source_joints[:,:,2]-=offset
    model=mj.MjModel.from_xml_path(str(a.model))
    if model.nq!=35 or qref.shape!=(len(times),35) or len(names)!=28:
        raise ValueError('expected one HDMI Vita28 track')
    body_names=[mj.mj_id2name(model,mj.mjtObj.mjOBJ_BODY,i) for i in range(1,model.nbody)]

    def qrot(v):
        x,y,z,w=[v[i] for i in range(4)]
        n=x*x+y*y+z*z+w*w
        return ca.vertcat(ca.horzcat(1-2*(y*y+z*z)/n,2*(x*y-z*w)/n,2*(x*z+y*w)/n),
          ca.horzcat(2*(x*y+z*w)/n,1-2*(x*x+z*z)/n,2*(y*z-x*w)/n),
          ca.horzcat(2*(x*z-y*w)/n,2*(y*z+x*w)/n,1-2*(x*x+y*y)/n))

    class VitaCompiledCasadiFK:
        def __init__(self,*unused,**kw):
            self.nq=model.nq;self.nv=model.nv;self.base_q_start=0
            self.body_names=body_names
            self.joint_q_indices=np.arange(7,model.nq)
            self.joint_v_indices=np.arange(6,model.nv)
            self.q_lower=np.full(model.nq,-np.inf);self.q_upper=np.full(model.nq,np.inf)
            self.v_abs=np.full(model.nv,np.inf)
            self.dof_name_to_index={n:7+i for i,n in enumerate(names)}
            for j in range(model.njnt):
                if model.jnt_type[j]==mj.mjtJoint.mjJNT_HINGE:
                    self.q_lower[model.jnt_qposadr[j]],self.q_upper[model.jnt_qposadr[j]]=model.jnt_range[j]
            # Velocity limits are not defined in the shared compiled MJCF.
            # Keep the official conditional velocity constraints inactive
            # rather than inventing a different robot's URDF limits.
            q=ca.SX.sym('q',model.nq)
            pos=[ca.DM.zeros(3)];rot=[ca.DM.eye(3)]
            for body in range(1,model.nbody):
                parent=int(model.body_parentid[body])
                attached=[int(model.body_jntadr[body])+k for k in range(int(model.body_jntnum[body]))]
                if attached and model.jnt_type[attached[0]]==mj.mjtJoint.mjJNT_FREE:
                    p=q[:3];R=qrot(q[3:7])
                else:
                    p=pos[parent]+rot[parent]@ca.DM(model.body_pos[body])
                    local_R=Rotation.from_quat(model.body_quat[body][[1,2,3,0]]).as_matrix()
                    R=rot[parent]@ca.DM(local_R)
                    for j in attached:
                        qa=int(model.jnt_qposadr[j]);angle=q[qa]-float(model.qpos0[qa])
                        axis=np.asarray(model.jnt_axis[j]);x,y,z=axis
                        K=ca.DM([[0,-z,y],[z,0,-x],[-y,x,0]])
                        if model.jnt_type[j]==mj.mjtJoint.mjJNT_HINGE:
                            dR=ca.DM.eye(3)+ca.sin(angle)*K+(1-ca.cos(angle))*(K@K)
                            pivot=ca.DM(model.jnt_pos[j]);p=p+R@pivot-R@dR@pivot;R=R@dR
                        elif model.jnt_type[j]==mj.mjtJoint.mjJNT_SLIDE:p=p+R@ca.DM(axis)*angle
                        else:raise ValueError('Unsupported joint type')
                pos.append(p);rot.append(R)
            base=body_names.index('base_link')+1
            self.fk_world_pos={};self.fk_world_rot={};self.fk_rel_pos={};self.fk_rel_rot={}
            for idx,name in enumerate(body_names):
                p=pos[idx+1];R=rot[idx+1]
                self.fk_world_pos[name]=ca.Function('p_'+name,[q],[p])
                self.fk_world_rot[name]=ca.Function('r_'+name,[q],[R])
                self.fk_rel_pos[('base_link',name)]=ca.Function('bp_'+name,[q],[rot[base].T@(p-pos[base])])
                self.fk_rel_rot[('base_link',name)]=ca.Function('br_'+name,[q],[rot[base].T@R])
        def world_pos(self,name,q):return self.fk_world_pos[name](q)
        def world_rotmat(self,name,q):return self.fk_world_rot[name](q)
        def rel_pos(self,base,name,q):return self.fk_rel_pos[(base,name)](q)
        def rel_rotmat(self,base,name,q):return self.fk_rel_rot[(base,name)](q)

    fk=VitaCompiledCasadiFK();d=mj.MjData(model);rng=np.random.default_rng(20261028)
    pos_error=rot_error=0.
    for k in range(20):
        q=qref[k%len(times)].copy()
        q[7:]=np.clip(q[7:]+rng.normal(0,.12,28),fk.q_lower[7:],fk.q_upper[7:])
        d.qpos[:]=q;mj.mj_forward(model,d)
        qp=q.copy();qp[3:7]=q[[4,5,6,3]]
        for i,name in enumerate(body_names,1):
            pos_error=max(pos_error,float(np.max(np.abs(np.asarray(fk.world_pos(name,qp)).ravel()-d.xpos[i]))))
            rot_error=max(rot_error,float(np.max(np.abs(np.asarray(fk.world_rotmat(name,qp))-d.xmat[i].reshape(3,3)))))
    if pos_error>1e-8 or rot_error>1e-8:raise ValueError(('Vita FK verification failed',pos_error,rot_error))
    print(json.dumps({'fk_random_pose_count':20,'position_error_max':pos_error,'rotation_error_max':rot_error}),flush=True)
    sys.path.insert(0,str(a.source))
    # Avoid unrelated viewer imports in the vendored GMR package initializer.
    gmr_namespace=types.ModuleType('hoi_retarget.gmr');gmr_namespace.__path__=[str(a.source/'hoi_retarget/gmr')]
    sys.modules['hoi_retarget.gmr']=gmr_namespace
    from hoi_retarget.config import ProjectionConfig,TrackingWeights,WindowConfig
    # The original BasePinocchioProjector preprocessor does not use Pinocchio;
    # its unused postprocessor import is removed and never called.
    projector_path=a.source/'hoi_retarget/optimization/projector.py'
    p_ast=ast.parse(projector_path.read_text())
    p_ast.body=[n for n in p_ast.body if not(isinstance(n,ast.Import) and any(v.name=='pinocchio' for v in n.names))]
    p_mod=types.ModuleType('hoi_retarget.optimization.projector');p_mod.__package__='hoi_retarget.optimization'
    exec(compile(p_ast,str(projector_path),'exec'),p_mod.__dict__)
    sys.modules[p_mod.__name__]=p_mod
    backend=types.ModuleType('hoi_retarget.optimization.pinocchio_model')
    backend.PinocchioCasadiRobot=VitaCompiledCasadiFK
    sys.modules[backend.__name__]=backend
    from hoi_retarget.optimization.window import PinocchioMotionProjectorWindow
    window_path=a.source/'hoi_retarget/optimization/window.py'
    cfg=ProjectionConfig(urdf_path=str(a.model),base_body_name='base_link',
       tracking=TrackingWeights(link_weights_world={'neck_yaw_link':1.,'left_ankle_roll_link':100.,'right_ankle_roll_link':100.},link_weights_base={'waist_pitch_link':1.}),
       window=WindowConfig(window_size=a.window_size))
    world_p=[];world_q=[];local_p=[];local_q=[];feet=[]
    feet_speed=np.linalg.norm(np.gradient(source_joints[:,[10,11]],times,axis=0),axis=-1)
    floor=float(np.quantile(source_joints[:,[7,8],2],.05)-.08)
    for frame,q in enumerate(qref):
        d.qpos[:]=q;mj.mj_forward(model,d)
        base_p=d.xpos[1].copy();base_R=d.xmat[1].reshape(3,3).copy()
        wp=d.xpos[1:].copy();wr=d.xmat[1:].reshape(-1,3,3).copy()
        wq=Rotation.from_matrix(wr).as_quat()[:,[3,0,1,2]]
        world_p.append(wp);world_q.append(wq);local_p.append((wp-base_p)@base_R)
        local_q.append(Rotation.from_matrix(base_R.T@wr).as_quat()[:,[3,0,1,2]])
        toe=source_joints[frame,[10,11]]
        feet.append(((toe[:,2]-floor)<.15)&(feet_speed[frame]<.30))
    feet=np.asarray(feet)
    fps=1/float(np.median(np.diff(times)))
    md={'fps':fps,'root_pos':qref[:,:3],'root_rot':qref[:,3:7],
        'dof_pos':qref[:,7:],'dof_names':names,'link_body_list':body_names,
        'world_body_pos':np.asarray(world_p),'world_body_orient':np.asarray(world_q),
        'local_body_pos':np.asarray(local_p),'local_body_orient':np.asarray(local_q),
        'object_pos':np.zeros((len(times),3)),'object_rot':np.tile([1.,0.,0.,0.],(len(times),1)),
        'object_contact_sequence':np.zeros((len(times),1),dtype=bool),
        'object_ground_contact_sequence':np.ones((len(times),1),dtype=bool),
        'contact_link_names':[],'per_link_contact_flags':np.zeros((len(times),0),dtype=bool),
        'fixed_contact_points_per_frame_in_object_frame':np.zeros((len(times),0,3)),
        'robot_foot_ground_contact_flags':feet}
    solver=PinocchioMotionProjectorWindow(cfg,0.,[],compute_analysis=True)
    prep=solver._preprocess_motion(md)
    result=solver._unified_pass(prep)
    if result is None:raise RuntimeError('All official windows failed')
    q=np.asarray(result['q']);q[:,3:7]=q[:,[6,3,4,5]]
    q[:,3:7]/=np.linalg.norm(q[:,3:7],axis=1,keepdims=True)
    dest=a.output
    np.savez_compressed(dest,qpos=q,time_s=times,robot_joint_names=np.asarray(names),fps=fps,
                        inferred_toe_stance=feet)
    record={'method':'HOI-Retarget official window NLP + independent Stage-1 HDMI28 reference',
        'status':'kinematic_retarget_complete_not_physical_success','actors':1,'frames':len(times),
        'source_repository':'https://github.com/leggedrobotics/hoi-retarget',
        'commit':json.loads((a.source/'SOURCE_MANIFEST.json').read_text())['commit'],
        'official_window_source_sha256':sha(window_path),'official_preprocessor_source_sha256':sha(projector_path),
        'model_sha256':sha(a.model),'stage1_reference_sha256':sha(a.stage1_reference),
        'stage1_reference_path':str(a.stage1_reference),'output_sha256':sha(dest),
        'fk_verification':{'sample_count':20,'world_position_max_error_m':pos_error,'rotation_matrix_max_error':rot_error},
        'coordinate_estimate':frame_audit,
        'elapsed_seconds':time.perf_counter()-started,'window_status':solver.window_status,
        'inferred_stance_frames':feet.sum(axis=0).tolist(),
        'window_size':a.window_size,'pinocchio_backend_executed':False,'casadi_backend_executed':True,
        'object_contact_constraints_executed':False,
        'limitations':['Vita target and compiled-MuJoCo-to-CasADi FK are local adaptations',
          'Official window objective, constraints, IPOPT solve, stitching and status reporting executed',
          'No measured object contact labels or 6D object trajectory; object contact terms disabled',
          'Foot stance inferred from source toe speed and ground proxy',
          'Stage-1 reference provided by independent UMR or OmniRetarget track',
          'No dynamics or SONIC evaluation in this run']}
    if any(row['status']!='solved' for row in solver.window_status):
        record['status']='window_output_contains_failed_or_capped_windows'
    dest.with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)

if __name__=='__main__':main()
