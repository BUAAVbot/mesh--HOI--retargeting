"""Run official HOI-Retarget window NLP with a validated compiled-Vita CasADi FK.

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


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--shared',type=Path,required=True)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--gmr-reference',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--frames',type=int,default=172)
    ap.add_argument('--window-size',type=int,default=60)
    a=ap.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    started=time.perf_counter()
    sys.path.append('/data/grail-envs/retarget/lib/python3.10/site-packages')
    import numpy as np
    import mujoco as mj
    import casadi as ca
    from scipy.spatial.transform import Rotation
    manifest=json.loads((a.shared/'source_manifest.json').read_text())
    observations=np.load(a.shared/'common_observations.npz',allow_pickle=False)
    times=observations['time_s'][:a.frames]
    source_joints=observations['repaired_source_joints_zup'][:len(times)]
    gmr=np.load(a.gmr_reference,allow_pickle=False)
    qref=gmr['qpos']
    if qref.shape[0]==2:qref=qref.transpose(1,0,2)
    qref=qref[:len(times)].copy()
    model=mj.MjModel.from_xml_path(manifest['model'])
    names=manifest['dof_names']
    body_names=[mj.mj_id2name(model,mj.mjtObj.mjOBJ_BODY,i) for i in range(1,model.nbody)]
    contact_name='seat_contact_proxy'
    body_names.append(contact_name)
    # The virtual contact frame is an explicitly declared rear/lower pelvis
    # proxy. It is not an observed fingertip/contact point.
    contact_local=np.array([-.10,0.,-.20])

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
                if name==contact_name:p=pos[base]+rot[base]@ca.DM(contact_local);R=rot[base]
                else:p=pos[idx+1];R=rot[idx+1]
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
        q=qref[k%len(times),k%2].copy()
        q[7:]=np.clip(q[7:]+rng.normal(0,.12,52),fk.q_lower[7:],fk.q_upper[7:])
        d.qpos[:]=q;mj.mj_forward(model,d)
        qp=q.copy();qp[3:7]=q[[4,5,6,3]]
        for i,name in enumerate(body_names[:-1],1):
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
    support=json.loads(Path(manifest['support_geometry']).read_text())
    cfg=ProjectionConfig(urdf_path=manifest['model'],base_body_name='base_link',
       tracking=TrackingWeights(link_weights_world={'neck_yaw_link':1.,'left_ankle_roll_link':100.,'right_ankle_roll_link':100.},link_weights_base={'waist_pitch_link':1.}),
       window=WindowConfig(window_size=a.window_size))
    outputs=[];statuses=[];contact_counts=[];stance_counts=[]
    for actor in range(2):
        world_p=[];world_q=[];local_p=[];local_q=[];flags=[];contact_points=[];feet=[]
        feet_speed=np.linalg.norm(np.gradient(source_joints[:,actor,[10,11]],times,axis=0),axis=-1)
        for frame,q in enumerate(qref[:,actor]):
            d.qpos[:]=q;mj.mj_forward(model,d)
            base_p=d.xpos[1].copy();base_R=d.xmat[1].reshape(3,3).copy()
            wp=np.vstack([d.xpos[1:].copy(),base_p+base_R@contact_local])
            wr=np.concatenate([d.xmat[1:].reshape(-1,3,3).copy(),base_R[None]],axis=0)
            wq=Rotation.from_matrix(wr).as_quat()[:,[3,0,1,2]]
            world_p.append(wp);world_q.append(wq);local_p.append((wp-base_p)@base_R)
            local_q.append(Rotation.from_matrix(base_R.T@wr).as_quat()[:,[3,0,1,2]])
            # Inferred contact, independent of whether the optimizer later succeeds.
            hips=source_joints[frame,actor,1]-source_joints[frame,actor,2]
            forward=np.cross(hips,np.array([0.,0.,1.]));forward/=max(np.linalg.norm(forward),1e-9)
            rear=source_joints[frame,actor,0]-.10*forward+np.array([0.,0.,-.20])
            candidates=[]
            for seat in support['seats']:
                yaw=seat['yaw_rad'];R=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
                delta=(rear[:2]-seat['center_xy_m'])@R
                clipped=np.clip(delta,-np.asarray(seat['half_size_xy_m']),seat['half_size_xy_m'])
                target=np.r_[np.asarray(seat['center_xy_m'])+R@clipped,seat['top_z_m']]
                dist=np.linalg.norm(rear[:2]-target[:2]);dz=abs(rear[2]-target[2])
                candidates.append((dist+dz,target,dist<.18 and dz<.16))
            _,target,is_contact=min(candidates,key=lambda row:row[0])
            flags.append(bool(is_contact));contact_points.append(target)
            floor=np.array(support['ground']['z_eq_ax_by_c'])
            toe=source_joints[frame,actor,[10,11]]
            h=toe[:,2]-(toe[:,0]*floor[0]+toe[:,1]*floor[1]+floor[2])
            feet.append((h<.1)&(feet_speed[frame]<.30))
        flags=np.asarray(flags);feet=np.asarray(feet)
        md={'fps':30.,'root_pos':qref[:,actor,:3],'root_rot':qref[:,actor,3:7],
            'dof_pos':qref[:,actor,7:],'dof_names':names,'link_body_list':body_names,
            'world_body_pos':np.asarray(world_p),'world_body_orient':np.asarray(world_q),
            'local_body_pos':np.asarray(local_p),'local_body_orient':np.asarray(local_q),
            'object_pos':np.zeros((len(times),3)),'object_rot':np.tile([1.,0.,0.,0.],(len(times),1)),
            'object_contact_sequence':flags[:,None],'object_ground_contact_sequence':np.ones((len(times),1)),
            'contact_link_names':[contact_name],'per_link_contact_flags':flags[:,None],
            'fixed_contact_points_per_frame_in_object_frame':np.asarray(contact_points)[:,None],
            'robot_foot_ground_contact_flags':feet}
        solver=PinocchioMotionProjectorWindow(cfg,0.,[contact_name],compute_analysis=True)
        prep=solver._preprocess_motion(md)
        result=solver._unified_pass(prep)
        if result is None:raise RuntimeError('All official contact windows failed')
        q=np.asarray(result['q']);q[:,3:7]=q[:,[6,3,4,5]]
        q[:,3:7]/=np.linalg.norm(q[:,3:7],axis=1,keepdims=True)
        outputs.append(q);statuses.append(solver.window_status)
        contact_counts.append(int(flags.sum()));stance_counts.append(feet.sum(axis=0).tolist())
        np.savez_compressed(a.output/f'person_{actor+1:02d}.npz',qpos=q,time_s=times,
                            inferred_seat_contact=flags,inferred_toe_stance=feet)
    dest=a.output/'reference.npz'
    np.savez_compressed(dest,qpos=np.stack(outputs,axis=1),time_s=times,dof_names=np.asarray(names),frame_ids=observations['frame_ids'][:len(times)])
    record={'method':'HOI-Retarget official contact-window NLP + GMR stage1 + compiled-Vita CasADi FK adapter',
        'status':'kinematic_retarget_complete_not_physical_success','actors':2,'frames':len(times),
        'source_repository':'https://github.com/leggedrobotics/hoi-retarget',
        'commit':json.loads((a.source/'SOURCE_MANIFEST.json').read_text())['commit'],
        'official_window_source_sha256':sha(window_path),'official_preprocessor_source_sha256':sha(projector_path),
        'shared_model_sha256':sha(Path(manifest['model'])),'shared_manifest_sha256':sha(a.shared/'source_manifest.json'),
        'stage1_reference_sha256':sha(a.gmr_reference),'output_sha256':sha(dest),
        'fk_verification':{'sample_count':20,'world_position_max_error_m':pos_error,'rotation_matrix_max_error':rot_error},
        'elapsed_seconds':time.perf_counter()-started,'window_status':statuses,
        'inferred_seated_contact_frames':contact_counts,'inferred_stance_frames':stance_counts,
        'window_size':a.window_size,'pinocchio_backend_executed':False,'casadi_backend_executed':True,
        'limitations':['Vita target and compiled-MuJoCo-to-CasADi FK are local adaptations, not stock Pinocchio/Vita support',
          'Official window objective, constraints, IPOPT solve, stitching and status reporting executed without modification',
          'Stage1 is separately evaluated upstream GMR Vita adaptation, not stock G1 profile',
          'Seat-contact and toe-stance labels are explicitly inferred from the shared repaired source; no measured contact truth',
          'Pelvis rear/lower virtual frame is an engineering contact proxy, not observed MANO data',
          'No scene/trajectory rescaling; two actors solved independently; no inter-robot collision constraint',
          'Compiled MJCF has no joint-velocity limits; infinity leaves the official conditional velocity-limit branch inactive',
          'No dynamics or SONIC evaluation in this run']}
    if any(row['status']!='solved' for group in statuses for row in group):
        record['status']='contact_window_output_contains_failed_or_capped_windows'
    (a.output/'report.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)


if __name__=='__main__':main()
