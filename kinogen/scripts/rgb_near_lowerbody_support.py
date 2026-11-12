"""Explicit source-cylinder seat variant with world foot/hand targets retained.

This is kinematic reference preparation, never a physics success certificate.
The input reference and original source arrays are retained independently.
"""
from pathlib import Path
import argparse
import hashlib
import json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/data/KinoGen'))
    p.add_argument('--input',default='outputs/rgb_near_pipeline/reference_v16_O6_grasp.npz')
    p.add_argument('--scene',default='outputs/rgb_near_pipeline/assets/vita_bench_two52_O6_v5_source_hipcyl.xml')
    p.add_argument('--output',default='outputs/rgb_near_pipeline/reference_v17_source_hipcyl.npz')
    p.add_argument('--root-lowering',type=float,default=.040)
    p.add_argument('--frame-stop',type=int)
    p.add_argument('--max-nfev',type=int,default=45)
    a=p.parse_args()
    import numpy as np
    import mujoco as mj
    from scipy.optimize import least_squares
    root=a.root;inp=root/a.input;scene=root/a.scene;out=root/a.output
    z=np.load(inp,allow_pickle=False);payload={k:z[k]for k in z.files}
    original=np.asarray(z['qpos']);N=len(original);n=min(N,a.frame_stop or N)
    q=original[:n].copy();names=[str(s)for s in z['dof_names']]
    m=mj.MjModel.from_xml_path(str(scene));d=mj.MjData(m)
    def fk():mj.mj_kinematics(m,d);mj.mj_comPos(m,d)
    maps=[]
    for actor in range(2):
        pre=f'p{actor+1}_';rj=mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,pre+'vita_demo_free_root')
        js=np.array([mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,pre+s)for s in names])
        if min(rj,*js)<0:raise ValueError('Joint ABI mismatch')
        maps.append((int(m.jnt_qposadr[rj]),m.jnt_qposadr[js],js))
    def reset(frame,arr):
        for ac,(rq,qa,_)in enumerate(maps):
            d.qpos[rq:rq+7]=arr[frame,ac,:7];d.qpos[qa]=arr[frame,ac,7:]
        fk()
    targets={};errors=[];jp=np.zeros((3,m.nv));jr=np.zeros_like(jp)
    for ac,(rq,qa,js)in enumerate(maps):
      for side in ('left','right'):
       for limb in ('leg','arm'):
        if limb=='leg':
            selected=[k for k,s in enumerate(names)if s.startswith(side+'_')and any(v in s for v in('hip_','knee_','ankle_'))]
            body=f'p{ac+1}_{side}_ankle_roll_link'
        else:
            selected=[k for k,s in enumerate(names)if s.startswith(side+'_')and any(v in s for v in('shoulder_','elbow_','wrist_'))]
            body=f'p{ac+1}_{"lh" if side=="left"else"rh"}_hand_base_link'
        bid=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,body)
        if bid<0:raise ValueError(body)
        sel=np.array(selected);jointids=js[sel];qadr=qa[sel];vadr=m.jnt_dofadr[jointids]
        targetp=[];targetR=[]
        for f in range(n):
            reset(f,original);targetp.append(d.xpos[bid].copy());targetR.append(d.xmat[bid].reshape(3,3).copy())
        targetp=np.array(targetp);targetR=np.array(targetR);targets[body]=(targetp,targetR)
        # Lower both roots once; individual limb solves leave all other joints fixed.
        q[:,ac,2]=original[:n,ac,2]-a.root_lowering
        delta=np.zeros(len(sel));lim=m.jnt_range[jointids];maxpos=maxrot=0.;counts=[]
        for f in range(n):
            reset(f,q);old=original[f,ac,7+sel]
            initial=np.clip(old+delta,lim[:,0]+1e-8,lim[:,1]-1e-8)
            def fun(x):
                d.qpos[qadr]=x;fk();R=d.xmat[bid].reshape(3,3)
                return np.r_[10*(d.xpos[bid]-targetp[f]),.3*(R-targetR[f]).T.ravel(),1e-7*(x-old)]
            def jac(x):
                d.qpos[qadr]=x;fk();mj.mj_jacBody(m,d,jp,jr,bid);R=d.xmat[bid].reshape(3,3)
                rot=np.concatenate([np.cross(jr[:,vadr].T,R[:,k]).T for k in range(3)],axis=0)
                return np.vstack((10*jp[:,vadr],.3*rot,1e-7*np.eye(len(sel))))
            fit=least_squares(fun,initial,jac=jac,bounds=(lim[:,0]+1e-9,lim[:,1]-1e-9),
                              max_nfev=a.max_nfev,ftol=1e-11,xtol=1e-11,gtol=1e-11)
            q[f,ac,7+sel]=fit.x;delta=fit.x-old;fun(fit.x)
            ep=float(np.linalg.norm(d.xpos[bid]-targetp[f]));er=float(np.arccos(np.clip((np.trace(targetR[f].T@d.xmat[bid].reshape(3,3))-1)/2,-1,1)))
            errors.append({'frame':f,'body':body,'position_error_m':ep,'rotation_error_rad':er,'nfev':fit.nfev})
            maxpos=max(maxpos,ep);maxrot=max(maxrot,er);counts.append(fit.nfev)
        print(body,'position',maxpos,'rotation',maxrot,'mean_nfev',np.mean(counts),flush=True)
    # Verify shoe attachment geometry without replacing the preserved world targets.
    shoe_errors=[];switch=int(z['shoe_release_frame'])
    leftankle=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'p1_left_ankle_roll_link')
    righthand=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'p1_rh_hand_base_link')
    for f in range(n):
        reset(f,q);wear=f<switch;bid=leftankle if wear else righthand
        lp=z['shoe_wear_position_local_to_left_ankle']if wear else z['shoe_hand_position_local_to_right_hand']
        lr=z['shoe_wear_rotation_local_to_left_ankle']if wear else z['shoe_hand_rotation_local_to_right_hand']
        R=d.xmat[bid].reshape(3,3);expectedp=d.xpos[bid]+R@lp;expectedR=R@lr
        rawR=np.zeros(9);mj.mju_quat2Mat(rawR,z['object_target_pose'][f,3:])
        shoe_errors.append([float(np.linalg.norm(expectedp-z['object_target_pose'][f,:3])),float(np.arccos(np.clip((np.trace(rawR.reshape(3,3).T@expectedR)-1)/2,-1,1)))])
    for key,value in list(payload.items()):
        if value.ndim and value.shape[0]==N:payload[key]=value[:n]
    payload['qpos']=q;payload['qpos_p1']=q[:,0];payload['qpos_p2']=q[:,1]
    payload['pre_source_cylinder_reference_qpos']=original[:n]
    payload['source_cylinder_root_lowering_m']=np.full((n,2),a.root_lowering)
    payload['source_cylinder_attachment_consistency_error_m_rad']=np.array(shoe_errors)
    payload['source_cylinder_parent_reference_sha256']=np.array(sha(inp))
    payload['source_cylinder_model_sha256']=np.array(sha(scene))
    out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,**payload)
    report={'schema':'rgb_near_source_cylinder_seat_variant_v1','kinematic_only':True,
            'input':str(inp),'input_sha256':sha(inp),'output':str(out),'output_sha256':sha(out),
            'scene':str(scene),'scene_sha256':sha(scene),'frames':n,'root_lowering_m':a.root_lowering,
            'max_limb_position_error_m':max(x['position_error_m']for x in errors),
            'max_limb_rotation_error_rad':max(x['rotation_error_rad']for x in errors),
            'max_shoe_attachment_position_error_m':float(np.max(np.array(shoe_errors)[:,0])),
            'shoe_world_trajectory_unchanged':bool(np.array_equal(payload['object_target_pose'],z['object_target_pose'][:n])),
            'raw_source_preserved':True,'source_hand_invalid_grasp_inferred':True,
            'semantics':'Deliberate cylinder contact branch; root lowered and legs/arms re-solved to preserve previous foot/hand world poses. No physical equilibrium claim. Capsule training baseline retained.',
            'per_limb_frame_errors':errors}
    out.with_suffix('.json').write_text(json.dumps(report,indent=2));out.with_suffix('.generator.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:v for k,v in report.items()if k!='per_limb_frame_errors'}),flush=True)


if __name__=='__main__':main()
