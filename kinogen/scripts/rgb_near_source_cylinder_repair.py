"""Repair bounded limb reachability by explicit seated pelvis XY adjustment."""
from pathlib import Path
import argparse
import hashlib
import json


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/data/KinoGen'))
    p.add_argument('--original',default='outputs/rgb_near_pipeline/reference_v16_O6_grasp.npz')
    p.add_argument('--input',default='outputs/rgb_near_pipeline/reference_v17_source_hipcyl.npz')
    p.add_argument('--output',default='outputs/rgb_near_pipeline/reference_v18_source_hipcyl_xy.npz')
    p.add_argument('--scene',default='outputs/rgb_near_pipeline/assets/vita_bench_two52_O6_v5_source_hipcyl.xml')
    p.add_argument('--max-xy',type=float,default=.15)
    p.add_argument('--relax-root-z',action='store_true',help='Allow returning up to input root lowering where hard limb limits require it.')
    a=p.parse_args()
    import numpy as np
    import mujoco as mj
    from scipy.optimize import least_squares
    inp=a.root/a.input;orig=a.root/a.original;scene=a.root/a.scene;out=a.root/a.output
    z=np.load(inp,allow_pickle=False);raw=np.load(orig,allow_pickle=False)
    payload={k:z[k]for k in z.files};q=z['qpos'].copy();N=len(q);names=[str(s)for s in z['dof_names']]
    m=mj.MjModel.from_xml_path(str(scene));d=mj.MjData(m);jp=np.zeros((3,m.nv));jr=np.zeros_like(jp)
    def fk():mj.mj_kinematics(m,d);mj.mj_comPos(m,d)
    maps=[]
    for ac in range(2):
        pre=f'p{ac+1}_';rj=mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,pre+'vita_demo_free_root')
        js=np.array([mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,pre+s)for s in names])
        if min(rj,*js)<0:raise ValueError('Schema mismatch')
        maps.append((int(m.jnt_qposadr[rj]),m.jnt_qposadr[js],js))
    def reset(f,arr):
        for ac,(rq,qa,js)in enumerate(maps):d.qpos[rq:rq+7]=arr[f,ac,:7];d.qpos[qa]=arr[f,ac,7:]
        fk()
    allerrors=[]
    for ac,(rq,qa,js)in enumerate(maps):
        sel=np.array([k for k,s in enumerate(names)if not s.startswith(('lh_','rh_','neck_','waist_'))])
        qa_sel=qa[sel];jid=js[sel];va=m.jnt_dofadr[jid];nj=len(sel)
        bodynames=[f'p{ac+1}_left_ankle_roll_link',f'p{ac+1}_right_ankle_roll_link',f'p{ac+1}_lh_hand_base_link',f'p{ac+1}_rh_hand_base_link']
        bodies=[mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,b)for b in bodynames]
        tp=[];tR=[]
        for f in range(N):reset(f,raw['qpos']);tp.append(d.xpos[bodies].copy());tR.append(d.xmat[bodies].reshape(4,3,3).copy())
        nr=3 if a.relax_root_z else 2
        tp=np.array(tp);tR=np.array(tR);prev=np.zeros(nr);previous_delta=np.zeros(nj)
        lim=m.jnt_range[jid];rlo=[-a.max_xy]*2+([0.]if nr==3 else[]);rhi=[a.max_xy]*2+([float(z['source_cylinder_root_lowering_m'][:,ac].max())]if nr==3 else[])
        lo=np.r_[rlo,lim[:,0]+1e-9];hi=np.r_[rhi,lim[:,1]-1e-9]
        for f in range(N):
            reset(f,q);base=q[f,ac,:nr].copy();base[:2]=raw['qpos'][f,ac,:2];old=raw['qpos'][f,ac,7+sel]
            initial=np.r_[prev,np.clip(q[f,ac,7+sel]+previous_delta*.1,lim[:,0]+1e-8,lim[:,1]-1e-8)]
            def fun(x):
                d.qpos[rq:rq+nr]=base+x[:nr];d.qpos[qa_sel]=x[nr:];fk()
                rows=[]
                for k,b in enumerate(bodies):rows.extend([10*(d.xpos[b]-tp[f,k]),.3*(d.xmat[b].reshape(3,3)-tR[f,k]).T.ravel()])
                return np.concatenate(rows+[.002*x[:nr],1e-6*(x[nr:]-old)])
            def jac(x):
                d.qpos[rq:rq+nr]=base+x[:nr];d.qpos[qa_sel]=x[nr:];fk();rows=[]
                for b in bodies:
                    mj.mj_jacBody(m,d,jp,jr,b);R=d.xmat[b].reshape(3,3)
                    rot=np.concatenate([np.cross(jr[:,va].T,R[:,k]).T for k in range(3)],axis=0)
                    rows.extend([np.c_[10*np.eye(3)[:,:nr],10*jp[:,va]],np.c_[np.zeros((9,nr)),.3*rot]])
                return np.vstack(rows+[np.c_[.002*np.eye(nr),np.zeros((nr,nj))],np.c_[np.zeros((nj,nr)),1e-6*np.eye(nj)]])
            fit=least_squares(fun,np.clip(initial,lo+1e-8,hi-1e-8),jac=jac,bounds=(lo,hi),max_nfev=85,ftol=1e-10,xtol=1e-10,gtol=1e-10)
            fun(fit.x);q[f,ac,:nr]=base+fit.x[:nr];q[f,ac,7+sel]=fit.x[nr:];prev=fit.x[:nr];previous_delta=fit.x[nr:]-old
            for k,b in enumerate(bodies):
                allerrors.append({'frame':f,'body':bodynames[k],'position_error_m':float(np.linalg.norm(d.xpos[b]-tp[f,k])),
                                  'rotation_error_rad':float(np.arccos(np.clip((np.trace(tR[f,k].T@d.xmat[b].reshape(3,3))-1)/2,-1,1)))})
            if f%50==0:print('actor',ac,'frame',f,'xy',prev,'residual',np.linalg.norm(fit.fun[:48]),flush=True)
    payload['qpos']=q;payload['qpos_p1']=q[:,0];payload['qpos_p2']=q[:,1]
    payload['source_cylinder_root_xy_adjustment_m']=q[:,:,:2]-raw['qpos'][:N,:,:2]
    payload['source_cylinder_root_lowering_m']=raw['qpos'][:N,:,2]-q[:,:,2]
    payload['source_cylinder_world_pose_repair_parent_sha256']=np.array(sha(inp))
    # Recompute foot/hand rigid attachment agreement without altering target poses.
    switch=int(z['shoe_release_frame']);sq=int(m.jnt_qposadr[mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,'shoe_free')]);shoeerrors=[]
    ankle=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'p1_left_ankle_roll_link');hand=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'p1_rh_hand_base_link')
    for f in range(N):
        reset(f,q);wear=f<switch;bid=ankle if wear else hand;R=d.xmat[bid].reshape(3,3)
        lp=z['shoe_wear_position_local_to_left_ankle']if wear else z['shoe_hand_position_local_to_right_hand']
        lr=z['shoe_wear_rotation_local_to_left_ankle']if wear else z['shoe_hand_rotation_local_to_right_hand'];tr=np.zeros(9);mj.mju_quat2Mat(tr,z['object_target_pose'][f,3:])
        shoeerrors.append([float(np.linalg.norm(d.xpos[bid]+R@lp-z['object_target_pose'][f,:3])),float(np.arccos(np.clip((np.trace(tr.reshape(3,3).T@R@lr)-1)/2,-1,1)))])
    payload['source_cylinder_attachment_consistency_error_m_rad']=np.array(shoeerrors)
    np.savez_compressed(out,**payload)
    report={'schema':'rgb_near_source_cylinder_world_pose_repair_v1','kinematic_only':True,'frames':N,
            'input':str(inp),'input_sha256':sha(inp),'original':str(orig),'original_sha256':sha(orig),
            'output':str(out),'output_sha256':sha(out),'scene':str(scene),'scene_sha256':sha(scene),
            'max_world_position_error_m':max(r['position_error_m']for r in allerrors),
            'max_world_rotation_error_rad':max(r['rotation_error_rad']for r in allerrors),
            'max_shoe_attachment_error_m':float(np.max(np.array(shoeerrors)[:,0])),
            'max_root_xy_adjustment_m':float(np.linalg.norm(payload['source_cylinder_root_xy_adjustment_m'],axis=-1).max()),
            'root_lowering_range_m':[float(payload['source_cylinder_root_lowering_m'].min()),float(payload['source_cylinder_root_lowering_m'].max())],
            'max_root_xy_speed_m_s':float(np.linalg.norm(np.diff(q[:,:,:2],axis=0)/np.diff(z['time_s'])[:,None,None],axis=-1).max()),
            'shoe_world_target_unchanged':bool(np.array_equal(z['object_target_pose'],raw['object_target_pose'][:N])),
            'per_body_frame_errors':allerrors}
    out.with_suffix('.json').write_text(json.dumps(report,indent=2));out.with_suffix('.generator.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:v for k,v in report.items()if k!='per_body_frame_errors'}),flush=True)


if __name__=='__main__':main()
