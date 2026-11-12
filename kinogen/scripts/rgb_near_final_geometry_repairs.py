"""Local reference repairs: p2 late support and nonholding-hand clearance.

No integration; preserves the input and raw source. Task shoe is unchanged.
"""
from pathlib import Path
import argparse
import hashlib
import json


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('/data/KinoGen'))
    p.add_argument('--input',default='outputs/rgb_near_pipeline/reference_v25_planted_com_full398.npz')
    p.add_argument('--scene',default='outputs/rgb_near_pipeline/assets/vita_bench_two52_O6_v5_source_hipcyl.xml')
    p.add_argument('--output',default='outputs/rgb_near_pipeline/reference_v26_geometry_repaired_full398.npz')
    a=p.parse_args()
    import numpy as np
    import mujoco as mj
    from scipy.optimize import least_squares
    z=np.load(a.root/a.input,allow_pickle=False);old=z['qpos'];q=old.copy();names=[str(v)for v in z['dof_names']];t=z['time_s'];N=len(t)
    m=mj.MjModel.from_xml_path(str(a.root/a.scene));d=mj.MjData(m);maps=[]
    for pre in('p1_','p2_'):
        r=mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,pre+'vita_demo_free_root');js=np.array([mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,pre+n)for n in names]);maps.append((int(m.jnt_qposadr[r]),m.jnt_qposadr[js],js))
    sq=int(m.jnt_qposadr[mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,'shoe_free')]);jp=np.zeros((3,m.nv));jr=np.zeros_like(jp)
    def fk():mj.mj_kinematics(m,d);mj.mj_comPos(m,d)
    def gn(g):return mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,int(g))or''
    def bn(b):return mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,int(b))or''
    def reset(f,arr):
        for ac,(rq,qa,_)in enumerate(maps):d.qpos[rq:rq+7]=arr[f,ac,:7];d.qpos[qa]=arr[f,ac,7:]
        d.qpos[sq:sq+7]=z['object_target_pose'][f];fk()
    feet={}
    for ac in range(2):
        for side in('left','right'):feet[ac,side]=[g for g in range(m.ngeom)if m.geom_contype[g]and gn(g).startswith(f'p{ac+1}_collision_foot_{side[0]}_')]
    def floorz(ac,side):return min(d.geom_xpos[g,2]-m.geom_size[g,0]-m.geom_size[g,1]*abs(d.geom_xmat[g].reshape(3,3)[2,2])for g in feet[ac,side])
    seat=mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,'bench_seat_proxy');hips=[mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,'p2_collision_'+s+'_hip_roll_cylinder')for s in('l','r')];ankles=[mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'p2_'+s+'_ankle_roll_link')for s in('left','right')]
    p2rows=[];rq,qa,js=maps[1];lim=m.jnt_range[js[:12]];previous=np.zeros(2)
    for f in range(N):
        if t[f]<12.7:continue
        blend=np.clip((t[f]-12.7)/.9,0,1);blend=blend*blend*(3-2*blend);reset(f,old);targetp=d.xpos[ankles].copy();targetR=d.xmat[ankles].reshape(2,3,3).copy();oldgap=np.array([mj.mj_geomDistance(m,d,g,seat,1.,np.zeros(6))for g in hips]);oldfloor=np.array([floorz(1,s)for s in('left','right')]);goal_gap=(1-blend)*oldgap+blend*(-.001);goal_floor=(1-blend)*oldfloor+blend*(-.0008);targetp[:,2]+=goal_floor-oldfloor;rp=old[f,1,:3].copy();rQ=old[f,1,3:7].copy();leg=old[f,1,7:19].copy()
        def bind(x):
            d.qpos[rq:rq+3]=rp+[0,0,x[0]];mj.mju_mulQuat(d.qpos[rq+3:rq+7],rQ,np.array([np.cos(x[1]/2),np.sin(x[1]/2),0,0]));d.qpos[qa[:12]]=x[2:];fk()
        def fun(x):
            bind(x);rows=[]
            for k,b in enumerate(ankles):rows.extend([8*(d.xpos[b]-targetp[k]),.25*(d.xmat[b].reshape(3,3)-targetR[k]).ravel()])
            return np.r_[np.concatenate(rows),40*(np.array([mj.mj_geomDistance(m,d,g,seat,1.,np.zeros(6))for g in hips])-goal_gap),40*(np.array([floorz(1,s)for s in('left','right')])-goal_floor),.001*x[:2]]
        fit=least_squares(fun,np.r_[previous,leg],bounds=(np.r_[-.03,-.1,lim[:,0]+1e-9],np.r_[.02,.1,lim[:,1]-1e-9]),max_nfev=65,ftol=1e-10,xtol=1e-10,gtol=1e-10);fun(fit.x);previous=fit.x[:2];q[f,1,:7]=d.qpos[rq:rq+7];q[f,1,7:19]=fit.x[2:]
        p2rows.append({'frame':f,'extra_root_dz_roll':fit.x[:2].tolist(),'foot_goal_error_m':np.linalg.norm(d.xpos[ankles]-targetp,axis=1).tolist(),'hip_gaps_m':[float(mj.mj_geomDistance(m,d,g,seat,1.,np.zeros(6)))for g in hips],'foot_floor_z_m':[float(floorz(1,s))for s in('left','right')]})
    # Keep p2 world wrists where they were before the millimetre support repair.
    for side,hpre in [('left','lh_'),('right','rh_')]:
        hand=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'p2_'+hpre+'hand_base_link');sel=np.array([k for k,s in enumerate(names)if s.startswith(side+'_')and any(v in s for v in('shoulder','elbow','wrist'))]);j=js[sel];va=m.jnt_dofadr[j];qal=qa[sel];limits=m.jnt_range[j]
        for f in range(N):
            if t[f]<12.7:continue
            reset(f,old);tp=d.xpos[hand].copy();R=d.xmat[hand].reshape(3,3).copy();reset(f,q)
            def afun(x):d.qpos[qal]=x;fk();return np.r_[10*(d.xpos[hand]-tp),.3*(d.xmat[hand].reshape(3,3)-R).T.ravel()]
            def ajac(x):
                afun(x);mj.mj_jacBody(m,d,jp,jr,hand);r=d.xmat[hand].reshape(3,3);return np.vstack([10*jp[:,va],.3*np.concatenate([np.cross(jr[:,va].T,r[:,k]).T for k in range(3)],axis=0)])
            fit=least_squares(afun,q[f,1,7+sel],jac=ajac,bounds=(limits[:,0]+1e-9,limits[:,1]-1e-9),max_nfev=50,ftol=1e-10,xtol=1e-10,gtol=1e-10);q[f,1,7+sel]=fit.x
    # Ensure the supporting bare left foot remains a real contact when p1's
    # opposite foot lifts late; change only its tiny vertical clearance.
    rq,qa,js=maps[0];sel=np.arange(6);j=js[sel];va=m.jnt_dofadr[j];qal=qa[sel];limits=m.jnt_range[j];ankle=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'p1_left_ankle_roll_link');lateleft=[]
    for f in range(N):
        if t[f]<13.:continue
        reset(f,q);leftz=floorz(0,'left');rightz=floorz(0,'right')
        if rightz<.02 or abs(leftz)>.02:continue
        tp=d.xpos[ankle].copy();R=d.xmat[ankle].reshape(3,3).copy();tp[2]+=-.0012-leftz
        def lfun(x):d.qpos[qal]=x;fk();return np.r_[10*(d.xpos[ankle]-tp),.3*(d.xmat[ankle].reshape(3,3)-R).T.ravel()]
        def ljac(x):
            lfun(x);mj.mj_jacBody(m,d,jp,jr,ankle);r=d.xmat[ankle].reshape(3,3);return np.vstack([10*jp[:,va],.3*np.concatenate([np.cross(jr[:,va].T,r[:,k]).T for k in range(3)],axis=0)])
        fit=least_squares(lfun,q[f,0,7:13],jac=ljac,bounds=(limits[:,0]+1e-9,limits[:,1]-1e-9),max_nfev=40);lfun(fit.x);q[f,0,7:13]=fit.x;lateleft.append({'frame':f,'vertical_shift_m':float(-.0012-leftz),'final_floor_z_m':float(floorz(0,'left'))})
    # Nonholding left arm moves modestly outward/forward, with actual native
    # penetrating hand-shoe/bench contacts penalized; final SAT audit separate.
    hand=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'p1_lh_hand_base_link');sel=np.array([k for k,s in enumerate(names)if s.startswith('left_')and any(v in s for v in('shoulder','elbow','wrist'))]);j=js[sel];qal=qa[sel];limits=m.jnt_range[j];va=m.jnt_dofadr[j];hg=[g for g in range(m.ngeom)if m.geom_contype[g]and bn(m.geom_bodyid[g]).startswith(('p1_lh_','p1_left_wrist'))];ob=[g for g in range(m.ngeom)if m.geom_contype[g]and gn(g).startswith(('shoe_','bench_'))];pairs={tuple(sorted((g,b))):i for i,(g,b)in enumerate((g,b)for g in hg for b in ob)};leftrows=[];previous_delta=np.zeros(len(sel))
    for f in range(N):
        if f<250:continue
        reset(f,q);original=q[f,0,7+sel].copy();tp=d.xpos[hand].copy();R=d.xmat[hand].reshape(3,3).copy();u=np.clip((t[f]-t[250])/.8,0,1);u=u*u*(3-2*u);tp+=u*np.array([.04,-.02,.015])
        def hfun(x):
            d.qpos[qal]=x;fk();mj.mj_collision(m,d);gaps=np.full(len(pairs),.3)
            for c in d.contact:
                key=tuple(sorted((int(c.geom1),int(c.geom2))))
                if key in pairs:gaps[pairs[key]]=min(gaps[pairs[key]],float(c.dist))
            return np.r_[3*(d.xpos[hand]-tp),.08*(d.xmat[hand].reshape(3,3)-R).ravel(),80*np.minimum(gaps+.0004,0),.002*(x-original)]
        step=3*float(t[f]-t[f-1]);lower=np.maximum(limits[:,0]+1e-8,original+previous_delta-step);upper=np.minimum(limits[:,1]-1e-8,original+previous_delta+step)
        if np.any(lower>=upper):lower=limits[:,0]+1e-8;upper=limits[:,1]-1e-8
        fit=least_squares(hfun,np.clip(original+previous_delta,lower+1e-9,upper-1e-9),bounds=(lower,upper),max_nfev=90,diff_step=1e-5);hfun(fit.x);q[f,0,7+sel]=fit.x;previous_delta=fit.x-original;leftrows.append({'frame':f,'hand_goal_error_m':float(np.linalg.norm(d.xpos[hand]-tp)),'max_added_joint_angle_rad':float(abs(previous_delta).max())})
    payload={k:z[k]for k in z.files};payload['qpos']=q;payload['qpos_p1']=q[:,0];payload['qpos_p2']=q[:,1];payload['pre_final_geometry_repair_qpos']=old;payload['final_geometry_repair_parent_sha256']=np.array(sha(a.root/a.input));payload['O6_grasp_geometric_quality_passed']=np.array(False)
    out=a.root/a.output;np.savez_compressed(out,**payload);report={'schema':'rgb_near_final_geometry_repairs_v1','kinematic_only':True,'input_sha256':sha(a.root/a.input),'output_sha256':sha(out),'p2_late_support':p2rows,'p1_late_left_grounding':lateleft,'p1_nonholding_left_arm':leftrows,'shoe_world_target_unchanged':bool(np.array_equal(payload['object_target_pose'],z['object_target_pose']))};out.with_suffix('.json').write_text(json.dumps(report,indent=2));out.with_suffix('.generator.py').write_bytes(Path(__file__).read_bytes());print(json.dumps({k:v for k,v in report.items()if not isinstance(v,list)}),flush=True)


if __name__=='__main__':main()
