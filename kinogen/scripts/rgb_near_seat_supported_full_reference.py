"""Keyframe-only seated lift feasibility probe; no dynamics certification."""
from pathlib import Path
import argparse
import json
import hashlib


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('/data/KinoGen'))
    p.add_argument('--input',default='outputs/rgb_near_pipeline/reference_v21_grounded_full398.npz')
    p.add_argument('--scene',default='outputs/rgb_near_pipeline/assets/vita_bench_two52_O6_v5_source_hipcyl.xml')
    p.add_argument('--output',default='outputs/rgb_near_pipeline/reference_v23_seat_supported_full398.npz')
    p.add_argument('--backshift',type=float,default=.10)
    p.add_argument('--shift-start',type=float,default=7.)
    p.add_argument('--shift-end',type=float,default=8.5)
    p.add_argument('--unshift-start',type=float,default=12.)
    p.add_argument('--unshift-end',type=float,default=13.5)
    a=p.parse_args()
    import numpy as np
    import mujoco as mj
    from scipy.optimize import least_squares
    from scipy.spatial import ConvexHull
    z=np.load(a.root/a.input,allow_pickle=False);q=z['qpos'];names=list(z['dof_names']);m=mj.MjModel.from_xml_path(str(a.root/a.scene));d=mj.MjData(m)
    rj=mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,'p1_vita_demo_free_root');rq=int(m.jnt_qposadr[rj]);js=np.array([mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,'p1_'+str(n))for n in names]);qa=m.jnt_qposadr[js]
    ankles=[mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'p1_'+s+'_ankle_roll_link')for s in('left','right')]
    hips=[mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,'p1_collision_'+s+'_hip_roll_cylinder')for s in('l','r')];seat=mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,'bench_seat_proxy')
    def gn(g):return mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,g)or''
    rfeet=[g for g in range(m.ngeom)if gn(g).startswith('p1_collision_foot_r_')and m.geom_contype[g]]
    def footz():return min(d.geom_xpos[g,2]-m.geom_size[g,0]-m.geom_size[g,1]*abs(d.geom_xmat[g].reshape(3,3)[2,2])for g in rfeet)
    results=[];newq=q.copy();previous=None
    for f in range(len(q)):
        d.qpos[rq:rq+7]=q[f,0,:7];d.qpos[qa]=q[f,0,7:];mj.mj_forward(m,d);tp=d.xpos[ankles].copy();tR=d.xmat[ankles].reshape(2,3,3).copy();rootq=q[f,0,3:7].copy();rootp=q[f,0,:3].copy();original=q[f,0,7:19].copy();lim=m.jnt_range[js[:12]];target_right_floor_z=footz()
        def bind(x):
            d.qpos[rq:rq+3]=rootp+[0,x[0],x[1]];roll=np.array([np.cos(x[2]/2),np.sin(x[2]/2),0,0]);mj.mju_mulQuat(d.qpos[rq+3:rq+7],rootq,roll);d.qpos[qa[:12]]=x[3:];mj.mj_kinematics(m,d);mj.mj_comPos(m,d)
        def fun(x):
            bind(x);gaps=np.array([mj.mj_geomDistance(m,d,g,seat,1.,np.zeros(6))for g in hips]);rows=[]
            for k,b in enumerate(ankles):rows.extend([8*(d.xpos[b]-tp[k]),.25*(d.xmat[b].reshape(3,3)-tR[k]).ravel()])
            return np.concatenate(rows+[12*(gaps+.0008),np.array([30*(footz()-target_right_floor_z)]),.01*x[:3],.0001*(x[3:]-original)])
        u=float(np.clip((z['time_s'][f]-a.shift_start)/(a.shift_end-a.shift_start),0,1));v=float(np.clip((z['time_s'][f]-a.unshift_start)/(a.unshift_end-a.unshift_start),0,1));dy=a.backshift*u*u*(3-2*u)*(1-v*v*(3-2*v))
        lo=np.r_[dy,-.05,-.55,lim[:,0]+1e-9];hi=np.r_[dy+1e-7,.03,.55,lim[:,1]-1e-9]
        best=None
        seeds=[np.r_[dy,previous[:2]if previous is not None else [0.,0.],original]]
        if f==0:seeds.append(np.r_[dy,-.001,0.,original])
        for x0 in seeds:
            fit=least_squares(fun,np.clip(x0,lo+1e-8,hi-1e-8),bounds=(lo,hi),max_nfev=100,diff_step=1e-5,ftol=1e-9,xtol=1e-9,gtol=1e-9)
            if best is None or fit.cost<best.cost:best=fit
        previous=best.x[1:3].copy()
        bind(best.x);mj.mj_forward(m,d)
        base=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'p1_base_link');support=[]
        for c in d.contact:
            pair=gn(int(c.geom1))+gn(int(c.geom2))
            if'p1_'in pair and('bench_seat_proxy'in pair or'ground_proxy'in pair):support.append(c.pos[:2].copy())
        margin=None
        if len(support)>=3:
            hull=ConvexHull(np.array(support));margin=float(-np.max(hull.equations[:,:2]@d.subtree_com[base,:2]+hull.equations[:,2]))
        row={'frame':f,'time_s':float(z['time_s'][f]),'dy_dz_roll':best.x[:3].tolist(),'qpos_p1':np.r_[d.qpos[rq:rq+7],d.qpos[qa]].tolist(),'COM_world':d.subtree_com[base].tolist(),'COM_support_polygon_margin_m':margin,'foot_position_error_m':np.linalg.norm(d.xpos[ankles]-tp,axis=1).tolist(),'foot_rotation_matrix_norm_error':np.linalg.norm(d.xmat[ankles].reshape(2,3,3)-tR,axis=(1,2)).tolist(),'hip_seat_gap_m':[float(mj.mj_geomDistance(m,d,g,seat,1.,np.zeros(6)))for g in hips],'right_foot_floor_z_m':float(footz()),'cost':float(best.cost),'nfev':best.nfev}
        newq[f,0]=np.array(row['qpos_p1'])
        if f%40==0:print(json.dumps({k:v for k,v in row.items()if k!='qpos_p1'}),flush=True)
        results.append(row)
    # Preserve both previous world wrists while changing the trunk, independently
    # of seat/foot optimization. Source hands were invalid; failures are reported.
    arm_errors=[];jp=np.zeros((3,m.nv));jr=np.zeros_like(jp)
    for side,hpre in [('left','lh_'),('right','rh_')]:
        hand=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'p1_'+hpre+'hand_base_link')
        sel=np.array([k for k,s in enumerate(names)if str(s).startswith(side+'_')and any(v in str(s)for v in('shoulder','elbow','wrist'))]);jarm=js[sel];qarm=qa[sel];va=m.jnt_dofadr[jarm];limits=m.jnt_range[jarm];previous_delta=np.zeros(len(sel))
        for f in range(len(q)):
            d.qpos[rq:rq+7]=q[f,0,:7];d.qpos[qa]=q[f,0,7:];mj.mj_kinematics(m,d);tp=d.xpos[hand].copy();tR=d.xmat[hand].reshape(3,3).copy()
            d.qpos[rq:rq+7]=newq[f,0,:7];d.qpos[qa]=newq[f,0,7:];oldarm=q[f,0,7+sel]
            def afun(x):d.qpos[qarm]=x;mj.mj_kinematics(m,d);mj.mj_comPos(m,d);return np.r_[10*(d.xpos[hand]-tp),.3*(d.xmat[hand].reshape(3,3)-tR).T.ravel(),1e-6*(x-oldarm)]
            def ajac(x):
                afun(x);mj.mj_jacBody(m,d,jp,jr,hand);R=d.xmat[hand].reshape(3,3);rot=np.concatenate([np.cross(jr[:,va].T,R[:,k]).T for k in range(3)],axis=0);return np.vstack([10*jp[:,va],.3*rot,1e-6*np.eye(len(sel))])
            fit=least_squares(afun,np.clip(oldarm+previous_delta,limits[:,0]+1e-8,limits[:,1]-1e-8),jac=ajac,bounds=(limits[:,0]+1e-9,limits[:,1]-1e-9),max_nfev=70,ftol=1e-10,xtol=1e-10,gtol=1e-10)
            afun(fit.x);newq[f,0,7+sel]=fit.x;previous_delta=fit.x-oldarm
            arm_errors.append({'frame':f,'side':side,'position_error_m':float(np.linalg.norm(d.xpos[hand]-tp)),'rotation_matrix_error':float(np.linalg.norm(d.xmat[hand].reshape(3,3)-tR))})
    payload={k:z[k]for k in z.files};payload['qpos']=newq;payload['qpos_p1']=newq[:,0];payload['qpos_p2']=newq[:,1];payload['pre_seat_supported_reference_qpos']=q
    payload['seat_support_root_dy_dz_roll']=np.array([r['dy_dz_roll']for r in results]);payload['seat_support_backshift_ramp_s']=np.array([a.shift_start,a.shift_end,a.unshift_start,a.unshift_end]);payload['O6_grasp_geometric_quality_passed']=np.array(False)
    ankle=ankles[0];hand=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'p1_rh_hand_base_link');switch=int(z['shoe_release_frame']);shoe=[];wearonly=[]
    for f in range(len(q)):
        d.qpos[rq:rq+7]=newq[f,0,:7];d.qpos[qa]=newq[f,0,7:];mj.mj_kinematics(m,d);R=d.xmat[ankle].reshape(3,3);wp=np.zeros(7);wp[:3]=d.xpos[ankle]+R@z['shoe_wear_position_local_to_left_ankle'];mj.mju_mat2Quat(wp[3:],(R@z['shoe_wear_rotation_local_to_left_ankle']).ravel());wearonly.append(wp)
        if f<switch:shoe.append(wp)
        else:
            H=d.xmat[hand].reshape(3,3)
            if f==switch:
                WR=np.zeros(9);mj.mju_quat2Mat(WR,wp[3:]);gp=H.T@(wp[:3]-d.xpos[hand]);gR=H.T@WR.reshape(3,3)
            pose=np.zeros(7);pose[:3]=d.xpos[hand]+H@gp;mj.mju_mat2Quat(pose[3:],(H@gR).ravel());shoe.append(pose)
    payload['object_target_pose']=np.array(shoe);payload['contact_optimized_shoe_pose']=np.array(shoe);payload['shoe_wear_only_reference_pose']=np.array(wearonly);payload['shoe_hand_position_local_to_right_hand']=gp;payload['shoe_hand_rotation_local_to_right_hand']=gR
    payload['shoe_source_position_correction_m']=np.linalg.norm(payload['object_target_pose'][:,:3]-z['raw_source_object_pose_xyz_wxyz'][:,:3],axis=1)
    out=a.root/a.output;np.savez_compressed(out,**payload)
    report={'schema':'seated_lift_reference_v1','kinematic_only':True,'reference_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256((a.root/a.input).read_bytes()).hexdigest(),'results':results,'world_wrist_errors':arm_errors,'max_foot_position_error_m':max(max(r['foot_position_error_m'])for r in results),'max_wrist_position_error_m':max(r['position_error_m']for r in arm_errors)}
    out.with_suffix('.json').write_text(json.dumps(report,indent=2));out.with_suffix('.generator.py').write_bytes(Path(__file__).read_bytes());print(json.dumps({k:v for k,v in report.items()if k not in('results','world_wrist_errors')}),flush=True)


if __name__=='__main__':main()
