"""Keyframe-only seated lift feasibility probe; no dynamics certification."""
from pathlib import Path
import argparse
import json
import hashlib


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('/data/KinoGen'))
    p.add_argument('--input',default='outputs/rgb_near_pipeline/reference_v21_grounded_full398.npz')
    p.add_argument('--scene',default='outputs/rgb_near_pipeline/assets/vita_bench_two52_O6_v5_source_hipcyl.xml')
    p.add_argument('--output',default='outputs/rgb_near_pipeline/reference_v25_planted_com_full398.npz')
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
    from scipy.ndimage import gaussian_filter1d
    from scipy.spatial.transform import Rotation,Slerp
    z=np.load(a.root/a.input,allow_pickle=False);q=z['qpos'];names=list(z['dof_names']);m=mj.MjModel.from_xml_path(str(a.root/a.scene));d=mj.MjData(m)
    rj=mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,'p1_vita_demo_free_root');rq=int(m.jnt_qposadr[rj]);js=np.array([mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,'p1_'+str(n))for n in names]);qa=m.jnt_qposadr[js]
    ankles=[mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'p1_'+s+'_ankle_roll_link')for s in('left','right')]
    hips=[mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,'p1_collision_'+s+'_hip_roll_cylinder')for s in('l','r')];seat=mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,'bench_seat_proxy')
    def gn(g):return mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,g)or''
    rfeet=[g for g in range(m.ngeom)if gn(g).startswith('p1_collision_foot_r_')and m.geom_contype[g]]
    lfeet=[g for g in range(m.ngeom)if gn(g).startswith('p1_collision_foot_l_')and m.geom_contype[g]]
    def footz():return min(d.geom_xpos[g,2]-m.geom_size[g,0]-m.geom_size[g,1]*abs(d.geom_xmat[g].reshape(3,3)[2,2])for g in rfeet)
    def leftfootz():return min(d.geom_xpos[g,2]-m.geom_size[g,0]-m.geom_size[g,1]*abs(d.geom_xmat[g].reshape(3,3)[2,2])for g in lfeet)
    allp=[];allR=[];allfloor=[]
    for f in range(len(q)):
        d.qpos[rq:rq+7]=q[f,0,:7];d.qpos[qa]=q[f,0,7:];mj.mj_forward(m,d);allp.append(d.xpos[ankles].copy());allR.append(d.xmat[ankles].reshape(2,3,3).copy());allfloor.append(footz())
    allp=np.array(allp);allR=np.array(allR);allfloor=np.array(allfloor);plannedp=allp.copy();plannedR=allR.copy();times=z['time_s']
    anchorp=np.median(allp[(times>=8.5)&(times<=11.5),1],axis=0);anchorR=allR[210,1].copy()
    def smooth(x):x=float(np.clip(x,0,1));return x*x*(3-2*x)
    leftsmooth=gaussian_filter1d(allp[:,0],2.,axis=0,mode='nearest')
    for f in range(len(q)):
        tt=float(times[f]);plant=smooth((tt-7.5)/.8)*(1-smooth((tt-12.)/.7));plannedp[f,1]=(1-plant)*allp[f,1]+plant*anchorp
        plannedR[f,1]=Slerp([0,1],Rotation.from_matrix(np.array([allR[f,1],anchorR])))(plant).as_matrix()
        blend=smooth((tt-7.8)/.45)*(1-smooth((tt-9.5)/.4));plannedp[f,0]=(1-blend)*allp[f,0]+blend*leftsmooth[f]
        d.qpos[rq:rq+7]=q[f,0,:7];d.qpos[qa]=q[f,0,7:];mj.mj_forward(m,d)
        b=ankles[1];geoms=rfeet
        # Explicitly compute the rigid right-foot support extent after its
        # desired rotation; keep its actual capsule surface on the ground.
        relative=[]
        for g in geoms:
            relative.append((d.xmat[b].reshape(3,3).T@(d.geom_xpos[g]-d.xpos[b]),d.xmat[b].reshape(3,3).T@d.geom_xmat[g].reshape(3,3),m.geom_size[g].copy()))
        bottom=min((plannedp[f,1]+plannedR[f,1]@pos)[2]-size[0]-size[1]*abs((plannedR[f,1]@R)[2,2])for pos,R,size in relative)
        if plant>1e-8:plannedp[f,1,2]+=(allfloor[f]-bottom)*plant
    results=[];newq=q.copy();previous=None
    for f in range(len(q)):
        d.qpos[rq:rq+7]=q[f,0,:7];d.qpos[qa]=q[f,0,7:];mj.mj_forward(m,d);tp=plannedp[f].copy();tR=plannedR[f].copy();rootq=q[f,0,3:7].copy();rootp=q[f,0,:3].copy();original=q[f,0,7:19].copy();lim=m.jnt_range[js[:12]];target_right_floor_z=allfloor[f];target_left_floor_z=leftfootz()+plannedp[f,0,2]-allp[f,0,2]
        def bind(x):
            d.qpos[rq:rq+3]=rootp+[0,x[0],x[1]];roll=np.array([np.cos(x[2]/2),np.sin(x[2]/2),0,0]);mj.mju_mulQuat(d.qpos[rq+3:rq+7],rootq,roll);d.qpos[qa[:12]]=x[3:];mj.mj_kinematics(m,d);mj.mj_comPos(m,d)
        def fun(x):
            bind(x);gaps=np.array([mj.mj_geomDistance(m,d,g,seat,1.,np.zeros(6))for g in hips]);rows=[]
            for k,b in enumerate(ankles):rows.extend([8*(d.xpos[b]-tp[k]),.25*(d.xmat[b].reshape(3,3)-tR[k]).ravel()])
            return np.concatenate(rows+[40*(gaps+.0008),np.array([40*(footz()-target_right_floor_z),40*(leftfootz()-target_left_floor_z)]),.01*x[:3],.0001*(x[3:]-original)])
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
    # Finite, disclosed torso/non-task-arm adaptation to leave a positive
    # projected support margin, while preserving the right task wrist.
    comrows=[];base=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'p1_base_link');taskhand=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'p1_rh_hand_base_link')
    shoejoint=mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,'shoe_free');sq=int(m.jnt_qposadr[shoejoint]);shoebody=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'shoe')
    handg=[g for g in range(m.ngeom)if m.geom_contype[g]and(mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[g]))or'').startswith(('p1_lh_','p1_rh_','p1_left_wrist','p1_right_wrist'))];benchg=[g for g in range(m.ngeom)if m.geom_contype[g]and gn(g).startswith('bench_')];shoeg=[g for g in range(m.ngeom)if m.geom_contype[g]and gn(g).startswith('shoe_')]
    csel=np.array([k for k,s in enumerate(names)if str(s).startswith(('waist_','left_shoulder','left_elbow','left_wrist','right_shoulder','right_elbow','right_wrist'))]);cj=js[csel];cqa=qa[csel];crange=m.jnt_range[cj];previous_delta=np.zeros(len(csel))
    for f in range(len(q)):
        if times[f]<7.:continue
        d.qpos[rq:rq+7]=newq[f,0,:7];d.qpos[qa]=newq[f,0,7:];mj.mj_forward(m,d);hp=d.xpos[taskhand].copy();hR=d.xmat[taskhand].reshape(3,3).copy();support=[]
        for c in d.contact:
            pair=gn(int(c.geom1))+gn(int(c.geom2))
            if'p1_'in pair and('bench_seat_proxy'in pair or'ground_proxy'in pair):support.append(c.pos[:2].copy())
        if len(support)<3:continue
        try:hull=ConvexHull(np.array(support))
        except Exception:continue
        A=hull.equations[:,:2];B=hull.equations[:,2];orig=newq[f,0,7+csel].copy();before=float(-np.max(A@d.subtree_com[base,:2]+B))
        maximum=np.array([.35 if str(names[k]).startswith('waist_')else .7 for k in csel]);lower=np.maximum(crange[:,0]+1e-6,orig-maximum);upper=np.minimum(crange[:,1]-1e-6,orig+maximum)
        # Additional adaptation changes smoothly even when the source itself
        # moves quickly. This bound concerns the added motion, not hidden time.
        if f>0:
            step=3.*float(times[f]-times[f-1]);lower=np.maximum(lower,orig+previous_delta-step);upper=np.minimum(upper,orig+previous_delta+step)
        def cf(x):
            d.qpos[cqa]=x;mj.mj_kinematics(m,d);mj.mj_comPos(m,d)
            violation=np.maximum(A@d.subtree_com[base,:2]+B+.008,0)
            collision_pairs=[tuple(sorted((g,b)))for g in handg for b in benchg]
            if f<int(z['shoe_release_frame']):
                R=d.xmat[ankles[0]].reshape(3,3);d.qpos[sq:sq+3]=d.xpos[ankles[0]]+R@z['shoe_wear_position_local_to_left_ankle'];mj.mju_mat2Quat(d.qpos[sq+3:sq+7],(R@z['shoe_wear_rotation_local_to_left_ankle']).ravel());mj.mj_kinematics(m,d)
                collision_pairs.extend(tuple(sorted((g,b)))for g in handg for b in shoeg)
            pairmap={pair:i for i,pair in enumerate(collision_pairs)};gaps=np.full(len(collision_pairs),.3);mj.mj_collision(m,d)
            for c in d.contact:
                key=tuple(sorted((int(c.geom1),int(c.geom2))))
                if key in pairmap:gaps[pairmap[key]]=min(gaps[pairmap[key]],float(c.dist))
            return np.r_[12*violation,10*(d.xpos[taskhand]-hp),.25*(d.xmat[taskhand].reshape(3,3)-hR).ravel(),.003*(x-orig),60*np.minimum(gaps+.0005,0)]
        if np.any(lower>=upper):lower=np.maximum(crange[:,0]+1e-6,orig-maximum);upper=np.minimum(crange[:,1]-1e-6,orig+maximum)
        fit=least_squares(cf,np.clip(orig+previous_delta,lower+1e-8,upper-1e-8),bounds=(lower,upper),max_nfev=100,ftol=1e-8,xtol=1e-8,gtol=1e-8);cf(fit.x);newq[f,0,7+csel]=fit.x;previous_delta=fit.x-orig
        comrows.append({'frame':f,'margin_before_m':before,'margin_after_m':float(-np.max(A@d.subtree_com[base,:2]+B)),'max_joint_adjustment_rad':float(abs(previous_delta).max()),'task_wrist_error_m':float(np.linalg.norm(d.xpos[taskhand]-hp))})
    payload={k:z[k]for k in z.files};payload['qpos']=newq;payload['qpos_p1']=newq[:,0];payload['qpos_p2']=newq[:,1];payload['pre_seat_supported_reference_qpos']=q
    payload['support_foot_pre_plant_position_world_m']=allp;payload['support_foot_planned_position_world_m']=plannedp;payload['support_foot_planned_rotation_world']=plannedR;payload['support_right_foot_plant_anchor_world_m']=anchorp;payload['support_right_foot_plant_ramp_s']=np.array([7.5,8.3,12.,12.7]);payload['support_left_lift_filter_sigma_frames']=np.array(2.)
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
    report={'schema':'seated_lift_reference_v1','kinematic_only':True,'reference_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256((a.root/a.input).read_bytes()).hexdigest(),'results':results,'world_wrist_errors':arm_errors,'COM_refinement':comrows,'max_foot_position_error_m':max(max(r['foot_position_error_m'])for r in results),'max_wrist_position_error_m':max(r['position_error_m']for r in arm_errors),'right_foot_max_plant_adaptation_m':float(np.linalg.norm(plannedp[:,1]-allp[:,1],axis=1).max()),'left_lift_peak_source_speed_m_s':float(np.linalg.norm(np.gradient(allp[:,0],times,axis=0)[(times>=8.3)&(times<=9.3)],axis=1).max()),'left_lift_peak_planned_speed_m_s':float(np.linalg.norm(np.gradient(plannedp[:,0],times,axis=0)[(times>=8.3)&(times<=9.3)],axis=1).max())}
    out.with_suffix('.json').write_text(json.dumps(report,indent=2));out.with_suffix('.generator.py').write_bytes(Path(__file__).read_bytes());print(json.dumps({k:v for k,v in report.items()if k not in('results','world_wrist_errors','COM_refinement')}),flush=True)


if __name__=='__main__':main()
