"""Ground the initial seated support feet/shoe using real collision surfaces.

Explicit kinematic adaptation: original source remains separate, no force or
pose injection in a dynamic simulation. This script prepares a short prefix.
"""
from pathlib import Path
import argparse
import hashlib
import json


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/data/KinoGen'))
    p.add_argument('--input',default='outputs/rgb_near_pipeline/reference_v17_source_hipcyl.npz')
    p.add_argument('--scene',default='outputs/rgb_near_pipeline/assets/vita_bench_two52_O6_v5_source_hipcyl.xml')
    p.add_argument('--output',default='outputs/rgb_near_pipeline/reference_v21_grounded_full398.npz')
    p.add_argument('--seat-lowering',type=float,default=.0015)
    p.add_argument('--floor-target',type=float,default=-.0002)
    a=p.parse_args()
    import numpy as np
    import mujoco as mj
    from scipy.optimize import least_squares
    z=np.load(a.root/a.input,allow_pickle=False);payload={k:z[k]for k in z.files};q=z['qpos'].copy();old=q.copy();N=len(q);names=[str(v)for v in z['dof_names']]
    switch=int(z['shoe_release_frame'])
    m=mj.MjModel.from_xml_path(str(a.root/a.scene));d=mj.MjData(m);maps=[]
    for ac in range(2):
        pre=f'p{ac+1}_';r=mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,pre+'vita_demo_free_root');js=np.array([mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,pre+s)for s in names]);maps.append((int(m.jnt_qposadr[r]),m.jnt_qposadr[js],js))
    sj=mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,'shoe_free');sq=int(m.jnt_qposadr[sj])
    def fk():mj.mj_kinematics(m,d);mj.mj_comPos(m,d)
    def reset(f,arr):
        for ac,(rq,qa,js)in enumerate(maps):d.qpos[rq:rq+7]=arr[f,ac,:7];d.qpos[qa]=arr[f,ac,7:]
        d.qpos[sq:sq+7]=z['object_target_pose'][f];fk()
    def gn(g):return mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,g)or''
    def minz(g):
        R=d.geom_xmat[g].reshape(3,3);size=m.geom_size[g];typ=m.geom_type[g]
        if typ==mj.mjtGeom.mjGEOM_BOX:return float(d.geom_xpos[g,2]-np.dot(np.abs(R[2]),size))
        if typ==mj.mjtGeom.mjGEOM_CAPSULE:return float(d.geom_xpos[g,2]-size[0]-size[1]*abs(R[2,2]))
        if typ==mj.mjtGeom.mjGEOM_SPHERE:return float(d.geom_xpos[g,2]-size[0])
        raise ValueError('Unsupported support collider '+gn(g))
    shoeg=[g for g in range(m.ngeom)if gn(g).startswith('shoe_')and m.geom_contype[g]]
    targets={};shifts=np.zeros((N,2));errors=[];wearposes=[];phase_support=np.zeros((N,2),dtype=bool)
    ankle=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'p1_left_ankle_roll_link')
    hand=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'p1_rh_hand_base_link')
    for f in range(N):
        reset(f,old);R=d.xmat[ankle].reshape(3,3);pose=np.zeros(7);pose[:3]=d.xpos[ankle]+R@z['shoe_wear_position_local_to_left_ankle'];mj.mju_mat2Quat(pose[3:],(R@z['shoe_wear_rotation_local_to_left_ankle']).ravel());wearposes.append(pose)
    wearposes=np.array(wearposes)
    # p2 was already supported and is deliberately unchanged.
    for side in ('left','right'):
        bid=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'p1_'+side+'_ankle_roll_link');footg=[g for g in range(m.ngeom)if gn(g).startswith('p1_collision_foot_'+side[0]+'_')and m.geom_contype[g]]
        pp=[];RR=[]
        for f in range(N):
            reset(f,old)
            if side=='left'and f<switch:d.qpos[sq:sq+7]=wearposes[f];fk()
            surface=shoeg if side=='left'and f<switch else footg
            gap=min(minz(g)for g in surface)
            # A continuous release envelope preserves the lifted source leg.
            # Ground small near-floor reconstruction offsets; fade adaptation
            # over 35-65 mm instead of snapping a raised foot to the floor.
            blend=float(np.clip((.065-gap)/.030,0,1));blend=blend*blend*(3-2*blend)
            dz=(a.floor_target-gap)*blend if gap>=a.floor_target else a.floor_target-gap
            phase_support[f,0 if side=='left'else 1]=gap<=.035
            pp.append(d.xpos[bid].copy()+[0,0,dz]);RR.append(d.xmat[bid].reshape(3,3).copy());shifts[f,0 if side=='left'else 1]=dz
        targets[side]=(bid,np.array(pp),np.array(RR))
    # Slight extra root settling engages both existing source-cylinder hips.
    q[:,0,2]-=a.seat_lowering
    jp=np.zeros((3,m.nv));jr=np.zeros_like(jp)
    for side in ('left','right'):
        rq,qa,js=maps[0];sel=np.array([k for k,s in enumerate(names)if s.startswith(side+'_')and any(x in s for x in('hip_','knee_','ankle_'))]);jsel=js[sel];va=m.jnt_dofadr[jsel];qa_sel=qa[sel];lim=m.jnt_range[jsel];bid,tp,tR=targets[side]
        for f in range(N):
            reset(f,q)
            def fun(x):d.qpos[qa_sel]=x;fk();return np.r_[10*(d.xpos[bid]-tp[f]),.3*(d.xmat[bid].reshape(3,3)-tR[f]).T.ravel()]
            def jac(x):
                d.qpos[qa_sel]=x;fk();mj.mj_jacBody(m,d,jp,jr,bid);R=d.xmat[bid].reshape(3,3);rot=np.concatenate([np.cross(jr[:,va].T,R[:,k]).T for k in range(3)],axis=0);return np.vstack([10*jp[:,va],.3*rot])
            fit=least_squares(fun,q[f,0,7+sel],jac=jac,bounds=(lim[:,0]+1e-9,lim[:,1]-1e-9),max_nfev=45,ftol=1e-12,xtol=1e-12,gtol=1e-12);fun(fit.x);q[f,0,7+sel]=fit.x
            errors.append(float(np.linalg.norm(d.xpos[bid]-tp[f])))
    payload['qpos']=q;payload['qpos_p1']=q[:,0];payload['qpos_p2']=q[:,1]
    payload['pre_ground_support_reference_qpos']=old;payload['support_foot_vertical_adaptation_m']=shifts
    extra_lowering=np.zeros((N,2));extra_lowering[:,0]=a.seat_lowering
    payload['support_seat_additional_root_lowering_m']=extra_lowering
    payload['support_parent_reference_sha256']=np.array(sha(a.root/a.input));payload['O6_grasp_geometric_quality_passed']=np.array(False)
    payload['O6_grasp_distance_audit_superseded']=np.array('MuJoCo 3.7.0 native GJK zero-distance defect; independent convex audit disproves prior fingertip contact. Soft-assisted grip only.')
    newshoe=[];newwear=[];newgrip_p=None;newgrip_R=None
    for f in range(N):
        reset(f,q);R=d.xmat[ankle].reshape(3,3);wear=np.zeros(7);wear[:3]=d.xpos[ankle]+R@z['shoe_wear_position_local_to_left_ankle'];mj.mju_mat2Quat(wear[3:],(R@z['shoe_wear_rotation_local_to_left_ankle']).ravel());newwear.append(wear)
        if f<switch:newshoe.append(wear)
        else:
            R=d.xmat[hand].reshape(3,3)
            if f==switch:
                WR=np.zeros(9);mj.mju_quat2Mat(WR,wear[3:]);newgrip_p=R.T@(wear[:3]-d.xpos[hand]);newgrip_R=R.T@WR.reshape(3,3)
            held=np.zeros(7);held[:3]=d.xpos[hand]+R@newgrip_p;mj.mju_mat2Quat(held[3:],(R@newgrip_R).ravel());newshoe.append(held)
    payload['object_target_pose']=np.array(newshoe);payload['contact_optimized_shoe_pose']=np.array(newshoe);payload['shoe_wear_only_reference_pose']=np.array(newwear)
    payload['support_phase_planted_inferred']=phase_support
    payload['shoe_hand_position_local_to_right_hand']=newgrip_p;payload['shoe_hand_rotation_local_to_right_hand']=newgrip_R
    payload['support_grip_position_recalibration_m']=newgrip_p-z['shoe_hand_position_local_to_right_hand']
    payload['shoe_source_position_correction_m']=np.linalg.norm(payload['object_target_pose'][:,:3]-payload['raw_source_object_pose_xyz_wxyz'][:,:3],axis=-1)
    floorchecks=[];contacts=[]
    for f in range(N):
        reset(f,q);d.qpos[sq:sq+7]=payload['object_target_pose'][f];mj.mj_forward(m,d)
        right=[g for g in range(m.ngeom)if gn(g).startswith('p1_collision_foot_r_')and m.geom_contype[g]]
        left=[g for g in range(m.ngeom)if gn(g).startswith('p1_collision_foot_l_')and m.geom_contype[g]]
        floorchecks.append([min(minz(g)for g in(shoeg if f<switch else left)),min(minz(g)for g in right)])
        if f==0:contacts=[{'geom1':gn(int(c.geom1)),'geom2':gn(int(c.geom2)),'distance_m':float(c.dist),'position_world':c.pos.tolist()}for c in d.contact if'p1_'in gn(int(c.geom1))+gn(int(c.geom2))or'shoe_'in gn(int(c.geom1))+gn(int(c.geom2))]
    payload['support_collision_surface_floor_z_m']=np.array(floorchecks)
    out=a.root/a.output;np.savez_compressed(out,**payload)
    continuity=float(np.linalg.norm(payload['object_target_pose'][switch,:3]-payload['shoe_wear_only_reference_pose'][switch,:3]))
    report={'schema':'rgb_near_full_ground_support_v1','kinematic_only':True,'frames':N,'input_sha256':sha(a.root/a.input),'output_sha256':sha(out),'scene_sha256':sha(a.root/a.scene),'seat_extra_lowering_m':a.seat_lowering,'foot_vertical_shift_range_m':[shifts.min(axis=0).tolist(),shifts.max(axis=0).tolist()],'max_foot_IK_position_error_m':max(errors),'all_support_surface_z_range_m':[np.min(floorchecks,axis=0).tolist(),np.max(floorchecks,axis=0).tolist()],'initial_contacts':contacts,'handover_position_mismatch_m':continuity,'grasp_geometry_legacy_result_invalidated':True,'support_phase_inferred':'near-floor <=35mm planted; smooth release 35-65mm; raised source leg retained; post-release left bare foot supports independently'}
    out.with_suffix('.json').write_text(json.dumps(report,indent=2));out.with_suffix('.generator.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(report),flush=True)


if __name__=='__main__':main()
