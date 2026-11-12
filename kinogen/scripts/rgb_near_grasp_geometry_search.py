"""Search inferred wrist targets while retaining the continuous frame260 grip."""
from pathlib import Path
import argparse
import hashlib
import json


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--switch-frame',type=int,default=260)
    p.add_argument('--transition-seconds',type=float,default=.65)
    p.add_argument('--staged',action='store_true',help='Extract along foot axis, lift above knees, then move outside')
    p.add_argument('--first-feasible',action='store_true')
    a=p.parse_args()
    import numpy as np
    import mujoco as mj
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation,Slerp
    m=mj.MjModel.from_xml_path(str(a.model));d=mj.MjData(m)
    z=np.load(a.reference,allow_pickle=False);q=z['qpos'];times=z['time_s'];count=len(q)
    hand=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'p1_rh_hand_base_link')
    shoe=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'shoe')
    sj=mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,'shoe_free');sq=int(m.jnt_qposadr[sj])
    right_joints=[j for j in range(m.njnt)if(mj.mj_id2name(m,mj.mjtObj.mjOBJ_JOINT,j)or'').startswith(('p1_right_shoulder_','p1_right_elbow_','p1_right_wrist_'))]
    qa=m.jnt_qposadr[right_joints];va=m.jnt_dofadr[right_joints]
    limits=m.jnt_range[right_joints]
    hand_geoms=[g for g in range(m.ngeom)if m.geom_contype[g] and(mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[g]))or'').startswith('p1_rh_')]
    shoe_geoms=[g for g in range(m.ngeom)if m.geom_contype[g] and m.geom_bodyid[g]==shoe]
    body_names=[mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,i)or'world'for i in range(m.nbody)]
    geom_names=[mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,i)or str(i)for i in range(m.ngeom)]
    def reset(frame):
        d.qpos[:59]=q[frame,0];d.qpos[59:118]=q[frame,1]
        d.qpos[sq:sq+7]=z['object_target_pose'][frame]
        mj.mj_fwdPosition(m,d)
    shoulder=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'p1_right_shoulder_pitch_link')
    foot=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'p1_left_ankle_roll_link')
    handP=[];handR=[];shoulderP=[]
    for frame in range(count):
        reset(frame);handP.append(d.xpos[hand].copy());handR.append(d.xmat[hand].reshape(3,3).copy());shoulderP.append(d.xpos[shoulder].copy())
    handP=np.asarray(handP);handR=np.asarray(handR)
    reset(a.switch_frame)
    gripR=handR[a.switch_frame].T@d.xmat[shoe].reshape(3,3)
    gript=handR[a.switch_frame].T@(d.xpos[shoe]-handP[a.switch_frame])
    sole=mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,'shoe_sole_proxy')
    shellQ=np.empty(9);mj.mju_quat2Mat(shellQ,m.geom_quat[sole]);shellQ=shellQ.reshape(3,3)
    shoeR0=d.xmat[shoe].reshape(3,3).copy()
    foot_forward=d.xmat[foot].reshape(3,3)[:,0].copy()
    safe=[('freeze_capture',shoeR0)]
    # Toe down, sole facing forward. Fixed grip is retained; the wrist rotates.
    canonical=np.column_stack([[0.,0.,-1.],[1.,0.,0.],[0.,-1.,0.]])
    for yaw in [-35.,0.,35.]:
        rz=Rotation.from_euler('z',yaw,degrees=True).as_matrix()
        safe.append((f'toe_down_yaw{yaw:g}',rz@canonical@shellQ.T))
    canonical=np.column_stack([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    safe.append(('toe_forward_upright',canonical@shellQ.T))
    canonical=np.column_stack([[0.,-1.,0.],[0.,0.,-1.],[1.,0.,0.]])
    safe.append(('toe_forward_sideways',canonical@shellQ.T))
    frames=np.asarray(sorted(set([a.switch_frame,266,274,280,285,295,305,315,321,330,345,360,380,count-1])))
    frames=frames[(frames>=a.switch_frame)&(frames<count)]
    blend=np.clip((times-times[a.switch_frame])/a.transition_seconds,0,1)
    blend=blend*blend*(3-2*blend)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    scores=[];best=None
    jp=np.empty((3,m.nv));jr=np.empty((3,m.nv))
    def solve(frame,targetP,targetR):
        reset(frame);seed=np.clip(d.qpos[qa].copy(),limits[:,0]+1e-6,limits[:,1]-1e-6);cache={}
        def calc(x):
            if 'x'in cache and np.array_equal(x,cache['x']):return cache['r'],cache['J']
            d.qpos[qa]=x;mj.mj_fwdPosition(m,d)
            pe=d.xpos[hand]-targetP
            oe=Rotation.from_matrix(d.xmat[hand].reshape(3,3)@targetR.T).as_rotvec()
            mj.mj_jacBody(m,d,jp,jr,hand)
            r=np.r_[10*pe,oe,.01*(x-seed)]
            J=np.vstack([10*jp[:,va],jr[:,va],.01*np.eye(len(qa))])
            cache.update(x=x.copy(),r=r,J=J);return r,J
        fit=least_squares(lambda x:calc(x)[0],seed,jac=lambda x:calc(x)[1],bounds=(limits[:,0]+1e-6,limits[:,1]-1e-6),
                          max_nfev=35,ftol=1e-5,xtol=1e-5,gtol=1e-5)
        calc(fit.x)
        realP=d.xpos[hand].copy();realR=d.xmat[hand].reshape(3,3).copy()
        shoeP=realP+realR@gript;shoeR=realR@gripR
        quat=np.empty(4);mj.mju_mat2Quat(quat,shoeR.ravel())
        d.qpos[sq:sq+7]=np.r_[shoeP,quat];mj.mj_fwdPosition(m,d)
        collision=[];otherpen=0.;handpen=0.;benchpen=0.
        for ci in range(d.ncon):
            c=d.contact[ci];b1=int(m.geom_bodyid[c.geom1]);b2=int(m.geom_bodyid[c.geom2]);pen=max(-float(c.dist),0.)
            if shoe in(b1,b2):
                other=b2 if b1==shoe else b1
                is_hand=body_names[other].startswith('p1_rh_')
                if is_hand:handpen=max(handpen,pen)
                else:otherpen=max(otherpen,pen)
                if pen>.001:collision.append([geom_names[int(c.geom1)],geom_names[int(c.geom2)],pen])
            bn=[body_names[b1],body_names[b2]];gn=[geom_names[int(c.geom1)],geom_names[int(c.geom2)]]
            if any(x.startswith(('p1_right_','p1_rh_'))for x in bn)and any(x.startswith('bench_')for x in gn):benchpen=max(benchpen,pen)
        gap=min(float(mj.mj_geomDistance(m,d,g,s,.25,np.zeros(6)))for g in hand_geoms for s in shoe_geoms)
        return {'frame':int(frame),'other_shoe_penetration_m':otherpen,'hand_shoe_penetration_m':handpen,
                'right_arm_bench_penetration_m':benchpen,'hand_shoe_distance_m':gap,
                'hand_position_ik_error_m':float(np.linalg.norm(realP-targetP)),
                'hand_orientation_ik_error_rad':float(np.linalg.norm(Rotation.from_matrix(realR@targetR.T).as_rotvec())),
                'collision_pairs':collision}
    stop=False
    if a.staged:safe=[row for row in safe if row[0]in('freeze_capture','toe_forward_upright','toe_down_yaw0')]
    for label,RshoeSafe in safe:
        safeHand=RshoeSafe@gripR.T
        sl=Slerp([0,1],Rotation.from_matrix(np.stack([handR[a.switch_frame],safeHand])))
        rotation_weight=blend
        if a.staged:
            rotation_weight=np.clip((times-times[a.switch_frame]-.9)/.65,0,1)
            rotation_weight=rotation_weight*rotation_weight*(3-2*rotation_weight)
        rotations=sl(rotation_weight).as_matrix()
        rotations[:a.switch_frame]=handR[:a.switch_frame]
        cases=[]
        if a.staged:
            elapsed=times-times[a.switch_frame]
            def smooth(x):
                x=np.clip(x,0,1);return x*x*(3-2*x)
            extract_w=smooth(elapsed/.30);lift_w=smooth((elapsed-.30)/.60);outside_w=smooth((elapsed-.90)/.65)
            for extract in [.14,.08]:
                for height in [.95,1.02,.88]:
                    for lateral in [.18,.28,.08]:
                        positions=np.repeat(handP[a.switch_frame][None],count,axis=0)
                        positions+=extract_w[:,None]*extract*foot_forward
                        positions[:,2]+=lift_w*(height-positions[:,2])
                        outside=np.asarray([float(np.median(np.asarray(shoulderP)[a.switch_frame:,0]))-lateral,-.55,height])
                        positions=(1-outside_w[:,None])*positions+outside_w[:,None]*outside
                        positions[:a.switch_frame]=handP[:a.switch_frame]
                        cases.append(({'extract_along_initial_foot_axis_m':extract,'hand_height_world_m':height,'lateral_from_right_shoulder_m':lateral,'outside_hand_target_world_m':outside.tolist()},positions))
        else:
            for forward in [.04,.10,.16,.22]:
                for up in [0.,.04,.08]:
                    cases.append(({'forward_shift_m':forward,'up_shift_m':up},handP+blend[:,None]*np.asarray([0.,-forward,up])))
        for parameters,positions in cases:
                rows=[solve(int(f),positions[f],rotations[f])for f in frames]
                worst={k:max(r[k]for r in rows)for k in ['other_shoe_penetration_m','hand_shoe_penetration_m','right_arm_bench_penetration_m','hand_position_ik_error_m','hand_orientation_ik_error_rad']}
                worst['max_hand_shoe_gap_m']=max(max(r['hand_shoe_distance_m'],0.)for r in rows)
                feasible=worst['other_shoe_penetration_m']<=.004 and worst['hand_shoe_penetration_m']<=.004 and worst['right_arm_bench_penetration_m']<=.01 and worst['hand_position_ik_error_m']<=.025 and worst['hand_orientation_ik_error_rad']<=.2 and worst['max_hand_shoe_gap_m']<=.01
                adjustment=float(np.max(np.linalg.norm(positions-handP,axis=1)))
                score=100*worst['other_shoe_penetration_m']+50*worst['hand_shoe_penetration_m']+20*worst['right_arm_bench_penetration_m']+10*worst['hand_position_ik_error_m']+worst['hand_orientation_ik_error_rad']+.2*adjustment
                record={'orientation':label,**parameters,'max_hand_target_adjustment_m':adjustment,'feasible':bool(feasible),'score':score,'worst':worst,'frames':rows}
                scores.append(record)
                if best is None or (not best['record']['feasible']and feasible)or(feasible==best['record']['feasible']and score<best['record']['score']):
                    best={'record':record,'position':positions.copy(),'rotation':rotations.copy(),'safe_shoe_rotation_world':RshoeSafe.copy()}
                print(json.dumps({k:v for k,v in record.items()if k!='frames'}),flush=True)
                if feasible and a.first_feasible:stop=True;break
        if stop:break
    selected=best['record'];target=a.output_dir/'right_hand_grasp_targets.npz'
    np.savez_compressed(target,right_hand_base_position_world_m=best['position'],right_hand_base_rotation_world=best['rotation'],
        inferred_grasp_rel_position=gript,inferred_grasp_rel_rotation=gripR,time_s=times,
        switch_frame=np.asarray(a.switch_frame),transition_weight=rotation_weight,
        source_hand_base_position_world_m=handP,source_hand_base_rotation_world=handR,
        selected_safe_shoe_rotation_world=best['safe_shoe_rotation_world'],
        hand_position_adjustment_m=best['position']-handP,
        synthetic_hand_contact_adaptation=np.asarray(True))
    report={'schema':'rgb_near.inferred_grasp_target_search.v1','model':str(a.model),'model_sha256':sha(a.model),
            'reference':str(a.reference),'reference_sha256':sha(a.reference),'selected':selected,
            'switch_frame':a.switch_frame,'transition_seconds':a.transition_seconds,'candidate_count':len(scores),
            'staged_extraction_lift_outside':a.staged,'staged_durations_s':[.30,.60,.65]if a.staged else None,
            'sample_frames':frames.tolist(),'targets':str(target),'targets_sha256':sha(target),
            'grip_relative_pose_kept_from_capture':True,'initial_grasp_no_teleport':True,
            'note':'Candidate-only IK/contact geometry checks; full398 temporal/collision optimization and physical simulation remain required.',
            'candidates':scores}
    (a.output_dir/'search_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'selected':selected,'targets':str(target)}),flush=True)


if __name__=='__main__':main()
