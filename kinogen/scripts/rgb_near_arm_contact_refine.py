"""Refine arm keyframes against bench/worn-shoe geometry, then repair failures.

Legs and root trajectories remain the input reference. Hand observations in
the source are invalid, so every clearance or grip adjustment is an inferred
task constraint and is recorded independently from raw source arrays.
"""
from pathlib import Path
import argparse
import hashlib
import json


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/data/KinoGen'))
    p.add_argument('--input',default='outputs/rgb_near_pipeline/reference_v6_release260.npz')
    p.add_argument('--output',default='outputs/rgb_near_pipeline/reference_v7_armclearance.npz')
    p.add_argument('--scene',default='outputs/rgb_near_pipeline/assets/vita_bench_two52_wear_v2.xml')
    p.add_argument('--targets',type=Path)
    p.add_argument('--key-stride',type=int,default=10)
    p.add_argument('--bench-clearance',type=float,default=.012)
    p.add_argument('--shoe-clearance',type=float,default=.001)
    p.add_argument('--max-nfev',type=int,default=45)
    p.add_argument('--joint-speed-limit',type=float,default=6.)
    p.add_argument('--orientation-weight',type=float,default=.25)
    p.add_argument('--held-shoe-clearance',type=float,default=.015)
    p.add_argument('--only-task-arm',action='store_true')
    p.add_argument('--unmodified-ik-reference',type=Path)
    p.add_argument('--actor-side',action='append',choices=['p1_left','p1_right','p2_left','p2_right'])
    p.add_argument('--left-extra-lift',type=float,default=.015)
    p.add_argument('--p2right-extra-lift',type=float,default=.020)
    p.add_argument('--pregrasp-only',action='store_true')
    a=p.parse_args()
    generator_bytes=Path(__file__).read_bytes();generator_sha256=hashlib.sha256(generator_bytes).hexdigest()
    import numpy as np
    import mujoco as mj
    from scipy.optimize import least_squares
    root=a.root;inp=root/a.input;out=root/a.output;scene=root/a.scene
    z=np.load(inp,allow_pickle=False);payload={k:z[k]for k in z.files}
    q=np.asarray(z['qpos']).copy();initial_q=q.copy();t=z['time_s'];names=[str(x)for x in z['dof_names']]
    model=mj.MjModel.from_xml_path(str(scene));d=mj.MjData(model)
    targets=np.load(a.targets if a.targets.is_absolute()else root/a.targets,allow_pickle=False) if a.targets else None
    switch=int(z['shoe_release_frame']);sj=mj.mj_name2id(model,mj.mjtObj.mjOBJ_JOINT,'shoe_free');sq=int(model.jnt_qposadr[sj])
    maps=[]
    for pre in('p1_','p2_'):
        rj=mj.mj_name2id(model,mj.mjtObj.mjOBJ_JOINT,pre+'vita_demo_free_root')
        ids=[mj.mj_name2id(model,mj.mjtObj.mjOBJ_JOINT,pre+n)for n in names]
        if rj<0 or min(ids)<0:raise ValueError('Reference joint names do not match scene actor '+pre)
        maps.append((int(model.jnt_qposadr[rj]),np.asarray([model.jnt_qposadr[j]for j in ids]),ids))
    def gn(g):return mj.mj_id2name(model,mj.mjtObj.mjOBJ_GEOM,int(g))or''
    def bn(b):return mj.mj_id2name(model,mj.mjtObj.mjOBJ_BODY,int(b))or'world'
    bench=[g for g in range(model.ngeom)if gn(g).startswith('bench_')and model.geom_contype[g]]
    shoe=[g for g in range(model.ngeom)if gn(g).startswith('shoe_')and model.geom_contype[g]]
    holder=mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,'p1_rh_hand_base_link')
    grip_p=z['shoe_hand_position_local_to_right_hand'];grip_R=z['shoe_hand_rotation_local_to_right_hand']
    obstacles=[g for g in range(model.ngeom)if model.geom_contype[g]and g not in shoe and
               not bn(model.geom_bodyid[g]).startswith('p1_rh_')]
    jp=np.zeros((3,model.nv));jr=np.zeros_like(jp)
    def fk():mj.mj_kinematics(model,d);mj.mj_comPos(model,d)
    def bind_shoe():
        R=d.xmat[holder].reshape(3,3);d.qpos[sq:sq+3]=d.xpos[holder]+R@grip_p
        mj.mju_mat2Quat(d.qpos[sq+3:sq+7],(R@grip_R).ravel());fk()
    def reset(f):
        for actor,(rq,qa,ids)in enumerate(maps):d.qpos[rq:rq+7]=q[f,actor,:7];d.qpos[qa]=q[f,actor,7:]
        d.qpos[sq:sq+7]=z['shoe_wear_only_reference_pose'][f]if f<switch or(a.pregrasp_only and f==switch) else z['object_target_pose'][f]
        fk()
        if f>=switch and not a.pregrasp_only:bind_shoe()
    def skew(v):return np.asarray([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    allfits=[];repair_counts=[]
    for actor in range(2):
      for side,handpre in [('left','lh_'),('right','rh_')]:
        if a.only_task_arm and not(actor==0 and side=='right'):continue
        if a.actor_side and f'p{actor+1}_{side}'not in a.actor_side:continue
        pre=f'p{actor+1}_';rq,qa,ids=maps[actor]
        sel=np.asarray([k for k,nm in enumerate(names)if nm.startswith((side+'_shoulder',side+'_elbow',side+'_wrist'))])
        qa_sel=qa[sel];j_sel=np.asarray(ids)[sel];va=model.jnt_dofadr[j_sel];nj=len(sel)
        hand=mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,pre+handpre+'hand_base_link')
        elbow=mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,pre+side+'_elbow_link')
        if min(hand,elbow)<0:raise ValueError('Scene arm/body names do not match retarget contract')
        hand_geoms=[g for g in range(model.ngeom)if model.geom_contype[g]and bn(model.geom_bodyid[g]).startswith((pre+handpre,pre+side+'_wrist'))]
        oldfk=[];oldrot=[];elbowfk=[]
        for f in range(len(t)):
            reset(f);oldfk.append(d.xpos[hand].copy());oldrot.append(d.xmat[hand].reshape(3,3).copy());elbowfk.append(d.xpos[elbow].copy())
        oldfk=np.asarray(oldfk);oldrot=np.asarray(oldrot);elbowfk=np.asarray(elbowfk)
        key=set(range(0,len(t),a.key_stride))|{len(t)-1,switch}
        if actor==0 and side=='right':key.update(range(200,min(272,len(t))))
        key=sorted(key);solutions={};previous=None
        if a.pregrasp_only:key=[f for f in key if f<=switch]
        def constraints(f):
            pairs=[(g,w,a.bench_clearance)for g in hand_geoms for w in bench]
            if actor==0 and f<=switch:pairs +=[(g,w,a.shoe_clearance)for g in hand_geoms for w in shoe]
            if actor==0 and side=='left'and f>switch:pairs +=[(g,w,a.bench_clearance)for g in hand_geoms for w in shoe]
            if actor==0 and side=='right'and f>=switch and not a.pregrasp_only:
                # The wearing foot is still inside its open shoe at handover.
                # Demand separation progressively during removal, without
                # manufacturing a 15mm instantaneous gap at the capture frame.
                held_margin=a.held_shoe_clearance*float(np.clip((t[f]-t[switch])/.35,0,1))
                pairs +=[(g,w,held_margin)for g in shoe for w in obstacles]
            return pairs
        def fit(f,seed,previous_frame=None,neighbor_bounds=False):
            reset(f);fixed=d.qpos.copy();pairs=constraints(f)
            if a.only_task_arm and f==switch:
                # Preserve the independently verified hand/foot/shoe capture
                # pose. Otherwise recomputing a new grip after optimization
                # changes the shoe rigid transform and invalidates all held
                # collision constraints that were just optimized.
                allfits.append({'actor':actor+1,'side':side,'frame':f,'cost':0.,'nfev':0,
                                'hand_position_delta_m':0.,'handover_anchor_preserved':True})
                return initial_q[f,actor,7+sel].copy()
            target=oldfk[f].copy();rotation=oldrot[f].copy()
            if actor==1 and side=='right':target[2]+=a.p2right_extra_lift
            elif actor==0 and side=='left':target[2]+=a.left_extra_lift
            if targets is not None and actor==0 and side=='right'and f>=switch:
                target=np.asarray(targets['right_hand_base_position_world_m'][f]);rotation=np.asarray(targets['right_hand_base_rotation_world'][f])
            low=model.jnt_range[j_sel,0]+1e-6;hi=model.jnt_range[j_sel,1]-1e-6
            if a.only_task_arm:
                anchor=initial_q[switch,actor,7+sel];reach=a.joint_speed_limit*abs(float(t[f]-t[switch]))
                low=np.maximum(low,anchor-reach);hi=np.minimum(hi,anchor+reach)
            if previous_frame is not None:
                delta=a.joint_speed_limit*float(t[f]-t[previous_frame])
                low=np.maximum(low,seed-delta);hi=np.minimum(hi,seed+delta)
            if neighbor_bounds:
                for other in(f-1,f+1):
                    if 0<=other<len(t):
                        delta=a.joint_speed_limit*abs(float(t[f]-t[other]))
                        low=np.maximum(low,q[other,actor,7+sel]-delta)
                        hi=np.minimum(hi,q[other,actor,7+sel]+delta)
            if np.any(high_gap:=hi-low<=1e-7):
                hi=np.maximum(hi,low+2e-7)
            seed=np.clip(seed,low+1e-8,hi-1e-8)
            cache={}
            def eval(x):
                if 'x'in cache and np.array_equal(cache['x'],x):return cache['r'],cache['J']
                d.qpos[:]=fixed;d.qpos[qa_sel]=x;fk()
                if f>=switch and not a.pregrasp_only:bind_shoe()
                rr=[];JJ=[]
                def add(v,j,w):rr.extend(np.asarray(v).ravel()*w);JJ.extend(np.asarray(j).reshape(-1,nj)*w)
                mj.mj_jacBody(model,d,jp,jr,hand);hpj=jp[:,va].copy();hrj=jr[:,va].copy()
                add(d.xpos[hand]-target,hpj,4.)
                R=d.xmat[hand].reshape(3,3)
                for ax in(0,2):add(R[:,ax]-rotation[:,ax],-skew(R[:,ax])@hrj,a.orientation_weight)
                mj.mj_jacBody(model,d,jp,jr,elbow);add(d.xpos[elbow]-elbowfk[f],jp[:,va],.5)
                for g,w,clearance in pairs:
                    witness=np.zeros(6);dist=float(mj.mj_geomDistance(model,d,g,w,.10,witness));r=min(dist-clearance,0.)
                    dj=np.zeros(nj)
                    if r<0 and np.linalg.norm(witness[:3]-witness[3:])>1e-10:
                        delta=witness[:3]-witness[3:];normal=delta/np.linalg.norm(delta)*(1 if dist>=0 else -1)
                        if g in shoe and f>=switch:
                            # The shoe is a declared rigid grasp relative to the
                            # hand during reference optimization. Differentiate
                            # its witness point through the holder arm, not its
                            # unrelated freejoint coordinates.
                            mj.mj_jac(model,d,jp,jr,witness[:3],holder);j1=jp[:,va].copy()
                            mj.mj_jac(model,d,jp,jr,witness[3:],int(model.geom_bodyid[w]));dj=normal@(j1-jp[:,va])
                        else:
                            mj.mj_jac(model,d,jp,jr,witness[:3],int(model.geom_bodyid[g]));dj=normal@jp[:,va]
                    add([r],dj.reshape(1,-1),30.)
                add(x-initial_q[f,actor,7+sel],np.eye(nj),.015)
                add(x-seed,np.eye(nj),.012)
                cache.update(x=x.copy(),r=np.asarray(rr),J=np.asarray(JJ));return cache['r'],cache['J']
            result=least_squares(lambda x:eval(x)[0],seed,jac=lambda x:eval(x)[1],bounds=(low,hi),max_nfev=a.max_nfev,ftol=1e-6,xtol=1e-6,gtol=1e-6)
            eval(result.x)
            row={'actor':actor+1,'side':side,'frame':f,'cost':float(result.cost),'nfev':result.nfev,'hand_position_delta_m':float(np.linalg.norm(d.xpos[hand]-oldfk[f]))}
            allfits.append(row);return result.x
        prev_frame=None
        for f in key:
            initial=q[f,actor,7+sel]if previous is None else previous
            solutions[f]=fit(f,initial,previous_frame=prev_frame);previous=solutions[f];prev_frame=f
        tt=np.asarray([t[f]for f in key]);xx=np.asarray([solutions[f]for f in key])
        for j,k in enumerate(sel):
            if a.pregrasp_only:
                q[:switch+1,actor,7+k]=np.interp(t[:switch+1],tt,xx[:,j])
                delta=q[switch,actor,7+k]-initial_q[switch,actor,7+k]
                u=np.clip((t[switch+1:]-t[switch])/.5,0,1);fade=1-(3*u*u-2*u*u*u)
                q[switch+1:,actor,7+k]=initial_q[switch+1:,actor,7+k]+delta*fade
                for f in range(switch+1,len(t)):
                    limit=a.joint_speed_limit*float(t[f]-t[f-1])
                    q[f,actor,7+k]=np.clip(q[f,actor,7+k],q[f-1,actor,7+k]-limit,q[f-1,actor,7+k]+limit)
            else:q[:,actor,7+k]=np.interp(t,tt,xx[:,j])
        # Inspect every frame, solve only clearance failures. A small allowance
        # avoids repairing <1mm interpolation/solver discrepancies repeatedly.
        bad=[]
        for f in range(len(t)):
            if a.pregrasp_only and f>switch:continue
            reset(f)
            if any(float(mj.mj_geomDistance(model,d,g,w,.10,np.zeros(6)))<clearance-.001 for g,w,clearance in constraints(f)):
                bad.append(f)
        for f in bad:q[f,actor,7+sel]=fit(f,q[f,actor,7+sel],neighbor_bounds=True)
        repair_counts.append({'actor':actor+1,'side':side,'keyframes':len(key),'repaired_frames':len(bad)})
        print(json.dumps(repair_counts[-1]),flush=True)
    # Keep a continuous rigid grasp transform at the actual reference contact
    # frame. Shoe positions before handover remain foot-bound independently.
    hand=mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,'p1_rh_hand_base_link')
    reset(switch);R=d.xmat[hand].reshape(3,3);s=z['shoe_wear_only_reference_pose'][switch]
    Rs=np.empty(9);mj.mju_quat2Mat(Rs,s[3:]);relp=R.T@(s[:3]-d.xpos[hand]);relR=R.T@Rs.reshape(3,3)
    grip_translation_change=float(np.linalg.norm(relp-grip_p));grip_rotation_change=float(np.linalg.norm(relR-grip_R))
    if a.only_task_arm and(grip_translation_change>1e-7 or grip_rotation_change>1e-7):
        raise ValueError('Frozen handover grip changed during post-grasp optimization')
    objectposes=[]
    for f in range(len(t)):
        if f<switch:objectposes.append(z['shoe_wear_only_reference_pose'][f]);continue
        reset(f);R=d.xmat[hand].reshape(3,3);pose=np.empty(7);pose[:3]=d.xpos[hand]+R@relp;mj.mju_mat2Quat(pose[3:],(R@relR).ravel());objectposes.append(pose)
    objectposes=np.asarray(objectposes)
    if a.unmodified_ik_reference:
        base_path=a.unmodified_ik_reference if a.unmodified_ik_reference.is_absolute()else root/a.unmodified_ik_reference
        base=np.load(base_path,allow_pickle=False)
        if not np.array_equal(base['time_s'],t)or not np.array_equal(base['dof_names'],z['dof_names']):
            raise ValueError('Unmodified IK source uses a different timeline or joint schema')
        payload['reference_unmodified_fullbody_ik_qpos']=base['qpos'].copy()
        payload['reference_unmodified_fullbody_ik_sha256']=np.asarray(sha(base_path))
    elif 'synthetic_arm_clearance_adjustment'not in z.files or not bool(np.asarray(z['synthetic_arm_clearance_adjustment']).item()):
        payload['reference_unmodified_fullbody_ik_qpos']=initial_q.copy()
        payload['reference_unmodified_fullbody_ik_sha256']=np.asarray(sha(inp))
    elif 'reference_unmodified_fullbody_ik_sha256'not in payload:
        payload.pop('reference_unmodified_fullbody_ik_qpos',None)
        payload['reference_input_kinematic_qpos']=initial_q.copy()
    history=json.loads(str(payload['reference_adaptation_history_json'].item())) if 'reference_adaptation_history_json'in payload else []
    history.append({'script_sha256':generator_sha256,'input_reference_sha256':sha(inp),'scene_sha256':sha(scene),
                    'targets_sha256':sha(a.targets if a.targets.is_absolute()else root/a.targets)if a.targets else None,
                    'only_task_arm':a.only_task_arm,'joint_speed_limit_rad_s':a.joint_speed_limit,
                    'bench_clearance_m':a.bench_clearance,'held_shoe_clearance_m':a.held_shoe_clearance})
    payload.update(qpos=q,qpos_p1=q[:,0],qpos_p2=q[:,1],object_target_pose=objectposes,contact_optimized_shoe_pose=objectposes,
        shoe_hand_position_local_to_right_hand=relp,shoe_hand_rotation_local_to_right_hand=relR,
        pre_arm_refinement_qpos=initial_q,synthetic_arm_clearance_adjustment=np.asarray(True),
        reference_adaptation_history_json=np.asarray(json.dumps(history)),
        shoe_source_position_correction_m=np.linalg.norm(objectposes[:,:3]-z['shoe_raw_source_pose'][:,:3],axis=1))
    out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,**payload)
    report={'schema':'rgb_near_arm_contact_refine.v1','input_sha256':sha(inp),'reference_sha256':sha(out),'scene_sha256':sha(scene),
            'script_sha256':generator_sha256,'frames':len(t),'actors':2,'bench_clearance_m':a.bench_clearance,
            'shoe_clearance_m':a.shoe_clearance,'targets':str(a.targets)if a.targets else None,
            'targets_sha256':history[-1]['targets_sha256'],'only_task_arm':a.only_task_arm,
            'actor_side':a.actor_side,'left_extra_lift_m':a.left_extra_lift,'p2right_extra_lift_m':a.p2right_extra_lift,
            'pregrasp_only':a.pregrasp_only,
            'grip_translation_change_m':grip_translation_change,'grip_rotation_matrix_change':grip_rotation_change,
            'joint_adjustment_abs_max_rad':float(np.max(np.abs(q[:,:,7:]-initial_q[:,:,7:]))),
            'joint_speed_limit_rad_s':a.joint_speed_limit,'orientation_weight':a.orientation_weight,
            'held_shoe_clearance_m':a.held_shoe_clearance,
            'held_shoe_clearance_ramp_s':.35,
            'joint_speed_abs_max_rad_s':float(np.max(np.abs(np.diff(q[:,:,7:],axis=0))/np.diff(t)[:,None,None])),
            'root_leg_trajectories_preserved':True,'source_observations_modified':False,
            'repair_counts':repair_counts,'fits':allfits}
    out.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    out.with_suffix('.generator.py').write_bytes(generator_bytes)
    print(json.dumps({k:v for k,v in report.items()if k!='fits'}),flush=True)


if __name__=='__main__':main()
