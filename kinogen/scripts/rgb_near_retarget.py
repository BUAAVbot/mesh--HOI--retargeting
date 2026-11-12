"""Full-sequence seated Vita retarget from numeric SMPL-X posed-mesh targets.

Raw source joints, shoe poses and timing are retained. Shoe/foot/hand contact
adaptation is an explicit inferred task model; source finger observations are
invalid. This script generates references and never integrates dynamics.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import copy
import xml.etree.ElementTree as ET


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/data/KinoGen'))
    p.add_argument('--source-targets',type=Path)
    p.add_argument('--model',type=Path)
    p.add_argument('--data',type=Path)
    p.add_argument('--scene',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--max-nfev',type=int,default=35)
    p.add_argument('--seat-height',type=float,default=.397)
    p.add_argument('--release-time',type=float,default=11.3)
    p.add_argument('--wear-foot-lift',type=float,default=0.)
    p.add_argument('--foot-floor-weight',type=float,default=5.)
    p.add_argument('--frame-stop',type=int,default=398)
    a=p.parse_args()
    generator_bytes=Path(__file__).read_bytes()
    generator_sha256=hashlib.sha256(generator_bytes).hexdigest()
    import numpy as np
    import mujoco as mj
    from scipy.optimize import least_squares
    root=a.root
    def resolve(path,default):
        x=Path(path)if path is not None else Path(default)
        return x if x.is_absolute()else root/x
    src=resolve(a.data,'data/rgb_261026_near_handoff')
    source_path=resolve(a.source_targets,'outputs/rgb_near_pipeline/source_joint_targets.npz')
    model_path=resolve(a.model,'outputs/rgb_near_pipeline/assets/vita_single_proxy.xml')
    output=resolve(a.output,'outputs/rgb_near_pipeline/reference_v1.npz')
    output.parent.mkdir(parents=True,exist_ok=True)
    s=np.load(source_path,allow_pickle=False)
    t=np.asarray(s['time_s'][:a.frame_stop]);n=len(t)
    combined=resolve(a.scene,'outputs/rgb_near_pipeline/assets/vita_bench_two52.xml')
    if combined.exists():
        tree=ET.parse(model_path);world=tree.getroot().find('worldbody')
        for geom in ET.parse(combined).getroot().find('worldbody').findall('geom'):
            if (geom.get('name')or'').startswith('bench_') and int(geom.get('contype','1')):
                world.append(copy.deepcopy(geom))
        model_path=output.with_suffix('.ik_scene.xml')
        tree.write(model_path,encoding='unicode')
    model=mj.MjModel.from_xml_path(str(model_path))
    d=mj.MjData(model)
    js=[j for j in range(model.njnt)if model.jnt_type[j]==mj.mjtJoint.mjJNT_HINGE]
    js=sorted(js,key=lambda j:model.jnt_qposadr[j])
    names=[mj.mj_id2name(model,mj.mjtObj.mjOBJ_JOINT,j) for j in js]
    if len(names)!=52:raise ValueError('Expected full Vita52')
    qa=np.asarray([model.jnt_qposadr[j]for j in js]); va=np.asarray([model.jnt_dofadr[j]for j in js])
    selected=np.asarray([k for k,name in enumerate(names)if not name.startswith(('lh_','rh_'))])
    xqa=np.r_[np.arange(3),qa[selected]];xva=np.r_[np.arange(3),va[selected]]
    nj=len(xqa)
    bids={name:mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,name) for name in (
        'base_link','left_hip_pitch_link','right_hip_pitch_link','left_knee_link','right_knee_link',
        'left_ankle_roll_link','right_ankle_roll_link','left_shoulder_pitch_link','right_shoulder_pitch_link',
        'left_elbow_link','right_elbow_link','left_wrist_pitch_link','right_wrist_pitch_link',
        'lh_hand_base_link','rh_hand_base_link','neck_yaw_link')}
    if min(bids.values())<0:raise ValueError('Missing required named robot bodies')
    body_targets=[('left_knee_link',4,2.),('right_knee_link',5,2.),
        ('left_ankle_roll_link',7,3.),('right_ankle_roll_link',8,3.),
        ('left_shoulder_pitch_link',16,1.5),('right_shoulder_pitch_link',17,1.5),
        ('left_elbow_link',18,1.5),('right_elbow_link',19,1.5),
        ('left_wrist_pitch_link',20,1.),('right_wrist_pitch_link',21,1.)]
    foot_geoms={side:[g for g in range(model.ngeom)if
        (mj.mj_id2name(model,mj.mjtObj.mjOBJ_GEOM,g)or'').startswith('collision_foot_'+short+'_')]
        for side,short in [('left','l'),('right','r')]}
    if not all(foot_geoms.values()):raise ValueError('Missing foot collision geoms')
    seat_gid=mj.mj_name2id(model,mj.mjtObj.mjOBJ_GEOM,'bench_seat_proxy')
    scene_geoms=[g for g in range(model.ngeom)if
                 (mj.mj_id2name(model,mj.mjtObj.mjOBJ_GEOM,g)or'').startswith('bench_')]
    robot_geoms=[g for g in range(model.ngeom)if int(model.geom_bodyid[g])>0 and int(model.geom_contype[g])]
    sit_geoms=[g for g in robot_geoms if 'hip_front_roll' in(mj.mj_id2name(model,mj.mjtObj.mjOBJ_GEOM,g)or'')]
    geom_vertices={}
    for g in range(model.ngeom):
        if model.geom_type[g]==mj.mjtGeom.mjGEOM_MESH:
            m=int(model.geom_dataid[g]);st=int(model.mesh_vertadr[m]);ct=int(model.mesh_vertnum[m])
            geom_vertices[g]=model.mesh_vert[st:st+ct].copy()

    def fk():mj.mj_kinematics(model,d);mj.mj_comPos(model,d)
    jp=np.zeros((3,model.nv));jr=np.zeros_like(jp)
    def point(body,local=None):
        b=bids[body];pos=d.xpos[b].copy()
        if local is not None:pos+=d.xmat[b].reshape(3,3)@local
        mj.mj_jac(model,d,jp,jr,pos,b)
        return pos,jp[:,xva].copy(),jr[:,xva].copy()
    def bottom(g):
        R=d.geom_xmat[g].reshape(3,3);pos=d.geom_xpos[g].copy();size=model.geom_size[g];typ=int(model.geom_type[g])
        if typ==mj.mjtGeom.mjGEOM_CAPSULE:
            pos-=np.sign(R[2,2])*size[1]*R[:,2];pos[2]-=size[0]
        elif typ==mj.mjtGeom.mjGEOM_SPHERE:pos[2]-=size[0]
        elif typ==mj.mjtGeom.mjGEOM_BOX:pos+=R@(-np.sign(R[2])*size)
        elif typ==mj.mjtGeom.mjGEOM_MESH:
            vv=geom_vertices[g]@R.T+pos;pos=vv[np.argmin(vv[:,2])]
        else:pos[2]-=model.geom_rbound[g]
        b=int(model.geom_bodyid[g]);mj.mj_jac(model,d,jp,jr,pos,b)
        return pos,jp[:,xva].copy()
    def foot_bottom(side):return min([bottom(g)for g in foot_geoms[side]],key=lambda x:x[0][2])
    def unit(x):return x/max(np.linalg.norm(x),1e-12)
    def skew(v):return np.asarray([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    allq=[];metrics=[];allfk=[];sole=[];rawtargets=[];adaptedtargets=[]
    for person in(1,2):
        joints=np.asarray(s[f'person{person}_joints_world'][:n],dtype=float)
        raw_root=np.asarray(s[f'person{person}_root_xyz_wxyz'][:n],dtype=float)
        q0=np.zeros(model.nq);q0[3]=1.
        q0[:3]=raw_root[0,:3]
        # Seated seed is a numerical initialization, not an inserted motion.
        for side in('left','right'):
            for token,val in [('hip_pitch_joint',-1.4),('knee_joint',1.55),('ankle_pitch_joint',-.15)]:
                q0[qa[names.index(side+'_'+token)]]=val
        prev=q0.copy();personq=[];personfk=[];personsole=[];personerrors=[];personraw=[];personadapt=[]
        yaw0=float(np.arctan2(*(np.cross(unit(joints[0,16]-joints[0,17]),[0,0,1])[[1,0]])))
        # Source yaw is observed from shoulders; median fixed base yaw avoids
        # converting torso turns into seat/foot yaw sliding.
        source_yaws=[]
        for f in range(n):
            lateral=joints[f,16]-joints[f,17];lateral[2]=0;fw=np.cross(unit(lateral),[0,0,1]);source_yaws.append(np.arctan2(fw[1],fw[0]))
        yaw=float(np.median(np.unwrap(source_yaws)))
        quat=np.array([np.cos(yaw/2),0,0,np.sin(yaw/2)])
        for f in range(n):
            raw=joints[f].copy();target=raw.copy()
            # Ankle centers lie above a flat sole; floor-correct only source
            # ankle targets near the observed ground, keeping lifted legs.
            for idx in(7,8):target[idx,2]=max(float(raw[idx,2])+.070,.074)
            wear_lift=a.wear_foot_lift*np.clip((a.release_time+.5-t[f])/.5,0,1) if person==1 else 0.
            target[7,2]+=wear_lift
            base_target=raw_root[f,:3].copy();base_target[2]+=.015
            qfixed=prev.copy();qfixed[3:7]=quat
            qfixed[qa[[i for i,nm in enumerate(names)if nm.startswith(('lh_','rh_'))]]]=0.
            dt=float(t[f]-t[f-1]) if f else 1/23
            low=np.r_[raw_root[0,:2]-.18,.43,model.jnt_range[np.asarray(js)[selected],0]+1e-5]
            high=np.r_[raw_root[0,:2]+.18,.78,model.jnt_range[np.asarray(js)[selected],1]-1e-5]
            if f:
                low[3:]=np.maximum(low[3:],prev[xqa][3:]-max(.12,dt*5.))
                high[3:]=np.minimum(high[3:],prev[xqa][3:]+max(.12,dt*5.))
            seed=np.clip(prev[xqa],low+1e-7,high-1e-7)
            cache={}
            def evaluate(x):
                if 'x'in cache and np.array_equal(x,cache['x']):return cache['r'],cache['J']
                d.qpos[:]=qfixed;d.qpos[xqa]=x;fk();rr=[];JJ=[]
                def add(v,j,w):rr.extend(np.asarray(v).ravel()*w);JJ.extend(np.asarray(j).reshape(-1,nj)*w)
                add(d.qpos[:3]-base_target,np.c_[np.eye(3),np.zeros((3,nj-3))],1.3)
                for body,idx,w in body_targets:
                    pos,j,_=point(body)
                    if person==1 and body=='right_wrist_pitch_link' and 9.7<t[f]<12.2:w*=.3
                    add(pos-target[idx],j,w)
                # Flat foot support when planted, with collision-shape minima.
                for side,idx,toe in [('left',7,10),('right',8,11)]:
                    pos,jpnt,jrot=point(side+'_ankle_roll_link')
                    up=d.xmat[bids[side+'_ankle_roll_link']].reshape(3,3)[:,2]
                    planted=raw[idx,2]<.10
                    if planted:
                        bp,bj=foot_bottom(side)
                        floor_target=.002+(wear_lift if side=='left' else 0.)
                        add([bp[2]-floor_target],bj[2:3],a.foot_floor_weight)
                        add(up-[0,0,1],-skew(up)@jrot,.12)
                    else:add(up-[0,0,1],-skew(up)@jrot,.025)
                # Seat support comes from proximal thigh proxies at the pelvis.
                # The explicit seat-height prior regularizes a seated base;
                # static collision audit refines it before dynamics.
                add([x[2]-.600],np.r_[0.,0.,1.,np.zeros(nj-3)].reshape(1,-1),.2)
                # Real primitive distance/witness points impose bench collision
                # inequalities. This does not move a dynamic state; it only
                # projects the reference into the declared contact geometry.
                seat_support=[]
                for g in robot_geoms:
                    for wall in scene_geoms:
                        witness=np.zeros(6)
                        dist=float(mj.mj_geomDistance(model,d,g,wall,.10,witness))
                        if dist>=.10:
                            add([0.],np.zeros((1,nj)),18.)
                            continue
                        delta=witness[:3]-witness[3:]
                        length=np.linalg.norm(delta)
                        if length<1e-10:
                            add([0.],np.zeros((1,nj)),18.)
                            continue
                        normal=delta/length*(1 if dist>=0 else -1)
                        mj.mj_jac(model,d,jp,jr,witness[:3],int(model.geom_bodyid[g]))
                        dj=normal@jp[:,xva]
                        add([min(dist+.001,0.)],dj.reshape(1,-1) if dist<-.001 else np.zeros((1,nj)),18.)
                        if wall==seat_gid and g in sit_geoms:seat_support.append((dist,dj))
                if seat_support:
                    support=min(seat_support,key=lambda x:x[0]);add([support[0]-.001],support[1].reshape(1,-1),2.)
                else:add([0.],np.zeros((1,nj)),2.)
                if person==1:
                    tt=float(t[f]);approach=np.clip((tt-9.7)/.7,0,1)*np.clip((12.5-tt)/1.2,0,1)
                    if approach>0:
                        hp,hj,_=point('rh_hand_base_link',np.array([.015,0,.105]))
                        fp,fj,_=point('left_ankle_roll_link',np.array([.04,0,-.03]))
                        add(hp-fp,hj-fj,4.*approach)
                # Weak joint and temporal regularization resolves redundant IK.
                reg=np.zeros((len(selected),nj));reg[:,3:]=np.eye(len(selected))
                add(x[3:]-seed[3:],reg,.025 if f else .008)
                cache.update(x=x.copy(),r=np.asarray(rr),J=np.asarray(JJ));return cache['r'],cache['J']
            fit=least_squares(lambda x:evaluate(x)[0],seed,jac=lambda x:evaluate(x)[1],bounds=(low,high),
                              max_nfev=100 if f==0 else a.max_nfev,ftol=1e-6,xtol=1e-6,gtol=1e-6)
            evaluate(fit.x);q=d.qpos.copy();prev=q.copy()
            actual=np.array([d.xpos[bids[b]] for b,_,_ in body_targets])
            desired=np.array([target[idx]for _,idx,_ in body_targets])
            row={'actor':person,'frame':f,'time_s':float(t[f]),'ik_position_rms_m':float(np.sqrt(np.mean((actual-desired)**2))),
                 'max_target_error_m':float(np.max(np.linalg.norm(actual-desired,axis=1))),
                 'cost':float(fit.cost),'nfev':fit.nfev,'root_xyz':q[:3].tolist()}
            personq.append(q);personfk.append(actual);personsole.append([foot_bottom(x)[0][2]for x in('left','right')]);personerrors.append(row)
            personraw.append(raw);personadapt.append(target)
            if f%50==0 or f==n-1:print(json.dumps(row),flush=True)
        allq.append(personq);allfk.append(personfk);sole.append(personsole);metrics.extend(personerrors)
        rawtargets.append(personraw);adaptedtargets.append(personadapt)
    q=np.asarray(allq).transpose(1,0,2)
    shoe=np.load(src/'objects/shoe/poses_world.npz',allow_pickle=False)
    rawshoe=np.c_[shoe['position_m'][:n],shoe['quaternion_wxyz'][:n]]
    optimized=[];phase=[];foot_attachment=[];hand_attachment=[]
    rfix=np.array([[0,1,0],[-1,0,0],[0,0,1.]])
    for f in range(n):
        d.qpos[:]=q[f,0];fk()
        ankle=bids['left_ankle_roll_link'];hand=bids['rh_hand_base_link']
        Rf=d.xmat[ankle].reshape(3,3);Rh=d.xmat[hand].reshape(3,3)
        pp=d.xpos[ankle]+Rf@np.array([.04,0,-.03]);Rf=Rf@rfix
        hp=d.xpos[hand]+Rh@np.array([.015,0,.105])
        foot_attachment.append(pp)
        hand_attachment.append(hp)
        if t[f]<=a.release_time:pos=pp;rot=Rf;phase.append(0)
        else:
            pos=hp;rot=Rh@np.array([[0,0,1],[1,0,0],[0,1,0.]])
            phase.append(2)
        qq=np.empty(4);mj.mju_mat2Quat(qq,rot.ravel());optimized.append(np.r_[pos,qq])
    opt=np.asarray(optimized)
    # All output conventions are explicit; preserve original source trajectory
    # independently from contact-consistent task targets.
    payload=dict(qpos=q,qpos_p1=q[:,0],qpos_p2=q[:,1],dof_names=np.asarray(names),time_s=t,
        frame_ids=np.arange(n),actor_names=np.asarray(['person_01','person_02']),
        raw_source_joint_targets=np.asarray(rawtargets).transpose(1,0,2,3),
        adapted_joint_targets=np.asarray(adaptedtargets).transpose(1,0,2,3),
        source_joint_regressor=s['smplx_joint_regressor'],
        raw_source_root_pose_xyz_wxyz=np.stack([s['person1_root_xyz_wxyz'][:n],s['person2_root_xyz_wxyz'][:n]],axis=1),
        fk_body_target_positions=np.asarray(allfk).transpose(1,0,2,3),
        foot_sole_min_z=np.asarray(sole).transpose(1,0,2),
        object_target_pose=opt,contact_optimized_shoe_pose=opt,shoe_raw_source_pose=rawshoe,
        raw_source_object_pose_xyz_wxyz=rawshoe,object_ids=np.asarray([0]),
        shoe_contact_phase=np.asarray(phase),shoe_contact_phase_names=np.asarray(['wear_left_foot','handover','right_hand','free']),
        shoe_attachment_name=np.where(np.asarray(phase)==0,'shoe_wear_left','shoe_grasp_right'),
        source_shoe_valid=shoe['valid'][:n],source_shoe_orientation_observed=shoe['orientation_observed'][:n],
        synthetic_task_adaptation=np.asarray(True),synthetic_hand_contact_adaptation=np.asarray(True),
        source_hand_valid=np.zeros((n,2,2),dtype=bool),
        contact_candidate_observed=np.stack([np.c_[np.zeros(n,dtype=bool),t>=9.7],np.zeros((n,2),dtype=bool)],axis=1)[:,:,:,None],
        contact_candidate_confidence=np.stack([np.c_[np.zeros(n),np.where(t>=9.7,.5,0.)],np.zeros((n,2))],axis=1)[:,:,:,None],
        foot_shoe_site_world=np.asarray(foot_attachment),hand_shoe_site_world=np.asarray(hand_attachment),
        shoe_source_position_correction_m=np.linalg.norm(opt[:,:3]-rawshoe[:,:3],axis=1),
        inferred_attachment_bodies=np.asarray(['p1_left_ankle_roll_link','p1_rh_hand_base_link']),
        adaptation_description=np.asarray('Seated whole-body IK from posed-mesh joint estimates; floor corrected ankle targets, fixed median root yaw, inferred left-foot shoe wear then right-hand removal at11.3s; raw shoe and body estimates retained. Finger observations unavailable.'))
    output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(output,**payload)
    report={'schema':'rgb_near_full_retarget.v1','frames':n,'actors':2,'duration_s':float(t[-1]),'model':str(model_path),'model_sha256':sha(model_path),
            'scene':str(combined),'scene_sha256':sha(combined),'data':str(src),'script_sha256':generator_sha256,
            'source_targets':str(source_path),'source_targets_sha256':sha(source_path),'reference_sha256':sha(output),
            'source_frame_subset':False if n==398 else True,'source_positions_modified':True,
            'raw_source_arrays_modified':False,'reference_positions_adapted':True,
            'legacy_source_positions_modified_meaning':'Only derived reference targets are adapted; saved raw source arrays are unchanged.',
            'object_source_error_m_max':float(np.max(payload['shoe_source_position_correction_m'])),
            'object_source_error_m_median':float(np.median(payload['shoe_source_position_correction_m'])),
            'ik_rms_m_median':float(np.median([r['ik_position_rms_m']for r in metrics])),
            'foot_sole_min_z_m':float(np.min(payload['foot_sole_min_z'])),
            'limitations':['Joint targets regressed from posed source mesh; not independently observed joints.',
                'Original hands invalid; hand/shoe task constraints inferred from source video.',
                'Reference construction is kinematic; dynamic simulation must pass separately.'], 'frames_metrics':metrics}
    output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    output.with_suffix('.generator.py').write_bytes(generator_bytes)
    print(json.dumps({k:v for k,v in report.items()if k!='frames_metrics'}),flush=True)


if __name__=='__main__':main()
