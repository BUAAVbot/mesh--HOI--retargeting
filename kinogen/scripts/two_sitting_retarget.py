"""Contact-aware kinematic Vita references for two people rising/swapping/sitting.

This generates references only. It never overwrites an executing dynamic state.
Raw observations and source timing remain separate from repaired/adapted targets.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inside(xy,polygon,np):
    polygon=np.asarray(polygon);edge=np.roll(polygon,-1,axis=0)-polygon;d=np.asarray(xy)-polygon
    cross=edge[:,0]*d[:,1]-edge[:,1]*d[:,0]
    return bool(np.all(cross>=-1e-8)or np.all(cross<=1e-8))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'data/two_sitting_people/rgb_174114_sim_handoff')
    p.add_argument('--assets',type=Path,default=ROOT/'outputs/two_sitting_pipeline/assets')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-nfev',type=int,default=45)
    p.add_argument('--time-scale',type=float,default=1.)
    p.add_argument('--source-smoothing-sigma',type=float,default=1.)
    p.add_argument('--seat-xy-adaptation',action='store_true',help='Explicitly shift seated robot pelvis toward the observed support patch, with smooth rise/sit blending')
    p.add_argument('--continuous-foot-support',action='store_true',help='Infer at least one grounded foot outside seat support; not an observed contact label')
    p.add_argument('--frame-stop',type=int,default=172)
    a=p.parse_args()
    import numpy as np,mujoco as mj
    from scipy.ndimage import gaussian_filter1d,binary_closing
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    if a.time_scale<=0:raise ValueError('Positive time-scale required')
    if a.output.exists():raise FileExistsError('Use a new output: '+str(a.output))
    spec=json.loads((a.assets/'support_geometry.json').read_text())
    U=np.asarray(spec['source_to_sim_U']);ground=np.asarray(spec['ground']['z_eq_ax_by_c'])
    model_path=a.assets/'vita_single_proxy.xml';m=mj.MjModel.from_xml_path(str(model_path.resolve()));d=mj.MjData(m)
    jointids=np.array([i for i in range(m.njnt)if m.jnt_type[i]==mj.mjtJoint.mjJNT_HINGE]);jointids=jointids[np.argsort(m.jnt_qposadr[jointids])]
    names=[mj.mj_id2name(m,mj.mjtObj.mjOBJ_JOINT,int(j))for j in jointids]
    if len(names)!=52 or m.nq!=59:raise ValueError('Expected original full Vita52 proxy')
    qa=m.jnt_qposadr[jointids];va=m.jnt_dofadr[jointids]
    selected=np.array([i for i,n in enumerate(names)if not n.startswith(('lh_','rh_','neck_'))]);nx=3+len(selected)
    H=np.zeros((m.nv,nx));H[:3,:3]=np.eye(3);H[va[selected],np.arange(3,nx)]=1
    TJ=np.c_[np.eye(3),np.zeros((3,nx-3))]
    bodytargets=[('left_knee_link',4,2.),('right_knee_link',5,2.),('left_shoulder_pitch_link',16,1.),('right_shoulder_pitch_link',17,1.),('left_elbow_link',18,1.),('right_elbow_link',19,1.),('left_wrist_pitch_link',20,.7),('right_wrist_pitch_link',21,.7),('neck_yaw_link',12,.6)]
    bids={n:mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,n)for n in ['base_link','left_ankle_roll_link','right_ankle_roll_link']+[n for n,_,_ in bodytargets]}
    if min(bids.values())<0:raise ValueError('Canonical Vita body names missing')
    geoms=lambda token:[g for g in range(m.ngeom)if token in(mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,g)or'')and m.geom_contype[g]]
    footgeoms=[geoms('collision_foot_'+s+'_')for s in('l','r')]
    hipgeoms=[geoms('collision_'+s+'_hip_roll_cylinder')+geoms('collision_'+s+'_hip_front_roll_cylinder')for s in('l','r')]
    if not all(footgeoms):raise ValueError('Canonical source foot colliders missing')
    robot=[g for g in range(m.ngeom)if m.geom_bodyid[g]>0 and m.geom_contype[g]];rindex={g:k for k,g in enumerate(robot)}
    worldids={g for g in range(m.ngeom)if m.geom_bodyid[g]==0 and m.geom_contype[g]}
    localmesh={g:m.mesh_vert[int(m.mesh_vertadr[m.geom_dataid[g]]):int(m.mesh_vertadr[m.geom_dataid[g]])+int(m.mesh_vertnum[m.geom_dataid[g]])].copy()for g in robot if m.geom_type[g]==mj.mjtGeom.mjGEOM_MESH}
    jp=np.zeros((3,m.nv));jr=np.zeros_like(jp)
    def jac(body,pos):mj.mj_jac(m,d,jp,jr,pos,body);return jp@H,jr@H
    def bottom(g):
        R=d.geom_xmat[g].reshape(3,3);p=d.geom_xpos[g].copy();s=m.geom_size[g];kind=m.geom_type[g]
        if kind==mj.mjtGeom.mjGEOM_CAPSULE:p-=np.sign(R[2,2])*s[1]*R[:,2];p[2]-=s[0]
        elif kind==mj.mjtGeom.mjGEOM_SPHERE:p[2]-=s[0]
        elif kind==mj.mjtGeom.mjGEOM_BOX:p+=R@(-np.sign(R[2])*s)
        elif kind==mj.mjtGeom.mjGEOM_CYLINDER:
            # Exact vertical support function for a cylinder, unlike a capsule.
            p[2]-=s[1]*abs(R[2,2])+s[0]*np.sqrt(max(0.,1-R[2,2]**2))
        elif g in localmesh:
            vertices=localmesh[g]@R.T+p;p=vertices[np.argmin(vertices[:,2])]
        else:p[2]-=m.geom_rbound[g]
        return p,jac(int(m.geom_bodyid[g]),p)[0]
    def skew(v):return np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    neutral=m.qpos0.copy();neutral[2]=1.;neutral[3:7]=[1,0,0,0]
    for i,n in enumerate(names):
        if 'hip_pitch' in n or 'ankle_pitch' in n:neutral[qa[i]]=-.18
        elif 'knee_joint'in n:neutral[qa[i]]=.36
        elif 'shoulder_roll'in n:neutral[qa[i]]=.25 if n.startswith('left')else-.25
    d.qpos[:]=neutral;mj.mj_kinematics(m,d);mj.mj_comPos(m,d)
    foot_offsets=[float(d.xpos[bids[side+'_ankle_roll_link'],2]-min(bottom(g)[0][2]for g in fg))for side,fg in zip(('left','right'),footgeoms)]
    standing_base_above_floor=1.-min(bottom(g)[0][2]for fg in footgeoms for g in fg)+.002
    timeline=np.load(a.data/'timeline.npz',allow_pickle=False);times=np.asarray(timeline['pts_seconds'][:a.frame_stop],float);times-=times[0];count=len(times)
    if count<2 or np.any(np.diff(times)<=0):raise ValueError('Invalid timeline')
    allq=[];rawall=[];adaptall=[];repairall=[];supportall=[];seatall=[];fkall=[];metrics=[];rootsall=[];footall=[];yawall=[];seatweightsall=[];seatshiftall=[];forcedfeetall=[]
    sources={}
    for person in(1,2):
        hpath=a.data/f'human/person_{person:02d}/smplx_meshes_world.npz';h=np.load(hpath,allow_pickle=False)
        raw=np.asarray(h['joints_world'][:count],float)@U.T
        sources[f'person{person}']=sha(hpath)
        repaired=raw.copy();mask=np.zeros(count,bool)
        if person==2:mask[63:min(70,count)]=True
        mask|=~np.isfinite(raw).all(axis=(1,2))
        good=np.flatnonzero(~mask)
        if len(good)<2:raise ValueError('Insufficient source observations')
        for j in range(raw.shape[1]):
            for c in range(3):repaired[:,j,c]=np.interp(times,times[good],raw[good,j,c])
        smooth=gaussian_filter1d(repaired,a.source_smoothing_sigma,axis=0)if a.source_smoothing_sigma>0 else repaired.copy()
        lateral=smooth[:,1]-smooth[:,2];forward=np.cross(lateral,[0,0,1]);yaw=gaussian_filter1d(np.unwrap(np.arctan2(forward[:,1],forward[:,0])),1.5)
        floorz=smooth[:,0,:2]@ground[:2]+ground[2]
        source_standing=float(np.quantile(smooth[:,0,2]-floorz,.85))
        height_offset=source_standing-standing_base_above_floor
        thigh=np.linalg.norm(smooth[:,4]-smooth[:,1],axis=1);shin=np.linalg.norm(smooth[:,7]-smooth[:,4],axis=1)
        mj.mj_kinematics(m,d)
        # Morphology is one constant per actor; retain world foot/root XY paths.
        scale=float(np.clip(.76/np.median(thigh+shin),.72,1.08))
        ankle=smooth[:,[7,8]].copy();anklevel=np.linalg.norm(np.gradient(ankle,times,axis=0),axis=-1)
        baseline=np.quantile(ankle[:,:,2]-(ankle[:,:,:2]@ground[:2]+ground[2]),.12,axis=0)
        footheight=ankle[:,:,2]-(ankle[:,:,:2]@ground[:2]+ground[2])-baseline
        expected=(footheight<.055)&(anklevel<.75)
        for k in range(2):expected[:,k]=binary_closing(expected[:,k],iterations=1,border_value=1)
        foot=ankle.copy()
        for k in range(2):
            on=np.r_[False,expected[:,k],False];starts=np.flatnonzero(np.diff(on.astype(int))==1);ends=np.flatnonzero(np.diff(on.astype(int))==-1)
            for start,end in zip(starts,ends):foot[start:end,k,:2]=np.median(ankle[start:end,k,:2],axis=0)
            foot[:,k,2]=foot[:,:,:2][:,k]@ground[:2]+ground[2]+foot_offsets[k]+np.maximum(footheight[:,k],0.)
            foot[expected[:,k],k,2]=foot[expected[:,k],k,:2]@ground[:2]+ground[2]+foot_offsets[k]+.001
        seatids=np.full(count,-1,int);seated=np.zeros(count,bool)
        for f in range(count):
            # Sitting is inferred from pelvis height above an observed seat and
            # horizontal closeness, not hardcoded to initial/final frame ranges.
            for k,seat in enumerate(spec['seats']):
                distance=np.linalg.norm(smooth[f,0,:2]-seat['center_xy_m'])
                dz=smooth[f,0,2]-seat['top_z_m']
                if distance<.45 and .08<dz<.42:seatids[f]=k;seated[f]=True;break
        seatweight=np.zeros(count);seatshift=np.zeros((count,2));forcedfeet=np.zeros((count,2),bool)
        if a.seat_xy_adaptation:
            for f in range(count):
                k=int(np.argmin([np.linalg.norm(smooth[f,0,:2]-seat['center_xy_m'])for seat in spec['seats']]))
                seat=spec['seats'][k];distance=np.linalg.norm(smooth[f,0,:2]-seat['center_xy_m']);dz=smooth[f,0,2]-seat['top_z_m']
                # Purely reference-domain adaptation: support geometry is never
                # enlarged or moved to follow the reconstructed human.
                w=float(np.clip((.60-dz)/.30,0.,1.))*float(np.clip((.75-distance)/.20,0.,1.))
                if w>0:
                    seatids[f]=k;seatweight[f]=w
                    front=np.array([np.cos(yaw[f]),np.sin(yaw[f])])
                    target=np.asarray(seat['center_xy_m'])+.065*front
                    seatshift[f]=w*(target-smooth[f,0,:2])
            seatweight=gaussian_filter1d(seatweight,2.)
            seatshift=gaussian_filter1d(seatshift,2.,axis=0)
            seated=seatweight>.65
        else:seatweight=seated.astype(float)
        if a.continuous_foot_support:
            for f in range(count):
                if seated[f]:
                    forcedfeet[f]=~expected[f];expected[f]=True
                elif not expected[f].any():
                    k=int(np.argmin(footheight[f]+.015*anklevel[f]));expected[f,k]=True;forcedfeet[f,k]=True
            # Recompute the stance locks after filling source contact gaps.
            for k in range(2):
                foot[:,k,:2]=ankle[:,k,:2]
                edge=np.diff(np.r_[False,expected[:,k],False].astype(int))
                for start,end in zip(np.flatnonzero(edge==1),np.flatnonzero(edge==-1)):
                    foot[start:end,k,:2]=np.median(ankle[start:end,k,:2],axis=0)
                foot[:,k,2]=foot[:,k,:2]@ground[:2]+ground[2]+foot_offsets[k]+np.maximum(footheight[:,k],0.)
                foot[expected[:,k],k,2]=foot[expected[:,k],k,:2]@ground[:2]+ground[2]+foot_offsets[k]-.0005
        prev=neutral.copy();personq=[];adapt=[];fk=[];root_targets=[]
        for f in range(count):
            root=smooth[f,0].copy();root[2]-=height_offset
            root[:2]+=seatshift[f]
            if seated[f]:root[2]=max(root[2],spec['seats'][seatids[f]]['top_z_m']+.175)
            desired=root+scale*(smooth[f]-smooth[f,0]);root_targets.append(root.copy())
            quat=np.roll(Rotation.from_euler('z',yaw[f]).as_quat(),1);fixed=prev.copy();fixed[3:7]=quat
            lim=m.jnt_range[jointids[selected]];low=np.r_[root[:2]-.16,max(.30,root[2]-.18),lim[:,0]+1e-6];high=np.r_[root[:2]+.16,root[2]+.18,lim[:,1]-1e-6]
            if f:
                step=6.*(times[f]-times[f-1]);low[3:]=np.maximum(low[3:],prev[qa[selected]]-step);high[3:]=np.minimum(high[3:],prev[qa[selected]]+step)
            seed=np.clip(np.r_[prev[:3],prev[qa[selected]]],low+1e-8,high-1e-8)
            if f==0:
                seed[:3]=root
                if seated[f]:
                    for side in('left','right'):
                        for token,val in(('hip_pitch_joint',-1.4),('knee_joint',1.5),('ankle_pitch_joint',-.1)):
                            ix=names.index(side+'_'+token);seed[3+list(selected).index(ix)]=val
                seed=np.clip(seed,low+1e-8,high-1e-8)
            cache={}
            def evaluate(x):
                if 'x'in cache and np.array_equal(x,cache['x']):return cache['r'],cache['J']
                d.qpos[:]=fixed;d.qpos[:3]=x[:3];d.qpos[qa[selected]]=x[3:];mj.mj_kinematics(m,d);mj.mj_comPos(m,d)
                residual=[];J=[]
                def add(v,j,w):residual.extend(np.asarray(v).reshape(-1)*w);J.extend(np.asarray(j).reshape(-1,nx)*w)
                add(x[:3]-root,TJ,4. if a.seat_xy_adaptation and seated[f]else 3.)
                for name,index,w in bodytargets:
                    body=bids[name];pos=d.xpos[body];add(pos-desired[index],jac(body,pos)[0],w)
                for k,side in enumerate(('left','right')):
                    body=bids[side+'_ankle_roll_link'];pos=d.xpos[body];jj,jrot=jac(body,pos)
                    add(pos-foot[f,k],jj,16. if expected[f,k]else 5.)
                    up=d.xmat[body].reshape(3,3)[:,2];add(up-[0,0,1],-skew(up)@jrot,.7 if expected[f,k]else .08)
                    if expected[f,k]:
                        bp,bj=min([bottom(g)for g in footgeoms[k]],key=lambda z:z[0][2]);n=np.r_[-ground[:2],1.]
                        add([bp@n-ground[2]+.0005],(n@bj).reshape(1,-1),20. if a.continuous_foot_support else 15.)
                if seated[f]:
                    seat=spec['seats'][seatids[f]]
                    for subset in hipgeoms:
                        if subset:
                            bp,bj=min([bottom(g)for g in subset],key=lambda z:z[0][2]);add([bp[2]-seat['top_z_m']+.001],bj[2:3],12.)
                mj.mj_collision(m,d);cr=np.zeros(len(robot));cj=np.zeros((len(robot),nx))
                for c in d.contact[:d.ncon]:
                    g1,g2=int(c.geom1),int(c.geom2)
                    if g1 in rindex and g2 in worldids:g=g1;normal=-np.array(c.frame[:3])
                    elif g2 in rindex and g1 in worldids:g=g2;normal=np.array(c.frame[:3])
                    else:continue
                    k=rindex[g]
                    if c.dist<-.002 and c.dist+.002<cr[k]:cr[k]=c.dist+.002;cj[k]=normal@jac(int(m.geom_bodyid[g]),np.array(c.pos))[0]
                add(cr,cj,70.)
                if f:add(x[3:]-prev[qa[selected]],np.c_[np.zeros((len(selected),3)),np.eye(len(selected))],.04)
                cache.update(x=x.copy(),r=np.array(residual),J=np.array(J));return cache['r'],cache['J']
            fit=least_squares(lambda x:evaluate(x)[0],seed,jac=lambda x:evaluate(x)[1],bounds=(low,high),max_nfev=100 if f==0 else a.max_nfev,ftol=2e-5,xtol=2e-5,gtol=2e-5)
            evaluate(fit.x);q=d.qpos.copy();prev=q.copy();personq.append(q);adapt.append(desired)
            actual=np.array([d.xpos[bids[name]]for name,_,_ in bodytargets]);fk.append(actual)
            penetration=max([max(0.,-float(c.dist))for c in d.contact[:d.ncon]]or[0.])
            row={'actor':person,'frame':f,'time_s':float(times[f]),'nfev':int(fit.nfev),'cost':float(fit.cost),'source_repaired':bool(mask[f]),'seated_inferred':bool(seated[f]),'max_static_penetration_m':penetration,'body_position_rms_m':float(np.sqrt(np.mean((actual-desired[[i for _,i,_ in bodytargets]])**2))),'foot_target_error_m':[float(np.linalg.norm(d.xpos[bids[s+'_ankle_roll_link']]-foot[f,k]))for k,s in enumerate(('left','right'))]}
            metrics.append(row)
            if f%30==0 or f==count-1:print(json.dumps(row),flush=True)
        allq.append(personq);rawall.append(raw);adaptall.append(adapt);repairall.append(mask);supportall.append(expected);seatall.append(seated);fkall.append(fk);rootsall.append(root_targets);footall.append(foot);yawall.append(yaw)
        seatweightsall.append(seatweight);seatshiftall.append(seatshift);forcedfeetall.append(forcedfeet)
    q=np.asarray(allq).transpose(1,0,2);a.output.parent.mkdir(parents=True,exist_ok=True)
    payload={'qpos':q,'qpos_p1':q[:,0],'qpos_p2':q[:,1],'time_s':times*a.time_scale,'source_time_s':times,'frame_ids':np.asarray(timeline['frame_ids'][:count]),'dof_names':np.asarray(names),'source_to_sim_U':U,'raw_source_joints_world_zup':np.asarray(rawall).transpose(1,0,2,3),'adapted_joint_targets':np.asarray(adaptall).transpose(1,0,2,3),'source_repaired_frame_mask':np.asarray(repairall).T,'source_hand_valid':np.zeros((count,2,2),bool),'foot_support_expected':np.asarray(supportall).transpose(1,0,2),'seat_support_expected':np.asarray(seatall).T,'foot_target_ankle_world':np.asarray(footall).transpose(1,0,2,3),'root_target_world':np.asarray(rootsall).transpose(1,0,2),'root_yaw_world_rad':np.asarray(yawall).T,'fk_body_target_positions':np.asarray(fkall).transpose(1,0,2,3),'contact_inferred':np.asarray(True),'reference_adapted':np.asarray(True)}
    if not np.isfinite(q).all():raise ValueError('Nonfinite reference')
    payload.update(seat_xy_reference_adaptation_m=np.asarray(seatshiftall).transpose(1,0,2),seat_support_blend_weight=np.asarray(seatweightsall).T,foot_support_filled_inference=np.asarray(forcedfeetall).transpose(1,0,2),seat_xy_adaptation_enabled=np.asarray(a.seat_xy_adaptation),continuous_foot_support_inference_enabled=np.asarray(a.continuous_foot_support))
    np.savez_compressed(a.output,**payload)
    report={'schema':'two_sitting_retarget.v1','frames':count,'actors':2,'source_duration_s':float(times[-1]),'execution_duration_s':float(times[-1]*a.time_scale),'time_scale':a.time_scale,'model_sha256':sha(model_path),'reference_sha256':sha(a.output),'source_hashes':sources,'source_hand_observed':False,'physics_passed':False,'explicit_repair':'person2 frames63-69 interpolated from surrounding source joint observations before smoothing','source_raw_preserved':True,'max_static_penetration_m':max(r['max_static_penetration_m']for r in metrics),'median_body_rms_m':float(np.median([r['body_position_rms_m']for r in metrics])),'rows':metrics,'limitations':['Kinematic reference only; dynamic physical sequence requires independent rollout and replay audit.','Seat/foot contacts and source morphology adaptation are inferred; no measured force ground truth.','Robot yaw follows observed pelvis hip direction, root roll/pitch fixed upright in this baseline.']}
    report.update(seat_xy_adaptation_enabled=a.seat_xy_adaptation,continuous_foot_support_inference_enabled=a.continuous_foot_support,max_seated_xy_reference_shift_m=float(np.max(np.linalg.norm(payload['seat_xy_reference_adaptation_m'],axis=-1))),additional_inferred_foot_contact_frames=int(payload['foot_support_filled_inference'].sum()))
    a.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    a.output.with_suffix('.generator.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:v for k,v in report.items()if k!='rows'}),flush=True)


if __name__=='__main__':main()
