"""Numeric SMPL-X to Vita reference with explicit, inferred terrain contacts."""
from pathlib import Path
import argparse,hashlib,json


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('data/rgb_145317_sim_handoff'))
    p.add_argument('--model',type=Path,default=Path('outputs/rgb145_pipeline/assets/vita_inference_O6_training.xml'))
    p.add_argument('--support',type=Path,default=Path('outputs/rgb145_pipeline/support_geometry.json'))
    p.add_argument('--weights',type=Path,default=Path('outputs/rgb145_pipeline/numeric_model/smplx_numeric_weights.npz'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-nfev',type=int,default=35)
    p.add_argument('--frame-stop',type=int,default=350)
    p.add_argument('--max-joint-speed',type=float,default=6.)
    p.add_argument('--minimum-seated-root-above-seat',type=float,default=.19)
    p.add_argument('--terrain-foot-projection',action='store_true')
    p.add_argument('--max-root-speed',type=float,default=1.5)
    p.add_argument('--max-root-acceleration',type=float,default=8.)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    import numpy as np,mujoco as mj
    from scipy.optimize import least_squares
    from scipy.ndimage import gaussian_filter1d,binary_closing
    from scipy.spatial.transform import Rotation
    from rgb145_support import height_at_xy,inside_convex_polygon
    U=np.array([[1.,0,0],[0,0,-1],[0,1,0]])
    h=np.load(a.source/'human/person_01/smplx_meshes_world.npz',allow_pickle=False)
    timeline=np.load(a.source/'timeline.npz',allow_pickle=False)
    rp=np.load(a.source/'human/person_01/root_poses_world.npz',allow_pickle=False)
    count=min(a.frame_stop,len(h['vertices_world']))
    time=np.asarray(h['pts_seconds'][:count],float);time-=time[0]
    rawj=np.asarray(h['joints_world'][:count])@U.T
    verts=np.asarray(h['vertices_world'][:count])@U.T
    transforms=np.asarray(rp['world_from_pelvis'][:count]).copy()
    transforms[:,:3,:]=np.einsum('ij,tjk->tik',U,transforms[:,:3,:])
    weights=np.load(a.weights,allow_pickle=False)['weights']
    footidx=[np.flatnonzero(weights[:,ank]+weights[:,toe]>.5)for ank,toe in ((7,10),(8,11))]
    low=np.stack([np.quantile(verts[:,idx,2],.015,axis=1)for idx in footidx],axis=1)
    spec=json.loads(a.support.read_text())
    ground=np.asarray(spec['ground']['z_eq_ax_by_c'])
    baseline=np.median(low[:min(count,80)]-(rawj[:min(count,80),[7,8],:2]@ground[:2]+ground[2]),axis=0)
    sole=gaussian_filter1d(low-baseline,1.2,axis=0)
    joints=gaussian_filter1d(rawj,1.2,axis=0)
    lateral=joints[:,1]-joints[:,2];fw=np.cross(lateral,[0,0,1]);yaw=gaussian_filter1d(np.unwrap(np.arctan2(fw[:,1],fw[:,0])),2.)
    vsole=np.gradient(sole,time,axis=0)
    surface=np.zeros((count,2));surface_names=np.full((count,2),'ground_support',dtype='<U40')
    correction=np.zeros((count,2,2))
    expected=np.zeros((count,2),bool);confidence=np.full((count,2),.7)
    def project_patch(xy,polygon):
        poly=np.asarray(polygon);edges=np.roll(poly,-1,axis=0)-poly;normals=np.c_[-edges[:,1],edges[:,0]];normals/=np.maximum(np.linalg.norm(normals,axis=1)[:,None],1e-12)
        if np.sum(poly[:,0]*np.roll(poly[:,1],-1)-poly[:,1]*np.roll(poly[:,0],-1))<0:normals=-normals
        bound=np.sum(poly*normals,axis=1)+.035;pt=xy.copy()
        for _ in range(30):
            delta=bound-normals@pt
            if max(delta)<1e-7:break
            for normal,b in zip(normals,bound):pt+=max(0.,b-normal@pt)*normal
        return pt
    for f in range(count):
        front=np.array([np.cos(yaw[f]),np.sin(yaw[f])]);left=np.array([-front[1],front[0]])
        for k,ank in enumerate((7,8)):
            xy=joints[f,ank,:2];samples=np.array([xy,xy+.13*front,xy-.08*front,xy+.05*left,xy-.05*left])
            candidates=[(float(xy@ground[:2]+ground[2]),'ground_support',xy.copy(),0.)]
            for item in spec['supports']:
                if a.terrain_foot_projection or np.any(inside_convex_polygon(samples,item['xy_polygon_m'])):
                    projected=project_patch(xy,item['xy_polygon_m'])if a.terrain_foot_projection else xy.copy();distance=float(np.linalg.norm(projected-xy))
                    if distance>.55:continue
                    b=np.asarray(item['top_z_eq_ax_by_c']);candidates.append((float(projected@b[:2]+b[2]),item['name'],projected,distance))
            best=min(candidates,key=lambda x:abs(x[0]-sole[f,k])+.30*x[3])
            surface[f,k],surface_names[f,k]=best[:2];correction[f,k]=best[2]-xy
            expected[f,k]=abs(sole[f,k]-best[0])<.065 and abs(vsole[f,k])<.45
        if not expected[f].any():
            gap=abs(sole[f]-surface[f])+.08*abs(vsole[f]);k=int(np.argmin(gap))
            if gap[k]<.18:expected[f,k]=True;confidence[f,k]=.55
    for k in range(2):expected[:,k]=binary_closing(expected[:,k],iterations=2,border_value=1)
    correction=gaussian_filter1d(correction,3.,axis=0)
    root_correction=gaussian_filter1d(np.sum(correction*expected[:,:,None],axis=1)/np.maximum(expected.sum(1)[:,None],1),3.,axis=0)
    m=mj.MjModel.from_xml_path(str(a.model.resolve()));d=mj.MjData(m)
    jids=np.array([j for j in range(m.njnt)if m.jnt_type[j]==mj.mjtJoint.mjJNT_HINGE])
    jids=jids[np.argsort(m.jnt_qposadr[jids])]
    names=[mj.mj_id2name(m,mj.mjtObj.mjOBJ_JOINT,int(j))for j in jids]
    qa=m.jnt_qposadr[jids];va=m.jnt_dofadr[jids]
    keep=np.array([not n.startswith(('lh_','rh_','neck_'))for n in names]);body=np.flatnonzero(keep)
    xqa=np.r_[np.arange(3),qa[body]];xva=np.r_[np.arange(3),va[body]];nx=len(xqa)
    bids={n:mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,n)for n in ['base_link','left_knee_link','right_knee_link','left_ankle_roll_link','right_ankle_roll_link','left_shoulder_pitch_link','right_shoulder_pitch_link','left_elbow_link','right_elbow_link','left_wrist_pitch_link','right_wrist_pitch_link','neck_yaw_link']}
    if min(bids.values())<0 or len(names)!=52:raise ValueError('Vita body contract mismatch')
    targets=[('left_knee_link',4,.8),('right_knee_link',5,.8),('left_shoulder_pitch_link',16,1.),('right_shoulder_pitch_link',17,1.),('left_elbow_link',18,1.5),('right_elbow_link',19,1.5),('left_wrist_pitch_link',20,1.),('right_wrist_pitch_link',21,1.),('neck_yaw_link',12,.6)]
    fg=[[g for g in range(m.ngeom)if (mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,g)or'').startswith('collision_foot_'+s+'_')]for s in ('l','r')]
    hips=[[g for g in range(m.ngeom)if any(tag in(mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,g)or'')for tag in tags)]for tags in [('collision_l_hip_roll_cylinder','collision_l_hip_front_roll_cylinder'),('collision_r_hip_roll_cylinder','collision_r_hip_front_roll_cylinder')]]
    robot=[g for g in range(m.ngeom)if m.geom_bodyid[g]>0 and m.geom_contype[g]];rindex={g:k for k,g in enumerate(robot)}
    support_ids={g for g in range(m.ngeom)if m.geom_bodyid[g]==0 and m.geom_contype[g]}
    localmesh={g:m.mesh_vert[int(m.mesh_vertadr[m.geom_dataid[g]]):int(m.mesh_vertadr[m.geom_dataid[g]])+int(m.mesh_vertnum[m.geom_dataid[g]])].copy()for g in robot if m.geom_type[g]==mj.mjtGeom.mjGEOM_MESH}
    jp=np.zeros((3,m.nv));jr=np.zeros_like(jp)
    def point(b,pos):mj.mj_jac(m,d,jp,jr,pos,b);return jp[:,xva].copy(),jr[:,xva].copy()
    def bottom(g):
        R=d.geom_xmat[g].reshape(3,3);v=d.geom_xpos[g].copy();sz=m.geom_size[g];typ=m.geom_type[g]
        if typ==mj.mjtGeom.mjGEOM_CAPSULE:v-=np.sign(R[2,2])*sz[1]*R[:,2];v[2]-=sz[0]
        elif typ==mj.mjtGeom.mjGEOM_MESH:
            pts=localmesh[g]@R.T+v;v=pts[np.argmin(pts[:,2])]
        elif typ==mj.mjtGeom.mjGEOM_SPHERE:v[2]-=sz[0]
        elif typ==mj.mjtGeom.mjGEOM_BOX:v+=R@(-np.sign(R[2])*sz)
        else:v[2]-=m.geom_rbound[g]
        return v,point(int(m.geom_bodyid[g]),v)[0]
    def skew(v):return np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    seed=m.qpos0.copy()
    for k,n in enumerate(names):
        if 'hip_pitch' in n or 'ankle_pitch' in n:seed[qa[k]]=-.2
        if 'knee_joint'in n:seed[qa[k]]=.4
        if 'shoulder_roll'in n:seed[qa[k]]=.42 if n.startswith('left')else-.42
    initial_height=float(seed[2]);root_offset=float(joints[0,0,2]-initial_height)
    solutions=[];rows=[];foot_desired=[];root_targets=[];actualfoot=[]
    limits=m.jnt_range[jids[body]]
    for f in range(count):
        sit=float(np.clip((time[f]-9.0)/1.2,0,1));sit=sit*sit*(3-2*sit)
        root_target=joints[f,0].copy();root_target[2]-=(1-sit)*root_offset+sit*.025
        root_target[:2]+=root_correction[f]
        desired=root_target+.84*(joints[f]-joints[f,0])
        feet=[]
        for k,ank in enumerate((7,8)):
            v=joints[f,ank].copy();v[:2]=joints[f,0,:2]+.90*(v[:2]-joints[f,0,:2])+correction[f,k]
            v[2]=(surface[f,k]-.0002 if expected[f,k]else max(sole[f,k],surface[f,k]+.015))+.070
            feet.append(v)
        feet=np.asarray(feet);root_targets.append(root_target.copy());foot_desired.append(feet.copy())
        fixed=seed.copy();fixed[3:7]=[np.cos(yaw[f]/2),0,0,np.sin(yaw[f]/2)]
        lowbound=np.r_[root_target[:2]-.30,max(.40,root_target[2]-.20),limits[:,0]+1e-5]
        highbound=np.r_[root_target[:2]+.30,root_target[2]+.20,limits[:,1]-1e-5]
        if sit>.8:
            candidates=[item for item in spec['supports']if item['kind']=='seat'and inside_convex_polygon(root_target[:2],item['xy_polygon_m'])]
            if candidates:
                item=min(candidates,key=lambda it:abs(it['mode_height_m']-.436));b=np.asarray(item['top_z_eq_ax_by_c'])
                lowbound[2]=max(lowbound[2],root_target[:2]@b[:2]+b[2]+a.minimum_seated_root_above_seat)
        if f:
            delta=float(time[f]-time[f-1]);lowbound[3:]=np.maximum(lowbound[3:],seed[xqa][3:]-a.max_joint_speed*delta);highbound[3:]=np.minimum(highbound[3:],seed[xqa][3:]+a.max_joint_speed*delta)
            velocity=(solutions[-1][:3]-solutions[-2][:3])/(time[f-1]-time[f-2])if f>1 else np.zeros(3)
            step_lo=np.maximum(-a.max_root_speed*delta,velocity*delta-a.max_root_acceleration*delta*delta)
            step_hi=np.minimum(a.max_root_speed*delta,velocity*delta+a.max_root_acceleration*delta*delta)
            temporal_lo=seed[:3]+step_lo;temporal_hi=seed[:3]+step_hi
            merged_lo=np.maximum(lowbound[:3],temporal_lo);merged_hi=np.minimum(highbound[:3],temporal_hi)
            invalid=merged_lo>=merged_hi
            lowbound[:3]=np.where(invalid,temporal_lo,merged_lo);highbound[:3]=np.where(invalid,temporal_hi,merged_hi)
        cache={}
        def evaluate(x):
            if 'x'in cache and np.array_equal(x,cache['x']):return cache['r'],cache['J']
            d.qpos[:]=fixed;d.qpos[xqa]=x;mj.mj_kinematics(m,d);mj.mj_comPos(m,d)
            rr=[];JJ=[]
            def add(v,J,w):rr.extend(np.asarray(v).reshape(-1)*w);JJ.extend(np.asarray(J).reshape(-1,nx)*w)
            add(x[:3]-root_target,np.c_[np.eye(3),np.zeros((3,nx-3))],1.2)
            for name,idx,w in targets:
                b=bids[name];pos=d.xpos[b];add(pos-desired[idx],point(b,pos)[0],w)
            for k,side in enumerate(('left','right')):
                b=bids[side+'_ankle_roll_link'];pos=d.xpos[b];J,Jr=point(b,pos)
                add(pos-feet[k],J,30. if expected[f,k]else 12.)
                up=d.xmat[b].reshape(3,3)[:,2];add(up-[0,0,1],-skew(up)@Jr,1.0 if expected[f,k]else .7)
                if expected[f,k]:
                    bp,bj=min([bottom(g)for g in fg[k]],key=lambda q:q[0][2]);beta=ground if surface_names[f,k]=='ground_support'else np.array(next(s['top_z_eq_ax_by_c']for s in spec['supports']if s['name']==surface_names[f,k]))
                    normal=np.r_[-beta[:2],1.];add([bp@normal-beta[2]+.0002],(normal@bj).reshape(1,-1),15.)
            if sit:
                for subset in hips:
                    bp,bj=min([bottom(g)for g in subset],key=lambda q:q[0][2]);seats=[s for s in spec['supports']if s['kind']=='seat'and inside_convex_polygon(bp[:2],s['xy_polygon_m'])]
                    if seats:
                        s=min(seats,key=lambda q:abs(bp[2]-(bp[:2]@np.array(q['top_z_eq_ax_by_c'])[:2]+q['top_z_eq_ax_by_c'][2])));b=np.asarray(s['top_z_eq_ax_by_c']);normal=np.r_[-b[:2],1.];add([bp@normal-b[2]+.001],(normal@bj).reshape(1,-1),12.*sit)
                    else:add([0.],np.zeros((1,nx)),0.)
            mj.mj_collision(m,d);cr=np.zeros(len(robot));cj=np.zeros((len(robot),nx))
            for c in d.contact[:d.ncon]:
                g1,g2=int(c.geom1),int(c.geom2)
                if g1 in rindex and g2 in support_ids:g=g1;normal=-np.array(c.frame[:3])
                elif g2 in rindex and g1 in support_ids:g=g2;normal=np.array(c.frame[:3])
                else:continue
                k=rindex[g]
                if c.dist<-.001 and c.dist+.001<cr[k]:cr[k]=c.dist+.001;cj[k]=normal@point(int(m.geom_bodyid[g]),np.array(c.pos))[0]
            add(cr,cj,50.)
            if f:add(x[3:]-seed[xqa][3:],np.c_[np.zeros((nx-3,3)),np.eye(nx-3)],.025)
            cache.update(x=x.copy(),r=np.array(rr),J=np.array(JJ));return cache['r'],cache['J']
        fit=least_squares(lambda x:evaluate(x)[0],np.clip(seed[xqa],lowbound+1e-7,highbound-1e-7),jac=lambda x:evaluate(x)[1],bounds=(lowbound,highbound),max_nfev=100 if f==0 else a.max_nfev,ftol=2e-5,xtol=2e-5,gtol=2e-5)
        evaluate(fit.x);seed=d.qpos.copy();solutions.append(seed.copy())
        actualfoot.append([d.xpos[bids[side+'_ankle_roll_link']].copy()for side in ('left','right')])
        pen=max([max(0.,-float(c.dist))for c in d.contact[:d.ncon]]or[0.])
        rows.append({'frame':f,'cost':float(fit.cost),'nfev':int(fit.nfev),'max_penetration_m':pen,'foot_error_m':np.linalg.norm(np.array(actualfoot[-1])-feet,axis=1).tolist()})
        if f%25==0:print(json.dumps(rows[-1]),flush=True)
    q=np.asarray(solutions);a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output,qpos=q,dof_names=np.asarray(names),time_s=time,source_time_s=time.copy(),frame_ids=h['frame_ids'][:count],foot_support_expected=expected,foot_support_geom_name=surface_names,foot_support_confidence=confidence,source_joints_world_zup=rawj,source_root_world_from_pelvis_zup=transforms,source_smplx_frame_valid=h['valid'][:count],source_foot_surface_low_zup=low,source_foot_surface_z_bias=baseline,foot_target_ankle_world=np.asarray(foot_desired),foot_actual_ankle_world=np.asarray(actualfoot),root_target_world=np.asarray(root_targets),source_to_sim_U=U,inferred_contact=np.asarray(True),terrain_foot_horizontal_correction_m=correction,terrain_root_horizontal_correction_m=root_correction)
    report={'schema':'rgb145.reference.v1','frames':count,'duration_s':float(time[-1]),'reference_sha256':sha(a.output),'model_sha256':sha(a.model),'source_smplx_mesh_sha256':sha(a.source/'human/person_01/smplx_meshes_world.npz'),'source_timeline_sha256':sha(a.source/'timeline.npz'),'source_foot_vertical_bias_m':baseline.tolist(),'standing_root_height_offset_m':root_offset,'max_static_penetration_m':max(r['max_penetration_m']for r in rows),'max_ankle_target_error_m':np.max(np.linalg.norm(np.asarray(actualfoot)-np.asarray(foot_desired),axis=-1),axis=0).tolist(),'body_reference_peak_velocity_rad_s':float(np.max(abs(np.diff(q[:,qa[body]],axis=0)/np.diff(time)[:,None]))),'raw_source_preserved':True,'source_hands_observed':False,'physics_passed':False,'rows':rows}
    a.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');a.output.with_suffix('.generator.py').write_bytes(Path(__file__).read_bytes());print(json.dumps({k:v for k,v in report.items()if k!='rows'}),flush=True)


if __name__=='__main__':main()
