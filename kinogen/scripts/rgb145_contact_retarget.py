"""Contact-balanced SMPL-X to Vita retarget; a kinematic reference, not a physics rollout."""
from pathlib import Path
import argparse,hashlib,json


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def project_swing_targets(targets,expected,expected_hips,max_reach):
    """Precompute free-swing retraction; do not make a target depend on IK variables."""
    import numpy as np
    original=np.asarray(targets,float);adjusted=original.copy();fade=np.zeros(expected.shape)
    if max_reach<=0:return adjusted,fade
    for side in range(2):
        swing=~expected[:,side];edge=np.diff(np.r_[0,swing.astype(int),0]);starts=np.flatnonzero(edge==1);ends=np.flatnonzero(edge==-1)
        for lo,hi in zip(starts,ends):
            index=np.arange(hi-lo);u=np.minimum(1.,np.minimum(index,hi-lo-1-index)/3.);weight=u*u*(3-2*u);fade[lo:hi,side]=weight
            vector=original[lo:hi,side]-expected_hips[lo:hi,side];distance=np.linalg.norm(vector,axis=1);scale=np.minimum(1.,max_reach/np.maximum(distance,1e-12))
            projected=expected_hips[lo:hi,side]+vector*scale[:,None]
            adjusted[lo:hi,side]+=weight[:,None]*(projected-original[lo:hi,side])
    if not np.array_equal(adjusted[expected],original[expected]):raise AssertionError('Swing retraction changed a stance target')
    return adjusted,fade


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('data/rgb_145317_sim_handoff'))
    p.add_argument('--model',type=Path,default=Path('outputs/rgb145_pipeline/physics_sonic_v2_full/model.xml'),help='Authoritative effective controller plant, including its actual joint ranges. Explicit --model takes priority; no fallback to older asset ranges.')
    p.add_argument('--support',type=Path,default=Path('outputs/rgb145_pipeline/support_geometry.json'))
    p.add_argument('--weights',type=Path,default=Path('outputs/rgb145_pipeline/numeric_model/smplx_numeric_weights.npz'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-nfev',type=int,default=35)
    p.add_argument('--frame-stop',type=int,default=350)
    p.add_argument('--max-joint-speed',type=float,default=6.)
    p.add_argument('--greedy-source-rate-bounds',action='store_true',help='Optional legacy per-frame source-clock joint/RP rate bounds; off by default. Execution speed is validated by downstream retiming.')
    p.add_argument('--minimum-seated-root-above-seat',type=float,default=.19)
    p.add_argument('--root-height-lower-adaptation',type=float,default=.25,help='Allowed geometric root-height lowering from its source target, in metres; floor and seat bounds still apply')
    p.add_argument('--root-height-upper-adaptation',type=float,default=.30,help='Allowed geometric root-height raising from its source target, in metres')
    p.add_argument('--swing-leg-max-reach',type=float,default=0.,help='Optional fixed-target hip-to-swing-ankle reach in metres;0 disables. Only free swing is projected with a3frame fade at each endpoint.')
    p.add_argument('--stance-knee-load-weight',type=float,default=0.,help='Optional analytic static knee-load proxy cost;0 disables optimization cost')
    p.add_argument('--stance-knee-load-target-ratio',type=float,default=.8,help='Soft target for raw knee actuator demand / actual signed force capacity; no static-friction assistance')
    p.add_argument('--terrain-foot-projection',action='store_true')
    p.add_argument('--max-root-speed',type=float,default=1.5)
    p.add_argument('--max-root-acceleration',type=float,default=8.)
    p.add_argument('--root-speed-cost-weight',type=float,default=.2,help='Soft excess-speed penalty in the source clock; not a hard bound')
    p.add_argument('--root-acceleration-cost-weight',type=float,default=.002,help='Soft excess-acceleration penalty in the source clock; not a hard bound')
    p.add_argument('--com-support-margin',type=float,default=.01)
    p.add_argument('--com-support-weight',type=float,default=12.)
    p.add_argument('--wrist-neutral-weight',type=float,default=.65)
    p.add_argument('--root-yaw-adaptation-window',type=float,default=.70)
    p.add_argument('--root-yaw-source-prior-weight',type=float,default=.2)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if not a.model.is_file():p.error('Effective model not found; provide --model explicitly. Older asset joint ranges are not used as a fallback.')
    if a.root_height_lower_adaptation<0 or a.root_height_upper_adaptation<0:p.error('Root-height adaptation allowances must be nonnegative')
    if a.swing_leg_max_reach<0:p.error('Swing-leg maximum reach must be nonnegative')
    if a.stance_knee_load_weight<0 or a.stance_knee_load_target_ratio<0:p.error('Knee-load weight and target ratio must be nonnegative')
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
    from rgb145_knee_load import KneeLoadProxy
    knee_load_proxy=KneeLoadProxy(m)
    jids=np.array([j for j in range(m.njnt)if m.jnt_type[j]==mj.mjtJoint.mjJNT_HINGE])
    jids=jids[np.argsort(m.jnt_qposadr[jids])]
    names=[mj.mj_id2name(m,mj.mjtObj.mjOBJ_JOINT,int(j))for j in jids]
    qa=m.jnt_qposadr[jids];va=m.jnt_dofadr[jids]
    keep=np.array([not n.startswith(('lh_','rh_','neck_'))for n in names]);body=np.flatnonzero(keep)
    nx=6+len(body);H=np.zeros((m.nv,nx));H[:3,:3]=np.eye(3);H[va[body],np.arange(5,nx-1)]=1.
    translation_jac=np.c_[np.eye(3),np.zeros((3,nx-3))]
    yaw_jac=np.zeros((1,nx));yaw_jac[0,-1]=1.
    bids={n:mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,n)for n in ['base_link','left_knee_link','right_knee_link','left_ankle_roll_link','right_ankle_roll_link','left_shoulder_pitch_link','right_shoulder_pitch_link','left_elbow_link','right_elbow_link','left_wrist_pitch_link','right_wrist_pitch_link','neck_yaw_link']}
    if min(bids.values())<0 or len(names)!=52:raise ValueError('Vita body contract mismatch')
    targets=[('left_knee_link',4,.8),('right_knee_link',5,.8),('left_shoulder_pitch_link',16,1.),('right_shoulder_pitch_link',17,1.),('left_elbow_link',18,1.5),('right_elbow_link',19,1.5),('left_wrist_pitch_link',20,1.),('right_wrist_pitch_link',21,1.),('neck_yaw_link',12,.6)]
    fg=[[g for g in range(m.ngeom)if (mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,g)or'').startswith('collision_foot_'+s+'_')]for s in ('l','r')]
    # Derive conservative sole envelopes from the actual Vita collider ABI, not a guessed disk.
    neutral=mj.MjData(m);neutral.qpos[7:]=0.;mj.mj_kinematics(m,neutral)
    footprints=[];sole_offsets=[]
    for side,geoms in zip(('left','right'),fg):
        b=bids[side+'_ankle_roll_link'];R=neutral.xmat[b].reshape(3,3);origin=neutral.xpos[b];points=[];bottoms=[]
        for g in geoms:
            center=R.T@(neutral.geom_xpos[g]-origin);rot=R.T@neutral.geom_xmat[g].reshape(3,3);typ=m.geom_type[g];radius=float(m.geom_size[g,0])
            if typ==mj.mjtGeom.mjGEOM_CAPSULE:ends=[center-m.geom_size[g,1]*rot[:,2],center+m.geom_size[g,1]*rot[:,2]]
            elif typ==mj.mjtGeom.mjGEOM_SPHERE:ends=[center]
            else:raise ValueError('Expected canonical training capsule/sphere foot geometry: '+str(typ))
            for end in ends:
                theta=np.arange(16)*2*np.pi/16;points.extend(end[:2]+(radius/np.cos(np.pi/16))*np.c_[np.cos(theta),np.sin(theta)]);bottoms.append(end[2]-radius)
        footprint=np.asarray(points);span=np.ptp(footprint,axis=0)
        if span[0]<1.2*span[1]or footprint[:,0].max()<=abs(footprint[:,0].min()):raise ValueError('Vita sole +X forefoot convention not confirmed')
        footprints.append(footprint);sole_offsets.append(float(-min(bottoms)))
    from rgb145_contact_plan import build_plan,rotate_xy,convex_hull,halfspaces
    legacy_expected=expected.copy()
    planned_feet,foot_yaw,expected,confidence,contact_audit=build_plan(joints,sole,vsole,time,surface,surface_names,legacy_expected,footprints,sole_offsets,spec,body_yaw=yaw)
    pre_no_flight_expected=np.asarray(contact_audit.pop('pre_no_flight_expected'),bool);no_flight_adapted_mask=np.asarray(contact_audit.pop('no_flight_adapted_mask'),bool)
    source_toe_yaw_before_adaptation=np.asarray(contact_audit.pop('source_toe_yaw_before_robot_adaptation_rad'));turn_yaw_override_mask=np.asarray(contact_audit.pop('turn_yaw_override_mask'),bool);turn_lift_mask=np.asarray(contact_audit.pop('turn_lift_mask'),bool);pre_turn_expected=np.asarray(contact_audit.pop('pre_turn_expected'),bool)
    correction=planned_feet[:,:,:2]-joints[:,[7,8],:2]
    root_correction=gaussian_filter1d(np.sum(correction*expected[:,:,None],axis=1)/np.maximum(expected.sum(1)[:,None],1),3.,axis=0)
    outside=0
    for f in range(count):
        for k in range(2):
            item=next((s for s in spec['supports']if s['name']==surface_names[f,k]),None)
            if expected[f,k]and item:outside+=int(not np.all(inside_convex_polygon(planned_feet[f,k,:2]+rotate_xy(footprints[k],foot_yaw[f,k]),item['xy_polygon_m'])))
    contact_audit['stance_footprints_outside_assigned_polygon_count']=outside
    if outside:raise ValueError('Planned support footprint outside its named collider')
    hips=[[g for g in range(m.ngeom)if any(tag in(mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,g)or'')for tag in tags)]for tags in [('collision_l_hip_roll_cylinder','collision_l_hip_front_roll_cylinder'),('collision_r_hip_roll_cylinder','collision_r_hip_front_roll_cylinder')]]
    robot=[g for g in range(m.ngeom)if m.geom_bodyid[g]>0 and m.geom_contype[g]];rindex={g:k for k,g in enumerate(robot)}
    support_ids={g for g in range(m.ngeom)if m.geom_bodyid[g]==0 and m.geom_contype[g]}
    localmesh={g:m.mesh_vert[int(m.mesh_vertadr[m.geom_dataid[g]]):int(m.mesh_vertadr[m.geom_dataid[g]])+int(m.mesh_vertnum[m.geom_dataid[g]])].copy()for g in robot if m.geom_type[g]==mj.mjtGeom.mjGEOM_MESH}
    jp=np.zeros((3,m.nv));jr=np.zeros_like(jp)
    def point(b,pos):mj.mj_jac(m,d,jp,jr,pos,b);return jp@H,jr@H
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
    wrist_targets={};wrist_sign_audit={}
    for side,prefix in (('left','lh_'),('right','rh_')):
        wb=bids[side+'_wrist_pitch_link'];hb=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,prefix+'hand_base_link')
        if hb<0:raise ValueError('Missing O6 palm frame')
        palm_long=neutral.xmat[wb].reshape(3,3).T@neutral.xmat[hb].reshape(3,3)[:,2]
        j=mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,side+'_wrist_pitch_joint');derivative_z=float(np.cross(m.jnt_axis[j],palm_long)[2])
        if abs(derivative_z)<.2:raise ValueError('Cannot confirm wrist pitch droop sign from palm geometry')
        wrist_sign_audit[side]={'palm_long_axis_in_wrist_frame':palm_long.tolist(),'positive_pitch_palm_dz':derivative_z}
        for axis in ('roll','yaw','pitch'):wrist_targets[side+'_wrist_'+axis+'_joint']=float(-np.sign(derivative_z)*.08)if axis=='pitch'else 0.
    for k,n in enumerate(names):
        if n in wrist_targets:seed[qa[k]]=wrist_targets[n]
    # Keep the unobserved fingers relaxed, and solve every passive follower from its unchanged polynomial.
    followers=set(map(int,m.eq_obj1id));finger_drivers=[int(j)for j,n in zip(jids,names)if n.startswith(('lh_','rh_'))and int(j)not in followers]
    for j in finger_drivers:
        desired=.04;lo,hi=m.jnt_range[j];grid=np.unique(np.r_[np.clip(desired,lo,hi),np.linspace(lo,hi,401)]);valid=np.ones(len(grid),bool)
        for eq in np.flatnonzero(m.eq_obj2id==j):
            follower=int(m.eq_obj1id[eq]);value=m.qpos0[m.jnt_qposadr[follower]]+np.polynomial.polynomial.polyval(grid-m.qpos0[m.jnt_qposadr[j]],m.eq_data[eq,:5]);flo,fhi=m.jnt_range[follower];valid&=(value>=flo-1e-9)&(value<=fhi+1e-9)
        if not valid.any():raise ValueError('No joint-limit-consistent relaxed finger coupling for '+str(j))
        candidate=grid[valid];seed[m.jnt_qposadr[j]]=candidate[np.argmin(abs(candidate-desired))]
    for _ in range(m.neq):
        for eq in range(m.neq):
            f1,f2=int(m.eq_obj1id[eq]),int(m.eq_obj2id[eq]);driver_value=0. if f2<0 else seed[m.jnt_qposadr[f2]]-m.qpos0[m.jnt_qposadr[f2]]
            seed[m.jnt_qposadr[f1]]=m.qpos0[m.jnt_qposadr[f1]]+np.polynomial.polynomial.polyval(driver_value,m.eq_data[eq,:5])
    eqcheck=mj.MjData(m);eqcheck.qpos[:]=seed;mj.mj_forward(m,eqcheck)
    eqmask=eqcheck.efc_type[:eqcheck.nefc]==int(mj.mjtConstraint.mjCNSTR_EQUALITY)
    initial_finger_eq_residual=float(np.max(abs(eqcheck.efc_pos[:eqcheck.nefc][eqmask])))if np.any(eqmask)else 0.
    if initial_finger_eq_residual>1e-7:raise ValueError('Unchanged O6 polynomial initial-state residual '+str(initial_finger_eq_residual))
    wrist_cols=np.asarray([i for i,k in enumerate(body)if names[k]in wrist_targets]);wrist_goal=np.asarray([wrist_targets[names[body[i]]]for i in wrist_cols]);wrist_jac=np.zeros((len(wrist_cols),nx));wrist_jac[np.arange(len(wrist_cols)),5+wrist_cols]=1.
    initial_height=float(seed[2]);root_offset=float(joints[0,0,2]-initial_height)
    # These targets are fixed before least_squares, preserving the analytic residual Jacobians.
    preplanned_roots=[]
    for f in range(count):
        sit=float(np.clip((time[f]-9.0)/1.2,0,1));sit=sit*sit*(3-2*sit)
        root_target=joints[f,0].copy();root_target[2]-=(1-sit)*root_offset+sit*.025;root_target[:2]+=root_correction[f];preplanned_roots.append(root_target)
    preplanned_roots=np.asarray(preplanned_roots)
    hip_ids=np.asarray([mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,side+'_hip_pitch_link')for side in('left','right')])
    if np.any(hip_ids<0):raise ValueError('Missing true-model hip-pitch FK origins')
    base=bids['base_link'];base_R=neutral.xmat[base].reshape(3,3);hip_offsets=(neutral.xpos[hip_ids]-neutral.xpos[base])@base_R
    nominal_hip_ankle_length=np.asarray([np.linalg.norm(neutral.xpos[bids[side+'_ankle_roll_link']]-neutral.xpos[hip])for side,hip in zip(('left','right'),hip_ids)])
    heading_R=Rotation.from_euler('Z',yaw).as_matrix();expected_hip_world=preplanned_roots[:,None,:]+np.einsum('tij,kj->tki',heading_R,hip_offsets)
    swing_target_before=planned_feet.copy();planned_feet,swing_projection_fade=project_swing_targets(swing_target_before,expected,expected_hip_world,a.swing_leg_max_reach)
    swing_target_delta=planned_feet-swing_target_before;swing_distance_before=np.linalg.norm(swing_target_before-expected_hip_world,axis=2);swing_distance_after=np.linalg.norm(planned_feet-expected_hip_world,axis=2)
    solutions=[];rows=[];foot_desired=[];root_targets=[];actualfoot=[];com_actual=[];com_goals=[];support_hulls=[];adapted_yaw=[];knee_load_rows=[]
    previous_x=np.r_[seed[:3],0.,0.,seed[qa[body]],yaw[0]]
    limits=m.jnt_range[jids[body]].copy()
    for i,k in enumerate(body):
        if 'wrist_pitch_joint'in names[k]:limits[i]=[max(limits[i,0],-.25),min(limits[i,1],.25)]
    for f in range(count):
        sit=float(np.clip((time[f]-9.0)/1.2,0,1));sit=sit*sit*(3-2*sit)
        root_target=preplanned_roots[f].copy()
        desired=root_target+.84*(joints[f]-joints[f,0])
        feet=planned_feet[f].copy();root_targets.append(root_target.copy())
        contact_points=[feet[k,:2]+rotate_xy(footprints[k],foot_yaw[f,k])for k in range(2)if expected[f,k]]
        # A small assumed hip support patch is admitted only in the late seat-contact phase.
        if sit>.8:
            seats=[s for s in spec['supports']if s['kind']=='seat'and inside_convex_polygon(root_target[:2],s['xy_polygon_m'])]
            if seats:
                seat=min(seats,key=lambda s:abs(s['mode_height_m']-.436));center=project_patch(root_target[:2],seat['xy_polygon_m']);angle=np.arange(8)*2*np.pi/8;patch=center+.04*np.c_[np.cos(angle),np.sin(angle)];patch=patch[inside_convex_polygon(patch,seat['xy_polygon_m'])]
                if len(patch):contact_points.append(patch)
        hull=convex_hull(np.concatenate(contact_points))if contact_points else np.empty((0,2));support_hulls.append(hull)
        support_n,support_b=halfspaces(hull)if len(hull)>=3 else(np.empty((0,2)),np.empty(0))
        com_goal=hull.mean(0)if len(hull)else root_target[:2].copy();com_goals.append(com_goal.copy())
        fixed=seed.copy()
        lowbound=np.r_[root_target[:2]-.80,max(.40,root_target[2]-a.root_height_lower_adaptation),-.45,-.55,limits[:,0]+1e-5,yaw[f]-a.root_yaw_adaptation_window]
        highbound=np.r_[root_target[:2]+.80,root_target[2]+a.root_height_upper_adaptation,.45,.55,limits[:,1]-1e-5,yaw[f]+a.root_yaw_adaptation_window]
        if sit>.8:
            candidates=[item for item in spec['supports']if item['kind']=='seat'and inside_convex_polygon(root_target[:2],item['xy_polygon_m'])]
            if candidates:
                item=min(candidates,key=lambda it:abs(it['mode_height_m']-.436));b=np.asarray(item['top_z_eq_ax_by_c'])
                lowbound[2]=max(lowbound[2],root_target[:2]@b[:2]+b[2]+a.minimum_seated_root_above_seat)
        if f:
            delta=float(time[f]-time[f-1])
            if a.greedy_source_rate_bounds:
                lowbound[5:-1]=np.maximum(lowbound[5:-1],previous_x[5:-1]-a.max_joint_speed*delta);highbound[5:-1]=np.minimum(highbound[5:-1],previous_x[5:-1]+a.max_joint_speed*delta)
                lowbound[3:5]=np.maximum(lowbound[3:5],previous_x[3:5]-2.*delta);highbound[3:5]=np.minimum(highbound[3:5],previous_x[3:5]+2.*delta)
            velocity=(solutions[-1][:3]-solutions[-2][:3])/(time[f-1]-time[f-2])if f>1 else np.zeros(3)
        cache={}
        def evaluate(x):
            if 'x'in cache and np.array_equal(x,cache['x']):return cache['r'],cache['J']
            d.qpos[:]=fixed;d.qpos[:3]=x[:3];d.qpos[3:7]=np.roll(Rotation.from_euler('ZYX',[x[-1],x[4],x[3]]).as_quat(),1);d.qpos[qa[body]]=x[5:-1]
            H[3:6,3]=[1,0,0];H[3:6,4]=[0,np.cos(x[3]),-np.sin(x[3])]
            H[3:6,-1]=[-np.sin(x[4]),np.cos(x[4])*np.sin(x[3]),np.cos(x[4])*np.cos(x[3])]
            mj.mj_kinematics(m,d);mj.mj_comPos(m,d);mj.mj_subtreeVel(m,d)
            rr=[];JJ=[]
            def add(v,J,w):rr.extend(np.asarray(v).reshape(-1)*w);JJ.extend(np.asarray(J).reshape(-1,nx)*w)
            add(x[:3]-root_target,translation_jac,1.2)
            add(x[3:5],np.c_[np.zeros((2,3)),np.eye(2),np.zeros((2,nx-5))],.15)
            add([x[-1]-yaw[f]],yaw_jac,a.root_yaw_source_prior_weight)
            if f:add(x[:3]-(previous_x[:3]+.8*velocity*delta),translation_jac,3.)
            if f:
                root_velocity=(x[:3]-previous_x[:3])/delta
                root_acceleration=(root_velocity-velocity)/delta
                add(np.maximum(abs(root_velocity)-a.max_root_speed,0.),np.diag(np.where(abs(root_velocity)>a.max_root_speed,np.sign(root_velocity)/delta,0.))@translation_jac,a.root_speed_cost_weight)
                add(np.maximum(abs(root_acceleration)-a.max_root_acceleration,0.),np.diag(np.where(abs(root_acceleration)>a.max_root_acceleration,np.sign(root_acceleration)/(delta*delta),0.))@translation_jac,a.root_acceleration_cost_weight)
            cj_com=np.zeros((3,m.nv));mj.mj_jacSubtreeCom(m,d,cj_com,bids['base_link'])
            if len(support_n):
                violation=support_b+a.com_support_margin-support_n@d.subtree_com[bids['base_link'],:2]
                jacobian=-support_n@(cj_com@H)[:2];jacobian[violation<=0]=0.
                add(np.maximum(violation,0.),jacobian,a.com_support_weight)
            add(x[5+wrist_cols]-wrist_goal,wrist_jac,a.wrist_neutral_weight)
            for name,idx,w in targets:
                b=bids[name];pos=d.xpos[b];add(pos-(desired[idx]+x[:3]-root_target),point(b,pos)[0]-translation_jac,w)
            for k,side in enumerate(('left','right')):
                b=bids[side+'_ankle_roll_link'];pos=d.xpos[b];J,Jr=point(b,pos)
                add(pos-feet[k],J,30. if expected[f,k]else 8.)
                up=d.xmat[b].reshape(3,3)[:,2];add(up-[0,0,1],-skew(up)@Jr,1.0 if expected[f,k]else .7)
                forward=d.xmat[b].reshape(3,3)[:,0];goal_forward=np.array([np.cos(foot_yaw[f,k]),np.sin(foot_yaw[f,k]),0.])
                add(forward-goal_forward,-skew(forward)@Jr,1.2 if expected[f,k]else .35)
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
            add(cr,cj,100.)
            if f:add(x[5:-1]-previous_x[5:-1],np.c_[np.zeros((nx-6,5)),np.eye(nx-6),np.zeros((nx-6,1))],.025)
            if a.stance_knee_load_weight>0:
                load_r,load_J,_=knee_load_proxy.single_stance_cost(d,expected[f],seated=bool(sit)or time[f]>=9.,velocity_map=H,ratio_threshold=a.stance_knee_load_target_ratio)
                add(load_r,load_J,a.stance_knee_load_weight)
            cache.update(x=x.copy(),r=np.array(rr),J=np.array(JJ));return cache['r'],cache['J']
        if f==0:
            trial=np.clip(previous_x,lowbound+1e-6,highbound-1e-6);trial[3:5]=[.1,-.1]
            r0,J0=evaluate(trial);errs=[]
            for col in (3,4,nx-1):
                plus=trial.copy();minus=trial.copy();plus[col]+=1e-6;minus[col]-=1e-6
                fd=(evaluate(plus)[0]-evaluate(minus)[0])/2e-6
                # Collision minimum switches are nonsmooth; verify the torso/COM block.
                errs.append(float(np.max(abs(fd[:38]-J0[:38,col]))))
            if max(errs)>1e-4:raise AssertionError({'root_roll_pitch_yaw_jacobian_error':errs})
        fit=least_squares(lambda x:evaluate(x)[0],np.clip(previous_x,lowbound+1e-7,highbound-1e-7),jac=lambda x:evaluate(x)[1],bounds=(lowbound,highbound),max_nfev=100 if f==0 else a.max_nfev,ftol=2e-5,xtol=2e-5,gtol=2e-5)
        evaluate(fit.x);seed=d.qpos.copy();solutions.append(seed.copy());previous_x=fit.x.copy();adapted_yaw.append(float(fit.x[-1]))
        effective_feet=feet.copy();foot_desired.append(effective_feet)
        com_actual.append(d.subtree_com[bids['base_link']].copy())
        actualfoot.append([d.xpos[bids[side+'_ankle_roll_link']].copy()for side in ('left','right')])
        pen=max([max(0.,-float(c.dist))for c in d.contact[:d.ncon]]or[0.])
        com_violation=float(np.maximum(support_b+a.com_support_margin-support_n@com_actual[-1][:2],0.).max())if len(support_n)else 0.
        # Re-evaluate the final pose independently instead of reusing optimization cache diagnostics.
        _,_,load=knee_load_proxy.single_stance_cost(d,expected[f],seated=bool(sit)or time[f]>=9.,velocity_map=H,ratio_threshold=a.stance_knee_load_target_ratio)
        load_info=None if load is None else{k:(v.tolist()if isinstance(v,np.ndarray)else v)for k,v in load.items()if not k.startswith('jacobian_')}
        if load_info is not None:
            gear=knee_load_proxy.knees[load_info['side']]['gear'];load_info['original_joint_torque_range_Nm']=np.sort(np.asarray(load_info['original_actuator_force_range'])*gear).tolist();load_info['gear']=float(gear)
        knee_load_rows.append(load_info)
        rows.append({'frame':f,'cost':float(fit.cost),'nfev':int(fit.nfev),'max_penetration_m':pen,'foot_error_m':np.linalg.norm(np.array(actualfoot[-1])-effective_feet,axis=1).tolist(),'com_target_error_m':float(np.linalg.norm(com_actual[-1][:2]-com_goal)),'com_support_margin_violation_m':com_violation,'support_hull_vertices':len(hull),'root_translation_adaptation_m':float(np.linalg.norm(seed[:3]-root_target)),'root_yaw_source_delta_rad':float(fit.x[-1]-yaw[f]),'root_z_reference_m':float(seed[2]),'root_z_target_m':float(root_target[2]),'root_z_lower_bound_m':float(lowbound[2]),'root_z_upper_bound_m':float(highbound[2]),'root_z_lower_bound_gap_m':float(seed[2]-lowbound[2]),'root_z_at_lower_bound':bool(seed[2]-lowbound[2]<=1e-4),'root_z_at_upper_bound':bool(highbound[2]-seed[2]<=1e-4),'ankle_actual_z_m':np.asarray(actualfoot[-1])[:,2].tolist(),'ankle_target_z_m':effective_feet[:,2].tolist(),'ankle_height_error_m':(np.asarray(actualfoot[-1])-effective_feet)[:,2].tolist(),'swing_target_projection_delta_norm_m':np.linalg.norm(swing_target_delta[f],axis=1).tolist(),'swing_target_projection_fade':swing_projection_fade[f].tolist(),'actual_hip_ankle_distance_m':np.linalg.norm(np.asarray(actualfoot[-1])-d.xpos[hip_ids],axis=1).tolist(),'stance_knee_load_proxy':load_info})
        if f%25==0:print(json.dumps(rows[-1]),flush=True)
    q=np.asarray(solutions);a.output.parent.mkdir(parents=True,exist_ok=True)
    physical_ranges=m.jnt_range[jids].copy();physical_limited=m.jnt_limited[jids].astype(bool);physical_violation=np.maximum(np.maximum(physical_ranges[:,0]-q[:,qa],q[:,qa]-physical_ranges[:,1]),0.);physical_violation[:,~physical_limited]=0.;physical_gate=bool(physical_violation.max()<=.05)
    np.savez_compressed(a.output,qpos=q,dof_names=np.asarray(names),time_s=time,source_time_s=time.copy(),frame_ids=h['frame_ids'][:count],foot_support_expected=expected,foot_support_geom_name=surface_names,foot_support_confidence=confidence,source_joints_world_zup=rawj,source_root_world_from_pelvis_zup=transforms,source_smplx_frame_valid=h['valid'][:count],source_foot_surface_low_zup=low,source_foot_surface_z_bias=baseline,foot_target_ankle_world=np.asarray(foot_desired),foot_actual_ankle_world=np.asarray(actualfoot),root_target_world=np.asarray(root_targets),source_to_sim_U=U,inferred_contact=np.asarray(True),terrain_foot_horizontal_correction_m=correction,terrain_root_horizontal_correction_m=root_correction,contact_plan_version=np.asarray('v17_optional_stance_knee_load'),stance_knee_load_valid=np.asarray([v is not None for v in knee_load_rows]),stance_knee_load_side=np.asarray([v['side']if v is not None else''for v in knee_load_rows]),stance_knee_joint_torque_Nm=np.asarray([v['joint_torque_Nm']if v is not None else 0. for v in knee_load_rows]),stance_knee_actuator_force=np.asarray([v['actuator_force']if v is not None else 0. for v in knee_load_rows]),stance_knee_effort_ratio=np.asarray([v['effort_ratio']if v is not None else 0. for v in knee_load_rows]),stance_knee_original_actuator_force_range=np.asarray([v['original_actuator_force_range']if v is not None else[0.,0.]for v in knee_load_rows]),stance_knee_original_joint_torque_range_Nm=np.asarray([v['original_joint_torque_range_Nm']if v is not None else[0.,0.]for v in knee_load_rows]),stance_knee_load_cost_weight=np.asarray(a.stance_knee_load_weight),stance_knee_load_target_ratio=np.asarray(a.stance_knee_load_target_ratio),swing_leg_max_reach_m=np.asarray(a.swing_leg_max_reach),swing_target_before_projection_world=swing_target_before,swing_target_after_projection_world=planned_feet,swing_target_projection_delta_world_m=swing_target_delta,swing_target_projection_fade=swing_projection_fade,swing_expected_hip_world=expected_hip_world,swing_hip_offset_root_local_m=hip_offsets,neutral_model_hip_to_ankle_distance_m=nominal_hip_ankle_length,swing_target_hip_distance_before_m=swing_distance_before,swing_target_hip_distance_after_m=swing_distance_after,source_raw_ankle_world_zup=rawj[:,[7,8]],root_z_lower_bound_m=np.asarray([r['root_z_lower_bound_m']for r in rows]),root_z_upper_bound_m=np.asarray([r['root_z_upper_bound_m']for r in rows]),root_z_at_lower_bound=np.asarray([r['root_z_at_lower_bound']for r in rows]),ankle_height_error_m=np.asarray([r['ankle_height_error_m']for r in rows]),physical_joint_limit_model_sha256=np.asarray(sha(a.model)),physical_joint_names=np.asarray(names),physical_joint_limited=physical_limited,physical_joint_lower_rad=physical_ranges[:,0],physical_joint_upper_rad=physical_ranges[:,1],physical_joint_violation_tolerance_rad=np.asarray(.05),physical_joint_limit_violation_rad=physical_violation,physical_joint_limits_passed=np.asarray(physical_gate),source_toe_yaw_before_robot_adaptation_rad=source_toe_yaw_before_adaptation,turn_yaw_override_mask=turn_yaw_override_mask,turn_lift_mask=turn_lift_mask,pre_turn_foot_support_expected=pre_turn_expected,turn_contact_override_mask=expected!=pre_turn_expected,source_raw_toe_yaw_world_rad=np.arctan2((rawj[:,[10,11]]-rawj[:,[7,8]])[:,:,1],(rawj[:,[10,11]]-rawj[:,[7,8]])[:,:,0]),source_hips_heading_yaw_rad=yaw,adapted_root_yaw_rad=np.asarray(adapted_yaw),root_yaw_source_delta_rad=np.asarray(adapted_yaw)-yaw,greedy_source_rate_bounds_enabled=np.asarray(a.greedy_source_rate_bounds),pre_no_flight_foot_support_expected=pre_no_flight_expected,no_flight_adapted_contact_mask=no_flight_adapted_mask,legacy_foot_support_expected=legacy_expected,foot_target_yaw_world_rad=foot_yaw,footprint_left_ankle_local_xy_m=footprints[0],footprint_right_ankle_local_xy_m=footprints[1],reference_com_world=np.asarray(com_actual),reference_com_goal_world_xy=np.asarray(com_goals),reference_com_constraint_mode=np.asarray('support_hull_one_sided_hinge'),reference_support_hull_vertices_xy_m=np.concatenate(support_hulls)if sum(map(len,support_hulls))else np.empty((0,2)),reference_support_hull_offsets=np.r_[0,np.cumsum([len(v)for v in support_hulls])],wrist_reference_joint_names=np.asarray(list(wrist_targets)),wrist_reference_joint_targets_rad=np.asarray(list(wrist_targets.values())))
    report={'schema':'rgb145.reference.v1','frames':count,'duration_s':float(time[-1]),'reference_sha256':sha(a.output),'model_sha256':sha(a.model),'source_smplx_mesh_sha256':sha(a.source/'human/person_01/smplx_meshes_world.npz'),'source_timeline_sha256':sha(a.source/'timeline.npz'),'source_foot_vertical_bias_m':baseline.tolist(),'standing_root_height_offset_m':root_offset,'max_static_penetration_m':max(r['max_penetration_m']for r in rows),'max_ankle_target_error_m':np.max(np.linalg.norm(np.asarray(actualfoot)-np.asarray(foot_desired),axis=-1),axis=0).tolist(),'body_reference_peak_velocity_rad_s':float(np.max(abs(np.diff(q[:,qa[body]],axis=0)/np.diff(time)[:,None]))),'raw_source_preserved':True,'source_hands_observed':False,'physics_passed':False,'rows':rows}
    rv=np.diff(q[:,:3],axis=0)/np.diff(time)[:,None];ra=np.diff(rv,axis=0)/np.diff(time)[1:,None]
    report.update(method='v17: optional analytic single-stance raw knee-load cost; optional fixed-target free-swing leg-reach projection; explicit source-relative root-height adaptation bounds and effective-model physical limits; explicit floor-turn lift/replant gait adaptation before source3.2s; free root-yaw adaptation within source window; geometric-feasibility-first IK with optional legacy source-rate bounds; explicit no-flight walking contact adaptation, support-hull COM inequality, neutral relaxed wrists and equality-consistent relaxed fingers; locked full-footprint contacts and clear swing arcs',contact_plan_audit=contact_audit,maximum_root_translation_adaptation_m=max(r['root_translation_adaptation_m']for r in rows),maximum_com_target_error_m=max(r['com_target_error_m']for r in rows),root_roll_pitch_jacobian_error=errs,root_speed_source_m_s_peak=float(np.linalg.norm(rv,axis=1).max()),root_acceleration_source_m_s2_peak=float(np.linalg.norm(ra,axis=1).max()),root_temporal_limits={'mode':'explicit_soft_excess_penalties_not_hard_constraints','speed_threshold_source_m_s':a.max_root_speed,'acceleration_threshold_source_m_s2':a.max_root_acceleration,'speed_cost_weight':a.root_speed_cost_weight,'acceleration_cost_weight':a.root_acceleration_cost_weight},foot_ankle_to_sole_offsets_m=sole_offsets)
    valid_loads=[v for v in knee_load_rows if v is not None]
    report['stance_knee_load_proxy']={'optimization_enabled':a.stance_knee_load_weight>0,'cost_weight':a.stance_knee_load_weight,'target_effort_ratio':a.stance_knee_load_target_ratio,'eligible_phase':'Exactly one expected supporting foot, non-seated and source time<9s','valid_frames':len(valid_loads),'effort_ratio_max':max([v['effort_ratio']for v in valid_loads]or[0.]),'joint_torque_abs_Nm_max':max([abs(v['joint_torque_Nm'])for v in valid_loads]or[0.]),'frames_exceeding_actual_motor_capacity':[i for i,v in enumerate(knee_load_rows)if v is not None and v['effort_ratio']>1.],'static_friction_assistance_used':False,'postsolve_diagnostics':'Recomputed independently at every solved pose, including when optimization weight is0; null/valid mask outside eligible phases','model_or_torque_ranges_changed':False,'dynamic_feasibility_claimed':False,'assumption':'Static proximal-body gravity load at the supporting knee; independent LP and full physical rollout remain required'}
    report['swing_leg_reach_adaptation']={'enabled':a.swing_leg_max_reach>0,'max_reach_m':a.swing_leg_max_reach,'target_precomputation':'Fixed source-scaled root_target positions with source hip heading and zero roll/pitch; true model hip-pitch origin offsets from neutral FK. Targets never follow optimized root variables inside residuals.','hip_offsets_root_local_m':hip_offsets.tolist(),'nominal_model_hip_to_ankle_distance_m':nominal_hip_ankle_length.tolist(),'stance_targets_changed':False,'surface_labels_and_foot_yaw_changed':False,'source_raw_ankles_changed':False,'endpoint_fade_source_frames':3,'fade_rule':'smoothstep(min(1,min(index,last-index)/3)); first/last swing samples remain unchanged; max-reach is intentionally not a hard bound inside fades','applied_frames_per_side':np.sum(np.linalg.norm(swing_target_delta,axis=2)>1e-10,axis=0).tolist(),'target_delta_norm_m_max':float(np.linalg.norm(swing_target_delta,axis=2).max()),'assumption':'Explicit unladen robot-leg retraction/earlier lift for human-to-Vita morphology; no online body IK or additional torque'}
    report['root_height_adaptation']={'lower_allowance_m':a.root_height_lower_adaptation,'upper_allowance_m':a.root_height_upper_adaptation,'absolute_floor_m':.40,'minimum_seated_root_above_seat_m':a.minimum_seated_root_above_seat,'bound_activity_tolerance_m':1e-4,'frames_at_lower_bound':[r['frame']for r in rows if r['root_z_at_lower_bound']],'frames_at_upper_bound':[r['frame']for r in rows if r['root_z_at_upper_bound']],'per_frame_bounds_and_actual_ankle_errors':'Included in rows and the numeric reference; lower adaptation does not change geometry, physical joint ranges, source data, or seat-height constraints'}
    report['physical_joint_limit_gate']={'passed':physical_gate,'max_violation_rad':float(physical_violation.max()),'allowed_violation_rad':.05,'model_path':str(a.model.resolve()),'model_sha256':sha(a.model),'joint_names':names,'lower_rad':physical_ranges[:,0].tolist(),'upper_rad':physical_ranges[:,1].tolist(),'limited':physical_limited.tolist(),'model_ranges_changed':False,'reference_joint_clipping_applied':False,'provenance':'IK bounds are compiled from this explicitly selected effective runtime plant; the older canonical asset ranges are not substituted'}
    report['root_yaw_adaptation']={'window_rad':a.root_yaw_adaptation_window,'source_prior_weight':a.root_yaw_source_prior_weight,'absolute_delta_rad_max':float(np.max(abs(np.asarray(adapted_yaw)-yaw))),'per_frame_delta_rad':(np.asarray(adapted_yaw)-yaw).tolist(),'roll_pitch_yaw_jacobian_errors':errs,'raw_source_root_orientation_preserved':True,'physical_joint_ranges_and_footprints_changed':False}
    report['source_rate_contract']={'greedy_source_rate_bounds_enabled':a.greedy_source_rate_bounds,'optional_joint_speed_rad_per_source_s':a.max_joint_speed,'optional_root_roll_pitch_speed_rad_per_source_s':2.,'source_pose_speed_bounded_claim':False,'physical_joint_ranges_collision_costs_and_soft_temporal_costs_preserved':True,'execution_speed_and_acceleration':'Must be controlled and independently verified by downstream adaptive execution-clock retiming'}
    report['com_constraint']={'mode':'single_sided_normalized_halfspace_hinge; no attraction to hull center; no COM residual without planned support','inward_margin_m':a.com_support_margin,'weight':a.com_support_weight,'maximum_margin_violation_m':max(r['com_support_margin_violation_m']for r in rows),'seat_support':'Assumed40mm-radius hip contact patch inside reconstructed seat only when sit transition>.8; no measured contact force claim','legacy_com_target_error_field':'Distance to support hull centroid for diagnostics only; not an optimization objective'}
    report['wrist_reference']={'joint_targets_rad':wrist_targets,'regularization_weight':a.wrist_neutral_weight,'reference_pitch_window_rad':[-.25,.25],'droop_sign_verified_from_actual_model':wrist_sign_audit,'physical_joint_limits_and_control_policy_changed':False}
    report['unobserved_hand_pose']={'assumption':'Near-neutral relaxed drivers, unchanged polynomial followers; no observed hand pose claim','driver_targets_rad':{names[k]:float(q[0,qa[k]])for k,j in enumerate(jids)if int(j)in finger_drivers},'passive_follower_targets_rad':{names[k]:float(q[0,qa[k]])for k,j in enumerate(jids)if int(j)in followers},'initial_mujoco_joint_equality_residual_max':initial_finger_eq_residual,'passive_motor_commands':'Remain zero in unchanged physics controller'}
    helper=Path(__file__).with_name('rgb145_contact_plan.py');report['contact_plan_helper_sha256']=sha(helper)
    knee_helper=Path(__file__).with_name('rgb145_knee_load.py');report['knee_load_helper_sha256']=sha(knee_helper);a.output.with_suffix('.knee_load.py').write_bytes(knee_helper.read_bytes())
    a.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');a.output.with_suffix('.generator.py').write_bytes(Path(__file__).read_bytes());a.output.with_suffix('.contact_plan.py').write_bytes(helper.read_bytes());print(json.dumps({k:v for k,v in report.items()if k!='rows'}),flush=True)
    return 0 if physical_gate else 3


if __name__=='__main__':raise SystemExit(main())
