"""Static mesh-landmark rim pinch search; no simulation or applied forces.

Thumb is inside the hollow rim, index outside. Fingertip mesh support points
and analytic source-table halfspace avoid differentiating penetrating EPA.
Candidates are then checked using real MuJoCo collision contact normals.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import mujoco as mj
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from fourth_arm_object_feedback import ArmObjectFeedback

p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--trial',type=Path,required=True);p.add_argument('--settled-objects',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--starts',type=int,default=18);p.add_argument('--max-nfev',type=int,default=180);p.add_argument('--coupled-o6',action='store_true');p.add_argument('--root-x',type=float);p.add_argument('--root-y',type=float);p.add_argument('--yaw',type=float);p.add_argument('--seed-reference-frame',type=int);a=p.parse_args()
m=mj.MjModel.from_xml_path(str(a.model.resolve()));d=mj.MjData(m);ref=dict(np.load(a.reference,allow_pickle=False));trace=np.load(a.trial/'physical_trace.npz');fi=int(np.argmin(abs(trace['time']-3.0)));qseed=trace['qpos'][fi].copy();settled=np.load(a.settled_objects)['final_object_pose_world_xyz_wxyz'];qseed[59:]=settled.reshape(-1)
if a.seed_reference_frame is not None:qseed[:59]=ref['qpos'][a.seed_reference_frame]
if a.root_x is not None:qseed[0]=a.root_x
if a.root_y is not None:qseed[1]=a.root_y
if a.yaw is not None:qseed[3:7]=[np.cos(a.yaw/2),0,0,np.sin(a.yaw/2)]
d.qpos[:]=qseed;mj.mj_forward(m,d)
h=ArmObjectFeedback(m,ref);side=1;oi=0;obj=h.object_bids[oi];pobj=d.xpos[obj].copy();robj=d.xmat[obj].reshape(3,3).copy();plane=np.array([.010271048830894367,.02292500590934548,.999684425036206,1.0058665495441206]);plane/=np.linalg.norm(plane[:3])
hand_geoms=h.hand_geoms[side];parts={}
for g in hand_geoms+h.object_geoms[oi]:
    if m.geom_type[g]==mj.mjtGeom.mjGEOM_MESH:
        mesh=int(m.geom_dataid[g]);v=m.mesh_vert[m.mesh_vertadr[mesh]:m.mesh_vertadr[mesh]+m.mesh_vertnum[mesh]].astype(float)
    elif m.geom_type[g]==mj.mjtGeom.mjGEOM_CYLINDER:
        rr,hh=m.geom_size[g,:2];angles=np.linspace(0,2*np.pi,32,endpoint=False);v=np.array([[rr*np.cos(t),rr*np.sin(t),z]for t in angles for z in [-hh,hh]])
    else:
        size=m.geom_size[g].copy()
        if m.geom_type[g]==mj.mjtGeom.mjGEOM_CAPSULE:size=np.array([size[0],size[0],size[0]+size[1]])
        if m.geom_type[g]==mj.mjtGeom.mjGEOM_SPHERE:size[:]=size[0]
        v=np.array([[x,y,z]for x in [-size[0],size[0]]for y in [-size[1],size[1]]for z in [-size[2],size[2]]])
    world=v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g];bid=int(m.geom_bodyid[g]);local=(world-d.xpos[bid])@d.xmat[bid].reshape(3,3)
    parts[g]=(bid,local)
def vertices(g):
    bid,local=parts[g];return local@d.xmat[bid].reshape(3,3).T+d.xpos[bid]
def plane_clearance(g):
    typ=m.geom_type[g];r=d.geom_xmat[g].reshape(3,3);size=m.geom_size[g];centre=float(d.geom_xpos[g]@plane[:3]+plane[3])
    if typ==mj.mjtGeom.mjGEOM_CAPSULE:return centre-size[0]-size[1]*abs(r[:,2]@plane[:3])
    if typ==mj.mjtGeom.mjGEOM_CYLINDER:
        axial=abs(r[:,2]@plane[:3]);return centre-size[1]*axial-size[0]*np.sqrt(max(0,1-axial*axial))
    if typ==mj.mjtGeom.mjGEOM_SPHERE:return centre-size[0]
    return float(np.min(vertices(g)@plane[:3]+plane[3]))
all_object=np.concatenate([vertices(g)for g in h.object_geoms[oi]]);ov=(all_object-pobj)@robj
radius=np.linalg.norm(ov[:,:2],axis=1);top=float(ov[:,2].max());bottom=float(ov[:,2].min());upper=radius[ov[:,2]>top-.001]
outer=float(np.max(radius));inner=float(np.min(upper[upper>outer*.5]));midz=top-.006
wall_hulls=[]
for g in h.object_geoms[oi]:
    v=(vertices(g)-pobj)@robj
    if m.geom_type[g]==mj.mjtGeom.mjGEOM_MESH and np.linalg.norm(v.mean(axis=0)[:2])>outer*.5:wall_hulls.append(ConvexHull(v).equations)
def wall_interval(radial,z):
    intervals=[]
    for eq in wall_hulls:
        aa=eq[:,:3]@radial;bb=eq[:,2]*z+eq[:,3];lo,hi=0.,np.inf
        if np.any((abs(aa)<1e-9)&(bb>1e-7)):continue
        for x,y in zip(aa,bb):
            if x>1e-9:hi=min(hi,-y/x)
            elif x< -1e-9:lo=max(lo,-y/x)
        if lo<=hi and hi>0:intervals.append((lo,hi))
    if not intervals:raise ValueError('No hollow wall intersects the chosen rim ray')
    return min(v[0]for v in intervals),max(v[1]for v in intervals)
thumb=next(g for g in hand_geoms if (mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[g]))or'')=='rh_thumb_distal')
index=next(g for g in hand_geoms if (mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[g]))or'')=='rh_index_distal')
def principal(g):
    v=parts[g][1];_,_,vv=np.linalg.svd(v-v.mean(axis=0));axis=vv[0]
    return axis if axis[2]>=0 else-axis
axes={g:principal(g)for g in [thumb,index]}
names=[mj.mj_id2name(m,mj.mjtObj.mjOBJ_JOINT,j)for j in range(m.njnt)]
arm=h.chains[1]['joint_ids'];finger_names=(['rh_thumb_cmc_yaw','rh_thumb_cmc_pitch','rh_index_mcp_pitch','rh_middle_mcp_pitch','rh_ring_mcp_pitch','rh_pinky_mcp_pitch']if a.coupled_o6 else['rh_thumb_cmc_yaw','rh_thumb_cmc_pitch','rh_thumb_ip','rh_index_mcp_pitch','rh_index_dip']);finger=[names.index(n)for n in finger_names];selected=np.asarray(arm+finger);qa=m.jnt_qposadr[selected];lower=m.jnt_range[selected,0]+1e-5;upper=m.jnt_range[selected,1]-1e-5
couplings={}
if a.coupled_o6:
    for eq in range(m.neq):
        if m.eq_type[eq]==mj.mjtEq.mjEQ_JOINT:
            slave,master=int(m.eq_obj1id[eq]),int(m.eq_obj2id[eq])
            if (names[slave]or'').startswith(('lh_','rh_')):
                if np.max(abs(m.eq_data[eq,[0,2,3,4]]))>1e-12:raise ValueError('Expected linear zero-offset O6 mimic')
                couplings[slave]=(master,float(m.eq_data[eq,1]))
    if len(couplings)!=10:raise ValueError('Expected O6 model with all ten mechanical mimic equalities')
    for passive,(master,ratio)in couplings.items():
        if master in selected:
            j=int(np.flatnonzero(selected==master)[0]);lower[j]=max(lower[j],m.jnt_range[passive,0]/ratio+1e-5);upper[j]=min(upper[j],m.jnt_range[passive,1]/ratio-1e-5)
finger_all=[j for j,n in enumerate(names)if n and n.startswith('rh_')]
rest=qseed.copy()
for j in finger_all:
    if j not in finger:rest[m.jnt_qposadr[j]]=1.1 if 'mcp'in names[j]else 1.35
def setpose(q):
    d.qpos[:]=rest;d.qpos[qa]=q
    for passive,(master,ratio)in couplings.items():d.qpos[m.jnt_qposadr[passive]]=ratio*d.qpos[m.jnt_qposadr[master]]
    mj.mj_kinematics(m,d);mj.mj_comPos(m,d)
base=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'base_link');radial=robj.T@(d.xpos[base]-pobj);az0=float(np.arctan2(radial[1],radial[0]));rng=np.random.default_rng(20261027);rows=[];best_score=np.inf;best=None
print(json.dumps({'plate_outer_radius_m':outer,'plate_inner_top_radius_m':inner,'plate_top_z_local_m':top,'plate_bottom_z_local_m':bottom,'thumb_mesh_vertices':len(parts[thumb][1]),'index_mesh_vertices':len(parts[index][1])}),flush=True)
for attempt in range(a.starts):
    az=az0+([0.,.3,-.3,.6,-.6][attempt%5]);r_local=np.array([np.cos(az),np.sin(az),0.]);r_world=robj@r_local;tangent=robj@np.array([-np.sin(az),np.cos(az),0.]);z_world=robj[:,2]
    ray_inner,ray_outer=wall_interval(r_local,midz)
    def tip_landmark(g,sign):
        v=parts[g][1];v=v[v[:,2]>np.quantile(v[:,2],.85)];x=np.quantile(v[:,0],.6 if sign>0 else .4);v=v[v[:,0]>=x]if sign>0 else v[v[:,0]<=x];return v.mean(axis=0)
    thumb_point=tip_landmark(thumb,1 if attempt%2 else-1);index_point=tip_landmark(index,1 if (attempt//2)%2 else-1)
    swap=bool((attempt//4)%2)
    if attempt==0:seed=qseed[qa].copy()
    else:
        seed=qseed[qa]+rng.normal(0,.45,len(qa));seed[7:]=rng.uniform(lower[7:],upper[7:])
    if a.coupled_o6 and attempt<8:seed[7:]=[.7,.3,1.,1.2,1.2,1.2]
    seed=np.clip(seed,lower,upper)
    def residual(q):
        setpose(q)
        tv=vertices(thumb);iv=vertices(index);tc=tv.mean(axis=0)-pobj;ic=iv.mean(axis=0)-pobj
        if a.coupled_o6:
            tp=thumb_point@d.xmat[parts[thumb][0]].reshape(3,3).T+d.xpos[parts[thumb][0]]
            ip=index_point@d.xmat[parts[index][0]].reshape(3,3).T+d.xpos[parts[index][0]]
            tr=(ray_outer-.0003)if swap else(ray_inner+.0003);ir=(ray_inner+.0003)if swap else(ray_outer-.0003)
            tt=pobj+r_world*tr-tangent*.003+z_world*midz;it=pobj+r_world*ir+tangent*.003+z_world*midz
            positions=np.r_[30*(tp-tt),30*(ip-it)]
        else:positions=[30*(float(np.max((tv-pobj)@r_world))-ray_inner-.0003),30*(float(np.min((iv-pobj)@r_world))-ray_outer+.0003),20*(float(tc@tangent)+.006),20*(float(ic@tangent)-.006),25*(float(tc@z_world)-midz),25*(float(ic@z_world)-midz)]
        orientation=[]
        for g in [thumb,index]:
            axis=d.xmat[parts[g][0]].reshape(3,3)@axes[g]
            orientation.extend((.005 if a.coupled_o6 else .06)*np.cross(axis,tangent))
        clear=np.array([plane_clearance(g)for g in hand_geoms]);palm_height=float(h.palm(d,1)@plane[:3]+plane[3])
        return np.r_[positions,orientation,200*np.minimum(clear-.002,0),.4*(palm_height-.065),.003*(q[:7]-qseed[qa][:7])]
    fit=least_squares(residual,seed,bounds=(lower,upper),max_nfev=a.max_nfev,ftol=2e-5,xtol=2e-5,gtol=2e-5,diff_step=2e-5)
    setpose(fit.x);d.qvel[:]=0;mj.mj_forward(m,d)
    minplane=float(min(plane_clearance(g)for g in hand_geoms));contacts=[];norms={'thumb':[],'index':[]};maxpen=0.
    for c in d.contact:
        g1,g2=int(c.geom1),int(c.geom2);b1,b2=int(m.geom_bodyid[g1]),int(m.geom_bodyid[g2])
        if obj not in (b1,b2):continue
        hg=g1 if g1 in hand_geoms else g2 if g2 in hand_geoms else None
        if hg is None:continue
        gn=mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,hg)or'';normal=np.asarray(c.frame).reshape(3,3)[0]*(1 if b2==obj else-1);maxpen=max(maxpen,float(-c.dist));contacts.append({'hand_geom':gn,'distance_m':float(c.dist),'normal_on_object_world':normal.tolist()})
        for digit in norms:
            if digit in gn:norms[digit].append(normal)
    dot=min((float(t@i)for t in norms['thumb']for i in norms['index']),default=1.);feasible=bool(dot<-.25 and minplane>=.0019 and maxpen<=.003)
    error=float(np.linalg.norm(residual(fit.x)));score=error+100*max(.002-minplane,0)+10*max(maxpen-.003,0)+(.2 if dot>=-.25 else 0)
    palm=h.palm(d,1);handr=d.xmat[h.hand_bids[1]].reshape(3,3);rotation=robj.T@handr;quat=Rotation.from_matrix(rotation).as_quat()[[3,0,1,2]]
    template={'hand':'right','palm_offset_in_object_m':((palm-pobj)@robj).tolist(),'hand_rotation_in_object_wxyz':quat.tolist(),'finger_joint_targets':{names[j]:float(d.qpos[m.jnt_qposadr[j]])for j in finger_all}}
    row={'attempt':attempt,'residual_norm':error,'thumb_outside_index_inside':swap,'minimum_hand_plane_clearance_m':minplane,'maximum_hand_object_penetration_m':maxpen,'thumb_index_minimum_normal_dot':dot,'feasible_static_candidate':feasible,'contacts':contacts,'template':template,'arm_joint_targets':{names[j]:float(d.qpos[m.jnt_qposadr[j]])for j in arm},'qpos':d.qpos.tolist()};rows.append(row)
    if score<best_score:best_score=score;best=row
    print(json.dumps({k:row[k]for k in ['attempt','residual_norm','minimum_hand_plane_clearance_m','maximum_hand_object_penetration_m','thumb_index_minimum_normal_dot','feasible_static_candidate']}),flush=True)
    if feasible:break
report={'status':'static_mesh_rim_grasp_candidate_found_not_physics_verified'if any(r['feasible_static_candidate']for r in rows)else'no_static_opposed_grasp_found','scope':'Private FK and real MuJoCo collision normals only; zero physics steps/no actual force writes','coupled_o6_six_independent_commands':a.coupled_o6,'mimic_ratios':{'four_DIP_per_MCP':.9043182459,'thumb_IP_per_pitch':1.968329967}if a.coupled_o6 else{},'model':str(a.model),'model_sha256':hashlib.sha256(a.model.read_bytes()).hexdigest(),'plane_world_abcd':plane.tolist(),'object_id':1,'best':next((r for r in rows if r['feasible_static_candidate']),best),'attempts':rows};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');np.savez_compressed(a.output.with_suffix('.npz'),qpos=np.asarray(report['best']['qpos']),object_pose=settled,plane_world_abcd=plane)
