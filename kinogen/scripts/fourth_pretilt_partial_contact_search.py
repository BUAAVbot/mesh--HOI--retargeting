"""Private one-finger upward outer-wall contact search for supported pre-tilt.

This stage intentionally retains table support. It makes no unsupported-grasp
or executed-tilt claim and never edits canonical control modules or raw poses.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import mujoco as mj
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from fourth_arm_object_feedback import ArmObjectFeedback
from fourth_convex_separation import ConvexSeparationOracle

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--stand',type=Path,required=True);p.add_argument('--settled-candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--starts',type=int,default=4);p.add_argument('--max-nfev',type=int,default=150);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
m=mj.MjModel.from_xml_path(str(a.model.resolve()));d=mj.MjData(m);ref=dict(np.load(a.reference,allow_pickle=False));stand=dict(np.load(a.stand,allow_pickle=False));saved=json.loads(a.settled_candidate.read_text())['best'];h=ArmObjectFeedback(m,ref);oracle=ConvexSeparationOracle(m)
refnames=list(map(str,ref['dof_names']));standnames=list(map(str,stand['dof_names']));robotj=[m.joint(n).id for n in refnames];seed=np.asarray(saved['qpos']);seed[m.jnt_qposadr[robotj]]=[stand['qpos'][0,7+standnames.index(n)]for n in refnames]
allnames=[m.joint(j).name for j in range(m.njnt)];arm=h.chains[1]['joint_ids'];master=[allnames.index('rh_'+name)for name in ['thumb_cmc_yaw','thumb_cmc_pitch','index_mcp_pitch','middle_mcp_pitch','ring_mcp_pitch','pinky_mcp_pitch']];selected=np.array(arm+master);qa=m.jnt_qposadr[selected];finger=[j for j,name in enumerate(allnames)if name and name.startswith('rh_')];allj=np.array(arm+finger);allv=m.jnt_dofadr[allj];K=np.zeros((len(allj),len(selected)));P=np.zeros((m.nv,6));relations=[]
for i,j in enumerate(allj):
    if j in selected:K[i,int(np.flatnonzero(selected==j)[0])]=1
for i,j in enumerate(master):P[m.jnt_dofadr[j],i]=1
for e in range(m.neq):
    if m.eq_type[e]!=mj.mjtEq.mjEQ_JOINT:continue
    child,parent=int(m.eq_obj1id[e]),int(m.eq_obj2id[e]);ratio=float(m.eq_data[e,1]);relations.append((child,parent,ratio))
    if child in allj and parent in selected:K[int(np.flatnonzero(allj==child)[0]),int(np.flatnonzero(selected==parent)[0])]=ratio
    if parent in master:P[m.jnt_dofadr[child],master.index(parent)]=ratio
lo=m.jnt_range[selected,0]+1e-5;hi=m.jnt_range[selected,1]-1e-5
for child,parent,ratio in relations:
    if parent in selected:
        k=int(np.flatnonzero(selected==parent)[0]);lo[k]=max(lo[k],m.jnt_range[child,0]/ratio+1e-5);hi[k]=min(hi[k],m.jnt_range[child,1]/ratio-1e-5)
hand=h.hand_geoms[1];meshes=[g for g in hand if m.geom_type[g]==mj.mjtGeom.mjGEOM_MESH];index=m.geom('collision_rh_index_distal_mesh').id;objects=sum(h.object_geoms,[]);JP=np.zeros((3,m.nv));JR=np.zeros_like(JP)
ni=m.numeric('source_table_plane_abcd').id;ad=int(m.numeric_adr[ni]);plane=m.numeric_data[ad:ad+4].copy();plane/=np.linalg.norm(plane[:3]);up=plane[:3]
def setq(q):
    d.qpos[:]=seed;d.qpos[qa]=q
    for child,parent,ratio in relations:d.qpos[m.jnt_qposadr[child]]=ratio*d.qpos[m.jnt_qposadr[parent]]
    mj.mj_kinematics(m,d);mj.mj_comPos(m,d)
def vertices(g):return oracle._shape(g)['vertices']@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
def bounds(g):
    R=d.geom_xmat[g].reshape(3,3);c=d.geom_xpos[g]+R@m.geom_aabb[g,:3];ex=abs(R)@m.geom_aabb[g,3:];return c-ex,c+ex
def plane_gap(g):
    kind=m.geom_type[g];R=d.geom_xmat[g].reshape(3,3);s=m.geom_size[g];centre=float(d.geom_xpos[g]@up+plane[3]);proj=float(R[:,2]@up)
    if kind==mj.mjtGeom.mjGEOM_MESH:return float(np.min(vertices(g)@up+plane[3]))
    if kind==mj.mjtGeom.mjGEOM_CAPSULE:return centre-s[0]-s[1]*abs(proj)
    if kind==mj.mjtGeom.mjGEOM_CYLINDER:return centre-s[1]*abs(proj)-s[0]*np.sqrt(max(0,1-proj*proj))
    if kind==mj.mjtGeom.mjGEOM_SPHERE:return centre-s[0]
    return centre-float(abs(R.T@up)@s)
setq(seed[qa]);ob=h.object_bids[0];pobj=d.xpos[ob].copy();Robj=d.xmat[ob].reshape(3,3).copy();wall=[]
for g in h.object_geoms[0]:
    if m.geom_type[g]==mj.mjtGeom.mjGEOM_MESH:
        local=(vertices(g)-pobj)@Robj;wall.append((g,local,ConvexHull(local).equations))
wallpoints=np.concatenate([item[1]for item in wall]);zmin,zmax=wallpoints[:,2].min(),wallpoints[:,2].max()
oldtemplate=saved['template'];oldR=Rotation.from_quat(np.asarray(oldtemplate['hand_rotation_in_object_wxyz'])[[1,2,3,0]]).as_matrix();oldaz=np.arctan2(oldtemplate['palm_offset_in_object_m'][1],oldtemplate['palm_offset_in_object_m'][0]);results=[]
def patch_at(az,frac):
    # Use an actual sector centre; exact nominal angular boundaries can lie
    # in the tiny gap between the independent convex wall parts.
    centres=[np.arctan2(v[:,1].mean(),v[:,0].mean())for _,v,_ in wall]
    az=min(centres,key=lambda value:abs(np.arctan2(np.sin(value-az),np.cos(value-az))))
    radial=np.array([np.cos(az),np.sin(az),0.]);zz=float(zmin+frac*(zmax-zmin));choices=[]
    for g,v,eq in wall:
        aa=eq[:,:3]@radial;bb=eq[:,2]*zz+eq[:,3];lower,upper=0.,np.inf;limiter=None
        if np.any((abs(aa)<1e-9)&(bb>1e-7)):continue
        for i,(x,y)in enumerate(zip(aa,bb)):
            if x>1e-9:
                bound=-y/x
                if bound<upper:upper=bound;limiter=i
            elif x< -1e-9:lower=max(lower,-y/x)
        if lower<=upper and upper>0 and np.isfinite(upper):choices.append((upper,g,eq[limiter,:3]))
    radius,g,outward=max(choices,key=lambda item:item[0]);force=-Robj@outward;force/=np.linalg.norm(force)
    return pobj+Robj@(radial*radius+np.array([0,0,zz])),force,g
for attempt in range(a.starts):
    root=[[-.50,2.25,-2.4],[-.50,2.30,-2.4]][(attempt//2)%2];seed[0:2]=root[:2];seed[3:7]=[np.cos(root[2]/2),0,0,np.sin(root[2]/2)]
    az=np.deg2rad([45.,63.][attempt%2]);point,normal,target_geom=patch_at(az,.85)
    desiredR=Robj@Rotation.from_rotvec([0,0,az-oldaz]).as_matrix()@oldR
    palm=pobj+Robj@np.array([.135*np.cos(az),.135*np.sin(az),.075])
    q0=seed[qa].copy();q0[7:]=[1.0,.28,.65,1.45,1.45,1.45];q0=np.clip(q0,lo,hi)
    def init_res(q):
        qq=q0.copy();qq[:7]=q;setq(qq);return np.r_[30*(h.palm(d,1)-palm),Rotation.from_matrix(d.xmat[h.hand_bids[1]].reshape(3,3)@desiredR.T).as_rotvec()]
    init=least_squares(init_res,q0[:7],bounds=(lo[:7],hi[:7]),max_nfev=100);q0[:7]=init.x;setq(q0)
    obounds=np.array([bounds(g)for g in objects]);olo,ohi=obounds[:,0],obounds[:,1];cache={'q':None};calls=[0]
    def evaluate(q):
        if cache['q']is not None and np.array_equal(q,cache['q']):return cache['r'],cache['J']
        setq(q);r=[];J=[];v=vertices(index);projection=v@normal;pt=v[projection>=projection.max()-1e-10].mean(0);goal=point-normal*.001
        mj.mj_jac(m,d,JP,None,pt,int(m.geom_bodyid[index]));r.extend(60*(pt-goal));J.extend(60*JP[:,allv]@K)
        for g in meshes:
            lower,upper=bounds(g);near=np.linalg.norm(np.maximum(np.maximum(lower-ohi,olo-upper),0.),axis=1)<.008
            for k,og in enumerate(objects):
                gap,row=oracle.query(d,g,og,allv)if near[k]else(.008,np.zeros(len(allv)))
                threshold=.0006 if g==index else .002
                r.append(250*min(gap-threshold,0));J.append(250*row@K if gap<threshold else np.zeros(len(q)))
        for g in hand:
            pg=plane_gap(g)
            if pg>=.002:gap,row=pg,np.zeros(len(allv))
            elif m.geom_type[g]in(mj.mjtGeom.mjGEOM_MESH,mj.mjtGeom.mjGEOM_CYLINDER):gap,row=oracle.query(d,g,h.table,allv)
            else:
                witness=np.zeros(6);gap=float(mj.mj_geomDistance(m,d,g,h.table,1.,witness));diff=witness[:3]-witness[3:];n=diff/max(np.linalg.norm(diff),1e-9);mj.mj_jac(m,d,JP,None,witness[:3],int(m.geom_bodyid[g]));row=n@JP[:,allv]
            r.append(250*min(gap-.002,0));J.append(250*row@K if gap<.002 else np.zeros(len(q)))
        r.extend(.003*(q-q0));J.extend(.003*np.eye(len(q)));calls[0]+=1
        cache.update(q=q.copy(),r=np.asarray(r),J=np.asarray(J))
        if calls[0]%30==0:print(json.dumps({'attempt':attempt,'calls':calls[0],'residual':float(np.linalg.norm(r))}),flush=True)
        return cache['r'],cache['J']
    fit=least_squares(lambda q:evaluate(q)[0],q0,jac=lambda q:evaluate(q)[1],bounds=(lo,hi),max_nfev=a.max_nfev,ftol=2e-6,xtol=2e-6,gtol=2e-6);setq(fit.x)
    mingap=min(oracle.query(d,g,og)[0]for g in meshes for og in objects)
    tablegap=min(oracle.query(d,g,h.table)[0]for g in meshes)
    witness=np.zeros(6);distance=float(mj.mj_geomDistance(m,d,index,target_geom,.1,witness));delta=witness[0:3]-witness[3:6];force=-delta/max(np.linalg.norm(delta),1e-12)
    qfull=d.qpos.copy();mj.mj_forward(m,d);mj.mj_jac(m,d,JP,None,witness[:3],int(m.geom_bodyid[index]));normal_force=1.35
    torque=P.T@d.qfrc_bias+P.T@JP.T@(normal_force*force)
    rscene=[]
    for contact in d.contact[:d.ncon]:
        b=[int(m.geom_bodyid[contact.geom1]),int(m.geom_bodyid[contact.geom2])];bn=[m.body(i).name or'world'for i in b];robot=[i>0 and not name.startswith('target_object_')for i,name in zip(b,bn)]
        if any(robot)and not all(robot)and contact.dist<-.001:rscene.append({'bodies':bn,'depth_m':-float(contact.dist)})
    good=bool(mingap>=.0004 and tablegap>=.0018 and 1e-5<distance<.0025 and force@up>.6 and np.max(abs(torque))<.35 and not rscene)
    row={'attempt':attempt,'base_xy_yaw':root,'outerwall_patch_world_m':point.tolist(),'requested_force_on_object_world':normal.tolist(),'target_geom':m.geom(target_geom).name,
        'minimum_all_hand_object_sat_gap_m':float(mingap),'minimum_finite_table_mesh_sat_gap_m':float(tablegap),'actual_index_gap_m':distance,'actual_force_on_object_normal_world':force.tolist(),
        'actual_force_normal_up_dot':float(force@up),'hypothetical_contact_normal_force_N':normal_force,'required_master_torque_for_test_Nm':torque.tolist(),'robot_scene_penetrations_gt1mm':rscene,
        'partial_support_pregrasp_passed':good,'qpos':qfull.tolist(),'residual_norm':float(np.linalg.norm(fit.fun)),'query_calls':calls[0],
        'template':{'hand':'right','palm_offset_in_object_m':((h.palm(d,1)-pobj)@Robj).tolist(),'hand_rotation_in_object_wxyz':Rotation.from_matrix(Robj.T@d.xmat[h.hand_bids[1]].reshape(3,3)).as_quat()[[3,0,1,2]].tolist(),'finger_joint_targets':{allnames[j]:float(qfull[m.jnt_qposadr[j]])for j in finger}}}
    results.append(row);best=min(results,key=lambda item:(not item['partial_support_pregrasp_passed'],max(0,.6-item['actual_force_normal_up_dot'])*10+max(0,.0004-item['minimum_all_hand_object_sat_gap_m'])*500+item['residual_norm']))
    report={'schema':'fourth_pretilt_partial_contact_geometry.v1','status':'partial_support_candidate_requires_physics'if best['partial_support_pregrasp_passed']else'no_positive_upward_candidate_yet',
        'scope':'Disposable geometry search only. Table remains support; no unsupported grasp or actual tilt claim. Raw/model/reference/controller files unchanged.',
        'model':str(a.model),'model_sha256':hashlib.sha256(a.model.read_bytes()).hexdigest(),'best':best,'attempts':results,'oracle':oracle.report()}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');np.savez_compressed(a.output.with_suffix('.npz'),qpos=np.array(best['qpos']))
    print(json.dumps({key:row[key]for key in ['attempt','base_xy_yaw','minimum_all_hand_object_sat_gap_m','minimum_finite_table_mesh_sat_gap_m','actual_index_gap_m','actual_force_normal_up_dot','required_master_torque_for_test_Nm','partial_support_pregrasp_passed']}),flush=True)
    if good:break
