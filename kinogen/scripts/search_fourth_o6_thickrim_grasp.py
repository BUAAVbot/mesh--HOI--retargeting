"""Private FK search for a three-patch O6 support grasp, with wrench audit.

The table's real convex prism is used; the fitted plane is only an upper-side
early separation certificate. Optional object translation is a hypothetical
pre-slide planning case, never an executed change of physical object state.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import mujoco as mj
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from fourth_arm_object_feedback import ArmObjectFeedback
from fourth_convex_separation import ConvexSeparationOracle
from fourth_contact_wrench import contact_columns,wrench_lp

p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--trial',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--starts',type=int,default=12);p.add_argument('--max-nfev',type=int,default=130);p.add_argument('--pre-slide-m',type=float,default=0.);p.add_argument('--pre-tilt-deg',type=float,default=0.);p.add_argument('--settled-objects',type=Path);p.add_argument('--seed-from-below',action='store_true');p.add_argument('--allow-waist',action='store_true');p.add_argument('--initial-pose-json',type=Path);p.add_argument('--free-support-patches',action='store_true');p.add_argument('--support-radius-target-m',type=float);p.add_argument('--rim-rise-m',type=float,default=.008);a=p.parse_args()
m=mj.MjModel.from_xml_path(str(a.model.resolve()));d=mj.MjData(m);ref=dict(np.load(a.reference));trace=dict(np.load(a.trial/'physical_trace.npz'));seed=trace['qpos'][0].copy();h=ArmObjectFeedback(m,ref);oracle=ConvexSeparationOracle(m);names=[m.joint(i).name for i in range(m.njnt)];side=1;oi=0;obj=h.object_bids[oi];waist=[j for j,n in enumerate(names)if n and n.startswith('waist')]if a.allow_waist else[];arm=waist+h.chains[side]['joint_ids'];narm=len(arm);master=[names.index('rh_'+n)for n in ['thumb_cmc_yaw','thumb_cmc_pitch','index_mcp_pitch','middle_mcp_pitch','ring_mcp_pitch','pinky_mcp_pitch']];selected=np.asarray(arm+master);qa=m.jnt_qposadr[selected];finger=[i for i,n in enumerate(names)if n and n.startswith('rh_')];allj=np.asarray(arm+finger);allv=m.jnt_dofadr[allj];K=np.zeros((len(allv),len(selected)));couplings=[];P=np.zeros((m.nv,6))
for i,j in enumerate(allj):
    if j in selected:K[i,int(np.flatnonzero(selected==j)[0])]=1
for k,j in enumerate(master):P[m.jnt_dofadr[j],k]=1
for e in range(m.neq):
    if m.eq_type[e]!=mj.mjtEq.mjEQ_JOINT:continue
    child,parent=int(m.eq_obj1id[e]),int(m.eq_obj2id[e]);ratio=float(m.eq_data[e,1]);couplings.append((child,parent,ratio))
    if child in allj and parent in selected:K[int(np.flatnonzero(allj==child)[0]),int(np.flatnonzero(selected==parent)[0])]=ratio
    if parent in master:P[m.jnt_dofadr[child],master.index(parent)]=ratio
lower=m.jnt_range[selected,0]+1e-5;upper=m.jnt_range[selected,1]-1e-5
for k,j in enumerate(waist):
    cap=1.8 if'yaw'in names[j]else .8;lower[k]=max(lower[k],-cap);upper[k]=min(upper[k],cap)
for child,parent,ratio in couplings:
    if parent in selected:
        k=int(np.flatnonzero(selected==parent)[0]);lower[k]=max(lower[k],m.jnt_range[child,0]/ratio+1e-5);upper[k]=min(upper[k],m.jnt_range[child,1]/ratio-1e-5)
if a.settled_objects:
    settled=np.load(a.settled_objects)['final_object_pose_world_xyz_wxyz']
    for j,adr in enumerate(h.object_qadr):seed[adr:adr+7]=settled[j]
outward=np.array([.35216658756772096,.9359373347617386,0.]);outward/=np.linalg.norm(outward);outward[2]=-(.010271048830894367*outward[0]+.02292500590934548*outward[1])/.999684425036206;outward/=np.linalg.norm(outward);seed[h.object_qadr[oi]:h.object_qadr[oi]+3]+=a.pre_slide_m*outward
if a.initial_pose_json:
    candidate=json.loads(a.initial_pose_json.read_text());rows=candidate.get('rows',[candidate.get('best',candidate)]);chosen=min(rows,key=lambda row:row.get('position_error_m',0)+.05*row.get('orientation_error_rad',0)+100*row.get('robot_table_max_penetration_m',0));saved=np.asarray(chosen['qpos']);seed[:59]=saved[:59]
hand=h.hand_geoms[side];meshes=[g for g in hand if m.geom_type[g]==mj.mjtGeom.mjGEOM_MESH];targets={digit:next(g for g in meshes if m.body(int(m.geom_bodyid[g])).name=='rh_'+digit+'_distal')for digit in ['thumb','index','middle','ring','pinky']};ogeoms=sum(h.object_geoms,[]);plane=np.array([.010271048830894367,.02292500590934548,.999684425036206,1.0058665495441206]);plane/=np.linalg.norm(plane[:3]);JP=np.zeros((3,m.nv));JR=np.zeros_like(JP)

def setq(q):
    d.qpos[:]=seed;d.qpos[qa]=q
    for child,parent,ratio in couplings:d.qpos[m.jnt_qposadr[child]]=ratio*d.qpos[m.jnt_qposadr[parent]]
    mj.mj_kinematics(m,d);mj.mj_comPos(m,d)
def verts(g):return oracle._shape(g)['vertices']@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
def aabb(g):
    R=d.geom_xmat[g].reshape(3,3);c=d.geom_xpos[g]+R@m.geom_aabb[g,:3];ex=abs(R)@m.geom_aabb[g,3:];return c-ex,c+ex
def plane_gap(g):
    kind=m.geom_type[g]
    if kind==mj.mjtGeom.mjGEOM_MESH:return float(np.min(verts(g)@plane[:3]+plane[3]))
    R=d.geom_xmat[g].reshape(3,3);s=m.geom_size[g];c=d.geom_xpos[g]@plane[:3]+plane[3];proj=abs(R[:,2]@plane[:3])
    if kind==mj.mjtGeom.mjGEOM_CAPSULE:return float(c-s[0]-s[1]*proj)
    if kind==mj.mjtGeom.mjGEOM_CYLINDER:return float(c-s[1]*proj-s[0]*np.sqrt(max(0,1-proj*proj)))
    return float(c-s[0])
setq(seed[qa]);pobj=d.xpos[obj].copy();Robj=d.xmat[obj].reshape(3,3).copy();world=np.concatenate([verts(g)for g in h.object_geoms[0]]);ov=(world-pobj)@Robj;top=float(ov[:,2].max());bottom=float(ov[:,2].min());radius=float(np.linalg.norm(ov[:,:2],axis=1).max());rim_bottom=top-a.rim_rise_m-.004;wallv=np.concatenate([((verts(g)-pobj)@Robj)for g in h.object_geoms[0]if m.geom_type[g]==mj.mjtGeom.mjGEOM_MESH]);zbase=float(wallv[:,2].min());baser=float(np.linalg.norm(wallv[wallv[:,2]<zbase+1e-5,:2],axis=1).max());tilt_translation=np.zeros(3)
if a.pre_tilt_deg:
    rr=Robj.T@outward;rr[2]=0;rr/=np.linalg.norm(rr);tt=Robj@np.cross([0.,0.,1.],rr);bottomgeom=next(g for g in h.object_geoms[0]if m.geom_type[g]==mj.mjtGeom.mjGEOM_CYLINDER);pivot=pobj-Robj@rr*m.geom_size[bottomgeom,0]+Robj[:,2]*bottom;rot=Rotation.from_rotvec(-tt*np.deg2rad(a.pre_tilt_deg)).as_matrix();newpos=pivot+rot@(pobj-pivot);tilt_translation=newpos-pobj;newR=rot@Robj;adr=h.object_qadr[0];seed[adr:adr+3]=newpos;seed[adr+3:adr+7]=Rotation.from_matrix(newR).as_quat()[[3,0,1,2]];setq(seed[qa]);pobj=d.xpos[obj].copy();Robj=d.xmat[obj].reshape(3,3).copy()
radial=Robj.T@(d.xpos[m.body('base_link').id]-pobj);az0=float(np.arctan2(radial[1],radial[0]));obounds=np.array([aabb(g)for g in ogeoms]);olo,ohi=obounds[:,0],obounds[:,1];results=[];rng=np.random.default_rng(20261027)
print(json.dumps({'object_origin':pobj.tolist(),'radius':radius,'base_radius':baser,'zbase':zbase,'top':top,'pre_slide_planning_translation_m':(a.pre_slide_m*outward).tolist(),'pre_tilt_planning_deg':a.pre_tilt_deg,'pre_tilt_pivot_translation_m':tilt_translation.tolist()}),flush=True)
for attempt in range(a.starts):
    az=(float(np.arctan2(*(Robj.T@outward)[[1,0]]))if a.seed_from_below else az0)+[0.,.12,-.12,.25,-.25][attempt%5];rl=np.array([np.cos(az),np.sin(az),0.]);rw=Robj@rl;tw=Robj@np.array([-np.sin(az),np.cos(az),0.]);zw=Robj[:,2]
    frac=[.25,.50,.75][attempt%3];ro=baser+frac*(radius-baser);zz=zbase+frac*(top-zbase);slope=radius-baser;nhat=Robj@(-rl*(top-zbase)+np.array([0,0,slope]));nhat/=np.linalg.norm(nhat)
    # Two lower outer-wall patches and a top-rim thumb patch. No +/- tangent
    # offset is imposed for the thumb/index pair; the third point adds wrench rank.
    patch={'thumb':(pobj+rw*(radius-.002)+zw*top,-zw),'index':(pobj+rw*ro-tw*.009+zw*zz,nhat),'middle':(pobj+rw*ro+tw*.009+zw*zz,nhat)}
    if a.seed_from_below:
        under_radius=radius-.004;patch={'thumb':(pobj+rw*(radius-.002)+zw*top,-zw),'index':(pobj+rw*under_radius-tw*.009+zw*bottom,zw),'middle':(pobj+rw*under_radius+tw*.009+zw*bottom,zw)}
    patch={'thumb':(pobj+rw*(radius-.004)+zw*top,-zw),**{digit:(pobj+rw*(radius-.004)+tw*t+zw*rim_bottom,zw)for digit,t in zip(['index','middle','ring','pinky'],[-.027,-.009,.009,.027])}}
    q0=seed[qa].copy()
    if not(a.free_support_patches and a.initial_pose_json):q0[narm:]=[1.20,.18,.8,.8,.8,.8]if a.seed_from_below else[.7,.30,1.1,1.1,1.30,1.10]
    q0=np.clip(q0,lower,upper)
    desiredR=np.column_stack([-zw,-tw,-rw]);palm=pobj+rw*(radius+.035)+zw*[.08,.10,.12][(attempt//3)%3]
    if a.seed_from_below:
        angle=np.deg2rad([20,10,30][(attempt//3)%3]);desiredR=np.column_stack([-rw*np.sin(angle)+zw*np.cos(angle),tw,-rw*np.cos(angle)-zw*np.sin(angle)]);palm=pobj+rw*.15;palm[2]=-(plane[0]*palm[0]+plane[1]*palm[1]+plane[3])/plane[2]+.015
    # Disposable arm-only initialization, with exact native joint limits.
    def init_res(q):
        qfull=q0.copy();qfull[:narm]=q;setq(qfull);R=d.xmat[h.hand_bids[1]].reshape(3,3);return np.r_[20*(h.palm(d,1)-palm),.8*Rotation.from_matrix(desiredR@R.T).as_rotvec()]
    if not(a.free_support_patches and a.initial_pose_json):
        init=least_squares(init_res,q0[:narm],bounds=(lower[:narm],upper[:narm]),max_nfev=90,ftol=1e-7,xtol=1e-7,gtol=1e-7);q0[:narm]=init.x
    print(json.dumps({'attempt':attempt,'arm_initialization_residual':float(np.linalg.norm(init_res(q0[:narm]))),'under_support_radius_m':under_radius if a.seed_from_below else None}),flush=True)
    cache={'q':None};calls=[0]
    def evaluate(q):
        if cache['q']is not None and np.array_equal(cache['q'],q):return cache['r'],cache['J']
        setq(q);r=[];J=[]
        for digit,(point,normal)in patch.items():
            g=targets[digit];v=verts(g);pr=v@normal;pt=v[pr>=pr.max()-1e-10].mean(0);goal=point-normal*.001;mj.mj_jac(m,d,JP,None,pt,int(m.geom_bodyid[g]));jac=JP[:,allv]@K
            if a.free_support_patches:
                # A real planar support region, not an arbitrary exact XYZ
                # landmark that can conflict with the hand's fixed finger spacing.
                height=float((pt-pobj)@zw);wanted=top+.0008 if digit=='thumb'else rim_bottom-.0008;r.append(45*(height-wanted));J.append(45*zw@jac)
                projected=pt-pobj-zw*((pt-pobj)@zw);rho=float(np.linalg.norm(projected));radial_jac=projected@jac/max(rho,1e-10);lo,hi=(radius-.008,radius+.001)
                for sign,bound in [(-1.,lo),(1.,hi)]:
                    err=sign*(rho-bound);r.append(30*max(err,0));J.append(30*sign*radial_jac if err>0 else np.zeros(len(q)))
                if digit!='thumb'and a.support_radius_target_m is not None:r.append(20*(rho-a.support_radius_target_m));J.append(20*radial_jac)
                # Finger ordering follows native geometry; do not force index
                # and middle to cross by prescribing an assumed tangent sign.
                tangent=float((pt-pobj)@tw);low,high=-.05,.05
                for sign,bound in [(-1.,low),(1.,high)]:
                    err=sign*(tangent-bound);r.append(15*max(err,0));J.append(15*sign*tw@jac if err>0 else np.zeros(len(q)))
                if digit=='thumb':
                    gap,row=min((oracle.query(d,g,og,allv)for og in h.object_geoms[0]),key=lambda value:value[0]);r.append(60*(gap-.001));J.append(60*row@K)
            else:r.extend(35*(pt-goal));J.extend(35*jac)
        for g in meshes:
            lo,hi=aabb(g);near=np.linalg.norm(np.maximum(np.maximum(lo-ohi,olo-hi),0),axis=1)<.006
            for oi2,og in enumerate(ogeoms):
                gap,row=oracle.query(d,g,og,allv)if near[oi2]else(.006,np.zeros(len(allv)))
                r.append(250*min(gap-.00065,0));J.append(250*row@K if gap<.00065 else np.zeros(len(q)))
        for g in hand:
            pg=plane_gap(g)
            if pg>=.002:gap,row=pg,np.zeros(len(allv))
            elif m.geom_type[g]in(mj.mjtGeom.mjGEOM_MESH,mj.mjtGeom.mjGEOM_CYLINDER):gap,row=oracle.query(d,g,h.table,allv)
            else:
                w=np.zeros(6);gap=float(mj.mj_geomDistance(m,d,g,h.table,1.,w));delta=w[:3]-w[3:];normal=delta/max(np.linalg.norm(delta),1e-9);mj.mj_jac(m,d,JP,None,w[:3],int(m.geom_bodyid[g]));row=normal@JP[:,allv]
            r.append(250*min(gap-.002,0));J.append(250*row@K if gap<.002 else np.zeros(len(q)))
        r.extend(.003*(q-q0));J.extend(.003*np.eye(len(q)));calls[0]+=1;cache.update(q=q.copy(),r=np.asarray(r),J=np.asarray(J))
        if calls[0]%30==0:print(json.dumps({'attempt':attempt,'calls':calls[0],'residual':float(np.linalg.norm(r))}),flush=True)
        return cache['r'],cache['J']
    fit=least_squares(lambda q:evaluate(q)[0],q0,jac=lambda q:evaluate(q)[1],bounds=(lower,upper),max_nfev=a.max_nfev,ftol=2e-6,xtol=2e-6,gtol=2e-6);setq(fit.x);d.qvel[:]=0
    mingap=min(oracle.query(d,g,og)[0]for g in meshes for og in ogeoms);tablegap=min(oracle.query(d,g,h.table)[0]for g in meshes);qfull=d.qpos.copy();original_margin=m.geom_margin.copy();m.geom_margin[meshes]=.0015;mj.mj_forward(m,d)
    cols=contact_columns(m,d,obj,('rh_thumb','rh_index','rh_middle','rh_ring','rh_pinky'),projection=P,max_gap_m=.003);lp=wrench_lp(cols['columns'],cols['normal_rows'],np.r_[m.body_mass[obj]*m.opt.gravity,np.zeros(3)],normal_cap_N=None,motor_torque_matrix=cols['motor_torque_matrix'],motor_gravity_bias_Nm=P.T@d.qfrc_bias,motor_cap_Nm=.4);m.geom_margin[:]=original_margin
    Rhand=d.xmat[h.hand_bids[1]].reshape(3,3);quat=Rotation.from_matrix(Robj.T@Rhand).as_quat()[[3,0,1,2]];template={'hand':'right','palm_offset_in_object_m':((h.palm(d,1)-pobj)@Robj).tolist(),'hand_rotation_in_object_wxyz':quat.tolist(),'finger_joint_targets':{names[j]:float(qfull[m.jnt_qposadr[j]])for j in finger},'controlled_digits':['thumb','index','middle','ring','pinky']};good=bool(mingap>=.0004 and tablegap>=.0018 and lp['feasible'])
    row={'attempt':attempt,'residual_norm':float(np.linalg.norm(fit.fun)),'minimum_hand_all_six_objects_SAT_m':float(mingap),'minimum_mesh_real_table_prism_SAT_m':float(tablegap),'wrench':lp,'prospective_contacts':cols['patches'],'geometry_and_hard_motor_wrench_pass':good,'qpos':qfull.tolist(),'template':template,'required_waist_reference_rad':{names[j]:float(qfull[m.jnt_qposadr[j]])for j in waist},'queries':calls[0]};results.append(row);print(json.dumps({k:row[k]for k in ['attempt','residual_norm','minimum_hand_all_six_objects_SAT_m','minimum_mesh_real_table_prism_SAT_m','wrench','geometry_and_hard_motor_wrench_pass']}),flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({'status':'candidate_requires_physics'if good else'search_in_progress','scope':'Private geometry and static LP, not executed motion; no physical object initialization is changed. Prospective contact query margins .0015m per hand geom are restored.','model':str(a.model),'model_sha256':hashlib.sha256(a.model.read_bytes()).hexdigest(),'pre_slide_planning_translation_m':(a.pre_slide_m*outward).tolist(),'pre_tilt_planning_deg':a.pre_tilt_deg,'pre_tilt_pivot_translation_m':tilt_translation.tolist(),'best':min(results,key=lambda x:(not x['geometry_and_hard_motor_wrench_pass'],x['wrench']['residual_force_equivalent_L1']+max(0,.0004-x['minimum_hand_all_six_objects_SAT_m'])*500+x['residual_norm'])),'attempts':results},indent=2)+'\n')
    if good:break
