"""Refine a six-command O6 rim grasp with exact convex SAT inequalities.

Only disposable FK state is changed. Every physical collision part is retained;
positive-gap witness normals are used for the pre-grasp (no force claim).
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import mujoco as mj
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from fourth_arm_object_feedback import ArmObjectFeedback
from fourth_convex_separation import ConvexSeparationOracle
p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--search',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--max-nfev',type=int,default=120);p.add_argument('--candidates',type=int,default=5);a=p.parse_args()
m=mj.MjModel.from_xml_path(str(a.model.resolve()));d=mj.MjData(m);ref=dict(np.load(a.reference,allow_pickle=False));h=ArmObjectFeedback(m,ref);oracle=ConvexSeparationOracle(m);old=json.loads(a.search.read_text());names=[mj.mj_id2name(m,mj.mjtObj.mjOBJ_JOINT,j)for j in range(m.njnt)];arm=h.chains[1]['joint_ids'];masters=[names.index('rh_'+n)for n in ['thumb_cmc_yaw','thumb_cmc_pitch','index_mcp_pitch','middle_mcp_pitch','ring_mcp_pitch','pinky_mcp_pitch']];selected=np.array(arm+masters);qa=m.jnt_qposadr[selected];jv=m.jnt_dofadr[selected];finger=[j for j,n in enumerate(names)if n and n.startswith('rh_')];allj=np.array(arm+finger);allv=m.jnt_dofadr[allj];K=np.zeros((len(allv),len(selected)));couplings=[]
for i,j in enumerate(allj):
    if j in selected:K[i,int(np.flatnonzero(selected==j)[0])]=1.
for eq in range(m.neq):
    if m.eq_type[eq]!=mj.mjtEq.mjEQ_JOINT:continue
    child,master=int(m.eq_obj1id[eq]),int(m.eq_obj2id[eq]);ratio=float(m.eq_data[eq,1]);couplings.append((child,master,ratio))
    if child in allj and master in selected:K[int(np.flatnonzero(allj==child)[0]),int(np.flatnonzero(selected==master)[0])]=ratio
lower=m.jnt_range[selected,0]+1e-5;upper=m.jnt_range[selected,1]-1e-5
for child,master,ratio in couplings:
    if master in selected:
        j=int(np.flatnonzero(selected==master)[0]);lower[j]=max(lower[j],m.jnt_range[child,0]/ratio+1e-5);upper[j]=min(upper[j],m.jnt_range[child,1]/ratio-1e-5)
plane=np.array([.010271048830894367,.02292500590934548,.999684425036206,1.0058665495441206]);plane/=np.linalg.norm(plane[:3]);hand=h.hand_geoms[1];meshes=[g for g in hand if m.geom_type[g]==mj.mjtGeom.mjGEOM_MESH];obj=h.object_bids[0];objectgeoms=h.object_geoms[0]
thumb=next(g for g in meshes if(mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[g]))or'')=='rh_thumb_distal');index=next(g for g in meshes if(mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[g]))or'')=='rh_index_distal')
JP=np.zeros((3,m.nv));JR=np.zeros_like(JP);results=[]
def shape_world(g):
    shape=oracle._shape(g);return shape['vertices']@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
def bounds(g):
    r=d.geom_xmat[g].reshape(3,3);c=d.geom_xpos[g]+r@m.geom_aabb[g,:3];ex=abs(r)@m.geom_aabb[g,3:];return c-ex,c+ex
def setq(seed,q):
    d.qpos[:]=seed;d.qpos[qa]=q
    for child,master,ratio in couplings:d.qpos[m.jnt_qposadr[child]]=ratio*d.qpos[m.jnt_qposadr[master]]
    mj.mj_kinematics(m,d);mj.mj_comPos(m,d)
def table_row(g):
    kind=m.geom_type[g];r=d.geom_xmat[g].reshape(3,3);size=m.geom_size[g];bid=int(m.geom_bodyid[g])
    if kind==mj.mjtGeom.mjGEOM_MESH:
        v=shape_world(g);val=v@plane[:3]+plane[3];minimum=val.min();point=v[val<=minimum+1e-10].mean(axis=0);mj.mj_jac(m,d,JP,None,point,bid);return float(minimum),plane[:3]@JP[:,allv]@K
    centre=float(d.geom_xpos[g]@plane[:3]+plane[3]);mj.mj_jac(m,d,JP,JR,d.geom_xpos[g],bid);row=plane[:3]@JP[:,allv];axis=r[:,2];proj=float(axis@plane[:3]);rot=(np.cross(JR[:,allv].T,axis)@plane[:3])
    if kind==mj.mjtGeom.mjGEOM_CAPSULE:return centre-size[0]-size[1]*abs(proj),(row-size[1]*np.sign(proj)*rot)@K
    if kind==mj.mjtGeom.mjGEOM_CYLINDER:
        radial=np.sqrt(max(1e-12,1-proj*proj));return centre-size[1]*abs(proj)-size[0]*radial,(row-size[1]*np.sign(proj)*rot+size[0]*proj/radial*rot)@K
    raise ValueError('Unsupported hand primitive')
candidates=sorted(old['attempts'],key=lambda row:(row['thumb_index_minimum_normal_dot']>=-.25,row['maximum_hand_object_penetration_m'],row['residual_norm']))[:a.candidates]
for number,source in enumerate(candidates):
    seed=np.asarray(source['qpos']);start=seed[qa].copy();setq(seed,start);pairs=[]
    for g in meshes:
        lo,hi=bounds(g)
        for og in objectgeoms:
            ol,oh=bounds(og);delta=np.maximum(np.maximum(lo-oh,ol-hi),0)
            if np.linalg.norm(delta)<.045:pairs.append((g,og))
    ti=[i for i,(g,_)in enumerate(pairs)if g==thumb];ii=[i for i,(g,_)in enumerate(pairs)if g==index]
    if not ti or not ii:continue
    cache={'q':None};calls=[0]
    def evaluate(q):
        if cache['q']is not None and np.array_equal(q,cache['q']):return cache['value'],cache['jac']
        setq(seed,q);gaps=[];jac=[]
        for g,og in pairs:
            gap,row=oracle.query(d,g,og,allv);gaps.append(gap);jac.append(row@K)
        gaps=np.asarray(gaps);jac=np.asarray(jac);r=[];J=[]
        for gap,row in zip(gaps,jac):r.append(150*min(gap-.0007,0));J.append(150*row if gap<.0007 else np.zeros(len(q)))
        for group in [ti,ii]:
            k=group[int(np.argmin(gaps[group]))];r.append(60*(gaps[k]-.001));J.append(60*jac[k])
        for g in hand:
            gap,row=table_row(g);r.append(250*min(gap-.002,0));J.append(250*row if gap<.002 else np.zeros(len(q)))
        r.extend(.02*(q-start));J.extend(.02*np.eye(len(q)))
        cache.update(q=q.copy(),value=np.asarray(r),jac=np.asarray(J));calls[0]+=1
        if calls[0]%30==0:print(json.dumps({'candidate':number,'calls':calls[0],'minimum_sat_m':float(gaps.min()),'thumb_gap_m':float(min(gaps[ti])),'index_gap_m':float(min(gaps[ii]))}),flush=True)
        return cache['value'],cache['jac']
    fit=least_squares(lambda q:evaluate(q)[0],np.clip(start,lower,upper),jac=lambda q:evaluate(q)[1],bounds=(lower,upper),max_nfev=a.max_nfev,ftol=1e-6,xtol=1e-6,gtol=1e-6)
    setq(seed,fit.x);allgaps=[oracle.query(d,g,og)[0]for g,og in pairs];mingap=float(min(oracle.query(d,g,og)[0]for g in meshes for og in objectgeoms));clear=float(min(table_row(g)[0]for g in hand));normal={};closest={}
    for digit,group in [('thumb',ti),('index',ii)]:
        order=sorted(group,key=lambda k:allgaps[k]);found=None
        for k in order:
            g,og=pairs[k];w=np.zeros(6);distance=float(mj.mj_geomDistance(m,d,g,og,.1,w));delta=w[3:]-w[:3]
            if distance>1e-6 and abs(np.linalg.norm(delta)-distance)<1e-4:found=(distance,delta/np.linalg.norm(delta),g,og);break
        if found is not None:closest[digit]={'distance_m':found[0],'hand_geom':m.geom(found[2]).name,'object_geom':m.geom(found[3]).name};normal[digit]=found[1]
    dot=float(normal['thumb']@normal['index'])if len(normal)==2 else 1.
    qfull=d.qpos.copy();palm=h.palm(d,1);obp=d.xpos[obj].copy();obr=d.xmat[obj].reshape(3,3).copy();hr=d.xmat[h.hand_bids[1]].reshape(3,3);quat=Rotation.from_matrix(obr.T@hr).as_quat()[[3,0,1,2]]
    feasible=bool(mingap>=.0004 and clear>=.0019 and dot<-.25 and len(closest)==2 and max(v['distance_m']for v in closest.values())<=.0025)
    template={'hand':'right','palm_offset_in_object_m':((palm-obp)@obr).tolist(),'hand_rotation_in_object_wxyz':quat.tolist(),'finger_joint_targets':{names[j]:float(qfull[m.jnt_qposadr[j]])for j in finger}}
    row={'source_attempt':source['attempt'],'residual_norm':float(np.linalg.norm(fit.fun)),'minimum_all_mesh_object_sat_gap_m':mingap,'minimum_all_hand_table_plane_gap_m':clear,'positive_gap_mujoco_witness_normal_dot':dot,'pregrasp_surface_witnesses':closest,'feasible_positive_gap_pregrasp':feasible,'template':template,'qpos':qfull.tolist(),'selected_joint_targets':{names[j]:float(qfull[m.jnt_qposadr[j]])for j in selected},'query_calls':calls[0]};results.append(row);print(json.dumps({k:row[k]for k in ['source_attempt','residual_norm','minimum_all_mesh_object_sat_gap_m','minimum_all_hand_table_plane_gap_m','positive_gap_mujoco_witness_normal_dot','feasible_positive_gap_pregrasp']}),flush=True)
    if feasible:break
out={'status':'positive_gap_rim_pregrasp_found_not_physics_verified'if any(r['feasible_positive_gap_pregrasp']for r in results)else'no_nonpenetrating_opposed_pregrasp_found','scope':'Private FK/SAT and positive-gap real MuJoCo witness directions; no simulation, contact-force claim or physical collider change','model':str(a.model),'model_sha256':hashlib.sha256(a.model.read_bytes()).hexdigest(),'best':next((r for r in results if r['feasible_positive_gap_pregrasp']),min(results,key=lambda r:r['residual_norm'])),'candidates':results,'oracle':oracle.report()};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');np.savez_compressed(a.output.with_suffix('.npz'),qpos=np.asarray(out['best']['qpos']))
