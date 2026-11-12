"""Original-position wall push planning without imposing an arbitrary hand pose."""
from pathlib import Path
import argparse,copy,json,hashlib
import numpy as np
import mujoco as mj
from scipy.spatial import ConvexHull
from scipy.optimize import least_squares
from fourth_convex_separation import ConvexSeparationOracle

R=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--trial',type=Path,default=R/'outputs/fourth_full_sequence/physics_sonic15225_o6_obj1_pinchmask_v3_6s')
p.add_argument('--base-xy',nargs=2,type=float,default=[-.35,2.2]);p.add_argument('--max-nfev',type=int,default=140)
p.add_argument('--output',type=Path,default=R/'reports/fourth_o6_contact_preslide_free_v1.json')
p.add_argument('--settled-objects',type=Path);p.add_argument('--warm-plan',type=Path)
p.add_argument('--minimum-table-gap',type=float,default=.0018);p.add_argument('--minimum-other-gap',type=float,default=.0007);a=p.parse_args()
source=json.loads((a.trial/'report.json').read_text());mp=Path(source['model']['path']);m=mj.MjModel.from_binary_path(str(mp));z=dict(np.load(a.trial/'physical_trace.npz',allow_pickle=False));q0=z['qpos'][0].copy();q0[:2]=a.base_xy
if a.settled_objects:q0[59:]=np.load(a.settled_objects)['final_object_pose_world_xyz_wxyz'].reshape(-1)
names=[m.joint(j).name or''for j in range(m.njnt)];arm=[j for j,n in enumerate(names)if n.startswith(('right_shoulder','right_elbow','right_wrist'))];hand=[j for j,n in enumerate(names)if n.startswith('rh_')]
masters=[names.index('rh_'+n)for n in ['thumb_cmc_yaw','thumb_cmc_pitch','index_mcp_pitch','middle_mcp_pitch','ring_mcp_pitch','pinky_mcp_pitch']]
selected=arm+[masters[2]];qa=m.jnt_qposadr[selected];av=m.jnt_dofadr[arm];hv=m.jnt_dofadr[hand];hq=m.jnt_qposadr[hand];K=np.zeros((len(hand),6));couplings=[]
for i,j in enumerate(hand):
    if j in masters:K[i,masters.index(j)]=1
for e in range(m.neq):
    if m.eq_type[e]==mj.mjtEq.mjEQ_JOINT:
        c,pa=int(m.eq_obj1id[e]),int(m.eq_obj2id[e]);ratio=float(m.eq_data[e,1]);couplings.append((c,pa,ratio))
        if c in hand and pa in masters:K[hand.index(c),masters.index(pa)]=ratio
aa=np.asarray([mj.mj_name2id(m,mj.mjtObj.mjOBJ_ACTUATOR,'motor_'+names[j])for j in arm]);ma=np.asarray([mj.mj_name2id(m,mj.mjtObj.mjOBJ_ACTUATOR,'motor_'+names[j])for j in masters])
bn=[m.body(i).name or'world'for i in range(m.nbody)];obj=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'target_object_01');table=mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,'table_support');index=mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,'collision_rh_index_distal_mesh')
robot_geoms=[g for g in range(m.ngeom)if m.geom_contype[g]>0 and bn[int(m.geom_bodyid[g])].startswith(('rh_','right_wrist','right_elbow','right_shoulder'))]
handgeoms=[g for g in robot_geoms if bn[int(m.geom_bodyid[g])].startswith(('rh_','right_wrist'))]
query=copy.copy(m);query.geom_margin[robot_geoms]=max(.003,a.minimum_other_gap+.003);d=mj.MjData(query);oracle=ConvexSeparationOracle(m)
numeric=mj.mj_name2id(m,mj.mjtObj.mjOBJ_NUMERIC,'source_table_plane_abcd');adr=int(m.numeric_adr[numeric]);plane=m.numeric_data[adr:adr+4].copy();plane/=np.linalg.norm(plane[:3]);xy=np.asarray([.35216658756772096,.9359373347617386]);out=np.r_[xy,-plane[:2]@xy/plane[2]];out/=np.linalg.norm(out)
def setq(q):
    d.qpos[:]=q
    for c,pa,ratio in couplings:d.qpos[int(m.jnt_qposadr[c])]=ratio*d.qpos[int(m.jnt_qposadr[pa])]
    mj.mj_kinematics(query,d);mj.mj_comPos(query,d)
def verts(g):return oracle._shape(g)['vertices']@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
setq(q0);objectgeoms=[g for g in range(m.ngeom)if m.geom_contype[g]>0 and m.geom_bodyid[g]==obj];facets=[]
for g in objectgeoms:
    if m.geom_type[g]!=mj.mjtGeom.mjGEOM_MESH:continue
    vv=verts(g);ch=ConvexHull(vv)
    for eq in ch.equations:
        n=-eq[:3]
        if n@out<.3 or n@plane[:3]<.2:continue
        face=vv[abs(vv@eq[:3]+eq[3])<2e-7];h=face@plane[:3]+plane[3]
        if len(face)<3 or h.max()-h.min()<.008:continue
        if any(np.linalg.norm(n-f['normal'])<1e-4 for f in facets):continue
        facets.append({'normal':n,'bottom':face[h<h.min()+.001].mean(0),'top':face[h>h.max()-.001].mean(0),'geom':m.geom(g).name,'score':float(n@out)})
facets.sort(key=lambda f:f['score'],reverse=True);facets=facets[:2]
lower=m.jnt_range[selected,0]+1e-6;upper=m.jnt_range[selected,1]-1e-6;results=[];attempts=[]
for facet in facets:
    n=facet['normal'];t1=np.cross(n,plane[:3]);t1/=np.linalg.norm(t1);t2=np.cross(n,t1)
    for fraction in [.82,.94,.68]:
        target=(1-fraction)*facet['bottom']+fraction*facet['top']
        for other_closed in ([1.6,1.45,1.25]if a.settled_objects else[0.,1.25,.65]):
            q=q0.copy();roots=np.zeros(6);roots[3:]=other_closed
            if other_closed:roots[:2]=[.5,.3]
            q[hq]=K@roots
            seeds=[np.r_[q0[m.jnt_qposadr[arm]],.15],np.asarray([.8,-.5,0.,1.4,0.,0.,0.,.6]),np.asarray([.2,-1.,.6,.6,-.4,.6,.4,1.])]
            if a.warm_plan:seeds.insert(0,np.asarray(json.loads(a.warm_plan.read_text())['best']['arm_and_index_q']))
            for seed in seeds:
                def residual(x):
                    q[qa]=x;setq(q);vv=verts(index);c=vv.mean(0);gap=float(n@target-np.max(vv@n))
                    value=[25*(gap-.001),15*float((c-target)@t1),15*float((c-target)@t2)]
                    for g in handgeoms:
                        if m.geom_type[g]==mj.mjtGeom.mjGEOM_MESH:clear=float(np.min(verts(g)@plane[:3]+plane[3]))
                        else:
                            rr=d.geom_xmat[g].reshape(3,3);cc=d.geom_xpos[g]+rr@m.geom_aabb[g,:3];ext=np.abs(rr)@m.geom_aabb[g,3:];clear=float(cc@plane[:3]+plane[3]-abs(plane[:3])@ext)
                        value.append(160*min(clear-max(a.minimum_table_gap+.0002,.002),0.))
                    mj.mj_collision(query,d);penalty={g:0. for g in robot_geoms}
                    for contact in d.contact[:d.ncon]:
                        g1,g2=int(contact.geom1),int(contact.geom2);b1,b2=int(m.geom_bodyid[g1]),int(m.geom_bodyid[g2])
                        if (g1==index and b2==obj)or(g2==index and b1==obj):
                            if a.settled_objects:penalty[index]=min(penalty[index],float(contact.dist)-.0007)
                            continue
                        for gg,other in [(g1,g2),(g2,g1)]:
                            if gg in penalty and (m.geom(other).name or'')!='floor':penalty[gg]=min(penalty[gg],float(contact.dist)-max(.0015,a.minimum_other_gap+.0002))
                    value.extend(60*v for v in penalty.values());value.extend(.002*(x-seed));return np.asarray(value)
                fit=least_squares(residual,np.clip(seed,lower,upper),bounds=(lower,upper),max_nfev=a.max_nfev,ftol=1e-6,xtol=1e-6,gtol=1e-6)
                q[qa]=fit.x;setq(q);vv=verts(index);c=vv.mean(0);plane_gap=float(n@target-np.max(vv@n));tangent_error=float(np.linalg.norm([(c-target)@t1,(c-target)@t2]))
                minimum_table=1.;othergap=1.;indexgap=1.
                for g in handgeoms:
                    if m.geom_type[g]==mj.mjtGeom.mjGEOM_MESH:
                        gap,_=oracle.query(d,g,table)
                    else:
                        w=np.zeros(6);gap=float(mj.mj_geomDistance(m,d,g,table,2.,w))
                    minimum_table=min(minimum_table,gap)
                    for og in objectgeoms:
                        if m.geom_type[g]==mj.mjtGeom.mjGEOM_MESH:gap,_=oracle.query(d,g,og)
                        else:
                            w=np.zeros(6);gap=float(mj.mj_geomDistance(m,d,g,og,2.,w))
                        if g==index:indexgap=min(indexgap,gap)
                        else:othergap=min(othergap,gap)
                valid=minimum_table>=a.minimum_table_gap and othergap>=a.minimum_other_gap and .0002<=indexgap<=.0035 and tangent_error<.01
                row={'arm_and_index_q':fit.x.tolist(),'residual_norm':float(np.linalg.norm(fit.fun)),'minimum_table_gap_m':minimum_table,
                     'minimum_nonpusher_object_gap_m':othergap,'index_object_gap_m':indexgap,'normal_plane_gap_m':plane_gap,
                     'contact_tangent_alignment_error_m':tangent_error,'other_finger_master_angles_rad':roots.tolist(),'fraction':fraction,'valid_geometry':bool(valid)}
                attempts.append(row);print(json.dumps({k:row[k]for k in ['residual_norm','minimum_table_gap_m','minimum_nonpusher_object_gap_m','index_object_gap_m','contact_tangent_alignment_error_m','valid_geometry']}),flush=True)
                if not valid:continue
                # Re-evaluate gravity with the real physical model, not query margins.
                actual=mj.MjData(m);actual.qpos[:]=d.qpos;actual.qvel[:]=0;mj.mj_forward(m,actual);JP=np.zeros((3,m.nv));JR=np.zeros_like(JP);mj.mj_jac(m,actual,JP,JR,target,int(m.geom_bodyid[index]))
                N=2.1;F=N*n;handtau=K.T@(actual.qfrc_bias[hv]+JP[:,hv].T@F);armtau=actual.qfrc_bias[av]+JP[:,av].T@F
                torque_ok=np.max(abs(handtau))<=.4 and np.all(abs(armtau)<=np.max(abs(m.actuator_forcerange[aa]),axis=1))
                all_robot_scene=[]
                for contact in actual.contact[:actual.ncon]:
                    gs=[int(contact.geom1),int(contact.geom2)];bs=m.geom_bodyid[gs]
                    if any((b>0 and not bn[int(b)].startswith('target_object_'))for b in bs)and any((b==0 or bn[int(b)].startswith('target_object_'))for b in bs):
                        if not any((m.geom(g).name or'')=='floor'for g in gs)and contact.dist<-.0001:all_robot_scene.append([m.geom(g).name for g in gs])
                row.update(qpos=actual.qpos.tolist(),hand_independent_target_rad=actual.qpos[m.jnt_qposadr[masters]].tolist(),
                    contact_surface_target_world_m=target.tolist(),desired_force_on_object_unit=n.tolist(),surface_geom=facet['geom'],
                    required_hand_motor_torque_Nm=handtau.tolist(),required_arm_motor_torque_Nm=armtau.tolist(),test_normal_N=N,
                    hard_motor_torque_feasible=bool(torque_ok),actual_nonfloor_robot_scene_penetrations=all_robot_scene)
                if torque_ok and not all_robot_scene:results.append(row);break
            if results:break
        if results:break
    if results:break
best=results[0]if results else None
report={'schema':'fourth.o6_contact_preslide_free_hand_private_plan.v1','model':str(mp),'model_sha256':hashlib.sha256(mp.read_bytes()).hexdigest(),
        'background_trial':str(a.trial),'base_xy':a.base_xy,'best':best,'attempts':attempts,'outward_tangent_unit':out.tolist(),'table_plane':plane.tolist(),
        'source_object_initial_poses_unchanged':True,'private_collision_query_margin_m':.003,'physical_slide_performed':False}
report['original_object_settling_source']=str(a.settled_objects)if a.settled_objects else None
report['raw_initialization_then_settle_required_for_physics']=a.settled_objects is not None
a.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
if best:np.savez_compressed(a.output.with_suffix('.npz'),initial_qpos=np.asarray(best['qpos']))
print(json.dumps({'status':'safe_original_position_push_plan_found'if best else'no_safe_original_position_push_plan','output':str(a.output),'attempts':len(attempts)}),flush=True)
