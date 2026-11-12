"""Private FK plan for an actual outer-wall index push; objects stay unshifted.

This is only a geometry/static-torque receipt. A subsequent independent
actuator-only rollout must create any slide through real contacts.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import mujoco as mj
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from fourth_convex_separation import ConvexSeparationOracle

R=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--trial',type=Path,default=R/'outputs/fourth_full_sequence/physics_sonic15225_o6_obj1_pinchmask_v3_6s')
p.add_argument('--base-xy',nargs=2,type=float,default=[-.35,2.2])
p.add_argument('--output',type=Path,default=R/'reports/fourth_o6_contact_preslide_plan_v1.json');a=p.parse_args()
source=json.loads((a.trial/'report.json').read_text());modelpath=Path(source['model']['path']);m=mj.MjModel.from_binary_path(str(modelpath));z=dict(np.load(a.trial/'physical_trace.npz',allow_pickle=False))
q0=z['qpos'][0].copy();q0[:2]=a.base_xy;d=mj.MjData(m);d.qpos[:]=q0;mj.mj_kinematics(m,d);mj.mj_comPos(m,d)
names=[m.joint(j).name or''for j in range(m.njnt)]
arm=[j for j,n in enumerate(names)if n.startswith(('right_shoulder','right_elbow','right_wrist'))];aq=m.jnt_qposadr[arm];av=m.jnt_dofadr[arm];aa=np.asarray([mj.mj_name2id(m,mj.mjtObj.mjOBJ_ACTUATOR,'motor_'+names[j])for j in arm])
hand=[j for j,n in enumerate(names)if n.startswith('rh_')];hq=m.jnt_qposadr[hand];hv=m.jnt_dofadr[hand]
master=[names.index('rh_'+n)for n in ['thumb_cmc_yaw','thumb_cmc_pitch','index_mcp_pitch','middle_mcp_pitch','ring_mcp_pitch','pinky_mcp_pitch']];mq=m.jnt_qposadr[master]
K=np.zeros((len(hand),6));couplings=[]
for i,j in enumerate(hand):
    if j in master:K[i,master.index(j)]=1
for e in range(m.neq):
    if m.eq_type[e]==mj.mjtEq.mjEQ_JOINT:
        child,parent=int(m.eq_obj1id[e]),int(m.eq_obj2id[e]);factor=float(m.eq_data[e,1]);couplings.append((child,parent,factor))
        if child in hand and parent in master:K[hand.index(child),master.index(parent)]=factor
ma=np.asarray([mj.mj_name2id(m,mj.mjtObj.mjOBJ_ACTUATOR,'motor_'+names[j])for j in master])
base=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'rh_hand_base_link');index=mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,'collision_rh_index_distal_mesh');obj=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'target_object_01')
numeric=mj.mj_name2id(m,mj.mjtObj.mjOBJ_NUMERIC,'source_table_plane_abcd');adr=int(m.numeric_adr[numeric]);plane=m.numeric_data[adr:adr+4].copy();plane/=np.linalg.norm(plane[:3])
outxy=np.asarray([.35216658756772096,.9359373347617386]);out=np.r_[outxy,-float(plane[:2]@outxy)/plane[2]];out/=np.linalg.norm(out)
table=mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,'table_support');handgeoms=[g for g in range(m.ngeom)if m.geom_contype[g]>0 and(m.body(int(m.geom_bodyid[g])).name or'').startswith(('rh_','right_wrist'))]
objgeoms=[g for g in range(m.ngeom)if m.geom_contype[g]>0 and int(m.geom_bodyid[g])==obj]
oracle=ConvexSeparationOracle(m)
def world(g):
    s=oracle._shape(g);return s['vertices']@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
def setq(q):
    d.qpos[:]=q
    for child,parent,factor in couplings:d.qpos[int(m.jnt_qposadr[child])]=factor*d.qpos[int(m.jnt_qposadr[parent])]
    mj.mj_kinematics(m,d);mj.mj_comPos(m,d)
facets=[]
for g in objgeoms:
    if m.geom_type[g]!=mj.mjtGeom.mjGEOM_MESH:continue
    vv=world(g);hull=ConvexHull(vv)
    for eq in hull.equations:
        normal=-eq[:3];up=float(normal@plane[:3]);forward=float(normal@out)
        if up<.2 or forward<.25:continue
        ids=np.flatnonzero(abs(vv@eq[:3]+eq[3])<2e-7);face=vv[ids]
        heights=face@plane[:3]+plane[3]
        if len(face)<3 or heights.max()-heights.min()<.008:continue
        key=tuple(np.round(normal,5))
        if any(row['key']==key for row in facets):continue
        bottom=face[heights<heights.min()+.001].mean(0);top=face[heights>heights.max()-.001].mean(0)
        facets.append({'key':key,'normal':normal,'bottom':bottom,'top':top,'geom':m.geom(g).name,'score':forward+.3*up})
facets.sort(key=lambda row:row['score'],reverse=True);facets=facets[:2]
X=-plane[:3];Z=out;Y=np.cross(Z,X);Y/=np.linalg.norm(Y);Rbase=np.column_stack([X,Y,Z])
results=[];rejected=[];statistics={'ik_trials':0,'ik_pose_passed':0,'clearance_tested':0};lower=m.jnt_range[arm,0]+1e-6;upper=m.jnt_range[arm,1]-1e-6
for facet in facets:
    for angle in [.3,.5,.7,.9,.06,.10,.14,.18]:
        for height_fraction in [.72,.84,.94]:
            for roll in [0.,-.22,.22]:
                q=q0.copy();roots=np.zeros(6);roots[2]=angle
                if angle>=.3:roots[3:]=1.3
                q[hq]=K@roots;setq(q)
                direction=-1 if angle>=.3 else 1
                Rfamily=np.column_stack([X,direction*Y,direction*Z])
                handR=Rfamily@Rotation.from_rotvec([0,0,roll]).as_matrix();currentR=d.xmat[base].reshape(3,3);local=(world(index)-d.xpos[base])@currentR
                facepoint=(1-height_fraction)*facet['bottom']+height_fraction*facet['top'];normal=facet['normal']
                dots=(local@handR.T)@normal;support=local[dots>=dots.max()-1e-7].mean(0)
                target=facepoint-normal*.001-handR@support
                seeds=[q[aq].copy(),np.clip(np.asarray([.2,-.35,0.,.7,0.,0.,0.]),lower,upper)]
                for seed in seeds:
                    def residual(x):
                        q[aq]=x;setq(q);rot=handR.T@d.xmat[base].reshape(3,3)
                        return np.r_[d.xpos[base]-target,.10*Rotation.from_matrix(rot).as_rotvec(),.0005*(x-seed)]
                    fit=least_squares(residual,np.clip(seed,lower,upper),bounds=(lower,upper),max_nfev=80,ftol=1e-7,xtol=1e-7,gtol=1e-7)
                    q[aq]=fit.x;setq(q);position=float(np.linalg.norm(d.xpos[base]-target));rot_error=float(np.linalg.norm(Rotation.from_matrix(handR.T@d.xmat[base].reshape(3,3)).as_rotvec()))
                    statistics['ik_trials']+=1
                    if position>.002 or rot_error>.02:
                        rejected.append({'stage':'ik','position_m':position,'rotation_rad':rot_error,'angle':angle,'height_fraction':height_fraction,'roll':roll});continue
                    statistics['ik_pose_passed']+=1
                    tablegap=1.;othergap=1.;indexgap=1.;bad=False
                    for g in handgeoms:
                        if m.geom_type[g]==mj.mjtGeom.mjGEOM_MESH:
                            gg,_=oracle.query(d,g,table)
                        else:
                            w=np.zeros(6);gg=float(mj.mj_geomDistance(m,d,g,table,2.,w))
                        tablegap=min(tablegap,gg)
                        for og in objgeoms:
                            if m.geom_type[g]==mj.mjtGeom.mjGEOM_MESH:gap,_=oracle.query(d,g,og)
                            else:
                                w=np.zeros(6);gap=float(mj.mj_geomDistance(m,d,g,og,2.,w))
                            if g==index:indexgap=min(indexgap,gap)
                            else:othergap=min(othergap,gap)
                    statistics['clearance_tested']+=1
                    if tablegap<.0018 or othergap<.001 or indexgap<.0002 or indexgap>.003:
                        rejected.append({'stage':'clearance','table_gap_m':tablegap,'other_gap_m':othergap,'index_gap_m':indexgap,'angle':angle,'height_fraction':height_fraction,'roll':roll});continue
                    d.qvel[:]=0;mj.mj_forward(m,d);JP=np.zeros((3,m.nv));JR=np.zeros_like(JP);mj.mj_jac(m,d,JP,JR,facepoint,int(m.geom_bodyid[index]))
                    requested_N=2.1;F=requested_N*normal;required_hand=K.T@(d.qfrc_bias[hv]+JP[:,hv].T@F);required_arm=d.qfrc_bias[av]+JP[:,av].T@F
                    physical=bool(np.max(abs(required_hand))<=.4 and np.all(abs(required_arm)<=np.max(abs(m.actuator_forcerange[aa]),axis=1)))
                    table_mu=max(float(m.geom_friction[table,0]),min(float(m.geom_friction[g,0])for g in objgeoms));weight=float(m.body_mass[obj])*float(np.linalg.norm(m.opt.gravity))
                    breakaway=table_mu*weight/(float(normal@out)+table_mu*float(normal@plane[:3]))
                    result={'qpos':d.qpos.tolist(),'arm_target_rad':fit.x.tolist(),'finger_independent_targets_rad':roots.tolist(),
                            'hand_position_target_world_m':target.tolist(),'hand_rotation_world_wxyz':Rotation.from_matrix(handR).as_quat()[[3,0,1,2]].tolist(),
                            'contact_surface_target_world_m':facepoint.tolist(),'desired_force_on_object_unit':normal.tolist(),'surface_geom':facet['geom'],
                            'index_mcp_pitch_rad':angle,'height_fraction':height_fraction,'hand_roll_rad':roll,
                            'hand_base_position_error_m':position,'hand_base_rotation_error_rad':rot_error,
                            'minimum_table_sat_gap_m':tablegap,'minimum_nonindex_object_gap_m':othergap,'index_object_gap_m':indexgap,
                            'estimated_table_friction_coefficient':table_mu,'estimated_breakaway_normal_N':breakaway,'test_normal_N':requested_N,
                            'required_six_hand_motor_torque_Nm':required_hand.tolist(),'required_seven_arm_motor_torque_Nm':required_arm.tolist(),
                            'hand_and_arm_motor_static_limits_feasible':physical}
                    results.append(result);print(json.dumps({k:result[k]for k in ['minimum_table_sat_gap_m','minimum_nonindex_object_gap_m','index_object_gap_m','estimated_breakaway_normal_N','hand_and_arm_motor_static_limits_feasible']}),flush=True)
                    if physical:break
                if results and results[-1]['hand_and_arm_motor_static_limits_feasible']:break
            if results and results[-1]['hand_and_arm_motor_static_limits_feasible']:break
        if results and results[-1]['hand_and_arm_motor_static_limits_feasible']:break
    if results and results[-1]['hand_and_arm_motor_static_limits_feasible']:break
best=next((row for row in results if row['hand_and_arm_motor_static_limits_feasible']),None)
report={'schema':'fourth.o6_outer_wall_contact_preslide_private_plan.v1','model':str(modelpath),'model_sha256':hashlib.sha256(modelpath.read_bytes()).hexdigest(),
        'frozen_background_trial':str(a.trial),'robot_initial_base_xy':a.base_xy,'source_object_poses_initially_unmodified':True,
        'outward_surface_tangent_unit':out.tolist(),'table_plane':plane.tolist(),'best':best,'candidates':results,
        'physical_slide_performed':False,'actual_object_displacement_claimed_m':0.,
        'planning_statistics':statistics,'rejected':rejected,
        'scope':'Private IK/collision/static torque plan only; actual contact slide still required.'}
a.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
if best:np.savez_compressed(a.output.with_suffix('.npz'),initial_qpos=np.asarray(best['qpos']))
print(json.dumps({'status':'private_push_plan_found'if best else'no_safe_push_plan_found','output':str(a.output),'candidates':len(results),'wall_facets_considered':len(facets)}),flush=True)
