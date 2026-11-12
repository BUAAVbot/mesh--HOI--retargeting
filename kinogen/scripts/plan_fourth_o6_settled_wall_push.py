"""Private settled-original wall pusher: geometry transfer, wrist roll, clearance."""
from pathlib import Path
import argparse,copy,json,hashlib,itertools
import numpy as np
import mujoco as mj
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from fourth_convex_separation import ConvexSeparationOracle
R=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=R/'reports/fourth_o6_contact_preslide_settled_wall_v5.json');a=p.parse_args()
s=json.loads((R/'reports/fourth_o6_contact_preslide_clearance_v2.json').read_text());m=mj.MjModel.from_binary_path(s['model']);old=np.asarray(s['best']['qpos']);q=old.copy();settled=R/'outputs/fourth_full_sequence/assets/vita_six_objects_newft_source_table_v1_settled_2s.npz';q[59:]=np.load(settled)['final_object_pose_world_xyz_wxyz'].reshape(-1)
bn=[m.body(b).name or'world'for b in range(m.nbody)];names=[m.joint(j).name or''for j in range(m.njnt)];arm=np.asarray([j for j,n in enumerate(names)if n.startswith(('right_shoulder','right_elbow','right_wrist'))]);aq=m.jnt_qposadr[arm];av=m.jnt_dofadr[arm];masters=[names.index('rh_'+n)for n in ['thumb_cmc_yaw','thumb_cmc_pitch','index_mcp_pitch','middle_mcp_pitch','ring_mcp_pitch','pinky_mcp_pitch']];rh=[j for j,n in enumerate(names)if n.startswith('rh_')];P=np.zeros((m.nv,6));couplings=[]
for col,j in enumerate(masters):P[int(m.jnt_dofadr[j]),col]=1
for e in range(m.neq):
    if m.eq_type[e]==mj.mjtEq.mjEQ_JOINT:
        child,parent=int(m.eq_obj1id[e]),int(m.eq_obj2id[e]);ratio=float(m.eq_data[e,1]);couplings.append((child,parent,ratio))
        if parent in masters:P[int(m.jnt_dofadr[child]),masters.index(parent)]=ratio
d=mj.MjData(m);d.qpos[:]=old;mj.mj_forward(m,d);site=mj.mj_name2id(m,mj.mjtObj.mjOBJ_SITE,'rh_palm_control_site');obj=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'target_object_01');oq=int(m.jnt_qposadr[int(m.body_jntadr[obj])]);oldR=d.xmat[obj].reshape(3,3).copy();oldp=d.xpos[obj].copy();oldhand=d.site_xpos[site].copy();oldhandR=d.site_xmat[site].reshape(3,3).copy()
d.qpos[:]=q;mj.mj_forward(m,d);change=d.xmat[obj].reshape(3,3)@oldR.T;basep=d.xpos[obj]+change@(oldhand-oldp);baseR=change@oldhandR;surface=d.xpos[obj]+change@(np.asarray(s['best']['contact_surface_target_world_m'])-oldp);normal=change@np.asarray(s['best']['desired_force_on_object_unit']);up=np.asarray(s['table_plane'][:3]);out=np.asarray(s['outward_tangent_unit']);side=np.cross(out,up);side/=np.linalg.norm(side)
hg=[g for g in range(m.ngeom)if m.geom_contype[g]>0 and bn[int(m.geom_bodyid[g])].startswith(('rh_','right_wrist'))];index=mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,'collision_rh_index_distal_mesh');table=mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,'table_support');query=copy.copy(m);query.geom_margin[hg]=.008;qd=mj.MjData(query);oracle=ConvexSeparationOracle(m);attempts=[];best=None
def bind(x,qb):
    d.qpos[:]=qb;d.qpos[aq]=x
    for child,parent,ratio in couplings:d.qpos[int(m.jnt_qposadr[child])]=ratio*d.qpos[int(m.jnt_qposadr[parent])]
    mj.mj_kinematics(m,d);mj.mj_comPos(m,d)
angles=[0.,.15,-.15,.3,-.3,.45,-.45]
for close,turn,tilt,lift in itertools.product([1.6,1.45,1.25],angles,[0.,.15,-.15],[0.,.0015]):
    qb=q.copy();qb[m.jnt_qposadr[masters[3:]]]=close;rotation=Rotation.from_rotvec(up*turn+side*tilt).as_matrix();targetR=rotation@baseR;target=surface+rotation@(basep-surface)
    # Move along the wall to raise all finger vertices without increasing its
    # penetration along the wall normal.
    target+=lift*(up-out*(normal@up)/(normal@out))
    def f(x):
        bind(x,qb)
        return np.r_[d.site_xpos[site]-target,.2*Rotation.from_matrix(targetR@d.site_xmat[site].reshape(3,3).T).as_rotvec(),1e-6*(x-q[aq])]
    fit=least_squares(f,q[aq],bounds=(m.jnt_range[arm,0]+1e-6,m.jnt_range[arm,1]-1e-6),max_nfev=55,gtol=1e-9,ftol=1e-9,xtol=1e-9);bind(fit.x,qb)
    if np.linalg.norm(fit.fun[:6])>.0005:continue
    # Set a positive plane gap without driving any actual state.
    vertices=oracle._shape(index)['vertices']@d.geom_xmat[index].reshape(3,3).T+d.geom_xpos[index];gap=float(normal@surface-max(vertices@normal));target+=(gap-.0012)*normal
    fit=least_squares(f,fit.x,bounds=(m.jnt_range[arm,0]+1e-6,m.jnt_range[arm,1]-1e-6),max_nfev=40,gtol=1e-9,ftol=1e-9,xtol=1e-9);bind(fit.x,qb);qd.qpos[:]=d.qpos;mj.mj_kinematics(query,qd);mj.mj_comPos(query,qd);mj.mj_collision(query,qd)
    min_table=.008;min_other=.008;index_contacts=[];whole_hits=[]
    for c in qd.contact[:qd.ncon]:
        gs=[int(c.geom1),int(c.geom2)];bs=[int(query.geom_bodyid[g])for g in gs]
        for g,other,bother in [(gs[0],gs[1],bs[1]),(gs[1],gs[0],bs[0])]:
            if g not in hg:continue
            if other==table:min_table=min(min_table,float(c.dist))
            elif bn[bother].startswith('target_object_'):
                if g==index and bother==obj:
                    sign=1 if bs[1]==obj else-1;index_contacts.append({'gap_m':float(c.dist),'point':c.pos.tolist(),'normal_on_object':(sign*np.asarray(c.frame).reshape(3,3)[0]).tolist(),'geom':m.geom(other).name})
                else:min_other=min(min_other,float(c.dist))
        if c.dist<-.0001 and any(b>0 and not bn[b].startswith('target_object_')for b in bs)and any(b==0 or bn[b].startswith('target_object_')for b in bs)and not any((m.geom(g).name or'')=='floor'for g in gs):whole_hits.append([m.geom(g).name for g in gs])
    min_index=min([x['gap_m']for x in index_contacts]+[.008]);valid=min_table>=.003 and min_other>=.004 and .0005<=min_index<=.0025 and not whole_hits
    row={'other_finger_mcp_rad':close,'palm_world_yaw_adjust_rad':turn,'palm_world_side_tilt_rad':tilt,'surface_raise_m':lift,'min_native_table_gap_m':min_table,'min_native_other_gap_m':min_other,'min_native_index_gap_m':min_index,'candidate_valid':bool(valid)};attempts.append(row)
    if len(attempts)%12==1:print(json.dumps(row),flush=True)
    if not valid:continue
    witness=min(index_contacts,key=lambda x:x['gap_m']);Fdir=np.asarray(witness['normal_on_object'])
    if Fdir@out<.25 or Fdir@up<.2:continue
    # Independent convex distance checks at the accepted pose only.
    sat_table=[];sat_other=[];sat_index=[]
    for g in hg:
        if m.geom_type[g]!=mj.mjtGeom.mjGEOM_MESH:continue
        sat_table.append(oracle.query(d,g,table)[0])
        for og in range(m.ngeom):
            if m.geom_contype[og]>0 and bn[int(m.geom_bodyid[og])].startswith('target_object_'):
                distance=oracle.query(d,g,og)[0]
                (sat_index if g==index and m.geom_bodyid[og]==obj else sat_other).append(distance)
    if min(sat_table)<.003 or min(sat_other)<.0037 or min(sat_index)<.0005:continue
    mj.mj_forward(m,d);JP=np.zeros((3,m.nv));JR=np.zeros_like(JP);point=np.asarray(witness['point']);mj.mj_jac(m,d,JP,JR,point,int(m.geom_bodyid[index]));htau=P.T@(d.qfrc_bias+JP.T@(2.1*Fdir));atau=d.qfrc_bias[av]+JP[:,av].T@(2.1*Fdir)
    if max(abs(htau))>.4:continue
    aa=np.asarray([mj.mj_name2id(m,mj.mjtObj.mjOBJ_ACTUATOR,'motor_'+names[j])for j in arm])
    if np.any(abs(atau)>np.max(abs(m.actuator_forcerange[aa]),axis=1)):continue
    best={**row,'qpos':d.qpos.tolist(),'arm_and_index_q':np.r_[d.qpos[aq],d.qpos[int(m.jnt_qposadr[masters[2]])]].tolist(),'hand_independent_target_rad':d.qpos[m.jnt_qposadr[masters]].tolist(),'required_hand_motor_torque_Nm':htau.tolist(),'required_arm_motor_torque_Nm':atau.tolist(),'desired_force_on_object_unit':Fdir.tolist(),'contact_surface_target_world_m':point.tolist(),'prospective_index_patches':index_contacts,'minimum_table_gap_m':min(sat_table),'minimum_nonpusher_object_gap_m':min(sat_other),'index_object_gap_m':min(sat_index),'test_normal_N':2.1,'hard_motor_torque_feasible':True,'actual_nonfloor_robot_scene_penetrations':whole_hits};break
report={**s,'schema':'fourth.o6_original_settled_wall_pusher_private_plan.v5','best':best,'attempts':attempts,'original_object_settling_source':str(settled),'physical_slide_performed':False,'raw_initialization_then_settle_required_for_physics':True}
a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':'found'if best else'no_safe_pusher','attempts':len(attempts),'best':{k:v for k,v in best.items()if k!='qpos'}if best else None}),flush=True)
