"""Raise an original-position pusher tangentially along its measured wall."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import mujoco as mj
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from fourth_convex_separation import ConvexSeparationOracle
R=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--input',type=Path,default=R/'reports/fourth_o6_contact_preslide_free_v1.json')
p.add_argument('--output',type=Path,default=R/'reports/fourth_o6_contact_preslide_clearance_v2.json');a=p.parse_args()
s=json.loads(a.input.read_text());old=s['best'];m=mj.MjModel.from_binary_path(s['model']);d=mj.MjData(m);q=np.asarray(old['qpos']);d.qpos[:]=q;mj.mj_forward(m,d)
arm=[j for j in range(m.njnt)if(m.joint(j).name or'').startswith(('right_shoulder','right_elbow','right_wrist'))];qa=m.jnt_qposadr[arm];av=m.jnt_dofadr[arm]
site=mj.mj_name2id(m,mj.mjtObj.mjOBJ_SITE,'rh_palm_control_site');targetR=d.site_xmat[site].reshape(3,3).copy();pos=d.site_xpos[site].copy();up=np.asarray(s['table_plane'][:3]);out=np.asarray(s['outward_tangent_unit']);n=np.asarray(old['desired_force_on_object_unit'])
delta=.0015*(up-out*(n@up)/(n@out));target=pos+delta;seed=q[qa].copy()
def f(x):
    d.qpos[:]=q;d.qpos[qa]=x;mj.mj_kinematics(m,d)
    return np.r_[d.site_xpos[site]-target,.2*Rotation.from_matrix(targetR@d.site_xmat[site].reshape(3,3).T).as_rotvec(),1e-5*(x-seed)]
fit=least_squares(f,seed,bounds=(m.jnt_range[arm,0]+1e-6,m.jnt_range[arm,1]-1e-6),max_nfev=120,xtol=1e-11,gtol=1e-11,ftol=1e-11)
d.qpos[:]=q;d.qpos[qa]=fit.x;d.qvel[:]=0;mj.mj_forward(m,d);oracle=ConvexSeparationOracle(m)
bn=[m.body(b).name or'world'for b in range(m.nbody)];hand=[g for g in range(m.ngeom)if m.geom_contype[g]>0 and bn[int(m.geom_bodyid[g])].startswith(('rh_','right_wrist'))];objs=[g for g in range(m.ngeom)if m.geom_contype[g]>0 and bn[int(m.geom_bodyid[g])]=='target_object_01'];table=mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,'table_support');ig=mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,'collision_rh_index_distal_mesh')
minimum_table=(np.inf,None);minimum_nonpusher=(np.inf,None);minimum_index=(np.inf,None)
for g in hand:
    def gap(og):
        if m.geom_type[g]==mj.mjtGeom.mjGEOM_MESH:return oracle.query(d,g,og)[0]
        return float(mj.mj_geomDistance(m,d,g,og,2.,np.zeros(6)))
    distance=gap(table)
    if distance<minimum_table[0]:minimum_table=(distance,m.geom(g).name)
    for og in objs:
        distance=gap(og)
        if g==ig and distance<minimum_index[0]:minimum_index=(distance,m.geom(og).name)
        if g!=ig and distance<minimum_nonpusher[0]:minimum_nonpusher=(distance,[m.geom(g).name,m.geom(og).name])
masters=[mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,'rh_'+name)for name in ['thumb_cmc_yaw','thumb_cmc_pitch','index_mcp_pitch','middle_mcp_pitch','ring_mcp_pitch','pinky_mcp_pitch']];P=np.zeros((m.nv,6))
for col,j in enumerate(masters):P[int(m.jnt_dofadr[j]),col]=1
for e in range(m.neq):
    if m.eq_type[e]==mj.mjtEq.mjEQ_JOINT and m.eq_obj2id[e]in masters:P[int(m.jnt_dofadr[int(m.eq_obj1id[e])]),masters.index(int(m.eq_obj2id[e]))]=m.eq_data[e,1]
surface=np.asarray(old['contact_surface_target_world_m'])+delta;JP=np.zeros((3,m.nv));JR=np.zeros_like(JP);mj.mj_jac(m,d,JP,JR,surface,int(m.geom_bodyid[ig]));F=2.1*n;handtau=P.T@(d.qfrc_bias+JP.T@F);armtau=d.qfrc_bias[av]+JP[:,av].T@F
hits=[]
for c in d.contact[:d.ncon]:
    gs=[int(c.geom1),int(c.geom2)];bs=m.geom_bodyid[gs]
    if any(b>0 and not bn[int(b)].startswith('target_object_')for b in bs)and any(b==0 or bn[int(b)].startswith('target_object_')for b in bs)and c.dist<-.0001 and not any((m.geom(g).name or'')=='floor'for g in gs):hits.append({'geoms':[m.geom(g).name for g in gs],'penetration_m':-float(c.dist)})
checks={k:bool(v)for k,v in {'table_clearance_ge_3mm':minimum_table[0]>=.003,'nonpusher_clearance_ge_0_7mm':minimum_nonpusher[0]>=.0007,'pusher_positive_close_gap':.0002<=minimum_index[0]<=.0035,'hand_static_torque_hard_0_4':max(abs(handtau))<=.4,'whole_robot_no_scene_penetration':not hits,'all_object_poses_unchanged':np.array_equal(d.qpos[59:],q[59:])}.items()}
best=dict(old);best.update(qpos=d.qpos.tolist(),arm_and_index_q=np.r_[d.qpos[qa],d.qpos[int(m.jnt_qposadr[masters[2]])]].tolist(),minimum_table_gap_m=minimum_table[0],minimum_nonpusher_object_gap_m=minimum_nonpusher[0],index_object_gap_m=minimum_index[0],contact_surface_target_world_m=surface.tolist(),required_hand_motor_torque_Nm=handtau.tolist(),required_arm_motor_torque_Nm=armtau.tolist())
report=dict(s);report.update(schema='fourth.o6_original_wall_pusher_clearance_refinement.v2',best=best,refinement_translation_m=delta.tolist(),refinement_checks=checks,minimum_table_pair=minimum_table,minimum_nonpusher_pair=minimum_nonpusher,minimum_index_pair=minimum_index,actual_scene_penetrations=hits,physical_slide_performed=False)
a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'checks':checks,'table':minimum_table,'nonpusher':minimum_nonpusher,'index':minimum_index,'handtau':handtau.tolist(),'armtau':armtau.tolist(),'translation':delta.tolist()}),flush=True)
if not all(checks.values()):raise SystemExit(2)
