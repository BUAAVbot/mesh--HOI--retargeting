"""Independent private geometry/wrench audit of the post-slide v7 candidate.

The plate translation is hypothetical planning input, never a physics result.
"""
from pathlib import Path
import argparse,copy,hashlib,json
import numpy as np
import mujoco as mj
from scipy.spatial.transform import Rotation
from fourth_convex_separation import ConvexSeparationOracle
from fourth_contact_wrench import contact_columns,wrench_lp

R=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--candidate',type=Path,default=R/'reports/fourth_o6_load_grasp_under_preslide5cm_v7.json')
p.add_argument('--output',type=Path,default=R/'reports/fourth_o6_preslide_under_grasp_v7_independent_audit.json');a=p.parse_args()
source=json.loads(a.candidate.read_text());q=np.asarray(source['best']['qpos']);mp=Path(source['model']);mp=mp if mp.is_absolute()else R/mp
if hashlib.sha256(mp.read_bytes()).hexdigest()!=source['model_sha256']:raise ValueError('Candidate model hash mismatch')
m=mj.MjModel.from_xml_path(str(mp));d=mj.MjData(m);d.qpos[:]=q;d.qvel[:]=0;mj.mj_forward(m,d)
original=json.loads((R/'reports/fourth_o6_static_grasp_refined_v1.json').read_text())['best'];raw=np.asarray(original['qpos'])
object_rows=[];expected=np.asarray([.017602793494044835,.046782154266720326,-.0012536759432943737])
for oid in range(1,7):
    j=mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,f'target_object_{oid:02d}_free');qa=int(m.jnt_qposadr[j]);delta=q[qa:qa+3]-raw[qa:qa+3]
    object_rows.append({'object_id':oid,'planning_origin_delta_m':delta.tolist(),'planning_displacement_norm_m':float(np.linalg.norm(delta)),
                        'orientation_unchanged':bool(np.array_equal(q[qa+3:qa+7],raw[qa+3:qa+7])),
                        'delta_matches_declared_hypothesis':bool(np.max(abs(delta-(expected if oid==1 else 0.)))<1e-11)})
limits=[];mimic=[];names=[m.joint(j).name or''for j in range(m.njnt)]
for j,name in enumerate(names):
    if m.jnt_type[j]!=mj.mjtJoint.mjJNT_HINGE:continue
    qa=int(m.jnt_qposadr[j]);limits.append({'joint':name,'angle_rad':float(q[qa]),'in_model_range':bool(m.jnt_range[j,0]-1e-10<=q[qa]<=m.jnt_range[j,1]+1e-10)})
for e in range(m.neq):
    if m.eq_type[e]!=mj.mjtEq.mjEQ_JOINT:continue
    c,pa=int(m.eq_obj1id[e]),int(m.eq_obj2id[e]);cq,pq=int(m.jnt_qposadr[c]),int(m.jnt_qposadr[pa]);err=q[cq]-m.qpos0[cq]-np.polynomial.polynomial.polyval(q[pq]-m.qpos0[pq],m.eq_data[e,:5])
    mimic.append({'child':names[c],'parent':names[pa],'residual_rad':float(err)})
hand=[g for g in range(m.ngeom)if m.geom_contype[g]>0 and(m.body(int(m.geom_bodyid[g])).name or'').startswith(('rh_','right_wrist'))]
meshes=[g for g in hand if m.geom_type[g]==mj.mjtGeom.mjGEOM_MESH];table=mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,'table_support');oracle=ConvexSeparationOracle(m)
minimum=(np.inf,None,None);table_values=[]
for g in hand:
    if g in meshes:gap,_=oracle.query(d,g,table)
    else:
        w=np.zeros(6);gap=float(mj.mj_geomDistance(m,d,g,table,2.,w))
    table_values.append({'geom':m.geom(g).name,'finite_prism_clearance_m':gap})
for g in meshes:
    for og in range(m.ngeom):
        if m.geom_contype[og]<=0 or not(m.body(int(m.geom_bodyid[og])).name or'').startswith('target_object_'):continue
        gap,_=oracle.query(d,g,og)
        if gap<minimum[0]:minimum=(gap,m.geom(g).name,m.geom(og).name)
actual_robot_contacts=[]
for c in d.contact[:d.ncon]:
    gs=[int(c.geom1),int(c.geom2)];bs=m.geom_bodyid[gs];bn=[m.body(int(b)).name or'world'for b in bs]
    robot=[b>0 and not n.startswith('target_object_')for b,n in zip(bs,bn)]
    if any(robot)and any(b==0 or n.startswith('target_object_')for b,n in zip(bs,bn))and not any((m.geom(g).name or'')=='floor'for g in gs):
        actual_robot_contacts.append({'geoms':[m.geom(g).name for g in gs],'signed_distance_m':float(c.dist)})
master_names=['rh_'+n for n in ['thumb_cmc_yaw','thumb_cmc_pitch','index_mcp_pitch','middle_mcp_pitch','ring_mcp_pitch','pinky_mcp_pitch']]
masters=[names.index(n)for n in master_names];P=np.zeros((m.nv,6))
for col,j in enumerate(masters):P[int(m.jnt_dofadr[j]),col]=1
for e in range(m.neq):
    if m.eq_type[e]==mj.mjtEq.mjEQ_JOINT:
        c,pa=int(m.eq_obj1id[e]),int(m.eq_obj2id[e])
        if pa in masters:P[int(m.jnt_dofadr[c]),masters.index(pa)]=m.eq_data[e,1]
query=copy.copy(m);query.geom_margin[meshes]=.0015;qd=mj.MjData(query);qd.qpos[:]=q;mj.mj_forward(query,qd)
obj=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'target_object_01');items=contact_columns(m,qd,obj,('rh_thumb','rh_index','rh_middle'),P,max_gap_m=.0015)
external=np.r_[m.body_mass[obj]*m.opt.gravity,np.zeros(3)];bias=P.T@d.qfrc_bias
lp=wrench_lp(items['columns'],items['normal_rows'],external,normal_cap_N=3.,motor_torque_matrix=items['motor_torque_matrix'],motor_gravity_bias_Nm=bias,motor_cap_Nm=.4)
triangle_area=None
if len(items['patches'])==3:
    points=np.asarray([v['point_world_m']for v in items['patches']]);triangle_area=float(np.linalg.norm(np.cross(points[1]-points[0],points[2]-points[0]))/2)
template=source['best']['template'];site=mj.mj_name2id(m,mj.mjtObj.mjOBJ_SITE,'rh_palm_control_site');base=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'rh_hand_base_link');objR=d.xmat[obj].reshape(3,3)
local=objR.T@(d.site_xpos[site]-d.xpos[obj]);targetR=Rotation.from_quat(np.asarray(template['hand_rotation_in_object_wxyz'])[[1,2,3,0]]).as_matrix()
position_error=float(np.linalg.norm(local-template['palm_offset_in_object_m']));rotation_error=float(np.linalg.norm(objR.T@d.xmat[base].reshape(3,3)-targetR))
checks={'all_hypothetical_object_deltas_match':all(row['delta_matches_declared_hypothesis']and row['orientation_unchanged']for row in object_rows),
        'all_native_hinge_limits':all(row['in_model_range']for row in limits),'all_ten_mimic_relations':len(mimic)==10 and max(abs(row['residual_rad'])for row in mimic)<1e-10,
        'rh_mesh_all_objects_nonpenetration':minimum[0]>=.0005,'finite_table_prism_clearance':min(row['finite_prism_clearance_m']for row in table_values)>=.002,
        'no_actual_robot_scene_contacts':not actual_robot_contacts,'three_noncollinear_prospective_patches':len(items['patches'])>=3 and triangle_area is not None and triangle_area>1e-6,
        'gravity_wrench_and_K_projected_hard_motor_limits':lp['feasible'],'template_pose_contract':position_error<1e-10 and rotation_error<1e-10}
report={'schema':'fourth.o6_hypothetical_preslide_undergrasp_independent_audit.v1','candidate':str(a.candidate),'model':str(mp),'model_sha256':source['model_sha256'],
        'checks':checks,'private_candidate_passed':all(checks.values()),'object_planning_deltas':object_rows,'mimic_residuals':mimic,'native_limits':limits,
        'rh_mesh_all_six_objects_minimum_sat_m':minimum[0],'limiting_pair':minimum[1:],'hand_wrist_finite_table_clearances':table_values,
        'actual_robot_scene_contacts_at_positive_gaps':actual_robot_contacts,'prospective_collision_query_margin_m':.0015,'prospective_patches':items['patches'],
        'prospective_triangle_area_m2':triangle_area,'wrench_lp_with_static_gravity_and_K_motor_caps':lp,'motor_gravity_bias_Nm':bias.tolist(),
        'template_palm_error_m':position_error,'template_rotation_matrix_error':rotation_error,
        'physical_preslide_or_grasp_performed':False,'actual_object_displacement_claimed_m':0.,
        'limitation':'Only valid after the object physically slides the declared5cm; raw reset must keep original object pose. Static motor reserve is narrow and does not prove dynamic success.'}
a.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:report[k]for k in ['private_candidate_passed','checks','rh_mesh_all_six_objects_minimum_sat_m','prospective_patches','prospective_triangle_area_m2','wrench_lp_with_static_gravity_and_K_motor_caps']}),flush=True)
if not report['private_candidate_passed']:raise SystemExit(2)
