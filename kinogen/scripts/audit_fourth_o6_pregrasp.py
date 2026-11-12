"""Independent full-model static audit of a saved O6 pregrasp candidate."""
from pathlib import Path
import argparse,hashlib,json
import xml.etree.ElementTree as ET
import numpy as np
import mujoco as mj
from scipy.spatial.transform import Rotation
from fourth_convex_separation import ConvexSeparationOracle

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--candidate',type=Path,default=ROOT/'reports/fourth_o6_static_grasp_refined_v1.json')
p.add_argument('--output',type=Path,default=ROOT/'reports/fourth_o6_static_pregrasp_independent_audit.json')
a=p.parse_args();source=json.loads(a.candidate.read_text());candidate=source['best']
path=Path(source['model']);path=path if path.is_absolute()else ROOT/path
sha=hashlib.sha256(path.read_bytes()).hexdigest()
if sha!=source['model_sha256']:raise ValueError('Candidate model receipt mismatch')
m=mj.MjModel.from_xml_path(str(path));d=mj.MjData(m);d.qpos[:]=np.asarray(candidate['qpos']);mj.mj_forward(m,d)
npz_path=a.candidate.with_suffix('.npz');npz_check=None
if npz_path.exists():
    saved=dict(np.load(npz_path,allow_pickle=False));matches=[key for key,value in saved.items()if value.shape==d.qpos.shape]
    npz_check={'path':str(npz_path),'keys':list(saved),'matching_qpos_fields':[key for key in matches if np.max(abs(saved[key]-d.qpos))<1e-12]}
    if matches and not npz_check['matching_qpos_fields']:raise ValueError('NPZ candidate disagrees with JSON')
source_dir=ROOT/'data/linkerhand_o6_06_20261027/06'
coupling=[];limits=[]
for folder,name in [('LEFT','linkerhand_o6_left_test.urdf'),('RIGHT','linkerhand_o6_right.urdf')]:
    for joint in ET.parse(source_dir/folder/name).getroot().findall('joint'):
        if joint.get('type')=='fixed':continue
        j=mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,joint.get('name'));q=int(m.jnt_qposadr[j]);value=float(d.qpos[q]);lim=joint.find('limit')
        limits.append({'joint':joint.get('name'),'angle_rad':value,'inside_source_urdf_range':float(lim.get('lower'))-1e-10<=value<=float(lim.get('upper'))+1e-10})
        mimic=joint.find('mimic')
        if mimic is not None:
            parent=mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,mimic.get('joint'));pq=int(m.jnt_qposadr[parent])
            expected=float(mimic.get('multiplier','1'))*float(d.qpos[pq])+float(mimic.get('offset','0'))
            coupling.append({'joint':joint.get('name'),'source_parent':mimic.get('joint'),'absolute_residual_rad':abs(value-expected)})
contacts=[];robot_scene=[];hand_object=[]
for c in d.contact[:d.ncon]:
    g1,g2=int(c.geom1),int(c.geom2);b1,b2=int(m.geom_bodyid[g1]),int(m.geom_bodyid[g2]);names=[m.body(b1).name or'world',m.body(b2).name or'world']
    item={'geoms':[m.geom(g1).name,m.geom(g2).name],'bodies':names,'signed_distance_m':float(c.dist)};contacts.append(item)
    isobject=[n.startswith('target_object_')for n in names];ishand=[n.startswith(('rh_','lh_'))for n in names]
    if any(isobject)and any(ishand):hand_object.append(item)
    if any(n=='world'or n.startswith('target_object_')for n in names)and any(n!='world'and not n.startswith('target_object_')for n in names):robot_scene.append(item)
oracle=ConvexSeparationOracle(m);rh=[g for g in range(m.ngeom)if m.geom_contype[g]>0 and m.body(int(m.geom_bodyid[g])).name.startswith('rh_')]
obj=[[g for g in range(m.ngeom)if m.geom_contype[g]>0 and m.body(int(m.geom_bodyid[g])).name==f'target_object_{oi:02d}']for oi in range(1,7)]
all_hand_object=[]
for oi,geoms in enumerate(obj):
    best=(np.inf,None,None)
    for hand in rh:
        for target in geoms:
            gap,_=oracle.query(d,hand,target)
            if gap<best[0]:best=(gap,m.geom(hand).name,m.geom(target).name)
    all_hand_object.append({'object_id':oi+1,'minimum_all_rh_mesh_sat_gap_m':best[0],'limiting_geoms':best[1:]})
numeric=mj.mj_name2id(m,mj.mjtObj.mjOBJ_NUMERIC,'source_table_plane_abcd');adr=int(m.numeric_adr[numeric]);plane=m.numeric_data[adr:adr+4]
plane/=np.linalg.norm(plane[:3]);table_values=[]
for g in range(m.ngeom):
    body=m.body(int(m.geom_bodyid[g])).name or''
    if m.geom_contype[g]<=0 or not body.startswith(('rh_','right_wrist')):continue
    if m.geom_type[g]==mj.mjtGeom.mjGEOM_MESH:
        mesh=int(m.geom_dataid[g]);start=int(m.mesh_vertadr[mesh]);number=int(m.mesh_vertnum[mesh]);verts=m.mesh_vert[start:start+number]
        points=verts@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
        minimum=float(np.min(points@plane[:3]+plane[3]))
    else:
        rotate=d.geom_xmat[g].reshape(3,3);centre=d.geom_xpos[g]+rotate@m.geom_aabb[g,:3];half=np.abs(rotate)@m.geom_aabb[g,3:]
        minimum=float(plane[:3]@centre+plane[3]-abs(plane[:3])@half)
    table_values.append({'geom':m.geom(g).name,'source_plane_lower_clearance_m':minimum})
witnesses={}
for digit in ['thumb','index']:
    hand=mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,f'collision_rh_{digit}_distal_mesh');best=(np.inf,None,None)
    for target in obj[0]:
        pair=np.zeros(6);gap=float(mj.mj_geomDistance(m,d,hand,target,1.,pair))
        if gap<=0 or abs(np.linalg.norm(pair[:3]-pair[3:])-gap)>1e-4:continue
        if gap<best[0]:best=(gap,pair,target)
    if best[1]is None:raise ValueError('No reliable positive-gap witness '+digit)
    normal=(best[1][:3]-best[1][3:])/best[0];normal/=np.linalg.norm(normal)
    witnesses[digit]={'distance_m':best[0],'normal_object_to_hand_world':normal.tolist(),'geom':m.geom(best[2]).name,
                     'hand_point_world_m':best[1][:3].tolist(),'object_point_world_m':best[1][3:].tolist()}
dot=float(np.asarray(witnesses['thumb']['normal_object_to_hand_world'])@witnesses['index']['normal_object_to_hand_world'])
baseyaw=float(Rotation.from_quat(d.qpos[[4,5,6,3]]).as_euler('xyz')[2])
body_check=bool(np.max(abs(d.qpos[:2]-[-.25,2.]))<1e-12 and abs(baseyaw-(-np.pi+.75))<1e-7)
template=candidate['template'];site=mj.mj_name2id(m,mj.mjtObj.mjOBJ_SITE,'rh_palm_control_site');ob=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'target_object_01');hb=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'rh_hand_base_link')
objR=d.xmat[ob].reshape(3,3);actual_offset=objR.T@(d.site_xpos[site]-d.xpos[ob]);actual_R=objR.T@d.xmat[hb].reshape(3,3)
Rtarget=Rotation.from_quat(np.asarray(template['hand_rotation_in_object_wxyz'])[[1,2,3,0]]).as_matrix()
offset_error=float(np.linalg.norm(actual_offset-template['palm_offset_in_object_m']));rotation_error=float(np.linalg.norm(actual_R-Rtarget))
bad_robot_scene=[item for item in robot_scene if item['signed_distance_m']<-.020]
all_object_no_penetration=bool(all(x['minimum_all_rh_mesh_sat_gap_m']>=-1e-6 for x in all_hand_object))
passed=bool(body_check and all(item['inside_source_urdf_range']for item in limits)and max(x['absolute_residual_rad']for x in coupling)<1e-10 and offset_error<1e-10 and rotation_error<1e-10 and all_object_no_penetration and min(x['source_plane_lower_clearance_m']for x in table_values)>=.002 and dot<-.2 and not bad_robot_scene)
report={'schema':'fourth.o6_pregrasp_independent_static_audit.v1','model':str(path),'model_sha256':sha,'candidate':str(a.candidate),'source_attempt':candidate['source_attempt'],
        'npz_consistency':npz_check,'base_xyz_yaw':[float(x)for x in d.qpos[:3]]+[baseyaw],'base_contract_matched':body_check,
        'native_limits':limits,'source_urdf_mimic_residuals':coupling,'template_palm_position_error_m':offset_error,'template_hand_rotation_matrix_error':rotation_error,
        'real_mujoco_contact_count':int(d.ncon),'real_contacts':contacts,'robot_scene_contacts':robot_scene,'hand_object_contacts':hand_object,
        'all_rh_mesh_all_six_objects_sat_clearances':all_hand_object,'hand_wrist_source_plane_clearances':table_values,
        'positive_gap_witnesses':witnesses,'normal_dot':dot,'oracle':oracle.report(),'static_audit_passed':passed,
        'physical_rollout_performed':False,'physical_grasp_or_lift_success_claimed':False,
        'recommendation':'Proceed to controlled real actuator-only acquire/lift test; static pregrasp is not a contact or lift claim.'if passed else'Inspect failing static contract before physical enable.'}
a.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:report[k]for k in ['static_audit_passed','base_xyz_yaw','real_mujoco_contact_count','robot_scene_contacts','all_rh_mesh_all_six_objects_sat_clearances','normal_dot','template_palm_position_error_m','template_hand_rotation_matrix_error','recommendation']}),flush=True)
if not passed:raise SystemExit(2)
