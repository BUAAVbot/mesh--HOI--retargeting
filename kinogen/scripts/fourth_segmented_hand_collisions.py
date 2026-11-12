"""Replace only hand collision proxies with the user's articulated segment meshes.

Visual geoms and all body/joint/actuator properties remain unchanged. Each mesh
is attached to its matching URDF link, so the eight non-thumb DIP joints move
real collision geometry instead of visual meshes alone.
"""
from pathlib import Path
import argparse
import hashlib
import json
import struct
import xml.etree.ElementTree as ET

import numpy as np
import mujoco as mj
from scipy.spatial.transform import Rotation

R=Path(__file__).resolve().parents[1]
URDF=R/'data/vita_demo_20261026/robot/vita02a/urdf/vita02a_linkerhand_meshcoll_relative.urdf'
A=R/'outputs/fourth_full_sequence/assets'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def fmt(values):return ' '.join(format(float(x),'.17g')for x in np.asarray(values).reshape(-1))


def stl_vertices(path):
    raw=Path(path).read_bytes()
    count=struct.unpack('<I',raw[80:84])[0]if len(raw)>=84 else-1
    if count>=0 and len(raw)==84+count*50:
        dtype=np.dtype([('n','<f4',(3,)),('v','<f4',(3,3)),('a','<u2')])
        return np.frombuffer(raw,offset=84,count=count,dtype=dtype)['v'].reshape(-1,3).astype(float)
    rows=[]
    for line in raw.decode('utf-8',errors='strict').splitlines():
        if line.strip().startswith('vertex '):rows.append([float(x)for x in line.split()[1:4]])
    if not rows:raise ValueError('STL has no vertices '+str(path))
    return np.asarray(rows)


def point_aabb_world(model,data,g):
    return data.geom_xpos[g]+data.geom_xmat[g].reshape(3,3)@model.geom_aabb[g,:3]


def nearest(model,data,g,objects):
    best=10.
    for gg in objects:
        w=np.zeros(6);distance=float(mj.mj_geomDistance(model,data,g,gg,10.,w))
        if not np.isfinite(distance)or abs(np.linalg.norm(w[:3]-w[3:])-abs(distance))>1e-4:continue
        okay=True
        for gid,pt in [(g,w[:3]),(gg,w[3:])]:
            rr=data.geom_xmat[gid].reshape(3,3);centre=point_aabb_world(model,data,gid);extent=np.abs(rr)@model.geom_aabb[gid,3:]
            okay&=bool(np.all(abs(pt-centre)<=extent+1e-4))
        if okay:best=min(best,distance)
    return best


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-model',type=Path,default=A/'vita_six_objects_newft_source_table_v1.xml')
    p.add_argument('--output',type=Path,default=A/'vita_six_objects_newft_source_table_segmented_hand_v1.xml')
    p.add_argument('--reference',type=Path,default=R/'outputs/fourth_full_sequence/reference_v6.npz')
    a=p.parse_args()
    if a.output.exists():raise ValueError('Never overwrite an existing collision variant')
    xml=ET.parse(a.base_model).getroot();source=ET.parse(URDF).getroot()
    old=mj.MjModel.from_xml_path(str(a.base_model))
    oldd=mj.MjData(old);mj.mj_kinematics(old,oldd)
    assets=xml.find('asset');rows=[];removed=[];palm_sites=[]
    for side,prefix in enumerate(['lh_','rh_']):
        palm=mj.mj_name2id(old,mj.mjtObj.mjOBJ_GEOM,'collision_'+prefix+'palm_mesh')
        handbase=mj.mj_name2id(old,mj.mjtObj.mjOBJ_BODY,prefix+'hand_base_link')
        local=oldd.xmat[handbase].reshape(3,3).T@(point_aabb_world(old,oldd,palm)-oldd.xpos[handbase])
        parent=xml.find(f".//body[@name='{prefix}hand_base_link']")
        ET.SubElement(parent,'site',name=prefix+'palm_control_site',pos=fmt(local),size='.001',rgba='0 0 0 0')
        palm_sites.append({'hand':prefix,'site':prefix+'palm_control_site','old_control_palm_aabb_site_local_m':local.tolist()})
    for body in xml.findall('.//body'):
        name=body.get('name','')
        if not name.startswith(('lh_','rh_')):continue
        prefix=name[:3]
        for geom in list(body.findall('geom')):
            if int(geom.get('contype','1'))or int(geom.get('conaffinity','1')):
                removed.append({'body':name,'name':geom.get('name'),'type':geom.get('type','sphere')});body.remove(geom)
        link=source.find(f"link[@name='{name}']")
        if link is None:raise ValueError('Hand link absent in full URDF '+name)
        palm_id=mj.mj_name2id(old,mj.mjtObj.mjOBJ_GEOM,'collision_'+prefix+'palm_mesh')
        for index,collision in enumerate(link.findall('collision')):
            mesh=collision.find('geometry/mesh')
            if mesh is None:raise ValueError('Expected original segmented collision mesh')
            path=(URDF.parent/mesh.get('filename')).resolve()
            if not path.is_file():raise FileNotFoundError(path)
            meshname='segmented_'+name+'_'+str(index)
            asset=ET.SubElement(assets,'mesh',name=meshname,file=str(path))
            if mesh.get('scale'):asset.set('scale',mesh.get('scale'))
            origin=collision.find('origin');xyz=np.zeros(3);rpy=np.zeros(3)
            if origin is not None:
                xyz=np.asarray([float(x)for x in origin.get('xyz','0 0 0').split()])
                rpy=np.asarray([float(x)for x in origin.get('rpy','0 0 0').split()])
            xyzw=Rotation.from_euler('xyz',rpy).as_quat()
            name_geom=collision.get('name')or f'collision_{name}_mesh_{index}'
            ET.SubElement(body,'geom',name=name_geom,type='mesh',mesh=meshname,pos=fmt(xyz),quat=fmt(xyzw[[3,0,1,2]]),
                          contype='1',conaffinity='2',density='0',group='0',rgba='.2 .8 .5 .5',
                          friction=fmt(old.geom_friction[palm_id]),solref=fmt(old.geom_solref[palm_id]),solimp=fmt(old.geom_solimp[palm_id]),
                          condim=str(int(old.geom_condim[palm_id])),margin=format(float(old.geom_margin[palm_id]),'.17g'),
                          gap=format(float(old.geom_gap[palm_id]),'.17g'),priority=str(int(old.geom_priority[palm_id])))
            verts=stl_vertices(path)
            visual=link.find('visual/geometry/mesh');visual_path=(URDF.parent/visual.get('filename')).resolve()if visual is not None else None
            vv=stl_vertices(visual_path)if visual_path is not None else np.empty((0,3))
            rows.append({'body':name,'collision_name':name_geom,'mesh':str(path),'sha256':sha(path),
                'urdf_origin_xyz_m':xyz.tolist(),'urdf_origin_rpy_rad':rpy.tolist(),
                'collision_mesh_local_aabb_m':[verts.min(0).tolist(),verts.max(0).tolist()],
                'visual_mesh':str(visual_path),'visual_mesh_local_aabb_m':[vv.min(0).tolist(),vv.max(0).tolist()]if len(vv)else None,
                'collision_to_visual_aabb_difference_max_m':float(np.max(abs(np.r_[verts.min(0)-vv.min(0),verts.max(0)-vv.max(0)])))if len(vv)else None,
                'convex_hull_scope':'One native URDF link only; no mesh is attached across a movable DIP joint.'})
    a.output.write_text(ET.tostring(xml,encoding='unicode'),encoding='utf-8')
    new=mj.MjModel.from_xml_path(str(a.output))
    if(new.nq,new.nv,new.nu,new.nbody)!=(old.nq,old.nv,old.nu,old.nbody):raise ValueError('Articulation changed')
    unchanged=['body_mass','body_inertia','body_ipos','body_iquat','body_pos','body_quat','jnt_type','jnt_pos','jnt_axis','jnt_range',
               'dof_armature','dof_damping','dof_frictionloss','actuator_gainprm','actuator_biasprm','actuator_forcerange']
    if any(not np.array_equal(getattr(old,k),getattr(new,k))for k in unchanged):raise ValueError('Noncollision dynamic properties changed')
    # Geom indices shift with extra pieces, so compare non-hand geometry by name.
    for g in range(old.ngeom):
        b=int(old.geom_bodyid[g]);bn=mj.mj_id2name(old,mj.mjtObj.mjOBJ_BODY,b)or'world'
        if bn.startswith(('lh_','rh_')):continue
        name=mj.mj_id2name(old,mj.mjtObj.mjOBJ_GEOM,g)
        if not name:continue
        gg=mj.mj_name2id(new,mj.mjtObj.mjOBJ_GEOM,name)
        if gg<0:raise ValueError('Lost unrelated geometry '+name)
        for key in ['geom_type','geom_pos','geom_quat','geom_size','geom_friction','geom_contype','geom_conaffinity']:
            if not np.array_equal(getattr(old,key)[g],getattr(new,key)[gg]):raise ValueError(('Non-hand geom changed',name,key))
    for entry in rows:
        g=mj.mj_name2id(new,mj.mjtObj.mjOBJ_GEOM,entry['collision_name'])
        b=mj.mj_name2id(new,mj.mjtObj.mjOBJ_BODY,entry['body'])
        if new.geom_bodyid[g]!=b:raise ValueError('Segment attached to wrong body')
        entry['attached_body_id']=int(b);entry['active_contype']=int(new.geom_contype[g]);entry['active_conaffinity']=int(new.geom_conaffinity[g])
    ref=dict(np.load(a.reference,allow_pickle=False));d=mj.MjData(new)
    names=list(map(str,ref['dof_names']));qa=[int(new.jnt_qposadr[mj.mj_name2id(new,mj.mjtObj.mjOBJ_JOINT,n)])for n in names]
    d.qpos[:7]=ref['qpos'][88,:7];d.qpos[qa]=ref['qpos'][88,7:]
    for oi,oid in enumerate(ref['object_ids']):
        q=int(new.jnt_qposadr[mj.mj_name2id(new,mj.mjtObj.mjOBJ_JOINT,f'target_object_{oid:02d}_free')]);d.qpos[q:q+7]=ref['object_target_pose'][88,oi]
    dips=[]
    for prefix in ['lh_','rh_']:
        for digit in ['index','middle','ring','pinky']:
            name=prefix+digit+'_dip';j=mj.mj_name2id(new,mj.mjtObj.mjOBJ_JOINT,name);q=int(new.jnt_qposadr[j])
            geom=mj.mj_name2id(new,mj.mjtObj.mjOBJ_GEOM,'collision_'+prefix+digit+'_distal_mesh')
            d.qpos[q]=float(new.jnt_range[j,0]);mj.mj_kinematics(new,d)
            before_pos=d.geom_xpos[geom].copy();before_rot=d.geom_xmat[geom].copy()
            object_groups=[[g for g in range(new.ngeom)if(mj.mj_id2name(new,mj.mjtObj.mjOBJ_BODY,int(new.geom_bodyid[g]))or'')==f'target_object_{i:02d}'and new.geom_contype[g]>0]for i in range(1,7)]
            before=[nearest(new,d,geom,group)for group in object_groups]
            d.qpos[q]=min(float(new.jnt_range[j,1]),float(new.jnt_range[j,0])+.8);mj.mj_kinematics(new,d)
            after=[nearest(new,d,geom,group)for group in object_groups]
            pos_delta=float(np.linalg.norm(d.geom_xpos[geom]-before_pos));rot_delta=float(np.max(abs(d.geom_xmat[geom]-before_rot)))
            item={'joint':name,'collision':new.geom(geom).name,'perturbation_rad':float(d.qpos[q]-new.jnt_range[j,0]),
                  'collision_position_change_m':pos_delta,'collision_rotation_max_abs_change':rot_delta,
                  'object_signed_distances_before_m':before,'object_signed_distances_after_m':after,
                  'max_object_distance_change_m':float(np.max(abs(np.asarray(after)-before))),'passed':bool(pos_delta>1e-4 and rot_delta>.1)}
            dips.append(item);d.qpos[q]=new.jnt_range[j,0]
    initial=mj.MjData(new);mj.mj_kinematics(new,initial)
    for item in palm_sites:
        prefix=item['hand'];g=mj.mj_name2id(new,mj.mjtObj.mjOBJ_GEOM,'collision_'+prefix+'palm_mesh');b=mj.mj_name2id(new,mj.mjtObj.mjOBJ_BODY,prefix+'hand_base_link')
        point=initial.xmat[b].reshape(3,3).T@(point_aabb_world(new,initial,g)-initial.xpos[b])
        item['new_collision_aabb_site_local_m']=point.tolist();item['aabb_center_change_m']=float(np.linalg.norm(point-item['old_control_palm_aabb_site_local_m']))
        item['control_site_definition_unchanged']=True
    report={'schema':'fourth.segmented_hand_collision_variant.v1','model':str(a.output),'model_sha256':sha(a.output),
        'base':str(a.base_model),'base_sha256':sha(a.base_model),'full_articulated_urdf':str(URDF),'full_urdf_sha256':sha(URDF),
        'removed_old_hand_colliders':removed,'new_hand_colliders':rows,'old_count_per_hand':len(removed)//2,'new_count_per_hand':len(rows)//2,
        'non_collision_physics_arrays_exactly_preserved':unchanged,'nonhand_named_geometries_preserved':True,
        'palm_control_sites':palm_sites,'dip_perturbation_audit':dips,'all_dip_collision_influence_checks_passed':all(x['passed']for x in dips),
        'masks':{'robot_hand_contype':1,'robot_hand_conaffinity':2,'robot_self_filtered_as_training':True,
                 'hand_object_and_hand_table_contacts_enabled':True},
        'limitations':['MuJoCo convexifies each individual link collision mesh; concavity within a single link remains an approximation.',
                       'Segment meshes intentionally overlap locally around joints; this does not attach one hull across two movable links.',
                       'This asset/kinematic audit does not establish dynamic grasp or full-sequence success.']}
    a.output.with_suffix('.report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'model':str(a.output),'sha256':sha(a.output),'old_per_hand':len(removed)//2,'new_per_hand':len(rows)//2,
                      'all_dip_checks_passed':report['all_dip_collision_influence_checks_passed'],'palm_sites':palm_sites,'dip_checks':dips},ensure_ascii=False),flush=True)
    if not report['all_dip_collision_influence_checks_passed']:raise SystemExit(3)


if __name__=='__main__':main()
