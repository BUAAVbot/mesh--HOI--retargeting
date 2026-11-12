"""Restore 24 auxiliary joints in the actual new-ft Isaac spawn URDF.

Matches the recovered VITA02A_CFG spawn: all cylinder collision primitives
become native capsules and robot self contacts are disabled in the matching
variant. Robot/object/world and object/object contacts remain enabled. A second
self-contact-enabled model is saved with identical inertia and geometry.
"""
from pathlib import Path
import copy
import hashlib
import json
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np

R=Path(__file__).resolve().parents[1]
SOURCE=Path('/data/yunhao.xia/GR00T-WholeBodyControl/gear_sonic/data/assets/robot_description/vita02a/urdf/vita02a_shoes.urdf')
CFG=Path('/data/yunhao.xia/GR00T-WholeBodyControl/gear_sonic/envs/manager_env/robots/vita02a.py')
AUX=R/'data/vita_demo_20261026/robot/vita02a/urdf/vita02a_linkerhand_meshcoll_relative.urdf'
SIX=R/'outputs/fourth_full_sequence/assets/vita_six_objects_dynamic.xml'
OUT=SIX.parent/'vita_six_objects_newft_isaac52.xml'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def to_spec(root):
    root=copy.deepcopy(root)
    for mesh in root.findall('.//mesh'):
        path=(SOURCE.parent/mesh.get('filename')).resolve()
        if not path.is_file():raise FileNotFoundError(path)
        mesh.set('filename',str(path))
    ext=root.find('mujoco')
    if ext is None:ext=ET.SubElement(root,'mujoco')
    compiler=ext.find('compiler')
    if compiler is None:compiler=ET.SubElement(ext,'compiler')
    compiler.attrib.update(discardvisual='false',fusestatic='false',strippath='false')
    spec=mj.MjSpec.from_string(ET.tostring(root,encoding='unicode'))
    spec.body('base_link').add_freejoint(name='vita_demo_free_root')
    spec.option.timestep=.005;spec.option.gravity=[0,0,-9.81]
    spec.option.integrator=mj.mjtIntegrator.mjINT_IMPLICITFAST
    spec.compile()
    return spec


def mask_allowed(m,a,b):
    return bool((int(m.geom_contype[a])&int(m.geom_conaffinity[b]))or(int(m.geom_contype[b])&int(m.geom_conaffinity[a])))


def main():
    src=ET.parse(SOURCE).getroot();aux=ET.parse(AUX).getroot();six=ET.parse(SIX).getroot()
    baseline=to_spec(src).compile()
    movable={j.get('name'):j for j in aux.findall('joint')if j.get('type')!='fixed'}
    restored=[];origin_changes=[]
    for j in list(src.findall('joint')):
        name=j.get('name')
        if j.get('type')=='fixed'and name in movable:
            old=ET.tostring(j.find('origin')) if j.find('origin')is not None else b''
            new=ET.tostring(movable[name].find('origin')) if movable[name].find('origin')is not None else b''
            if old!=new:origin_changes.append(name)
            index=list(src).index(j);src.remove(j);src.insert(index,copy.deepcopy(movable[name]));restored.append(name)
    if len(restored)!=24:raise ValueError(('Expected 24 restored auxiliary joints',restored))
    spec=to_spec(src);robot=ET.fromstring(spec.to_xml())
    # MjSpec's review XML rounds numerical values. Restore the source compile's
    # inertial arrays with round-trip precision before producing final MJCF.
    for b in range(1,baseline.nbody):
        name=mj.mj_id2name(baseline,mj.mjtObj.mjOBJ_BODY,b)
        element=robot.find(f".//body[@name='{name}']/inertial")
        if element is None:raise ValueError(('Missing inertial',name))
        fmt=lambda values:' '.join(format(float(x),'.17g')for x in values)
        element.set('mass',format(float(baseline.body_mass[b]),'.17g'))
        element.set('pos',fmt(baseline.body_ipos[b]));element.set('quat',fmt(baseline.body_iquat[b]))
        element.set('diaginertia',fmt(baseline.body_inertia[b]))
    converted=[]
    for g in robot.findall('.//geom'):
        active=int(g.get('contype','1'))!=0 or int(g.get('conaffinity','1'))!=0
        if active:
            if g.get('type')=='cylinder':g.set('type','capsule');converted.append(g.get('name'))
            g.set('contype','1');g.set('conaffinity','2')
    assets=robot.find('asset');world=robot.find('worldbody')
    for item in six.find('asset'):
        if item.get('name','').startswith('target_object_'):assets.append(copy.deepcopy(item))
    for item in six.find('worldbody'):
        if(item.tag=='geom'and item.get('name')in ['table_support','assumed_floor'])or(item.tag=='body'and item.get('name','').startswith('target_object_')):
            addition=copy.deepcopy(item)
            for g in ([addition]if addition.tag=='geom'else addition.findall('.//geom')):
                if int(g.get('contype','1'))!=0 or int(g.get('conaffinity','1'))!=0:
                    g.set('contype','2');g.set('conaffinity','3')
            world.append(addition)
    robot.append(copy.deepcopy(six.find('actuator')))
    OUT.write_text(ET.tostring(robot,encoding='unicode'),encoding='utf-8')
    m=mj.MjModel.from_xml_path(str(OUT))
    if(m.nq,m.nv,m.nu)!=(101,94,52):raise ValueError((m.nq,m.nv,m.nu))
    preserved=[];property_errors={key:0. for key in ['body_mass','body_inertia','body_ipos','body_iquat']}
    for b in range(1,baseline.nbody):
        name=mj.mj_id2name(baseline,mj.mjtObj.mjOBJ_BODY,b)
        bb=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,name)
        if bb<0:raise ValueError(('Lost original URDF link',name))
        for key in property_errors:
            error=float(np.max(np.abs(getattr(baseline,key)[b]-getattr(m,key)[bb])))
            property_errors[key]=max(property_errors[key],error)
        preserved.append(name)
    if max(property_errors.values())>1e-8:raise ValueError(('Inertia changed',property_errors))
    names=[mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,b)or'world'for b in range(m.nbody)]
    robot_bodies=[b for b,name in enumerate(names)if b>0 and not name.startswith('target_object_')]
    robot_geoms=[g for g in range(m.ngeom)if int(m.geom_bodyid[g])in robot_bodies and m.geom_contype[g]>0]
    object_geoms=[g for g in range(m.ngeom)if names[int(m.geom_bodyid[g])].startswith('target_object_')and m.geom_contype[g]>0]
    support_geoms=[mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,n)for n in ['assumed_floor','table_support']]
    mask_report={'robot_robot_allowed_pairs':sum(mask_allowed(m,a,b)for k,a in enumerate(robot_geoms)for b in robot_geoms[k+1:]),
                 'robot_object_all_allowed':all(mask_allowed(m,a,b)for a in robot_geoms for b in object_geoms),
                 'robot_scene_all_allowed':all(mask_allowed(m,a,b)for a in robot_geoms for b in support_geoms),
                 'object_object_all_allowed':all(mask_allowed(m,a,b)for k,a in enumerate(object_geoms)for b in object_geoms[k+1:]if m.geom_bodyid[a]!=m.geom_bodyid[b]),
                 'object_scene_all_allowed':all(mask_allowed(m,a,b)for a in object_geoms for b in support_geoms)}
    if mask_report['robot_robot_allowed_pairs']!=0 or not all(v for k,v in mask_report.items()if k!='robot_robot_allowed_pairs'):
        raise ValueError(('Contact bitmask mismatch',mask_report))
    variant=copy.deepcopy(robot)
    for g in variant.findall('.//geom'):
        if g.get('contype')=='1':g.set('conaffinity','3')
    variant_path=OUT.with_name(OUT.stem+'_selfcontacts.xml')
    variant_path.write_text(ET.tostring(variant,encoding='unicode'),encoding='utf-8')
    selfmodel=mj.MjModel.from_xml_path(str(variant_path))
    robot_mass=float(m.body_mass[robot_bodies].sum())
    report={'schema':'fourth.newft_actual_isaac_spawn52.v1','model':str(OUT),'model_sha256':sha(OUT),
            'self_contacts_variant':str(variant_path),'self_contacts_variant_sha256':sha(variant_path),
            'actual_spawn_urdf':str(SOURCE),'actual_spawn_urdf_sha256':sha(SOURCE),
            'training_cfg':str(CFG),'training_cfg_sha256':sha(CFG),
            'auxiliary_joint_source_sha256':sha(AUX),'six_object_scene_source_sha256':sha(SIX),
            'nq':m.nq,'nv':m.nv,'nu':m.nu,'original_fixed_hands_dimensions':[baseline.nq,baseline.nv,baseline.nu],
            'original_55_link_inertias_preserved':True,'link_property_max_abs_errors':property_errors,
            'preserved_links':preserved,'original_robot_mass_kg':float(baseline.body_mass.sum()),
            'extended_robot_mass_kg':robot_mass,'added_hand_mass_kg':robot_mass-float(baseline.body_mass.sum()),
            'restored_auxiliary_joints':restored,'joint_origin_xml_text_differences':origin_changes,
            'cylinders_converted_to_native_capsules':converted,
            'training_spawn_replace_cylinders_with_capsules':True,
            'training_spawn_enabled_self_collisions':False,'contact_mask_validation':mask_report,
            'physics_dt_s':float(m.opt.timestep),'integrator':'implicitfast',
            'root_joint':'vita_demo_free_root','object_qpos_addresses':[59,66,73,80,87,94],
            'limitations':['Restoring auxiliary joint mobility changes the original fixed-hands articulation, while preserving each link mass/inertia.',
                           'MuJoCo implicitfast/native position servos are explicit backend adaptations from Isaac; Controller must configure body actuation.',
                           'Matching variant omits intra-robot contacts as in the source Isaac cfg; separate selfcontact variant is provided.',
                           'Six object masses/friction/collision proxies remain assumptions.']}
    OUT.with_suffix('.report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k]for k in ['model','model_sha256','nq','nv','nu','original_robot_mass_kg','extended_robot_mass_kg','added_hand_mass_kg','link_property_max_abs_errors','contact_mask_validation']}),flush=True)


if __name__=='__main__':main()
