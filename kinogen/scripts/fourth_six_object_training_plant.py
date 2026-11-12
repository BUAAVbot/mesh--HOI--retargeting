"""Extend the actual new_ft15225 training MJCF with articulated hands and scene.

The recovered 28-DOF training body is copied verbatim before adding two neck
and 22 finger joints from the provided Vita52 model. This is an explicitly
reported articulated-hand extension, not an exact unchanged 28-DOF plant.
"""
from pathlib import Path
import copy
import hashlib
import json
import xml.etree.ElementTree as ET

import mujoco as mj

R=Path(__file__).resolve().parents[1]
SOURCE=Path('/data/yunhao.xia/GR00T-WholeBodyControl/gear_sonic/data/assets/robot_description/mjcf/vita02a/vita02a_28dof.xml')
SIX=R/'outputs/fourth_full_sequence/assets/vita_six_objects_dynamic.xml'
OUT=SIX.parent/'vita_six_objects_newft_training52.xml'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    src=ET.parse(SOURCE).getroot();six=ET.parse(SIX).getroot()
    original_xml=copy.deepcopy(src)
    for mesh in src.findall('./asset/mesh'):
        path=(SOURCE.parent/mesh.get('file')).resolve()
        if not path.is_file():
            path=SOURCE.parent/(mesh.get('name')+'.STL')
        if not path.is_file():raise FileNotFoundError(path)
        mesh.set('file',str(path))
    root=src.find(".//freejoint[@name='root']")
    if root is None:raise ValueError('Training root missing')
    root.set('name','vita_demo_free_root')
    world=src.find('worldbody');assets=src.find('asset')
    for geom in list(world.findall('geom')):
        world.remove(geom)
    # Matching source keeps its own 28 body geometry and inertial definitions.
    # The supplied articulated model provides only the missing distal limbs.
    graft=[('waist_pitch_link','neck_yaw_link'),
           ('left_wrist_pitch_link','lh_hand_base_link'),
           ('right_wrist_pitch_link','rh_hand_base_link')]
    selected_meshes=set();added_joints=[];added_bodies=[]
    for parent,name in graft:
        body=six.find(f".//body[@name='{name}']")
        target=src.find(f".//body[@name='{parent}']")
        if body is None or target is None:raise ValueError(('Missing subtree',parent,name))
        subtree=copy.deepcopy(body)
        selected_meshes.update(g.get('mesh') for g in subtree.findall('.//geom')if g.get('mesh'))
        added_joints.extend(j.get('name')for j in subtree.findall('.//joint'))
        added_bodies.extend(b.get('name')for b in subtree.iter('body'))
        target.append(subtree)
    if len(added_joints)!=24:raise ValueError(('Expected 24 auxiliary joints',added_joints))
    for item in six.find('asset'):
        if item.get('name')in selected_meshes or item.get('name','').startswith('target_object_'):
            assets.append(copy.deepcopy(item))
    for item in six.find('worldbody'):
        if item.tag=='geom'and item.get('name')in ['table_support','assumed_floor']:
            world.append(copy.deepcopy(item))
        elif item.tag=='body'and item.get('name','').startswith('target_object_'):
            world.append(copy.deepcopy(item))
    oldact=src.find('actuator')
    if oldact is not None:src.remove(oldact)
    src.append(copy.deepcopy(six.find('actuator')))
    opt=src.find('option')
    if opt is None:opt=ET.SubElement(src,'option')
    opt.set('timestep','.005');opt.set('integrator','implicitfast');opt.set('gravity','0 0 -9.81')
    OUT.write_text(ET.tostring(src,encoding='unicode'),encoding='utf-8')
    m=mj.MjModel.from_xml_path(str(OUT))
    if(m.nq,m.nv,m.nu)!=(101,94,52):raise ValueError((m.nq,m.nv,m.nu))
    # Load isolated original MJCF via resolved copied mesh paths, untouched body.
    original=copy.deepcopy(original_xml)
    for mesh in original.findall('./asset/mesh'):
        path=(SOURCE.parent/mesh.get('file')).resolve()
        if not path.is_file():path=SOURCE.parent/(mesh.get('name')+'.STL')
        mesh.set('file',str(path))
    original_path=OUT.parent/'newft_original_training28_resolved.xml'
    original_path.write_text(ET.tostring(original,encoding='unicode'),encoding='utf-8')
    training=mj.MjModel.from_xml_path(str(original_path))
    preservation=[]
    for oldbody in range(1,training.nbody):
        name=mj.mj_id2name(training,mj.mjtObj.mjOBJ_BODY,oldbody)
        body=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,name)
        if body<0:raise ValueError(('Lost training body',name))
        for key in ['body_mass','body_inertia','body_ipos','body_iquat','body_pos','body_quat']:
            import numpy as np
            if not np.array_equal(getattr(training,key)[oldbody],getattr(m,key)[body]):
                raise ValueError(('Training body changed',name,key))
        preservation.append(name)
    added_mass=sum(float(m.body_mass[mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,name)])for name in added_bodies)
    colliders=[]
    for side in ['left','right']:
        b=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,side+'_ankle_roll_link')
        colliders.append({'body':side+'_ankle_roll_link','mass_kg':float(m.body_mass[b]),
            'collision_names':[mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,g) for g in range(m.ngeom)if m.geom_bodyid[g]==b and m.geom_contype[g]>0]})
    report={'schema':'fourth.newft_training52_plant.v1','model':str(OUT),'model_sha256':sha(OUT),
            'true_training_mjcf':str(SOURCE),'true_training_mjcf_sha256':sha(SOURCE),
            'resolved_original28':str(original_path),'resolved_original28_sha256':sha(original_path),
            'auxiliary_scene_source':str(SIX),'auxiliary_scene_source_sha256':sha(SIX),
            'nq':m.nq,'nv':m.nv,'nu':m.nu,'original28_dimensions':[training.nq,training.nv,training.nu],
            'original_training_body_properties_exactly_preserved':True,'verified_preserved_bodies':preservation,
            'added_auxiliary_joint_names':added_joints,'added_auxiliary_body_names':added_bodies,
            'added_auxiliary_mass_kg':added_mass,'training_robot_mass_kg':float(training.body_mass.sum()),
            'extended_robot_mass_kg':float(training.body_mass.sum()+added_mass),
            'foot_colliders':colliders,'physics_dt_s':float(m.opt.timestep),
            'root_name_adapted_from':'root','root_name':'vita_demo_free_root',
            'actuator_bank':'52 named motor_<joint> native positions; SONIC Controller configures the 28 body gains/actuation explicitly.',
            'limitations':['Added neck/finger masses change composite dynamics despite preserving original 28 individual body properties.',
                           'Native position servo bank and implicitfast are declared control/integrator adaptations.',
                           'Six object masses, friction and collision proxies are assumptions.']}
    OUT.with_suffix('.report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k]for k in ['model','model_sha256','nq','nv','nu','original28_dimensions','added_auxiliary_mass_kg','training_robot_mass_kg','foot_colliders']}),flush=True)


if __name__=='__main__':main()
