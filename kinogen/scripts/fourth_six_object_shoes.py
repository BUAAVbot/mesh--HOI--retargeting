"""Clone the recovered SONIC shoe plant, restore auxiliary joints, add six objects.

Original source files and earlier frozen trials are never overwritten. Body/link
inertias and foot meshes come from the recovered SONIC URDF; 24 auxiliary neck
and finger joint definitions are grafted from the supplied articulated URDF.
"""
from pathlib import Path
import copy
import hashlib
import json
import xml.etree.ElementTree as ET

import mujoco as mj

R=Path(__file__).resolve().parents[1]
SOURCE=R/'dependencies/fourth_sonic/source/legged_lab/assets/vita02a/urdf/vita02a_shoes_sonic_hw.urdf'
ARTICULATED=R/'data/vita_demo_20261026/robot/vita02a/urdf/vita02a_linkerhand_meshcoll_relative.urdf'
SIX=R/'outputs/fourth_full_sequence/assets/vita_six_objects_dynamic.xml'
OUT=SIX.parent/'vita_six_objects_sonic_shoes52.xml'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    old=ET.parse(ARTICULATED).getroot();src=ET.parse(SOURCE).getroot()
    movable={j.get('name'):j for j in old.findall('joint')if j.get('type')!='fixed'}
    graft=[]
    for joint in list(src.findall('joint')):
        name=joint.get('name')
        if joint.get('type')=='fixed' and name in movable:
            index=list(src).index(joint);src.remove(joint);src.insert(index,copy.deepcopy(movable[name]));graft.append(name)
    if len(graft)!=24:raise ValueError(('Expected 24 restored neck/finger joints',graft))
    for mesh in src.findall('.//mesh'):
        path=(SOURCE.parent/mesh.get('filename')).resolve()
        if not path.is_file():raise FileNotFoundError(path)
        mesh.set('filename',str(path))
    ext=src.find('mujoco')
    if ext is None:ext=ET.SubElement(src,'mujoco')
    comp=ext.find('compiler')
    if comp is None:comp=ET.SubElement(ext,'compiler')
    comp.attrib.update(discardvisual='false',fusestatic='false',strippath='false')
    spec=mj.MjSpec.from_string(ET.tostring(src,encoding='unicode'))
    spec.body('base_link').add_freejoint(name='vita_demo_free_root')
    spec.option.timestep=.005
    spec.option.integrator=mj.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.gravity=[0,0,-9.81]
    spec.compile()
    robot=ET.fromstring(spec.to_xml());six=ET.parse(SIX).getroot()
    # Keep each shoe mesh and replace only the 14 tagged cylindrical strip
    # placeholders with native capsule geoms before compiling new bounds.
    capsules=[]
    for geom in robot.findall('.//geom'):
        name=geom.get('name','')
        if name.startswith(('collision_foot_l_capsule_','collision_foot_r_capsule_')):
            if geom.get('type')!='cylinder':raise ValueError('Unexpected shoe strip type')
            geom.set('type','capsule');capsules.append(name)
    if len(capsules)!=14:raise ValueError(('Missing shoe strips',capsules))
    assets=robot.find('asset');world=robot.find('worldbody')
    for item in six.find('asset'):
        if item.get('name','').startswith('target_object_'):assets.append(copy.deepcopy(item))
    for item in six.find('worldbody'):
        if item.tag=='geom' and item.get('name')in ['table_support','assumed_floor']:
            world.append(copy.deepcopy(item))
        elif item.tag=='body' and item.get('name','').startswith('target_object_'):
            world.append(copy.deepcopy(item))
    actuator=six.find('actuator')
    if actuator is None:raise ValueError('Six-object actuator bank missing')
    robot.append(copy.deepcopy(actuator))
    OUT.write_text(ET.tostring(robot,encoding='unicode'),encoding='utf-8')
    m=mj.MjModel.from_xml_path(str(OUT))
    if(m.nq,m.nv,m.nu)!=(101,94,52):raise ValueError((m.nq,m.nv,m.nu))
    feet=[]
    for side in ['left','right']:
        body=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,side+'_ankle_roll_link')
        feet.append({'body':side+'_ankle_roll_link','mass_kg':float(m.body_mass[body]),
                     'inertial_position_local_m':m.body_ipos[body].tolist(),
                     'principal_inertia_kg_m2':m.body_inertia[body].tolist(),
                     'capsule_count':int(sum(int(m.geom_bodyid[g])==body and m.geom_type[g]==mj.mjtGeom.mjGEOM_CAPSULE for g in range(m.ngeom))),
                     'mesh_collision_preserved':any(int(m.geom_bodyid[g])==body and (mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,g)or'').endswith('_mesh') for g in range(m.ngeom))})
    objects=[]
    for oid in range(1,7):
        name=f'target_object_{oid:02d}';j=mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,name+'_free')
        objects.append({'id':oid,'body':name,'freejoint':name+'_free','qpos':int(m.jnt_qposadr[j]),'qvel':int(m.jnt_dofadr[j])})
    report={'schema':'fourth.sonic_shoes52_assets.v1','model':str(OUT),'model_sha256':sha(OUT),
            'source_urdf':str(SOURCE),'source_urdf_sha256':sha(SOURCE),
            'auxiliary_joint_source':str(ARTICULATED),'auxiliary_joint_source_sha256':sha(ARTICULATED),
            'six_object_scene_source_sha256':sha(SIX),'nq':m.nq,'nv':m.nv,'nu':m.nu,
            'restored_auxiliary_joints':graft,'shoe_capsules':capsules,'foot_inertia':feet,'objects':objects,
            'physics_dt_s':float(m.opt.timestep),'integrator':'implicitfast',
            'source_link_inertias_and_geometry_preserved':True,'fingers_and_neck_articulated':True,
            'gains':'Copied 52-channel demo bank; SONIC Controller must override matching 28 body gains and armature at reset.',
            'limitations':['The recovered source fixes fingers/neck; articulating 24 auxiliary joints is an explicit plant adaptation.',
                           'Native implicit position actuators and implicitfast are a numerical control adaptation to the source explicit backend.',
                           'Six-object mass/friction/collision proxies remain declared engineering assumptions.']}
    OUT.with_suffix('.report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k]for k in ['model','model_sha256','nq','nv','nu','foot_inertia','objects']}),flush=True)


if __name__=='__main__':main()
