"""Build Vita52 + dynamic bowl + finite tabletop for fourth-scene physics trials.

The source meshes and robot inertias are preserved. The bowl's hollow collision
proxy, body inertial parameters, friction and support patch are declared
engineering assumptions, not measured quantities or ground-truth contacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/fourth_sim_handoff'
URDF = ROOT / 'data/vita_demo_20261026/robot/vita02a/urdf/vita02a_linkerhand_meshcoll_relative.urdf'
CONTRACT = ROOT / 'reports/vita17-old-policy-contract.json'
OUTPUT = ROOT / 'outputs/fourth_vita_demo/assets'


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def num(values) -> str:
    return ' '.join(f'{float(x):.12g}' for x in values)


def quat_from_z(normal):
    normal = np.asarray(normal, dtype=float)
    normal /= np.linalg.norm(normal)
    cross = np.cross([0., 0., 1.], normal)
    quaternion = np.r_[1.+normal[2], cross]
    return quaternion/np.linalg.norm(quaternion)


def load_source(object_id: int, start: int, stop: int):
    poses = np.load(DATA/'objects/poses_sim.npz', allow_pickle=False)
    index = object_id-1
    if not 0 <= index < 6 or not 0 <= start < stop <= 923:
        raise ValueError('Invalid object ID or source frame interval')
    valid = poses['valid'][start:stop, index]
    if not np.all(valid):
        raise ValueError('Requested source interval includes invalid object poses')
    info = json.loads((DATA/f'objects/object_{object_id:02d}/model_info.json').read_text(encoding='utf-8'))
    qa = json.loads((DATA/'provenance/object_pose_qa.json').read_text(encoding='utf-8'))
    positions = np.asarray(poses['position_m'][start:stop, index], dtype=float)
    quats = np.asarray(poses['quaternion_wxyz'][start:stop, index], dtype=float)
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(quats)):
        raise ValueError('Nonfinite source poses')
    return info, qa, positions, quats, np.asarray(poses['status'][start:stop, index], dtype=int)


def make_urdf_spec():
    root = ET.parse(URDF).getroot()
    for mesh in root.findall('.//mesh'):
        name = mesh.get('filename')
        if name.startswith('package://'):
            raise ValueError('Use the audited relative URDF')
        path = (URDF.parent/name).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        mesh.set('filename', str(path))
    ext = root.find('mujoco')
    if ext is None: ext = ET.SubElement(root, 'mujoco')
    comp = ext.find('compiler')
    if comp is None: comp = ET.SubElement(ext, 'compiler')
    comp.attrib.update(discardvisual='false', fusestatic='false', strippath='false')
    for key in ('boundmass', 'boundinertia', 'balanceinertia'):
        comp.attrib.pop(key, None)
    spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding='unicode'))
    spec.body('base_link').add_freejoint(name='vita_demo_free_root')
    spec.option.timestep = .005
    spec.option.gravity = [0., 0., -9.81]
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.compile()  # Assert the original mass/inertia geometry is valid.
    return root, ET.fromstring(spec.to_xml())


def build(args):
    info, qa, positions, quats, statuses = load_source(args.object_id, args.start, args.stop)
    urdf_root, mjxml = make_urdf_spec()
    world = mjxml.find('worldbody')
    asset = mjxml.find('asset')
    if asset is None: asset = ET.SubElement(mjxml, 'asset')

    # ARKit +Y-up -> sim +Z-up; the plane is fitted from the source capture.
    source_plane = qa['table_plane']
    arkit_origin = np.asarray(source_plane['origin_world'], dtype=float)
    arkit_normal = np.asarray(source_plane['normal_world'], dtype=float)
    sim_origin = np.asarray([arkit_origin[0], -arkit_origin[2], arkit_origin[1]])
    sim_normal = np.asarray([arkit_normal[0], -arkit_normal[2], arkit_normal[1]])
    sim_normal /= np.linalg.norm(sim_normal)
    centre_xy = .5*(np.min(positions[:, :2], axis=0)+np.max(positions[:, :2], axis=0))
    span_xy = np.ptp(positions[:, :2], axis=0)
    if args.variant == 'v2':
        # Bound support to the object swept XY footprint. The broad v1 patch
        # intersected the retargeted robot's left hip at the edge; this changes
        # real table geometry, not collision masks or a state trajectory.
        footprint_radius = float(max(info['dimensions_xyz_m'][:2]))/2
        table_half_xy = span_xy/2 + footprint_radius + .005
    else:
        table_half_xy = np.maximum(span_xy/2+.12, [.22, .22])
    table_top = (np.dot(sim_normal, sim_origin)-np.dot(sim_normal[:2], centre_xy))/sim_normal[2]
    table_half_z = .025
    table_centre = np.r_[centre_xy, table_top]-sim_normal*table_half_z
    support_quat = quat_from_z(sim_normal)
    ET.SubElement(world, 'geom', name='table_support', type='box', pos=num(table_centre),
                  quat=num(support_quat), size=num([*table_half_xy, table_half_z]),
                  rgba='.39 .29 .20 .8', friction='.75 .01 .0005',
                  contype='1', conaffinity='1', group='2')
    floor_z = float(args.floor_z) if args.floor_z is not None else table_top-.75
    ET.SubElement(world, 'geom', name='assumed_floor', type='plane',
                  pos=num([0., 0., floor_z]), size='0 0 .02',
                  rgba='.24 .25 .28 1', friction='1 .005 .0001',
                  contype='1', conaffinity='1', group='2')

    # Visual mesh only: source geometry has inferred unseen surfaces. Dynamics
    # use an explicit hollow bowl proxy instead of a solid convex hull.
    object_name = f'target_object_{args.object_id:02d}'
    mesh_path = (DATA/f'objects/object_{args.object_id:02d}/mesh.obj').resolve()
    ET.SubElement(asset, 'mesh', name=object_name+'_visual', file=str(mesh_path))
    body = ET.SubElement(world, 'body', name=object_name,
                         pos=num(positions[0]), quat=num(quats[0]))
    ET.SubElement(body, 'freejoint', name=object_name+'_free')
    dims = np.asarray(info['dimensions_xyz_m'], dtype=float)
    radius = float(max(dims[:2])/2)
    height = float(dims[2])
    mass = float(args.assumed_mass)
    if mass <= 0 or radius <= 0 or height <= 0:
        raise ValueError('Mass, radius and height must be positive')
    # Approximate inertia of a circular shell; values declared, not measured.
    ix = mass*(3*radius*radius+height*height)/12
    iz = mass*radius*radius/2
    ET.SubElement(body, 'inertial', pos=num([0.,0.,height/2]),
                  mass=f'{mass:.12g}', diaginertia=num([ix, ix, iz]))
    ET.SubElement(body, 'geom', name=object_name+'_visual_geom', type='mesh',
                  mesh=object_name+'_visual', rgba='.58 .73 .81 1',
                  contype='0', conaffinity='0', density='0', group='1')
    wall = min(.005, radius*.09)
    bottom_h = min(.003, height*.12)
    ET.SubElement(body, 'geom', name=object_name+'_bottom_collision', type='cylinder',
                  pos=num([0,0,bottom_h]), size=num([radius-wall, bottom_h]),
                  density='0', contype='1', conaffinity='1',
                  friction='.65 .01 .0005', group='0')
    nwall = 16
    half_tangent = (radius-wall/2)*math.sin(math.pi/nwall)*1.07
    wall_height = max(.005, height-2*bottom_h)
    for k in range(nwall):
        angle = 2*math.pi*k/nwall
        centre = [(radius-wall/2)*math.cos(angle),
                  (radius-wall/2)*math.sin(angle), bottom_h+wall_height/2]
        q = [math.cos(angle/2),0,0,math.sin(angle/2)]
        ET.SubElement(body, 'geom', name=f'{object_name}_wall_{k:02d}', type='box',
                      pos=num(centre), quat=num(q),
                      size=num([wall/2, half_tangent, wall_height/2]),
                      density='0', contype='1', conaffinity='1',
                      friction='.65 .01 .0005', group='0')

    # Complete 52-channel native implicit position actuators. The 28 body
    # gains are an old Vita checkpoint contract; hand/neck gains are explicit
    # assumptions, and are not represented as learned SONIC policy actions.
    old = json.loads(CONTRACT.read_text(encoding='utf-8'))['contract']
    body_gain = {n: (float(kp),float(kd)) for n,kp,kd in zip(
        old['joint_names'],old['joint_stiffness'],old['joint_damping'])}
    actuators = mjxml.find('actuator')
    if actuators is None: actuators = ET.SubElement(mjxml, 'actuator')
    joint_names = []
    for joint in urdf_root.findall('joint'):
        if joint.get('type') == 'fixed': continue
        name = joint.get('name'); joint_names.append(name)
        effort = float(joint.find('limit').get('effort'))
        kp,kd = body_gain.get(name,(8., .2))
        motor_cap = min(effort, 3.) if name.startswith(('lh_','rh_')) else effort
        ET.SubElement(actuators, 'position', name='motor_'+name, joint=name,
                      gear='1', kp=f'{kp:.12g}', kv=f'{kd:.12g}',
                      forcelimited='true', forcerange=num([-motor_cap,motor_cap]))

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    suffix = '_'+args.variant if args.variant else ''
    xml_path = output/f'vita_object_{args.object_id:02d}_dynamic{suffix}.xml'
    xml_text = ET.tostring(mjxml, encoding='unicode')
    model = mujoco.MjModel.from_xml_string(xml_text)
    if model.nq != 66 or model.nv != 64 or model.nu != 52:
        raise RuntimeError(f'Unexpected model nq={model.nq},nv={model.nv},nu={model.nu}')
    xml_path.write_text(xml_text, encoding='utf-8')
    def joint_address(name):
        jid = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
        return {'qpos':int(model.jnt_qposadr[jid]),'qvel':int(model.jnt_dofadr[jid])}
    object_joint = joint_address(object_name+'_free')
    robot_joint = joint_address('vita_demo_free_root')
    data = mujoco.MjData(model)
    # Static support probe: only set initial state, then integrate unforced.
    data.qpos[robot_joint['qpos']:robot_joint['qpos']+3] = [20,20,2]
    data.qpos[object_joint['qpos']:object_joint['qpos']+3] = [centre_xy[0],centre_xy[1],table_top+.001]
    data.qpos[object_joint['qpos']+3:object_joint['qpos']+7] = [1,0,0,0]
    mujoco.mj_forward(model,data)
    initial_contacts = [model.geom(i).name+' / '+model.geom(j).name
        for i,j in [(int(data.contact[k].geom1),int(data.contact[k].geom2))for k in range(data.ncon)]]
    for _ in range(100):
        mujoco.mj_step(model,data)
    support_z = float(data.qpos[object_joint['qpos']+2])
    if not np.all(np.isfinite(data.qpos)) or support_z < table_top-.025:
        raise RuntimeError(f'Dynamic bowl escaped support: {support_z} vs table {table_top}')
    report = {
        'schema':'fourth_vita_sim_assets.v1','source_object_id':args.object_id,
        'source_frames':[args.start,args.stop-1],'source_valid_count':int(len(positions)),
        'source_status_counts':{str(k):int(v)for k,v in zip(*np.unique(statuses,return_counts=True))},
        'model':str(xml_path),'model_sha256':digest(xml_path),
        'urdf_sha256':digest(URDF),'object_mesh_sha256':digest(mesh_path),
        'compiled':True,'nq':model.nq,'nv':model.nv,'nu':model.nu,
        'robot_freejoint':'vita_demo_free_root','robot_freejoint_address':robot_joint,
        'object_body':object_name,'object_freejoint':object_name+'_free',
        'object_freejoint_address':object_joint,
        'actuator_prefix':'motor_','actuator_joint_names':joint_names,
        'object_initial_pose':np.r_[positions[0],quats[0]].tolist(),
        'table_support':{'name':'table_support','centre':table_centre.tolist(),
                         'top_z_at_centre':float(table_top),'normal':sim_normal.tolist(),
                         'half_size_xy_m':table_half_xy.tolist(),
                         'source':'object_pose_qa fitted plane + local source object XY extent'},
        'floor_support':{'name':'assumed_floor','z_m':floor_z,'source':'assumed table minus 0.75m'},
        'object_collision':{'name':object_name,'body_type':'dynamic_freejoint_not_mocap',
                            'bottom_disc':1,'wall_segments':nwall,'hollow_interior':True,
                            'radius_m':radius,'height_m':height},
        'assumptions':{'mass_kg':mass,'ixx_iyy_kg_m2':ix,'izz_kg_m2':iz,
                       'table_friction':.75,'object_friction':.65,
                       'not_measured':True,'hand_neck_gains_are_assumed':True,
                       'robot_mass_inertia_from_supplied_urdf_preserved':True},
        'support_probe':{'initial_contacts':initial_contacts,'steps':100,
                         'timestep_s':model.opt.timestep,'final_object_bottom_z_m':support_z,
                         'finite_state':True,'passed':True},
        'physics_validation_scope':'Static bowl-on-table drop only; no hand interaction, policy success or real material calibration claimed.'}
    report['variant'] = args.variant or 'v1'
    report_path = output/f'vita_object_{args.object_id:02d}_assets_report{suffix}.json'
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'model':str(xml_path),'report':str(report_path),
                      'nq':model.nq,'nv':model.nv,'nu':model.nu,
                      'object':object_name,'object_qpos':object_joint['qpos'],
                      'table_top':table_top,'support_probe_z':support_z}),flush=True)


def build_visual_context(args):
    """Append the observed static scene surface for rendering only.

    This is a separate model and cannot replace the frozen physics XML in
    trajectory or contact audits. The extra geom has collisions disabled and
    adds no bodies, joints or actuators.
    """
    output = Path(args.output_dir)
    base_path = output/f'vita_object_{args.object_id:02d}_dynamic.xml'
    if not base_path.is_file():
        raise FileNotFoundError(base_path)
    root = ET.parse(base_path).getroot()
    asset = root.find('asset')
    world = root.find('worldbody')
    mesh_path = (DATA/'scene/visual_mesh.obj').resolve()
    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)
    ET.SubElement(asset,'mesh',name='observed_scene_visual',file=str(mesh_path))
    ET.SubElement(world,'geom',name='observed_scene_visual_geom',type='mesh',
                  mesh='observed_scene_visual',rgba='.8 .8 .8 1',
                  density='0',contype='0',conaffinity='0',group='4')
    text = ET.tostring(root,encoding='unicode')
    original = mujoco.MjModel.from_xml_path(str(base_path))
    context = mujoco.MjModel.from_xml_string(text)
    if (context.nq,context.nv,context.nu)!=(original.nq,original.nv,original.nu):
        raise RuntimeError('Visual context unexpectedly changed state/action sizes')
    context_path=output/f'vita_object_{args.object_id:02d}_context.xml'
    context_path.write_text(text,encoding='utf-8')
    report={'schema':'fourth_vita_render_context.v1',
            'base_physics_model':str(base_path),'base_physics_sha256':digest(base_path),
            'render_context_model':str(context_path),'render_context_sha256':digest(context_path),
            'scene_visual_mesh':str(mesh_path),'scene_mesh_sha256':digest(mesh_path),
            'nq':context.nq,'nv':context.nv,'nu':context.nu,
            'scene_geom_group':4,'scene_geom_contype':0,'scene_geom_conaffinity':0,
            'scene_geom_is_collision':False,
            'other_five_objects_not_added':True,
            'scope':'Display-only observed surface. Use base_physics_model for all contact, torque and deterministic-replay audits.'}
    report_path=output/f'vita_object_{args.object_id:02d}_context_report.json'
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'context_model':str(context_path),'report':str(report_path),
                      'nq':context.nq,'nv':context.nv,'nu':context.nu}),flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--object-id',type=int,default=5)
    ap.add_argument('--start',type=int,default=567)
    ap.add_argument('--stop',type=int,default=627)
    ap.add_argument('--assumed-mass',type=float,default=.25)
    ap.add_argument('--floor-z',type=float)
    ap.add_argument('--output-dir',default=str(OUTPUT))
    ap.add_argument('--variant',choices=['v2'])
    ap.add_argument('--context-only',action='store_true')
    args=ap.parse_args()
    build_visual_context(args) if args.context_only else build(args)
