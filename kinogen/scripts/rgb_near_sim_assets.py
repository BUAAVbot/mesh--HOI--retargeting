"""Build two dynamic Vita52 robots, a fixed bench and a dynamic shoe.

Visual meshes are source reconstructions. Collision proxies and shoe physical
parameters are explicit engineering assumptions, not measured material data.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def num(x):
    return ' '.join(f'{float(v):.17g}' for v in x)


def save_xml(root, path):
    ET.indent(root)
    path.write_text(ET.tostring(root, encoding='unicode')+'\n', encoding='utf-8')


def source_robot(source):
    r = ET.parse(source).getroot()
    robot = r.find("worldbody/body[@name='base_link']")
    if robot is None:
        raise ValueError('Expected the original near-source Vita root')
    used = {g.get('mesh') for g in robot.findall('.//geom') if g.get('mesh')}
    assets = ET.Element('asset')
    for child in r.find('asset'):
        if child.get('name') in used:
            assets.append(copy.deepcopy(child))
    if len(assets) != len(used):
        raise ValueError('A robot visual/collision mesh dependency is missing')
    return copy.deepcopy(robot), assets, copy.deepcopy(r.find('actuator'))


def base_document():
    r = ET.Element('mujoco', model='rgb_near_two_vita_bench_shoe')
    ET.SubElement(r, 'compiler', angle='radian')
    ET.SubElement(r, 'option', timestep='.005', integrator='implicitfast',
                  gravity='0 0 -9.81', iterations='100', cone='elliptic')
    vis = ET.SubElement(r, 'visual')
    ET.SubElement(vis, 'global', offwidth='1280', offheight='800')
    ET.SubElement(vis, 'headlight', ambient='.45 .45 .45', diffuse='.65 .65 .65')
    ET.SubElement(r, 'statistic', center='-.4 -.1 .7', extent='2.5')
    return r


def robot_proxy(source, out, mujoco):
    robot, assets, actuator = source_robot(source)
    r = base_document()
    r.append(assets)
    w = ET.SubElement(r, 'worldbody')
    robot.set('pos', '0 0 .9')
    w.append(robot)
    ET.SubElement(w, 'geom', name='ground_proxy', type='plane', pos='0 0 0',
                  size='0 0 .05', contype='2', conaffinity='3', friction='1 .01 .0005',
                  rgba='.3 .34 .28 1')
    r.append(actuator)
    path = out/'vita_single_proxy.xml'
    save_xml(r, path)
    model = mujoco.MjModel.from_xml_path(str(path))
    if (model.nq, model.nv, model.nu) != (59, 58, 52):
        raise ValueError('Unexpected unprefixed single-robot dimensions')
    mujoco.mj_saveModel(model, str(out/'vita_single_proxy.mjb'))
    return path


def prefix_robot(body, actuators, prefix, collision_type, affinity):
    b = copy.deepcopy(body)
    for elem in b.iter():
        if elem.get('name'):
            elem.set('name', prefix+elem.get('name'))
        if elem.tag == 'geom' and int(elem.get('contype', '0')):
            elem.set('contype', str(collision_type))
            elem.set('conaffinity', str(affinity))
    a = []
    for original in actuators:
        elem = copy.deepcopy(original)
        elem.set('name', prefix+elem.get('name'))
        elem.set('joint', prefix+elem.get('joint'))
        a.append(elem)
    return b, a


def weighted_median(v, w, np):
    order = np.argsort(v)
    return float(v[order][np.searchsorted(np.cumsum(w[order]), np.sum(w)/2)])


def bench_fit(data, np):
    z = np.load(data/'scene/bench_genrecon.npz', allow_pickle=False)
    v, faces = z['vertices'].astype(float), z['faces']
    triangles = v[faces]
    centre = triangles.mean(axis=1)
    cross = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    area = np.linalg.norm(cross, axis=1)/2
    normals = cross/np.maximum(2*area[:, None], 1e-20)
    horizontal = (abs(normals[:, 2]) > .85) & (centre[:, 2] > .2) & (centre[:, 2] < .6)
    bins = np.floor(centre[horizontal, 2]*100).astype(int)
    histogram = np.bincount(bins-bins.min(), weights=area[horizontal])
    peak = (int(np.argmax(histogram))+int(bins.min()))/100
    selected = horizontal & (centre[:, 2] >= peak-.005) & (centre[:, 2] < peak+.015) & (centre[:, 1] < .4)
    top = weighted_median(centre[selected, 2], area[selected], np)
    eigenvalue, eigvec = np.linalg.eigh(np.cov(v[:, :2].T))
    along = eigvec[:, -1]
    if along[0] < 0:
        along = -along
    across = np.asarray([-along[1], along[0]])
    axes = np.column_stack([along, across])
    xy = v[:, :2] @ axes
    # Full source width; the support depth is bounded by the source seat front
    # and the lower backrest region. These are disclosed simple boxes.
    low_long, high_long = np.quantile(xy[:, 0], [.002, .998])
    low_depth = float(np.min(xy[:, 1]))
    high_depth = .44
    return {'top_z_m': top, 'seat_bottom_z_m': top-.055,
            'long_min_max_m': [float(low_long), float(high_long)],
            'depth_min_max_m': [low_depth, high_depth],
            'yaw_rad': float(np.arctan2(along[1], along[0])),
            'axes_xy': axes.tolist(), 'horizontal_area_peak_z_m': peak,
            'source_bounds': [v.min(axis=0).tolist(), v.max(axis=0).tolist()],
            'fit_method': 'horizontal triangle-area mode/median; PCA long axis; simple support boxes',
            'back_depth_centre_m': .58, 'back_vertical_bounds_m': [.44, .85]}


def texture_mesh(asset, data, mesh_name, stem, material_name):
    path = data/stem
    ET.SubElement(asset, 'mesh', name=mesh_name, file=str(path.with_suffix('.obj').resolve()))
    ET.SubElement(asset, 'texture', name=material_name+'_texture', type='2d',
                  file=str(path.with_name(path.name+'_color.png').resolve()))
    ET.SubElement(asset, 'material', name=material_name, texture=material_name+'_texture',
                  specular='0', shininess='0', rgba='1 1 1 1')


def validate_shoe_support(assets, shoe, out, mujoco, np):
    r = base_document()
    selected = ET.SubElement(r, 'asset')
    for elem in assets:
        if elem.get('name') in {'source_shoe', 'shoe_rgb', 'shoe_rgb_texture'}:
            selected.append(copy.deepcopy(elem))
    world = ET.SubElement(r, 'worldbody')
    ET.SubElement(world, 'geom', name='ground_proxy', type='plane', pos='0 0 0', size='0 0 .05',
                  contype='2', conaffinity='13', friction='1 .01 .0005')
    world.append(copy.deepcopy(shoe))
    path = out/'shoe_static_support.xml'
    save_xml(r, path)
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    boxes = [g for g in range(model.ngeom) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX]
    minimum = min(float(data.geom_xpos[g, 2] - abs(data.geom_xmat[g].reshape(3,3)[2])@model.geom_size[g]) for g in boxes)
    reset_lift = max(.003-minimum, 0.)
    data.qpos[2] += reset_lift
    mujoco.mj_forward(model, data)
    initial = data.qpos.copy()
    max_penetration = 0.
    for _ in range(600):
        mujoco.mj_step(model, data)
        max_penetration = max(max_penetration, max([max(-float(data.contact[j].dist),0.) for j in range(data.ncon)]+[0.]))
    speed = float(np.linalg.norm(data.qvel[:3]))
    passed = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all() and speed < .02 and max_penetration < .02 and data.ncon > 0)
    result = {'status':'passed_static_shoe_support' if passed else 'failed_static_shoe_support',
              'model':str(path.resolve()), 'model_sha256':sha(path), 'steps':600, 'duration_s':float(data.time),
              'initial_qpos':initial.tolist(), 'source_initial_floor_lift_m':reset_lift,
              'final_qpos':data.qpos.tolist(), 'final_linear_speed_m_s':speed,
              'max_contact_penetration_m':max_penetration,
              'ground_contact_points_final':int(data.ncon), 'object_dynamic_freejoint':model.nq == 7,
              'object_actuators':model.nu, 'external_forces':False,
              'validates_full_task':False}
    (out/'shoe_static_support_report.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    if not passed:
        raise ValueError('Standalone shoe/ground support test failed: '+json.dumps(result))
    return result


def audit_inertias(model, proxy_model, mujoco, np):
    result = {}
    for prefix in ['p1_', 'p2_']:
        largest = {key:0. for key in ['body_mass', 'body_inertia', 'body_ipos', 'body_iquat']}
        ids = []
        for bid in range(1, proxy_model.nbody):
            name = mujoco.mj_id2name(proxy_model, mujoco.mjtObj.mjOBJ_BODY, bid)
            target = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prefix+name)
            if target < 0:
                raise ValueError('Robot hierarchy body missing: '+prefix+name)
            ids.append(target)
            for key in largest:
                largest[key] = max(largest[key], float(np.max(abs(getattr(model,key)[target]-getattr(proxy_model,key)[bid]))))
        if any(v != 0. for v in largest.values()):
            raise ValueError('Robot inertias changed during composition')
        result[prefix] = {'links':len(ids), 'mass_kg':float(np.sum(model.body_mass[ids])), 'max_abs_property_errors':largest}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=ROOT/'data/rgb_261026_near_handoff')
    p.add_argument('--source', type=Path, default=ROOT/'outputs/fourth_full_sequence/assets/vita_six_objects_newft_isaac52_isolated.xml')
    p.add_argument('--output-dir', type=Path, default=ROOT/'outputs/rgb_near_pipeline/assets')
    p.add_argument('--proxy-only', action='store_true')
    p.add_argument('--audit-existing', action='store_true', help='Audit frozen XML without modifying it')
    a = p.parse_args()
    import numpy as np
    import mujoco
    a.output_dir.mkdir(parents=True, exist_ok=True)
    if a.audit_existing:
        model = mujoco.MjModel.from_xml_path(str(a.output_dir/'vita_bench_two52.xml'))
        proxy_model = mujoco.MjModel.from_xml_path(str(a.output_dir/'vita_single_proxy.xml'))
        path = a.output_dir/'assets_report.json'
        report = json.loads(path.read_text())
        report['robot_original_inertia_audit'] = audit_inertias(model, proxy_model, mujoco, np)
        temp = path.with_suffix('.partial.json')
        temp.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        temp.replace(path)
        print(json.dumps(report['robot_original_inertia_audit']))
        return
    proxy = robot_proxy(a.source, a.output_dir, mujoco)
    if a.proxy_only:
        print(json.dumps({'proxy': str(proxy)}))
        return
    robot, assets, actuators = source_robot(a.source)
    r = base_document()
    r.append(assets)
    texture_mesh(assets, a.data, 'source_near_background', Path('scene/near_background'), 'background_rgb')
    texture_mesh(assets, a.data, 'source_bench', Path('scene/bench_genrecon'), 'bench_rgb')
    texture_mesh(assets, a.data, 'source_shoe', Path('objects/shoe/mesh'), 'shoe_rgb')
    w = ET.SubElement(r, 'worldbody')
    ET.SubElement(w, 'light', name='scene_light', pos='0 -3 4', dir='0 1 -1', diffuse='.6 .6 .6')
    for name, mesh, material in [('background_visual', 'source_near_background', 'background_rgb'),
                                 ('bench_visual', 'source_bench', 'bench_rgb')]:
        ET.SubElement(w, 'geom', name=name, type='mesh', mesh=mesh, material=material,
                      contype='0', conaffinity='0', group='1')
    ET.SubElement(w, 'geom', name='ground_proxy', type='plane', pos='0 0 0',
                  size='0 0 .05', contype='2', conaffinity='13', friction='1 .01 .0005',
                  rgba='.25 .31 .22 1', group='2')
    fit = bench_fit(a.data, np)
    axes = np.asarray(fit['axes_xy'])
    long0, long1 = fit['long_min_max_m']
    depth0, depth1 = fit['depth_min_max_m']
    yaw = fit['yaw_rad']
    def box(name, centre_ldz, halfsize, rgba='.6 .36 .18 .45'):
        xy = axes @ np.asarray(centre_ldz[:2])
        ET.SubElement(w, 'geom', name=name, type='box', pos=num([*xy, centre_ldz[2]]),
                      euler=num([0, 0, yaw]), size=num(halfsize), rgba=rgba,
                      contype='2', conaffinity='13', friction='.8 .01 .0005', group='0')
    box('bench_seat_proxy', [(long0+long1)/2, (depth0+depth1)/2, fit['top_z_m']-.0275],
        [(long1-long0)/2, (depth1-depth0)/2, .0275])
    box('bench_back_proxy', [(long0+long1)/2, fit['back_depth_centre_m'], .645],
        [(long1-long0)/2, .035, .205])
    for i, along in enumerate([long0+.18, long1-.18]):
        for j, across in enumerate([depth0+.10, depth1-.10]):
            h = fit['seat_bottom_z_m']/2
            box(f'bench_leg_{i}_{j}_proxy', [along, across, h], [.035, .035, h])
    action = ET.SubElement(r, 'actuator')
    for person, ctype, affinity in [(1, 1, 6), (2, 4, 3)]:
        body, motors = prefix_robot(robot, actuators, f'p{person}_', ctype, affinity)
        pose = np.load(a.data/f'human/person_{person:02d}/root_poses_world.npz', allow_pickle=False)
        body.set('pos', num(pose['position_m'][0]))
        w.append(body)
        action.extend(motors)
    mesh = np.load(a.data/'objects/shoe/mesh.npz', allow_pickle=False)
    vertices = mesh['vertices'].astype(float)
    poses = np.load(a.data/'objects/shoe/poses_world.npz', allow_pickle=False)
    covariance_values, covariance_axes = np.linalg.eigh(np.cov(vertices.T))
    up = covariance_axes[:, 1]
    if (poses['world_from_object'][0, :3, :3] @ up)[2] < 0:
        up = -up
    length = covariance_axes[:, 2]
    width = np.cross(up, length)
    qrot = np.column_stack([length, width, up])
    local = vertices @ qrot
    low, high = local.min(axis=0), local.max(axis=0)
    # Source shoe is a partial side surface, so the absent cross-shoe width
    # cannot be inferred. Use a disclosed 0.11 m shell width, preserve length
    # and height extents, and leave the top/heel open for fingers/foot volume.
    low[1], high[1] = -.055, .055
    centre = (low+high)/2
    dims = high-low
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, qrot.ravel())
    mass = .30
    inertia = mass/12*np.asarray([dims[1]**2+dims[2]**2, dims[0]**2+dims[2]**2, dims[0]**2+dims[1]**2])
    shoe = ET.SubElement(w, 'body', name='shoe', pos=num(poses['position_m'][0]), quat=num(poses['quaternion_wxyz'][0]))
    ET.SubElement(shoe, 'freejoint', name='shoe_free')
    ET.SubElement(shoe, 'inertial', pos=num(qrot@centre), quat=num(q), mass=str(mass), diaginertia=num(inertia))
    ET.SubElement(shoe, 'geom', name='shoe_visual', type='mesh', mesh='source_shoe', material='shoe_rgb',
                  contype='0', conaffinity='0', density='0', group='1')
    thickness = .004
    def shell_box(name, mid, half):
        ET.SubElement(shoe, 'geom', name=name, type='box', pos=num(qrot@np.asarray(mid)), quat=num(q), size=num(half),
                      density='0', contype='8', conaffinity='7', friction='.9 .01 .0005',
                      rgba='.25 .25 .28 .45', group='0')
    shell_box('shoe_sole_proxy', [centre[0], centre[1], low[2]+thickness/2], [dims[0]/2, dims[1]/2, thickness/2])
    for side, y in [('left', low[1]+thickness/2), ('right', high[1]-thickness/2)]:
        shell_box('shoe_'+side+'_wall_proxy', [centre[0], y, centre[2]], [dims[0]/2, thickness/2, dims[2]/2])
    shell_box('shoe_toe_wall_proxy', [high[0]-thickness/2, centre[1], centre[2]], [thickness/2, dims[1]/2, dims[2]/2])
    for name, point in [('shoe_wear_anchor', [low[0]+.025, centre[1], high[2]-.01]),
                        ('shoe_grasp_anchor', [low[0]+.012, centre[1], high[2]-.005])]:
        ET.SubElement(shoe, 'site', name=name, pos=num(qrot@np.asarray(point)),
                      size='.006', rgba='1 .2 .1 .7', group='3')
    equality = ET.SubElement(r, 'equality')
    for label, body in [('wear_left', 'p1_left_ankle_roll_link'),
                        ('wear_right', 'p1_right_ankle_roll_link'),
                        ('grasp_left', 'p1_lh_hand_base_link'),
                        ('grasp_right', 'p1_rh_hand_base_link')]:
        ET.SubElement(equality, 'weld', name='shoe_'+label, body1=body, body2='shoe',
                      active='false', solref='.06 1', solimp='.9 .95 .001', torquescale='.03')
    path = a.output_dir/'vita_bench_two52.xml'
    save_xml(r, path)
    model = mujoco.MjModel.from_xml_path(str(path))
    if (model.nq, model.nv, model.nu) != (125, 122, 104):
        raise ValueError(f'Unexpected combined dimensions {(model.nq, model.nv, model.nu)}')
    mujoco.mj_saveModel(model, str(a.output_dir/'vita_bench_two52.mjb'))
    shoe_support = validate_shoe_support(assets, shoe, a.output_dir, mujoco, np)
    proxy_model = mujoco.MjModel.from_xml_path(str(proxy))
    inertia_audit = audit_inertias(model, proxy_model, mujoco, np)
    report = {'schema':'rgb_near.two_vita_dynamic_assets.v1', 'model':str(path.resolve()), 'model_sha256':sha(path),
              'single_proxy':str(proxy.resolve()), 'single_proxy_sha256':sha(proxy),
              'source_robot_model':str(a.source.resolve()), 'source_robot_model_sha256':sha(a.source),
              'source_coordinates':'right-handed, +Z up, approximate metres, unchanged',
              'nq':model.nq, 'nv':model.nv, 'nu':model.nu,
              'robot_prefixes':['p1_', 'p2_'], 'robot_roots':['p1_vita_demo_free_root','p2_vita_demo_free_root'],
              'robot_original_inertia_audit':inertia_audit,
              'shoe_body':'shoe', 'shoe_freejoint':'shoe_free', 'shoe_dynamic':True, 'shoe_mocap':False,
              'shoe_static_support_test':shoe_support,
              'bench_fixed_to_world':True, 'bench_collision_fit':fit,
              'inferred_contact_constraints':{'default_active':False,
                  'welds':['shoe_wear_left','shoe_wear_right','shoe_grasp_left','shoe_grasp_right'],
                  'solref':[.06,1.], 'torquescale_m':.03,
                  'semantics':'Optional compliant footwear/grasp hypotheses; phase activation, relative poses and constraint forces must be explicitly recorded by the physics driver. Not force ground truth.'},
              'shoe_collision_proxy':{'kind':'open top/heel shoe shell, sole plus two sides and toe',
                                      'source_partial_surface':True, 'assumed_width_m':.11, 'wall_thickness_m':thickness,
                                      'local_axes_columns':qrot.tolist(), 'proxy_dimensions_m':dims.tolist(),
                                      'assumed_mass_kg':mass, 'assumed_friction':.9},
              'contact_masks':'same-robot self contacts disabled per near-source training; inter-robot and world/shoe contacts enabled',
              'limitations':['Bench support boxes are fitted engineering proxies for generated visual geometry, not measured collision surfaces.',
                             'Shoe mass, inertia, friction and missing side width are assumed; shoe rotation in source is unobserved.',
                             'Bench is an anchored environment body; no claim of calibrated movable furniture dynamics.']}
    (a.output_dir/'assets_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
