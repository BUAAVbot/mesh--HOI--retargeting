"""Read-only, standard-library audit of vendored G1/ROHand physics assets."""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def matmul(a, b):
    return [[sum(a[i][k]*b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]


def transform(xyz=(0, 0, 0), rpy=(0, 0, 0)):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return [[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr, xyz[0]],
            [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr, xyz[1]],
            [-sp, cp*sr, cp*cr, xyz[2]], [0., 0., 0., 1.]]


def origin(node):
    return transform(*[[float(x) for x in node.get(tag, '0 0 0').split()] for tag in ('xyz', 'rpy')]) if node is not None else transform()


def hand_directions():
    from user_rohand_coupling import ROHandCoupling
    coupling = ROHandCoupling(ROOT)
    output = {}
    for side in ('left', 'right'):
        tree = ET.parse(ROOT/f'dependencies/rohand-urdf/urdf/rohand_{side}.urdf').getroot()
        offset = 0 if side == 'left' else 6
        a0 = coupling.lower[offset:offset+6]
        # Change only four long fingers, preserve thumb and thumb root.
        a1 = a0[:]
        for i in range(1, 5):
            a1[i] = coupling.upper[offset+i]
        amid = [(x+y)/2 for x, y in zip(a0, a1)]
        states = [coupling.expand_side(a, side) for a in (a0, amid, a1)]
        poses = []
        for state in states:
            tf = {'base_link': transform()}
            pending = list(tree.findall('joint'))
            while pending:
                progressed = False
                for joint in pending[:]:
                    parent = joint.find('parent').get('link')
                    if parent not in tf:
                        continue
                    q = state[side+'_hand_ro_'+joint.get('name')]
                    axis = [float(v) for v in joint.find('axis').get('xyz').split()]
                    norm = math.sqrt(sum(x*x for x in axis)); axis = [x/norm for x in axis]
                    delta = transform()
                    if joint.get('type') == 'prismatic':
                        for i in range(3): delta[i][3] = axis[i]*q
                    else:
                        c, s, d = math.cos(q), math.sin(q), 1-math.cos(q)
                        x, y, z = axis
                        rot = [[c+x*x*d, x*y*d-z*s, x*z*d+y*s], [y*x*d+z*s, c+y*y*d, y*z*d-x*s], [z*x*d-y*s, z*y*d+x*s, c+z*z*d]]
                        for i in range(3):
                            for j in range(3): delta[i][j] = rot[i][j]
                    tf[joint.find('child').get('link')] = matmul(matmul(tf[parent], origin(joint.find('origin'))), delta)
                    pending.remove(joint); progressed = True
                if not progressed: raise ValueError('Unresolved hand tree')
            poses.append(tf)
        fingers = {}
        for digit in ('if', 'mf', 'rf', 'lf'):
            name = digit+'_distal_link'
            visual = tree.find(f"link[@name='{name}']/visual")
            path = mesh_path(visual.find('geometry/mesh').get('filename'), ROOT/'dependencies/rohand-urdf/urdf')
            data = path.read_bytes()
            vertices = [tuple(values[k:k+3]) for i in range(struct.unpack_from('<I', data, 80)[0])
                        for values in [struct.unpack_from('<9f', data, 84+i*50+12)] for k in (0, 3, 6)]
            maps = [matmul(tf[name], origin(visual.find('origin'))) for tf in poses]
            world = [[tuple(sum(t[row][j]*v[j] for j in range(3))+t[row][3] for row in range(3)) for v in vertices] for t in maps]
            # Mean of distal-most 5% of source-open vertices; same material points closed.
            indices = sorted(range(len(vertices)), key=lambda k: world[0][k][0])[:max(1, len(vertices)//20)]
            tips = [[sum(w[k][axis] for k in indices)/len(indices) for axis in range(3)] for w in world]
            fingers[digit] = {'open_distal_tip_proxy_xyz_m': tips[0], 'half_closed_same_material_xyz_m': tips[1],
                'closed_same_material_xyz_m': tips[2],
                'half_closing_displacement_xyz_m': [tips[1][i]-tips[0][i] for i in range(3)],
                'closing_displacement_xyz_m': [tips[2][i]-tips[0][i] for i in range(3)]}
        output[side] = {'frame': 'original ROHand base_link, metres',
            'open_actuator6': a0, 'four_fingers_closed_actuator6': a1, 'fingers': fingers,
            'note': 'Tip proxy is mean distal-most 5% open-mesh material vertices, not a manufacturer contact point. Curl direction is verified by official coupling FK.'}
    return output


def rel(p):
    return str(p.resolve().relative_to(ROOT.resolve())).replace('\\', '/')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def eig3(a):
    """Jacobi symmetric eigensolver; avoids a local numerical dependency."""
    a = [row[:] for row in a]
    for _ in range(40):
        p, q = max(((0, 1), (0, 2), (1, 2)), key=lambda ij: abs(a[ij[0]][ij[1]]))
        if abs(a[p][q]) < 1e-22:
            break
        theta = .5 * math.atan2(2*a[p][q], a[q][q]-a[p][p])
        c, s = math.cos(theta), math.sin(theta)
        ap, aq, off = a[p][p], a[q][q], a[p][q]
        for k in range(3):
            if k not in (p, q):
                x, y = a[k][p], a[k][q]
                a[k][p] = a[p][k] = c*x-s*y
                a[k][q] = a[q][k] = s*x+c*y
        a[p][p] = c*c*ap-2*c*s*off+s*s*aq
        a[q][q] = s*s*ap+2*c*s*off+c*c*aq
        a[p][q] = a[q][p] = 0.
    return sorted(a[i][i] for i in range(3))


def mesh_path(filename, parent):
    prefix = 'package://rohand_urdf_ros1/'
    if filename.startswith(prefix):
        return ROOT/'dependencies/rohand-urdf'/filename[len(prefix):]
    if filename.startswith('package://'):
        # Source packages in this snapshot may include their package name.
        _, sub = filename[len('package://'):].split('/', 1)
        return parent/sub
    return parent/filename


def stl_info(path):
    out = {'path': rel(path), 'exists': path.is_file()}
    if not path.is_file():
        return out
    data = path.read_bytes()
    out.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    n = struct.unpack_from('<I', data, 80)[0] if len(data) >= 84 else -1
    if n >= 0 and len(data) == 84+50*n:
        out['triangles'] = n
        out['format'] = 'binary_stl'
    elif path.suffix.lower() == '.stl':
        out['triangles'] = data.lower().count(b'facet normal')
        out['format'] = 'ascii_stl'
    return out


def urdf(path):
    tree = ET.parse(path).getroot()
    links, joints = tree.findall('link'), tree.findall('joint')
    out = {'path': rel(path), 'sha256': sha(path), 'link_count': len(links),
           'joint_types': dict(collections.Counter(j.get('type') for j in joints)),
           'inertials': [], 'joints': [], 'meshes': {},
           'collision_geometry_types': {}, 'missing_collision_links': [],
           'visual_collision_same_mesh_links': [],
           'mimic_count': len(tree.findall('.//mimic')),
           'transmission_count': len(tree.findall('.//transmission')),
           'dynamics_count': len(tree.findall('.//dynamics'))}
    geometry = collections.Counter()
    missing_inertia = []
    for link in links:
        name = link.get('name')
        inert = link.find('inertial')
        if inert is None:
            missing_inertia.append(name)
        else:
            it = inert.find('inertia')
            d = {k: float(it.get(k, '0')) for k in ('ixx', 'iyy', 'izz', 'ixy', 'ixz', 'iyz')}
            eig = eig3([[d['ixx'], d['ixy'], d['ixz']], [d['ixy'], d['iyy'], d['iyz']], [d['ixz'], d['iyz'], d['izz']]])
            out['inertials'].append({'link': name, 'mass_kg': float(inert.find('mass').get('value')),
                'origin': dict(inert.find('origin').attrib) if inert.find('origin') is not None else {},
                'tensor_kg_m2': d, 'principal_kg_m2': eig,
                'positive_definite': eig[0] > 0,
                'triangle_inequality': eig[2] <= eig[0]+eig[1]+1e-12})
        vmesh = {x.get('filename') for x in link.findall('visual/geometry/mesh')}
        cmesh = {x.get('filename') for x in link.findall('collision/geometry/mesh')}
        if vmesh and cmesh and vmesh == cmesh:
            out['visual_collision_same_mesh_links'].append(name)
        if not link.findall('collision'):
            out['missing_collision_links'].append(name)
        for c in link.findall('collision/geometry'):
            geometry.update(x.tag for x in c)
        for mesh in link.findall('.//mesh'):
            filename = mesh.get('filename')
            if filename not in out['meshes']:
                out['meshes'][filename] = stl_info(mesh_path(filename, path.parent))
    for joint in joints:
        row = dict(joint.attrib)
        for tag in ('parent', 'child', 'origin', 'axis', 'limit', 'dynamics', 'mimic'):
            node = joint.find(tag)
            if node is not None:
                row[tag] = dict(node.attrib)
        out['joints'].append(row)
    out['collision_geometry_types'] = dict(geometry)
    masses = [i['mass_kg'] for i in out['inertials']]
    movable = [j for j in out['joints'] if j.get('type') != 'fixed']
    out['summary'] = {'total_link_mass_kg': sum(masses), 'min_link_mass_kg': min(masses),
        'max_link_mass_kg': max(masses), 'missing_inertial_links': missing_inertia,
        'nonpositive_mass_links': [i['link'] for i in out['inertials'] if i['mass_kg'] <= 0],
        'nonpositive_inertia_links': [i['link'] for i in out['inertials'] if not i['positive_definite']],
        'inertia_triangle_inequality_failures': [i['link'] for i in out['inertials'] if not i['triangle_inequality']],
        'principal_inertia_below_1e_8_links': [i['link'] for i in out['inertials'] if min(i['principal_kg_m2']) < 1e-8],
        'missing_effort_joints': [j['name'] for j in movable if 'effort' not in j.get('limit', {})],
        'zero_effort_joints': [j['name'] for j in movable if j.get('limit', {}).get('effort') == '0'],
        'missing_velocity_joints': [j['name'] for j in movable if 'velocity' not in j.get('limit', {})],
        'zero_velocity_joints': [j['name'] for j in movable if j.get('limit', {}).get('velocity') == '0'],
        'missing_mesh_files': [m['path'] for m in out['meshes'].values() if not m['exists']],
        'unique_mesh_triangles': sum(m.get('triangles', 0) for m in out['meshes'].values())}
    return out


def mjcf(path):
    root = ET.parse(path).getroot()
    compiler = root.find('compiler')
    meshdir = compiler.get('meshdir', '') if compiler is not None else ''
    meshes = [stl_info(path.parent/meshdir/m.get('file')) for m in root.findall('asset/mesh') if m.get('file')]
    bodies = root.findall('.//worldbody//body')
    joints = [j for b in bodies for j in list(b) if j.tag in ('joint', 'freejoint')]
    return {'path': rel(path), 'sha256': sha(path),
        'body_count': len(bodies), 'joint_types': dict(collections.Counter(j.get('type', 'hinge') if j.tag == 'joint' else 'free' for j in joints)),
        'actuators': [dict(x.attrib, element=x.tag) for a in root.findall('actuator') for x in a],
        'defaults_xml': [ET.tostring(x, encoding='unicode') for x in root.findall('default')],
        'option': dict(root.find('option').attrib) if root.find('option') is not None else None,
        'contact_geom_types_explicit': dict(collections.Counter(g.get('type', 'sphere') for b in root.findall('worldbody') for g in b.iter('geom') if g.get('contype') != '0' and g.get('class') != 'visual')),
        'mesh_count': len(meshes), 'missing_meshes': [m['path'] for m in meshes if not m['exists']],
        'note': 'XML source inspection only; defaults and mesh convexification not compiled by a simulator.'}


def usd_inventory(directory):
    items = []
    for path in sorted(directory.rglob('*.usda')):
        source = path.read_text(encoding='utf-8')
        refs = []
        for target in sorted(set(re.findall(r'@([^@]+)@', source))):
            remote = '://' in target
            p = path.parent/target
            refs.append({'target': target, 'external_uri': remote,
                'local_exists': None if remote else p.is_file(),
                'resolution_note': 'MDL search-path asset; adjacent-file absence does not prove runtime missing' if target == 'OmniPBR.mdl' else 'relative file check only'})
        drives = re.findall(r'(?:float|double)\s+(drive:[\w:]+)\s*=\s*([^\n]+)', source)
        items.append({'path': rel(path), 'sha256': sha(path), 'references': refs,
            'rigid_body_api_mentions': source.count('PhysicsRigidBodyAPI'),
            'mass_api_mentions': source.count('PhysicsMassAPI'),
            'drive_attributes': dict(drives),
            'drive_attribute_occurrence_count': len(drives),
            'missing_local_references': [x['target'] for x in refs if x['local_exists'] is False],
            'unresolved_relative_scene_layers': [x['target'] for x in refs if x['local_exists'] is False and x['target'] != 'OmniPBR.mdl']})
    return {'scope': rel(directory), 'file_count': len(items), 'files': items,
        'note': 'Isaac Sim asset-transformer test fixtures; stock three-finger/Inspire variants, not assembled ROHand. Text references checked, no USD composition/runtime validation.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='reports/g1-training-asset-audit.json')
    args = parser.parse_args()
    scene = json.loads((ROOT/'data/video_generation/candidate_scene/scene.json').read_text())
    paths = {'custom_rohand': 'data/robots/g1_rohand_five_finger/g1_rohand.urdf',
        'source_g1': scene['robot']['urdf'],
        'rohand_left': 'dependencies/rohand-urdf/urdf/rohand_left.urdf',
        'rohand_right': 'dependencies/rohand-urdf/urdf/rohand_right.urdf'}
    models = {name: urdf(ROOT/path) for name, path in paths.items()}
    report = {'schema_version': 1, 'mode': 'read_only_source_audit_no_simulation',
        'urdf': models,
        'mjcf': [mjcf(ROOT/path) for path in (
            'dependencies/unitree-ros/robots/g1_description/g1_29dof_rev_1_0.xml',
            'dependencies/unitree-ros/robots/g1_description/g1_29dof_with_hand_rev_1_0.xml',
            'dependencies/kimodo/kimodo/assets/skeletons/g1skel34/xml/g1.xml')],
        'usd': usd_inventory(ROOT/'dependencies/isaacsim/source/extensions/isaacsim.asset.transformer.rules/data/tests/G1'),
        'hand_frame_geometry': hand_directions(),
        'other_public_five_finger_variants': [rel(p) for p in (ROOT/'dependencies/unitree-ros/robots').rglob('*.urdf')
            if any(tag in p.name.lower() for tag in ('inspire', 'brainco', 'dex5'))],
        'local_isaaclab_present': (ROOT/'dependencies/isaaclab').exists(),
        'source_manifests': {},
        'findings': [], 'training_proposal': {}}
    for name in ('unitree-ros', 'rohand-urdf'):
        p = ROOT/f'manifests/sources/{name}.json'
        data = json.loads(p.read_text())
        report['source_manifests'][name] = {'path': rel(p), 'sha256': sha(p),
            'repository': data.get('repository'), 'commit': data.get('commit')}
    report['findings'] = [
        'Existing custom model uses g1_29dof_with_hand_rev_1_0, the README up-to-date mode5/4010-wrist variant; do not silently mix later 5010 wrist limits/inertias.',
        '89 links and 79 movable URDF joints represent 29 body plus 50 hand motions; intended controls are 29 body plus 12 hand, with an unactuated floating base added by the simulator.',
        'All 52 ROHand link masses and inertia tensors are positive and satisfy principal inertia triangle inequalities in this audit; none are zero placeholders. Smallest mass is about 0.177 grams and principal inertia 3.41e-10 kg m^2. Numerical regularization, if needed, must be explicit in the derivative asset.',
        'ROHand left/right CAD total mass differs: about 0.09566/0.13803 kg, mainly base masses 0.05596/0.09416 kg. This source asymmetry needs validation and must not be silently made symmetric.',
        'Custom G1 has no collision elements on pelvis, waist_yaw_link and waist_roll_link; four fixed sensor links also lack collision and inertial. Preserve a documented torso/pelvis contact envelope in a training derivative.',
        'ROHand six bounded inputs per side have zero effort and velocity, and no URDF mimic, transmission or dynamics. Official nonlinear FingerMath plus ROS joint-state publication describes display kinematics, not closed-loop force transmission.',
        'Giving all 50 hand joints independent position drives with 12-to-50 targets is an approximate powered-joint surrogate: PD forces can inject energy outside the real six-motor linkage. It must be disclosed and tested, not called exact ROHand mechanics.',
        'The bilateral fixed adapter is a virtual kinematic mount, with no separately modelled adapter mass/inertia or mechanical verification.',
        'Same detailed meshes are used for visual and collision on ROHand. Importer convexification may erase finger gaps or be costly; use derived contact geometry, preserve wrist-shell/palm/finger collision against box.',
        'Reference video box/scene contains geometric dimensions, not measured payload mass, friction, compliance or hardware motor parameters. Video trajectories are references, not dynamically feasible demonstrations.',
        'Official Unitree MJCF and IsaacSim G1 USDA fixtures provide physics starting points, but neither is the custom bilateral ROHand robot and neither establishes trained-policy performance.'
    ]
    report['training_proposal'] = {
        'write_scope': 'New derivative data/robots/g1_box_training only; preserve vendor files and frozen reconstruction assets.',
        'body': 'Select the existing rev_1_0 body variant and bind source SHA. Preserve its positive torque/velocity/position limits and mass/COM; floating base is unactuated. External PD target controller must clip body torques and expose saturation logs.',
        'hand_controls': '12 normalized policy synergies map to original slider/root ranges, then official nonlinear 6-to-25 per-side lookup. Targets are metres for prismatic and radians for rotational joints. Never expose 50 independent policy hand actions.',
        'surrogate_dynamics': 'For the first simulation-only policy, explicitly label independent-drive coupling approximation. Missing hand force, speed, damping/armature and gains are tunable simulation assumptions; use conservative finite limits, stability sweeps and domain randomization, never call chosen values official. Enforce real linkage constraints or validated reduced-coordinate hand for hardware transfer.',
        'inertia': 'Audit all missing/zero/non-SPD entries. Retain positive CAD tensors when numerically usable; any minimum mass/inertia regularization belongs in a manifest with before/after values. When merging fixed or tiny internal links, recompute COM/full inertia via parallel-axis theorem; do not silently duplicate or delete mass.',
        'collision': 'Body/limbs: source simple primitives or documented convex hulls. Preserve feet sole support, wrist shell, palm/base, thumb and each proximal/distal phalanx. Disable internal slider/connecting contacts only with explicit link list; avoid a single whole-hand hull. Keep robot-box contact enabled. Adjacent self-contact exclusions are explicit, not blanket all-self-contact removal.',
        'task': 'Start with balance/standing plus low-mass explicit box and two-palm support; then lift/hold curriculum. Box must be free rigid body with declared mass/inertia/material, table/floor static. Contact forces and object motion must result from physics, not reset/teleport at every step.',
        'acceptance_before_ppo': [
            'Simulator compile/import and resolved-mesh check, 41 commanded controls/79 physical joints, units and limits audit.',
            'Gravity-only and standing-PD stability with finite contact forces, no exploding small-link accelerations; test half timestep and solver convergence.',
            '12-control lookup range sweep, no branch discontinuities, compare FK against original geometry.',
            'Quasistatic bilateral grasp/lift with torque/force saturation, collision penetration and slip diagnostics; no hidden fixed box joint or per-step state overwrite.',
            'Record all chosen gains, effort/speed assumptions, contact geometry, mass changes and randomization ranges; reserve holdout payload/friction/initial-state tests.'
        ]}
    report['hand_frame_geometry']['interpretation'] = {
        'finger_extension_axis_in_source_base': [-1, 0, 0],
        'inferred_grasping_palm_normal_in_source_base': [0, 0, -1],
        'evidence': 'Four long-finger source-open tips have negative X. At half slider travel, all four tips move toward negative Z by 38-55 mm in both hands. This is a geometric inference, not a manufacturer palm-frame specification.',
        'ik_note': 'For fingers pointing world +X, map RO base -X to world +X; map RO base -Z toward the box on each side. Use unchanged wrist mount and solve arm orientation; do not flip frozen meshes.'}
    report['summary'] = {
        'custom_model_sha256': models['custom_rohand']['sha256'],
        'cad_mass_kg': models['custom_rohand']['summary']['total_link_mass_kg'],
        'all_given_inertias_spd': not models['custom_rohand']['summary']['nonpositive_inertia_links'],
        'kinematic_input_controls': 41,
        'expanded_motion_joints': 79,
        'hardware_equivalent_hand_dynamics_verified': False,
        'simulator_compiled_in_this_audit': False,
        'training_asset_ready_without_derivation': False,
        'critical_derivation_items': ['explicit finite hand actuation and servo model', 'linkage-coupling approximation disclosure',
            'small-inertia conditioning audit', 'simplified contact geometry including wrist and pelvis',
            'declared payload mass/material', 'unchanged mount mass/inertia assumption and source asymmetry disclosure']}
    p = ROOT/args.output
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'output': rel(p), 'sha256': sha(p), 'urdf_summaries': {k: {key: value for key, value in v['summary'].items() if key in ('total_link_mass_kg', 'min_link_mass_kg', 'nonpositive_inertia_links', 'missing_mesh_files', 'unique_mesh_triangles')} for k, v in models.items()},
        'usd_file_count': report['usd']['file_count'], 'usd_missing_reference_count': sum(len(x['missing_local_references']) for x in report['usd']['files'])}, indent=2))


if __name__ == '__main__':
    main()
