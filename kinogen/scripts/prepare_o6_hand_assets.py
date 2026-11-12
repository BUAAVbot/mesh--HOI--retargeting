"""Read-only source audit and bounded extraction of the user's O6 hand package."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import xml.etree.ElementTree as ET
import zipfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--zip', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    a = p.parse_args()
    root = a.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    payload = {}
    with zipfile.ZipFile(a.zip) as z:
        for i in z.infolist():
            rel = PurePosixPath(i.filename)
            if (not rel.parts or rel.parts[0]!='06' or rel.is_absolute() or '..' in rel.parts or
                    '\\' in i.filename or ':' in i.filename or stat.S_ISLNK(i.external_attr>>16)):
                raise ValueError('Unsafe archive path')
            if i.is_dir():
                continue
            target = root.joinpath(*rel.parts[1:]).resolve()
            if root not in target.parents:
                raise ValueError('Archive path outside destination')
            data = z.read(i)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and digest(target.read_bytes()) != digest(data):
                raise ValueError('Refusing to overwrite different data: '+str(target))
            if not target.exists():
                with target.open('xb') as f:
                    f.write(data)
            payload[str(target.relative_to(root))] = {'bytes':len(data), 'sha256':digest(data)}
    hands = {}
    for side in ('LEFT','RIGHT'):
        folder = root/side
        urdf = next(folder.glob('*.urdf'))
        profile_path = next(folder.glob('*.json'))
        robot = ET.parse(urdf).getroot()
        profile = json.loads(profile_path.read_text(encoding='utf-8'))
        joints = robot.findall('joint')
        meshes = [e.get('filename') for e in robot.findall('.//mesh')]
        if any(not (urdf.parent/v).is_file() for v in meshes):
            raise ValueError('URDF refers to a missing mesh')
        passive = [j.get('name') for j in joints if j.find('mimic') is not None]
        if len(profile['command_names']) != 6 or len(joints) != 11 or len(passive) != 5:
            raise ValueError('Unexpected O6 articulation/command structure')
        expected_left = profile.get('quality',{}).get('transfer_provenance',{}).get('transferred_left_urdf_sha256')
        left_receipt_matches = (digest(urdf.read_bytes()) == expected_left) if expected_left else None
        hands[side.lower()] = {'urdf':str(urdf), 'urdf_sha256':digest(urdf.read_bytes()),
            'profile':str(profile_path), 'profile_sha256':digest(profile_path.read_bytes()),
            'links':len(robot.findall('link')), 'collision_elements':len(robot.findall('.//collision')),
            'inertial_elements':len(robot.findall('.//inertial')), 'physical_joints':len(joints),
            'independent_command_channels':profile['command_names'], 'passive_joints':passive,
            'calibration_scope':profile['calibration_scope'], 'profile_quality':profile['quality'],
            'per_joint_calibration_status':{k:v.get('calibration_status') for k,v in profile['joints'].items()},
            'baseline_command_u8':profile['baseline_command_u8'], 'curve_input_domain':profile['curve_input_domain'],
            'runtime_curve_policy':profile['runtime_curve_policy'],
            'left_published_urdf_sha_matches':left_receipt_matches,
            'force_calibration_included':False,
            'urdf_effort_note':'The URDF effort=100 field is not itself evidence of a measured motor torque limit.'}
    result = {'schema':'linkerhand_o6_source_audit.v1','status':'extracted_and_model_structure_verified',
        'archive':str(a.zip.resolve()), 'archive_sha256':digest(a.zip.read_bytes()),
        'archive_bytes':a.zip.stat().st_size, 'payload_files':len(payload), 'destination':str(root),
        'hands':hands,'members':payload,'source_executables_run':False,
        'interpretation':'Right partial measurement plus cross-finger transfer; left mirrored transfer, not a complete two-hand measured calibration.'}
    a.receipt.parent.mkdir(parents=True, exist_ok=True)
    a.receipt.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps({'status':result['status'], 'archive_sha256':result['archive_sha256'],
                      'payload_files':len(payload), 'hands':{k:{'links':v['links'],'collision_elements':v['collision_elements'],
                       'physical_joints':v['physical_joints'],'independent_commands':len(v['independent_command_channels']),
                       'left_published_urdf_sha_matches':v['left_published_urdf_sha_matches']} for k,v in hands.items()}}))


if __name__ == '__main__':
    main()
