"""Safely unpack and inspect the user-supplied O6 hand data; executes no payload."""
import argparse
import collections
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import xml.etree.ElementTree as ET
import zipfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Use a fresh output directory; never overwrite an earlier package')
    root = a.output.resolve()
    manifest, models, calibrations = [], {}, {}
    with zipfile.ZipFile(a.archive) as z:
        members = z.infolist()
        names = set()
        for item in members:
            rel = PurePosixPath(item.filename.replace('\\', '/'))
            if rel.is_absolute() or '..' in rel.parts or ':' in item.filename:
                raise ValueError('Unsafe archive path: '+item.filename)
            if stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError('Archive symlinks are unsupported')
            target = (root / Path(*rel.parts)).resolve()
            if not target.is_relative_to(root) or str(rel).casefold() in names:
                raise ValueError('Unsafe or duplicate archive target')
            names.add(str(rel).casefold())
        root.mkdir(parents=True)
        for item in members:
            if item.is_dir():
                continue
            raw = z.read(item)
            target = root.joinpath(*PurePosixPath(item.filename.replace('\\', '/')).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            manifest.append(dict(path=target.relative_to(root).as_posix(), bytes=len(raw), sha256=sha(raw)))
            if target.suffix == '.urdf':
                r = ET.fromstring(raw)
                side = 'left' if 'LEFT' in item.filename else 'right'
                joints = []
                for j in r.findall('joint'):
                    def attrs(tag):
                        e = j.find(tag)
                        return e.attrib if e is not None else None
                    joints.append(dict(name=j.get('name'), type=j.get('type'), origin=attrs('origin'),
                                       axis=attrs('axis'), limit=attrs('limit'), mimic=attrs('mimic')))
                meshes = sorted(set(m.get('filename') for m in r.findall('.//mesh')))
                for mesh in meshes:
                    resolved = (target.parent / mesh).resolve()
                    if not resolved.is_relative_to(root):
                        raise ValueError('Mesh reference escapes package')
                models[side] = dict(path=target.relative_to(root).as_posix(), sha256=sha(raw), joints=joints,
                    independent_revolute_joints=sum(j['type']=='revolute' and not j['mimic'] for j in joints),
                    mimic_joints=sum(bool(j['mimic']) for j in joints), collision_geometries=len(r.findall('.//collision')),
                    meshes=meshes)
            elif target.suffix == '.json':
                d = json.loads(raw)
                calibrations[d['side']] = dict(path=target.relative_to(root).as_posix(), sha256=sha(raw),
                    scope=d.get('calibration_scope'), quality=d.get('quality'),
                    command_names=d.get('command_names'), curve_input_domain=d.get('curve_input_domain'),
                    runtime_curve_policy=d.get('runtime_curve_policy'),
                    calibration_status_counts=dict(collections.Counter(j.get('calibration_status') for j in d['joints'].values())),
                    joint_metadata={name:{k:v for k,v in joint.items() if not isinstance(v,list) or len(v)<20}
                                    for name,joint in d['joints'].items()})
    for model in models.values():
        for mesh in model['meshes']:
            if not (root/model['path']).parent.joinpath(mesh).is_file():
                raise ValueError('Missing referenced mesh: '+mesh)
    report = dict(schema='fourth.o6_package_inspection.v1', archive=str(a.archive.resolve()),
                  archive_sha256=sha(a.archive.read_bytes()), payload_files=len(manifest),
                  models=models, calibrations=calibrations, files=manifest,
                  payload_executed=False, simulation_run=False,
                  limits=['Calibration is partial; left hand is mirrored from right, not independently measured.',
                          'URDF linear mimic is an endpoint fallback; JSON runtime coupling curves are a distinct model.',
                          'Thumb zero is already encoded in URDF origin and must not be applied twice.'])
    (root/'package_audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(archive_sha256=report['archive_sha256'], payload_files=len(manifest),
                         models={s:{k:m[k] for k in ['independent_revolute_joints','mimic_joints','collision_geometries']} for s,m in models.items()},
                         audit=str(root/'package_audit.json'))),flush=True)


if __name__ == '__main__':
    main()
