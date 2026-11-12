"""Prepare the supplied Vita archives using only the local standard library.

Pickle globals are replaced by inert metadata containers, never imported from
the input. No training, simulation, or numerical runtime runs locally.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import pickle
import struct
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(data):
    return hashlib.sha256(data).hexdigest()


class DType:
    def __init__(self, code, *args):
        self.code = code

    def __setstate__(self, state):
        self.state = state


class Array:
    def __setstate__(self, state):
        self.version, self.shape, self.dtype, self.fortran, self.data = state
        if len(self.shape) > 8 or math.prod(self.shape) > 20_000_000:
            raise ValueError('Unexpected array size')


def reconstruct(*args):
    return Array()


class RestrictedReader(pickle.Unpickler):
    def find_class(self, module, name):
        if module in ('numpy._core.multiarray', 'numpy.core.multiarray') and name == '_reconstruct':
            return reconstruct
        if module == 'numpy' and name == 'ndarray':
            return Array
        if module == 'numpy' and name == 'dtype':
            return DType
        raise ValueError('Unsupported pickle global: ' + module + '.' + name)

    def persistent_load(self, pid):
        raise ValueError('Persistent objects are disabled')


def npy_bytes(shape, descr, payload):
    header = repr({'descr': descr, 'fortran_order': False, 'shape': tuple(shape)})
    padding = (64 - ((10 + len(header) + 1) % 64)) % 64
    header = (header + ' ' * padding + '\n').encode('latin1')
    return b'\x93NUMPY\x01\x00' + struct.pack('<H', len(header)) + header + payload


def text_array(values):
    width = max(map(len, values))
    data = b''.join(x.encode('utf-32le').ljust(width * 4, b'\0') for x in values)
    return npy_bytes((len(values),), '<U' + str(width), data)


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--robot-archive', type=Path, required=True)
    p.add_argument('--motions-archive', type=Path, required=True)
    p.add_argument('--output', type=Path, default=ROOT / 'data/vita_demo_20261026')
    args = p.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(ROOT / 'data'):
        raise ValueError('Output must be inside project data')
    if (out / 'input_manifest.json').exists():
        raise FileExistsError('Completed inputs already exist; use a fresh output directory')
    out.mkdir(parents=True, exist_ok=True)
    manifest = {'schema': 'vita17.inputs.1', 'local_ml_runtime_used': False,
                'source_robot_zip': str(args.robot_archive.resolve()),
                'source_motion_tar': str(args.motions_archive.resolve()),
                'source_motion_tar_sha256': sha(args.motions_archive.read_bytes()),
                'assets': [], 'motions': [], 'checkpoint': None}
    with zipfile.ZipFile(args.robot_archive) as z:
        top = z.namelist()[0].split('/')[0] + '/'
        archive_manifest = json.loads(z.read(top + 'manifest.json'))
        for info in z.infolist():
            rel = info.filename[len(top):]
            if info.is_dir() or not (rel.startswith('robot/') or rel.startswith('checkpoints/new_ft15225/')):
                continue
            path = PurePosixPath(rel)
            if path.is_absolute() or '..' in path.parts or '\\' in rel:
                raise ValueError('Unsafe archive member')
            dest = out / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            if dest.exists():
                digest.update(dest.read_bytes())
                expected = next(x for x in archive_manifest['files'] if x['path'] == rel)
                if digest.hexdigest() != expected['sha256']:
                    raise ValueError('Existing preparation artifact differs from source')
            else:
                with z.open(info) as src, dest.open('xb') as dst:
                    while block := src.read(4 * 1024 * 1024):
                        digest.update(block)
                        dst.write(block)
            row = {'path': rel, 'bytes': dest.stat().st_size, 'sha256': digest.hexdigest()}
            if rel.endswith('.pt'):
                if row['sha256'] != archive_manifest['new_model']['checkpoint_sha256']:
                    raise ValueError('Checkpoint SHA differs from supplied manifest')
                manifest['checkpoint'] = row
            else:
                manifest['assets'].append(row)
    motions = out / 'motions'
    motions.mkdir(exist_ok=True)
    with tarfile.open(args.motions_archive) as tar:
        for info in tar.getmembers():
            if not info.isfile():
                continue
            if info.size > 16 * 1024 * 1024 or not info.name.endswith('.pkl'):
                raise ValueError('Unexpected input member')
            raw = tar.extractfile(info).read()
            pickle_digest = sha(raw)
            data = RestrictedReader(io.BytesIO(raw)).load()
            n = data['root_pos'].shape[0]
            if data['robot'] != 'vita02a' or data['root_rot_format'] != 'xyzw':
                raise ValueError('Unexpected robot or quaternion order')
            names = list(data['dof_names'].data)
            if len(names) != 52 or len(set(names)) != 52:
                raise ValueError('Expected 52 named joint states')
            name = PurePosixPath(info.name).stem + '.npz'
            (motions / name).unlink(missing_ok=True)  # An interrupted, uncommitted preparation only.
            with zipfile.ZipFile(motions / name, 'x', compression=zipfile.ZIP_DEFLATED) as npz:
                for field, source, shape in [('root_pos', 'root_pos', (n, 3)),
                                              ('root_rot_xyzw', 'root_rot', (n, 4)),
                                              ('dof_pos', 'dof_pos', (n, 52))]:
                    a = data[source]
                    if tuple(a.shape) != shape or a.dtype.code not in ('f4', 'f8'):
                        raise ValueError('Unsupported numeric array layout')
                    width = 4 if a.dtype.code == 'f4' else 8
                    if len(a.data) != math.prod(shape) * width:
                        raise ValueError('Unexpected numeric payload size')
                    if not all(math.isfinite(x[0]) for x in struct.iter_unpack('<f' if width == 4 else '<d', a.data)):
                        raise ValueError('Nonfinite source trajectory')
                    raw = a.data
                    if a.fortran:
                        if len(shape) != 2:
                            raise ValueError('Unsupported column-major dimensionality')
                        rows, cols = shape
                        raw = b''.join(a.data[(i + j * rows) * width:(i + j * rows + 1) * width]
                                       for i in range(rows) for j in range(cols))
                    npz.writestr(field + '.npy', npy_bytes(shape, '<' + a.dtype.code, raw))
                npz.writestr('dof_names.npy', text_array(names))
                npz.writestr('fps.npy', npy_bytes((), '<i8', struct.pack('<q', int(data['fps']))))
                npz.writestr('source_pickle_sha256.npy', text_array([pickle_digest]))
            manifest['motions'].append({'path': 'motions/' + name, 'source_member': info.name,
                                        'source_pickle_sha256': pickle_digest,
                                        'sha256': sha((motions / name).read_bytes()), 'frames': n,
                                        'fps': int(data['fps']), 'joint_count': len(names)})
    if len(manifest['motions']) != 17:
        raise ValueError('Expected all 17 source clips')
    (out / 'input_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    bundle = out / 'assets_and_motions.zip'
    with zipfile.ZipFile(bundle, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as z:
        for source in sorted(out.rglob('*')):
            if not source.is_file() or source == bundle or source.suffix == '.pt':
                continue
            z.write(source, source.relative_to(out).as_posix())
    print(json.dumps({'directory': str(out), 'bundle_bytes': bundle.stat().st_size,
                      'bundle_sha256': sha(bundle.read_bytes()), 'clips': len(manifest['motions']),
                      'frames': sum(x['frames'] for x in manifest['motions']),
                      'checkpoint': manifest['checkpoint']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
