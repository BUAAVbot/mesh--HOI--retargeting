"""Safely verify and unpack the user-provided fourth-stage visual handoff."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

EXPECTED_ARCHIVE_SHA = '5d3b56568a041c2934af1ec62a83db58b59388dc80f7bb126bdcc4f0fa3cda6c'
PREFIX = 'fourth_sim_handoff/'


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('target', type=Path)
    args = parser.parse_args()
    archive = args.archive.resolve()
    target = args.target.resolve()
    staging = target.with_name(target.name+'.staging')
    if target.exists() or staging.exists():
        raise FileExistsError('Refusing to overwrite target or staging directory')
    if digest(archive) != EXPECTED_ARCHIVE_SHA:
        raise ValueError('Archive SHA256 differs from inspected user file')
    staging.mkdir(parents=True)
    receipts = []
    with tarfile.open(archive, 'r:gz') as bundle:
        members = bundle.getmembers()
        if len(members) != 92 or any(not m.isfile() for m in members):
            raise ValueError('Unexpected tar membership or non-regular file')
        names = [m.name for m in members]
        if len(set(names)) != len(names):
            raise ValueError('Duplicate archive member')
        for m in members:
            if not m.name.startswith(PREFIX) or m.size < 0 or m.size > 150_000_000:
                raise ValueError('Unexpected member name or size')
            rel = PurePosixPath(m.name.removeprefix(PREFIX))
            if rel.is_absolute() or '..' in rel.parts or not rel.parts:
                raise ValueError('Unsafe archive path')
            out = (staging/Path(*rel.parts)).resolve()
            if not out.is_relative_to(staging.resolve()):
                raise ValueError('Archive member escapes staging')
            out.parent.mkdir(parents=True, exist_ok=True)
            h = hashlib.sha256()
            size = 0
            with bundle.extractfile(m) as src, out.open('xb') as dst:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
                    h.update(chunk)
                    size += len(chunk)
            if size != m.size:
                raise ValueError('Member size mismatch')
            receipts.append({'path': str(rel), 'bytes': size, 'sha256': h.hexdigest()})
    reported = json.loads((staging/'delivery_manifest.json').read_text(encoding='utf-8'))['files']
    byname = {r['path']: r for r in receipts}
    if len(reported) != 91 or len(byname) != 92:
        raise ValueError('Manifest/member count differs from inspected archive')
    for entry in reported:
        if byname.get(entry['path']) != entry:
            raise ValueError('Delivered member does not match bundle manifest: '+entry['path'])
    staging.rename(target)
    result = {'archive_sha256': EXPECTED_ARCHIVE_SHA, 'target': str(target),
              'files': len(receipts), 'payload_verified': len(reported),
              'unpacked_bytes': sum(r['bytes'] for r in receipts), 'scripts_executed': False}
    report = target.parent/'fourth_sim_handoff_extraction.json'
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
