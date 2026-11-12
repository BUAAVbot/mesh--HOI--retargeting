"""Verify the retarget comparison ZIP and extract it into the local output folder."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--expected-sha256', required=True)
    args = parser.parse_args()
    archive_hash = hashlib.sha256(args.archive.read_bytes()).hexdigest()
    if archive_hash != args.expected_sha256:
        raise RuntimeError(f'Archive SHA mismatch: {archive_hash}')
    prefix = 'retarget_compare_20261028/'
    with ZipFile(args.archive) as packed:
        if packed.testzip():
            raise RuntimeError('ZIP CRC check failed')
        manifest = json.loads(packed.read(prefix + 'PACKAGE_MANIFEST.json'))
        assert manifest['archive_prefix'] == 'retarget_compare_20261028'
        assert len(manifest['files']) == 482
        expected = {prefix + row['path'] for row in manifest['files']}
        actual = set(packed.namelist()) - {prefix + 'PACKAGE_MANIFEST.json'}
        if actual != expected:
            raise RuntimeError('Manifest/file list mismatch')
        target = args.destination.resolve()
        for row in manifest['files']:
            entry = prefix + row['path']
            content = packed.read(entry)
            if len(content) != row['bytes'] or digest(content) != row['sha256']:
                raise RuntimeError(f'Payload mismatch: {entry}')
            output = (target / entry).resolve()
            if target not in output.parents:
                raise RuntimeError(f'Escaping path: {entry}')
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(content)
        manifest_output = target / prefix / 'PACKAGE_MANIFEST.json'
        manifest_output.write_bytes(packed.read(prefix + 'PACKAGE_MANIFEST.json'))
    print(json.dumps({'archive_sha256': archive_hash, 'payloads_verified': len(expected),
                      'crc_ok': True, 'output': str(target / prefix)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
