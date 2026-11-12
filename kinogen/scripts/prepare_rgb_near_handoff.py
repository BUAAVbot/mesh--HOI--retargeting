"""Safely extract and verify the user's RGB near-scene handoff ZIP."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import zipfile


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--zip', type=Path, required=True)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    a = p.parse_args()
    a.parent.mkdir(parents=True, exist_ok=True)
    root = a.parent.resolve()
    prefix = 'rgb_261026_near_handoff'
    with zipfile.ZipFile(a.zip) as z:
        members = z.infolist()
        if sum(i.file_size for i in members) > 2_000_000_000:
            raise ValueError('Unexpected expanded archive size')
        names = set()
        for item in members:
            rel = PurePosixPath(item.filename)
            if (rel.is_absolute() or '..' in rel.parts or '\\' in item.filename or
                    not rel.parts or rel.parts[0] != prefix or ':' in item.filename or
                    stat.S_ISLNK(item.external_attr >> 16) or item.filename in names):
                raise ValueError('Unsafe/duplicate archive member: '+item.filename)
            names.add(item.filename)
            dest = root.joinpath(*rel.parts).resolve()
            if root not in dest.parents:
                raise ValueError('Archive member escapes destination')
            if item.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            data = z.read(item)
            if dest.exists():
                if sha(dest) != hashlib.sha256(data).hexdigest():
                    raise ValueError('Refusing to overwrite different existing data: '+str(dest))
            else:
                with dest.open('xb') as f:
                    f.write(data)
        target = root/prefix
        checksum_file = target/'SHA256SUMS.txt'
        verified = []
        for line in checksum_file.read_text(encoding='utf-8').splitlines():
            m = re.fullmatch(r'([0-9a-f]{64})\s+\*?(.+)', line)
            if not m:
                raise ValueError('Invalid checksum record')
            rel = PurePosixPath(m[2])
            dest = target.joinpath(*rel.parts).resolve()
            if target not in dest.parents or sha(dest) != m[1]:
                raise ValueError('Payload checksum failed: '+m[2])
            verified.append(m[2])
        payload = {i.filename[len(prefix)+1:] for i in members if not i.is_dir()}
        if set(verified) != payload-{'SHA256SUMS.txt'}:
            raise ValueError('Checksum manifest does not cover all payload files')
    result = {'status':'source_extracted_and_checksums_verified', 'zip':str(a.zip.resolve()),
              'zip_sha256':sha(a.zip), 'zip_bytes':a.zip.stat().st_size,
              'expanded_bytes':sum(i.file_size for i in members),
              'destination':str(target), 'payload_files_verified':len(verified),
              'source_scripts_executed':False}
    a.receipt.parent.mkdir(parents=True, exist_ok=True)
    a.receipt.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
