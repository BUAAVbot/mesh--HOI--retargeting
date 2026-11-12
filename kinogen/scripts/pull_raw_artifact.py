#!/usr/bin/env python3
"""Pull small project artifacts using raw 128 KiB SSH/dd chunks and SHA checks.

Only Python's standard library is required locally. Metadata is obtained with
one SSH request per source file; each subsequent dd request carries raw bytes.
All chunks and the complete file are verified before an atomic local replace.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
CHUNK_BYTES = 128 * 1024
MAX_BYTES = 16 * 1024 * 1024


def checked_relative(value):
    relative = PurePosixPath(value)
    if (not value or relative.is_absolute() or relative == PurePosixPath('.')
            or '..' in relative.parts or '\\' in value or ':' in value
            or any(c in value for c in '\0\r\n')):
        raise ValueError('Supply a project-relative POSIX file path')
    return relative


def local_destination(relative_path, local_path=None):
    relative = checked_relative(relative_path)
    # Downloaded user previews belong to the local preview folder.
    parts = relative.parts
    mapped = PurePosixPath('previews', *parts[2:]) if parts[:2] == ('outputs', 'previews') else relative
    target = (ROOT / str(mapped) if local_path is None else Path(local_path)).resolve()
    if not target.is_relative_to(ROOT.resolve()) or target == ROOT.resolve():
        raise ValueError('Local destination must remain inside the KinoGen project')
    return target


def file_sha(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def download(relative_path, local_path=None, timeout=120, replace=False,
             workers=3, retries=2, progress=True):
    started = time.monotonic()
    relative = checked_relative(relative_path)
    target = local_destination(relative_path, local_path)
    settings = json.loads((ROOT / 'configs/execution-targets.json').read_text(encoding='utf-8'))['server']
    remote_root = PurePosixPath(settings['project_root'])
    if not remote_root.is_absolute() or '..' in remote_root.parts:
        raise ValueError('Remote project root must be an absolute POSIX path')
    host = settings['ssh_host']
    if not isinstance(host, str) or not host or host.startswith('-') or any(c in host for c in '\0\r\n'):
        raise ValueError('Invalid SSH host')
    remote_path = str(remote_root / relative)
    ssh = ['ssh', '-o', 'BatchMode=yes', host]
    timeout = float(timeout)
    if timeout <= 0 or int(retries) < 1:
        raise ValueError('Timeout and retry count must be positive')
    workers = min(max(int(workers), 1), 3)

    metadata_code = '\n'.join([
        'from pathlib import Path', 'import hashlib,json',
        'p=Path(' + repr(remote_path) + ')',
        'if not p.is_file(): raise ValueError("Source is not a file")',
        'size=p.stat().st_size',
        f'if not 0 <= size <= {MAX_BYTES}: raise ValueError("Artifact exceeds 16 MiB limit")',
        'checksum=hashlib.sha256(); chunks=[]; offset=0',
        "with p.open('rb') as stream:",
        ' while True:',
        f'  block=stream.read({CHUNK_BYTES})',
        '  if not block: break',
        "  checksum.update(block); chunks.append({'offset':offset,'bytes':len(block),'sha256':hashlib.sha256(block).hexdigest()}); offset+=len(block)",
        'if offset != size: raise RuntimeError("Source size changed during metadata read")',
        "print(json.dumps({'bytes':size,'sha256':checksum.hexdigest(),'chunk_bytes':" + str(CHUNK_BYTES) + ",'chunks':chunks}),flush=True)",
    ])
    metadata_started = time.monotonic()
    response = subprocess.run(ssh + ['/usr/bin/python3.12 -c ' + shlex.quote(metadata_code)],
                              capture_output=True, check=True, timeout=timeout)
    metadata = json.loads(response.stdout.decode('utf-8'))
    metadata_seconds = time.monotonic() - metadata_started
    size = metadata['bytes']
    if (not isinstance(size, int) or not 0 <= size <= MAX_BYTES
            or metadata['chunk_bytes'] != CHUNK_BYTES):
        raise RuntimeError('Invalid remote artifact size or chunk size')
    expected_count = (size + CHUNK_BYTES - 1) // CHUNK_BYTES
    chunks = metadata['chunks']
    if len(chunks) != expected_count:
        raise RuntimeError('Invalid chunk count')
    for index, chunk in enumerate(chunks):
        if chunk['offset'] != index * CHUNK_BYTES or chunk['bytes'] != min(CHUNK_BYTES, size - chunk['offset']):
            raise RuntimeError('Invalid chunk offsets or lengths')
    result = {'source': str(relative), 'local_path': str(target), 'bytes': size,
              'sha256': metadata['sha256'], 'chunk_bytes': CHUNK_BYTES,
              'chunk_count': expected_count, 'metadata_seconds': metadata_seconds}
    if target.is_file():
        if target.stat().st_size == size and file_sha(target) == metadata['sha256']:
            return {**result, 'already_verified': True, 'elapsed_seconds': time.monotonic() - started}
        if not replace:
            raise FileExistsError('A different local artifact exists; use --replace explicitly')

    def read_chunk(index):
        chunk = chunks[index]
        command = ('dd if=' + shlex.quote(remote_path)
                   + f' bs={CHUNK_BYTES} skip={index} count=1 status=none')
        error = None
        for attempt in range(1, int(retries) + 1):
            try:
                reply = subprocess.run(ssh + [command], capture_output=True,
                                       check=True, timeout=timeout)
                block = reply.stdout
                if len(block) != chunk['bytes'] or hashlib.sha256(block).hexdigest() != chunk['sha256']:
                    raise RuntimeError('Raw chunk length/SHA mismatch or source changed')
                return index, block
            except (subprocess.SubprocessError, RuntimeError) as problem:
                error = problem
                if progress:
                    print(f'{relative}: chunk {index + 1}/{expected_count}, attempt {attempt}/{retries} failed: {problem}',
                          file=sys.stderr, flush=True)
        raise RuntimeError(f'Failed raw artifact chunk at offset {chunk["offset"]}: {error}')

    received = {}
    transfer_started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(read_chunk, index) for index in range(expected_count)]
        try:
            for future in as_completed(futures):
                index, block = future.result()
                received[index] = block
                if progress:
                    total = sum(len(piece) for piece in received.values())
                    print(f'{relative}: {len(received)}/{expected_count} chunks, {total}/{size} raw bytes SHA verified',
                          file=sys.stderr, flush=True)
        except BaseException:
            # Avoid continuing an entire queue after one unrecoverable chunk.
            for future in futures:
                future.cancel()
            raise
    payload = b''.join(received[index] for index in range(expected_count))
    if len(payload) != size or hashlib.sha256(payload).hexdigest() != metadata['sha256']:
        raise RuntimeError('Complete file size/SHA mismatch or source changed')
    # Recheck replacement authorization after the network round trip.
    if target.is_file() and file_sha(target) != metadata['sha256'] and not replace:
        raise FileExistsError('A different local artifact appeared; use --replace explicitly')
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.kinogen-raw-', suffix='.part', delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
    try:
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return {**result, 'already_verified': False, 'chunks_verified': expected_count,
            'transfer_seconds': time.monotonic() - transfer_started,
            'elapsed_seconds': time.monotonic() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs='+', help='Project-relative remote files')
    parser.add_argument('--local-path', type=Path, help='Explicit destination for one source only')
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--workers', type=int, default=3, choices=(1, 2, 3))
    parser.add_argument('--retries', type=int, default=2)
    parser.add_argument('--replace', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()
    if args.local_path is not None and len(args.paths) != 1:
        parser.error('--local-path requires exactly one source')
    results = [download(path, args.local_path, args.timeout, args.replace,
                        args.workers, args.retries, not args.quiet) for path in args.paths]
    print(json.dumps(results[0] if len(results) == 1 else results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
