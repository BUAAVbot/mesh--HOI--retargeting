#!/usr/bin/env python3
"""Pull a project artifact over SSH using compressed, 76-column ASCII base64.

Some gateways stall on binary streams or very long single lines. This avoids
SFTP, prints no file payload locally, and checks size/SHA before writing.
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import tempfile
import zlib

ROOT = Path(__file__).resolve().parents[1]


def download(relative_path, local_path=None, timeout=90, replace=False, workers=3):
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or ".." in relative.parts or "\\" in relative_path or ":" in relative_path:
        raise ValueError("Supply a project-relative POSIX path")
    target = (ROOT / relative_path if local_path is None else Path(local_path)).resolve()
    if not target.is_relative_to(ROOT.resolve()):
        raise ValueError("Local destination must remain within this KinoGen project")
    settings = json.loads((ROOT / "configs/execution-targets.json").read_text(encoding="utf-8"))["server"]
    remote_root = PurePosixPath(settings["project_root"])
    if not remote_root.is_absolute():
        raise ValueError("Remote project root must be absolute")
    remote_path = str(remote_root / relative)
    ssh = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
           "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=2", settings["ssh_host"]]
    def execute(code):
        return subprocess.run(ssh + ["/usr/bin/python3.12 -c " + shlex.quote(code)],
                              capture_output=True, check=True, timeout=timeout).stdout.decode("utf-8")
    code = "\n".join([
        "from pathlib import Path", "import json,hashlib",
        "p=Path(" + repr(remote_path) + ")",
        "with p.open('rb') as f: checksum=hashlib.file_digest(f,'sha256').hexdigest()",
        "print(json.dumps({'bytes':p.stat().st_size,'sha256':checksum}),flush=True)",
    ])
    metadata = json.loads(execute(code))
    if not 0 <= metadata["bytes"] <= 16 * 1024 * 1024:
        raise ValueError("This artifact helper is limited to files of at most 16 MiB")
    if target.is_file() and target.stat().st_size == metadata["bytes"] and hashlib.sha256(target.read_bytes()).hexdigest() == metadata["sha256"]:
        return {"source": relative_path, "local_path": str(target), "already_verified": True, **metadata}
    # Incompressible video can expand past an SSH gateway window after base64.
    # 32 KiB raw guarantees less than 45 KiB ASCII even without compression.
    chunk_size = 32 * 1024
    bounds = [(start, min(chunk_size, metadata["bytes"] - start)) for start in range(0, metadata["bytes"], chunk_size)]
    def read_chunk(pair):
        offset, size = pair
        code = "\n".join([
            "from pathlib import Path", "import json,hashlib,zlib,base64,sys",
            "p=Path(" + repr(remote_path) + ")",
            "with p.open('rb') as f: f.seek(" + str(offset) + "); b=f.read(" + str(size) + ")",
            "print('KINOGEN_BEGIN:'+json.dumps({'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}),flush=True)",
            "sys.stdout.write(base64.encodebytes(zlib.compress(b)).decode('ascii'))",
            "print('KINOGEN_END',flush=True)",
        ])
        last_error = None
        for _ in range(2):
            try:
                lines = execute(code).splitlines()
                start = next(i for i, line in enumerate(lines) if line.startswith("KINOGEN_BEGIN:"))
                end = lines.index("KINOGEN_END", start + 1)
                chunk_meta = json.loads(lines[start].split(":", 1)[1])
                payload = zlib.decompress(base64.b64decode("".join(lines[start + 1:end]), validate=True))
                if len(payload) != size or len(payload) != chunk_meta["bytes"] or hashlib.sha256(payload).hexdigest() != chunk_meta["sha256"]:
                    raise RuntimeError("Artifact chunk size/SHA mismatch")
                return offset, payload
            except (subprocess.SubprocessError, ValueError, RuntimeError) as error:
                last_error = error
        raise RuntimeError(f"Failed artifact chunk at offset {offset}: {last_error}")
    pieces = {}
    with ThreadPoolExecutor(max_workers=min(max(int(workers), 1), 3)) as pool:
        futures = [pool.submit(read_chunk, pair) for pair in bounds]
        for future in as_completed(futures):
            offset, payload = future.result()
            pieces[offset] = payload
    payload = b"".join(pieces[start] for start, _ in bounds)
    if len(payload) != metadata["bytes"] or hashlib.sha256(payload).hexdigest() != metadata["sha256"]:
        raise RuntimeError("Complete artifact size/SHA mismatch or source changed during transfer")
    if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() != metadata["sha256"] and not replace:
        raise FileExistsError("A different local artifact exists; use --replace for regenerated project outputs")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".kinogen-ascii-", suffix=".part", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
    try:
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return {"source": relative_path, "local_path": str(target), **metadata}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Remote project-relative file")
    parser.add_argument("--local-path", type=Path)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--workers", type=int, default=3, choices=(1, 2, 3))
    args = parser.parse_args()
    print(json.dumps(download(args.path, args.local_path, args.timeout, args.replace, args.workers), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
