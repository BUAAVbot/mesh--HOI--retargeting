"""Atomically sync explicitly selected project files over the existing SSH profile."""
import argparse
import base64
import hashlib
import json
from pathlib import PurePosixPath
import shlex
import subprocess
from prepare_dependencies import ROOT, now, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    target = json.loads((ROOT / "configs/execution-targets.json").read_text(encoding="utf-8"))["server"]
    records = []
    for relative in args.paths:
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT.resolve()) or not path.is_file() or path.stat().st_size > 16 * 2**20:
            raise ValueError("Expected a small file inside the KinoGen project")
        relative = path.relative_to(ROOT).as_posix()
        wheel = relative.startswith("downloads/runtime-wheels/") and relative.endswith(".whl")
        if PurePosixPath(relative).parts[0] not in {"scripts", "configs", "reports", "docs", "manifests", "data"} and not (relative.startswith("requirements-") or relative.endswith(".md") or wheel):
            raise ValueError("Only runtime scripts, configurations and reports are supported")
        data = path.read_bytes()
        records.append({"path": relative, "sha256": hashlib.sha256(data).hexdigest(), "data": base64.b64encode(data).decode()})
    code = "\n".join([
        "import json, sys, base64, hashlib",
        "from pathlib import Path",
        "root = Path(" + repr(target["project_root"]) + ").resolve()",
        "verified = []",
        "for record in json.load(sys.stdin):",
        "    path = (root / record['path']).resolve()",
        "    if not path.is_relative_to(root): raise ValueError('Unsafe destination')",
        "    data = base64.b64decode(record['data'])",
        "    if hashlib.sha256(data).hexdigest() != record['sha256']: raise ValueError('Payload checksum mismatch')",
        "    path.parent.mkdir(parents=True, exist_ok=True)",
        "    temporary = path.with_name(path.name + '.kinogen-upload.tmp')",
        "    temporary.write_bytes(data)",
        "    temporary.replace(path)",
        "    verified.append(dict(path=record['path'], sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=len(data)))",
        "print(json.dumps(verified))",
    ])
    result = subprocess.run(["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", target["ssh_host"], "/usr/bin/python3.12 -c " + shlex.quote(code)], input=json.dumps(records).encode(), capture_output=True, check=True)
    verified = json.loads(result.stdout)
    stamp = now().replace(":", "").replace(".", "").replace("+", "")
    write_json(ROOT / f"manifests/deployment/small-sync-{stamp}.json", {"server": target["ssh_host"], "verified_at_utc": now(), "files": verified})
    print(json.dumps({"verified_files": len(verified), "paths": [item["path"] for item in verified]}))


if __name__ == "__main__":
    main()
