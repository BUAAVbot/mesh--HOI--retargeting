"""Collect small server-side component evidence; do not download model weights."""
import base64
import json
import shlex
import subprocess
import zlib
from prepare_dependencies import ROOT, now, write_json


def main():
    target = json.loads((ROOT / "configs/execution-targets.json").read_text(encoding="utf-8"))["server"]
    code = "\n".join([
        "import json, base64, zlib, sys",
        "from pathlib import Path",
        "root = Path(" + repr(target["project_root"]) + ")",
        "paths = list((root / 'reports').glob('*smoke*.json')) + list((root / 'reports').glob('user-*.json')) + list((root / 'outputs/previews').glob('kimodo_g1_*.json'))",
        "flow_path = root / 'reports/component-flow-latest.json'",
        "if flow_path.is_file():",
        "    paths.append(flow_path)",
        "    flow = json.loads(flow_path.read_text())",
        "    paths.extend(Path(value) for value in flow.get('reports', {}).values())",
        "    paths.append(Path(flow['run_directory']) / 'flow.json')",
        "environment_path = root / 'reports/server-component-environments.json'",
        "if environment_path.is_file(): paths.append(environment_path)",
        "for name in ('llama3-modelscope-server.json', 'kimodo-text-latest.json', 'kimodo-text-preflight.json', 'kimodo-text-environment.json'):",
        "    candidate = root / 'reports' / name",
        "    if candidate.is_file(): paths.append(candidate)",
        "result = {}",
        "for path in paths:",
        "    if path.resolve().is_relative_to(root.resolve()) and path.is_file() and not path.is_symlink() and path.stat().st_size < 2**20:",
        "        result[path.relative_to(root).as_posix()] = json.loads(path.read_text())",
        "sys.stdout.write(base64.encodebytes(zlib.compress(json.dumps(result).encode())).decode('ascii'))",
    ])
    result = subprocess.run(["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", target["ssh_host"], "/usr/bin/python3.12 -c " + shlex.quote(code)], capture_output=True, check=True, timeout=60)
    records = json.loads(zlib.decompress(base64.b64decode(b''.join(result.stdout.split()), validate=True)))
    destination = ROOT / "reports/server-components"
    for relative, report in records.items():
        path = (destination / relative).resolve()
        if not path.is_relative_to(destination.resolve()):
            raise ValueError("Unexpected report path")
        write_json(path, report)
    write_json(destination / "index.json", {"server": target["ssh_host"], "collected_at_utc": now(), "reports": list(records)})
    print(json.dumps({"collected_reports": len(records), "destination": str(destination)}))


if __name__ == "__main__":
    main()
