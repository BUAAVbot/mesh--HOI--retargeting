"""Run offline inventory verification on the chosen server and retain evidence locally."""
import json
import shlex
import subprocess
from prepare_dependencies import ROOT, write_json


def main():
    target = json.loads((ROOT / "configs/execution-targets.json").read_text(encoding="utf-8"))["server"]
    code = "\n".join([
        "import json, subprocess",
        "from pathlib import Path",
        "root = Path(" + repr(target["project_root"]) + ")",
        "python = str(root / '.venvs/tools/bin/python')",
        "inventory = subprocess.run([python, 'scripts/verify_downloads.py'], cwd=root, capture_output=True, text=True)",
        "preflight = subprocess.run([python, 'scripts/check_setup.py'], cwd=root, capture_output=True, text=True)",
        "reports = {name: json.loads((root / ('reports/' + name + '.json')).read_text()) for name in ['dependency_inventory', 'preflight']}",
        "print(json.dumps(dict(inventory_exit_code=inventory.returncode, preflight_exit_code=preflight.returncode, inventory_log=inventory.stdout + inventory.stderr, preflight_log=preflight.stdout + preflight.stderr, reports=reports)))",
    ])
    result = subprocess.run(["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", target["ssh_host"], "/usr/bin/python3.12 -c " + shlex.quote(code)], capture_output=True, check=True)
    evidence = json.loads(result.stdout)
    for name, report in evidence["reports"].items():
        write_json(ROOT / f"reports/server-{name.replace('_', '-')}.json", report)
    write_json(ROOT / "reports/server-preparation-check.json", {key: value for key, value in evidence.items() if key != "reports"})
    print(evidence["inventory_log"].strip())
    print("Algorithm preflight status: " + evidence["reports"]["preflight"]["status"])
    print("Server evidence retained in reports/server-*.json")
    return evidence["inventory_exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
