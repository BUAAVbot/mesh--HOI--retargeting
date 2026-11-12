"""Read-only preflight; writes a local report, never imports model packages."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    config = json.loads((ROOT / "configs/reproduction.json").read_text(encoding="utf-8"))
    blockers = []
    targets = json.loads((ROOT / config["execution_targets"]).read_text(encoding="utf-8")) if config.get("execution_targets") else {}
    local_preview_only = targets.get("local", {}).get("role") == "video_preview_only"
    upstream = config["upstream"]
    for key in ("repository_url", "commit", "python_version", "pytorch_version", "cuda_version", "isaac_sim_version", "entrypoint"):
        if not upstream.get(key):
            blockers.append(f"Official upstream configuration missing: {key}")
    checkout = ROOT / upstream["checkout"]
    if not (checkout / ".git").exists():
        blockers.append("Official algorithm checkout missing: upstream/")
    else:
        for arguments, expected in ((["rev-parse", "HEAD"], upstream.get("commit")), (["remote", "get-url", "origin"], upstream.get("repository_url"))):
            result = subprocess.run(["git", "-C", str(checkout), *arguments], capture_output=True, text=True, timeout=15)
            if result.returncode or not expected or result.stdout.strip() != expected:
                blockers.append(f"Upstream identity could not be verified: {' '.join(arguments)}")
    gpu = {"nvidia_smi": shutil.which("nvidia-smi"), "output": None}
    if gpu["nvidia_smi"]:
        try:
            result = subprocess.run([gpu["nvidia_smi"], "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
            gpu["output"] = result.stdout.strip()
            if result.returncode or not gpu["output"]:
                blockers.append("NVIDIA GPU query failed")
        except (OSError, subprocess.TimeoutExpired) as exc:
            gpu["error"] = str(exc)
            blockers.append("NVIDIA GPU query failed")
    elif not local_preview_only:
        blockers.append("nvidia-smi unavailable; GPU readiness is unverified")
    if local_preview_only:
        runtime_path = ROOT / "reports/server-runtime.json"
        runtime = json.loads(runtime_path.read_text(encoding="utf-8")) if runtime_path.is_file() else {}
        gpu["execution_target"] = targets["server"]["ssh_host"]
        gpu["server_runtime_evidence"] = runtime
        if not runtime.get("cuda_available") or runtime.get("cuda_kernel_smoke_result") != 5.0 or runtime.get("environment_prefix") != targets["server"].get("gpu_baseline_environment"):
            blockers.append("Server GPU baseline has not passed validation")
    assets = {}
    for name, relative_path in config["required_assets"].items():
        path = ROOT / relative_path if relative_path else None
        exists = path is not None and path.exists()
        assets[name] = {"path": relative_path, "exists": exists}
        if not exists:
            blockers.append(f"Required asset unconfigured or missing: {name}")
    evidence = {"available": False}
    manifest_path = ROOT / "sources/manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        evidence = {"available": True, "files": manifest["files"], "failures": manifest["failures"]}
        if manifest["failures"]:
            blockers.append("Latest official-source download had failures")
        for entry in manifest["files"]:
            source_path = ROOT / "sources" / entry["file"]
            if not source_path.is_file() or hashlib.sha256(source_path.read_bytes()).hexdigest() != entry["sha256"]:
                blockers.append(f"Source snapshot hash mismatch or missing: {entry['file']}")
            if entry.get("code_coming_soon"):
                blockers.append("Official project snapshot says Code coming soon; check snapshot timestamp")
            if entry.get("repository_list_complete") is False:
                blockers.append("GitHub repository evidence is incomplete (pagination required)")
    else:
        blockers.append("Official source snapshots have not been downloaded")
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "blocked" if blockers else "prerequisites_present_runtime_not_validated",
        "algorithm_executed": False,
        "tooling_python": {"executable": sys.executable, "version": platform.python_version(), "platform": platform.platform()},
        "gpu": gpu,
        "packages_visible_in_tooling_environment": {name: importlib.util.find_spec(name) is not None for name in ("torch", "numpy")},
        "assets": assets, "source_evidence": evidence, "blockers": blockers,
    }
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    output = reports / "preflight.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Status: {report['status']}")
    for blocker in blockers:
        print(f"- {blocker}")
    print(f"Report: {output}")
    return 2 if blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
