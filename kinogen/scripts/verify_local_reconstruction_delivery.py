"""Verify downloaded reconstruction files using server audit hashes; no ML runtime."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-variant", choices=("baseline", "contact"), default="contact")
    parser.add_argument("--preview-only", action="store_true", help="Keep large GLB assets on the audited server, as requested for local video preview")
    args = parser.parse_args()
    suffix = "-contact" if args.run_variant == "contact" else ""
    summary_path = ROOT / f"reports/user-reconstruction-v2-summary{suffix}.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("contract_audit_passed") is not True:
        raise RuntimeError("Server data contract did not pass")
    expected = summary["artifacts"] + summary["videos"] + [summary["review"]]
    expected += summary.get("provenance_files", [])
    seen, verified, missing, server_only = set(), [], [], []
    for record in expected:
        relative = record["path"].replace("outputs/previews/", "previews/", 1)
        parsed = PurePosixPath(relative)
        if parsed.is_absolute() or ".." in parsed.parts or ":" in relative or "\\" in relative:
            raise RuntimeError("Audit contains an unsafe relative path")
        if relative in seen:
            continue
        seen.add(relative)
        path = ROOT / relative
        if not path.is_file():
            if args.preview_only and relative.startswith("outputs/user_reconstruction_v2/") and parsed.suffix == ".glb":
                server_only.append(record)
                continue
            missing.append(relative)
            continue
        actual = sha(path)
        if actual != record["sha256"]:
            raise RuntimeError("Downloaded artifact hash mismatch: " + relative)
        verified.append({"path": relative, "bytes": path.stat().st_size, "sha256": actual})
    report = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": ("local_preview_delivery_sha256_passed" if args.preview_only else "local_delivery_sha256_passed") if not missing else "local_delivery_incomplete",
        "delivery_mode": "local_video_preview_server_glb" if args.preview_only else "all_artifacts_local",
        "run_variant": args.run_variant,
        "server_audit": {"path": str(summary_path.relative_to(ROOT)), "sha256": sha(summary_path)},
        "server_quality_status": summary["status"],
        "verified_files": verified,
        "missing_files": missing,
        "server_resident_glb_files": server_only,
        "server_project": "yz-ai-4090-01.bce:/data/KinoGen",
        "strict_metric_reconstruction_valid": False,
        "physical_hardware_validated": False,
        "local_ml_runtime_used": False,
        "note": "This verifies local transfer integrity. In preview-only mode, listed GLBs remain on the SHA-audited server and are not claimed to exist locally. Server image/geometry uncertainty and failed frame flags remain authoritative.",
    }
    destination = ROOT / f"reports/user-local-delivery-v2{suffix}.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "verified_files": len(verified), "server_resident_glb_files": len(server_only), "missing_files": missing}, indent=2))
    return 0 if not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
