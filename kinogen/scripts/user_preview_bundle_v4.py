"""Pack or verify a bounded preview delivery, without loading an ML runtime."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
LIMIT = 32 * 1024 * 1024


def digest(data):
    return hashlib.sha256(data).hexdigest()


def destination(relative):
    parts = PurePosixPath(relative)
    if parts.is_absolute() or ".." in parts.parts or ":" in relative or "\\" in relative:
        raise ValueError("Unsafe delivery path")
    if not (relative.startswith("outputs/user_reconstruction_v4/") or relative.startswith("outputs/previews/user_v4_")
            or relative.startswith("previews/user_v4_") or relative.startswith("reports/user-")):
        raise ValueError("Delivery is restricted to named reconstruction outputs and reports")
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError("Delivery path leaves the project")
    return path


def pack(args):
    records, entries, total = [], {}, 0
    for name in args.paths:
        path = destination(name)
        data = path.read_bytes()
        total += len(data)
        if total > LIMIT:
            raise ValueError("Preview bundle exceeds 32 MiB uncompressed")
        local = name.replace("outputs/previews/", "previews/", 1)
        destination(local)
        member = "files/" + local
        if member in entries:
            raise ValueError("Duplicate delivery member")
        entries[member] = data
        records.append({"server_path": name, "local_path": local, "member": member, "bytes": len(data), "sha256": digest(data)})
    server_only = []
    if args.scene_report:
        scene = json.loads(destination(args.scene_report).read_text(encoding="utf-8"))
        for record in scene["outputs"]:
            if record["path"].endswith(".glb"):
                path = destination(record["path"])
                if digest(path.read_bytes()) != record["sha256"]:
                    raise ValueError("Server scene file differs from audited hash")
                server_only.append(record)
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "mode": "local_video_preview_server_glb",
                "server_project": "yz-ai-4090-01.bce:/data/KinoGen", "files": records,
                "server_resident_glb_files": server_only,
                "strict_metric_reconstruction_valid": False, "physical_hardware_validated": False}
    archive = destination(args.archive)
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=archive.parent, prefix=".preview-bundle-", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as zipped:
            zipped.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
            for name, data in entries.items():
                zipped.writestr(name, data)
        temporary.replace(archive)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({"archive": args.archive, "bytes": archive.stat().st_size, "sha256": digest(archive.read_bytes()),
                      "files": len(records), "server_resident_glbs": len(server_only)}, indent=2))


def unpack(args):
    archive = destination(args.archive)
    verified = []
    with zipfile.ZipFile(archive) as zipped:
        infos = zipped.infolist()
        if len(infos) > 256 or sum(info.file_size for info in infos) > LIMIT + 1024 * 1024:
            raise ValueError("Delivery archive exceeds bounds")
        if len({info.filename for info in infos}) != len(infos):
            raise ValueError("Duplicate ZIP entries")
        manifest = json.loads(zipped.read("manifest.json"))
        members = {"manifest.json"}
        pending = []
        for record in manifest["files"]:
            path = destination(record["local_path"])
            if record["member"] != "files/" + record["local_path"]:
                raise ValueError("Manifest/member mismatch")
            data = zipped.read(record["member"])
            if len(data) != record["bytes"] or digest(data) != record["sha256"]:
                raise ValueError("Delivery member size/SHA mismatch")
            if path.exists() and digest(path.read_bytes()) != record["sha256"] and not args.replace:
                raise FileExistsError("Different local artifact exists; use --replace for an authorized new result")
            pending.append((path, data, record))
            members.add(record["member"])
        if members != {info.filename for info in infos}:
            raise ValueError("Archive contains unlisted entries")
        for path, data, record in pending:
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists() or digest(path.read_bytes()) != record["sha256"]:
                with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".preview-delivery-", suffix=".tmp", delete=False) as stream:
                    stream.write(data)
                    temporary = Path(stream.name)
                try:
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
            if digest(path.read_bytes()) != record["sha256"]:
                raise RuntimeError("Local artifact verification failed")
            verified.append(record)
    for record in manifest.get("server_resident_glb_files", []):
        destination(record["path"])
        if not record["path"].endswith(".glb"):
            raise ValueError("Unexpected server-resident artifact type")
    receipt = {"verified_at_utc": datetime.now(timezone.utc).isoformat(), "status": "local_preview_delivery_sha256_passed",
               "archive": {"path": args.archive, "sha256": digest(archive.read_bytes())}, "verified_files": verified,
               "server_project": manifest["server_project"], "server_resident_glb_files": manifest.get("server_resident_glb_files", []),
               "strict_metric_reconstruction_valid": False, "physical_hardware_validated": False,
               "local_ml_runtime_used": False, "note": "File integrity only; server quality failures and uncertainty flags remain authoritative. GLBs listed as server-resident are not claimed to have been downloaded."}
    report = ROOT / "reports/user-v4-local-delivery.json"
    report.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "verified_files": len(verified), "server_resident_glbs": len(receipt["server_resident_glb_files"])}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("pack", "unpack"))
    parser.add_argument("--archive", default="outputs/user_reconstruction_v4/preview_delivery.zip")
    parser.add_argument("--scene-report")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("paths", nargs="*")
    args = parser.parse_intermixed_args()
    pack(args) if args.mode == "pack" else unpack(args)


if __name__ == "__main__":
    main()
