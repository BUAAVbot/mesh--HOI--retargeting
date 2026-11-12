"""Package selected project artifacts; runtime mode includes user-provided licensed assets."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import tarfile
from prepare_dependencies import ROOT, native_path, write_json, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["base", "weights", "metadata", "runtime", "wheels"])
    args = parser.parse_args()
    paths = {}
    for name in ["README.zh-CN.md", "ACCESS_REQUIREMENTS.zh-CN.md", "PREPARATION_STATUS.zh-CN.md", "requirements-tools.lock.txt", ".gitignore"]:
        if (ROOT / name).is_file():
            paths[name] = {}
    for requirement in ROOT.glob("requirements-*.txt"):
        paths[requirement.name] = {}
    for folder in ["scripts", "configs", "manifests", "sources", "reports", "docs"]:
        for path in (ROOT / folder).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".tmp":
                paths[path.relative_to(ROOT).as_posix()] = {}
    # Retain the small source archives used by archive-based verification.
    for source in (ROOT / "manifests/sources").glob("*.json"):
        metadata = json.loads(source.read_text(encoding="utf-8"))
        if metadata.get("sha256") and ("files" not in metadata or metadata.get("files_indexed_from_verified_archive")):
            paths[metadata["path"]] = {"sha256": metadata["sha256"]}
    if args.mode == "base":
        for source in (ROOT / "manifests/sources").glob("*.json"):
            manifest = json.loads(source.read_text(encoding="utf-8"))
            directory = manifest["source_directory"]
            if "files" in manifest:
                for item in manifest["files"]:
                    paths[directory + "/" + item["path"]] = item
            else:
                base = native_path(ROOT / directory)
                for path in base.rglob("*"):
                    if path.is_file():
                        paths[directory + "/" + path.relative_to(base).as_posix()] = {}
    elif args.mode == "weights":
        report = json.loads((ROOT / "reports/dependency_inventory.json").read_text(encoding="utf-8"))
        if report["public_weight_files_pending"]:
            raise RuntimeError("Finish and verify public weights before packaging")
        for item in report["weights"]:
            if not item["status"].startswith("verified"):
                raise RuntimeError("Weight integrity verification has not passed")
            paths[item["path"]] = item
    elif args.mode == "runtime":
        for name in ("MANO_RIGHT.pkl", "MANO_LEFT.pkl", "LICENSE.txt"):
            path = ROOT / "weights/wilor/mano_data" / name
            if path.is_file():
                paths[path.relative_to(ROOT).as_posix()] = {}
        for path in (ROOT / "data/smoke_inputs").rglob("*"):
            if path.is_file():
                paths[path.relative_to(ROOT).as_posix()] = {}
    elif args.mode == "wheels":
        paths = {"manifests/runtime-wheels.json": {}}
        for item in json.loads((ROOT / "manifests/runtime-wheels.json").read_text(encoding="utf-8"))["files"]:
            paths[item["path"]] = item
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    receipt_path = ROOT / f"manifests/deployment/{args.mode}-{stamp}.json"
    archive = ROOT / f"downloads/deployment/{args.mode}-{stamp}.tar"
    archive.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with tarfile.open(archive, "w", format=tarfile.PAX_FORMAT) as output:
        for relative, metadata in sorted(paths.items()):
            normal = (ROOT / relative).resolve()
            if not normal.is_relative_to(ROOT.resolve()):
                raise ValueError("Unsafe deployment path")
            path = native_path(normal)
            info = output.gettarinfo(str(path), arcname=relative)
            mode = metadata.get("mode")
            info.mode = (int(mode, 8) & 0o777) if mode else (0o755 if path.suffix == ".sh" else 0o644)
            with path.open("rb") as stream:
                output.addfile(info, stream)
            records.append({"path": relative, "bytes": path.stat().st_size, "sha256": metadata.get("sha256") or sha256(path)})
        receipt = {"kind": args.mode, "created_at_utc": datetime.now(timezone.utc).isoformat(), "files": records}
        write_json(receipt_path, receipt)
        output.add(receipt_path, arcname=receipt_path.relative_to(ROOT).as_posix())
    description = {"archive": archive.relative_to(ROOT).as_posix(), "receipt": receipt_path.relative_to(ROOT).as_posix(), "bytes": archive.stat().st_size, "sha256": sha256(archive), "kind": args.mode}
    write_json(ROOT / f"manifests/deployment/latest-{args.mode}.json", description)
    print(json.dumps(description), flush=True)


if __name__ == "__main__":
    main()
