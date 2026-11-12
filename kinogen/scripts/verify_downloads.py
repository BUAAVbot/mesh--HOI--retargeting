"""Verify staged files without importing Torch or executing checkpoint pickles."""
import base64
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from prepare_dependencies import ROOT, REPOS, write_json, native_path


def digest(path, algorithm="sha256"):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def main():
    records = []
    for plan_path in (ROOT / "configs/downloads.json", ROOT / "configs/downloads-extra.json"):
        for item in json.loads(plan_path.read_text(encoding="utf-8")):
            path = ROOT / item["path"]
            record = {"component": item["component"], "path": item["path"], "status": "missing", "url": item["url"]}
            if path.is_file():
                checksum = digest(path)
                size_ok = item.get("bytes") is None or path.stat().st_size == item["bytes"]
                checksum_ok = item.get("sha256") is None or checksum == item["sha256"]
                gcs_hash = item.get("upstream_gcs_hash", "")
                md5_ok = True
                if "md5=" in gcs_hash:
                    md5_expected = gcs_hash.split("md5=", 1)[1].split(",", 1)[0]
                    md5_actual = base64.b64encode(bytes.fromhex(digest(path, "md5"))).decode()
                    md5_ok = md5_actual == md5_expected
                record.update({"status": "verified" if size_ok and checksum_ok and md5_ok else "invalid", "bytes": path.stat().st_size, "sha256": checksum, "official_sha256_checked": bool(item.get("sha256")), "official_gcs_md5_checked": "md5=" in gcs_hash})
            else:
                part = path.with_suffix(path.suffix + ".part")
                chunks = part.with_suffix(part.suffix + ".chunks")
                partial_bytes = part.stat().st_size if part.is_file() else 0
                segment_bytes = sum(p.stat().st_size for p in chunks.iterdir() if p.is_file()) if chunks.is_dir() else 0
                if partial_bytes or segment_bytes:
                    record.update({"status": "partial", "partial_bytes": partial_bytes, "segment_bytes": segment_bytes})
            records.append(record)
    fp_manifest = ROOT / "manifests/foundationpose-pytorch.json"
    if fp_manifest.is_file():
        entries = json.loads(fp_manifest.read_text(encoding="utf-8"))
        for item in entries["files"]:
            path = ROOT / item["path"]
            valid = path.is_file() and path.stat().st_size == item["bytes"] and digest(path) == item["sha256"]
            records.append({**item, "component": "FoundationPose original PyTorch", "status": "verified_local_hash" if valid else "invalid", "official_sha256_checked": False})
        for item in entries["failures"]:
            records.append({"component": "FoundationPose original PyTorch", "path": item["path"], "status": "missing", "error": item["error"]})
    sources = []
    for name, repo in REPOS.items():
        manifest_path = ROOT / "manifests/sources" / f"{name}.json"
        record = {"name": name, "repository": f"https://github.com/{repo}", "status": "missing_or_incomplete"}
        if manifest_path.is_file():
            metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
            destination = ROOT / metadata["source_directory"]
            errors = []
            if "files" in metadata:
                for item in metadata["files"]:
                    path = native_path(destination / item["path"])
                    if not path.is_file() or digest(path) != item["sha256"]:
                        errors.append(item["path"])
            if metadata.get("sha256") and ("files" not in metadata or metadata.get("files_indexed_from_verified_archive")):
                archive = ROOT / metadata["path"]
                if not archive.is_file() or digest(archive) != metadata["sha256"]:
                    errors.append("source_archive")
            record.update({"status": "verified_source_snapshot" if destination.is_dir() and not errors else "invalid", "commit": metadata["commit"], "source_directory": metadata["source_directory"], "omitted_entries": len(metadata.get("omitted", [])), "unhydrated_lfs_pointers": metadata.get("unhydrated_lfs_pointers", []), "lfs_hydrated_paths": metadata.get("lfs_hydrated_paths", 0), "errors": errors, "scope": metadata.get("scope", "Full source tarball; submodules and LFS content not automatically hydrated")})
        sources.append(record)
    missing = [item for item in records if item["status"] not in ("verified", "verified_local_hash")]
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "scope": "Download integrity only; independent runtime evidence: RUNTIME_STATUS.zh-CN.md", "algorithm_executed": False, "model_load_tested": False, "weights": records, "sources": sources, "complete_file_bytes": sum(item.get("bytes", 0) for item in records if item["status"].startswith("verified")), "public_weight_files_pending": len(missing), "account_requirements": "ACCESS_REQUIREMENTS.zh-CN.md"}
    write_json(ROOT / "reports/dependency_inventory.json", report)
    groups = defaultdict(list)
    for item in records:
        groups[item["component"]].append(item)
    lines = ["# KinoGen 依赖准备状态", "", f"核验时间（UTC）：{report['checked_at_utc']}", "", "本报告仅核对下载计划文件的大小与哈希，本校验脚本不执行推理。另行完成的模型推理及组件验收见 RUNTIME_STATUS.zh-CN.md；此处清单不含后续单独部署的 SAM3、MANO 等授权模型。", "", "## 公开权重", "", "| 组件 | 完整文件 / 计划文件 | 完整文件大小（MiB） |", "| --- | ---: | ---: |"]
    for name, entries in groups.items():
        ready = [item for item in entries if item["status"].startswith("verified")]
        lines.append(f"| {name} | {len(ready)} / {len(entries)} | {sum(item.get('bytes', 0) for item in ready)/2**20:.1f} |")
    lines += ["", "## 官方源码", "", "| 组件 | 状态 | 固定提交 |", "| --- | --- | --- |"]
    for item in sources:
        lines.append(f"| {item['name']} | {item['status']} | {item.get('commit', '待补齐')} |")
    lines += ["", "逐文件源码快照会另列省略的大媒体、子模块、符号链接或 LFS；完整范围见 `manifests/sources/`。厂商基础模型与论文中的机器人组装并不等价。", ""]
    for item in sources:
        if item.get("lfs_hydrated_paths") or item.get("unhydrated_lfs_pointers"):
            lines += [f"{item['name']}：已补齐 {item.get('lfs_hydrated_paths', 0)} 个 LFS 文件，未补齐 {len(item.get('unhydrated_lfs_pointers', []))} 个。", ""]
    lines += ["## 未完成传输", ""]
    lines += [f"- `{item['path']}`：{item['status']}" for item in missing] or ["公开权重计划中的文件已全部完成。"]
    lines += ["", "账号、人工批准、许可和托管视频服务见 [访问条件](ACCESS_REQUIREMENTS.zh-CN.md)。网络中断与授权缺失分别记录。", "", "全部逐文件校验结果见 `reports/dependency_inventory.json`。KinoGen 官方集成与版本选型仍待发布。", ""]
    (ROOT / "PREPARATION_STATUS.zh-CN.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Complete files: {sum(item['status'].startswith('verified') for item in records)}/{len(records)}; complete bytes: {report['complete_file_bytes']/2**20:.1f} MiB; pending public weight files: {len(missing)}")
    print("Report: PREPARATION_STATUS.zh-CN.md and reports/dependency_inventory.json")
    return 2 if missing or any(item["status"] != "verified_source_snapshot" or item.get("unhydrated_lfs_pointers") for item in sources) else 0


if __name__ == "__main__":
    raise SystemExit(main())
