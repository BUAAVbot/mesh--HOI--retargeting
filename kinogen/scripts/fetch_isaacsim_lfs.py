"""Hydrate public upstream LFS assets using commit-pinned paths and SHA256."""
import concurrent.futures
import hashlib
import json
import re
import shutil
import time
from urllib.parse import quote
import requests
from prepare_dependencies import ROOT, native_path, sha256, write_json, now


def main():
    path = ROOT / "manifests/sources/isaacsim.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    plan = json.loads((ROOT / "manifests/isaacsim-lfs-assets.json").read_text(encoding="utf-8"))["files"]
    unique = {}
    for item in plan:
        unique.setdefault(item["sha256"], item)
    print(f"Public Isaac Sim LFS: {len(plan)} paths, {len(unique)} distinct objects", flush=True)

    def one(item):
        cache = ROOT / "downloads/lfs" / item["sha256"]
        if cache.is_file() and cache.stat().st_size == item["bytes"] and sha256(cache) == item["sha256"]:
            return item["sha256"]
        url = f"https://media.githubusercontent.com/media/isaac-sim/IsaacSim/{manifest['commit']}/" + quote(item["path"], safe="/")
        for attempt in range(3):
            try:
                response = requests.get(url, timeout=(15, 45))
                response.raise_for_status()
                data = response.content
                if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
                    raise ValueError("Official LFS object identity mismatch")
                cache.parent.mkdir(parents=True, exist_ok=True)
                temporary = cache.with_suffix(".part")
                temporary.write_bytes(data)
                temporary.replace(cache)
                return item["sha256"]
            except (requests.RequestException, OSError, ValueError):
                if attempt == 2:
                    raise
                time.sleep(2)

    ready, failures = set(), []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(one, item): item for item in unique.values()}
        for future in concurrent.futures.as_completed(futures):
            try:
                ready.add(future.result())
                if len(ready) % 100 == 0:
                    print(f"Verified LFS objects: {len(ready)}/{len(unique)}", flush=True)
            except Exception as exc:
                item = futures[future]
                failures.append({"path": item["path"], "sha256": item["sha256"], "error": type(exc).__name__})
    by_path = {item["path"]: item for item in plan}
    hydrated, pending = [], []
    for record in manifest["files"]:
        item = by_path.get(record["path"])
        if not item:
            continue
        if item["sha256"] not in ready:
            pending.append(record["path"])
            continue
        target = native_path(ROOT / manifest["source_directory"] / record["path"])
        if not target.is_file() or target.stat().st_size != item["bytes"] or sha256(target) != item["sha256"]:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".lfs-tmp")
            shutil.copyfile(ROOT / "downloads/lfs" / item["sha256"], temporary)
            temporary.replace(target)
        pointer = record.pop("git_lfs_pointer_not_hydrated", None)
        if pointer:
            record["git_lfs_pointer"] = pointer
            record["pointer_sha256"] = record["sha256"]
        record.update({"git_lfs_hydrated": True, "sha256": item["sha256"], "bytes": item["bytes"]})
        hydrated.append(record["path"])
    manifest["unhydrated_lfs_pointers"] = pending
    manifest["lfs_hydrated_paths"] = len(hydrated)
    manifest["lfs_checked_at_utc"] = now()
    write_json(path, manifest)
    write_json(ROOT / "manifests/isaacsim-lfs-downloads.json", {"hydrated_paths": hydrated, "pending_paths": pending, "failures": failures})
    print(f"Isaac Sim LFS ready: {len(hydrated)} paths; pending: {len(pending)}", flush=True)
    return bool(pending)


if __name__ == "__main__":
    raise SystemExit(main())
