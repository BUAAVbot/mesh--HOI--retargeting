"""Recover official source files individually, checking Git blob identities.

This avoids restarting large tarballs on unstable connections. Large media and
Git LFS objects are recorded separately, never mistaken for installed assets.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
import time
import requests
from prepare_dependencies import ROOT, REPOS, get_json, write_json, now


def fetch_source(name):
    repo = REPOS[name]
    existing = ROOT / "manifests/sources" / f"{name}.json"
    if existing.exists():
        print(f"SOURCE already staged: {name}", flush=True)
        return
    candidates = list((ROOT / "downloads/sources").glob(name + "-*.tar.gz*"))
    commits = [re.search(r"-([a-f0-9]{40})\.tar\.gz", p.name).group(1) for p in candidates if re.search(r"-([a-f0-9]{40})\.tar\.gz", p.name)]
    commit = commits[0] if commits else get_json(f"https://api.github.com/repos/{repo}/commits/HEAD")["sha"]
    tree_path = ROOT / "manifests/trees" / f"{name}-{commit}.json"
    if tree_path.exists():
        tree = json.loads(tree_path.read_text(encoding="utf-8"))
    else:
        tree = get_json(f"https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1")
        write_json(tree_path, tree)
    if tree.get("truncated"):
        raise ValueError("GitHub tree response is truncated")
    destination = ROOT / "dependencies" / name
    selected, omitted = [], []
    for entry in tree["tree"]:
        path = entry["path"]
        if entry["type"] == "tree":
            continue
        if name == "unitree-ros" and "/" in path and not path.startswith("robots/g1_description/"):
            omitted.append({**entry, "omission_reason": "outside_g1_task_scope"})
            continue
        # Code/config/license/build files and small robot geometry are included.
        # Optional large demos, docs media and checkpoints are separately logged.
        if entry["type"] != "blob" or entry.get("mode") == "120000" or entry.get("size", 0) > 16 * 2**20 or Path(path).suffix.lower() in {".mp4", ".mov", ".gif", ".jpg", ".jpeg", ".png", ".webp", ".pdf"}:
            omitted.append(entry)
        else:
            selected.append(entry)
    records, failures, pointers = [], [], []
    selected.sort(key=lambda item: (Path(item["path"]).suffix.lower() in {".stl", ".dae", ".obj", ".npy", ".npz"}, item["path"]))
    print(f"SOURCE fetching {name}: {len(selected)} files, {sum(item.get('size', 0) for item in selected)/2**20:.1f} MiB", flush=True)

    def one(entry):
        path = destination / entry["path"]
        expected = entry["sha"]
        def valid(data):
            return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() == expected
        if path.is_file() and valid(path.read_bytes()):
            data = path.read_bytes()
        else:
            url = f"https://raw.githubusercontent.com/{repo}/{commit}/{entry['path']}"
            for attempt in range(3):
                try:
                    response = requests.get(url, timeout=(15, 30))
                    response.raise_for_status()
                    data = response.content
                    if not valid(data):
                        raise ValueError("Git blob hash mismatch")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
                    break
                except (requests.RequestException, ValueError):
                    if attempt == 2:
                        raise
                    time.sleep(2)
        record = {"path": entry["path"], "git_blob_sha1": expected, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        if data.startswith(b"version https://git-lfs.github.com/spec/v1"):
            record["git_lfs_pointer_not_hydrated"] = data.decode()
        return record

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(one, entry): entry["path"] for entry in selected}
        for future in concurrent.futures.as_completed(futures):
            try:
                record = future.result()
                records.append(record)
                if "git_lfs_pointer_not_hydrated" in record:
                    pointers.append(record["path"])
                if len(records) % 100 == 0:
                    print(f"SOURCE progress {name}: {len(records)}/{len(selected)}", flush=True)
            except Exception as exc:
                failures.append({"path": futures[future], "error": type(exc).__name__})
    scope = "Official source/build/config files; images, videos, PDFs, symlinks, submodules and files over 16 MiB are separately omitted"
    if name == "unitree-ros":
        scope += "; Unitree assets scoped to robots/g1_description and repository root metadata"
    report = {"repository": f"https://github.com/{repo}", "commit": commit, "checked_at_utc": now(), "source_directory": destination.relative_to(ROOT).as_posix(), "scope": scope, "files": sorted(records, key=lambda item: item["path"]), "omitted": omitted, "unhydrated_lfs_pointers": pointers, "failures": failures}
    target = existing if not failures else ROOT / "manifests/source-progress" / f"{name}.json"
    write_json(target, report)
    print(f"SOURCE {name}: {len(records)} verified files, {len(omitted)} optional/large/gitlink entries omitted, {len(failures)} failures", flush=True)
    if failures:
        raise RuntimeError(f"Incomplete source files: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="+", choices=REPOS)
    args = parser.parse_args()
    failed = False
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(fetch_source, name): name for name in args.names}
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"FAILED SOURCE {futures[future]}: {type(exc).__name__}: {str(exc)[:100]}", flush=True)
                failed = True
    return failed


if __name__ == "__main__":
    raise SystemExit(main())
