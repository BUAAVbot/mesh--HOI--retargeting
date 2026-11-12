"""Stage pinned official sources and weights without installing or executing them."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import re
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
import requests

ROOT = Path(__file__).resolve().parents[1]
REPOS = {
    "sam3": "facebookresearch/sam3",
    "tapnet": "google-deepmind/tapnet",
    "video-depth-anything": "DepthAnything/Video-Depth-Anything",
    "kimodo": "nv-tlabs/kimodo",
    "wilor": "rolpotamias/WiLoR",
    "foundationpose": "NVlabs/FoundationPose",
    "dno-reference": "korrawe/Diffusion-Noise-Optimization",
    "isaacsim": "isaac-sim/IsaacSim",
    "unitree-ros": "unitreerobotics/unitree_ros",
    "rohand-urdf": "oymotion/rohand_urdf_ros1",
    "pytorch3d": "facebookresearch/pytorch3d",
    "nvdiffrast": "NVlabs/nvdiffrast",
    "chumpy": "mattloper/chumpy",
}


def now():
    return datetime.now(timezone.utc).isoformat()


def native_path(path):
    """Use the Windows extended path form for deep upstream source trees."""
    path = Path(path).absolute()
    if os.name == "nt" and not str(path).startswith("\\\\?\\"):
        return Path("\\\\?\\" + str(path))
    return path


def get_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": "KinoGen-dependency-preparation/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code != 403:
            raise
        # Public API responses can cache an expired anonymous rate-limit error.
        # Retry once with a fresh URL; do not use credentials or gated mirrors.
        separator = "&" if "?" in url else "?"
        request = urllib.request.Request(f"{url}{separator}kg_checked={int(time.time())}", headers={"User-Agent": "KinoGen-dependency-preparation/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(url, path, expected_sha=None, expected_size=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and (expected_size is None or path.stat().st_size == expected_size):
        digest = sha256(path)
        if expected_sha is None or digest == expected_sha:
            return {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": digest, "url": url}
    part = path.with_suffix(path.suffix + ".part")
    if expected_size is not None and expected_size > 32 * 1024 * 1024:
        parallel_ranges(url, part, expected_size)
    # Explicit byte ranges work with the official large-file CDNs and support
    # resuming without loading checkpoints into memory. Never send credentials.
    transfer_size = expected_size
    failures = 0
    while transfer_size is None or not part.exists() or part.stat().st_size < transfer_size:
        offset = part.stat().st_size if part.exists() else 0
        end = offset + 16 * 1024 * 1024 - 1
        if transfer_size is not None:
            end = min(end, transfer_size - 1)
        separator = "&" if "?" in url else "?"
        transfer_url = f"{url}{separator}download=true&kg_range={offset}-{end}"
        try:
            with requests.get(transfer_url, headers={"Range": f"bytes={offset}-{end}"}, stream=True, timeout=(20, 45)) as response:
                response.raise_for_status()
                if response.status_code == 206:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if not match or int(match[1]) != offset:
                        raise ValueError("Server returned an unexpected byte range")
                    total = int(match[3])
                    if transfer_size is not None and total != transfer_size:
                        raise ValueError("Remote size differs from official metadata")
                    transfer_size = total
                elif offset:
                    # Source archives commonly ignore Range. Restart only this
                    # task's incomplete .part file and keep the final target.
                    offset = 0
                with part.open("ab" if offset else "wb") as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        stream.write(chunk)
                if response.status_code == 200:
                    if transfer_size is None:
                        transfer_size = part.stat().st_size
                    break
            failures = 0
        except (requests.RequestException, OSError) as exc:
            failures += 1
            if failures >= 5:
                raise RuntimeError(f"Transfer failed after retries: {type(exc).__name__}") from exc
            time.sleep(min(failures * 2, 8))
    if transfer_size is not None and part.stat().st_size != transfer_size:
        raise ValueError(f"Incomplete transfer: {path.name}")
    if expected_size is not None and part.stat().st_size != expected_size:
        raise ValueError(f"Size mismatch: {path.name}")
    digest = sha256(part)
    if expected_sha is not None and digest != expected_sha:
        raise ValueError(f"SHA256 mismatch: {path.name}")
    part.replace(path)
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": digest, "url": url, "downloaded_at_utc": now()}


def parallel_ranges(url, part, total):
    """Resume verified-size byte ranges; final official SHA256 checks all bytes."""
    offset = part.stat().st_size if part.exists() else 0
    if offset >= total:
        return
    chunk_size = 1024 * 1024
    chunks = part.with_suffix(part.suffix + ".chunks")
    chunks.mkdir(exist_ok=True)
    ranges = [(start, min(start + chunk_size - 1, total - 1)) for start in range(offset, total, chunk_size)]

    def one(bounds):
        start, end = bounds
        destination = chunks / f"{start}-{end}"
        if destination.exists() and destination.stat().st_size == end - start + 1:
            return destination
        for attempt in range(5):
            try:
                separator = "&" if "?" in url else "?"
                with requests.get(f"{url}{separator}download=true&kg_range={start}-{end}", headers={"Range": f"bytes={start}-{end}"}, timeout=(20, 45)) as response:
                    response.raise_for_status()
                    if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{total}":
                        raise ValueError("Range response does not match requested bytes")
                    data = response.content
                    if len(data) != end - start + 1:
                        raise ValueError("Truncated range response")
                    destination.write_bytes(data)
                    return destination
            except (requests.RequestException, OSError, ValueError):
                if attempt == 4:
                    raise
                time.sleep(attempt + 1)

    workers = int(os.environ.get("KINOGEN_DOWNLOAD_WORKERS", "4"))
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, bounds): bounds for bounds in ranges}
        for future in concurrent.futures.as_completed(futures):
            future.result()
            completed += 1
            if completed % 128 == 0 or completed == len(ranges):
                print(f"TRANSFER {part.parent.name}/{part.name}: {completed}/{len(ranges)} segments", flush=True)
    with part.open("ab") as output:
        for start, end in ranges:
            segment = chunks / f"{start}-{end}"
            with segment.open("rb") as stream:
                shutil.copyfileobj(stream, output)
    # Remove only the exact segments just appended, keeping retries recoverable.
    for start, end in ranges:
        (chunks / f"{start}-{end}").unlink()
    if not any(chunks.iterdir()):
        chunks.rmdir()


def source(name, repo):
    lock_path = ROOT / "manifests" / "sources" / f"{name}.json"
    if lock_path.exists():
        record = json.loads(lock_path.read_text(encoding="utf-8"))
        if (ROOT / record["source_directory"]).is_dir():
            print(f"SOURCE cached: {name}", flush=True)
            return record
    metadata = get_json(f"https://api.github.com/repos/{repo}")
    commit = get_json(f"https://api.github.com/repos/{repo}/commits/{metadata['default_branch']}")["sha"]
    archive = ROOT / "downloads" / "sources" / f"{name}-{commit}.tar.gz"
    record = download(f"https://codeload.github.com/{repo}/tar.gz/{commit}", archive)
    destination = ROOT / "dependencies" / name
    unpack = ROOT / "downloads" / "unpack" / name
    unpack.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as stream:
        stream.extractall(unpack, filter="data")
    folders = list(unpack.iterdir())
    if len(folders) != 1 or not folders[0].is_dir():
        raise ValueError("Unexpected source archive layout")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    destination.parent.mkdir(exist_ok=True)
    shutil.move(str(folders[0]), str(destination))
    unpack.rmdir()
    record.update({"repository": f"https://github.com/{repo}", "commit": commit, "source_directory": destination.relative_to(ROOT).as_posix(), "license_metadata": metadata.get("license"), "note": "Current official source snapshot, not a KinoGen-pinned dependency; Git submodules and LFS objects are not automatically fetched."})
    write_json(lock_path, record)
    print(f"SOURCE ready: {name} @ {commit[:12]}", flush=True)
    return record


def hf_metadata(repo, kind="models"):
    metadata = get_json(f"https://huggingface.co/api/{kind}/{repo}?blobs=true")
    path = ROOT / "manifests" / "hub" / (repo.replace("/", "--") + ".json")
    write_json(path, metadata)
    print(json.dumps({"repo": repo, "revision": metadata.get("sha"), "gated": metadata.get("gated"), "license": metadata.get("cardData", {}).get("license"), "files": [{"name": f["rfilename"], "size": f.get("size"), "lfs": f.get("lfs")} for f in metadata.get("siblings", [])]}, ensure_ascii=False), flush=True)


def weights(plan_path="configs/downloads.json"):
    plan = json.loads((ROOT / plan_path).read_text(encoding="utf-8"))
    def one(item):
        name = item["name"]
        record = download(item["url"], ROOT / item["path"], item.get("sha256"), item.get("bytes"))
        record.update({"name": name, "component": item["component"], "role": item.get("role"), "verification": "upstream_sha256" if item.get("sha256") else "local_sha256_and_transfer_success"})
        write_json(ROOT / "manifests" / "weights" / f"{name}.json", record)
        print(f"WEIGHT ready: {name} ({record['bytes'] / 2**20:.1f} MiB)", flush=True)
    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(one, item): item["name"] for item in plan}
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                errors.append({"name": futures[future], "error": str(exc)})
                print(f"FAILED {futures[future]}: {exc}", flush=True)
    write_json(ROOT / "manifests" / (Path(plan_path).stem + "-failures.json"), errors)
    return bool(errors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["sources", "hub", "weights"])
    parser.add_argument("--repo")
    parser.add_argument("--kind", default="models")
    parser.add_argument("--plan", default="configs/downloads.json")
    parser.add_argument("--source-name", action="append")
    args = parser.parse_args()
    if args.mode == "sources":
        errors = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(source, name, repo): name for name, repo in REPOS.items() if not args.source_name or name in args.source_name}
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    errors.append({"name": futures[future], "error": str(exc)})
                    print(f"FAILED {futures[future]}: {exc}", flush=True)
        write_json(ROOT / "manifests" / "source_failures.json", errors)
        return bool(errors)
    if args.mode == "hub":
        hf_metadata(args.repo, args.kind)
        return 0
    return weights(args.plan)


if __name__ == "__main__":
    raise SystemExit(main())
