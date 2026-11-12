"""Server-only, resumable download of the audited original Llama3 mirror."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
MODEL = "LLM-Research/Meta-Llama-3-8B-Instruct"
REVISION = "e9f7e7d3fa08b550ea228e38bb5501d35be92c0d"
DEST = ROOT / "weights/text-encoders/meta-llama/Meta-Llama-3-8B-Instruct"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=16, choices=range(1, 33))
    parser.add_argument("--chunk-mib", type=int, default=4, choices=(1, 2, 4, 8, 16))
    args = parser.parse_args()
    if platform.system() != "Linux":
        parser.error("Download on the server; the Windows machine only previews outputs")
    import requests
    audit = json.loads((ROOT / "reports/llama3-modelscope-check.json").read_text(encoding="utf-8"))
    if audit["revision"] != REVISION or audit["model_id"] != MODEL:
        raise ValueError("Unexpected model audit/revision")
    entries = [e for e in audit["filetree_response"]["Data"]["Files"]
               if e["Type"] == "blob" and "/" not in e["Path"] and e["Path"] != ".gitattributes"]
    shards = [e for e in entries if e["Path"].endswith(".safetensors")]
    if len(entries) != 14 or len(shards) != 4:
        raise ValueError("Expected 14 audited files and four original Transformers shards")
    for item in entries:
        if Path(item["Path"]).name != item["Path"] or len(item["Sha256"]) != 64:
            raise ValueError("Invalid audited file entry")
    DEST.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(DEST).free < 40 * 2**30:
        raise RuntimeError("Need at least 40 GiB free for resumable chunks and assembled model")
    local = threading.local()
    start_time = time.monotonic()
    total = sum(e["Size"] for e in entries)
    completed = 0
    receipt = {"source": audit["source"], "model_id": MODEL, "revision": REVISION,
               "destination": str(DEST), "model_loaded": False, "files": []}

    def session():
        if not hasattr(local, "session"):
            local.session = requests.Session()
            local.session.trust_env = False
        return local.session

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for entry in sorted(entries, key=lambda item: item["Path"].endswith(".safetensors")):
            name, size, checksum = entry["Path"], entry["Size"], entry["Sha256"]
            destination = DEST / name
            if destination.exists():
                if not destination.is_file() or destination.is_symlink() or destination.stat().st_size != size or sha(destination) != checksum:
                    raise FileExistsError(f"Preserving different existing file: {destination}")
                completed += size
                print(f"Verified existing {name}", flush=True)
            else:
                chunk_root = DEST / ".chunks" / (name + "." + checksum[:16])
                chunk_root.mkdir(parents=True, exist_ok=True)
                chunk_size = size if size < 16 * 2**20 else args.chunk_mib * 2**20
                bounds = [(offset, min(size - 1, offset + chunk_size - 1)) for offset in range(0, size, chunk_size)]

                def one(pair):
                    first, last = pair
                    path = chunk_root / f"{first}-{last}"
                    length = last - first + 1
                    if path.is_file() and path.stat().st_size == length:
                        return length
                    partial = path.with_name(path.name + ".part")
                    for attempt in range(6):
                        try:
                            with session().get("https://www.modelscope.cn/api/v1/models/" + MODEL + "/repo",
                                    params={"Revision": REVISION, "FilePath": name},
                                    headers={"Range": f"bytes={first}-{last}"}, stream=True,
                                    timeout=(15, 40)) as response:
                                response.raise_for_status()
                                full_small_file = size < 16 * 2**20 and first == 0 and last == size - 1 and response.status_code == 200
                                exact_range = response.status_code == 206 and response.headers.get("Content-Range") == f"bytes {first}-{last}/{size}"
                                if not (full_small_file or exact_range):
                                    raise ValueError(f"Unexpected HTTP range: status={response.status_code}, Content-Range={response.headers.get('Content-Range')}, file={name}, wanted={first}-{last}/{size}")
                                with partial.open("wb") as stream:
                                    for data in response.iter_content(256 * 1024):
                                        stream.write(data)
                            if partial.stat().st_size != length:
                                raise ValueError("Truncated download")
                            partial.replace(path)
                            return length
                        except (requests.RequestException, OSError, ValueError):
                            if attempt == 5:
                                raise
                            time.sleep(min(2 ** attempt, 10))

                futures = [pool.submit(one, pair) for pair in bounds]
                last_print = 0
                for future in as_completed(futures):
                    completed += future.result()
                    now = time.monotonic()
                    if now - last_print >= 25:
                        last_print = now
                        progress = {"status": "downloading", "file": name,
                                    "completed_bytes": completed, "total_bytes": total,
                                    "percent": round(completed * 100 / total, 2),
                                    "elapsed_seconds": round(now - start_time)}
                        write_json(ROOT / "reports/llama3-download-progress.json", progress)
                        print(json.dumps(progress), flush=True)
                partial = destination.with_name(destination.name + ".part")
                with partial.open("wb") as stream:
                    for first, last in bounds:
                        with (chunk_root / f"{first}-{last}").open("rb") as source:
                            shutil.copyfileobj(source, stream, 8 * 2**20)
                if partial.stat().st_size != size or sha(partial) != checksum:
                    raise ValueError(f"Full-file size/SHA256 mismatch: {name}")
                partial.replace(destination)
                # Remove only the exact chunks just used, after full SHA256 passed.
                for first, last in bounds:
                    (chunk_root / f"{first}-{last}").unlink()
                if not any(chunk_root.iterdir()):
                    chunk_root.rmdir()
                print(f"SHA256 verified {name}: {size} bytes", flush=True)
            receipt["files"].append({"name": name, "bytes": size, "sha256": checksum})
            write_json(ROOT / "reports/llama3-modelscope-server.json", receipt)
    receipt.update({"verified_at_utc": datetime.now(timezone.utc).isoformat(),
                    "status": "all_files_sha256_verified", "files_count": len(entries),
                    "bytes": total, "elapsed_seconds": round(time.monotonic() - start_time, 2)})
    write_json(ROOT / "reports/llama3-modelscope-server.json", receipt)
    write_json(ROOT / "reports/llama3-download-progress.json", {"status": "all_files_sha256_verified", "percent": 100, "total_bytes": total})
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
