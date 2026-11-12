"""Download the user-selected SAM 3 mirror and verify every published SHA256."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
import requests

ROOT = Path(__file__).resolve().parents[1]
REVISION = "96f3e1b404ba14f2cfac60ee6ae87c269a7b7923"
API = "https://www.modelscope.cn/api/v1/models/facebook/sam3"
DESTINATION = ROOT / "weights/sam3"
SAM3_SHA = "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def fetch_file(entry):
    name, size = entry["Path"], entry["Size"]
    destination = DESTINATION / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = entry["Sha256"]
    if destination.exists():
        if destination.stat().st_size == size and sha(destination) == expected:
            print(f"Verified existing {name}", flush=True)
            return
        raise FileExistsError(f"Different existing file preserved: {destination}")
    temporary = destination.with_name(destination.name + ".part")
    params = {"Revision": REVISION, "FilePath": name}
    if size < 16 * 2**20:
        r = requests.get(API + "/repo", params=params, timeout=(20, 60))
        r.raise_for_status()
        temporary.write_bytes(r.content)
    else:
        chunks = destination.with_name(destination.name + ".chunks")
        chunks.mkdir(exist_ok=True)
        chunk_size = 2**20
        bounds = [(start, min(size - 1, start + chunk_size - 1))
                  for start in range(0, size, chunk_size)]

        def one(pair):
            start, end = pair
            path = chunks / f"{start}-{end}"
            length = end - start + 1
            if path.exists() and path.stat().st_size == length:
                return
            last_error = None
            for attempt in range(4):
                try:
                    with requests.get(API + "/repo", params=params,
                            headers={"Range": f"bytes={start}-{end}"},
                            stream=True, timeout=(20, 30)) as response:
                        response.raise_for_status()
                        if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                            raise ValueError("ModelScope returned an unexpected byte range")
                        partial = path.with_name(path.name + ".part")
                        with partial.open("wb") as f:
                            for data in response.iter_content(256 * 1024):
                                f.write(data)
                        if partial.stat().st_size != length:
                            raise ValueError("Truncated ModelScope range")
                        partial.replace(path)
                    return
                except (requests.RequestException, OSError, ValueError) as error:
                    last_error = error
                    time.sleep(attempt + 1)
            raise RuntimeError(f"Failed range {start}-{end}: {last_error}")

        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(one, pair) for pair in bounds]
            for count, future in enumerate(as_completed(futures), 1):
                future.result()
                if count % 128 == 0 or count == len(bounds):
                    print(f"{name}: {count}/{len(bounds)} chunks ({100*count/len(bounds):.0f}%), {time.monotonic()-started:.0f}s", flush=True)
        with temporary.open("wb") as dst:
            for start, end in bounds:
                with (chunks / f"{start}-{end}").open("rb") as src:
                    shutil.copyfileobj(src, dst, 8 * 2**20)
    actual = sha(temporary)
    if temporary.stat().st_size != size or actual != expected:
        raise ValueError(f"Size/SHA256 mismatch for {name}")
    temporary.replace(destination)
    if size >= 16 * 2**20:
        for start, end in bounds:
            (chunks / f"{start}-{end}").unlink()
        if not any(chunks.iterdir()):
            chunks.rmdir()
    print(f"SHA256 verified: {name} ({size} bytes)", flush=True)


def main():
    response = requests.get(API + "/repo/files",
        params={"Revision": REVISION, "Recursive": "true"}, timeout=(20, 60))
    response.raise_for_status()
    metadata = response.json()
    if not metadata.get("Success"):
        raise RuntimeError("ModelScope did not return a successful file listing")
    write_json(ROOT / "manifests/modelscope/facebook--sam3.json", metadata)
    entries = [e for e in metadata["Data"]["Files"] if e["Type"] == "blob"
               and e["Path"] not in ["model.safetensors", ".gitattributes"]]
    checkpoint = next(e for e in entries if e["Path"] == "sam3.pt")
    if checkpoint["Sha256"] != SAM3_SHA:
        raise ValueError("Checkpoint no longer matches original SAM 3 SHA256")
    for entry in sorted(entries, key=lambda e: e["Path"] == "sam3.pt"):
        fetch_file(entry)
    receipt = {"verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "https://www.modelscope.cn/models/facebook/sam3", "revision": REVISION,
        "checkpoint_matches_huggingface_original_sha256": True,
        "checkpoint_loaded": False,
        "files": [{"path": (DESTINATION / e["Path"]).relative_to(ROOT).as_posix(),
                   "bytes": e["Size"], "sha256": e["Sha256"]} for e in entries]}
    write_json(ROOT / "reports/sam3-modelscope-download.json", receipt)
    config_path = ROOT / "configs/reproduction.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["required_assets"]["segmentation_weights"] = "weights/sam3/sam3.pt"
    write_json(config_path, config)
    print("SAM3 download complete; checkpoint verified against ModelScope and original-model hashes.", flush=True)


if __name__ == "__main__":
    main()
