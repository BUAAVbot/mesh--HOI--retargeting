"""Fetch a small Linux wheelhouse locally; never install GPU packages locally."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import time
from urllib.parse import urljoin, urlparse, unquote
import requests
from bs4 import BeautifulSoup
from prepare_dependencies import ROOT, download, parallel_ranges, write_json


def fetch(spec):
    name, version = spec.split("==", 1)
    metadata_path = ROOT / f"manifests/pypi/{name}--{version}.json"
    if metadata_path.is_file():
        info = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        for attempt in range(5):
            try:
                response = requests.get(f"https://pypi.org/pypi/{name}/{version}/json", timeout=(15, 30))
                response.raise_for_status()
                info = response.json()
                write_json(metadata_path, info)
                break
            except requests.RequestException:
                if attempt == 4:
                    raise
                time.sleep(attempt + 1)
    wheels = [item for item in info["urls"] if item["packagetype"] == "bdist_wheel"]
    eligible = [item for item in wheels if item["filename"].endswith("none-any.whl")]
    if not eligible:
        eligible = [item for item in wheels if "x86_64" in item["filename"] and "manylinux" in item["filename"] and any(tag in item["filename"] for tag in ("cp310-cp310", "cp39-abi3", "cp38-abi3", "cp37-abi3"))]
    if not eligible:
        raise RuntimeError("No compatible Python 3.10 Linux wheel: " + spec)
    item = sorted(eligible, key=lambda row: row["filename"])[0]
    url = item["url"]
    path = ROOT / "downloads/runtime-wheels" / item["filename"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file() and item["size"] > 2**20:
        parallel_ranges(url, path.with_suffix(path.suffix + ".part"), item["size"])
    record = download(url, path, expected_sha=item["digests"]["sha256"], expected_size=item["size"])
    print("Verified wheel: " + item["filename"], flush=True)
    return {**record, "package": spec, "official_pypi_url": item["url"], "official_pypi_sha256_checked": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packages", nargs="*", default=["transformers==5.1.0", "peft==0.18.1", "huggingface-hub==1.33.0", "tokenizers==0.22.2", "httpx==0.28.1", "httpcore==1.0.9", "ultralytics==8.1.34", "thop==0.1.1.post2209072238", "ftfy==6.1.1"])
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(fetch, args.packages))
    manifest = ROOT / "manifests/runtime-wheels.json"
    previous = json.loads(manifest.read_text(encoding="utf-8"))["files"] if manifest.is_file() else []
    combined = {record["path"]: record for record in [*previous, *records]}
    write_json(manifest, {"target": "Linux x86_64 Python 3.10, pure-Python wheels also usable by 3.12", "files": list(combined.values())})
    print(json.dumps({"verified_wheels": len(records), "bytes": sum(item["bytes"] for item in records)}))


if __name__ == "__main__":
    main()
