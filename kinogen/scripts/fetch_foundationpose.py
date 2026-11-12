"""Fetch the author's public Drive files without login or browser cookies."""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
import gdown

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    ("2023-10-28-18-33-37/config.yml", "1477-st1s1TxXN6oqfM5ZnsQwd8BCzVg1"),
    ("2023-10-28-18-33-37/model_best.pth", "1E9FPB5WFIBMLrOJqZLpoVOK4Mjzrrxhv"),
    ("2024-01-11-20-02-45/config.yml", "1kQkQG-q_VvLRozv30hyeLB7P_jiEEqiE"),
    ("2024-01-11-20-02-45/model_best.pth", "1Zdjnkn4EHOI5_k08apofwRgTjWpai4E4"),
]


def main():
    records, failures = [], []
    for name, file_id in FILES:
        path = ROOT / "weights/foundationpose-pytorch" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if not path.is_file():
                result = gdown.download(id=file_id, output=str(path), quiet=True, use_cookies=False, resume=True, timeout=(15, 30), retries=2)
                if result is None:
                    raise RuntimeError("Public Drive download returned no file")
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            records.append({"component": "FoundationPose original PyTorch", "path": path.relative_to(ROOT).as_posix(), "url": f"https://drive.google.com/file/d/{file_id}/view", "bytes": path.stat().st_size, "sha256": digest, "verification": "local_sha256_and_transfer_success"})
            print(f"READY {name}: {path.stat().st_size} bytes", flush=True)
        except Exception as exc:
            failures.append({"path": str(path.relative_to(ROOT)), "error": type(exc).__name__})
            print(f"FAILED {name}: {type(exc).__name__}", flush=True)
        manifest = ROOT / "manifests/foundationpose-pytorch.json"
        manifest.parent.mkdir(exist_ok=True)
        manifest.write_text(json.dumps({"checked_at_utc": datetime.now(timezone.utc).isoformat(), "files": records, "failures": failures}, indent=2) + "\n", encoding="utf-8")
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
