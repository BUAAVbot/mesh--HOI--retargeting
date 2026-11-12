"""Verify only explicitly listed deployment files, using the standard library."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    receipt = json.loads((root / args.receipt).read_text(encoding="utf-8"))
    errors = []
    for item in receipt["files"]:
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size != item["bytes"]:
            errors.append(item["path"])
            continue
        with path.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        if checksum != item["sha256"]:
            errors.append(item["path"])
    print(json.dumps({"verified": len(receipt["files"]) - len(errors), "errors": errors}))
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
