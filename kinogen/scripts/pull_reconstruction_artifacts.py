"""Fetch reconstruction artifacts through bounded 32 KiB raw SSH chunks."""
import argparse
import json
from pathlib import PurePosixPath
from datetime import datetime, timezone

from prepare_dependencies import ROOT
from pull_ascii_artifact import download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--replace", action="store_true", help="Replace an older local derived artifact")
    parser.add_argument("--timeout", type=float, default=45, help="Timeout per small SSH chunk, seconds")
    parser.add_argument("--workers", type=int, default=3, choices=(1, 2, 3))
    args = parser.parse_args()
    target = json.loads((ROOT / "configs/execution-targets.json").read_text(encoding="utf-8"))["server"]
    records = []
    for name in args.paths:
        relative = PurePosixPath(name)
        safe = (name.startswith("outputs/previews/user_") or
                name.startswith("outputs/user_reconstruction/") or name.startswith("outputs/user_reconstruction_v2/") or name.startswith("reports/user-"))
        if not safe or relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
            parser.error("Expected a reconstruction artifact relative to the server project")
        if relative.suffix.lower() not in {".json", ".mp4", ".png", ".jpg", ".ply", ".npz", ".glb"}:
            parser.error("Unsupported artifact type")
        local_name = name.replace("outputs/previews/", "previews/", 1)
        destination = ROOT / local_name
        if not destination.resolve().is_relative_to(ROOT.resolve()) or destination.is_symlink() or destination.is_junction():
            raise RuntimeError("Unsafe local artifact destination")
        record = download(name, destination, timeout=args.timeout, replace=args.replace, workers=args.workers)
        record.update(server_path=name, local_path=local_name)
        records.append(record)
        print(json.dumps(record), flush=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    manifest = ROOT / "manifests/deployment" / ("reconstruction-pull-" + stamp + ".json")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"server": target["ssh_host"], "files": records}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
