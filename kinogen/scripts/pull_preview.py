"""List or fetch server-exported videos using the existing SSH profile."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
from prepare_dependencies import ROOT

VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".mkv"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", nargs="?", help="Video name from --list")
    parser.add_argument("--list", action="store_true", help="List exported videos")
    parser.add_argument("--open", action="store_true", help="Open the downloaded video locally")
    args = parser.parse_args()
    if not args.list and not args.name:
        parser.error("Use --list or supply a video name")
    if args.name and (Path(args.name).name != args.name or "/" in args.name or "\\" in args.name or ":" in args.name or Path(args.name).is_reserved() or Path(args.name).suffix.lower() not in VIDEO_SUFFIXES):
        parser.error("Supply a video filename, without a directory")
    target = json.loads((ROOT / "configs/execution-targets.json").read_text(encoding="utf-8"))["server"]
    remote_directory = target.get("preview_export_directory", target["project_root"] + "/outputs/previews")
    ssh = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", target["ssh_host"]]
    code = "\n".join([
        "from pathlib import Path",
        "import json, hashlib",
        "root = Path(" + repr(remote_directory) + ").resolve()",
        "files = []",
        "for path in sorted(root.iterdir()) if root.is_dir() else []:",
        "    if path.is_file() and not path.is_symlink() and path.suffix.lower() in " + repr(VIDEO_SUFFIXES) + ":",
        "        with path.open('rb') as stream:",
        "            checksum = hashlib.file_digest(stream, 'sha256').hexdigest()",
        "        files.append(dict(name=path.name, bytes=path.stat().st_size, sha256=checksum))",
        "print(json.dumps(files))",
    ])
    listing = subprocess.run(ssh + ["/usr/bin/python3.12 -c " + shlex.quote(code)], capture_output=True, check=True)
    files = json.loads(listing.stdout)
    if args.list:
        print(json.dumps(files, ensure_ascii=False, indent=2))
        if not files:
            print("No exported videos yet: " + remote_directory)
        return 0
    match = next((item for item in files if item["name"] == args.name), None)
    if match is None:
        parser.error("Video not found; use --list to see exports")
    directory = ROOT / "previews"
    if directory.is_symlink() or directory.is_junction():
        parser.error("The local previews directory must not be a link or junction")
    directory.mkdir(exist_ok=True)
    destination = directory / args.name
    if destination.is_symlink() or destination.is_junction():
        parser.error("The local video path must not be a link or junction")
    if destination.is_file():
        with destination.open("rb") as stream:
            already_verified = hashlib.file_digest(stream, "sha256").hexdigest() == match["sha256"]
        if not already_verified:
            parser.error("A different local video already uses this name; rename it first")
    else:
        with tempfile.NamedTemporaryFile(dir=directory, prefix=".kinogen-video-", suffix=".part", delete=False) as stream:
            partial = Path(stream.name)
            try:
                subprocess.run(ssh + ["cat -- " + shlex.quote(remote_directory + "/" + args.name)], stdout=stream, check=True)
            except BaseException:
                stream.close()
                partial.unlink(missing_ok=True)
                raise
        try:
            with partial.open("rb") as stream:
                valid = partial.stat().st_size == match["bytes"] and hashlib.file_digest(stream, "sha256").hexdigest() == match["sha256"]
            if not valid:
                raise RuntimeError("Video changed during transfer or failed integrity verification; retry")
            if destination.exists() or destination.is_symlink():
                parser.error("The local video path appeared during transfer; retry")
            partial.rename(destination)
        finally:
            partial.unlink(missing_ok=True)
    print(str(destination))
    if args.open:
        if os.name != "nt":
            raise RuntimeError("--open currently supports the local Windows preview host")
        os.startfile(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
