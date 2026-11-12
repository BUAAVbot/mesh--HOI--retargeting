"""Validate the official source ZIP against its Git tree before staging it."""
import hashlib
import json
import re
from pathlib import Path
import zipfile
from prepare_dependencies import ROOT, write_json, now, sha256, native_path

COMMIT = "87c95274f50836068b44eeb030bf62aa2108bdf8"


def main():
    archive = ROOT / f"downloads/sources/isaacsim-{COMMIT}.zip"
    tree = json.loads((ROOT / f"manifests/trees/isaacsim-{COMMIT}.json").read_text(encoding="utf-8"))
    entries = {entry["path"]: entry for entry in tree["tree"] if entry["type"] != "tree"}
    destination = ROOT / "dependencies/isaacsim"
    records, omitted, pointers = [], [], []
    with zipfile.ZipFile(archive) as source:
        invalid = source.testzip()
        if invalid:
            raise ValueError(f"ZIP CRC mismatch: {invalid}")
        prefix = source.infolist()[0].filename.split("/", 1)[0] + "/"
        for relative, entry in entries.items():
            if entry["type"] != "blob" or entry.get("mode") == "120000":
                omitted.append(entry)
                continue
            data = source.read(prefix + relative)
            git_hash = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            eol_conversion = None
            if git_hash != entry["sha"]:
                normalized = data.replace(b"\r\n", b"\n")
                canonical_hash = hashlib.sha1(b"blob " + str(len(normalized)).encode() + b"\0" + normalized).hexdigest()
                if canonical_hash != entry["sha"]:
                    raise ValueError(f"Git blob mismatch: {relative}")
                git_hash = canonical_hash
                eol_conversion = "Official Git archive applies CRLF; LF canonical bytes match Git blob"
            target = (destination / relative).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError("Unsafe archive entry")
            target = native_path(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            pointer = data.decode() if data.startswith(b"version https://git-lfs.github.com/spec/v1") else None
            asset_sha = re.search(r"oid sha256:([0-9a-f]+)", pointer)[1] if pointer else None
            asset_size = int(re.search(r"(?m)^size (\d+)", pointer)[1]) if pointer else None
            asset_ready = bool(pointer and target.is_file() and target.stat().st_size == asset_size and sha256(target) == asset_sha)
            if not asset_ready and (not target.is_file() or hashlib.sha256(target.read_bytes()).digest() != hashlib.sha256(data).digest()):
                target.write_bytes(data)
            record = {"path": relative, "git_blob_sha1": git_hash, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "mode": entry.get("mode")}
            if eol_conversion:
                record["archive_eol_conversion"] = eol_conversion
            if asset_ready:
                record.update({"git_lfs_hydrated": True, "git_lfs_pointer": pointer, "pointer_sha256": record["sha256"], "sha256": asset_sha, "bytes": asset_size})
            elif pointer:
                pointers.append(relative)
                record["git_lfs_pointer_not_hydrated"] = pointer
            records.append(record)
    write_json(ROOT / "manifests/sources/isaacsim.json", {"repository": "https://github.com/isaac-sim/IsaacSim", "commit": COMMIT, "checked_at_utc": now(), "source_directory": "dependencies/isaacsim", "archive_path": archive.relative_to(ROOT).as_posix(), "archive_sha256": sha256(archive), "scope": "Full official source ZIP, checked against Git tree; symlinks/gitlinks recorded separately; runtime/Kit dependencies not installed", "files": records, "omitted": omitted, "unhydrated_lfs_pointers": pointers, "lfs_hydrated_paths": sum(bool(item.get('git_lfs_hydrated')) for item in records), "failures": []})
    print(f"Isaac Sim source: {len(records)} Git-verified files; {len(omitted)} symlinks/gitlinks recorded separately; {len(pointers)} unhydrated LFS pointers")


if __name__ == "__main__":
    main()
