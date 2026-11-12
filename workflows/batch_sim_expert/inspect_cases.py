"""Inspect unpacked SIM handoffs without changing source data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile
from pathlib import PurePosixPath
import zipfile

import numpy as np


def inspect(path: Path) -> dict:
    path = path.resolve()
    if path.is_file():
        return inspect_archive(path)
    manifest_path = path / "handoff_manifest.json"
    timeline_path = path / "timeline.npz"
    if not manifest_path.is_file() or not timeline_path.is_file():
        raise ValueError(f"not a SIM handoff: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    people = sorted((path / "human").glob("person_*"))
    if not people:
        raise ValueError(f"no person_* directory: {path}")
    for person in people:
        for filename in ("smplx_params_world.npz", "smplx_meshes_world.npz"):
            if not (person / filename).is_file():
                raise ValueError(f"missing {person / filename}")
    with np.load(timeline_path, allow_pickle=False) as timeline:
        time_s = np.asarray(timeline["pts_seconds"], dtype=float)
        frame_ids = np.asarray(timeline["frame_ids"])
    if len(time_s) < 2 or len(time_s) != len(frame_ids) or not np.all(np.diff(time_s) > 0):
        raise ValueError(f"invalid timeline: {path}")
    if manifest.get("frames") != len(time_s):
        raise ValueError(f"frame count differs from manifest: {path}")
    kind = {1: "single_stairs", 2: "dual_sit"}.get(len(people), "unsupported_actor_count")
    return {"id": path.name, "source": str(path), "kind": kind,
            "actors": len(people), "frames": len(time_s),
            "fps_estimate": float(1.0 / np.median(np.diff(time_s))),
            "duration_s": float(time_s[-1] - time_s[0]),
            "format": manifest.get("format"), "eligible_for_current_adapters":
            kind == "dual_sit" and len(time_s) == 172}


def inspect_archive(path: Path) -> dict:
    """Read a SIM manifest from ZIP/TAR without extracting or trusting member paths."""
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            names = [item.filename for item in archive.infolist() if not item.is_dir()]
            for item in archive.infolist():
                if (item.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError(f"archive contains symlink: {item.filename}")
            read = archive.read
            result = _archive_manifest(path, names, read)
    elif tarfile.is_tarfile(path):
        with tarfile.open(path, "r:*") as archive:
            members = archive.getmembers()
            if any(not (m.isfile() or m.isdir()) for m in members):
                raise ValueError(f"archive contains links or special files: {path}")
            names = [m.name for m in members if m.isfile()]
            def read(name):
                stream = archive.extractfile(name)
                if stream is None:
                    raise ValueError(f"missing archive member: {name}")
                return stream.read()
            result = _archive_manifest(path, names, read)
    else:
        raise ValueError(f"unsupported archive format: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    result["archive_sha256"] = digest.hexdigest()
    return result


def _archive_manifest(path: Path, names: list[str], read) -> dict:
    normalized = {}
    for name in names:
        value = name.replace("\\", "/")
        member = PurePosixPath(value)
        if member.is_absolute() or ".." in member.parts or value.startswith("/"):
            raise ValueError(f"unsafe archive member: {name}")
        if value in normalized:
            raise ValueError(f"duplicate archive member: {value}")
        normalized[value] = name
    candidates = [name for name in normalized if name.endswith("handoff_manifest.json")]
    if len(candidates) != 1:
        raise ValueError(f"expected one handoff_manifest.json in {path}, got {len(candidates)}")
    manifest_name = candidates[0]
    prefix = manifest_name[:-len("handoff_manifest.json")]
    manifest = json.loads(read(normalized[manifest_name]))
    humans = manifest.get("humans", [])
    if not isinstance(humans, list) or not humans:
        raise ValueError(f"invalid humans manifest in {path}")
    required = [prefix + "timeline.npz"]
    for human in humans:
        required.extend(prefix + human[key].replace("\\", "/") for key in ("meshes", "parameters"))
    missing = sorted(set(required) - set(normalized))
    if missing:
        raise ValueError(f"missing required archive members: {missing[:4]}")
    actors = len(humans)
    frames = int(manifest["frames"])
    if frames < 2:
        raise ValueError(f"invalid frame count in {path}")
    kind = {1: "single_stairs", 2: "dual_sit"}.get(actors, "unsupported_actor_count")
    return {"id": path.name.removesuffix(".tar.gz").removesuffix(".zip").removesuffix(".tgz").removesuffix(".tar"),
            "source": str(path), "archive_member_root": prefix, "kind": kind,
            "actors": actors, "frames": frames, "fps_estimate": float(manifest["fps"]),
            "duration_s": (frames - 1) / float(manifest["fps"]),
            "format": manifest.get("format"), "eligible_for_current_adapters":
            kind == "dual_sit" and frames == 172}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = [inspect(path) for path in args.sources]
    text = json.dumps({"schema": "sim_batch.case_inventory.v1", "cases": rows},
                      ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
