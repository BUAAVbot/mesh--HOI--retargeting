"""Index and verify extracted source files against the retained, pinned archives."""
import hashlib
import json
from pathlib import PurePosixPath
import tarfile
from prepare_dependencies import ROOT, native_path, sha256, write_json


def main():
    for name in ("tapnet", "video-depth-anything"):
        manifest_path = ROOT / f"manifests/sources/{name}.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        archive = ROOT / manifest["path"]
        if sha256(archive) != manifest["sha256"]:
            raise RuntimeError(f"Source archive checksum mismatch: {name}")
        files = []
        with tarfile.open(archive, "r:gz") as source:
            for member in source.getmembers():
                if not member.isfile():
                    continue
                relative = PurePosixPath(member.name).parts[1:]
                if not relative or ".." in relative:
                    raise ValueError("Unexpected archive member path")
                relative = PurePosixPath(*relative).as_posix()
                with source.extractfile(member) as stream:
                    expected = hashlib.file_digest(stream, "sha256").hexdigest()
                path = native_path(ROOT / manifest["source_directory"] / relative)
                if not path.is_file() or path.stat().st_size != member.size or sha256(path) != expected:
                    raise RuntimeError(f"Extracted source mismatch: {name}/{relative}")
                files.append({"path": relative, "bytes": member.size, "sha256": expected, "mode": format(member.mode, "06o")})
        manifest["files"] = files
        manifest["files_indexed_from_verified_archive"] = True
        write_json(manifest_path, manifest)
        print(f"Verified archive and {len(files)} extracted files: {name}")


if __name__ == "__main__":
    main()
