"""Audit every HOI trajectory, official-window status, and Omni initializer hash."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main(root: Path) -> None:
    rows = []
    for path in sorted(root.glob("*/person_*/hoi/reference.npz")):
        report = json.loads(path.with_suffix(".json").read_text())
        stage1 = Path(report["stage1_reference_path"])
        with np.load(path, allow_pickle=False) as result, np.load(stage1, allow_pickle=False) as source:
            q = result["qpos"]
            sq = source["qpos"]
            fps = float(result["fps"])
            shape_ok = q.ndim == 2 and q.shape[1] == 35 and q.shape == sq.shape
            finite = bool(np.isfinite(q).all())
        stages_match = sha256(stage1) == report["stage1_reference_sha256"]
        windows = report["window_status"]
        solved = sum(item["status"] == "solved" for item in windows)
        row = {"task": path.parents[2].name, "person": path.parents[1].name,
               "frames": q.shape[0], "shape_ok": shape_ok, "finite": finite,
               "fps": fps, "stage1_hash_current": stages_match,
               "window_count": len(windows), "solved_windows": solved,
               "status": report["status"]}
        rows.append(row)
    summary = {"tracks": len(rows), "windows": sum(r["window_count"] for r in rows),
               "solved_windows": sum(r["solved_windows"] for r in rows),
               "all_valid": len(rows) == 10 and all(r["shape_ok"] and r["finite"] and
                   r["stage1_hash_current"] and r["window_count"] == r["solved_windows"] for r in rows),
               "rows": rows}
    print(json.dumps(summary))
    if not summary["all_valid"]:
        sys.exit(1)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
