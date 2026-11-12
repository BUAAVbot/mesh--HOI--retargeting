"""Run official HOI-Retarget windows on selected exo tracks initialized by Omni."""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import subprocess
import sys
from pathlib import Path


SKIP = {"videomimic_gallery_07_v4", "videomimic_single_input_v2"}


def run(item: tuple[Path, Path, Path], args: argparse.Namespace) -> dict:
    src, stage1, out = item
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        return {"input": str(src), "status": "existing", "output": str(out)}
    cmd = [sys.executable, str(args.solver), "--input", str(src), "--model", str(args.model),
           "--source", str(args.source), "--stage1-reference", str(stage1), "--output", str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    (out.parent / "run.log").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    return {"input": str(src), "stage1": str(stage1), "output": str(out),
            "status": "success" if proc.returncode == 0 else "failed", "returncode": proc.returncode,
            "last_output": (proc.stdout + proc.stderr)[-1000:]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--solver", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    items = []
    for src in sorted(a.input_root.glob("*/humans/person_*/smplx_human_motion.npz")):
        task = src.parents[2].name
        if task in SKIP:
            continue
        person = src.parent.name
        parent = a.output_root / task / person
        stage1 = parent / "omni" / "reference.npz"
        if not stage1.exists():
            raise FileNotFoundError(stage1)
        items.append((src, stage1, parent / "hoi" / "reference.npz"))
    print(json.dumps({"count": len(items), "tasks": sorted({p.parents[2].name for p, _, _ in items})}), flush=True)
    with cf.ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(run, item, a) for item in items]
        for future in cf.as_completed(futures):
            print(json.dumps(future.result()), flush=True)


if __name__ == "__main__":
    main()
