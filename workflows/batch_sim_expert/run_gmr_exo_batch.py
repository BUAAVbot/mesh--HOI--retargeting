"""Run genuine GMR retargeting for every selected exo person track."""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import subprocess
import sys
from pathlib import Path


SKIP = {"videomimic_gallery_07_v4", "videomimic_single_input_v2"}


def run(item: tuple[Path, Path], args: argparse.Namespace) -> dict:
    src, out = item
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(args.solver), "--input", str(src), "--model", str(args.model),
           "--output", str(out), "--gmr-source", str(args.gmr_source)]
    if out.exists():
        return {"input": str(src), "output": str(out), "status": "existing"}
    result = subprocess.run(cmd, capture_output=True, text=True)
    (out.parent / "run.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    return {"input": str(src), "output": str(out), "status": "success" if result.returncode == 0 else "failed",
            "returncode": result.returncode, "last_output": (result.stdout + result.stderr)[-1000:]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--solver", type=Path, required=True)
    ap.add_argument("--gmr-source", type=Path, default=Path("/data/grail-tools/gmr-pinned-source"))
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    items = []
    for src in sorted(a.input_root.glob("*/humans/person_*/smplx_human_motion.npz")):
        task = src.parents[2].name
        if task in SKIP:
            continue
        person = src.parent.name
        items.append((src, a.output_root / task / person / "gmr" / "reference.npz"))
    print(json.dumps({"count": len(items), "tasks": sorted({p.parents[2].name for p, _ in items})}), flush=True)
    with cf.ThreadPoolExecutor(max_workers=a.workers) as pool:
        futs = [pool.submit(run, item, a) for item in items]
        for fut in cf.as_completed(futs):
            print(json.dumps(fut.result()), flush=True)


if __name__ == "__main__":
    main()
