"""Create an auditable frame slice of a canonical Stage-2 reference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import sha256, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--zero-initial-velocity", action="store_true")
    args = parser.parse_args()
    with np.load(args.source, allow_pickle=False) as archive:
        total = len(archive["qpos_ref"])
        if not 2 <= args.frames <= total:
            raise ValueError(f"frames must be in [2, {total}]")
        arrays = {
            key: (value[:args.frames].copy()
                  if value.ndim and value.shape[0] == total else value.copy())
            for key, value in ((key, archive[key]) for key in archive.files)
        }
    if args.zero_initial_velocity:
        for key in ("qvel_ref", "qvel"):
            if key in arrays:
                arrays[key][0] = 0.0
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    report = {
        "schema": "stage2-canonical-slice-v1",
        "source": str(Path(args.source)),
        "source_sha256": sha256(args.source),
        "frames": args.frames,
        "zero_initial_velocity": args.zero_initial_velocity,
        "output_sha256": sha256(output),
    }
    write_json(output.with_suffix(".manifest.json"), report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
