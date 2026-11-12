"""Independent single/batch frozen-actor zero-residual comparison, no RL."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend-dir", type=Path, required=True)
    p.add_argument("--runtime-root", type=Path, required=True)
    p.add_argument("--motion", type=Path, required=True)
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    sys.path.insert(0, str(args.backend_dir))
    from backend import HdmiDynamics, sha
    rows = []
    for count in (1, 4):
        b = HdmiDynamics(args.runtime_root, args.motion, args.scene, args.checkpoint, count=count)
        try:
            cost, metrics, _ = b.rollout(np.zeros((count, b.frames - 1, 28), np.float32))
            row = dict(count=count, origins=b.env.scene.env_origins.cpu().numpy().tolist(),
                       cost=cost.tolist(), metrics={key: value.tolist() for key, value in metrics.items()})
            rows.append(row)
            print(json.dumps(row), flush=True)
        finally:
            b.close()
    # Retain the list interface consumed by acceptance; provenance sits in a
    # same-stem receipt so historical audit loaders remain compatible.
    args.output.write_text(json.dumps(rows, indent=2))
    args.output.with_suffix(".provenance.json").write_text(json.dumps(dict(
        source_reference_sha256=sha(args.motion), scene_sha256=sha(args.scene),
        checkpoint_sha256=sha(args.checkpoint), backend_sha256=sha(args.backend_dir / "backend.py"),
        root_forcing=False, rl_training=False), indent=2))


if __name__ == "__main__":
    main()
