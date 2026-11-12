"""Validate a normalized backend result before metrics or video rendering."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import load_canonical, sha256, write_json


def validate(reference_path: str, result_path: str, method: str) -> dict:
    reference = load_canonical(reference_path)
    frames = reference["qpos_ref"].shape[0]
    with np.load(result_path, allow_pickle=False) as archive:
        result = {key: archive[key] for key in archive.files}
    expected = {"qpos": (frames, 35), "qvel": (frames, 34), "ctrl": (frames, 28)}
    for key, shape in expected.items():
        if key not in result or result[key].shape != shape:
            raise ValueError(f"{key} must have shape {shape}")
        if not np.isfinite(result[key]).all():
            raise ValueError(f"{key} contains non-finite values")
    if method == "prime":
        for key in ("tau", "contact_lambda", "dynamics_residual", "friction_residual"):
            if key not in result:
                raise ValueError(f"PRIME result lacks strict Stage-2 field {key!r}")
            if not np.isfinite(result[key]).all():
                raise ValueError(f"{key} contains non-finite values")
    report = {
        "schema": "stage2-normalized-result-validation-v1",
        "method": method,
        "frames": frames,
        "reference_sha256": sha256(reference_path),
        "result_sha256": sha256(result_path),
        "valid": True,
    }
    write_json(Path(result_path).with_suffix(".validation.json"), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--method", choices=("spider", "sbto", "prime"), required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.reference, args.result, args.method), indent=2))


if __name__ == "__main__":
    main()

