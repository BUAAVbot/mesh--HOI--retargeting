#!/usr/bin/env python3
"""Build an audited low-resolution hfield used only for SPIDER optimization.

MuJoCo-Warp caps heightfield collision candidates at 50 per geom.  The frozen
153x104, 25 mm field overflows that cap for the Vita foot collision mesh when
SPIDER rolls out hundreds of worlds.  This tool changes only the hfield grid in
a copy of the compiled model.  Final acceptance must still replay controls in
the original frozen MJB.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def reconstruct(low: np.ndarray, rows: np.ndarray, cols: np.ndarray,
                nrow: int, ncol: int) -> np.ndarray:
    """Bilinearly reconstruct a sparse grid at every original grid point."""
    full_cols = np.arange(ncol, dtype=np.float64)
    horizontal = np.stack([
        np.interp(full_cols, cols, row) for row in low
    ])
    full_rows = np.arange(nrow, dtype=np.float64)
    return np.stack([
        np.interp(full_rows, rows, horizontal[:, column])
        for column in range(ncol)
    ], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--geom", default="calibrated_scene_surface")
    parser.add_argument("--strict-replay-model", type=Path, required=True)
    args = parser.parse_args()
    if args.stride < 2:
        raise ValueError("stride must be >= 2")
    if args.model.resolve() != args.strict_replay_model.resolve():
        raise ValueError("Source model must be the frozen strict replay model")

    model = mujoco.MjModel.from_binary_path(str(args.model))
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, args.geom)
    if geom_id < 0 or int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_HFIELD):
        raise ValueError(f"{args.geom!r} is not a heightfield geom")
    hfield_id = int(model.geom_dataid[geom_id])
    nrow, ncol = int(model.hfield_nrow[hfield_id]), int(model.hfield_ncol[hfield_id])
    address = int(model.hfield_adr[hfield_id])
    original_count = nrow * ncol
    original = np.asarray(
        model.hfield_data[address:address + original_count], dtype=np.float64
    ).reshape(nrow, ncol).copy()
    # Always retain exact boundary samples, so the proxy has identical extent.
    rows = np.unique(np.r_[np.arange(0, nrow, args.stride), nrow - 1]).astype(int)
    cols = np.unique(np.r_[np.arange(0, ncol, args.stride), ncol - 1]).astype(int)
    low = original[np.ix_(rows, cols)]
    restored = reconstruct(low, rows, cols, nrow, ncol)
    vertical_scale = float(model.hfield_size[hfield_id, 2])
    error_m = (restored - original) * vertical_scale

    model.hfield_nrow[hfield_id] = len(rows)
    model.hfield_ncol[hfield_id] = len(cols)
    model.hfield_data[address:address + low.size] = low.ravel()
    # Make unused tail deterministic; there is one heightfield in this model.
    model.hfield_data[address + low.size:address + original_count] = 0.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mujoco.mj_saveModel(model, str(args.output))

    check = mujoco.MjModel.from_binary_path(str(args.output))
    if (int(check.hfield_nrow[hfield_id]), int(check.hfield_ncol[hfield_id])) != low.shape:
        raise RuntimeError("Saved proxy did not retain reduced hfield shape")
    reloaded = np.asarray(
        check.hfield_data[address:address + low.size], dtype=np.float64
    ).reshape(low.shape)
    if not np.allclose(reloaded, low, atol=1e-7):
        raise RuntimeError("Saved proxy hfield samples changed")
    if (check.nq, check.nv, check.nu, check.nbody, check.ngeom) != (
        model.nq, model.nv, model.nu, model.nbody, model.ngeom
    ):
        raise RuntimeError("Proxy changed robot/model dimensions")

    report = {
        "schema": "spider-optimization-hfield-proxy-v1",
        "proxy_optimization_only": True,
        "acceptance_replay_must_use_original_model": True,
        "source_model": str(args.model.resolve()),
        "source_model_sha256": sha256(args.model),
        "proxy_model": str(args.output.resolve()),
        "proxy_model_sha256": sha256(args.output),
        "strict_replay_model": str(args.strict_replay_model.resolve()),
        "strict_replay_model_sha256": sha256(args.strict_replay_model),
        "geom_name": args.geom,
        "geom_id": geom_id,
        "hfield_id": hfield_id,
        "stride": args.stride,
        "original_shape_yx": [nrow, ncol],
        "proxy_shape_yx": list(low.shape),
        "original_spacing_multiplier": [
            float((nrow - 1) / (len(rows) - 1)),
            float((ncol - 1) / (len(cols) - 1)),
        ],
        "height_approximation_error_m": {
            "maximum_absolute": float(np.max(np.abs(error_m))),
            "rmse": float(np.sqrt(np.mean(error_m ** 2))),
            "mean_absolute": float(np.mean(np.abs(error_m))),
            "p95_absolute": float(np.quantile(np.abs(error_m), 0.95)),
        },
        "unchanged_dimensions": {
            "nq": int(check.nq), "nv": int(check.nv), "nu": int(check.nu),
            "nbody": int(check.nbody), "ngeom": int(check.ngeom),
        },
        "declared_model_changes": [
            "hfield_nrow", "hfield_ncol", "hfield_data[calibrated_scene_heightfield]"
        ],
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
