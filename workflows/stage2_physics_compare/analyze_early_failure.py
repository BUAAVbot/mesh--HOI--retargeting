"""Print the event order around early strict-replay failure."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def first_index(mask: np.ndarray) -> int | None:
    hits = np.flatnonzero(mask)
    return int(hits[0]) if hits.size else None


def describe(path: Path, reference: Path) -> None:
    run = np.load(path, allow_pickle=True)
    ref = np.load(reference, allow_pickle=True)
    print(f"PATH {path}")
    print("KEYS", [(key, run[key].shape) for key in run.files])
    qpos = run["qpos"] if "qpos" in run else run["qpos_record"]
    qref = ref["qpos_ref"]
    # Saved qpos records are at the reference rate; substep-only diagnostics are 200 Hz.
    indices = np.minimum(np.arange(len(qpos)), len(qref) - 1)
    qref = qref[indices]
    root_error = np.linalg.norm(qpos[:, :3] - qref[:, :3], axis=1)
    height_drop = qref[:, 2] - qpos[:, 2]
    print("FIRST_ROOT_ERROR_GT", {
        threshold: first_index(root_error > threshold)
        for threshold in (0.05, 0.10, 0.25, 0.50, 1.00)
    })
    print("FIRST_HEIGHT_DROP_GT", {
        threshold: first_index(height_drop > threshold)
        for threshold in (0.05, 0.10, 0.25, 0.50)
    })
    for key in ("physics_effort_ratio", "physics_penetration_m",
                "physics_support_force_N"):
        if key in run:
            value = run[key]
            print(key, "shape", value.shape, "first10", np.asarray(value[:10]).tolist())
    if "physics_effort_ratio" in run:
        effort = np.asarray(run["physics_effort_ratio"])
        if effort.ndim > 1:
            effort = effort.max(axis=1)
        print("FIRST_EFFORT_RATIO", {
            threshold: first_index(effort >= threshold)
            for threshold in (0.80, 0.95, 0.999)
        })
    if "physics_penetration_m" in run:
        penetration = np.asarray(run["physics_penetration_m"])
        if penetration.ndim > 1:
            penetration = penetration.max(axis=tuple(range(1, penetration.ndim)))
        print("FIRST_PENETRATION_GT", {
            threshold: first_index(penetration > threshold)
            for threshold in (0.005, 0.012)
        })
    if "physics_support_force_N" in run:
        support = np.asarray(run["physics_support_force_N"])
        if support.ndim > 1:
            support = support.mean(axis=1)
        print("FIRST_LOW_SUPPORT", {
            threshold: first_index(support < threshold)
            for threshold in (0.8 * 543.6836, 0.5 * 543.6836, 0.1 * 543.6836)
        })
        print("SUPPORT_FIRST20", support[:20].tolist())
    if "contact_ref" in ref:
        print("CONTACT_REF_FIRST20", np.asarray(ref["contact_ref"][:20]).tolist())
    if "contact_min_distance_m" in ref:
        print("CONTACT_DISTANCE_FIRST20_M",
              np.asarray(ref["contact_min_distance_m"][:20]).tolist())
    print("ROOT_SAMPLES", [
        {"i": i, "t": i * 0.02, "xyz": qpos[i, :3].tolist(),
         "ref": qref[i, :3].tolist(), "err": float(root_error[i])}
        for i in (0, 4, 8, 20, 40, 80, 120, 200, 400, min(800, len(qpos)-1))
        if i < len(qpos)
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.paths:
        describe(path, args.reference)


if __name__ == "__main__":
    main()
