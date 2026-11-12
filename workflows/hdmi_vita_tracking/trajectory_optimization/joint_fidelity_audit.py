"""Audit source hard limits and joint-wise tracking without creating a GPU env."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from acceptance import load_npz, names


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--record", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--prefix-frames", type=int, default=139)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    source, record = load_npz(args.source), load_npz(args.record)
    sj, rj = names(source, "joint_names"), names(record, "joint_names")
    model = mujoco.MjModel.from_binary_path(str(args.model))
    model_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
                   for jid in range(model.njnt)]
    rows = []
    last = min(args.prefix_frames, len(record["joint_pos"]), len(source["joint_pos"]))
    for si, name in enumerate(sj):
        ji = [j for j, n in enumerate(model_names) if n == name or (n and n.endswith("/" + name))]
        if len(ji) != 1:
            raise ValueError(f"Cannot unambiguously map {name}: {ji}")
        ji = ji[0]
        q = np.asarray(source["joint_pos"][:, si])
        error = np.asarray(record["joint_pos"][:last, rj.index(name)]) - q[:last]
        lo, hi = model.jnt_range[ji]
        excess = np.maximum(np.maximum(lo - q, q - hi), 0.) if model.jnt_limited[ji] else np.zeros_like(q)
        rows.append(dict(name=name, hard_limited=bool(model.jnt_limited[ji]),
                         lower_rad=float(lo), upper_rad=float(hi),
                         source_min_rad=float(q.min()), source_max_rad=float(q.max()),
                         source_hard_excess_max_rad=float(excess.max()),
                         source_hard_excess_peak_frame=int(excess.argmax()),
                         source_initial_hard_excess_rad=float(excess[0]),
                         prefix_joint_rmse_rad=float(np.sqrt(np.mean(error ** 2))),
                         prefix_joint_max_error_rad=float(np.abs(error).max()),
                         prefix_joint_max_error_frame=int(np.abs(error).argmax())))
    result = dict(prefix_frames=last, joints=rows,
                  source_max_hard_excess_rad=max(row["source_hard_excess_max_rad"] for row in rows),
                  rmse_ranked=sorted(rows, key=lambda row: row["prefix_joint_rmse_rad"], reverse=True))
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(prefix_frames=last, source_max_hard_excess_rad=result["source_max_hard_excess_rad"],
                          source_violations=[row for row in rows if row["source_hard_excess_max_rad"] > .001],
                          top_tracking_errors=result["rmse_ranked"][:10])))


if __name__ == "__main__":
    main()
