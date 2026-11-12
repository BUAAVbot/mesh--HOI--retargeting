"""Unit-aware comparison of two complete fresh physical recordings."""
import argparse
import json
from pathlib import Path

import numpy as np
from acceptance import angular_distance, load_npz, names, sha256, EE_BODIES


def summarize(value):
    v = np.asarray(value, np.float64)
    return dict(maximum=float(v.max()), rms=float(np.sqrt(np.mean(v ** 2))),
                final_maximum=float(v[-1].max()),
                peak_index=list(map(int, np.unravel_index(v.argmax(), v.shape))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--first", type=Path, required=True)
    p.add_argument("--second", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    a, b = load_npz(args.first), load_npz(args.second)
    if names(a, "joint_names") != names(b, "joint_names") or names(a, "body_names") != names(b, "body_names"):
        raise ValueError("Record ordering differs")
    result = dict(first_sha256=sha256(args.first), second_sha256=sha256(args.second))
    ar, br = a["root_state"], b["root_state"]
    result["root_xyz_m"] = summarize(np.linalg.norm(ar[:, :3] - br[:, :3], axis=-1))
    result["root_quaternion_angle_rad"] = summarize(angular_distance(ar[:, 3:7], br[:, 3:7]))
    result["joint_position_rad"] = summarize(np.abs(a["joint_pos"] - b["joint_pos"]))
    result["root_linear_velocity_m_s"] = summarize(np.linalg.norm(ar[:, 7:10] - br[:, 7:10], axis=-1))
    result["root_angular_velocity_rad_s"] = summarize(np.linalg.norm(ar[:, 10:13] - br[:, 10:13], axis=-1))
    result["joint_velocity_rad_s"] = summarize(np.abs(a["joint_vel"] - b["joint_vel"]))
    result["end_effectors"] = {}
    bodies = names(a, "body_names")
    for name in EE_BODIES:
        idx = bodies.index(name)
        diff = a["body_pos_w"][:, idx] - b["body_pos_w"][:, idx]
        result["end_effectors"][name] = dict(position_m=summarize(np.linalg.norm(diff, axis=-1)),
                                            height_m=summarize(np.abs(diff[:, 2])))
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
