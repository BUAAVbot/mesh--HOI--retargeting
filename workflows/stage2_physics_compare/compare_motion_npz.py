"""Compare two MJLab motion archives without assuming identical keys."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    args = parser.parse_args()
    left = np.load(args.left, allow_pickle=False)
    right = np.load(args.right, allow_pickle=False)
    result = {}
    for key in sorted(set(left.files) | set(right.files)):
        if key not in left or key not in right:
            result[key] = {"left": key in left, "right": key in right}
            continue
        a, b = left[key], right[key]
        row = {"left_shape": list(a.shape), "right_shape": list(b.shape),
               "same_dtype": a.dtype == b.dtype}
        if a.shape == b.shape and np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number):
            row["max_abs_difference"] = float(np.max(np.abs(a - b)))
            row["equal"] = bool(np.array_equal(a, b))
        result[key] = row
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
