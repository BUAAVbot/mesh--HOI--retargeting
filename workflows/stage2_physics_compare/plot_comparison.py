"""Render a compact, auditable Stage-2 comparison figure from comparison.csv."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.csv.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise SystemExit("comparison.csv has no rows")

    label_key = "run" if "run" in rows[0] else "method"
    pass_key = "passed" if "passed" in rows[0] else "status"
    labels = [row[label_key] for row in rows if row.get("root_rmse_m")]
    metric_rows = [row for row in rows if row.get("root_rmse_m")]
    root_cm = [100.0 * float(row["root_rmse_m"]) for row in metric_rows]
    penetration_mm = [
        float(row["max_penetration_mm"])
        if "max_penetration_mm" in row
        else 1000.0 * float(row["penetration_max_m"])
        for row in metric_rows
    ]
    colors = [
        "#2ca02c" if row[pass_key].lower() in {"true", "pass", "passed"} else "#d62728"
        for row in metric_rows
    ]

    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(rows) * 1.35), 5.2))
    axes[0].bar(labels, root_cm, color=colors)
    axes[0].axhline(10.0, color="black", linestyle="--", linewidth=1, label="gate 10 cm")
    axes[0].set_ylabel("Root RMSE (cm)")
    axes[0].set_title("Tracking")
    axes[0].legend()

    axes[1].bar(labels, penetration_mm, color=colors)
    axes[1].axhline(12.0, color="black", linestyle="--", linewidth=1, label="gate 12 mm")
    axes[1].set_ylabel("Max penetration (mm)")
    axes[1].set_title("Contact feasibility")
    axes[1].legend()

    for axis in axes:
        axis.tick_params(axis="x", labelrotation=35)
        axis.grid(axis="y", alpha=.25)
    fig.suptitle("Same-track Stage-2 strict no-reset replay (green=all gates passed)")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
