"""Build the auditable comparison table from fresh-replay reports."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRICS = (
    "root_rmse_m", "root_max_m", "final_root_error_m", "joint_rmse_rad",
    "penetration_max_m", "effort_ratio_max", "support_fraction",
)


def load_runtime(run_dir: Path) -> float | None:
    """Read wall-clock optimizer time without inventing a value."""
    for name in ("manifest.json", "run_manifest.json"):
        path = run_dir / name
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for key in ("runtime_s", "wall_time_s", "elapsed_s", "optimization_time_s"):
            if key in data:
                return float(data[key])
    return None


def violation_score(report: dict) -> float:
    metrics, checks = report["metrics"], report["checks"]
    score = 0.0
    for key, passed in checks.items():
        if passed:
            continue
        value = float(metrics[key])
        # The only lower-bound gate in the frozen contract is support_fraction.
        if key == "support_fraction":
            score += max(0.0, 0.8 / max(value, 1e-12) - 1.0)
        else:
            limits = {
                "root_max_m": .25, "root_rmse_m": .10, "final_root_error_m": .10,
                "ee_z_max_m": .25, "body_relative_max_m": .35,
                "body_relative_rmse_m": .15, "root_orientation_rmse_rad": .25,
                "final_root_orientation_rad": .30, "body_orientation_rmse_rad": .35,
                "joint_rmse_rad": .30, "joint_error_max_rad": 1.0,
                "penetration_max_m": .012, "joint_limit_excess_max_rad": .02,
                "effort_ratio_max": 1.001,
            }
            score += max(0.0, value / limits[key] - 1.0)
    return score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-baseline", action="store_true")
    args = parser.parse_args()
    rows = []
    for report_path in sorted(args.runs_root.glob("*/replay_a/report.json")):
        if report_path.parents[1].name == "baseline" and not args.include_baseline:
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        row = {
            "run": report_path.parents[1].name,
            "method": report["method"],
            "passed": bool(report["passed"]),
            "failed_gates": sum(not value for value in report["checks"].values()),
            "violation_score": violation_score(report),
            "runtime_s": load_runtime(report_path.parents[1]),
            **{key: report["metrics"][key] for key in METRICS},
            "report": str(report_path),
        }
        rows.append(row)
    if not rows:
        raise SystemExit("No replay_a/report.json files found")
    rows.sort(key=lambda row: (not row["passed"], row["failed_gates"],
                               row["violation_score"], row["root_rmse_m"]))
    args.output.mkdir(parents=True, exist_ok=True)
    fields = tuple(rows[0])
    with (args.output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    lines = [
        "# Stage-2 same-track comparison", "",
        "All rows use the same ③ Gallery08-P01 reference, Vita28 model, calibrated scene,",
        "HDMI PD gains, 0.005 s physics step and fresh no-reset replay.", "",
        "| Run | Method | Pass | Runtime (s) | Failed gates | Score | Root RMSE (m) | Penetration (m) | Support |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['run']} | {row['method']} | {row['passed']} | "
            f"{row['runtime_s'] if row['runtime_s'] is not None else 'N/A'} | {row['failed_gates']} | "
            f"{row['violation_score']:.3f} | {row['root_rmse_m']:.4f} | "
            f"{row['penetration_max_m']:.4f} | {row['support_fraction']:.3f} |"
        )
    lines += ["", f"Best auditable run: `{rows[0]['run']}`.", ""]
    (args.output / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
