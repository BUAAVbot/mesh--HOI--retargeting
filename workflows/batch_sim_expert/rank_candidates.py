"""Conservative kinematic gate before expensive SONIC physical screening."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def rank(reports: dict[str, Path], kind: str, top_k: int = 2) -> dict:
    rows = []
    for method, path in reports.items():
        report = json.loads(path.read_text(encoding="utf-8"))
        metrics = report["metrics"]
        required = ("max_joint_limit_violation_rad", "max_penetration_m",
                    "max_body_joint_speed_rad_s", "root_xy_rmse_m",
                    "mean_bone_direction_error_deg")
        missing = [key for key in required if key not in metrics]
        if missing:
            raise ValueError(f"{method}: missing comparable metrics {missing}")
        if any(not math.isfinite(float(metrics[key])) for key in required):
            raise ValueError(f"{method}: nonfinite metric")
        failures = []
        if metrics["max_joint_limit_violation_rad"] > 0.01:
            failures.append("joint_limit_over_0.01rad")
        if metrics["max_penetration_m"] > 0.03:
            failures.append("penetration_over_0.03m")
        if metrics["max_body_joint_speed_rad_s"] > 12.0:
            failures.append("joint_speed_over_12rad_s")
        contact = (metrics.get("seat_endpoint_absolute_distance_mean_m", math.inf)
                   if kind == "dual_sit" else metrics.get("stance_foot_speed_mean_m_s", math.inf))
        if not math.isfinite(float(contact)):
            failures.append("missing_task_contact_metric")
        score = (float(metrics["max_penetration_m"]), float(contact),
                 float(metrics["root_xy_rmse_m"]),
                 float(metrics["mean_bone_direction_error_deg"]))
        rows.append({"method": method, "report": str(path), "valid": not failures,
                     "failures": failures, "ranking_key": score,
                     "reference": report.get("reference")})
    valid = sorted((row for row in rows if row["valid"]), key=lambda row: row["ranking_key"])
    return {"schema": "sim_batch.kinematic_shortlist.v1", "kind": kind,
            "policy": "exploratory limits: 0.01rad joint, 0.03m penetration, 12rad/s speed; task contact then root/bone error",
            "candidates": rows, "send_to_sonic": [row["method"] for row in valid[:top_k]],
            "physical_success_claim": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("dual_sit", "single_stairs"), required=True)
    parser.add_argument("--report", action="append", required=True, metavar="METHOD=PATH")
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    reports = {}
    for item in args.report:
        method, separator, filename = item.partition("=")
        if not separator or not method or not filename or method in reports:
            parser.error(f"invalid or duplicate --report: {item}")
        reports[method] = Path(filename)
    result = rank(reports, args.kind, args.top_k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    if not result["send_to_sonic"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
