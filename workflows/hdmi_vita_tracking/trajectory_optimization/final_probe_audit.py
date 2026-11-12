"""Failed single-probe audit; never substitutes a probe for fresh acceptance."""
import argparse
import json
from pathlib import Path

import numpy as np

from acceptance import load_npz, sha256, tracking_metrics
from support_coverage_audit import audit as coverage_audit


def main():
    p = argparse.ArgumentParser()
    for key in ("source", "original-source", "record", "model", "scene", "heightfield",
                "scene-geometry-audit", "probe-report", "projection-report", "gates", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    args = p.parse_args()
    record = load_npz(args.record)
    calibrated = tracking_metrics(load_npz(args.source), record)
    original = tracking_metrics(load_npz(args.original_source), record)
    gates = json.loads(args.gates.read_text())
    coverage = coverage_audit(args.record, args.model, args.heightfield)
    coverage_path = args.output.with_name("final_support_coverage_audit.json")
    coverage_path.write_text(json.dumps(coverage, indent=2))
    metrics = dict(calibrated)
    fields = dict(penetration_max_m="physics_penetration_m",
                  joint_limit_excess_max_rad="physics_joint_excess_rad", effort_ratio_max="physics_effort_ratio")
    for label, key in fields.items():
        value = np.asarray(record.get(key, []), float)
        metrics[label] = float(value.max()) if value.size and np.isfinite(value).all() else None
    support = np.asarray(record.get("physics_support_force_N", []), float)
    weight = float(np.asarray(record.get("robot_weight_N", 0.)).reshape(-1)[0])
    metrics["support_fraction"] = float(np.mean(support >= .1 * weight)) if support.size and weight > 0 else None
    original_checks = {key: bool(value <= gates["maximum"][key])
                       for key, value in original.items() if key in gates["maximum"]}
    physical_checks = {key: value is not None and bool(value <= gates["maximum"][key])
                       for key, value in metrics.items() if key in gates["maximum"]}
    physical_checks.update({key: metrics.get(key) is not None and bool(metrics[key] >= value)
                            for key, value in gates["minimum"].items()})
    physical_checks["support_contact_filter_verified"] = bool(np.asarray(record.get("support_filter_verified", False)).reshape(-1)[0]) and str(np.asarray(record.get("support_filter_semantics", "")).reshape(-1)[0]) == "exact_per_static_body_excludes_self"
    physical_checks["actual_support_coverage_verified"] = coverage["passed"]
    probe = json.loads(args.probe_report.read_text())
    projection = json.loads(args.projection_report.read_text())
    result = dict(
        schema="hdmi.trajopt.acceptance.v1", passed=False, accepted=False,
        scope="continuous_nominal_simulation_trajectory", initial_state_changed=True,
        source_reference_sha256=sha256(args.source), source_sidecar_sha256=sha256(args.source.with_suffix(".json")),
        original_source_reference_sha256=sha256(args.original_source), original_source_sidecar_sha256=sha256(args.original_source.with_suffix(".json")),
        source_contact_projection_sha256=sha256(args.projection_report),
        record_sha256=sha256(args.record), model_sha256=sha256(args.model), scene_sha256=sha256(args.scene),
        heightfield_asset_sha256=sha256(args.heightfield), scene_geometry_audit_sha256=sha256(args.scene_geometry_audit),
        support_coverage_audit_sha256=sha256(coverage_path), checkpoint_sha256=probe.get("checkpoint_sha256"),
        gates_sha256=sha256(args.gates), gates=gates,
        metrics=metrics, calibrated_source_fidelity_checks=physical_checks,
        original_fidelity_metrics=original, original_fidelity_checks=original_checks,
        support_contact_filter_verified=physical_checks["support_contact_filter_verified"],
        fresh_replay=dict(performed=False, continuous=True, serialized_controls_reloaded=False,
                          no_midrollout_reset=True, no_root_state_write=True, no_external_root_force=True,
                          reason="Actual single probe already violates physical/fidelity gates; no duplicate GPU replay was run."),
        first_failure_frame=probe["metrics"]["first_failure_frame"],
        source_projection_static_report=projection,
        failed_checks=["single_probe_only_fresh_acceptance_not_completed"]
                      + ["calibrated_" + key for key, passed in physical_checks.items() if not passed]
                      + ["original_fidelity_" + key for key, passed in original_checks.items() if not passed],
        rl_training_started=False, new_rl_policy_verified=False, hardware_verified=False,
        conclusion="Static contact/IK cleanup passed its own checks, but the actual frozen-actor physical rollout failed; RL remains gated off.")
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(passed=False, first_failure_frame=result["first_failure_frame"], metrics=metrics,
                          original_fidelity_metrics=original, coverage=coverage["counts"],
                          coverage_failed_checks=coverage["failed_checks"], failed_checks=result["failed_checks"])))


if __name__ == "__main__":
    main()
