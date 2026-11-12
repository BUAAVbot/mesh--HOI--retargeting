"""Independent acceptance of a continuous HDMI Vita28 physical recording.

Tracking errors are ALWAYS measured against the frozen pre-optimization source,
never against the newly exported physical reference. This script does not run a
policy and cannot turn an optimizer completion into a success certificate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


SCHEMA = "hdmi.trajopt.acceptance.v1"
TRACK_BODIES = (
    "base_link", "waist_pitch_link", "left_hip_pitch_link", "right_hip_pitch_link",
    "left_knee_link", "right_knee_link", "left_ankle_roll_link", "right_ankle_roll_link",
    "left_shoulder_pitch_link", "right_shoulder_pitch_link", "left_elbow_link",
    "right_elbow_link", "left_wrist_pitch_link", "right_wrist_pitch_link",
)
EE_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link",
             "left_wrist_pitch_link", "right_wrist_pitch_link")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        result = {key: data[key] for key in data.files}
    # prepare_motion.py stores names in a same-stem JSON receipt, not in the
    # six-field mjlab NPZ. Use that explicit ordering instead of guessing it.
    sidecar = Path(path).with_suffix(".json")
    metadata = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.is_file() else {}
    for key in ("joint_names", "body_names"):
        if key in result and key in metadata and tuple(map(str, result[key])) != tuple(map(str, metadata[key])):
            raise ValueError(f"NPZ and sidecar disagree on explicit {key} ordering")
        if key not in result and key in metadata:
            result[key] = np.asarray(metadata[key])
    return result


def names(data, key):
    return tuple(str(value) for value in np.asarray(data[key]).reshape(-1))


def angular_distance(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    an = np.linalg.norm(a, axis=-1, keepdims=True)
    bn = np.linalg.norm(b, axis=-1, keepdims=True)
    if np.any(an < 1e-8) or np.any(bn < 1e-8):
        raise ValueError("Zero quaternion in physical recording or target")
    dot = np.sum((a / an) * (b / bn), axis=-1)
    return 2.0 * np.arccos(np.clip(np.abs(dot), 0.0, 1.0))


def align_arrays(source, record):
    sj = names(source, "joint_names")
    rj = names(record, "joint_names")
    sb = names(source, "body_names")
    rb = names(record, "body_names")
    if len(set(sj)) != len(sj) or len(set(rj)) != len(rj):
        raise ValueError("Duplicate joint names")
    if set(sj) != set(rj):
        raise ValueError("Physical/source joint sets differ")
    if not all(name in sb and name in rb for name in TRACK_BODIES):
        raise ValueError("Required tracking bodies missing")
    jp = np.asarray(record["joint_pos"], np.float64)[:, [rj.index(n) for n in sj]]
    source_jp = np.asarray(source["joint_pos"], np.float64)
    rp = np.asarray(record["body_pos_w"], np.float64)[:, [rb.index(n) for n in TRACK_BODIES]]
    sp = np.asarray(source["body_pos_w"], np.float64)[:, [sb.index(n) for n in TRACK_BODIES]]
    rq = np.asarray(record["body_quat_w"], np.float64)[:, [rb.index(n) for n in TRACK_BODIES]]
    sq = np.asarray(source["body_quat_w"], np.float64)[:, [sb.index(n) for n in TRACK_BODIES]]
    if jp.shape != source_jp.shape or rp.shape != sp.shape or rq.shape != sq.shape:
        raise ValueError("Recording must retain the exact frozen source time grid")
    sfps = float(np.asarray(source["fps"]).reshape(-1)[0])
    rfps = float(np.asarray(record["fps"]).reshape(-1)[0])
    if abs(sfps - 50.0) > 1e-6 or abs(rfps - sfps) > 1e-6:
        raise ValueError("Source and physical recording must both be 50 Hz")
    return source_jp, jp, sp, rp, sq, rq, sfps


def tracking_metrics(source, record):
    sjp, rjp, sp, rp, sq, rq, fps = align_arrays(source, record)
    finite = all(np.isfinite(v).all() for v in (sjp, rjp, sp, rp, sq, rq))
    if not finite:
        raise ValueError("Non-finite state/reference")
    root_error = np.linalg.norm(rp[:, 0] - sp[:, 0], axis=-1)
    world_body_error = np.linalg.norm(rp - sp, axis=-1)
    # Root-relative body shape is a separate metric; root tracking remains a
    # world-frame constraint, so a translation cannot hide task displacement.
    relative_body_error = np.linalg.norm(
        (rp - rp[:, :1]) - (sp - sp[:, :1]), axis=-1)
    ee_idx = [TRACK_BODIES.index(name) for name in EE_BODIES]
    ee_z = np.abs(rp[:, ee_idx, 2] - sp[:, ee_idx, 2])
    root_angle = angular_distance(rq[:, 0], sq[:, 0])
    body_angle = angular_distance(rq, sq)
    joint_error = rjp - sjp
    return dict(
        finite=True, frames=int(len(rjp)), fps=fps,
        duration_s=float((len(rjp) - 1) / fps),
        root_max_m=float(root_error.max()),
        root_rmse_m=float(np.sqrt(np.mean(root_error ** 2))),
        final_root_error_m=float(root_error[-1]),
        body_world_max_m=float(world_body_error.max()),
        body_relative_max_m=float(relative_body_error.max()),
        body_relative_rmse_m=float(np.sqrt(np.mean(relative_body_error ** 2))),
        ee_z_max_m=float(ee_z.max()),
        root_orientation_rmse_rad=float(np.sqrt(np.mean(root_angle ** 2))),
        final_root_orientation_rad=float(root_angle[-1]),
        body_orientation_rmse_rad=float(np.sqrt(np.mean(body_angle ** 2))),
        joint_rmse_rad=float(np.sqrt(np.mean(joint_error ** 2))),
        joint_error_max_rad=float(np.abs(joint_error).max()),
        joint_speed_max_rad_s=float(np.abs(np.asarray(record["joint_vel"])).max()),
        target="frozen_source_reference_not_optimized_reference",
        first_root_gate_failure_frame=(int(np.flatnonzero(root_error > .25)[0])
                                       if np.any(root_error > .25) else None),
        first_ee_gate_failure_frame=(int(np.flatnonzero(np.any(ee_z > .25, axis=-1))[0])
                                     if np.any(ee_z > .25) else None),
    )


def acceptance(source_path, record_path, controls_path, optimized_reference_path,
               manifest_path, gates_path, comparison_record_path=None,
               scene_audit_path=None, batch_audit_path=None,
               original_source_path=None, contact_projection_report_path=None,
               heightfield_asset_path=None, scene_geometry_audit_path=None,
               support_coverage_audit_path=None):
    source = load_npz(source_path)
    record = load_npz(record_path)
    exported = load_npz(optimized_reference_path)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    gates = json.loads(Path(gates_path).read_text(encoding="utf-8"))
    metrics = tracking_metrics(source, record)
    checks = {"finite": metrics["finite"], "frozen_gates": gates.get("frozen") is True}
    frames = metrics["frames"]
    for key, width in (("qpos", 35), ("qvel", 34)):
        value = np.asarray(record.get(key, []), np.float64)
        checks[key + "_complete_finite"] = value.shape == (frames, width) and bool(np.isfinite(value).all())
    times = np.asarray(record.get("times", []), np.float64)
    checks["physical_time_grid"] = (
        times.shape == (frames,) and np.allclose(times, np.arange(frames) / 50., rtol=0, atol=1e-8))
    s_joint, r_joint, s_body, r_body, _, _, _ = align_arrays(source, record)
    checks["initial_root_matches_command_reference"] = np.allclose(s_body[0, 0], r_body[0, 0], rtol=0, atol=1e-5)
    checks["initial_joint_matches_command_reference"] = np.allclose(s_joint[0], r_joint[0], rtol=0, atol=1e-5)
    if checks["qpos_complete_finite"]:
        rb = names(record, "body_names")
        checks["qpos_matches_joint_state"] = np.allclose(record["qpos"][:, 7:], record["joint_pos"], rtol=0, atol=1e-6)
        checks["qpos_matches_root_position"] = np.allclose(
            record["qpos"][:, :3], record["body_pos_w"][:, rb.index("base_link")], rtol=0, atol=1e-6)
        checks["qpos_matches_root_orientation"] = bool(np.max(angular_distance(
            record["qpos"][:, 3:7], record["body_quat_w"][:, rb.index("base_link")])) < 1e-5)
    expected_hashes = {
        "source_reference_sha256": sha256(source_path),
        "source_sidecar_sha256": sha256(Path(source_path).with_suffix(".json")),
        "record_sha256": sha256(record_path),
        "controls_sha256": sha256(controls_path),
        "optimized_reference_sha256": sha256(optimized_reference_path),
    }
    for key, value in expected_hashes.items():
        checks[key + "_matches"] = manifest.get(key) == value
    for key in ("source_reference_sha256", "scene_sha256", "checkpoint_sha256"):
        checks["record_" + key + "_matches"] = str(np.asarray(record.get(key, "")).reshape(-1)[0]) == manifest.get(key)
    checks["not_self_target"] = expected_hashes["source_reference_sha256"] != expected_hashes["optimized_reference_sha256"]
    runtime_fields = ("runtime_task_sha256", "runtime_rl_cfg_sha256", "robot_asset_sha256")
    for key in ("model_sha256", "scene_sha256", "checkpoint_sha256") + runtime_fields:
        value = manifest.get(key)
        checks[key + "_present"] = isinstance(value, str) and len(value) == 64
    audit_hashes = {}
    for label, path in (("scene_alignment_audit_sha256", scene_audit_path),
                        ("batch_invariance_audit_sha256", batch_audit_path)):
        available = path is not None and Path(path).is_file()
        checks[label + "_present"] = available
        audit_hashes[label] = sha256(path) if available else None
        checks[label + "_matches"] = available and manifest.get(label) == audit_hashes[label]
    scene_alignment_verified = False
    if scene_audit_path is not None and Path(scene_audit_path).is_file():
        scene_payload = json.loads(Path(scene_audit_path).read_text())
        if isinstance(scene_payload, dict):
            scene_audit = scene_payload["rows"]
            checks["scene_audit_binds_current_scene"] = scene_payload.get("scene_sha256") == manifest.get("scene_sha256")
            checks["scene_audit_binds_runtime_task"] = scene_payload.get("runtime_task_sha256") == manifest.get("runtime_task_sha256")
        else:
            scene_audit = scene_payload
            checks["scene_audit_binds_current_scene"] = False
            checks["scene_audit_binds_runtime_task"] = False
        zero_spacing = [row for row in scene_audit if row["count"] > 1 and row["spacing"] == 0.]
        single = [row for row in scene_audit if row["count"] == 1]
        scene_alignment_verified = bool(zero_spacing and single and all(
            max(box["position_error_m"]) < 1e-5
            for row in zero_spacing + single for box in row["boxes"]))
        checks["all_static_geometries_audited"] = bool(zero_spacing and single and all(
            row.get("static_geometries") and all(max(g["position_error_m"]) < 1e-5
                                                  for g in row["static_geometries"])
            for row in zero_spacing + single))
    checks["batch_single_world_scene_alignment_verified"] = (
        scene_alignment_verified and manifest.get("batch_single_world_scene_alignment_verified") is True)
    if batch_audit_path is not None and Path(batch_audit_path).is_file():
        batch_audit = json.loads(Path(batch_audit_path).read_text())
        single = [row for row in batch_audit if row["count"] == 1]
        multiple = [row for row in batch_audit if row["count"] > 1]
        origin_aligned = bool(single and multiple and all(
            np.max(np.abs(np.asarray(row["origins"]))) < 1e-6 for row in single + multiple))
        first_failure_consistent = bool(single and multiple and all(
            np.all(np.asarray(row["metrics"]["first_failure_frame"]) == single[0]["metrics"]["first_failure_frame"][0])
            for row in multiple))
        checks["batch_audit_origins_aligned"] = origin_aligned
        checks["batch_audit_baseline_first_failure_consistent"] = first_failure_consistent
    else:
        checks["batch_audit_origins_aligned"] = False
        checks["batch_audit_baseline_first_failure_consistent"] = False
    replay = dict(manifest.get("fresh_replay", {}))
    for key in ("performed", "continuous", "serialized_controls_reloaded",
                "no_midrollout_reset", "no_root_state_write", "no_external_root_force"):
        checks["fresh_" + key] = replay.get(key) is True
    checks["fixed_source_clock"] = manifest.get("source_clock_unchanged") is True
    checks["record_no_resets"] = int(np.asarray(record.get("reset_count", -1)).reshape(-1)[0]) == 0
    checks["record_no_root_state_writes"] = int(np.asarray(record.get("state_writes_after_initialization", -1)).reshape(-1)[0]) == 0
    checks["record_continuous_physics"] = bool(np.asarray(record.get("continuous_physics", False)).reshape(-1)[0])
    controls = load_npz(controls_path)
    saved_residual = np.asarray(controls.get("residual_actions", []))
    executed_residual = np.asarray(record.get("residual_actions", []))
    checks["serialized_residual_executed"] = (
        saved_residual.shape == executed_residual.shape == (frames - 1, 28)
        and np.allclose(saved_residual, executed_residual, rtol=0, atol=1e-7))
    checks["controls_joint_order"] = (
        "joint_names" in controls and names(controls, "joint_names") == names(record, "joint_names"))
    for key in ("reference_blend", "clip_control_targets_to_limits"):
        if key in controls:
            checks["serialized_" + key + "_executed"] = (
                key in record and controls[key].shape == record[key].shape
                and np.allclose(controls[key], record[key], rtol=0, atol=1e-7))
    # Exported RL reference must be the actual physical states in the record,
    # including all bodies and ordering, not the optimized PD target sequence.
    for key in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
                "body_lin_vel_w", "body_ang_vel_w"):
        checks["export_matches_record_" + key] = (
            key in exported and key in record and exported[key].shape == record[key].shape
            and np.allclose(exported[key], record[key], rtol=0, atol=1e-6))
    physics_fields = {
        "penetration_max_m": "physics_penetration_m",
        "joint_limit_excess_max_rad": "physics_joint_excess_rad",
        "effort_ratio_max": "physics_effort_ratio",
    }
    for metric, field in physics_fields.items():
        values = np.asarray(record.get(field, []), np.float64)
        checks[field + "_present_finite"] = values.size > 0 and bool(np.isfinite(values).all())
        checks[field + "_all_substeps"] = values.shape == (frames - 1, 4)
        metrics[metric] = float(values.max()) if checks[field + "_present_finite"] else None
    support = np.asarray(record.get("physics_support_force_N", []), np.float64)
    checks["support_force_present_finite"] = support.size > 0 and bool(np.isfinite(support).all())
    checks["support_force_all_substeps"] = support.size == (frames - 1) * 4
    support_filter_verified = bool(np.asarray(record.get("support_filter_verified", False)).reshape(-1)[0])
    support_filter_semantics = str(np.asarray(record.get("support_filter_semantics", "")).reshape(-1)[0])
    checks["support_contact_filter_verified"] = (
        support_filter_verified and support_filter_semantics == "exact_per_static_body_excludes_self")
    if checks["support_force_present_finite"]:
        # Force arrays must already contain scalar total support per substep.
        force = support.reshape(-1)
        weight = float(manifest.get("robot_weight_N", 0))
        checks["robot_weight_present"] = weight > 0
        fraction = float(np.mean(force >= .10 * weight)) if weight > 0 else 0.
        metrics["support_fraction"] = fraction
        metrics["support_force_peak_N"] = float(force.max())
    else:
        metrics["support_fraction"] = None
    for metric, limit in gates.get("maximum", {}).items():
        value = metrics.get(metric)
        checks[metric] = value is not None and np.isfinite(value) and value <= float(limit)
    for metric, limit in gates.get("minimum", {}).items():
        value = metrics.get(metric)
        checks[metric] = value is not None and np.isfinite(value) and value >= float(limit)
    initial_state_changed = manifest.get("initial_state_changed") is True
    original_metrics, original_checks, projection_hashes = None, {}, {}
    if initial_state_changed:
        available = (original_source_path is not None and Path(original_source_path).is_file()
                     and contact_projection_report_path is not None and Path(contact_projection_report_path).is_file())
        checks["original_source_and_contact_projection_present"] = available
        if available:
            projection_hashes = dict(
                original_source_reference_sha256=sha256(original_source_path),
                original_source_sidecar_sha256=sha256(Path(original_source_path).with_suffix(".json")),
                source_contact_projection_sha256=sha256(contact_projection_report_path))
            for key, digest in projection_hashes.items():
                checks[key + "_matches"] = manifest.get(key) == digest
            checks["calibrated_and_original_source_distinct"] = (
                projection_hashes["original_source_reference_sha256"] != expected_hashes["source_reference_sha256"])
            original_metrics = tracking_metrics(load_npz(original_source_path), record)
            # Apply the IDENTICAL fidelity thresholds to the original frozen
            # source as well. A physically easier calibrated command must not
            # silently turn a different task into an accepted reproduction.
            for metric, limit in gates.get("maximum", {}).items():
                if metric in original_metrics:
                    value = original_metrics[metric]
                    original_checks[metric] = value is not None and np.isfinite(value) and value <= float(limit)
            checks.update({"original_fidelity_" + key: value for key, value in original_checks.items()})
    else:
        checks["no_unreported_original_source"] = original_source_path is None and contact_projection_report_path is None
    heightfield_hashes = {}
    if heightfield_asset_path is not None or manifest.get("heightfield_asset_sha256"):
        for key, path in (("heightfield_asset_sha256", heightfield_asset_path),
                          ("scene_geometry_audit_sha256", scene_geometry_audit_path),
                          ("support_coverage_audit_sha256", support_coverage_audit_path)):
            present = path is not None and Path(path).is_file()
            checks[key + "_present"] = present
            heightfield_hashes[key] = sha256(path) if present else None
            checks[key + "_matches"] = present and manifest.get(key) == heightfield_hashes[key]
        helper_hash = manifest.get("runtime_ground_scene_sha256")
        checks["runtime_ground_scene_sha256_present"] = isinstance(helper_hash, str) and len(helper_hash) == 64
        heightfield_hashes["runtime_ground_scene_sha256"] = helper_hash
        if scene_geometry_audit_path is not None and Path(scene_geometry_audit_path).is_file():
            geometry = json.loads(Path(scene_geometry_audit_path).read_text())
            checks["heightfield_geometry_verified"] = (
                geometry.get("schema") == "hdmi.heightfield.audit_binding.v1"
                and bool(geometry.get("checks")) and all(value is True for value in geometry["checks"].values())
                and geometry.get("files_sha256", {}).get("heightfield.npz") == heightfield_hashes.get("heightfield_asset_sha256")
                and geometry.get("files_sha256", {}).get("scene.json") == manifest.get("scene_sha256"))
        else:
            checks["heightfield_geometry_verified"] = False
        if support_coverage_audit_path is not None and Path(support_coverage_audit_path).is_file():
            coverage = json.loads(Path(support_coverage_audit_path).read_text())
            counts = coverage.get("counts", {})
            checks["actual_loadbearing_support_coverage_verified"] = (
                coverage.get("schema") == "hdmi.trajopt.support_coverage.v1" and coverage.get("passed") is True
                and not coverage.get("failed_checks")
                and coverage.get("record_sha256") == expected_hashes["record_sha256"]
                and coverage.get("model_sha256") == manifest.get("model_sha256")
                and coverage.get("heightfield_asset_sha256") == heightfield_hashes.get("heightfield_asset_sha256")
                and counts.get("total", 0) > 0 and counts.get("unknown") == 0 and counts.get("outside") == 0)
        else:
            checks["actual_loadbearing_support_coverage_verified"] = False
    # Recompute reproducibility from both complete records, rather than trust
    # a scalar written by the optimization process into its own manifest.
    comparison_sha = None
    state_difference = None
    checks["comparison_record_present"] = comparison_record_path is not None and Path(comparison_record_path).is_file()
    if checks["comparison_record_present"]:
        comparison_sha = sha256(comparison_record_path)
        checks["comparison_record_sha256_matches"] = replay.get("comparison_record_sha256") == comparison_sha
        comparison = load_npz(comparison_record_path)
        differences = []
        replay["state_difference_by_field"] = {}
        for key in ("qpos", "qvel"):
            first = np.asarray(record.get(key, []), np.float64)
            second = np.asarray(comparison.get(key, []), np.float64)
            checks["comparison_" + key + "_complete"] = (
                first.shape == second.shape and first.size > 0 and bool(np.isfinite(second).all()))
            if checks["comparison_" + key + "_complete"]:
                difference = float(np.abs(first - second).max())
                differences.append(difference)
                replay["state_difference_by_field"][key] = difference
        if len(differences) == 2:
            state_difference = max(differences)
    replay["max_state_difference"] = state_difference
    checks["fresh_replay_consistency"] = (
        state_difference is not None and np.isfinite(state_difference)
        and state_difference <= float(gates["fresh_replay_state_tolerance"]))
    passed = bool(all(checks.values()))
    return dict(schema=SCHEMA, passed=passed, accepted=passed,
                scope="continuous_nominal_simulation_trajectory",
                **expected_hashes,
                **{k: manifest.get(k) for k in ("model_sha256", "scene_sha256", "checkpoint_sha256")},
                **{k: manifest.get(k) for k in runtime_fields},
                verification_code_sha256=manifest.get("verification_code_sha256", {}),
                gates_sha256=sha256(gates_path), gates=gates,
                comparison_record_sha256=comparison_sha,
                **audit_hashes,
                **projection_hashes,
                **heightfield_hashes,
                initial_state_changed=initial_state_changed,
                original_fidelity_metrics=original_metrics,
                original_fidelity_checks=original_checks,
                support_filter_verified=support_filter_verified,
                support_filter_semantics=support_filter_semantics,
                support_contact_filter_verified=checks["support_contact_filter_verified"],
                batch_single_world_scene_alignment_verified=scene_alignment_verified,
                fresh_replay=replay, metrics=metrics, checks=checks,
                failed_checks=[key for key, value in checks.items() if not value],
                teacher_student_algorithm_verified=False,
                new_rl_policy_verified=False, hardware_verified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--controls", type=Path)
    parser.add_argument("--optimized-reference", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--gates", type=Path)
    parser.add_argument("--comparison-record", type=Path,
                        help="Second fresh count=1 serialized-control replay; required for acceptance")
    parser.add_argument("--scene-audit", type=Path,
                        help="Independent fixed-entity coordinates audit; required for acceptance")
    parser.add_argument("--batch-audit", type=Path,
                        help="Spacing-corrected single/batch baseline audit; required for acceptance")
    parser.add_argument("--original-source", type=Path,
                        help="Frozen pre-calibration source; required when initial state changed")
    parser.add_argument("--contact-projection-report", type=Path,
                        help="Explicit source-contact calibration provenance when initial state changed")
    parser.add_argument("--heightfield-asset", type=Path)
    parser.add_argument("--scene-geometry-audit", type=Path)
    parser.add_argument("--support-coverage-audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true",
                        help="Report frozen-source fidelity and recorded physics; never accepts")
    args = parser.parse_args()
    if args.audit_only:
        source, record = load_npz(args.source), load_npz(args.record)
        metrics = tracking_metrics(source, record)
        physics = {}
        for key in ("physics_penetration_m", "physics_joint_excess_rad", "physics_effort_ratio",
                    "physics_support_force_N"):
            values = np.asarray(record.get(key, []), np.float64)
            physics[key] = {"present": values.size > 0, "shape": list(values.shape)}
            if values.size:
                physics[key].update(finite=bool(np.isfinite(values).all()),
                                    minimum=float(values.min()), maximum=float(values.max()),
                                    first_control_frame=values[:1].tolist())
                if key in ("physics_penetration_m", "physics_joint_excess_rad"):
                    bound = .012 if key == "physics_penetration_m" else .02
                    flat = values.reshape(-1)
                    indices = np.flatnonzero(flat > bound)
                    physics[key].update(
                        gate=bound,
                        first_gate_failure_physics_substep=int(indices[0]) if len(indices) else None,
                        first_gate_failure_control_frame=int(indices[0] // 4 + 1) if len(indices) else None,
                        first_80_control_frame_maximum=float(flat[:80 * 4].max()))
        result = dict(schema="hdmi.trajopt.audit.v1", passed=False, accepted=False,
                      source_reference_sha256=sha256(args.source), record_sha256=sha256(args.record),
                      metrics=metrics, physics=physics, failed_checks=["audit_only_not_acceptance"])
    else:
        required = ("controls", "optimized_reference", "manifest", "gates", "comparison_record", "scene_audit", "batch_audit")
        if any(getattr(args, key) is None for key in required):
            parser.error("Acceptance requires controls, optimized-reference, manifest, gates, comparison-record, scene-audit and batch-audit")
        result = acceptance(args.source, args.record, args.controls, args.optimized_reference,
                            args.manifest, args.gates, args.comparison_record,
                            args.scene_audit, args.batch_audit,
                            args.original_source, args.contact_projection_report,
                            args.heightfield_asset, args.scene_geometry_audit,
                            args.support_coverage_audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "failed_checks": result["failed_checks"],
                      "metrics": result["metrics"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
