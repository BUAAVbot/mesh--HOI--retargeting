"""Two fresh serialized-control physical replays and independent acceptance.

Never launches RL. Failed candidates retain recordings, references and reports
but their acceptance certificate cannot open the training gate.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from acceptance import acceptance, load_npz, names, sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runtime-root", type=Path, required=True)
    p.add_argument("--optimizer", type=Path, required=True)
    p.add_argument("--motion", type=Path, required=True)
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--controls", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gates", type=Path, required=True)
    p.add_argument("--scene-audit", type=Path, required=True)
    p.add_argument("--batch-audit", type=Path, required=True)
    p.add_argument("--original-source", type=Path)
    p.add_argument("--contact-projection-report", type=Path)
    p.add_argument("--heightfield-asset", type=Path)
    p.add_argument("--scene-geometry-audit", type=Path,
                   help="geometry_binding_manifest.json for the reconstructed terrain")
    p.add_argument("--gpu", type=int, default=4)
    args = p.parse_args()
    if (args.original_source is None) != (args.contact_projection_report is None):
        p.error("original-source and contact-projection-report must be provided together")
    if (args.heightfield_asset is None) != (args.scene_geometry_audit is None):
        p.error("heightfield-asset and scene-geometry-audit must be provided together")
    if args.output.exists():
        raise FileExistsError("Use a fresh candidate verification directory")
    occupied = subprocess.check_output([
        "nvidia-smi", f"--id={args.gpu}", "--query-gpu=memory.used",
        "--format=csv,noheader,nounits"], text=True).strip()
    if int(occupied) > 200:
        raise RuntimeError(f"GPU{args.gpu} no longer free: {occupied}MiB")
    args.output.mkdir(parents=True)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS="2")
    contract_fields = ("source_reference_sha256", "scene_sha256", "checkpoint_sha256", "model_sha256",
                       "runtime_task_sha256", "runtime_rl_cfg_sha256", "robot_asset_sha256")
    code_paths = [args.optimizer, args.optimizer.with_name("backend.py"), Path(__file__),
                  Path(__file__).with_name("acceptance.py")]
    ground_helper = None
    if args.heightfield_asset is not None:
        ground_helper = args.optimizer.with_name("ground_scene.py")
        training_ground_helper = Path(__file__).with_name("ground_scene.py")
        if not ground_helper.is_file() or not training_ground_helper.is_file() or sha256(ground_helper) != sha256(training_ground_helper):
            raise ValueError("Optimization/training reconstructed-ground helper source differs")
        code_paths += [ground_helper, training_ground_helper, Path(__file__).with_name("support_coverage_audit.py")]
    frozen_code = {str(path): sha256(path) for path in code_paths}
    reports = []
    for label in ("fresh_a", "fresh_b"):
        output = args.output / label
        command = [sys.executable, str(args.optimizer), "--root", str(args.runtime_root),
                   "--motion", str(args.motion), "--scene", str(args.scene),
                   "--checkpoint", str(args.checkpoint), "--controls", str(args.controls),
                   "--output", str(output), "--instrument"]
        with (args.output / f"{label}.log").open("x") as log:
            subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                           check=True, stdin=subprocess.DEVNULL)
        if any(sha256(Path(path)) != digest for path, digest in frozen_code.items()):
            raise ValueError("Verification/optimizer source changed during fresh replays")
        reports.append(json.loads((output / "report.json").read_text()))
        print(json.dumps(dict(stage=label, metrics=reports[-1]["metrics"])), flush=True)
    for key in contract_fields:
        if reports[0][key] != reports[1][key]:
            raise ValueError(f"Fresh runs changed {key}")
    source, record = load_npz(args.motion), load_npz(args.output / "fresh_a/physical_record.npz")
    if names(source, "joint_names") != names(record, "joint_names") or names(source, "body_names") != names(record, "body_names"):
        raise ValueError("Physical recording order differs from frozen source metadata")
    exported = {key: record[key] for key in (
        "joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w",
        "joint_names", "body_names", "fps")}
    reference = args.output / "optimized_reference.npz"
    np.savez_compressed(reference, **exported)
    reference.with_suffix(".json").write_text(json.dumps(dict(
        status="physical_candidate_acceptance_pending", fps=50., frames=len(record["joint_pos"]),
        joint_names=list(names(record, "joint_names")), body_names=list(names(record, "body_names")),
        source_reference_sha256=sha256(args.motion),
        semantics="actual_integrated_states_not_PD_targets"), indent=2))
    manifest = dict(
        **{key: reports[0][key] for key in contract_fields},
        source_sidecar_sha256=sha256(args.motion.with_suffix(".json")),
        record_sha256=sha256(args.output / "fresh_a/physical_record.npz"),
        controls_sha256=sha256(args.controls), optimized_reference_sha256=sha256(reference),
        source_clock_unchanged=True, robot_weight_N=reports[0]["robot_weight_N"],
        scene_alignment_audit_sha256=sha256(args.scene_audit),
        batch_invariance_audit_sha256=sha256(args.batch_audit),
        batch_single_world_scene_alignment_verified=True,
        verification_code_sha256=frozen_code,
        initial_state_changed=args.original_source is not None,
        fresh_replay=dict(performed=True, continuous=True, serialized_controls_reloaded=True,
                          no_midrollout_reset=True, no_root_state_write=True,
                          no_external_root_force=True,
                          comparison_record_sha256=sha256(args.output / "fresh_b/physical_record.npz")))
    if args.original_source is not None:
        manifest.update(original_source_reference_sha256=sha256(args.original_source),
                        original_source_sidecar_sha256=sha256(args.original_source.with_suffix(".json")),
                        source_contact_projection_sha256=sha256(args.contact_projection_report))
    coverage_path = None
    if args.heightfield_asset is not None:
        from support_coverage_audit import audit as coverage_audit
        coverage = coverage_audit(args.output / "fresh_a/physical_record.npz",
                                  args.output / "fresh_a/compiled_model.mjb", args.heightfield_asset)
        coverage_path = args.output / "support_coverage_audit.json"
        coverage_path.write_text(json.dumps(coverage, indent=2))
        manifest.update(heightfield_asset_sha256=sha256(args.heightfield_asset),
                        scene_geometry_audit_sha256=sha256(args.scene_geometry_audit),
                        support_coverage_audit_sha256=sha256(coverage_path),
                        runtime_ground_scene_sha256=sha256(ground_helper))
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    certificate = acceptance(
        args.motion, args.output / "fresh_a/physical_record.npz", args.controls, reference,
        manifest_path, args.gates, args.output / "fresh_b/physical_record.npz",
        args.scene_audit, args.batch_audit, args.original_source, args.contact_projection_report,
        args.heightfield_asset, args.scene_geometry_audit, coverage_path)
    (args.output / "acceptance.json").write_text(json.dumps(certificate, indent=2))
    print(json.dumps(dict(stage="independent_acceptance", passed=certificate["passed"],
                          failed_checks=certificate["failed_checks"], metrics=certificate["metrics"],
                          fresh_replay=certificate["fresh_replay"])), flush=True)


if __name__ == "__main__":
    main()
