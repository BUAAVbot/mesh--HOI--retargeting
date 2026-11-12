"""Train only a serialized reference with an independently accepted physical replay.

This entry point never turns an optimization score into a training permission.
The acceptance certificate and all bound artifacts must still match on disk.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_certificate(certificate: Path, artifacts: dict[str, Path]) -> dict:
    report = json.loads(certificate.read_text(encoding="utf-8"))
    if report.get("schema") != "hdmi.trajopt.acceptance.v1":
        raise ValueError("Unknown physical acceptance certificate schema")
    if report.get("passed") is not True or report.get("accepted") is not True:
        raise ValueError("Physical optimization has not passed independent acceptance")
    if report.get("scope") != "continuous_nominal_simulation_trajectory":
        raise ValueError("Acceptance scope does not describe a continuous physical trajectory")
    replay = report.get("fresh_replay", {})
    for name in ("performed", "continuous", "no_midrollout_reset", "no_root_state_write",
                 "no_external_root_force", "serialized_controls_reloaded"):
        if replay.get(name) is not True:
            raise ValueError(f"Independent physical replay guarantee missing: {name}")
    if report.get("failed_checks"):
        raise ValueError("Certificate still contains failed checks")
    if report.get("batch_single_world_scene_alignment_verified") is not True:
        raise ValueError("Batch/single-world robot and scene alignment is not verified")
    if report.get("support_contact_filter_verified") is not True:
        raise ValueError("Contact support measurement does not exclude robot self-contact")
    if report.get("initial_state_changed") is True and report.get("original_fidelity_passed") is not True:
        raise ValueError("Contact-calibrated reference has not passed original-source fidelity checks")
    for field, path in artifacts.items():
        if not path.is_file() or report.get(field) != sha256(path):
            raise ValueError(f"Accepted artifact is missing or has changed: {field}: {path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--original-source", type=Path)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--comparison-record", type=Path, required=True)
    parser.add_argument("--gates", type=Path, required=True)
    parser.add_argument("--scene-audit", type=Path, required=True)
    parser.add_argument("--batch-audit", type=Path, required=True)
    parser.add_argument("--scene-geometry-audit", type=Path)
    parser.add_argument("--support-coverage-audit", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--envs", type=int, default=2048)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    artifacts = {
        "source_reference_sha256": args.source,
        "source_sidecar_sha256": args.source.with_suffix(".json"),
        "optimized_reference_sha256": args.motion,
        "controls_sha256": args.controls,
        "record_sha256": args.record,
        "comparison_record_sha256": args.comparison_record,
        "gates_sha256": args.gates,
        "scene_alignment_audit_sha256": args.scene_audit,
        "batch_invariance_audit_sha256": args.batch_audit,
        "model_sha256": args.model,
        "scene_sha256": args.scene,
        "checkpoint_sha256": args.checkpoint,
        "runtime_task_sha256": args.runtime_root / "task.py",
        "runtime_rl_cfg_sha256": args.runtime_root / "rl_cfg.py",
        "robot_asset_sha256": args.runtime_root / "assets" / "vita28_shoes.xml",
    }
    if args.original_source is not None:
        artifacts["original_source_reference_sha256"] = args.original_source
    scene_description = json.loads(args.scene.read_text(encoding="utf-8"))
    field = scene_description.get("ground_heightfield")
    if field:
        field_path = Path(field["path"])
        if not field_path.is_absolute():
            field_path = args.scene.parent / field_path
        if args.scene_geometry_audit is None or args.support_coverage_audit is None:
            parser.error("Heightfield training requires scene geometry and actual support coverage audits")
        artifacts["heightfield_asset_sha256"] = field_path
        artifacts["scene_geometry_audit_sha256"] = args.scene_geometry_audit
        artifacts["support_coverage_audit_sha256"] = args.support_coverage_audit
        artifacts["runtime_ground_scene_sha256"] = Path(__file__).with_name("ground_scene.py")
    verify_certificate(args.certificate, artifacts)
    import numpy as np

    with np.load(args.motion, allow_pickle=False) as motion:
        positions = motion["joint_pos"]
        fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
        if positions.ndim != 2 or positions.shape[1] != 28 or fps != 50.0:
            raise ValueError("Accepted motion must use Vita28 at 50 Hz")
        for key in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
                    "body_lin_vel_w", "body_ang_vel_w"):
            if len(motion[key]) != len(positions) or not np.isfinite(motion[key]).all():
                raise ValueError(f"Invalid accepted motion field: {key}")
        metadata = json.loads(args.source.with_suffix(".json").read_text(encoding="utf-8"))
        for name in ("joint_names", "body_names"):
            if name not in motion or tuple(map(str, motion[name])) != tuple(metadata[name]):
                raise ValueError(f"Optimized motion changed the frozen source ordering: {name}")
        duration = (len(positions) - 1) / fps
    receipt = {
        "schema": "hdmi.trajopt.training_gate.v1",
        "certificate": str(args.certificate.resolve()),
        "certificate_sha256": sha256(args.certificate),
        "verified_artifacts": {key: str(value.resolve()) for key, value in artifacts.items()},
        "duration_s": duration,
        "training_episode_length_s": duration + 2.0,
        "sampling_mode": "adaptive",
        "independent_world_env_spacing_m": 0.0,
        "training_and_evaluation_anchor_threshold_m": 0.25,
        "training_and_evaluation_ee_z_threshold_m": 0.25,
        "observation_noise": False,
        "goal": "PPO tracking expert for independently optimized physical reference",
        "is_hdmi_ppo_roa_teacher_student": False,
    }
    if args.output.exists():
        raise FileExistsError(f"Use a new training output directory: {args.output}")
    args.output.mkdir(parents=True)
    (args.output / "training_gate.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2), flush=True)
    if args.preflight_only:
        return

    sys.path.insert(0, str(args.runtime_root.resolve()))
    from dataclasses import replace
    from mjlab.scripts.train import TrainConfig, launch_training
    from mjlab.tasks.registry import register_mjlab_task
    from mjlab.rl import MjlabOnPolicyRunner
    from rl_cfg import make_runner_cfg
    from task import make_env_cfg
    from ground_scene import configure_ground_scene

    class WarmStartRunner(MjlabOnPolicyRunner):
        def __init__(self, env, train_cfg, log_dir=None, device="cpu", registry_name=None):
            super().__init__(env, train_cfg, log_dir, device)
            self.load(str(args.checkpoint.resolve()), load_cfg={"actor": True},
                      strict=True, map_location=device)

    def configuration(play: bool):
        cfg = make_env_cfg(str(args.motion.resolve()), str(args.scene.resolve()),
                           play=play, profile="precision_adaptive_feet")
        configure_ground_scene(cfg, args.scene.resolve())
        if field:
            cfg.sim.nconmax = max(cfg.sim.nconmax, 1024)
        cfg.scene.num_envs = 1 if play else args.envs
        # Each mjwarp world is physically independent. Shared coordinates keep
        # robot/reference and fixed mocap scene entities in the same frame.
        cfg.scene.env_spacing = 0.0
        cfg.episode_length_s = duration + 2.0
        cfg.sim.mujoco.ccd_iterations = 50
        cfg.commands["motion"].sampling_mode = "start" if play else "adaptive"
        cfg.terminations["anchor_pos"].params["threshold"] = 0.25
        cfg.terminations["ee_body_pos"].params["threshold"] = 0.25
        cfg.terminations["ee_body_pos"].params["body_names"] = (
            "left_ankle_roll_link", "right_ankle_roll_link",
            "left_wrist_pitch_link", "right_wrist_pitch_link")
        for group in cfg.observations.values():
            group.enable_corruption = False
        cfg.events.pop("encoder_bias", None)
        return cfg

    task_id = "Mjlab-Tracking-HDMI-Vita28-Verified-Trajopt"
    runner = make_runner_cfg()
    runner.experiment_name = "hdmi_vita28_gallery08_verified_trajopt"
    runner.run_name = args.output.name
    runner.max_iterations = args.iterations
    runner.save_interval = args.save_interval
    register_mjlab_task(task_id, configuration(False), configuration(True),
                       runner, runner_cls=WarmStartRunner)
    config = replace(TrainConfig.from_task(task_id), gpu_ids=[args.gpu])
    launch_training(task_id, config)


if __name__ == "__main__":
    main()
