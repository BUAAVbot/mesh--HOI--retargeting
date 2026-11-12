#!/usr/bin/env python3
"""Run the official SPIDER optimizer on an explicit Vita trajectory.

This file is deliberately an adapter, not a reimplementation of SPIDER.  It
patches only the official runner's dataset/model loading hooks so that a
standalone ``.mjb`` scene and the experiment's canonical ``.npz`` can be used.
The optimizer, MuJoCo-Warp rollout, reward, sampling and MPC loop remain the
upstream implementation.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from common import configure_position_pd as _configure_shared_position_pd


_FREQ = 20.0 * math.pi
_ARM_LZ5735 = 35.0**2 * 1.23e-5
_ARM_RP70H = 20.0**2 * 3.975e-5
_ANKLE_ROLL_RATIO = 0.02175 / 0.05


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_model(path: Path):
    import mujoco

    if path.suffix.lower() == ".mjb":
        return mujoco.MjModel.from_binary_path(str(path))
    return mujoco.MjModel.from_xml_path(str(path))


def _joint_pd(name: str, armature: float) -> tuple[float, float]:
    """Return the HDMI/MJLab Vita28 position-servo gains."""
    key = name.lower()
    if "ankle_roll" in key or ("ankle" in key and "roll" in key):
        ratio2 = _ANKLE_ROLL_RATIO**2
        return 40.0 * ratio2, 2.0 * ratio2
    if "ankle" in key:
        return 40.0, 2.0
    if "wrist" in key:
        return 20.0, 1.0
    if "shoulder" in key or "elbow" in key:
        return _ARM_RP70H * _FREQ**2, 2.0 * _ARM_RP70H * _FREQ
    if "waist" in key:
        return _ARM_LZ5735 * _FREQ**2, 2.0 * _ARM_LZ5735 * _FREQ
    # Hips and knees use the armature stored in the shipped Vita model.
    return armature * _FREQ**2, 2.0 * armature * _FREQ


def configure_position_pd(model: Any, sim_dt: float) -> list[dict[str, Any]]:
    """Convert the compiled model's torque motors to HDMI position servos.

    MuJoCo's affine actuator force is ``kp * ctrl - kp * q - kd * qdot``.
    SPIDER therefore optimizes joint-position targets, matching the MJLab HDMI
    task rather than treating target angles as torques.
    """
    model.opt.timestep = float(sim_dt)
    import mujoco

    return _configure_shared_position_pd(model, mujoco)["actuators"]


def _validate_reference(path: Path, model: Any) -> dict[str, tuple[int, ...]]:
    with np.load(path, allow_pickle=False) as data:
        required = {"qpos", "qvel", "ctrl"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"canonical reference lacks {sorted(missing)}")
        shapes = {key: tuple(data[key].shape) for key in data.files}
        if shapes["qpos"][1] != model.nq:
            raise ValueError(f"qpos width {shapes['qpos'][1]} != model.nq {model.nq}")
        if shapes["qvel"][1] != model.nv:
            raise ValueError(f"qvel width {shapes['qvel'][1]} != model.nv {model.nv}")
        if shapes["ctrl"][1] != model.nu:
            raise ValueError(f"ctrl width {shapes['ctrl'][1]} != model.nu {model.nu}")
        frames = shapes["qpos"][0]
        if frames < 3 or shapes["qvel"][0] != frames or shapes["ctrl"][0] != frames:
            raise ValueError("qpos, qvel and ctrl must have the same T >= 3")
    return shapes


def _load_upstream_runner(repo: Path) -> ModuleType:
    sys.path.insert(0, str(repo))
    # The upstream runner imports an offline pandas evaluator at module import
    # time.  It is object-specific and is replaced below by our frozen humanoid
    # validator, so avoid making pandas a runtime dependency of the optimizer.
    evaluator_name = "spider.postprocess.get_success_rate"
    if evaluator_name not in sys.modules:
        evaluator = ModuleType(evaluator_name)
        evaluator.compute_object_tracking_error = lambda *unused, **unused_kw: {
            "obj_pos_err": float("nan"), "obj_quat_err": float("nan")
        }
        sys.modules[evaluator_name] = evaluator
    # The official module also imports optional OpenCV/Rerun/Viser viewers even
    # when all visualization flags are false.  Supply the same disabled-viewer
    # behavior without importing those optional UI stacks.
    viewer_name = "spider.viewers"
    if viewer_name not in sys.modules:
        viewers = ModuleType(viewer_name)

        class _DummyViewer:
            user_scn = None

            def is_running(self):
                return True

            def sync(self):
                return None

        @contextmanager
        def _run_viewer():
            yield _DummyViewer()

        viewers.setup_viewer = lambda *unused, **unused_kw: _run_viewer
        viewers.setup_renderer = lambda *unused, **unused_kw: None
        viewers.update_viewer = lambda *unused, **unused_kw: None
        viewers.log_frame = lambda *unused, **unused_kw: None
        viewers.render_image = lambda *unused, **unused_kw: None
        sys.modules[viewer_name] = viewers
    runner_path = repo / "examples" / "run_mjwp.py"
    spec = importlib.util.spec_from_file_location("vita_spider_upstream_runner", runner_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import upstream runner at {runner_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _repo_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown (source snapshot without .git)"


def _explicit_process_config(config: Any, model_path: Path, data_path: Path, output: Path):
    from spider.config import compute_noise_schedule, compute_steps

    model = _load_model(model_path)
    configure_position_pd(model, config.sim_dt)
    config = compute_steps(config)
    trace_steps = int(round(config.trace_dt / config.sim_dt))
    if not np.isclose(config.trace_dt, trace_steps * config.sim_dt, atol=1e-3):
        raise ValueError("trace_dt must be divisible by sim_dt")
    config.model_path = str(model_path)
    config.data_path = str(data_path)
    config.output_dir = str(output)
    config.nq_obj = 0
    config.nq = int(model.nq)
    config.nv = int(model.nv)
    config.nu = int(model.nu)
    config.npair = int(model.npair)
    config.contact_site_ids = []
    return compute_noise_schedule(config)


def _normalize_upstream_output(output: Path, reference_path: Path) -> dict[str, Any]:
    raw_path = output / "trajectory_mjwp.npz"
    if not raw_path.exists():
        raise FileNotFoundError(f"upstream SPIDER output missing: {raw_path}")
    with np.load(raw_path, allow_pickle=False) as raw:
        qpos = np.asarray(raw["qpos"]).reshape(-1, raw["qpos"].shape[-1])
        qvel = np.asarray(raw["qvel"]).reshape(-1, raw["qvel"].shape[-1])
        ctrl = np.asarray(raw["ctrl"]).reshape(-1, raw["ctrl"].shape[-1])
        if "time" in raw.files:
            timestamps = np.asarray(raw["time"]).reshape(-1)
        else:
            timestamps = np.arange(qpos.shape[0], dtype=np.float64)
    count = min(len(qpos), len(qvel), len(ctrl), len(timestamps))
    qpos, qvel, ctrl, timestamps = (
        qpos[:count], qvel[:count], ctrl[:count], timestamps[:count]
    )
    np.savez_compressed(output / "physical_record.npz", qpos=qpos, qvel=qvel, ctrl=ctrl, time=timestamps)
    np.savez_compressed(output / "controls.npz", ctrl=ctrl, time=timestamps)
    np.savez_compressed(output / "optimized_reference.npz", qpos=qpos, qvel=qvel, ctrl=ctrl, time=timestamps)
    with np.load(reference_path, allow_pickle=False) as reference:
        n = min(count, len(reference["qpos"]))
        root_error = np.linalg.norm(qpos[:n, :3] - reference["qpos"][:n, :3], axis=1)
    return {
        "frames": count,
        "root_position_rmse_m": float(np.sqrt(np.mean(root_error**2))),
        "root_position_max_m": float(np.max(root_error)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ref-dt", type=float, default=0.02)
    parser.add_argument("--sim-dt", type=float, default=0.005)
    parser.add_argument("--ctrl-dt", type=float, default=0.02)
    parser.add_argument("--horizon", type=float, default=0.4)
    parser.add_argument("--knot-dt", type=float, default=0.02)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--max-sim-steps", type=int, default=-1)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for path in (args.repo, args.model, args.reference):
        if not path.exists():
            raise FileNotFoundError(path)

    model = _load_model(args.model)
    pd_contract = configure_position_pd(model, args.sim_dt)
    reference_shapes = _validate_reference(args.reference, model)
    preflight = {
        "method": "SPIDER",
        "upstream_repo": "https://github.com/facebookresearch/spider",
        "upstream_commit": _repo_commit(args.repo),
        "model": str(args.model.resolve()),
        "model_sha256": _sha256(args.model),
        "reference": str(args.reference.resolve()),
        "reference_sha256": _sha256(args.reference),
        "reference_shapes": {key: list(value) for key, value in reference_shapes.items()},
        "control_semantics": "joint_position_target_via_hdmi_vita28_pd",
        "pd_contract": pd_contract,
        "device": args.device,
    }
    (args.output / "preflight.json").write_text(
        json.dumps(preflight, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if args.preflight_only:
        return 0

    runner = _load_upstream_runner(args.repo)
    import spider.simulators.mjwp as mjwp
    from spider.config import Config

    def load_pd_model(_config: Any):
        loaded = _load_model(args.model)
        configure_position_pd(loaded, args.sim_dt)
        return loaded

    runner.process_config = lambda config: _explicit_process_config(
        config, args.model, args.reference, args.output
    )
    runner.setup_mj_model = load_pd_model
    mjwp.setup_mj_model = load_pd_model
    # The upstream summary is object-specific; the comparison's validator
    # computes humanoid metrics from the normalized physical record instead.
    runner.compute_object_tracking_error = lambda *unused: {
        "obj_pos_err": float("nan"), "obj_quat_err": float("nan")
    }

    config = Config(
        robot_type="vita28_shoes",
        embodiment_type="humanoid",
        seed=args.seed,
        model_path=str(args.model),
        data_path=str(args.reference),
        output_dir=str(args.output),
        device=args.device,
        sim_dt=args.sim_dt,
        ref_dt=args.ref_dt,
        ctrl_dt=args.ctrl_dt,
        render_dt=0.02,
        trace_dt=0.02,
        horizon=args.horizon,
        knot_dt=args.knot_dt,
        max_sim_steps=args.max_sim_steps,
        num_samples=args.samples,
        max_num_iterations=args.iterations,
        contact_guidance=False,
        contact_rew_scale=0.0,
        show_viewer=False,
        wait_on_finish=False,
        save_video=False,
        save_info=True,
        save_rerun=False,
        save_viser=False,
        sanity_check_seconds=0.0,
    )
    started = time.perf_counter()
    runner.main(config)
    metrics = _normalize_upstream_output(args.output, args.reference)
    manifest = {
        **preflight,
        "status": "completed_unvalidated",
        "runtime_s": time.perf_counter() - started,
        "parameters": {
            "seed": args.seed,
            "sim_dt": args.sim_dt,
            "ref_dt": args.ref_dt,
            "ctrl_dt": args.ctrl_dt,
            "horizon": args.horizon,
            "knot_dt": args.knot_dt,
            "samples": args.samples,
            "iterations": args.iterations,
        },
        "adapter_scope": "explicit MJB/NPZ loading, HDMI PD semantics, normalized output only",
        "certificate": "unavailable: SPIDER is sampling MPC, not a dynamics certificate backend",
        "metrics": metrics,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
