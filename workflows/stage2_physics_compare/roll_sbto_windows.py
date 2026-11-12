#!/usr/bin/env python3
"""Run contact-aware SBTO as overlapping windows with physical state handoff."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import mujoco
import numpy as np

from common import configure_position_pd, load_canonical, sha256, write_json


def _window_reference(source: dict[str, np.ndarray], start: int, stop: int,
                      qpos: np.ndarray, qvel: np.ndarray, path: Path) -> None:
    frames = len(source["qpos_ref"])
    arrays = {
        key: (value[start:stop].copy()
              if value.ndim and value.shape[0] == frames else value.copy())
        for key, value in source.items()
    }
    arrays["qpos_ref"][0] = qpos
    arrays["qvel_ref"][0] = qvel
    np.savez_compressed(path, **arrays)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True,
                        help="Python executable containing the official SBTO environment")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--sbto-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon-frames", type=int, default=21)
    parser.add_argument("--commit-frames", type=int, default=10)
    parser.add_argument("--sim-dt", type=float, default=0.005)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.horizon_frames < 3 or not 1 <= args.commit_frames < args.horizon_frames:
        raise ValueError("Need horizon_frames >= 3 and 1 <= commit_frames < horizon_frames")

    source = load_canonical(args.reference)
    fps = float(np.asarray(source.get("fps", [50.0])).reshape(-1)[0])
    substeps = int(round((1.0 / fps) / args.sim_dt))
    if substeps <= 0 or not np.isclose(substeps * args.sim_dt, 1.0 / fps):
        raise ValueError("sim_dt must exactly divide the source frame interval")

    args.output.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_binary_path(str(args.model))
    model.opt.timestep = args.sim_dt
    configure_position_pd(model, mujoco)
    data = mujoco.MjData(model)
    data.qpos[:] = source["qpos_ref"][0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    controls: list[np.ndarray] = []
    qpos_frames = [data.qpos.copy()]
    qvel_frames = [data.qvel.copy()]
    windows: list[dict[str, object]] = []
    total_frames = len(source["qpos_ref"])
    elapsed_before = 0.0
    partial_path = args.output / "rolling_manifest.partial.json"
    if args.resume:
        if not partial_path.exists():
            raise FileNotFoundError(f"Cannot resume without {partial_path}")
        partial = json.loads(partial_path.read_text(encoding="utf-8"))
        windows = partial["completed_windows"]
        elapsed_before = float(partial.get("elapsed_s", 0.0))
        for window_id, item in enumerate(windows):
            start = int(item["start_frame"])
            stop = int(item["target_stop_frame"]) + 1
            if window_id != int(item["window"]) or start != window_id * args.commit_frames:
                raise ValueError("Partial manifest windows are not contiguous")
            control_path = (args.output / "windows" / f"{window_id:03d}_{start:04d}_{stop - 1:04d}"
                            / "optimized" / "controls.npz")
            with np.load(control_path, allow_pickle=False) as archive:
                window_controls = np.asarray(archive["pd_target"], dtype=np.float64)
            commit_intervals = int(item["committed_stop_frame"]) - start
            committed = window_controls[:commit_intervals * substeps]
            controls.append(committed)
            for step, target in enumerate(committed, 1):
                data.ctrl[:] = target
                mujoco.mj_step(model, data)
                if step % substeps == 0:
                    qpos_frames.append(data.qpos.copy())
                    qvel_frames.append(data.qvel.copy())
        resumed_intervals = sum(int(item["committed_stop_frame"]) - int(item["start_frame"])
                                for item in windows)
        if len(qpos_frames) != 1 + resumed_intervals:
            raise ValueError("Partial manifest has an incomplete commit window")
        print(json.dumps({"resumed_windows": len(windows),
                          "resumed_frame": len(qpos_frames) - 1}), flush=True)
    started = time.perf_counter()

    for window_id, start in enumerate(range(0, total_frames - 1, args.commit_frames)):
        if window_id < len(windows):
            continue
        stop = min(start + args.horizon_frames, total_frames)
        commit_intervals = min(args.commit_frames, total_frames - 1 - start)
        window_dir = args.output / "windows" / f"{window_id:03d}_{start:04d}_{stop - 1:04d}"
        window_dir.mkdir(parents=True, exist_ok=True)
        window_reference = window_dir / "reference.npz"
        _window_reference(source, start, stop, data.qpos.copy(), data.qvel.copy(), window_reference)
        command = [
            str(args.python), str(args.adapter),
            "--sbto-root", str(args.sbto_root),
            "--reference", str(window_reference),
            "--model", str(args.model),
            "--output", str(window_dir / "optimized"),
            "--sim-dt", str(args.sim_dt),
            "--samples", str(args.samples),
            "--step-knots", str(max(1, (stop - start - 1) // 2)),
            "--max-iterations-per-knot", str(args.iterations),
            "--threads", str(args.threads),
            "--seed", str(args.seed + window_id),
            "--root-xy-weight", "800",
            "--root-z-weight", "1500",
            "--root-orientation-weight", "800",
            "--contact-aware",
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        (window_dir / "optimizer.log").write_text(
            result.stdout + result.stderr, encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"SBTO window {window_id} failed; see {window_dir / 'optimizer.log'}")

        control_path = window_dir / "optimized" / "controls.npz"
        with np.load(control_path, allow_pickle=False) as archive:
            window_controls = np.asarray(archive["pd_target"], dtype=np.float64)
        commit_steps = commit_intervals * substeps
        if len(window_controls) < commit_steps or window_controls.shape[1] != model.nu:
            raise ValueError(f"Window {window_id} returned invalid control shape {window_controls.shape}")
        committed = window_controls[:commit_steps]
        controls.append(committed)
        for step, target in enumerate(committed, 1):
            data.ctrl[:] = target
            mujoco.mj_step(model, data)
            if step % substeps == 0:
                qpos_frames.append(data.qpos.copy())
                qvel_frames.append(data.qvel.copy())

        item = {
            "window": window_id,
            "start_frame": start,
            "target_stop_frame": stop - 1,
            "committed_stop_frame": start + commit_intervals,
            "initial_root": qpos_frames[-commit_intervals - 1][:3].tolist(),
            "final_root": data.qpos[:3].tolist(),
            "reference_sha256": sha256(window_reference),
            "controls_sha256": sha256(control_path),
        }
        windows.append(item)
        print(json.dumps(item), flush=True)
        write_json(args.output / "rolling_manifest.partial.json", {
            "completed_windows": windows,
            "elapsed_s": elapsed_before + time.perf_counter() - started,
        })

    joined = np.concatenate(controls, axis=0)
    expected_steps = (total_frames - 1) * substeps
    if joined.shape != (expected_steps, model.nu):
        raise AssertionError(f"Joined controls {joined.shape} != {(expected_steps, model.nu)}")
    if len(qpos_frames) != total_frames or len(qvel_frames) != total_frames:
        raise AssertionError("Physical handoff record does not cover every source frame")

    controls_path = args.output / "controls.npz"
    np.savez_compressed(controls_path, pd_target=joined, sim_dt=np.float64(args.sim_dt))
    np.savez_compressed(
        args.output / "handoff_record.npz",
        qpos=np.asarray(qpos_frames), qvel=np.asarray(qvel_frames),
        fps=np.asarray([fps]), reset_count=np.asarray([0]),
        state_writes_after_initialization=np.asarray([0]),
    )
    manifest = {
        "schema": "stage2-sbto-rolling-v1",
        "status": "completed_unvalidated",
        "reference": str(args.reference),
        "reference_sha256": sha256(args.reference),
        "model": str(args.model),
        "model_sha256": sha256(args.model),
        "frames": total_frames,
        "physics_steps": expected_steps,
        "horizon_frames": args.horizon_frames,
        "commit_frames": args.commit_frames,
        "window_count": len(windows),
        "samples": args.samples,
        "iterations": args.iterations,
        "wall_time_s": elapsed_before + time.perf_counter() - started,
        "controls_sha256": sha256(controls_path),
        "continuous_physics_handoff": True,
        "windows": windows,
    }
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
