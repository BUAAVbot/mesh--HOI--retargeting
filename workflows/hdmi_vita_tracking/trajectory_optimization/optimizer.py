"""Bounded, compact-support residual CEM using the frozen HDMI policy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.interpolate import CubicSpline

from backend import HdmiDynamics, sha


def scalar_metrics(metrics, index):
    return {k: np.asarray(v)[index].item() for k, v in metrics.items()}


def dense_residual(parameters, frames, names, left, right, scope="legs"):
    """C2 compact window; parameters are desired joint target offsets in rad."""
    n, knots, j = parameters.shape
    query = np.arange(frames - 1)
    nodes = np.linspace(left, right, knots)
    value = CubicSpline(nodes, parameters, axis=1, bc_type="clamped", extrapolate=False)(query)
    value = np.nan_to_num(value)
    phase = np.clip((query - left) / (right - left), 0, 1)
    envelope = 64 * phase**3 * (1 - phase)**3
    value *= envelope[None, :, None]
    out = np.zeros((n, frames - 1, 28), np.float32)
    ids = [i for i, name in enumerate(names) if scope == "all" or any(x in name for x in ("hip", "knee", "ankle", "waist"))]
    out[:, :, ids] = value
    # The active tracking profile uses 0.5 leg and 0.4 waist action scale.
    scales = np.array([.5 if any(x in name for x in ("hip", "knee", "ankle")) else
                       .3 if "wrist" in name else .4 for name in names])
    return out / scales[None, None]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--motion", type=Path, required=True)
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--iterations", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--knots", type=int, default=7)
    p.add_argument("--left", type=int, default=80)
    p.add_argument("--right", type=int, default=225)
    p.add_argument("--std", type=float, default=.045)
    p.add_argument("--bound", type=float, default=.18)
    p.add_argument("--baseline-only", action="store_true")
    p.add_argument("--controls", type=Path)
    p.add_argument("--instrument", action="store_true")
    p.add_argument("--resume", type=Path)
    p.add_argument("--joint-scope", choices=("legs", "all"), default="legs")
    p.add_argument("--blend-legs", type=float, default=0.)
    p.add_argument("--blend-arms", type=float, default=0.)
    p.add_argument("--blend-wrists", type=float, default=0.)
    p.add_argument("--clip-targets", action="store_true")
    p.add_argument("--optimize-blend", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    count = 1 if args.baseline_only or args.controls else args.samples
    b = HdmiDynamics(args.root, args.motion, args.scene, args.checkpoint, count=count,
                     device=args.device, seed=args.seed, instrument=args.instrument, clip_targets=args.clip_targets)
    blends = np.tile([args.blend_legs,args.blend_arms,args.blend_wrists], (count,1))
    b.export_model(args.output / "compiled_model.mjb")
    contract = {"source_reference_sha256": sha(args.motion), "scene_sha256": sha(args.scene),
                "checkpoint_sha256": sha(args.checkpoint), "model_sha256": sha(args.output / "compiled_model.mjb"),
                "continuous_physics": True, "reset_count": 0, "state_writes_after_initialization": 0,
                "external_root_force": False, "obs_noise": False, "control_hz": 50,
                "physics_hz": 200, "frames": b.frames, "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    contract["robot_weight_N"] = b.robot_weight_N
    contract["ground_metadata"] = b.ground_metadata
    contract["support_filter_verified"] = bool(args.instrument)
    contract["runtime_task_sha256"] = sha(args.root / "task.py")
    contract["runtime_rl_cfg_sha256"] = sha(args.root / "rl_cfg.py")
    contract["robot_asset_sha256"] = sha(args.root / "assets/vita28_shoes.xml")
    (args.output / "contract.json").write_text(json.dumps(contract, indent=2))
    started = time.time()
    try:
        if args.baseline_only or args.controls:
            residual = np.zeros((1, b.frames - 1, 28), np.float32)
            if args.controls:
                with np.load(args.controls, allow_pickle=False) as z:
                    if str(z["source_reference_sha256"]) != sha(args.motion):
                        raise ValueError("Serialized controls reference mismatch")
                    if str(z["scene_sha256"]) != sha(args.scene) or str(z["checkpoint_sha256"]) != sha(args.checkpoint):
                        raise ValueError("Serialized controls scene/checkpoint mismatch")
                    residual[0] = z["residual_actions"]
                    if "reference_blend" in z:
                        blends[0] = z["reference_blend"]
                    if "clip_control_targets_to_limits" in z:
                        b.clip_targets = bool(z["clip_control_targets_to_limits"])
            contract["reference_blend"] = blends[0].tolist()
            contract["clip_control_targets_to_limits"] = b.clip_targets
            (args.output / "contract.json").write_text(json.dumps(contract, indent=2))
            cost, metrics, record = b.rollout(residual, record=True, blends=blends)
            np.savez_compressed(args.output / "physical_record.npz", **record)
            report = {**contract, "metrics": scalar_metrics(metrics, 0), "cost": float(cost[0]),
                      "checkpoint": str(args.checkpoint), "first_failure_frame": int(metrics["first_failure_frame"][0]),
                      "elapsed_s": time.time() - started, "fresh_replay": bool(args.controls),
                      "passed": False, "gate_status": "pending_independent_acceptance"}
            (args.output / "report.json").write_text(json.dumps(report, indent=2))
            print(json.dumps(report), flush=True)
            return
        nj = 28 if args.joint_scope == "all" else sum(any(x in name for x in ("hip", "knee", "ankle", "waist")) for name in b.names)
        shape = (args.knots, nj)
        mean, sigma = np.zeros(shape), np.full(shape, args.std)
        mean_blend = blends[0].copy()
        sigma_blend = np.full(3, .08)
        if args.resume:
            with np.load(args.resume, allow_pickle=False) as z:
                mean = np.asarray(z["parameters"]).copy()
                if mean.shape != shape and shape[1] == 28 and mean.shape == (args.knots, 14):
                    expanded = np.zeros(shape)
                    ids = [i for i,name in enumerate(b.names) if any(x in name for x in ("hip","knee","ankle","waist"))]
                    expanded[:,ids] = mean
                    mean = expanded
                if mean.shape != shape:
                    raise ValueError("Resume parameter/basis shape mismatch")
                for key in ("left", "right"):
                    if key in z and int(z[key]) != getattr(args, key):
                        raise ValueError("Resume time basis mismatch")
                if "reference_blend" in z:
                    mean_blend = np.asarray(z["reference_blend"]).copy()
        best_params, best_cost, best_metrics = mean.copy(), float("inf"), None
        best_blend = mean_blend.copy()
        best_rank = (float("inf"), float("inf"), float("inf"))
        rng = np.random.default_rng(args.seed)
        history = []
        for iteration in range(args.iterations):
            params = np.clip(mean + rng.standard_normal((count, *shape)) * sigma, -args.bound, args.bound)
            params[0], params[1] = best_params, 0
            trial_blends = blends.copy()
            if args.optimize_blend:
                trial_blends = np.clip(mean_blend + rng.standard_normal((count,3)) * sigma_blend, 0, 1)
                trial_blends[0], trial_blends[1] = best_blend, 0
            residual = dense_residual(params, b.frames, b.names, args.left, args.right, args.joint_scope)
            costs, metrics, _ = b.rollout(residual, blends=trial_blends)
            ranks = np.lexsort((costs, metrics["normalized_gate_violation"], metrics["failed_gate_count"]))
            winner = int(ranks[0])
            rank = (float(metrics["failed_gate_count"][winner]),
                    float(metrics["normalized_gate_violation"][winner]), float(costs[winner]))
            if rank < best_rank:
                best_rank = rank
                best_cost, best_params = float(costs[winner]), params[winner].copy()
                best_blend = trial_blends[winner].copy()
                best_metrics = scalar_metrics(metrics, winner)
                best_residual = dense_residual(best_params[None], b.frames, b.names, args.left, args.right, args.joint_scope)[0]
                np.savez_compressed(args.output / "controls.npz", residual_actions=best_residual,
                                    parameters=best_params, source_reference_sha256=np.array(sha(args.motion)),
                                    scene_sha256=np.array(sha(args.scene)), checkpoint_sha256=np.array(sha(args.checkpoint)),
                                    joint_names=np.asarray(b.names), left=np.array(args.left), right=np.array(args.right),
                                    knots=np.array(args.knots), joint_scope=np.array(args.joint_scope),
                                    reference_blend=best_blend, clip_control_targets_to_limits=np.array(args.clip_targets),
                                    semantics=np.array("frozen_actor_normalized_action_residual"))
            elite = params[ranks[:max(3, count // 4)]]
            if args.optimize_blend:
                elite_blend = trial_blends[ranks[:max(3,count//4)]]
                mean_blend = .4 * mean_blend + .6 * elite_blend.mean(0)
                sigma_blend = np.sqrt(.65*sigma_blend**2 + .35*(elite_blend.var(0)+.012**2))
            mean = .4 * mean + .6 * elite.mean(0)
            sigma = np.sqrt(.65 * sigma**2 + .35 * (elite.var(0) + .008**2))
            row = {"iteration": iteration, "best_cost": best_cost, "best_metrics": best_metrics,
                   "best_rank": best_rank, "best_blend": best_blend.tolist(),
                   "sigma_max_rad": float(sigma.max()), "elapsed_s": time.time() - started}
            history.append(row)
            (args.output / "history.json").write_text(json.dumps(history, indent=2))
            print(json.dumps(row), flush=True)
        (args.output / "search_summary.json").write_text(json.dumps({**contract, "best_metrics": best_metrics,
                  "cost": best_cost, "elapsed_s": time.time() - started}, indent=2))
    finally:
        b.close()


if __name__ == "__main__":
    main()
