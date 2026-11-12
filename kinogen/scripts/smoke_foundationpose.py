#!/usr/bin/env python3
"""Strict official FoundationPose checkpoint + preprocessing/forward smoke.

Inputs are explicitly synthetic pinhole plane crops, not a captured object.
This validates two trained networks and their official crop preprocessing.
It does not run register(), estimate a real pose, or reproduce KinoGen.
No source snapshot edits, external downloads, or model training are performed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def synthetic_batch(torch, BatchPoseData, cfg, pairs):
    """Known pinhole plane; raw RGB 0..255 and camera-space XYZ in metres.

    We supply identity crop transforms because the generated images already
    have exactly the configured crop size. Official transform_batch performs
    RGB scaling, camera-centre subtraction, radius scaling, and invalid masking.
    A and B differ by a small depth/colour change; no pose accuracy is asserted.
    """
    height, width = map(int, cfg.input_resize)
    y, x = torch.meshgrid(
        torch.arange(height, device="cuda", dtype=torch.float32),
        torch.arange(width, device="cuda", dtype=torch.float32), indexing="ij",
    )
    focal = float(max(height, width) * 3)
    cx, cy = (width - 1) / 2, (height - 1) / 2
    foreground = ((x - cx) / (width * .38)).square() + ((y - cy) / (height * .33)).square() <= 1

    def observation(z, colour_shift):
        xyz = torch.stack(((x - cx) * z / focal, (y - cy) * z / focal, torch.full_like(x, z)))
        xyz[:, ~foreground] = 0
        rgb = torch.stack((40 + 160 * x / max(width - 1, 1), 40 + 160 * y / max(height - 1, 1), torch.full_like(x, 130 + colour_shift)))
        rgb[:, ~foreground] = 0
        return rgb[None].repeat(pairs, 1, 1, 1), xyz[None].repeat(pairs, 1, 1, 1)

    rgb_a, xyz_a = observation(.98, 0)
    rgb_b, xyz_b = observation(.985, 5)
    pose = torch.eye(4, device="cuda")[None].repeat(pairs, 1, 1)
    pose[:, 2, 3] = 1
    intrinsics = torch.tensor([[focal, 0, cx], [0, focal, cy], [0, 0, 1]], device="cuda")[None].repeat(pairs, 1, 1)
    return BatchPoseData(
        rgbAs=rgb_a, rgbBs=rgb_b, xyz_mapAs=xyz_a, xyz_mapBs=xyz_b,
        poseA=pose, Ks=intrinsics,
        tf_to_crops=torch.eye(3, device="cuda")[None].repeat(pairs, 1, 1),
        mesh_diameters=torch.full((pairs,), .5, device="cuda"),
    )


def tensor_summary(torch, value):
    if not bool(torch.isfinite(value).all()):
        raise RuntimeError("Network produced non-finite values")
    value = value.detach().float().cpu()
    return {"shape": list(value.shape), "min": float(value.min()), "max": float(value.max()), "mean": float(value.mean()), "values": value.tolist() if value.numel() <= 100 else None}


def run(args, report):
    import torch
    from omegaconf import OmegaConf
    if not torch.cuda.is_available():
        raise RuntimeError("GPU smoke requires CUDA; do not install/run this locally for video preview")
    torch.manual_seed(20261020)
    source = ROOT / "dependencies/foundationpose"
    sys.path.insert(0, str(source))
    native = ROOT / ".build/foundationpose-mycpp"
    if native.exists():
        sys.path.insert(0, str(native))
    from learning.models.refine_network import RefineNet
    from learning.models.score_network import ScoreNetMultiPair
    from learning.datasets.h5_dataset import PoseRefinePairH5Dataset, ScoreMultiPairH5Dataset
    from learning.datasets.pose_dataset import BatchPoseData
    import Utils

    report.update({
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "native_modules": {name: getattr(Utils, name, None) is not None for name in ("mycpp", "wp", "kornia")},
        "networks": {},
    })
    for name, run_id, model_class, dataset_class in (
        ("refiner", "2023-10-28-18-33-37", RefineNet, PoseRefinePairH5Dataset),
        ("scorer", "2024-01-11-20-02-45", ScoreNetMultiPair, ScoreMultiPairH5Dataset),
    ):
        start = time.perf_counter()
        folder = ROOT / "weights/foundationpose-pytorch" / run_id
        config_path, checkpoint_path = folder / "config.yml", folder / "model_best.pth"
        cfg = OmegaConf.load(config_path)
        if int(cfg.c_in) != 6 or bool(cfg.use_normal):
            raise RuntimeError("This smoke supports the published RGB+XYZ six-channel checkpoints only")
        cfg.zfar = float("inf")
        model = model_class(cfg=cfg, c_in=cfg.c_in)
        # These are the hash-verified public NVIDIA checkpoints, loaded locally.
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = state["model"] if "model" in state else state
        compatibility = model.load_state_dict(state, strict=True)
        if compatibility.missing_keys or compatibility.unexpected_keys:
            raise RuntimeError("Strict checkpoint key mismatch")
        model.cuda().eval()
        dataset = dataset_class(cfg=cfg, h5_file="", mode="test")
        batch = synthetic_batch(torch, BatchPoseData, cfg, args.pairs)
        height, width = map(int, cfg.input_resize)
        batch = dataset.transform_batch(batch, H_ori=height, W_ori=width, bound=1)
        a = torch.cat((batch.rgbAs, batch.xyz_mapAs), dim=1).float()
        b = torch.cat((batch.rgbBs, batch.xyz_mapBs), dim=1).float()
        if a.shape != (args.pairs, 6, height, width) or a.shape != b.shape:
            raise RuntimeError("Unexpected preprocessed crop shape")
        if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
            raise RuntimeError("Non-finite synthetic crop after preprocessing")
        with torch.inference_mode():
            output = model(a, b) if name == "refiner" else model(a, b, L=args.pairs)
        torch.cuda.synchronize()
        report["networks"][name] = {
            "run_id": run_id, "checkpoint_sha256": file_sha256(checkpoint_path),
            "config_sha256": file_sha256(config_path), "strict_checkpoint_load": True,
            "checkpoint_key_count": len(state), "input_shape": list(a.shape),
            "official_preprocessing_executed": True,
            "output": {key: tensor_summary(torch, value) for key, value in output.items()},
            "elapsed_seconds": round(time.perf_counter() - start, 3),
        }
        print(f"PASS {name}: strict load; official preprocessing; finite forward", flush=True)
        del model, state, output, a, b, batch
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=2, help="number of synthetic pairs per scoring group (1..8)")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/foundationpose_smoke.json")
    args = parser.parse_args()
    if not 1 <= args.pairs <= 8:
        parser.error("--pairs must be between 1 and 8")
    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(), "status": "running",
        "mode": "synthetic_crop_network_smoke", "synthetic_inputs": True,
        "real_pose_estimation_executed": False, "full_registration_executed": False,
        "kinogen_pipeline_executed": False, "training_executed": False,
        "scope": "Strict official refiner/scorer checkpoint loading, official RGB/XYZ crop normalization, and actual network forward; no pose accuracy claim",
    }
    try:
        run(args, report)
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Report: {args.report}", flush=True)


if __name__ == "__main__":
    main()
