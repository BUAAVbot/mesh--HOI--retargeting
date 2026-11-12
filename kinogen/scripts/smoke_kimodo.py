#!/usr/bin/env python3
"""Server-only, genuine Kimodo G1 empty-text sampling and skeleton video.

This exercises an upstream component, not the unreleased KinoGen pipeline.
The empty-only encoder implements the exact zero feature/zero text length used
by upstream Kimodo._generate for empty strings. It cannot encode text prompts.
No alternate language model, pretrained text embedding, or paid API is used.
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


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def render_video(joints, parents, names, fps, path: Path, root_distance_m: float = 0) -> None:
    """Two fixed orthographic views of actual generated world-space joints."""
    import av
    import numpy as np
    from PIL import Image, ImageDraw

    width, height = 960, 640
    # Kimodo uses Y up, +Z forward. Keep one camera/bounds for the whole clip.
    mins = joints.min(axis=(0, 1))
    maxs = joints.max(axis=(0, 1))
    centers = (mins + maxs) / 2
    extent = max(float(np.max(maxs - mins)), 1.5) * 1.25
    scale = 430 / extent

    def project(point, horizontal, panel):
        return (
            int(panel * 480 + 240 + (point[horizontal] - centers[horizontal]) * scale),
            int(355 - (point[1] - centers[1]) * scale),
        )

    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=int(fps))
        stream.width, stream.height = width, height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "20", "preset": "medium"}
        for index, pose in enumerate(joints):
            frame_image = Image.new("RGB", (width, height), "#101827")
            draw = ImageDraw.Draw(frame_image)
            title = "KINEMATIC CONSTRAINT ONLY" if root_distance_m > 0 else "UNCONDITIONAL"
            subtitle = f"Empty text; root waypoint target +Z {root_distance_m:g}m; not full KinoGen" if root_distance_m > 0 else "Empty text; no task/object constraints; kinematic sample; not full KinoGen"
            draw.text((22, 15), f"KIMODO G1 | {title} COMPONENT SMOKE", fill="#ffffff")
            draw.text((22, 34), subtitle, fill="#b8c9dd")
            draw.line((480, 65, 480, 600), fill="#334155", width=2)
            for panel, horizontal, label in ((0, 0, "FRONT: X/Y"), (1, 2, "SIDE: Z/Y")):
                draw.text((panel * 480 + 18, 69), label, fill="#b8c9dd")
                floor_y = project((0, 0, 0), horizontal, panel)[1]
                if 95 < floor_y < 580:
                    draw.line((panel * 480 + 15, floor_y, panel * 480 + 465, floor_y), fill="#3d5065", width=1)
                for joint_index, parent in enumerate(parents):
                    color = "#60a5fa" if names[joint_index].startswith("left_") else "#fb7185" if names[joint_index].startswith("right_") else "#e2e8f0"
                    point = project(pose[joint_index], horizontal, panel)
                    if int(parent) >= 0:
                        draw.line((project(pose[int(parent)], horizontal, panel), point), fill=color, width=5)
                    draw.ellipse((point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3), fill=color)
            draw.text((22, 610), f"Frame {index + 1}/{len(joints)} | {index / fps:.2f}s | {fps:g} FPS | world-space skeleton", fill="#b8c9dd")
            video_frame = av.VideoFrame.from_ndarray(np.asarray(frame_image), format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    # Decode the result to verify actual frame count, not just encoder exit.
    with av.open(str(path)) as container:
        decoded_frames = sum(1 for _ in container.decode(video=0))
    if decoded_frames != len(joints):
        raise RuntimeError(f"Video verification failed: {decoded_frames} != {len(joints)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--root-distance-m", type=float, default=0, help="Positive: use official start/end root XZ+heading constraints for +Z travel. Zero: unconditional.")
    parser.add_argument("--device", default="cuda:0", help="Visible CUDA device; select physical GPU via CUDA_VISIBLE_DEVICES")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "weights/kimodo-g1-rp-v1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/previews")
    args = parser.parse_args()
    if platform.system() != "Linux":
        parser.error("Run this script on the Linux GPU server. Local computer is for video preview only.")
    if args.frames < 2 or not (2 <= args.steps <= 1000):
        parser.error("Require frames >= 2 and 2 <= steps <= 1000")
    if not 0 <= args.root_distance_m < float("inf"):
        parser.error("Require finite root-distance-m >= 0")
    if not args.device.startswith("cuda"):
        parser.error("This smoke test is configured for server CUDA inference only")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(ROOT / "dependencies/kimodo"))

    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from kimodo.constraints import Root2DConstraintSet, save_constraints_lst
    from kimodo.model.loading import instantiate_from_dict

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the server environment")

    class EmptyTextEncoder(torch.nn.Module):
        """Only upstream's null text condition; deliberately rejects real text."""

        def forward(self, texts):
            if any(text.strip() for text in texts):
                raise ValueError("This encoder supports empty-text unconditional sampling only")
            return torch.zeros((len(texts), 1, 4096), device=args.device), [0] * len(texts)

    model_dir = args.model_dir.resolve()
    config = OmegaConf.load(model_dir / "config.yaml")
    config = OmegaConf.merge(config, {"checkpoint_dir": str(model_dir), "text_encoder": None})
    model_config = OmegaConf.to_container(config, resolve=True)
    model_config.pop("checkpoint_dir", None)
    print(f"Loading public Kimodo G1 checkpoint from {model_dir}", flush=True)
    started = time.perf_counter()
    model = instantiate_from_dict(model_config, overrides={"device": args.device})
    model.text_encoder = EmptyTextEncoder()
    model.eval()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    constraints = []
    sampling_kwargs = {"cfg_type": "nocfg"}
    mode = "unconditional"
    if args.root_distance_m > 0:
        # Official ground-plane convention is [x, z]; heading [cos, sin].
        # Only the two waypoint conditions are authored. All motion between
        # them, including limbs, comes from the trained diffusion model.
        constraints = [Root2DConstraintSet(
            skeleton=model.skeleton,
            frame_indices=torch.tensor([0, args.frames - 1], dtype=torch.long, device=args.device),
            smooth_root_2d=torch.tensor([[0., 0.], [0., args.root_distance_m]], dtype=torch.float32, device=args.device),
            global_root_heading=torch.tensor([[1., 0.], [1., 0.]], dtype=torch.float32, device=args.device),
        )]
        sampling_kwargs = {"cfg_type": "separated", "cfg_weight": [0., 2.]}
        mode = "kinematic"
    print(f"Sampling genuine empty-text {mode} motion; no language model loaded", flush=True)
    with torch.inference_mode():
        motion = model(
            prompts="", num_frames=args.frames, num_denoising_steps=args.steps,
            constraint_lst=constraints, post_processing=False,
            return_numpy=True,
            **sampling_kwargs,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    motion = {name: np.asarray(value) for name, value in motion.items()}
    joints = motion["posed_joints"]
    if joints.shape != (args.frames, 34, 3):
        raise RuntimeError(f"Unexpected G1 output shape: {joints.shape}")
    for name, value in motion.items():
        if not np.isfinite(value).all():
            raise RuntimeError(f"Nonfinite generated values in {name}")
    parents = model.skeleton.joint_parents.detach().cpu().numpy()
    names = list(model.skeleton.bone_order_names)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    stem = args.output_dir / f"kimodo_g1_{mode}_{stamp}"
    npz_path = stem.with_suffix(".npz")
    video_path = stem.with_suffix(".mp4")
    np.savez_compressed(npz_path, **motion, fps=np.asarray(model.fps), joint_parents=parents, joint_names=np.asarray(names))
    render_video(joints, parents, names, model.fps, video_path, args.root_distance_m)
    constraints_info = []
    constraint_checks = None
    if constraints:
        constraints_info = save_constraints_lst(str(stem) + ".constraints.json", constraints)
        indices = np.asarray([0, args.frames - 1])
        expected_xz = np.asarray([[0., 0.], [0., args.root_distance_m]])
        # Constraint targets refer to the smoothed root; report the actual
        # pelvis trajectory separately so their meanings cannot be confused.
        generated_xz = motion["smooth_root_pos"][indices][:, [0, 2]]
        errors = np.linalg.norm(generated_xz - expected_xz, axis=-1)
        heading = motion["global_root_heading"][indices]
        heading_errors = np.abs(np.arctan2(heading[:, 1], heading[:, 0]))
        constraint_checks = {
            "target_root_distance_m": args.root_distance_m,
            "frame_indices": indices.tolist(),
            "target_smooth_root_xz_m": expected_xz.tolist(),
            "generated_smooth_root_xz_m": generated_xz.tolist(),
            "smooth_root_start_error_m": float(errors[0]),
            "smooth_root_end_error_m": float(errors[1]),
            "heading_errors_rad": heading_errors.tolist(),
            "position_tolerance_m": .15,
            "heading_tolerance_rad": .26,
            "constraint_following_passed": bool(np.max(errors) <= .15 and np.max(heading_errors) <= .26),
            "smooth_root_displacement_xyz_m": (motion["smooth_root_pos"][-1] - motion["smooth_root_pos"][0]).tolist(),
            "pelvis_displacement_xyz_m": (motion["root_positions"][-1] - motion["root_positions"][0]).tolist(),
            "postprocessing_enabled": False,
            "generated_motion_modified_after_sampling": False,
        }
    meta = {
        "status": "component_smoke_passed", "full_kinogen_pipeline": False,
        "component": "Kimodo-G1-RP-v1", "conditioning": "kinematic_constraint_only_empty_text" if constraints else "unconditional_empty_text",
        "prompt": "", "constraints": constraints_info, "text_encoder_loaded": False,
        "sampling_configuration": sampling_kwargs,
        "constraint_checks": constraint_checks,
        "null_condition_reference": "dependencies/kimodo/kimodo/model/kimodo_model.py::_generate",
        "limitations": ["No text following", "No hand-object interaction", "No physical tracking or simulator validation", "No video-generation API stage", "KinoGen integration code is not released"],
        "seed": args.seed, "frames": args.frames, "fps": model.fps, "denoising_steps": args.steps,
        "inference_and_load_seconds": elapsed,
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(torch.device(args.device)),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "checkpoint_sha256": sha256(model_dir / "model.safetensors"),
        "output_shapes": {name: list(value.shape) for name, value in motion.items()},
        "all_arrays_finite": True,
        "joint_motion_std_m": float(joints.std(axis=0).mean()),
        "video_decoded_frame_count": len(joints),
        "files": [{"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in (npz_path, video_path)],
    }
    stem.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
