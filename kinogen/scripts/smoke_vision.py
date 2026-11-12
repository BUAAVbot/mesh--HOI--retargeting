#!/usr/bin/env python3
"""Real, independent public-model inference; this is not the KinoGen pipeline.

Run on the GPU server, e.g.:
  python scripts/smoke_vision.py --component all
No network downloads or training are performed by this script.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf8")


def label(frame, text):
    import cv2
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 29), (20, 20, 20), -1)
    cv2.putText(frame, text, (7, 20), cv2.FONT_HERSHEY_SIMPLEX, .43,
                (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def save_video(path, frames_bgr, fps):
    import cv2
    import imageio.v2 as imageio
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(path), fps=fps, codec="libx264", macro_block_size=1,
                            ffmpeg_params=["-crf", "18", "-pix_fmt", "yuv420p"]) as writer:
        for frame in frames_bgr:
            height, width = frame.shape[:2]
            # H.264 yuv420 requires even dimensions.
            frame = cv2.copyMakeBorder(frame, 0, height % 2, 0, width % 2,
                                       cv2.BORDER_REPLICATE)
            writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if not ok or count != len(frames_bgr):
        raise RuntimeError(f"Preview verification failed: {path}, frames={count}")
    return {"path": relative(path), "sha256": digest(path), "frames": count,
            "width": int(frame.shape[1]), "height": int(frame.shape[0]), "fps": fps}


def load_frames(args):
    import cv2
    import numpy as np
    if args.video:
        cap = cv2.VideoCapture(str(args.video))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or args.fps
        frames = []
        while len(frames) < args.frames:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        origin = {"kind": "user_supplied_video", "path": str(args.video),
                  "sha256": digest(args.video)}
    else:
        image = cv2.imread(str(args.image))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {args.image}")
        h, w = image.shape[:2]
        frames = []
        for index in range(args.frames):
            offset = 8.0 * index / max(args.frames - 1, 1)
            affine = np.float32([[1, 0, offset], [0, 1, .3 * offset]])
            frames.append(cv2.warpAffine(image, affine, (w, h),
                                         borderMode=cv2.BORDER_REFLECT_101))
        fps = args.fps
        origin = {"kind": "synthetic_camera_translation_of_official_demo_image",
                  "path": relative(args.image), "sha256": digest(args.image),
                  "note": "Static image translated by 0 to 8 pixels; not a generated action or an evaluation dataset."}
    if not frames:
        raise RuntimeError("No readable input frames")
    max_res = args.max_res
    scale = min(1., max_res / max(frames[0].shape[:2]))
    if scale < 1:
        h, w = frames[0].shape[:2]
        size = (max(2, int(w * scale) // 2 * 2), max(2, int(h * scale) // 2 * 2))
        frames = [cv2.resize(frame, size) for frame in frames]
    return np.stack(frames), fps, origin


def run_depth(args, frames, fps, out):
    import cv2
    import numpy as np
    import torch
    sys.path.insert(0, str(ROOT / "dependencies/video-depth-anything"))
    from video_depth_anything.video_depth import VideoDepthAnything
    configs = {"vits": {"features": 64, "out_channels": [48, 96, 192, 384]},
               "vitl": {"features": 256, "out_channels": [256, 512, 1024, 1024]}}
    suffix = "small" if args.encoder == "vits" else "large"
    checkpoint = ROOT / f"weights/video-depth-anything-{suffix}/video_depth_anything_{args.encoder}.pth"
    model = VideoDepthAnything(encoder=args.encoder, **configs[args.encoder])
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    model = model.to(args.device).eval()
    with torch.inference_mode():
        depths, output_fps = model.infer_video_depth(
            frames[..., ::-1].copy(), fps, input_size=args.depth_input_size,
            device=args.device, fp32=args.fp32)
    depths = np.asarray(depths)
    if not np.isfinite(depths).all() or float(depths.max() - depths.min()) <= 0:
        raise RuntimeError("Depth output is non-finite or constant")
    raw = out / "video_depth.npz"
    np.savez_compressed(raw, relative_depth=depths, fps=output_fps)
    lo, hi = np.percentile(depths, [1, 99])
    visual = []
    for rgb, depth in zip(frames, depths):
        colored = cv2.applyColorMap((np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        if colored.shape[:2] != rgb.shape[:2]:
            colored = cv2.resize(colored, (rgb.shape[1], rgb.shape[0]))
        visual.append(np.concatenate([label(rgb.copy(), "Input: synthetic camera translation" if not args.video else "Input video"),
                                      label(colored, "VideoDepthAnything: relative depth")], axis=1))
    preview = save_video(ROOT / "outputs/previews/video_depth_smoke.mp4", visual, output_fps)
    return {"checkpoint": relative(checkpoint), "encoder": args.encoder,
            "strict_state_dict": True, "depth_shape": list(depths.shape),
            "depth_min": float(depths.min()), "depth_max": float(depths.max()),
            "depth_units": "relative, not metric", "array": relative(raw), "preview": preview}


def old_mano_compatibility():
    # MANO's licensed legacy pickle contains Chumpy objects. These aliases are
    # scoped to this inference process; immutable upstream sources stay intact.
    import numpy as np
    for name, value in {"bool": bool, "int": int, "float": float, "complex": complex,
                        "object": object, "str": str, "unicode": str}.items():
        if name not in np.__dict__:
            setattr(np, name, value)
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec


def run_wilor(args, frames, fps, out):
    import cv2
    import numpy as np
    import torch
    old_mano_compatibility()
    sys.path.insert(0, str(ROOT / "dependencies/wilor"))
    from wilor.configs import get_config
    from wilor.models import WiLoR
    from wilor.datasets.vitdet_dataset import ViTDetDataset
    from wilor.utils import recursive_to
    from wilor.utils.renderer import cam_crop_to_full
    from ultralytics import YOLO
    weight_root = ROOT / "weights/wilor"
    checkpoint = weight_root / "pretrained_models/wilor_final.ckpt"
    mano = weight_root / "mano_data/MANO_RIGHT.pkl"
    if not mano.is_file():
        raise FileNotFoundError(f"Licensed MANO model missing: {mano}")
    cfg = get_config(str(weight_root / "pretrained_models/model_config.yaml"))
    cfg.defrost()
    cfg.MODEL.BBOX_SHAPE = [192, 256]
    cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS", None)
    cfg.MANO.DATA_DIR = str(weight_root / "mano_data")
    cfg.MANO.MODEL_PATH = str(weight_root / "mano_data")
    cfg.MANO.MEAN_PARAMS = str(weight_root / "mano_data/mano_mean_params.npz")
    cfg.freeze()
    model = WiLoR(cfg=cfg, init_renderer=False)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    incompat = model.load_state_dict(payload["state_dict"], strict=False)
    learned_missing = sorted(set(incompat.missing_keys) & set(dict(model.named_parameters())))
    if learned_missing:
        raise RuntimeError(f"Missing learned weights: {learned_missing}")
    del payload
    model = model.to(args.device).eval()
    detector = YOLO(str(weight_root / "pretrained_models/detector.pt")).to(args.device)
    image = frames[0].copy()
    detections = detector(image, conf=.3, verbose=False)[0].boxes
    boxes = detections.xyxy.detach().cpu().numpy()
    right = detections.cls.detach().cpu().numpy()
    if len(boxes) == 0:
        raise RuntimeError("Official detector did not detect any hands; no reconstruction success claimed")
    dataset = ViTDetDataset(cfg, image, boxes, right, rescale_factor=2.)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    all_verts, all_joints, all_cams, all_right, all_poses, all_shapes = [], [], [], [], [], []
    overlay = image.copy()
    with torch.inference_mode():
        for batch in loader:
            batch = recursive_to(batch, torch.device(args.device))
            predicted = model(batch)
            cam = predicted["pred_cam"].clone()
            cam[:, 1] *= 2 * batch["right"] - 1
            focal = cfg.EXTRA.FOCAL_LENGTH / cfg.MODEL.IMAGE_SIZE * batch["img_size"].max()
            cams = cam_crop_to_full(cam, batch["box_center"].float(), batch["box_size"].float(),
                                    batch["img_size"].float(), focal).cpu().numpy()
            for index in range(len(cams)):
                hand_right = float(batch["right"][index].item())
                verts = predicted["pred_vertices"][index].cpu().numpy().copy()
                joints = predicted["pred_keypoints_3d"][index].cpu().numpy().copy()
                verts[:, 0] *= 2 * hand_right - 1
                joints[:, 0] *= 2 * hand_right - 1
                projected = verts + cams[index]
                projected = projected[:, :2] / projected[:, 2:3] * float(focal) + np.array([image.shape[1]/2, image.shape[0]/2])
                color = (80, 230, 90) if hand_right else (230, 150, 50)
                for face in model.mano.faces:
                    pts = np.clip(projected[face], -10000, 10000).astype(np.int32)
                    cv2.polylines(overlay, [pts], True, color, 1, cv2.LINE_AA)
                all_verts.append(verts); all_joints.append(joints); all_cams.append(cams[index]); all_right.append(hand_right)
                all_poses.append(predicted["pred_mano_params"]["hand_pose"][index].cpu().numpy())
                all_shapes.append(predicted["pred_mano_params"]["betas"][index].cpu().numpy())
    raw = out / "wilor_first_frame.npz"
    arrays = {"vertices": np.stack(all_verts), "joints": np.stack(all_joints),
              "camera_translation": np.stack(all_cams), "is_right": np.array(all_right),
              "hand_pose_rotation_matrices": np.stack(all_poses), "betas": np.stack(all_shapes),
              "faces": model.mano.faces, "boxes": boxes}
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise RuntimeError("Non-finite WiLoR output")
    np.savez_compressed(raw, **arrays)
    still = np.concatenate([label(image, "Official WiLoR demo: first input frame"),
                            label(overlay, f"WiLoR + MANO: {len(all_verts)} hands, mesh wireframe")], axis=1)
    cv2.imwrite(str(out / "wilor_first_frame.jpg"), still)
    preview = save_video(ROOT / "outputs/previews/wilor_smoke.mp4", [still] * 18, 6)
    return {"checkpoint": relative(checkpoint), "hands": len(all_verts),
            "vertices_shape": list(arrays["vertices"].shape), "array": relative(raw),
            "missing_state_keys": list(incompat.missing_keys), "unexpected_state_keys": list(incompat.unexpected_keys),
            "missing_learned_parameters": learned_missing,
            "inferred_frames": 1, "preview_note": "Still first-frame reconstruction held for three seconds; not temporal hand tracking.",
            "preview": preview}


def run_tracking(args, frames, fps, out):
    import cv2
    import numpy as np
    import torch
    sys.path.insert(0, str(ROOT / "dependencies/tapnet"))
    from tapnet.tapnextpp.votsp2026.model import TAPNextPP
    checkpoint = ROOT / "weights/tapnextpp/tapnextpp_ckpt.pt"
    height, width = frames[0].shape[:2]
    query_mask = None
    if args.query_mask:
        mask = cv2.imread(str(args.query_mask), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Cannot read first-frame query mask: {args.query_mask}")
        original_shape = list(mask.shape)
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST) > 0
        candidates_yx = np.argwhere(mask)
        if len(candidates_yx) < 9:
            raise ValueError(f"Query mask must contain at least 9 foreground pixels after resizing; found {len(candidates_yx)}. No grid fallback.")
        candidates = candidates_yx[:, ::-1].astype(np.float32)
        # Start inside the mask, then deterministically spread nine distinct
        # queries by farthest-point sampling over foreground pixels only.
        interior = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
        first_y, first_x = np.unravel_index(interior.argmax(), interior.shape)
        selected = [np.array([first_x, first_y], dtype=np.float32)]
        nearest_distance = np.full(len(candidates), np.inf, dtype=np.float32)
        for _ in range(8):
            nearest_distance = np.minimum(nearest_distance, ((candidates - selected[-1]) ** 2).sum(axis=1))
            selected.append(candidates[int(nearest_distance.argmax())].copy())
        query = np.stack(selected)
        query_type = "9 farthest-spread foreground pixels from first-frame segmentation mask"
        query_label = "TAPNext++: 9 SAM3-mask queries"
        query_mask = {"path": str(args.query_mask.resolve()), "sha256": digest(args.query_mask),
                      "original_shape": original_shape, "resized_shape": [height, width],
                      "resize_interpolation": "nearest", "foreground_pixels": int(mask.sum())}
    else:
        # Explicit regular-grid queries; these are not SAM/object annotations.
        query = np.array([[x * width, y * height] for y in (.35, .5, .65)
                          for x in (.3, .5, .7)], dtype=np.float32)
        query_type = "regular image grid; no semantic mask"
        query_label = "TAPNext++: 9 first-frame grid queries"
    model = TAPNextPP.from_checkpoint(checkpoint, device=args.device, input_resolution=256)
    positions, visibility, rendered, state = [], [], [], None
    for index, frame in enumerate(frames):
        points, visible, state = model.track_frame(frame, query if index == 0 else None, state=state)
        positions.append(points); visibility.append(visible)
        overlay = frame.copy()
        for point, shown in zip(points, visible):
            if shown and np.isfinite(point).all():
                cv2.circle(overlay, tuple(np.clip(point, -10000, 10000).astype(int)), 4, (80, 255, 70), -1)
        rendered.append(label(overlay, query_label))
    tracks = np.stack(positions)
    if not np.isfinite(tracks).all():
        raise RuntimeError("Non-finite TAPNext++ tracks")
    raw = out / "tapnextpp_tracks.npz"
    np.savez_compressed(raw, positions_xy=tracks, visible=np.stack(visibility), query_xy=query, fps=fps)
    preview = save_video(ROOT / "outputs/previews/tapnextpp_smoke.mp4", rendered, fps)
    return {"checkpoint": relative(checkpoint), "tracks_shape": list(tracks.shape),
            "query_type": query_type, "query_mask": query_mask, "query_xy": query.tolist(), "array": relative(raw),
            "visible_observations": int(np.stack(visibility).sum()), "preview": preview}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=["depth", "wilor", "tracking", "all"], default="all")
    parser.add_argument("--image", type=Path, default=ROOT / "data/smoke_inputs/test1.jpg")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--query-mask", type=Path, help="First-frame segmentation mask for tracking queries; empty masks fail explicitly")
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--fps", type=float, default=6.)
    parser.add_argument("--max-res", type=int, default=512)
    parser.add_argument("--encoder", choices=["vits", "vitl"], default="vits")
    parser.add_argument("--depth-input-size", type=int, default=280)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fp32", action="store_true")
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be positive")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import torch
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; this smoke must run on the GPU server")
    frames, fps, origin = load_frames(args)
    output = ROOT / "outputs/vision_smoke"
    output.mkdir(parents=True, exist_ok=True)
    report = {"created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
              "scope": "Independent public model component smoke tests; not official KinoGen end-to-end reproduction",
              "training": False, "torch": torch.__version__, "device": args.device,
              "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
              "input": origin, "frames": len(frames), "components": {}}
    input_preview = save_video(ROOT / "outputs/previews/vision_smoke_input.mp4",
                               [label(f.copy(), "Smoke input: synthetic camera translation" if not args.video else "Smoke input video") for f in frames], fps)
    report["input_preview"] = input_preview
    runners = {"wilor": run_wilor, "depth": run_depth, "tracking": run_tracking}
    requested = runners if args.component == "all" else {args.component: runners[args.component]}
    for component, runner in requested.items():
        started = time.monotonic()
        try:
            value = runner(args, frames, fps, output)
            value["status"] = "passed"
        except Exception as exc:
            traceback.print_exc()
            value = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        value["elapsed_seconds"] = round(time.monotonic() - started, 3)
        report["components"][component] = value
        write_json(ROOT / f"reports/vision-smoke-{args.component}.json", report)
        print(json.dumps({"component": component, **value}, ensure_ascii=False), flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 0 if all(v["status"] == "passed" for v in report["components"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
