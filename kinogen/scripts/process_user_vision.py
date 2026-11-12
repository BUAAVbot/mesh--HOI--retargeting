#!/usr/bin/env python3
"""Process every frame and both uncalibrated views of the user's generated video.

SAM3 uses independent image/text inference for each frame, not its video tracker.
VDA Large estimates relative, nonmetric depth. No training or network API calls.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
VIDEO_SHA = "035c5852f9580cbbf5d612e2ad93a7f80065426a34ce355fb87db43876b20d6b"
SAM_SHA = "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"
VDA_SHA = "43df27c6b396042ba34ff7b798ab279f64d204d2e86d7a373968f8fa36d0e6fa"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path):
    return str(path.resolve().relative_to(ROOT)).replace("\\", "/")


def json_save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    temporary.replace(path)


def npz_save(path, **values):
    import numpy as np
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **values)
    temporary.replace(path)


def update_report(stage, record):
    path = ROOT / "reports/user-vision.json"
    report = json.loads(path.read_text(encoding="utf8")) if path.exists() else {}
    report.update(updated_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  source_video_sha256=VIDEO_SHA, training=False,
                  scope="Full 121-frame independent vision components on two candidate views; not calibrated multi-view KinoGen reproduction")
    report.setdefault("stages", {})[stage] = record
    json_save(path, report)


def load_input(args):
    import numpy as np
    actual = sha256(args.video)
    if actual != VIDEO_SHA:
        raise ValueError(f"Source video SHA256 mismatch: {actual}")
    with np.load(args.input, allow_pickle=False) as data:
        rgb = data["rgb"]
        timestamps = data["timestamps"]
        invalid = data["watermark_invalid"]
    if rgb.shape != (121, 2, 720, 640, 3) or rgb.dtype != np.uint8:
        raise ValueError(f"Unexpected full-video RGB shape/dtype: {rgb.shape}, {rgb.dtype}")
    if invalid.shape != rgb.shape[:-1] or invalid.dtype != np.bool_:
        raise ValueError(f"Unexpected watermark-invalid shape/dtype: {invalid.shape}, {invalid.dtype}")
    if timestamps.shape != (121,) or not np.isfinite(timestamps).all() or not (np.diff(timestamps) > 0).all():
        raise ValueError("Invalid complete input timestamps")
    return rgb, timestamps, invalid, {"video": rel(args.video), "sha256": actual,
                                      "input_array": rel(args.input), "input_array_sha256": sha256(args.input),
                                      "rgb_array_sha256": hashlib.sha256(memoryview(rgb)).hexdigest(),
                                      "frames": 121, "views": ["left", "right"], "view_hw": [720, 640],
                                      "coordinate_system": "Uncropped candidate half-view pixels, x=0..639,y=0..719",
                                      "frame_indices": list(range(121)), "timestamps_seconds": timestamps.tolist(),
                                      "watermark_invalid_pixels": int(invalid.sum()),
                                      "watermark_policy": "Input pixels remain unchanged; marked pixels excluded from observations"}


def runtime():
    import torch
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "6":
        raise RuntimeError("This user-video task is assigned exclusively to physical GPU6; require CUDA_VISIBLE_DEVICES=6")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU server CUDA unavailable")
    return {"python": sys.version, "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "physical_gpu": 6,
            "gpu": torch.cuda.get_device_name(), "ld_preload": os.environ.get("LD_PRELOAD", ""),
            "compile": False, "network_api_calls": False}


def preview(path, images, timestamps):
    import cv2
    import imageio.v2 as imageio
    fps = float(1 / __import__("numpy").median(__import__("numpy").diff(timestamps)))
    with imageio.get_writer(str(path), fps=fps, codec="libx264", macro_block_size=1,
                            ffmpeg_params=["-crf", "20", "-pix_fmt", "yuv420p"]) as writer:
        for frame in images:
            writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap = cv2.VideoCapture(str(path))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, first = cap.read()
    cap.release()
    if not ok or count != 121:
        raise RuntimeError(f"Preview frame count mismatch: {path}, {count}")
    return {"path": rel(path), "sha256": sha256(path), "frames": count,
            "fps": fps, "width": int(first.shape[1]), "height": int(first.shape[0])}


def annotate(image, text):
    import cv2
    cv2.rectangle(image, (0, 0), (image.shape[1], 28), (20, 20, 20), -1)
    cv2.putText(image, text, (7, 20), cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def segment(args, record, rgb, timestamps, invalid):
    import cv2
    import numpy as np
    from PIL import Image
    import torch
    sys.path.insert(0, str(ROOT / "dependencies/sam3"))
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    checkpoint = ROOT / "weights/sam3/sam3.pt"
    checkpoint_sha = sha256(checkpoint)
    if checkpoint_sha != SAM_SHA:
        raise ValueError("SAM3 checkpoint checksum mismatch")
    bpe = ROOT / "dependencies/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    record.update(checkpoint={"path": rel(checkpoint), "sha256": checkpoint_sha},
                  tokenizer_sha256=sha256(bpe), algorithm="SAM3 independent image text-prompt segmentation per frame/view",
                  runtime_support="Python3.10/Torch2.5 eager compatibility configuration previously tested; below official README versions",
                  confidence_threshold=args.threshold, prompts={"robot": "robot", "box": "cardboard box"},
                  processed_images=0)
    update_report("segmentation", record)
    loads = []
    original = torch.nn.Module.load_state_dict
    def checked_load(module, *values, **keywords):
        result = original(module, *values, **keywords)
        if module.__class__.__name__ == "Sam3Image":
            missing = sorted(set(result.missing_keys) & set(dict(module.named_parameters())))
            loads.append({"missing_learned_parameters": missing,
                          "missing_keys": list(result.missing_keys), "unexpected_keys": list(result.unexpected_keys)})
        return result
    try:
        torch.nn.Module.load_state_dict = checked_load
        model = build_sam3_image_model(checkpoint_path=str(checkpoint), bpe_path=str(bpe),
                                      load_from_HF=False, compile=False, device="cuda", eval_mode=True,
                                      enable_segmentation=True, enable_inst_interactivity=False)
    finally:
        torch.nn.Module.load_state_dict = original
    if not loads or any(row["missing_learned_parameters"] for row in loads):
        raise RuntimeError(f"Incomplete SAM3 learned weights: {loads}")
    record["state_dict_loads"] = loads
    processor = Sam3Processor(model.eval(), device="cuda", confidence_threshold=args.threshold)
    shape = rgb.shape[:-1]
    masks = {name: np.zeros(shape, dtype=bool) for name in ("robot", "box")}
    raw_masks = {name: np.zeros(shape, dtype=bool) for name in masks}
    scores = {name: np.zeros(shape[:2], dtype=np.float32) for name in masks}
    valid = {name: np.zeros(shape[:2], dtype=bool) for name in masks}
    counts = {name: np.zeros(shape[:2], dtype=np.int16) for name in masks}
    observations = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for frame in range(len(rgb)):
            for view in range(2):
                state = processor.set_image(Image.fromarray(rgb[frame, view]))
                for name, prompt in (("robot", "robot"), ("box", "cardboard box")):
                    processor.reset_all_prompts(state)
                    result = processor.set_text_prompt(prompt=prompt, state=state)
                    prediction = result["masks"].detach().cpu().numpy().reshape(-1, 720, 640)
                    confidence = result["scores"].detach().float().cpu().numpy()
                    bbox = result["boxes"].detach().float().cpu().numpy()
                    if not np.isfinite(confidence).all() or not np.isfinite(bbox).all():
                        raise RuntimeError(f"Non-finite prediction at {frame}/{view}/{name}")
                    union = prediction.any(axis=0) if len(prediction) else np.zeros((720, 640), dtype=bool)
                    observed = union & ~invalid[frame, view]
                    raw_masks[name][frame, view] = union
                    masks[name][frame, view] = observed
                    scores[name][frame, view] = float(confidence.max()) if len(confidence) else 0.
                    valid[name][frame, view] = observed.any()
                    counts[name][frame, view] = len(prediction)
                    observations.append({"frame": frame, "view": view, "class": name,
                                         "scores": confidence.tolist(), "boxes_xyxy": bbox.tolist(),
                                         "raw_area": int(union.sum()), "valid_area": int(observed.sum()),
                                         "valid": bool(observed.any()), "missing": not bool(observed.any())})
                del state
                record["processed_images"] += 1
            if frame % 10 == 0 or frame == len(rgb) - 1:
                print(f"SAM3 complete frame {frame + 1}/{len(rgb)}, images={record['processed_images']}", flush=True)
                update_report("segmentation", record)
    path = ROOT / "outputs/user_reconstruction/segmentation.npz"
    npz_save(path, robot=masks["robot"], box=masks["box"],
             robot_raw_for_qa=raw_masks["robot"], box_raw_for_qa=raw_masks["box"],
             robot_score=scores["robot"], box_score=scores["box"],
             robot_valid=valid["robot"], box_valid=valid["box"],
             robot_instance_count=counts["robot"], box_instance_count=counts["box"],
             observation_valid=~invalid, timestamps=timestamps,
             source_video_sha256=np.asarray(VIDEO_SHA), checkpoint_sha256=np.asarray(checkpoint_sha),
             frame_indices=np.arange(len(rgb)), view_names=np.asarray(["left", "right"]))
    json_save(path.with_name("segmentation_instances.json"), {"observations": observations})
    record.update(array={"path": rel(path), "sha256": sha256(path)},
                  raw_mask_policy="*_raw_for_qa are unfiltered model unions only, not accepted observations; robot/box exclude invalid pixels",
                  observations={name: {"valid": int(valid[name].sum()), "missing": int((~valid[name]).sum())} for name in masks},
                  processed_frames=121, processed_views=2, status="inference_complete_preview_pending")
    update_report("segmentation", record)
    def images():
        for frame in range(121):
            views = []
            for view in range(2):
                image = cv2.cvtColor(rgb[frame, view], cv2.COLOR_RGB2BGR)
                for name, color in (("robot", np.asarray([60, 210, 70])), ("box", np.asarray([20, 150, 250]))):
                    mask = masks[name][frame, view]
                    image[mask] = (.58 * image[mask] + .42 * color).astype(np.uint8)
                text = f"SAM3 frame {frame:03d} {'left' if view == 0 else 'right'}: robot {scores['robot'][frame, view]:.2f} box {scores['box'][frame, view]:.2f}"
                views.append(annotate(image, text))
            yield np.concatenate(views, axis=1)
    record["preview"] = preview(ROOT / "outputs/previews/user_segmentation.mp4", images(), timestamps)
    record["status"] = "passed"
    update_report("segmentation", record)


def depth(args, record, rgb, timestamps, invalid):
    import cv2
    import numpy as np
    import torch
    sys.path.insert(0, str(ROOT / "dependencies/video-depth-anything"))
    from video_depth_anything.video_depth import VideoDepthAnything
    checkpoint = ROOT / "weights/video-depth-anything-large/video_depth_anything_vitl.pth"
    checkpoint_sha = sha256(checkpoint)
    if checkpoint_sha != VDA_SHA:
        raise ValueError("VDA Large checkpoint checksum mismatch")
    record.update(checkpoint={"path": rel(checkpoint), "sha256": checkpoint_sha},
                  algorithm="VideoDepthAnything Large, full sequential video inference independently per candidate view",
                  depth_units="Relative/nonmetric; views have independent affine depth ambiguity",
                  processed_views=0, inference_input_sizes=[])
    model = VideoDepthAnything(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024])
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    model = model.to("cuda").eval()
    relative_depth = np.empty(rgb.shape[:-1], dtype=np.float32)
    fps = float(1 / np.median(np.diff(timestamps)))
    update_report("depth", record)
    for view in range(2):
        for input_size in sorted(set([args.depth_input_size, 420, 364, 280]), reverse=True):
            if input_size > args.depth_input_size:
                continue
            try:
                print(f"VDA Large view={view} frames=121 network_input_size={input_size}", flush=True)
                with torch.inference_mode():
                    predicted, returned_fps = model.infer_video_depth(rgb[:, view], fps,
                                                                    input_size=input_size, device="cuda", fp32=False)
                predicted = np.asarray(predicted, dtype=np.float32)
                if predicted.shape != (121, 720, 640) or not np.isfinite(predicted).all():
                    raise RuntimeError(f"Invalid full-video depth output: shape {predicted.shape}")
                if float(predicted.max() - predicted.min()) <= 0:
                    raise RuntimeError("Constant depth output")
                relative_depth[:, view] = predicted
                record["inference_input_sizes"].append(input_size)
                record["processed_views"] += 1
                record.setdefault("range_per_view", []).append([float(predicted.min()), float(predicted.max())])
                del predicted
                update_report("depth", record)
                break
            except torch.cuda.OutOfMemoryError:
                record.setdefault("oom_retries", []).append({"view": view, "input_size": input_size})
                gc.collect()
                torch.cuda.empty_cache()
                update_report("depth", record)
                if input_size == 280:
                    raise
        else:
            raise RuntimeError(f"Could not run VDA Large for view {view}")
    path = ROOT / "outputs/user_reconstruction/depth.npz"
    npz_save(path, relative_depth=relative_depth, observation_valid=~invalid,
             timestamps=timestamps, source_video_sha256=np.asarray(VIDEO_SHA),
             checkpoint_sha256=np.asarray(checkpoint_sha), frame_indices=np.arange(121),
             view_names=np.asarray(["left", "right"]), units=np.asarray("relative_nonmetric"),
             inference_input_sizes=np.asarray(record["inference_input_sizes"]))
    record.update(array={"path": rel(path), "sha256": sha256(path)}, processed_frames=121,
                  shape=list(relative_depth.shape), strict_state_dict=True,
                  watermark_policy="Depth values retained as model outputs; observation_valid=False on watermark pixels",
                  status="inference_complete_preview_pending")
    update_report("depth", record)
    record["preview"] = depth_preview(relative_depth, invalid, timestamps)
    record["status"] = "passed"
    update_report("depth", record)


def depth_preview(relative_depth, invalid, timestamps):
    import cv2
    import numpy as np
    limits = [np.percentile(relative_depth[:, view][~invalid[:, view]], [1, 99]) for view in range(2)]
    def images():
        for frame in range(121):
            views = []
            for view in range(2):
                lo, hi = limits[view]
                gray = (np.clip((relative_depth[frame, view] - lo) / max(hi-lo, 1e-6), 0, 1) * 255).astype(np.uint8)
                colored = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
                colored[invalid[frame, view]] = [70, 70, 70]
                text = f"VDA Large frame {frame:03d} {'left' if view == 0 else 'right'}: RELATIVE depth"
                views.append(annotate(colored, text))
            yield np.concatenate(views, axis=1)
    return preview(ROOT / "outputs/previews/user_depth.mp4", images(), timestamps)


def refresh_depth_validity(args, new_input, timestamps, invalid):
    """Update QA exclusions while preserving every original model depth value."""
    import numpy as np
    report_path = ROOT / "reports/user-vision.json"
    report = json.loads(report_path.read_text(encoding="utf8"))
    record = report["stages"]["depth"]
    if record.get("status") != "passed":
        raise RuntimeError("Cannot refresh validity before a successful full depth inference")
    path = ROOT / "outputs/user_reconstruction/depth.npz"
    old_file_sha = sha256(path)
    with np.load(path, allow_pickle=False) as data:
        payload = {key: data[key] for key in data.files}
    values = payload["relative_depth"]
    depth_values_sha = hashlib.sha256(memoryview(values)).hexdigest()
    old_input = record["input"]
    if old_input["sha256"] != new_input["sha256"]:
        raise RuntimeError("Source video changed; validity-only update is unsafe")
    if old_input.get("rgb_array_sha256") and old_input["rgb_array_sha256"] != new_input["rgb_array_sha256"]:
        raise RuntimeError("RGB array changed; validity-only update is unsafe")
    if not np.array_equal(payload["timestamps"], timestamps):
        raise RuntimeError("Depth timestamps changed")
    payload["observation_valid"] = ~invalid
    npz_save(path, **payload)
    with np.load(path, allow_pickle=False) as check:
        if hashlib.sha256(memoryview(check["relative_depth"])).hexdigest() != depth_values_sha:
            raise RuntimeError("Relative depth values changed during validity refresh")
    record.setdefault("validity_refreshes", []).append({
        "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "previous_depth_file_sha256": old_file_sha,
        "previous_input_array_sha256": old_input["input_array_sha256"],
        "new_input_array_sha256": new_input["input_array_sha256"],
        "relative_depth_values_sha256": depth_values_sha,
        "relative_depth_values_unchanged": True,
        "gpu_inference_repeated": False})
    record["input"] = new_input
    record["array"] = {"path": rel(path), "sha256": sha256(path)}
    record["preview"] = depth_preview(values, invalid, timestamps)
    update_report("depth", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["segmentation", "depth", "refresh-depth-validity"], required=True)
    parser.add_argument("--video", type=Path, default=ROOT / "data/video_generation/candidate_scene/generated.mp4")
    parser.add_argument("--input", type=Path, default=ROOT / "outputs/user_reconstruction/input.npz")
    parser.add_argument("--threshold", type=float, default=.3)
    parser.add_argument("--depth-input-size", type=int, default=518)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("MPLBACKEND", "Agg")
    record = {"started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "status": "running"}
    start = time.monotonic()
    try:
        record["runtime"] = runtime()
        rgb, timestamps, invalid, provenance = load_input(args)
        record["input"] = provenance
        (ROOT / "outputs/user_reconstruction").mkdir(parents=True, exist_ok=True)
        (ROOT / "outputs/previews").mkdir(parents=True, exist_ok=True)
        if args.stage == "refresh-depth-validity":
            record = refresh_depth_validity(args, provenance, timestamps, invalid)
        else:
            update_report(args.stage, record)
            (segment if args.stage == "segmentation" else depth)(args, record, rgb, timestamps, invalid)
    except Exception as exc:
        traceback.print_exc()
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    if args.stage == "refresh-depth-validity" and record.get("status") == "passed":
        record["validity_refresh_elapsed_seconds"] = round(time.monotonic() - start, 3)
        record["validity_updated_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    else:
        record["elapsed_seconds"] = round(time.monotonic() - start, 3)
        record["finished_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    update_report("depth" if args.stage == "refresh-depth-validity" else args.stage, record)
    print(json.dumps({"stage": args.stage, **record}, ensure_ascii=False), flush=True)
    return 0 if record["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
