#!/usr/bin/env python3
"""Offline SAM 3 image text segmentation, independent of the KinoGen pipeline.

The upstream README requires Python >=3.12, PyTorch >=2.7 and CUDA >=12.6.
Older eager-mode runtimes require an explicit --allow-unsupported-runtime flag;
they are experiments, not upstream-supported configurations.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import sys
import time
import traceback

from smoke_vision import ROOT, digest, label, relative, save_video, write_json


def version_pair(value):
    numbers = re.findall(r"\d+", str(value))
    return tuple(map(int, numbers[:2])) if len(numbers) >= 2 else (0, 0)


def inspect_runtime(args):
    import torch
    import torchvision
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from torchvision.transforms import v2
    from torch.utils._pytree import tree_map_only
    packages = {}
    for name in ("torch", "torchvision", "numpy", "timm", "ftfy", "regex", "iopath",
                 "huggingface-hub", "einops", "pycocotools", "scipy", "psutil", "triton", "setuptools"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    declared = (sys.version_info[:2] >= (3, 12)
                and version_pair(torch.__version__) >= (2, 7)
                and version_pair(torch.version.cuda) >= (12, 6))
    symbols = {"compiler_is_dynamo_compiling": hasattr(torch.compiler, "is_dynamo_compiling"),
               "sdpa_kernel": callable(sdpa_kernel), "sdpa_flash_backend": hasattr(SDPBackend, "FLASH_ATTENTION"),
               "torchvision_v2": hasattr(v2, "ToDtype"), "tree_map_only": callable(tree_map_only),
               "aten_addmm_activation": hasattr(torch.ops.aten, "_addmm_activation")}
    runtime = {"python": platform.python_version(), "torch": torch.__version__,
               "torchvision": torchvision.__version__, "cuda_runtime": torch.version.cuda,
               "cuda_available": torch.cuda.is_available(), "packages": packages,
               "required_symbols": symbols,
               "meets_upstream_readme_versions": declared,
               "mode": "image/eager, compile=False, FA3 disabled, no video tracker",
               "support_note": "Version checks do not prove inference compatibility; only the completed model run does."}
    if torch.cuda.is_available():
        runtime.update(gpu=torch.cuda.get_device_name(), bf16_supported=torch.cuda.is_bf16_supported())
    if not all(symbols.values()):
        raise RuntimeError(f"Required APIs missing: {symbols}")
    # The builder imports tracking modules even for image-only inference. Import
    # here exposes those dependencies without downloading or running a model.
    sys.path.insert(0, str(ROOT / "dependencies/sam3"))
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    runtime["sam3_import_passed"] = True
    return runtime


def run(args, report):
    import cv2
    import numpy as np
    from PIL import Image
    import torch
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    if not report["runtime"]["meets_upstream_readme_versions"] and not args.allow_unsupported_runtime:
        raise RuntimeError("Runtime is below upstream README requirements. Explicit --allow-unsupported-runtime is required for the isolated eager-mode experiment.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; do not run this smoke on the local preview machine")
    for path in (args.checkpoint, args.bpe, args.image):
        if not path.is_file():
            raise FileNotFoundError(path)
    checkpoint_sha = digest(args.checkpoint)
    if args.expected_sha256 and checkpoint_sha != args.expected_sha256:
        raise RuntimeError(f"SAM3 checkpoint SHA256 mismatch: {checkpoint_sha}")
    report["checkpoint"] = {"path": str(args.checkpoint), "bytes": args.checkpoint.stat().st_size,
                             "sha256": checkpoint_sha}
    report["tokenizer"] = {"path": str(args.bpe), "sha256": digest(args.bpe)}
    # Observe upstream state loading to reject a partially initialised model.
    # A temporary class wrapper is restored immediately after building. The
    # immutable upstream snapshot is never edited.
    original_load = torch.nn.Module.load_state_dict
    loads = []
    def recorded_load(module, *positional, **keywords):
        result = original_load(module, *positional, **keywords)
        if module.__class__.__name__ == "Sam3Image":
            learned = set(dict(module.named_parameters()))
            loads.append({"missing_keys": list(result.missing_keys),
                          "unexpected_keys": list(result.unexpected_keys),
                          "missing_learned_parameters": sorted(learned.intersection(result.missing_keys))})
        return result
    try:
        torch.nn.Module.load_state_dict = recorded_load
        model = build_sam3_image_model(
            checkpoint_path=str(args.checkpoint), bpe_path=str(args.bpe),
            load_from_HF=False, device="cuda", eval_mode=True,
            enable_segmentation=True, enable_inst_interactivity=False, compile=False)
    finally:
        torch.nn.Module.load_state_dict = original_load
    report["state_dict_loads"] = loads
    if not loads or any(row["missing_learned_parameters"] for row in loads):
        raise RuntimeError(f"Model learned weights were not completely loaded: {loads}")
    model.eval()
    processor = Sam3Processor(model, device="cuda", confidence_threshold=args.threshold)
    image = Image.open(args.image).convert("RGB")
    rgb = np.asarray(image)
    report["input"] = {"path": str(args.image), "sha256": digest(args.image),
                        "width": image.width, "height": image.height}
    output = ROOT / "outputs/sam3_smoke"
    output.mkdir(parents=True, exist_ok=True)
    report["prompts"] = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = processor.set_image(image)
        for prompt in args.prompts:
            processor.reset_all_prompts(state)
            result = processor.set_text_prompt(prompt=prompt, state=state)
            masks = result["masks"].detach().cpu().numpy().reshape(-1, image.height, image.width)
            scores = result["scores"].detach().float().cpu().numpy()
            boxes = result["boxes"].detach().float().cpu().numpy()
            if not np.isfinite(scores).all() or not np.isfinite(boxes).all():
                raise RuntimeError(f"Non-finite SAM3 predictions for {prompt}")
            safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", prompt)
            npz = output / f"{safe}_segmentation.npz"
            np.savez_compressed(npz, masks=masks, scores=scores, boxes_xyxy=boxes)
            union = masks.any(axis=0) if len(masks) else np.zeros(rgb.shape[:2], dtype=bool)
            mask_png = output / f"{safe}_mask.png"
            Image.fromarray(union.astype(np.uint8) * 255).save(mask_png)
            overlay = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
            for index, (mask, score, box) in enumerate(zip(masks, scores, boxes)):
                color = np.array([(53 + index*71) % 255, (220 + index*31) % 255,
                                  (65 + index*107) % 255], dtype=np.uint8)
                overlay[mask] = (.55 * overlay[mask] + .45 * color).astype(np.uint8)
                x0, y0, x1, y1 = np.round(box).astype(int)
                cv2.rectangle(overlay, (x0, y0), (x1, y1), tuple(map(int, color)), 1)
                cv2.putText(overlay, f"{float(score):.2f}", (x0, max(15, y0)),
                            cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
            labelled = label(overlay, f"SAM3 image: '{prompt}', {len(masks)} masks (still)")
            png = output / f"{safe}_overlay.png"
            cv2.imwrite(str(png), labelled)
            preview = save_video(ROOT / f"outputs/previews/sam3_{safe}_smoke.mp4", [labelled] * 18, 6)
            report["prompts"].append({"prompt": prompt, "instances": len(masks),
                                      "scores": scores.tolist(), "union_area_pixels": int(union.sum()),
                                      "mask": relative(mask_png), "overlay": relative(png),
                                      "array": relative(npz), "preview": preview})
    report["nonempty_prompts"] = sum(row["instances"] > 0 for row in report["prompts"])
    report["status"] = "passed" if report["nonempty_prompts"] else "ran_but_no_masks"
    report["note"] = "One image segmented per prompt. MP4 holds the static result for three seconds; no SAM3 video tracking or full KinoGen reproduction."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=ROOT / "data/smoke_inputs/test1.jpg")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "weights/sam3/sam3.pt")
    parser.add_argument("--bpe", type=Path, default=ROOT / "dependencies/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz")
    parser.add_argument("--expected-sha256", default="9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e")
    parser.add_argument("--prompts", nargs="+", default=["hand", "person"])
    parser.add_argument("--threshold", type=float, default=.3)
    parser.add_argument("--preflight", action="store_true", help="Import/API checks only; no model creation or GPU forward")
    parser.add_argument("--allow-unsupported-runtime", action="store_true")
    args = parser.parse_args()
    # Explicit offline flags prevent implicit HF traffic, even on an import path.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    report = {"created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
              "scope": "SAM3 independent image segmentation smoke", "training": False,
              "compile": False, "hf_downloads": False,
              "preflight_only": args.preflight, "allow_unsupported_runtime": args.allow_unsupported_runtime}
    started = time.monotonic()
    try:
        report["runtime"] = inspect_runtime(args)
        if args.preflight:
            report["status"] = "preflight_passed_inference_not_tested"
        else:
            run(args, report)
    except Exception as exc:
        traceback.print_exc()
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    write_json(ROOT / ("reports/sam3-preflight.json" if args.preflight else "reports/sam3-smoke.json"), report)
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["status"] in ("passed", "preflight_passed_inference_not_tested") else 2


if __name__ == "__main__":
    raise SystemExit(main())
