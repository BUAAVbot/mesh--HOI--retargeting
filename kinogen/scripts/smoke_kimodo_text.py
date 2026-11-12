#!/usr/bin/env python3
"""Offline original Llama3 + two LLM2Vec adapters -> real text-conditioned Kimodo.

Run only on server physical GPU 6/7. Encode one prompt at a time, save genuine
embeddings with provenance, release the 8B model, then sample 90-frame motions.
This is a public-component check, not quantitative language-alignment validation
or an end-to-end KinoGen reproduction. No training or external APIs are used.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def record_file(path):
    path = Path(path)
    return {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": digest(path)}


def bidirectional_preflight(torch, LlamaConfig, LlamaBiModel):
    """Tiny random CPU model checks API semantics, never supplies real embeddings."""
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=64, use_cache=False)
    cfg._attn_implementation = "eager"
    torch.manual_seed(123)
    model = LlamaBiModel(cfg).eval()
    a = torch.tensor([[1, 2, 3, 4]])
    b = torch.tensor([[1, 2, 3, 5]])
    with torch.inference_mode():
        x = model(input_ids=a, attention_mask=torch.ones_like(a)).last_hidden_state
        y = model(input_ids=b, attention_mask=torch.ones_like(b)).last_hidden_state
    difference = float((x[:, 0] - y[:, 0]).abs().max())
    if not difference > 1e-6:
        raise RuntimeError("Bidirectional semantics failed: first token cannot see a changed future token. Do not use this runtime for LLM2Vec.")
    del model
    return {"status": "passed", "test_model": "tiny random CPU model; not used for motion conditioning", "first_token_future_dependence_max_abs": difference}


def verify_loaded_adapter(torch, model, directory):
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    expected = load_file(str(directory / "adapter_model.safetensors"), device="cpu")
    loaded = get_peft_model_state_dict(model, adapter_name="default")
    if set(expected) != set(loaded):
        raise RuntimeError(f"Adapter state keys did not load exactly: {directory.name}")
    for key, value in expected.items():
        if not torch.equal(value.float(), loaded[key].detach().cpu().float()):
            raise RuntimeError(f"Loaded adapter differs from published tensor: {key}")
    return {"adapter": directory.name, "keys": len(expected), "all_tensors_match_published_checkpoint": True}


def render(joints, parents, names, fps, path, prompt):
    import av
    import numpy as np
    from PIL import Image, ImageDraw
    width, height = 960, 640
    center = (joints.min(axis=(0, 1)) + joints.max(axis=(0, 1))) / 2
    extent = max(float(np.max(joints.max(axis=(0, 1)) - joints.min(axis=(0, 1)))), 1.5) * 1.25
    scale = 420 / extent
    def point(position, axis, panel):
        return int(panel * 480 + 240 + (position[axis] - center[axis]) * scale), int(350 - (position[1] - center[1]) * scale)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=int(fps))
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        stream.options = {"crf": "20", "preset": "medium"}
        for index, pose in enumerate(joints):
            image = Image.new("RGB", (width, height), "#101827")
            draw = ImageDraw.Draw(image)
            draw.text((20, 12), "KIMODO G1 | ORIGINAL LLAMA3 + LLM2VEC TEXT CONDITION", fill="white")
            draw.text((20, 32), prompt[:130], fill="#c6d6e8")
            draw.text((20, 50), "Actual model sample; no quantitative semantic-accuracy claim; not full KinoGen", fill="#a8bad0")
            for panel, axis, title in ((0, 0, "FRONT X/Y"), (1, 2, "SIDE Z/Y")):
                draw.text((panel * 480 + 18, 77), title, fill="#c6d6e8")
                for joint_index, parent in enumerate(parents):
                    colour = "#60a5fa" if names[joint_index].startswith("left_") else "#fb7185" if names[joint_index].startswith("right_") else "#e2e8f0"
                    xy = point(pose[joint_index], axis, panel)
                    if int(parent) >= 0:
                        draw.line((point(pose[int(parent)], axis, panel), xy), fill=colour, width=5)
                    draw.ellipse((xy[0] - 3, xy[1] - 3, xy[0] + 3, xy[1] + 3), fill=colour)
            draw.line((480, 75, 480, 600), fill="#334155", width=2)
            draw.text((20, 610), f"{index + 1}/{len(joints)} frames | {fps:g} FPS | world-space skeleton", fill="#c6d6e8")
            frame = av.VideoFrame.from_ndarray(np.asarray(image), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    with av.open(str(path)) as container:
        count = sum(1 for _ in container.decode(video=0))
    if count != len(joints):
        raise RuntimeError(f"Video decode frame mismatch: {count} != {len(joints)}")
    return count


def run(args, report):
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from transformers import AutoConfig, AutoTokenizer, LlamaConfig
    from peft import PeftModel
    sys.path.insert(0, str(ROOT / "dependencies/kimodo"))
    from kimodo.model.llm2vec.models.bidirectional_llama import LlamaBiModel
    from kimodo.model.llm2vec.llm2vec import LLM2Vec
    from kimodo.model.loading import instantiate_from_dict
    from kimodo.sanitize import sanitize_texts
    runtime = json.loads(args.runtime.read_text(encoding="utf-8"))
    report["runtime_config"] = record_file(args.runtime)
    report["versions"] = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "tokenizers", "huggingface-hub")}
    for name, version in runtime["packages"].items():
        if report["versions"][name] != version:
            raise RuntimeError(f"Use the dedicated kimodo-text overlay: expected {name}=={version}, got {report['versions'][name]}")
    report["bidirectional_preflight"] = bidirectional_preflight(torch, LlamaConfig, LlamaBiModel)
    print("Bidirectional tiny-model preflight passed", flush=True)
    if args.preflight:
        report["status"] = "preflight_passed_real_weights_not_loaded"
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one physical GPU (6 or 7) to avoid LLM2Vec multiprocessing")
    base = ROOT / runtime["base_directory"]
    mntp = ROOT / runtime["mntp_directory"]
    supervised = ROOT / runtime["supervised_directory"]
    model_dir = ROOT / runtime["kimodo_directory"]
    report["verified_base_files"] = []
    for item in runtime["base_expected_files"]:
        actual = record_file(base / item["path"])
        if actual["bytes"] != item["bytes"] or actual["sha256"] != item["sha256"]:
            raise RuntimeError(f"Original base-model integrity mismatch: {item['path']}")
        report["verified_base_files"].append(actual)
    report["adapter_files"] = []
    for key, directory in (("mntp", mntp), ("supervised", supervised)):
        config = json.loads((directory / "adapter_config.json").read_text())
        if config["base_model_name_or_path"] != runtime["base_model_identity"]:
            raise RuntimeError(f"Unexpected {key} adapter base identity")
        weight = record_file(directory / "adapter_model.safetensors")
        if weight["sha256"] != runtime["adapter_expected_sha256"][key]:
            raise RuntimeError(f"Unexpected {key} adapter checkpoint hash")
        report["adapter_files"].extend((weight, record_file(directory / "adapter_config.json")))
    tokenizer = AutoTokenizer.from_pretrained(str(mntp), local_files_only=True, trust_remote_code=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    report["tokenizer_files"] = [record_file(mntp / name) for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")]
    config = AutoConfig.from_pretrained(str(base), local_files_only=True, trust_remote_code=False)
    config._name_or_path = runtime["base_model_identity"]
    config._attn_implementation = runtime["attention_implementation"]
    config.use_cache = False
    dtype = getattr(torch, runtime["encoding_dtype"])
    print("Loading local original Llama3 base; no Hub model resolution", flush=True)
    started = time.perf_counter()
    encoder_model, loading_info = LlamaBiModel.from_pretrained(str(base), config=config, local_files_only=True,
        trust_remote_code=False, torch_dtype=dtype, low_cpu_mem_usage=True,
        attn_implementation=runtime["attention_implementation"], output_loading_info=True)
    if loading_info.get("missing_keys") or loading_info.get("mismatched_keys") or loading_info.get("error_msgs"):
        raise RuntimeError(f"Base model did not completely load: {loading_info}")
    if any(key != "lm_head.weight" for key in loading_info.get("unexpected_keys", [])):
        raise RuntimeError(f"Unexpected base-model tensor mapping: {loading_info}")
    report["base_loading_info"] = loading_info
    encoder_model.config._name_or_path = runtime["base_model_identity"]
    # Explicitly supplied model + local adapter directories prevent the PEFT
    # adapter_config base name from resolving through a gated remote repo.
    encoder_model = PeftModel.from_pretrained(encoder_model, str(mntp), is_trainable=False, local_files_only=True)
    report["loaded_adapter_checks"] = [verify_loaded_adapter(torch, encoder_model, mntp)]
    encoder_model = encoder_model.merge_and_unload()
    encoder_model = PeftModel.from_pretrained(encoder_model, str(supervised), is_trainable=False, local_files_only=True)
    report["loaded_adapter_checks"].append(verify_loaded_adapter(torch, encoder_model, supervised))
    encoder_model.config._name_or_path = runtime["base_model_identity"]
    settings = {key: runtime[key] for key in ("pooling_mode", "max_length", "doc_max_length", "skip_instruction")}
    optional_settings = supervised / "llm2vec_config.json"
    if optional_settings.exists():
        settings.update(json.loads(optional_settings.read_text()))
    encoder = LLM2Vec(model=encoder_model, tokenizer=tokenizer, **settings).eval().to("cuda:0")
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    texts = sanitize_texts(args.prompt or ["A person walks forward.", "A person raises both arms above their head."])
    if not texts or len(set(texts)) != len(texts) or any(not text.strip() for text in texts):
        raise RuntimeError("Provide one or more distinct, nonempty prompts")
    with torch.inference_mode():
        features = encoder.encode(texts, batch_size=1, show_progress_bar=True, convert_to_tensor=True, device="cuda:0")
    features = features.detach().cpu().float().contiguous()
    if tuple(features.shape) != (len(texts), 4096) or not bool(torch.isfinite(features).all()):
        raise RuntimeError(f"Invalid genuine text-feature tensor: {tuple(features.shape)}")
    differences = {(str(i) + "-" + str(j)): float((features[i] - features[j]).abs().max()) for i in range(len(texts)) for j in range(i + 1, len(texts))}
    if len(texts) >= 2 and not all(value > 1e-6 for value in differences.values()):
        raise RuntimeError("Different prompts produced indistinguishable features")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = ROOT / "outputs/kimodo_text" / stamp
    output.mkdir(parents=True, exist_ok=False)
    cached = output / "llm2vec_features.npz"
    np.savez_compressed(cached, prompts=np.asarray(texts), text_features=features.numpy()[:, None], text_lengths=np.ones(len(texts), dtype=np.int64))
    report.update({"encoding_seconds": round(time.perf_counter() - started, 3),
        "prompts": texts, "encoder_settings": settings, "encoding_dtype": str(dtype),
        "features_shape": [len(texts), 1, 4096], "features_all_finite": True,
        "feature_comparison_status": "passed_distinct_features" if len(texts) >= 2 else "comparison_not_applicable_single_prompt",
        "different_prompt_max_abs_differences": differences if len(texts) >= 2 else None,
        "feature_cosine_similarity": torch.nn.functional.normalize(features, dim=-1).matmul(torch.nn.functional.normalize(features, dim=-1).T).tolist() if len(texts) >= 2 else None,
        "genuine_feature_cache": record_file(cached), "cache_origin": "Actual local original Llama3 + MNTP merge + supervised adapter; vendored bidirectional LLM2Vec; batch size 1",
        "output_directory": str(output), "gpu": torch.cuda.get_device_name(), "motion_samples": []})
    del encoder, encoder_model, tokenizer, config
    gc.collect()
    torch.cuda.empty_cache()
    report["cuda_memory_allocated_after_encoder_unload"] = torch.cuda.memory_allocated()
    print("Genuine embeddings cached; 8B encoder released before motion loading", flush=True)
    if args.encode_only:
        report["status"] = "real_text_encoding_passed_motion_not_tested"
        return

    class CachedOriginalEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.values = {text: features[i:i + 1, None].clone() for i, text in enumerate(texts)}
        def forward(self, requested):
            if any(text not in self.values for text in requested):
                raise ValueError("Text was not encoded by the verified original LLM2Vec in this run")
            return torch.cat([self.values[text] for text in requested]).to("cuda:0"), [1] * len(requested)

    cfg = OmegaConf.merge(OmegaConf.load(model_dir / "config.yaml"), {"checkpoint_dir": str(model_dir), "text_encoder": None})
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg.pop("checkpoint_dir", None)
    model = instantiate_from_dict(cfg, overrides={"device": "cuda:0"}).eval()
    model.text_encoder = CachedOriginalEncoder()
    report["motion_checkpoint"] = record_file(model_dir / "model.safetensors")
    parents = model.skeleton.joint_parents.detach().cpu().numpy()
    names = list(model.skeleton.bone_order_names)
    preview_dir = ROOT / "outputs/previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    previous = None
    for index, prompt in enumerate(texts):
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        started = time.perf_counter()
        with torch.inference_mode():
            motion = model(prompts=prompt, num_frames=args.frames, num_denoising_steps=args.steps,
                constraint_lst=[], cfg_type="regular", cfg_weight=args.cfg_weight,
                post_processing=False, return_numpy=True)
        torch.cuda.synchronize()
        motion = {key: np.asarray(value) for key, value in motion.items()}
        joints = motion["posed_joints"]
        if joints.shape != (args.frames, 34, 3) or not all(np.isfinite(value).all() for value in motion.values()):
            raise RuntimeError("Motion output has invalid shape or non-finite values")
        stem = f"kimodo_g1_text_{index + 1}_{stamp}"
        npz = output / f"{stem}.npz"
        np.savez_compressed(npz, **motion, fps=np.asarray(model.fps), joint_parents=parents, joint_names=np.asarray(names))
        video = preview_dir / f"{stem}.mp4"
        count = render(joints, parents, names, model.fps, video, prompt)
        item = {"prompt": prompt, "seed": args.seed, "frames": count, "fps": model.fps,
            "cfg_type": "regular", "cfg_weight": args.cfg_weight, "sampling_seconds": round(time.perf_counter() - started, 3),
            "all_arrays_finite": True, "posed_joints_shape": list(joints.shape),
            "files": [record_file(npz), record_file(video)]}
        if previous is not None:
            joint_difference = float(np.max(np.abs(joints - previous)))
            item["max_abs_joint_difference_from_first_prompt_same_seed"] = joint_difference
            if not joint_difference > 1e-6:
                raise RuntimeError("Different real text features did not change same-seed joint samples; text-to-motion data-flow check failed")
        else:
            previous = joints.copy()
        report["motion_samples"].append(item)
        print(f"PASS genuine text-conditioned motion {index + 1}: {prompt}", flush=True)
    report["text_to_motion_effect_status"] = "passed_different_prompt_changed_same_seed_joint_sample" if len(texts) >= 2 else "comparison_not_applicable_single_prompt"
    report["status"] = "real_text_conditioned_component_passed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=ROOT / "configs/kimodo-text-runtime.json")
    parser.add_argument("--prompt", action="append", help="One nonempty task prompt, or repeat with distinct prompts; omitted: two-prompt verification")
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg-weight", type=float, default=2.)
    parser.add_argument("--preflight", action="store_true", help="Tiny CPU bidirectional/API test only; no real model weights")
    parser.add_argument("--encode-only", action="store_true")
    args = parser.parse_args()
    if platform.system() != "Linux":
        parser.error("Run on the Linux server; the local machine is for video preview")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("6", "7"):
        parser.error("Set CUDA_VISIBLE_DEVICES to exactly physical GPU 6 or 7")
    if not 2 <= args.frames <= 600 or not 2 <= args.steps <= 1000 or not 0 < args.cfg_weight < 100:
        parser.error("Require frames 2..600, steps 2..1000, and finite positive CFG weight <100")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    report = {"checked_at": datetime.now(timezone.utc).isoformat(), "status": "running",
        "python": sys.executable, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "full_kinogen_pipeline": False, "training_executed": False, "external_api_used": False,
        "quantitative_semantic_alignment_validated": False,
        "limitations": ["Public Kimodo text-conditioned component only", "No quantitative semantic-accuracy evaluation", "No physics/simulator validation", "No video-generation API", "No KinoGen hand-object optimization or official integration"]}
    try:
        run(args, report)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        path = ROOT / "reports" / ("kimodo-text-preflight.json" if args.preflight else "kimodo-text-latest.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        path.write_text(text, encoding="utf-8")
        if report.get("output_directory"):
            (Path(report["output_directory"]) / "report.json").write_text(text, encoding="utf-8")
        print(text, flush=True)


if __name__ == "__main__":
    main()
