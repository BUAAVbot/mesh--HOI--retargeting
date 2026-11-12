"""Audit the final 10-track kinematic delivery against calibrated source assets."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar(value):
    array = np.asarray(value)
    if array.size == 1:
        item = array.reshape(-1)[0]
        return item.item() if hasattr(item, "item") else item
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--board", type=Path, required=True)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cases = json.loads((args.board / "cases.json").read_text(encoding="utf-8"))
    results = []
    for case in cases:
        folder = args.board / case["folder"]
        npz_path = folder / f"{case['stem']}_kinematic.npz"
        geometry = json.loads((folder / "geometry_check.json").read_text(encoding="utf-8"))
        with np.load(npz_path, allow_pickle=False) as data:
            qpos = data["qpos"]
            item = {
                **case,
                "delivery_npz": str(npz_path),
                "delivery_npz_sha256": sha256(npz_path),
                "qpos_shape": list(qpos.shape),
                "delivery_fps": float(scalar(data["fps"])),
                "delivery_status": str(scalar(data["status"])) if "status" in data else None,
                "delivery_geometry_pass": bool(scalar(data["kinematic_geometry_pass"]))
                if "kinematic_geometry_pass" in data else None,
                "geometry_reference_sha256": geometry.get("reference_sha256"),
                "source_qa_status": geometry.get("source_qa_status"),
                "source_missing_frames": geometry.get("source_missing_frames"),
            }

        calibration_dir = args.calibration_root / case["task"] / case["person"]
        reference = calibration_dir / "reference_calibrated.npz"
        support = calibration_dir / "support_scene.json"
        qa = calibration_dir / "qa.json"
        item.update({
            "calibration_dir": str(calibration_dir),
            "calibrated_reference_exists": reference.exists(),
            "calibrated_reference_sha256": sha256(reference),
            "support_scene_exists": support.exists(),
            "support_scene_sha256": sha256(support),
            "qa_exists": qa.exists(),
            "board_matches_calibrated_reference": (
                reference.exists()
                and geometry.get("reference_sha256") == sha256(reference)
            ),
        })
        if reference.exists():
            with np.load(reference, allow_pickle=True) as data:
                item["calibrated_keys"] = sorted(data.files)
                for key in ("qpos", "qpos_ref", "fps", "stance", "contact", "contact_ref"):
                    if key in data:
                        item[f"calibrated_{key}_shape"] = list(data[key].shape)
        if support.exists():
            support_data = json.loads(support.read_text(encoding="utf-8"))
            item["support_scene_keys"] = sorted(support_data)
            for key in ("boxes", "proxies", "support_proxies", "surfaces", "objects", "geometries"):
                if key in support_data and isinstance(support_data[key], list):
                    item["support_proxy_count"] = len(support_data[key])
                    item["support_proxy_key"] = key
                    break
        if qa.exists():
            qa_data = json.loads(qa.read_text(encoding="utf-8"))
            item["qa_top_keys"] = sorted(qa_data)
            for key in (
                "status", "passed", "strict_pass", "approved", "geometry_pass",
                "strict_geometry_pass", "strict_ready_for_render",
                "contact_calibration_passed", "full_support_collision_passed",
                "contact_force_validated", "dynamic_tracking_validated",
                "kinematic_precheck_passed", "contact_geometry_precheck_passed",
            ):
                if key in qa_data:
                    item[f"qa_{key}"] = qa_data[key]
        results.append(item)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
