"""Audit actual user-video outputs and export a compact, qualified delivery report."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/user_reconstruction"
VIDEO_SHA = "035c5852f9580cbbf5d612e2ad93a7f80065426a34ce355fb87db43876b20d6b"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import av
    import numpy as np
    from PIL import Image, ImageDraw

    def load(name):
        return json.loads((ROOT / "reports" / name).read_text(encoding="utf-8"))

    vision = load("user-vision.json")
    tracking = load("user-tracking.json")
    box_report = load("user-box-reconstruction.json")
    metric = load("user-metric-depth.json")
    watermark = load("user-watermark.json")
    assert sha(ROOT / "data/video_generation/candidate_scene/generated.mp4") == VIDEO_SHA
    with np.load(OUT / "input.npz", allow_pickle=False) as data:
        invalid = data["watermark_invalid"]
        times = data["timestamps"]
        assert str(data["video_sha256"]) == VIDEO_SHA
        assert invalid.shape == (121, 2, 720, 640)
        assert np.allclose(times, np.arange(121) / 24)
    with np.load(OUT / "segmentation.npz", allow_pickle=False) as data:
        assert str(data["source_video_sha256"]) == VIDEO_SHA
        assert np.array_equal(data["observation_valid"], ~invalid)
        counts = {}
        for kind in ("robot", "box"):
            mask = data[kind]
            assert mask.shape == invalid.shape and not np.any(mask & invalid)
            assert mask.any(axis=(-2, -1)).all()
            counts[kind] = int(mask.any(axis=(-2, -1)).sum())
            del mask
    with np.load(OUT / "depth.npz", allow_pickle=False) as data:
        assert str(data["source_video_sha256"]) == VIDEO_SHA
        assert np.array_equal(data["observation_valid"], ~invalid)
        depth = data["relative_depth"]
        assert depth.shape == invalid.shape and np.isfinite(depth).all()
        del depth
    with np.load(OUT / "tracking.npz", allow_pickle=False) as data:
        assert str(data["video_sha256"]) == VIDEO_SHA
        points, valid = data["positions_xy"], data["observation_valid"]
        assert points.shape == (121, 2, 650, 2)
        for frame, view in np.ndindex(121, 2):
            p = points[frame, view, valid[frame, view]]
            assert np.isfinite(p).all()
            xy = np.round(p).astype(int)
            assert ((xy[:, 0] >= 0) & (xy[:, 0] < 640) & (xy[:, 1] >= 0) & (xy[:, 1] < 720)).all()
            assert not invalid[frame, view, xy[:, 1], xy[:, 0]].any()
    with np.load(OUT / "sparse_stereo.npz", allow_pickle=False) as data:
        valid = data["valid"]
        assert valid.shape == (121, 37) and valid.any(axis=1).all()
        assert np.isfinite(data["points_world"][valid]).all()
        reprojection = data["reprojection_error_px"]
        assert (reprojection[np.broadcast_to(valid[:, None, :], reprojection.shape)] <= 3).all()
        assert (data["epipolar_sampson_px"][valid] <= 2).all()
        assert int(valid.sum()) == tracking["valid_stereo_observations"]
    with np.load(OUT / "box_trajectory.npz", allow_pickle=False) as data:
        valid, image_valid = data["valid"], data["image_fit_valid"]
        assert valid.shape == (121,) and int(valid.sum()) == box_report["valid_frames"]
        assert int(image_valid.sum()) == box_report["image_fit_valid_frames"]
        assert np.isnan(data["center_valid_only_xyz_m"][~valid]).all()
        assert np.isfinite(data["center_xyz_m"]).all()
        assert np.allclose(np.linalg.norm(data["quaternion_xyzw"], axis=1), 1)
    for name, expected in (("input.npz", tracking["input_npz_sha256"]),
                           ("geometry.npz", tracking["geometry_sha256"]),
                           ("segmentation.npz", tracking["segmentation_sha256"])):
        assert sha(OUT / name) == expected, "Stale tracking dependency: " + name
    for record in box_report["input_files"]:
        assert sha(ROOT / record["path"]) == record["sha256"], "Stale box dependency"
    for record in metric["inputs"].values():
        assert sha(Path(record["path"])) == record["sha256"], "Stale depth calibration dependency"
    assert metric["strict_metric_reconstruction_valid"] is False
    assert metric["dynamic_object_metric_validated"] is False
    assert watermark["watermark_revision"] == 3 and watermark["source_video_modified"] is False

    previews = []
    names = ["segmentation", "tracking", "depth", "box_reconstruction", "metric_depth"]
    sheet = Image.new("RGB", (1440, len(names) * 343), "#101827")
    draw = ImageDraw.Draw(sheet)
    for row, name in enumerate(names):
        path = ROOT / "outputs/previews" / ("user_" + name + ".mp4")
        count = 0
        with av.open(str(path)) as container:
            for count, frame in enumerate(container.decode(video=0), 1):
                expected_height = 840 if name == "box_reconstruction" else 720
                assert (frame.width, frame.height) == (1280, expected_height), name
                if count - 1 in (0, 60, 120):
                    column = (0, 60, 120).index(count - 1)
                    image = frame.to_image()
                    image.thumbnail((480, 315))
                    sheet.paste(image, (480 * column, row * 343 + 28))
                    draw.text((480 * column + 6, row * 343 + 8), f"{name} | frame {count-1:03d}", fill="white")
        assert count == 121, "Incomplete preview: " + name
        previews.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                         "sha256": sha(path), "decoded_frames": count, "width": 1280, "height": expected_height})
    sheet.save(ROOT / "outputs/previews/user_reconstruction_review.jpg", quality=91)
    artifacts = []
    for name in ("input.npz", "segmentation.npz", "depth.npz", "geometry.npz", "tracking.npz",
                 "sparse_stereo.npz", "metric_depth.npz", "box_trajectory.npz", "box_trajectory.json",
                 "box_trajectory.ply", "cloud_frame_000.ply", "cloud_frame_060.ply", "cloud_frame_120.ply"):
        path = OUT / name
        artifacts.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha(path)})
    fit_summaries = []
    for fit in metric["fits"]:
        selected = next(x for x in fit["alternatives"] if x["model"] == fit["selected_model"])
        fit_summaries.append({"view": fit["view"], "model": fit["selected_model"],
                              "spatial_group_heldout": selected["spatial_group_heldout"],
                              "heldout_per_surface": fit["heldout_per_surface"]})
    summary = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed_vision_and_conditional_3d_candidates_with_quality_flags",
        "source_video_sha256": VIDEO_SHA, "frames": 121, "views": 2, "fps": 24,
        "server": "yz-ai-4090-01.bce:/data/KinoGen", "inference_physical_gpu": 6,
        "sam3_mode": "independent image segmentation for every frame, not SAM3 video tracking",
        "nonempty_segmented_views": counts,
        "tracking": {key: tracking[key] for key in ("accepted_2d_observations", "shared_point_count", "valid_stereo_observations", "median_valid_reprojection_px")},
        "box": {key: box_report[key] for key in ("status", "valid_frames", "image_fit_valid_frames", "max_lift_relative_to_first_valid_frames_m", "independent_view_center_disagreement_median_m")},
        "depth_calibration": fit_summaries,
        "watermark": {key: watermark[key] for key in ("watermark_revision", "frames_with_candidates", "uncertain_frames", "invalid_fraction", "source_video_modified")},
        "previews": previews, "artifacts": artifacts,
        "training": False, "paid_video_api": False, "full_kinogen_pipeline": False,
        "strict_metric_reconstruction_valid": False, "whole_body_hands_reconstructed": False,
        "limitations": [
            "AI generated video and reference cameras are only conditionally compatible; no physical sensor calibration.",
            "The 37 reconstructed common robot points cover feet/ankles, not the articulated whole body or hands.",
            "57 box frames fail strict geometry checks; invalid frames remain marked and valid-only positions are NaN.",
            "Static depth holdout errors do not certify robot or box metric accuracy; clouds retain surface disagreement.",
            "Known cuboid silhouette fit is a custom baseline, not a FoundationPose registration or official KinoGen reconstruction.",
            "No whole-body/hand joint optimization, mesh reconstruction or physically tracked policy was completed."
        ],
    }
    path = ROOT / "reports/user-reconstruction-summary.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "report": str(path), "verified_artifacts": len(artifacts), "verified_previews": len(previews)}, indent=2))


if __name__ == "__main__":
    main()
