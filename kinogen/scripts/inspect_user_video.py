"""Decode the user video and export review frames without modifying the input."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    import av
    from PIL import Image, ImageDraw
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=ROOT / "data/video_generation/candidate_scene/generated.mp4")
    args = parser.parse_args()
    video = args.video.resolve()
    digest = hashlib.sha256()
    with video.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    checksum = digest.hexdigest()
    out = ROOT / "outputs/user_video" / checksum[:16]
    out.mkdir(parents=True, exist_ok=True)
    timestamps, sizes = [], set()
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        rate = float(stream.average_rate) if stream.average_rate else None
        codec = stream.codec_context.name
        for frame in container.decode(video=0):
            timestamps.append(float(frame.time) if frame.time is not None else None)
            sizes.add((frame.width, frame.height))
    if not timestamps or len(sizes) != 1:
        raise RuntimeError("No frames, or changing video dimensions")
    width, height = next(iter(sizes))
    indices = sorted(set(round(i * (len(timestamps) - 1) / 5) for i in range(6)))
    sheet = Image.new("RGB", (1280, 3 * 386), "white")
    draw = ImageDraw.Draw(sheet)
    samples = []
    with av.open(str(video)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index not in indices:
                continue
            number = indices.index(index)
            image = frame.to_image()
            path = out / f"frame_{index:06d}.png"
            image.save(path)
            if width % 2 == 0:
                for side, begin in (("left", 0), ("right", width // 2)):
                    image.crop((begin, 0, begin + width // 2, height)).save(out / f"frame_{index:06d}_{side}.png")
            preview = image.copy()
            preview.thumbnail((640, 360))
            x, y = (number % 2) * 640, (number // 2) * 386
            sheet.paste(preview, (x, y + 26))
            draw.text((x + 5, y + 5), f"Frame {index + 1}/{len(timestamps)} | t={timestamps[index]}", fill="black")
            samples.append({"frame_index": index, "time_seconds": timestamps[index], "path": str(path.relative_to(ROOT))})
    sheet_path = ROOT / "outputs/previews/user_video_contact_sheet.png"
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(sheet_path)
    valid_times = [t for t in timestamps if t is not None]
    duration = valid_times[-1] - valid_times[0] + (1 / rate if rate else 0) if valid_times else None
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "status": "decoded_for_visual_review",
              "input": str(video.relative_to(ROOT)), "bytes": video.stat().st_size, "sha256": checksum,
              "codec": codec, "width": width, "height": height, "average_fps": rate,
              "decoded_frames": len(timestamps), "duration_seconds_estimated": duration,
              "timestamps_strictly_increasing": len(valid_times) == len(timestamps) and all(a < b for a, b in zip(valid_times, valid_times[1:])),
              "matches_reference_aspect_ratio": width * 720 == height * 1280,
              "candidate_equal_halves_wh": [width // 2, height] if width % 2 == 0 else None,
              "equal_halves_are_only_a_candidate_split": True,
              "camera_calibration_validated": False, "cross_view_consistency_validated": False,
              "samples": samples, "contact_sheet": str(sheet_path.relative_to(ROOT)),
              "note": "Original video is preserved. Crop, camera motion, cross-view agreement, grasp, and geometry require visual review; no reconstruction or model inference is performed here."}
    text = json.dumps(report, indent=2) + "\n"
    (ROOT / "reports/user-video-inspection.json").write_text(text, encoding="utf-8")
    (out / "inspection.json").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
