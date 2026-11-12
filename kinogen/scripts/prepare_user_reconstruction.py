"""Prepare original two-view frames and per-frame watermark observation masks."""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def main():
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw
    from watermark_locator import WatermarkLocator
    cv2.setNumThreads(4)
    source = ROOT / "data/video_generation/candidate_scene/generated.mp4"
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    if checksum != "035c5852f9580cbbf5d612e2ad93a7f80065426a34ce355fb87db43876b20d6b":
        raise RuntimeError("This preparation was authored for the inspected user video")
    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if len(frames) != 121 or any(f.shape != (720, 1280, 3) for f in frames) or fps != 24:
        raise RuntimeError("Unexpected frame geometry or timebase")
    out = ROOT / "outputs/user_reconstruction"
    out.mkdir(parents=True, exist_ok=True)
    locator = WatermarkLocator(frames)
    x, y, w, h = 42, 35, 205, 60
    cv2.imwrite(str(out / "watermark_template.png"), frames[120][y:y+h, x:x+w])
    invalid = np.zeros((len(frames), 720, 1280), dtype=bool)
    detections = []
    selected_indices = [0, 24, 48, 72, 96, 120]
    sheet = Image.new("RGB", (1280, 3 * 386), "white")
    draw = ImageDraw.Draw(sheet)
    for index, frame in enumerate(frames):
        candidate = locator.locate(frame)[0]
        boxes = []
        score = candidate['score']
        uncertain = .30 <= score < .45
        if score >= .30:
            px, py, endx, endy = candidate['xyxy']
            padding = 6
            x0, y0 = max(0, px-padding), max(0, py-padding)
            x1, y1 = min(1280, endx+padding), min(720, endy+padding)
            invalid[index, y0:y1, x0:x1] = True
            boxes.append({"xyxy": [x0,y0,x1,y1], "score": float(score), "scale": candidate['scale']})
        detections.append({"frame": index, "time_seconds": index/fps, "boxes": boxes,
                           "best_score": float(score), "uncertain": bool(uncertain), "invalid_pixels": int(invalid[index].sum())})
        if index in selected_indices:
            row = selected_indices.index(index)
            image = frame.copy()
            for item in boxes:
                a,b,c,d = item["xyxy"]
                cv2.rectangle(image, (a,b), (c,d), (20,20,245), 3)
            image = cv2.resize(image, (640,360), interpolation=cv2.INTER_AREA)
            sheet.paste(Image.fromarray(image[...,::-1]), ((row%2)*640,(row//2)*386+26))
            draw.text(((row%2)*640+5,(row//2)*386+5), f"Frame {index} | mask candidates {len(boxes)} | score {score:.2f}",fill="black")
    rgb = np.stack([np.stack((f[:,:640,::-1], f[:,640:,::-1])) for f in frames])
    watermark = np.stack((invalid[:,:,:640],invalid[:,:,640:]),axis=1)
    path = out / "input.npz"
    temporary = path.with_name("input.pending.npz")
    np.savez_compressed(temporary, rgb=rgb, watermark_invalid=watermark,
                        timestamps=np.arange(len(frames),dtype=np.float64)/fps,
                        frame_indices=np.arange(len(frames),dtype=np.int64), fps=np.asarray(fps),
                        video_sha256=np.asarray(checksum))
    backup = out / "input.watermark-v1.npz"
    if path.exists() and not backup.exists():
        path.replace(backup)
    temporary.replace(path)
    sheet.save(ROOT / "outputs/previews/user_watermark_review.png")
    report = {"created_at_utc":datetime.now(timezone.utc).isoformat(), "status":"prepared_for_watermark_visual_review",
              "input_sha256":checksum,"frames":len(frames),"fps":fps,"rgb_shape":list(rgb.shape),
              "watermark_invalid_shape":list(watermark.shape),"input_npz":"outputs/user_reconstruction/input.npz",
              "watermark_revision":3,
              "watermark_method":"Observed logo on texture-free background; signed high-pass, temporal-residual template mask, multiscale masked correlation; padded observation-rejection rectangles",
              "template_source_frame":120,"template_source_xywh":[x,y,w,h],
              "source_video_modified":False,"inpainting_performed":False,
              "frames_with_candidates":sum(bool(r["boxes"]) for r in detections),
              "uncertain_frames":sum(r["uncertain"] for r in detections),
              "invalid_fraction":float(watermark.mean()),"detections":detections}
    (ROOT / "reports/user-watermark.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({key:value for key,value in report.items() if key!="detections"}),flush=True)


if __name__ == "__main__":
    main()
