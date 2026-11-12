"""Render the smoothed motion and an aligned before/after comparison."""
from settings import work_dir, load_settings
import hashlib
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np

BASE = work_dir()
HERE = BASE / "smoothing_v1"
FRAMES, FPS = load_settings()["frames"], load_settings()["fps"]
import render_preview as original

PREVIEW = HERE / "preview"


def writer(path):
    return imageio.get_writer(path, fps=FPS, codec="libx264", quality=8,
                              macro_block_size=1, ffmpeg_log_level="error")


def validate(path, frames, shape):
    cap = cv2.VideoCapture(str(path))
    count = 0
    while True:
        ok, image = cap.read()
        if not ok:
            break
        assert image.shape[:2] == shape
        count += 1
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    assert count == frames and abs(fps-FPS) < 1e-5, (path, count, fps)
    return {"path": str(path), "frames": count, "fps": fps,
            "width": shape[1], "height": shape[0], "duration_s": count/fps,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    PREVIEW.mkdir(exist_ok=True)
    arrays, scales = [], []
    for person in (1, 2):
        with np.load(HERE / f"results/person_{person:02d}_smoothed.npz", allow_pickle=True) as a:
            q = a["qpos"].astype(float)
            assert q.shape == (FRAMES, 35) and np.isfinite(q).all()
            assert np.allclose(np.linalg.norm(q[:, 3:7], axis=1), 1, atol=1e-5)
            assert np.array_equal(a["frame_ids"], np.arange(FRAMES))
            arrays.append(q); scales.append(float(a["smpl_scale"][0]))
    assert abs(scales[0]-scales[1]) < 1e-6
    scale = scales[0]
    motion = np.concatenate(arrays, axis=1)
    model = mujoco.MjModel.from_xml_path(str(BASE / "preview_grounded/dual_scene.xml"))
    data = mujoco.MjData(model)
    original.PREVIEW = PREVIEW
    report = original.collision_report(model, data, motion, inter_robot_avoidance_optimized=True)
    assert report["max_joint_limit_violation_rad"] < 1e-6
    for key, metric in report["collision_proxies"].items():
        assert metric["min_signed_distance_m"] > -.002, (key, metric)
        if key.endswith("floor"):
            assert metric["max_distance_m"] < .035, (key, metric)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [-.22*scale, .9*scale, .78*scale]
    camera.distance = 4.8*scale
    camera.elevation = -17
    model.vis.headlight.active = 1
    model.vis.headlight.ambient[:] = [.45]*3
    model.vis.headlight.diffuse[:] = [.65]*3
    option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(option)
    option.geomgroup[3:] = 0
    for geom in original.groups_robot_collision(model):
        model.geom_group[geom] = 3
    renderer = mujoco.Renderer(model, width=1280, height=900)
    source = cv2.VideoCapture(str(BASE / "source/input.mp4"))
    path = PREVIEW / "dual_vita02a_seat_exchange_smoothed.mp4"
    output = writer(path)
    samples = []
    try:
        for t, q in enumerate(motion):
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            canvas = np.full((1080, 1920, 3), (22, 28, 33), np.uint8)
            camera.azimuth = 90
            renderer.update_scene(data, camera=camera, scene_option=option)
            canvas[70:970, :1280] = renderer.render()
            camera.azimuth = 125
            renderer.update_scene(data, camera=camera, scene_option=option)
            canvas[430:880, 1280:] = cv2.resize(renderer.render(), (640, 450))
            ok, image = source.read()
            assert ok
            canvas[70:430, 1280:] = cv2.cvtColor(cv2.resize(image, (640, 360)), cv2.COLOR_BGR2RGB)
            text = original.put_text
            text(canvas, "Vita02A x2 | temporally smoothed", (28, 44), .95)
            text(canvas, f"Frame {t:03d}/{FRAMES-1} | {t/FPS:.2f}s | {FPS:g} FPS", (1320, 44), .65)
            text(canvas, "Person 1: cyan    Person 2: orange", (28, 1004), .75)
            text(canvas, "Whole-sequence smoothing + foot contact / seat collision constraints", (28, 1045), .62)
            text(canvas, "Source RGB (synchronized)", (1300, 100), .6)
            text(canvas, "Oblique view", (1300, 460), .6)
            text(canvas, "KINEMATIC PREVIEW", (1300, 925), .75, (255, 191, 100))
            text(canvas, "Not a physics rollout or robot-ready trajectory", (1300, 958), .53)
            output.append_data(canvas)
            if t in set(np.linspace(0, FRAMES-1, min(8, FRAMES), dtype=int)):
                samples.append(cv2.resize(canvas, (960, 540)))
                cv2.imwrite(str(PREVIEW / f"frame_{t:03d}.jpg"),
                            cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    finally:
        output.close(); source.release(); renderer.close()
    if len(samples) % 2:
        samples.append(np.zeros_like(samples[-1]))
    sheet = np.concatenate([np.concatenate(samples[i:i+2], axis=1)
                            for i in range(0, len(samples), 2)], axis=0)
    cv2.imwrite(str(PREVIEW / "contact_sheet.jpg"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    comparison_path = PREVIEW / "before_after_smoothing.mp4"
    comparison = writer(comparison_path)
    try:
        for repeat in (1, 2):
            before = cv2.VideoCapture(str(BASE / "preview_grounded/dual_vita02a_seat_exchange.mp4"))
            after = cv2.VideoCapture(str(path))
            try:
                for t in range(FRAMES):
                    ok1, a = before.read(); ok2, b = after.read()
                    assert ok1 and ok2
                    canvas = np.full((900, 1920, 3), (22, 28, 33), np.uint8)
                    canvas[80:755, :960] = cv2.cvtColor(cv2.resize(a[70:970, :1280], (960, 675)), cv2.COLOR_BGR2RGB)
                    canvas[80:755, 960:] = cv2.cvtColor(cv2.resize(b[70:970, :1280], (960, 675)), cv2.COLOR_BGR2RGB)
                    original.put_text(canvas, "BEFORE | grounded, frame-wise projection", (25, 40), .8)
                    original.put_text(canvas, "AFTER | constrained temporal smoothing", (985, 40), .8)
                    original.put_text(canvas, f"Source frame {t:03d}/{FRAMES-1} | {t/FPS:.2f}s", (25, 798), .75)
                    original.put_text(canvas, "Normal speed" if repeat == 1 else "0.5x replay", (985, 798), .85)
                    original.put_text(canvas, "Same scene, camera, robot model and source timeline", (25, 845), .65)
                    original.put_text(canvas, "Kinematic preview only", (985, 845), .65)
                    for _ in range(repeat):
                        comparison.append_data(canvas)
            finally:
                before.release(); after.release()
    finally:
        comparison.close()
    validation = {"smoothed": validate(path, FRAMES, (1080, 1920)),
                  "comparison": validate(comparison_path, 3*FRAMES, (900, 1920))}
    (PREVIEW / "video_validation.json").write_text(json.dumps(validation, indent=2)+"\n")
    print(json.dumps(validation, indent=2), flush=True)


if __name__ == "__main__":
    main()
