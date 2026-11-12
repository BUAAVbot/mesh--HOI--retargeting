"""Render both retargeted robots, the shared scene, and synchronized source video."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import os
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np
import trimesh
from settings import work_dir, load_settings

HERE = work_dir()
PREVIEW = HERE / "preview"
SETTINGS = load_settings()
ROBOT = Path(SETTINGS["robot_xml"])
FRAMES, FPS = SETTINGS["frames"], SETTINGS["fps"]
U = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
COLORS = [(0.08, .70, .88, 1), (1., .48, .10, 1)]


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def make_scene(scale):
    root = ET.Element("mujoco", model="two_vita02a_sitting")
    ET.SubElement(root, "compiler", angle="radian", fusestatic="false")
    ET.SubElement(root, "option", gravity="0 0 -9.81")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1920", offheight="1080")
    ET.SubElement(visual, "quality", shadowsize="2048")
    assets = ET.SubElement(root, "asset")
    ET.SubElement(assets, "texture", type="skybox", builtin="gradient",
                  rgb1=".78 .86 .93", rgb2=".94 .96 .98", width="512", height="3072")
    original = ET.parse(ROBOT).getroot()
    for asset in original.find("asset"):
        asset = copy.deepcopy(asset)
        if asset.get("file"):
            asset.set("file", str((ROBOT.parent / asset.get("file")).resolve()))
        assets.append(asset)
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0 -3 6", dir="0 0 -1", diffuse=".7 .7 .7")
    ET.SubElement(world, "light", pos="3 4 5", dir="-0.2 -0.3 -1", diffuse=".5 .5 .5")
    ET.SubElement(world, "geom", name="floor", type="plane", size="10 10 .1",
                  rgba=".82 .84 .83 1", pos="0 0 -.008")
    for person, color in enumerate(COLORS, 1):
        for original_body in original.find("worldbody"):
            if original_body.tag != "body":
                continue
            body = copy.deepcopy(original_body)
            for element in body.iter():
                if element.get("name"):
                    element.set("name", f"p{person}_" + element.get("name"))
                if element.tag == "geom" and element.get("group") == "1":
                    rgba = np.fromstring(element.get("rgba", ".5 .5 .5 1"), sep=" ")
                    # Tint light-colored covers, keeping the dark mechanical details.
                    if rgba[:3].mean() > .3:
                        element.set("rgba", " ".join(map(str, color)))
            world.append(body)
    seats = json.loads((HERE / "seats.json").read_text())
    for seat in seats:
        ET.SubElement(world, "geom", name=seat["name"], type="box",
                      pos=" ".join(map(str, np.array(seat["center_zup_m"]) * scale)),
                      size=" ".join(map(str, np.array(seat["size_m"]) * scale / 2)),
                      rgba=".19 .21 .22 1")
    scene = np.load(HERE / "source/scene/pi3_mesh.npz")
    v, f = scene["vertices"] @ U.T, scene["faces"]
    colors = scene["colors"].astype(float)
    if colors.max() > 1:
        colors /= 255
    # Vertex clustering reduces the visual-only RGB reconstruction, not collision geometry.
    _, inverse = np.unique(np.round(v / .035).astype(int), axis=0, return_inverse=True)
    count = np.bincount(inverse)
    vv = np.column_stack([np.bincount(inverse, weights=v[:, j]) / count for j in range(3)])
    cc = np.column_stack([np.bincount(inverse, weights=colors[:, j]) / count for j in range(3)])
    ff = inverse[f]
    keep = (ff[:, 0] != ff[:, 1]) & (ff[:, 1] != ff[:, 2]) & (ff[:, 0] != ff[:, 2])
    ff = ff[keep]
    ff = np.unique(ff, axis=0)
    centers = vv[ff].mean(axis=1)
    keep = ((abs(centers[:, 0]) < 3) & (centers[:, 1] > -.5) & (centers[:, 1] < 3.5))
    ff = ff[keep]
    face_colors = cc[ff].mean(axis=1)
    bins = np.minimum((face_colors * 4).astype(int), 3)
    ids = bins[:, 0] * 16 + bins[:, 1] * 4 + bins[:, 2]
    mesh_dir = PREVIEW / "scene_meshes"
    mesh_dir.mkdir(exist_ok=True)
    for index in np.unique(ids):
        selection = ids == index
        if selection.sum() < 3:
            continue
        mesh = trimesh.Trimesh(vv * scale, ff[selection], process=False)
        mesh.remove_unreferenced_vertices()
        path = mesh_dir / f"scene_{index:02d}.obj"
        mesh.export(path)
        name = f"scene_{index:02d}"
        ET.SubElement(assets, "mesh", name=name, file=str(path))
        color = np.r_[np.mean(face_colors[selection], axis=0), 1.]
        ET.SubElement(world, "geom", name=name, type="mesh", mesh=name,
                      contype="0", conaffinity="0", group="2", density="0",
                      rgba=" ".join(map(str, color)))
    path = PREVIEW / "dual_scene.xml"
    ET.ElementTree(root).write(path, encoding="unicode")
    return mujoco.MjModel.from_xml_path(str(path))


def load_motions(refined=False, grounded=False):
    motions, scales = [], []
    for person in (1, 2):
        suffix = "_grounded" if grounded else ("_refined" if refined else "")
        with np.load(HERE / f"results/person_{person:02d}{suffix}.npz", allow_pickle=True) as a:
            q = a["qpos"].astype(float)
            assert q.shape == (FRAMES, 35) and np.isfinite(q).all()
            assert np.array_equal(a["frame_ids"], np.arange(FRAMES))
            assert np.allclose(np.linalg.norm(q[:, 3:7], axis=1), 1, atol=1e-5)
            assert abs(float(np.ravel(a["fps"])[0]) - FPS) < 1e-5
            scales.append(float(np.ravel(a["smpl_scale"])[0]))
            motions.append(q)
    assert abs(scales[0] - scales[1]) < 1e-6, scales
    return np.concatenate(motions, axis=1), scales[0]


def collision_report(model, data, motion, inter_robot_avoidance_optimized=False):
    if motion.shape != (FRAMES, 70) or not np.isfinite(motion).all():
        raise ValueError("Invalid dual-robot motion shape or nonfinite values.")
    for offset in (0, 35):
        if not np.allclose(np.linalg.norm(motion[:, offset+3:offset+7], axis=1), 1, atol=1e-5):
            raise ValueError("Non-unit root quaternion.")
    groups = {key: [] for key in ["p1", "p2", "seats"]}
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    for g in range(model.ngeom):
        if not (model.geom_contype[g] or model.geom_conaffinity[g]):
            continue
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or ""
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if body.startswith("p1_"):
            groups["p1"].append(g)
        elif body.startswith("p2_"):
            groups["p2"].append(g)
        elif name.startswith("seat_"):
            groups["seats"].append(g)
    def distance(a, b):
        return min(mujoco.mj_geomDistance(model, data, i, j, .5, None) for i in a for j in b)
    rows = []
    for frame, q in enumerate(motion):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        rows.append([distance(groups["p1"], groups["p2"]),
                     distance(groups["p1"], groups["seats"]),
                     distance(groups["p2"], groups["seats"]),
                     distance(groups["p1"], [floor]),
                     distance(groups["p2"], [floor])])
    values = np.asarray(rows)
    metrics = {}
    labels = ["robot_robot", "robot1_seats", "robot2_seats", "robot1_floor", "robot2_floor"]
    for col, label in enumerate(labels):
        v = values[:, col]
        metrics[label] = {"min_signed_distance_m": float(v.min()),
                          "worst_frame": int(v.argmin()),
                          "frames_penetrating_over_5mm": int((v < -.005).sum()),
                          "median_distance_m": float(np.median(v)),
                          "p95_distance_m": float(np.quantile(v, .95)),
                          "max_distance_m": float(v.max())}
    violations = []
    for j in range(model.njnt):
        if model.jnt_limited[j] and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
            values_j = motion[:, model.jnt_qposadr[j]]
            low, high = model.jnt_range[j]
            violations.append(max(float((low - values_j).max()), float((values_j - high).max()), 0))
    np.savez_compressed(PREVIEW / "collision_distances.npz",
                        frame_ids=np.arange(FRAMES), distances_m=values, labels=labels)
    report = {"frames": FRAMES, "fps": FPS, "finite_qpos": True, "unit_root_quaternions": True,
              "max_joint_limit_violation_rad": max(violations), "collision_proxies": metrics,
              "dynamics_validated": False, "inter_robot_avoidance_optimized": inter_robot_avoidance_optimized,
              "rendered_floor_z_m": -.008}
    save_json(PREVIEW / "quality_report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def put_text(canvas, text, xy, size=.65, color=(230, 235, 240)):
    cv2.putText(canvas, text, xy, cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)


def main():
    global PREVIEW
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--refined", action="store_true")
    parser.add_argument("--grounded", action="store_true")
    parser.add_argument("--scene-only", action="store_true", help="Build the dual scene without rendering a video.")
    args = parser.parse_args()
    assert not (args.refined and args.grounded)
    PREVIEW = HERE / ("preview_grounded" if args.grounded else ("preview_refined" if args.refined else "preview"))
    PREVIEW.mkdir(exist_ok=True)
    if args.smoke:
        scale = 1.
        motion = None
    else:
        motion, scale = load_motions(refined=args.refined, grounded=args.grounded)
    model = make_scene(scale)
    if model.nq != 70 or model.nv != 68:
        raise ValueError("This preset requires two Vita02A robots, each nq=35 and nv=34.")
    if args.scene_only:
        print("Scene:", PREVIEW / "dual_scene.xml", flush=True)
        return
    data = mujoco.MjData(model)
    if motion is not None:
        collision_report(model, data, motion)
    else:
        motion = np.repeat(model.qpos0[None], 1, axis=0)
        motion[:, :3] = [-1., 1., 1.]
        motion[:, 35:38] = [1., 1., 1.]
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [-.22 * scale, .9 * scale, .78 * scale]
    camera.distance = 4.8 * scale
    camera.elevation = -17
    renderer = mujoco.Renderer(model, width=1280, height=900)
    option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(option)
    # Hide primitive robot collision proxies; fitted static seats remain visible in group 0.
    option.geomgroup[3:] = 0
    for g in groups_robot_collision(model):
        model.geom_group[g] = 3
    model.vis.headlight.active = 1
    model.vis.headlight.ambient[:] = [.45] * 3
    model.vis.headlight.diffuse[:] = [.65] * 3
    source = cv2.VideoCapture(str(HERE / "source/input.mp4"))
    output = PREVIEW / "dual_vita02a_seat_exchange.mp4"
    writer = None if args.smoke else imageio.get_writer(output, fps=FPS, codec="libx264",
                                                        quality=8, macro_block_size=1,
                                                        ffmpeg_log_level="error")
    samples = []
    sample_ids = set(np.linspace(0, FRAMES-1, min(8, FRAMES), dtype=int))
    try:
        for frame, q in enumerate(motion):
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
            if not ok:
                raise RuntimeError(f"Source video ended at frame {frame}")
            canvas[70:430, 1280:] = cv2.cvtColor(cv2.resize(image, (640, 360)), cv2.COLOR_BGR2RGB)
            put_text(canvas, "Vita02A x2 | UMR scene-aware seat exchange", (28, 44), .95)
            put_text(canvas, f"Frame {frame:03d}/{FRAMES-1}   |   {frame / FPS:.2f}s   |   {FPS:g} FPS",
                     (1320, 44), .65)
            put_text(canvas, "Person 1: cyan    Person 2: orange", (28, 1004), .75)
            subtitle = ("UMR + constrained foot-contact projection / shared scene" if args.grounded
                        else "Synchronized source / shared world / fitted seat proxies")
            put_text(canvas, subtitle, (28, 1045), .63)
            put_text(canvas, "Source RGB (synchronized)", (1300, 100), .6)
            put_text(canvas, "Oblique view", (1300, 460), .6)
            put_text(canvas, "KINEMATIC PREVIEW", (1300, 925), .75, (255, 191, 100))
            put_text(canvas, "Not a physics rollout or robot-ready trajectory", (1300, 958), .53)
            put_text(canvas, f"Shared scene scale: {scale:.5f}", (1300, 1004), .6)
            if writer is not None:
                writer.append_data(canvas)
            if frame in sample_ids or args.smoke:
                samples.append(cv2.resize(canvas, (960, 540)))
                cv2.imwrite(str(PREVIEW / f"frame_{frame:03d}.jpg"),
                            cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
            if frame % max(1, round(FPS)) == 0:
                print("rendered", frame, flush=True)
    finally:
        source.release()
        if writer:
            writer.close()
        renderer.close()
    if len(samples) > 1 and len(samples) % 2:
        samples.append(np.zeros_like(samples[-1]))
    contact = np.concatenate([np.concatenate(samples[i:i+2], axis=1)
                              for i in range(0, len(samples), 2)], axis=0) if len(samples) > 1 else samples[0]
    cv2.imwrite(str(PREVIEW / "contact_sheet.jpg"), cv2.cvtColor(contact, cv2.COLOR_RGB2BGR))
    if not args.smoke:
        cap = cv2.VideoCapture(str(output))
        decoded = 0
        while True:
            ok, image = cap.read()
            if not ok:
                break
            assert image.shape == (1080, 1920, 3)
            decoded += 1
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        assert decoded == FRAMES and abs(fps - FPS) < 1e-5
        save_json(PREVIEW / "video_validation.json",
                  {"decoded_frames": decoded, "fps": fps, "width": 1920, "height": 1080,
                   "duration_seconds": decoded / fps,
                   "bytes": output.stat().st_size,
                   "sha256": hashlib.sha256(output.read_bytes()).hexdigest()})
        print("VIDEO", output, flush=True)


def groups_robot_collision(model):
    for g in range(model.ngeom):
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or ""
        if body.startswith(("p1_", "p2_")) and model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            yield g


if __name__ == "__main__":
    main()
