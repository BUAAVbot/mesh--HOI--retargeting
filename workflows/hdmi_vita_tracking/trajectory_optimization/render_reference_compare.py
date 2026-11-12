"""Kinematic FK reference comparison only. No physics is integrated."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

from render_compare import sha, model_joint_map


def load_reference(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as z:
        data = {key: z[key] for key in z.files}
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    joint_names = tuple(map(str, data.get("joint_names", metadata["joint_names"])))
    body_names = tuple(map(str, data.get("body_names", metadata["body_names"])))
    base = body_names.index("base_link")
    qpos = np.concatenate((data["body_pos_w"][:, base], data["body_quat_w"][:, base], data["joint_pos"]), axis=1)
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    if qpos.shape[1] != 35 or len(joint_names) != 28 or not np.isfinite(qpos).all():
        raise ValueError("invalid named Vita28 reference")
    return qpos, joint_names, fps


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("model", "before", "after", "projection-report", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--fps", type=float, default=25.)
    args = p.parse_args()

    import imageio.v2 as imageio
    import mujoco as mj
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    before, names0, fps0 = load_reference(args.before)
    after, names1, fps1 = load_reference(args.after)
    projection = json.loads(args.projection_report.read_text(encoding="utf-8"))
    if len(before) != len(after) or fps0 != fps1 or projection["original_source_reference_sha256"] != sha(args.before):
        raise ValueError("reference/projection provenance or time grids differ")
    model = mj.MjModel.from_binary_path(str(args.model))
    model.vis.global_.offwidth, model.vis.global_.offheight = 640, 400
    model.vis.headlight.ambient[:] = .7
    model.vis.headlight.diffuse[:] = .8
    mask = model.geom_type == mj.mjtGeom.mjGEOM_HFIELD
    model.geom_rgba[mask] = [.30, .43, .50, 1.]
    model.geom_matid[mask] = -1
    option = mj.MjvOption()
    option.geomgroup[:] = 1
    option.sitegroup[:] = 0
    maps = [model_joint_map(model, names0), model_joint_map(model, names1)]
    center = np.median(before[:, :3], axis=0)
    center[2] -= .15
    spread = np.max(np.linalg.norm(before[:, :2] - center[:2], axis=1))
    distance = float(max(2.5, 2 * spread + 1.6))
    cameras = []
    for azimuth, elevation in ((135, -18), (90, -10)):
        camera = mj.MjvCamera()
        camera.type = mj.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = center
        camera.distance, camera.azimuth, camera.elevation = distance, azimuth, elevation
        cameras.append(camera)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    count = int(np.ceil(len(before) * args.fps / fps0))
    data = mj.MjData(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    screenshots = {0, int(np.ceil(141 / fps0 * args.fps)), count - 1}
    with mj.Renderer(model, height=400, width=640) as renderer, imageio.get_writer(str(args.output), fps=args.fps, codec="libx264", quality=8) as writer:
        for index in range(count):
            frame = min(round(index / args.fps * fps0), len(before) - 1)
            canvas = Image.new("RGB", (1280, 928), (238, 241, 245))
            draw = ImageDraw.Draw(canvas)
            for row, camera in enumerate(cameras):
                for column, trace in enumerate((before, after)):
                    free, addresses = maps[column]
                    data.qpos[free:free + 7] = trace[frame, :7]
                    data.qpos[addresses] = trace[frame, 7:]
                    mj.mj_forward(model, data)
                    renderer.update_scene(data, camera=camera, scene_option=option)
                    renderer.scene.flags[mj.mjtRndFlag.mjRND_SHADOW] = 0
                    x, y = column * 640, row * 440
                    canvas.paste(Image.fromarray(renderer.render().copy()), (x, y + 40))
                    draw.rectangle((x, y, x + 640, y + 40), fill=(135, 78, 10))
                    draw.text((x + 8, y + 2), "KINEMATIC REFERENCE ONLY / NOT PHYSICS", font=font, fill="white")
                    label = "BEFORE / original reference" if column == 0 else "AFTER / root-height + joint-contact projection"
                    draw.text((x + 8, y + 21), label + f" | frame {frame}", font=font, fill="white")
            draw.text((8, 887), "DYNAMICS NOT VERIFIED. FK pose is assigned every frame; no simulation, policy execution or success certificate.", fill=(130, 45, 0), font=font)
            draw.text((8, 907), f"Same calibrated heightfield / source clock. Initial rootZ correction {projection['initial_rootZ_correction_m'] * 1000:+.0f} mm; joint contact geometry adjusted.", fill=(20, 30, 45), font=font)
            if index in screenshots:
                canvas.save(args.output.parent / f"{args.output.stem}_frame{frame:04d}.jpg", quality=93)
            writer.append_data(np.asarray(canvas))
    manifest = dict(kind="kinematic_reference_comparison", integrates_physics=False,
                    pose_state_written_each_frame=True, dynamics_verified=False,
                    policy_executed=False, before_source_reference_sha256=sha(args.before),
                    after_source_reference_sha256=sha(args.after), original_source_reference_sha256=projection["original_source_reference_sha256"],
                    projection_report_sha256=sha(args.projection_report), model_sha256=sha(args.model),
                    frames=count, fps=args.fps, source_frames=len(before), source_fps=fps0,
                    camera_center=center.tolist(), camera_distance=distance,
                    note="Geometry preview only; corrected contact geometry does not imply dynamically executable motion.")
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
