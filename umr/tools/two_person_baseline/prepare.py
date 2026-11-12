"""Prepare synchronized Y-up SMPL-X inputs and fitted static-seat proxies."""
from pathlib import Path
import csv
import hashlib
import json
import sys
import zipfile
from xml.sax.saxutils import quoteattr

import mujoco
import numpy as np
import trimesh
from settings import REPO, work_dir, load_settings, write_settings

HERE = work_dir()
sys.path.insert(0, str(REPO / "scripts"))
from humanoid_retarget_config import load_config
from humanoid_retarget_pipeline import correspondence_slots_compatible

U = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def timeline_fps(frame_ids, times, valid):
    count = len(frame_ids)
    if count < 4 or not np.asarray(valid).all():
        raise ValueError("At least four synchronized, valid frames are required for both actors.")
    if not np.array_equal(frame_ids, np.arange(count)):
        raise ValueError("The baseline requires contiguous, zero-based frame IDs.")
    frames = np.arange(count, dtype=float)
    times = np.asarray(times, dtype=float)
    if times.shape != frames.shape or not np.isfinite(times).all():
        raise ValueError("Invalid source timestamps.")
    # A whole-timeline fit avoids accumulating float32 difference quantization.
    period = float(frames @ times / (frames @ frames))
    if period <= 0 or not np.allclose(times, frames * period, atol=1e-6, rtol=0):
        raise ValueError("A uniform source timeline starting at zero is required.")
    return 1. / period


def archive_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    settings = load_settings()
    archive = Path(settings["input_zip"])
    archive_sha256 = archive_digest(archive)
    previous = HERE / "preparation.json"
    if previous.is_file() and json.loads(previous.read_text())["archive_sha256"] != archive_sha256:
        raise ValueError("The input archive changed; use a new workspace.")
    for name in ["source", "objects", "inputs", "configs", "results", "logs", "preview"]:
        (HERE / name).mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        prefix = settings["archive_prefix"]
        names = ["scene/pi3_mesh.npz", "cameras_world.npz", "timeline.npz", "input.mp4"]
        names += [f"human/person_{i:02d}/{n}.npz" for i in (1, 2)
                  for n in ("smplx_params_world", "smplx_meshes_world")]
        for name in names:
            path = HERE / "source" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_bytes(z.read(prefix + name))
    timeline = np.load(HERE / "source/timeline.npz")
    count = len(timeline["frame_ids"])
    fps = timeline_fps(timeline["frame_ids"], timeline["pts_seconds"], timeline["human_valid"])
    settings.update(frames=count, fps=fps)
    write_settings(settings)
    scene = np.load(HERE / "source/scene/pi3_mesh.npz")
    vertices, faces = scene["vertices"] @ U.T, scene["faces"]
    tri = vertices[faces]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    centers = tri.mean(axis=1)
    seats = []
    seat_centers = np.asarray(settings["seat_centers"]).reshape(2, 2)
    for index, (x, y) in enumerate(seat_centers, 1):
        mask = ((abs(centers[:, 0] - x) < .65) & (abs(centers[:, 1] - y) < .9)
                & (centers[:, 2] > .3) & (centers[:, 2] < .8)
                & (abs(normal[:, 2]) > .85))
        points = centers[mask]
        if not len(points):
            raise ValueError(f"No horizontal seat surface near ({x}, {y}); adjust --seat-centers.")
        hist, edges = np.histogram(points[:, 2], bins=np.arange(.3, .81, .02))
        peak = hist.argmax()
        top = points[(points[:, 2] > edges[peak] - .02)
                     & (points[:, 2] < edges[peak + 1] + .02)]
        low, high = np.quantile(top[:, :2], [.005, .995], axis=0)
        height = float(np.median(top[:, 2]))
        size = np.r_[high - low, height]
        center = np.r_[(high + low) / 2, height / 2]
        name = f"seat_{index:02d}"
        mesh_path = HERE / "objects" / f"{name}.obj"
        trimesh.creation.box(extents=size).export(mesh_path)
        half = " ".join(map(str, size / 2))
        xml = (f'<mujoco><asset><mesh name="{name}_mesh" file={quoteattr(str(mesh_path))}/></asset>'
               f'<worldbody><body name="{name}"><freejoint/>'
               f'<geom type="mesh" mesh="{name}_mesh" group="2" contype="0" conaffinity="0"/>'
               f'<geom name="{name}_collision" type="box" size="{half}" group="3"/>'
               '</body></worldbody></mujoco>')
        (HERE / "objects" / f"{name}.xml").write_text(xml)
        with (HERE / "objects" / f"prop_{name}.csv").open("w") as stream:
            writer = csv.writer(stream)
            writer.writerow(["frame_id", "timestamp", "px", "py", "pz", "qx", "qy", "qz", "qw"])
            for frame, timestamp in zip(timeline["frame_ids"], timeline["pts_seconds"]):
                writer.writerow([int(frame), float(timestamp), *center, 0, 0, 0, 1])
        mujoco.MjModel.from_xml_path(str(HERE / "objects" / f"{name}.xml"))
        seats.append({"name": name, "center_zup_m": center.tolist(), "size_m": size.tolist(),
                      "method": "axis-aligned box fitted to local reconstructed top surface"})
    write_json(HERE / "seats.json", seats)
    base = load_config(REPO / "robot_configs/humanoid_retarget_vita02a_example.json")
    base["smplx_model_dir"] = settings["smplx_model_dir"]
    base["robot"]["xml"] = settings["robot_xml"]
    base["correspondence"]["slots"] = settings["slots"]
    base["correspondence"]["dataset"].update({
        "out": str(HERE / "correspondence/dataset.npz"),
        "smpl_models": [{"type": "smplx", "dir": settings["smplx_model_dir"], "genders": ["neutral"]}],
    })
    base["correspondence"]["train"].update({
        "out_dir": str(HERE / "correspondence"), "template_name": "smplx_neutral", "device": "cpu",
    })
    for person in (1, 2):
        name = f"person_{person:02d}"
        d = np.load(HERE / f"source/human/{name}/smplx_params_world.npz")
        if (not d["valid"].all() or not np.allclose(d["betas"], d["betas"][0])
                or float(d["model_scale"]) != 1):
            raise ValueError(f"{name}: valid frames, constant betas and model_scale=1 are required.")
        poses = np.concatenate([d["global_orient"], d["body_pose"],
                                np.zeros((count, 9), np.float32),
                                d["left_hand_pose"], d["right_hand_pose"]], axis=1)
        if poses.shape != (count, 165) or not all(np.isfinite(a).all() for a in (poses, d["transl"], d["betas"])):
            raise ValueError(f"{name}: invalid SMPL-X pose, translation or shape.")
        path = HERE / "inputs" / f"{name}.npz"
        np.savez_compressed(path, poses=poses, trans=d["transl"], betas=d["betas"][0],
                            gender="neutral", mocap_frame_rate=fps, output_up="y",
                            frame_ids=timeline["frame_ids"], pts_seconds=timeline["pts_seconds"])
        config = json.loads(json.dumps(base))
        config.pop("_config_path", None)
        config.pop("_config_dir", None)
        config["robot"]["slot_name"] = "vita02a_shoes"
        config["motion"] = {"data": str(path), "seq_key": name, "start": 0, "end": -1,
                            "stride": 1, "max_frames": 0}
        config["hsi_hoi"] = {"source_sequence_dir": str(HERE / "objects"),
                             "object": {"output_up": "z", "convert_y_up": False,
                                        "ground_align": False, "object_scale": 1.,
                                        "multiple_objects": True}}
        config["solver"].update({
            "body_segment": {"module": "retarget_body_segment_surface_hoi_hsi",
                             "schema": "smplx_55_to_19_upper_arm_v1",
                             "upper_arm_split": {"ratio_start": 0., "ratio_end": .3}},
            "self_contact_map_cost": 0.,
            "ground_contact_map_cost": 30., "ground_contact_anchor_cost": 10.,
            "object_contact_map_cost": 30., "object_contact_map_threshold": .10,
            "object_contact_map_snap_threshold": .015, "object_contact_map_samples": 4096,
            "object_contact_map_max_points": 256, "retarget_object_size": "scaled",
            "robot_object_penetration_soft_cost": 1000.,
            "robot_object_hard_constraint": True, "robot_object_margin": 0.,
            "robot_object_hard_slack": True, "robot_object_hard_slack_cost": 3000.,
            "robot_object_hard_failure_fallback_to_soft": True,
            "robot_object_threshold": .06, "robot_object_max_pairs": 128,
            "iters": 6, "pose_init_iters": 60, "trajectory_warm_start_mode": "bidirectional",
            "trajectory_filter_mode": "off", "temporal_smooth_cost": .4})
        config["retarget"].update({"out": str(HERE / "results" / f"{name}.npz"),
                                   "source_ground_align": "none", "smplx_device": "cpu",
                                   "smplx_batch_size": 32})
        config["view"] = {"enabled": False}
        config_path = HERE / "configs" / f"{name}.json"
        write_json(config_path, config)
        loaded = load_config(config_path)
        slots = Path(config["correspondence"]["slots"])
        if slots.is_file() and not correspondence_slots_compatible(slots, loaded):
            raise ValueError(f"Incompatible correspondence: {slots}")
        write_json(HERE / "configs" / f"{name}_refined.json", {
            "extends": f"{name}.json",
            "solver": {
                "ground_contact_anchor_cost": 100., "ground_contact_map_cost": 150.,
                "ground_contact_map_snap_threshold": .065, "iters": 12, "pose_init_iters": 100,
                "robot_object_hard_slack_cost": 100000., "robot_object_penetration_soft_cost": 10000.,
            },
            "retarget": {"out": str(HERE / "results" / f"{name}_refined.npz")},
        })
        print(name, "frames", count, "correspondence", "compatible" if slots.is_file() else "pending learning", flush=True)
    write_json(HERE / "preparation.json", {
        "archive": str(archive), "archive_sha256": archive_sha256,
        "frames": count, "fps": fps, "source_y_up_to_z_up": U.tolist(),
        "shared_scale_policy": "same neutral correspondence template and robot height for both actors",
        "seats": seats, "dynamics_validated": False})
    print(json.dumps(seats, indent=2))


if __name__ == "__main__":
    main()
