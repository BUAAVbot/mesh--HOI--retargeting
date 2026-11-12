"""Numerically audit fixed collision bodies versus per-world robot origins."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from acceptance import sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runtime-root", type=Path, required=True)
    p.add_argument("--motion", type=Path, required=True)
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    sys.path.insert(0, str(args.runtime_root))
    from task import make_env_cfg
    from mjlab.envs import ManagerBasedRlEnv
    import mujoco
    import warp as wp
    rows = []
    scene_description = json.loads(args.scene.read_text())
    expected = scene_description.get("boxes", [])
    ground_helper = None
    if scene_description.get("ground_heightfield"):
        from ground_scene import configure_ground_scene
        ground_helper = Path(__file__).with_name("ground_scene.py")
    for count, spacing in ((1, 2.), (4, 2.), (4, 0.)):
        cfg = make_env_cfg(str(args.motion), str(args.scene), play=True, profile="precision_start_limbs")
        if ground_helper is not None:
            configure_ground_scene(cfg, args.scene)
        cfg.scene.num_envs, cfg.scene.env_spacing = count, spacing
        cfg.events, cfg.rewards, cfg.terminations, cfg.metrics = {}, {}, {}, {}
        for group in cfg.observations.values():
            group.enable_corruption = False
        env = ManagerBasedRlEnv(cfg, device=args.device)
        try:
            env.reset()
            env.sim.forward()
            env.scene.update(dt=env.physics_dt)
            origins = env.scene.env_origins.detach().cpu().numpy()
            robot = env.scene["robot"].data.root_link_pos_w.detach().cpu().numpy()
            boxes = []
            for i, source in enumerate(expected):
                entity = env.scene[f"seat_{i}"]
                bidx = entity.body_names.index(f"seat_{i}")
                actual = entity.data.body_link_pos_w[:, bidx].detach().cpu().numpy()
                desired = origins + np.asarray(source["pos"])
                boxes.append(dict(name=f"seat_{i}", actual=actual.tolist(),
                                  expected=desired.tolist(),
                                  position_error_m=np.linalg.norm(actual - desired, axis=-1).tolist(),
                                  actual_in_robot_origin_frame=(actual - origins).tolist()))
            model = env.sim.mj_model
            native = mujoco.MjData(model)
            mujoco.mj_forward(model, native)
            geom_positions = wp.to_torch(env.sim.wp_data.geom_xpos).detach().cpu().numpy()
            static = []
            for gid in range(model.ngeom):
                gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
                if gname.startswith("robot/"):
                    continue
                actual = geom_positions[:, gid]
                desired = origins + native.geom_xpos[gid]
                delta = actual - desired
                is_plane = model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_PLANE
                # XY translation of an infinite horizontal plane has no
                # physical effect; finite heightfields and boxes must align.
                if is_plane:
                    delta[:, :2] = 0.
                static.append(dict(name=gname, type=int(model.geom_type[gid]),
                                   actual=actual.tolist(), expected=desired.tolist(),
                                   position_error_m=np.linalg.norm(delta, axis=-1).tolist()))
            rows.append(dict(count=count, spacing=spacing, origins=origins.tolist(),
                             robot_root_world=robot.tolist(), boxes=boxes,
                             static_geometries=static))
        finally:
            env.close()
    report = dict(schema="hdmi.trajopt.scene_coordinates_audit.v1",
                  scene_sha256=sha256(args.scene), source_reference_sha256=sha256(args.motion),
                  runtime_task_sha256=sha256(args.runtime_root / "task.py"), rows=rows)
    if ground_helper is not None:
        report["runtime_ground_scene_sha256"] = sha256(ground_helper)
    args.output.write_text(json.dumps(report, indent=2))
    for row in rows:
        print(json.dumps(dict(count=row["count"], spacing=row["spacing"], origins=row["origins"],
                              seat0=row["boxes"][0] if row["boxes"] else None,
                              static_geometry_max_error_m=max(max(g["position_error_m"]) for g in row["static_geometries"]),
                              robot_root_world=row["robot_root_world"])))


if __name__ == "__main__":
    main()
