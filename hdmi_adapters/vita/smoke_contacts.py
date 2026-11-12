"""Smoke test: Vita02a URDF spawn + static/kinematic boxes + filtered contact sensors with contact points."""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
app = AppLauncher(args).app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from active_adaptation.assets import ROBOTS

NUM_ENVS = 4
scene_cfg = InteractiveSceneCfg(num_envs=NUM_ENVS, env_spacing=4.0, replicate_physics=False)
scene_cfg.terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")
robot = ROBOTS["vita02a"].copy()
robot.prim_path = "{ENV_REGEX_NS}/Robot"
robot.init_state.pos = (0.0, 0.0, 0.943)
scene_cfg.robot = robot
mat = sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0)
# static collider box under the left foot, kinematic box under the right foot (both 5 cm tall)
scene_cfg.box_static = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/BoxStatic",
    spawn=sim_utils.CuboidCfg(size=(0.4, 0.3, 0.05), collision_props=sim_utils.CollisionPropertiesCfg(), physics_material=mat),
    init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.2, 0.025)),
)
scene_cfg.box_kin = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/BoxKin",
    spawn=sim_utils.CuboidCfg(size=(0.4, 0.3, 0.05), collision_props=sim_utils.CollisionPropertiesCfg(), physics_material=mat,
                              rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                              mass_props=sim_utils.MassPropertiesCfg(mass=10.0)),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.2, 0.025)),
)
FILTERS = {
    "parent": ["/World/ground", "{ENV_REGEX_NS}/BoxStatic", "{ENV_REGEX_NS}/BoxKin"],
    "mesh": ["/World/ground/terrain/GroundPlane/CollisionPlane", "{ENV_REGEX_NS}/BoxStatic/geometry/mesh", "{ENV_REGEX_NS}/BoxKin"],
}
import os
which = os.environ.get("SMOKE_FILTER", "mesh")
print("[SMOKE] filter mode", which)
for side in ("left", "right"):
    setattr(scene_cfg, f"c_{side}", ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + f"{side}_ankle_roll_link",
        history_length=0, track_contact_points=True, track_friction_forces=True,
        max_contact_data_count_per_prim=16, filter_prim_paths_expr=FILTERS[which]))
scene_cfg.c_net = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*_ankle_roll_link", history_length=0)

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device="cuda:0"))
scene = InteractiveScene(scene_cfg)
sim.reset()
from pxr import Usd
stage = sim.stage
print("[SMOKE] ground prims:", [str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath("/World/ground")) ][:12])
print("[SMOKE] box prims:", [str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath("/World/envs/env_0/BoxStatic"))])
r = scene["robot"]
print("[SMOKE] bodies", len(r.body_names), r.body_names)
print("[SMOKE] joints", len(r.joint_names), r.joint_names)
print("[SMOKE] mass total", r.root_physx_view.get_masses()[0].sum().item())
for k, a in r.actuators.items():
    print("[SMOKE] act", k, type(a).__name__, a.joint_names, a.stiffness[0].tolist())
# hold default pose
push = float(os.environ.get("SMOKE_PUSH", "0"))
fric_sum_hist = []
if push != 0.0:
    # env0 pushed +x, env1 pushed -x (world frame) on base_link; others unpushed
    f = torch.zeros(NUM_ENVS, 1, 3, device=sim.device)
    f[0, 0, 0] = push
    f[1, 0, 0] = -push
    r.permanent_wrench_composer.set_forces_and_torques(
        forces=f, torques=torch.zeros_like(f), body_ids=[0], is_global=True)
for i in range(60):
    r.set_joint_position_target(r.data.default_joint_pos)
    scene.write_data_to_sim()
    sim.step()
    scene.update(0.005)
    if i >= 20:
        tot = torch.zeros(NUM_ENVS, 3, device=sim.device)
        for side in ("left", "right"):
            ff = scene[f"c_{side}"].data.friction_forces_w  # (N, 1, M, 3)
            tot += torch.nan_to_num(ff).sum(dim=(1, 2))
        fric_sum_hist.append(tot.cpu())
if fric_sum_hist:
    m = torch.stack(fric_sum_hist).mean(0)
    print("[SMOKE] push", push, "mean summed foot friction per env (x,y,z):", m.tolist())
    print("[SMOKE] base x drift per env:", (r.data.root_pos_w[:, 0] - scene.env_origins[:, 0]).tolist())
print("[SMOKE] base pos", r.data.root_pos_w[:, 2].tolist())
print("[SMOKE] feet pos env0", r.data.body_pos_w[0, [r.body_names.index("left_ankle_roll_link"), r.body_names.index("right_ankle_roll_link")]].tolist(), "env origin", scene.env_origins[0].tolist())
for side in ("left", "right"):
    s = scene[f"c_{side}"]
    print(f"[SMOKE] {side} filter_count", s.contact_physx_view.filter_count, "net", s.data.net_forces_w[0].tolist())
    print(f"[SMOKE] {side} force_matrix", s.data.force_matrix_w[0].tolist())
    print(f"[SMOKE] {side} contact_pos", s.data.contact_pos_w[0].tolist())
    print(f"[SMOKE] {side} friction", s.data.friction_forces_w[0].tolist())
    forces, points, normals, dists, cnt, start = s.contact_physx_view.get_contact_data(dt=0.005)
    print(f"[SMOKE] {side} raw: n_points_total", int(cnt.sum()), "counts env0", cnt[0].tolist(), "forces shape", tuple(forces.shape))
print("[SMOKE] net", scene["c_net"].data.net_forces_w[0].tolist())
print("[SMOKE] weight N", r.root_physx_view.get_masses()[0].sum().item() * 9.81)
import sys
sys.stdout.flush()
os._exit(0)
