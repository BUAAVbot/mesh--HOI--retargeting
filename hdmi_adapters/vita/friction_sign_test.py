"""Determine the sign convention of PhysX friction data: 10 kg cube on ground, pushed +x by 30 N (static)."""
import os, sys
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg

cfg = InteractiveSceneCfg(num_envs=2, env_spacing=3.0, replicate_physics=False)
cfg.terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")
cfg.cube = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Cube",
    spawn=sim_utils.CuboidCfg(size=(0.3, 0.3, 0.3), collision_props=sim_utils.CollisionPropertiesCfg(),
                              rigid_props=sim_utils.RigidBodyPropertiesCfg(), mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
                              activate_contact_sensors=True,
                              physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0)),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.151)),
)
cfg.c = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Cube", history_length=0, track_contact_points=True,
                         track_friction_forces=True, max_contact_data_count_per_prim=16,
                         filter_prim_paths_expr=["/World/ground/terrain/GroundPlane/CollisionPlane"])
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device="cuda:0"))
scene = InteractiveScene(cfg)
sim.reset()
cube = scene["cube"]
f = torch.zeros(2, 1, 3, device=sim.device)
f[0, 0, 0] = 30.0
f[1, 0, 1] = -30.0
cube.permanent_wrench_composer.set_forces_and_torques(forces=f, torques=torch.zeros_like(f), is_global=True)
for i in range(200):
    scene.write_data_to_sim(); sim.step(); scene.update(0.005)
s = scene["c"]
print("[FRIC] applied env0 +30x, env1 -30y")
print("[FRIC] friction_forces_w:", s.data.friction_forces_w[:, 0, 0].tolist())
print("[FRIC] force_matrix_w (normal):", s.data.force_matrix_w[:, 0, 0].tolist())
ff, fp, cnt, st = s.contact_physx_view.get_friction_data(dt=0.005)
print("[FRIC] raw friction env0 pts:", ff[int(st[0,0]):int(st[0,0])+int(cnt[0,0])].tolist())
print("[FRIC] cube drift:", (cube.data.root_pos_w - scene.env_origins).tolist())
sys.stdout.flush(); os._exit(0)
