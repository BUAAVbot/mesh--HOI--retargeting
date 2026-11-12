"""Check that MuJoCo FK used by umr_to_motion.py matches Isaac (URDF) FK for the Vita02a asset."""
import argparse, json, os, sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--motion_dir", required=True)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
app = AppLauncher(args).app

import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.math import quat_error_magnitude
from active_adaptation.assets import ROBOTS

motion = np.load(os.path.join(args.motion_dir, "motion.npz"))
meta = json.load(open(os.path.join(args.motion_dir, "meta.json")))

scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=False)
scene_cfg.terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")
robot = ROBOTS["vita02a"].copy()
robot.prim_path = "{ENV_REGEX_NS}/Robot"
scene_cfg.robot = robot
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device="cuda:0"))
scene = InteractiveScene(scene_cfg)
sim.reset()
r = scene["robot"]
origin = scene.env_origins[0]
jidx = [meta["joint_names"].index(n) for n in r.joint_names]
bidx_isaac = [r.body_names.index(n) for n in meta["body_names"]]
root = meta["body_names"].index("base_link")

T = motion["joint_pos"].shape[0]
pos_err, ori_err = [], []
for t in range(0, T, max(1, T // 40)):
    pose = torch.tensor(np.r_[motion["body_pos_w"][t, root], motion["body_quat_w"][t, root]], dtype=torch.float32, device=sim.device)[None]
    pose[:, :3] += origin
    r.write_root_pose_to_sim(pose)
    r.write_root_velocity_to_sim(torch.zeros(1, 6, device=sim.device))
    jp = torch.tensor(motion["joint_pos"][t, jidx], dtype=torch.float32, device=sim.device)[None]
    r.write_joint_state_to_sim(jp, torch.zeros_like(jp))
    sim.forward()
    r.update(0.005)
    bp = (r.data.body_link_pos_w[0, bidx_isaac] - origin).cpu().numpy()
    bq = r.data.body_link_quat_w[0, bidx_isaac]
    pos_err.append(np.linalg.norm(bp - motion["body_pos_w"][t], axis=-1))
    ori_err.append(quat_error_magnitude(bq, torch.tensor(motion["body_quat_w"][t], device=sim.device)).cpu().numpy())
pos_err, ori_err = np.array(pos_err), np.array(ori_err)
print(f"[FK] frames checked {len(pos_err)}; pos err max {pos_err.max()*1000:.2f} mm mean {pos_err.mean()*1000:.3f} mm; ori err max {np.degrees(ori_err.max()):.3f} deg")
worst = pos_err.max(0).argsort()[::-1][:5]
print("[FK] worst bodies:", [(meta["body_names"][i], round(float(pos_err.max(0)[i]) * 1000, 2)) for i in worst])
sys.stdout.flush()
os._exit(0)
