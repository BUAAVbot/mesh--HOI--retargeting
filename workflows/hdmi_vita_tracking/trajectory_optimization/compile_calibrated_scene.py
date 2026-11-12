import json
from pathlib import Path
import sys
import mujoco
import numpy as np
from scipy.interpolate import RegularGridInterpolator

base=Path('/data/yihang.bai/training/hdmi_vita_mjlab')
out=base/'outputs/gallery08_p01_trajopt_20261030_2205'
sys.path.insert(0,str(out/'runtime_snapshot'))
from task import make_env_cfg
from ground_scene import configure_ground_scene
from mjlab.scene import Scene
motion=base/'batch_runs/exo_20261030_v1/gallery08_p01_contact_candidate_smooth3/mjlab_motion.npz'
scene=out/'mesh_calibrated_scene_v1/scene.json'
cfg=make_env_cfg(str(motion),str(scene),play=True,profile='precision_start_limbs')
cfg.scene.num_envs=1;cfg.scene.env_spacing=0
metadata=configure_ground_scene(cfg,scene)
m=Scene(cfg.scene,device='cpu').compile();d=mujoco.MjData(m);mujoco.mj_forward(m,d)
target=out/'mesh_calibrated_scene_v1/compiled_model.mjb';mujoco.mj_saveModel(m,str(target))
gid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'calibrated_scene_surface')
with np.load(metadata['path']) as z:
    x,y,h=z['x'],z['y'],z['height_m']
interpolate=RegularGridInterpolator((y,x),h)
points=((-0.589575,-1.153242),(-0.080876,.473388),(.269124,.123388),(.714,.534))
rows=[]
for px,py in points:
    distance=mujoco.mj_rayHfield(m,d,gid,np.array([px,py,2.]),np.array([0.,0.,-1.]))
    actual=2-distance if distance>=0 else None
    desired=float(interpolate((py,px)))
    rows.append(dict(x=px,y=py,expected_bilinear_m=desired,native_height_m=actual,error_m=abs(actual-desired) if actual is not None else None))
report=dict(metadata=metadata,native_ray_checks=rows,max_error_m=max(r['error_m'] for r in rows),
            passed=all(r['error_m']<.002 for r in rows),model=str(target))
(out/'mesh_calibrated_scene_v1/native_heightfield_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
