import json
import os
from pathlib import Path
import subprocess
import sys

base=Path('/data/yihang.bai/training/hdmi_vita_mjlab')
out=base/'outputs/gallery08_p01_trajopt_20261030_2205'
motion=base/'batch_runs/exo_20261030_v1/gallery08_p01_contact_candidate_smooth3/mjlab_motion.npz'
scene=out/'mesh_calibrated_scene_v1/scene.json'
ckpt=base/'logs/rsl_rl/hdmi_vita28_exo_exo_20261030_v1_videomimic_gallery_08_v2__person_01/2026-10-30_18-31-07_repair_smooth3_continuity_v1_gpu4/model_2297.pt'
jobs=[('calibrated_scene_baseline_v2', [sys.executable,str(base/'trajectory_optimization/optimizer.py'),
    '--root',str(out/'runtime_snapshot'),'--motion',str(motion),'--scene',str(scene),'--checkpoint',str(ckpt),
    '--output',str(out/'calibrated_scene_baseline_v2'),'--baseline-only','--instrument']),
    ('contact_projected_hfield_joints_v1',[sys.executable,str(base/'trajectory_optimization/project_reference_contacts.py'),
    '--source',str(motion),'--model',str(out/'mesh_calibrated_scene_v1/compiled_model.mjb'),
    '--scene',str(scene),'--output',str(out/'contact_projected_hfield_joints_v1'),'--joint-project',
    '--ground-audit',str(out/'independent_mesh_ground_audit.json')])]
records=[]
for tag,command in jobs:
    log=(out/f'{tag}.log').open('x')
    proc=subprocess.Popen(command,cwd=base,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),
        stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    records.append(dict(pid=proc.pid,tag=tag,command=command));log.close()
print(json.dumps(records,indent=2))
