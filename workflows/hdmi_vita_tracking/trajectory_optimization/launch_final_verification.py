import json
import os
from pathlib import Path
import subprocess
import sys

base=Path('/data/yihang.bai/training/hdmi_vita_mjlab')
out=base/'outputs/gallery08_p01_trajopt_20261030_2205'
motion=base/'batch_runs/exo_20261030_v1/gallery08_p01_contact_candidate_smooth3/mjlab_motion.npz'
checkpoint=base/'logs/rsl_rl/hdmi_vita28_exo_exo_20261030_v1_videomimic_gallery_08_v2__person_01/2026-10-30_18-31-07_repair_smooth3_continuity_v1_gpu4/model_2297.pt'
records=[]
jobs=((6,'verified_warmblend_s311_gpu6','warmblend_s311_gpu1'),
      (7,'verified_warmblend_s563_gpu7','warmblend_s563_gpu3'))
if '--retry6' in sys.argv:
    jobs=((6,'verified_warmblend_s311_gpu6_retry','warmblend_s311_gpu1'),)
for gpu,tag,source in jobs:
    if not (out/source/'search_summary.json').is_file():raise RuntimeError('Search still running')
    command=[sys.executable,str(out/'training_tools/verify_candidate.py'),
        '--runtime-root',str(out/'runtime_snapshot'),'--optimizer',str(base/'trajectory_optimization/optimizer.py'),
        '--motion',str(motion),'--scene',str(motion.with_name('scene.json')),'--checkpoint',str(checkpoint),
        '--controls',str(out/source/'controls.npz'),'--output',str(out/tag),
        '--gates',str(out/'training_tools/gallery08_gates.v1.json'),
        '--scene-audit',str(out/'independent_scene_coordinates_audit.json'),
        '--batch-audit',str(out/'batch_invariance_audit.json'),'--gpu',str(gpu)]
    log=(out/f'{tag}.log').open('x')
    proc=subprocess.Popen(command,cwd=base,env=dict(os.environ,OMP_NUM_THREADS='2'),stdout=log,
        stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    records.append(dict(gpu=gpu,pid=proc.pid,output=str(out/tag)));log.close()
print(json.dumps(records))
