import json
import os
from pathlib import Path
import subprocess
import sys

base = Path('/data/yihang.bai/training/hdmi_vita_mjlab')
out = base / 'outputs/gallery08_p01_trajopt_20261030_2205'
motion = base / 'batch_runs/exo_20261030_v1/gallery08_p01_contact_candidate_smooth3/mjlab_motion.npz'
checkpoint = base / 'logs/rsl_rl/hdmi_vita28_exo_exo_20261030_v1_videomimic_gallery_08_v2__person_01/2026-10-30_18-31-07_repair_smooth3_continuity_v1_gpu4/model_2297.pt'
records = []
for gpu, name, controls in ((0, 'baseline_instrumented', None),
                            (1, 'candidate_s42_instrumented', out / 'search_s42_gpu1/controls.npz')):
    command = [sys.executable, str(base / 'trajectory_optimization/optimizer.py'), '--root', str(out / 'runtime_snapshot'),
               '--motion', str(motion), '--scene', str(motion.with_name('scene.json')), '--checkpoint', str(checkpoint),
               '--output', str(out / name), '--instrument']
    command += ['--controls', str(controls)] if controls else ['--baseline-only']
    log = (out / f'{name}.log').open('x')
    process = subprocess.Popen(command, cwd=base, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2'),
        stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    records.append(dict(pid=process.pid, gpu=gpu, output=str(out / name)))
    log.close()
print(json.dumps(records))
