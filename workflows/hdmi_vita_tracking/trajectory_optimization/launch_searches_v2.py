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
for gpu, seed, left, right, scope in ((0, 17, -25, 310, 'legs'), (1, 42, -40, 320, 'legs'),
                                      (2, 83, -50, 330, 'all'), (3, 123, -30, 300, 'legs')):
    used = int(subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.used', '--format=csv,noheader,nounits']).decode().strip())
    if used > 300:
        raise RuntimeError(f'GPU{gpu} unexpectedly occupied: {used}MiB')
    tag = f'correctscene_s{seed}_gpu{gpu}'
    command = [sys.executable, str(base / 'trajectory_optimization/optimizer.py'), '--root', str(out / 'runtime_snapshot'),
        '--motion', str(motion), '--scene', str(motion.with_name('scene.json')), '--checkpoint', str(checkpoint),
        '--output', str(out/tag), '--samples', '64', '--iterations', '30', '--seed', str(seed),
        '--left', str(left), '--right', str(right), '--std', '.08', '--bound', '.3', '--joint-scope', scope]
    log = (out/f'{tag}.log').open('x')
    process = subprocess.Popen(command, cwd=base, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2'),
        stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    records.append(dict(pid=process.pid, gpu=gpu, command=command, log=log.name))
    log.close()
(out/'correctscene_launches.json').write_text(json.dumps(records,indent=2))
print(json.dumps(records,indent=2))
