import json
import os
from pathlib import Path
import subprocess
import sys

base=Path('/data/yihang.bai/training/hdmi_vita_mjlab')
out=base/'outputs/gallery08_p01_trajopt_20261030_2205'
motion=base/'batch_runs/exo_20261030_v1/gallery08_p01_contact_candidate_smooth3/mjlab_motion.npz'
records=[]
for name,joint in (('contact_projected_rootz_v1',False),('contact_projected_rootz_joints_v1',True)):
    command=[sys.executable,str(base/'trajectory_optimization/project_reference_contacts.py'),
        '--source',str(motion),'--model',str(out/'baseline_instrumented/compiled_model.mjb'),
        '--scene',str(motion.with_name('scene.json')),'--output',str(out/name),
        '--ground-audit',str(out/'independent_mesh_ground_audit.json')]
    if joint:command+=['--joint-project']
    log=(out/f'{name}.log').open('x')
    proc=subprocess.Popen(command,cwd=base,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),
        stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    records.append(dict(pid=proc.pid,output=str(out/name),joint_projection=joint));log.close()
print(json.dumps(records))
