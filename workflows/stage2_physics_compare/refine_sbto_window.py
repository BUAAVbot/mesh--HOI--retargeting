"""Refine the two failed Gallery08 gates from the first SBTO control solution."""
import json
import os
from pathlib import Path
import subprocess
import sys

from common import write_json

root=Path(sys.argv[1]).resolve()
variant=sys.argv[2] if len(sys.argv)>2 else 'sbto_refine'
seed_variant='sbto_refine' if variant=='sbto_self_guard' else 'window_0_80/sbto'
window=root/'window_0_80'
manifest=json.loads((window/'experiment.json').read_text())
base=Path('/data/yihang.bai/experiments/stage2_same_track_20261102')
code=Path(__file__).resolve().parent
env=os.environ.copy()
env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
command=[str(base/'envs/sbto_core/bin/python'),'-u',str(code/'sbto_adapter.py'),
    '--sbto-root',str(base/'repos/sbto'),'--reference',str(window/'reference.npz'),
    '--model',manifest['model'],'--output',str(root/variant),
    '--samples',('64' if variant=='sbto_self_guard' else '96'),
    '--max-iterations-per-knot',('2' if variant=='sbto_self_guard' else '3'),'--step-knots','20',
    '--threads','8','--seed','1','--sigma',('.02' if variant=='sbto_self_guard' else '.04'),
    '--root-xy-weight','1600','--root-z-weight','3000','--root-orientation-weight','800',
    '--joint-weight','50','--terminal-root-scale','10',
    '--warmstart-controls',str(root/seed_variant/'controls.npz'),'--contact-aware']
write_json(root/(variant+'_command.json'),dict(command=command))
with (root/(variant+'.log')).open('w') as stream:
    subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
reports=[]
for i in (1,2):
    replay=root/(variant+'_replay_'+str(i))
    command=[sys.executable,str(code/'replay_validate.py'),'--method','sbto',
        '--model',manifest['model'],'--reference',str(window/'reference.npz'),
        '--controls',str(root/variant/'controls.npz'),'--gates',manifest['gates'],'--output',str(replay)]
    with (root/(replay.name+'.log')).open('w') as stream:
        subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    reports.append(json.loads((replay/'report.json').read_text()))
write_json(root/(variant+'_comparison.json'),dict(passed=all(r['passed'] for r in reports),
    repeated_record_hash_equal=reports[0]['physical_record_sha256']==reports[1]['physical_record_sha256'],reports=reports))
