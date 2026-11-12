"""Execute a recorded N21 method command, then perform two frozen continuous replays."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from common import write_json,sha256


def main():
    spec_path=Path(sys.argv[1]).resolve();spec=json.loads(spec_path.read_text())
    root=spec_path.parent;tag=spec['tag'];method=spec['method'];code=Path(__file__).resolve().parent
    if method not in ('sbto','dsms') or not tag.replace('_','').isalnum():raise ValueError('Invalid method identity')
    output=root/tag;output.mkdir(exist_ok=True)
    adapter=Path(spec['command'][2])
    # Commands invoke python -u adapter.py, and bind the executed source before it can change.
    if adapter.suffix!='.py':raise ValueError('Expected a recorded Python adapter command')
    shutil.copy2(adapter,output/'adapter_used.py')
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
    env['LD_LIBRARY_PATH']='/data/yihang.bai/experiments/stage2_same_track_20261102/envs/prime_core/lib:'+env.get('LD_LIBRARY_PATH','')
    started=time.monotonic()
    with (root/(tag+'.log')).open('w') as log:
        job=subprocess.run(spec['command'],env=env,stdout=log,stderr=subprocess.STDOUT)
    result=dict(method=method,tag=tag,exit_code=job.returncode,wall_s=time.monotonic()-started,
        adapter_sha256=sha256(output/'adapter_used.py'),command=spec['command'])
    if job.returncode==0:
        reports=[]
        for repeat in (1,2):
            folder=root/(tag+'_replay_'+str(repeat))
            command=[sys.executable,str(code/'replay_validate.py'),'--method',method,
                '--reference',str(root/'input/reference_raw.npz'),'--model',str(root/'input/compiled_model.mjb'),
                '--controls',str(output/'controls.npz'),'--gates',str(root/'gates.json'),
                '--output',str(folder),'--sim-dt',str(1/300)]
            with (root/(folder.name+'.log')).open('w') as log:
                subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            reports.append(json.loads((folder/'report.json').read_text()))
        assert reports[0]['physical_record_sha256']==reports[1]['physical_record_sha256']
        result.update(passed=all(r['passed'] for r in reports),replay=reports[0],repeated_record_hash_equal=True)
    write_json(root/(tag+'_comparison.json'),result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
