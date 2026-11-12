"""Screen completed seeded specialist checkpoints; no policy/model mutations."""
from __future__ import annotations
import json,os,subprocess,time
from pathlib import Path


ROOT=Path('/data/KinoGen/outputs/sonic_expert_20261028')
PYTHON='/root/vita02a-mjlab/.venv310/bin/python'
RUNS=('expert_phc_capacity1_positive_pool',)


def free_gpu():
    result=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],
                          capture_output=True,text=True,check=True)
    memory={int(row.split(',')[0]):int(row.split(',')[1])for row in result.stdout.splitlines()}
    return next((index for index in (0,2,4)if memory[index]<100),None)


def main():
    began=time.monotonic()
    while time.monotonic()-began<7200:
        pending=[]
        for run in RUNS:
            for iteration in (0,100,200,300,400):
                path=ROOT/run/('initial.pt'if iteration==0 else f'checkpoint_{iteration:06d}.pt')
                output=ROOT/'checkpoint_screen'/f'{run}_iter{iteration:03d}_n8'
                if path.exists()and not(output/'report.json').exists():pending.append((run,iteration,path,output))
        if pending:
            run,iteration,checkpoint,output=pending[0]
            gpu=free_gpu()
            if gpu is None:time.sleep(5);continue
            time.sleep(2)  # checkpoint writes must finish before loading
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
            command=[PYTHON,'/data/KinoGen/scripts/evaluate_seeded_expert_batch.py','--checkpoint',str(checkpoint),
                     '--output',str(output),'--num-envs','8','--seed','88001']
            result=subprocess.run(command,env=env,capture_output=True,text=True)
            (ROOT/'checkpoint_screen'/f'{run}_iter{iteration:03d}_n8.log').write_text(result.stdout+result.stderr)
            if result.returncode:
                print(json.dumps({'run':run,'iteration':iteration,'error':result.stderr[-1000:]}),flush=True)
                break
            report=json.loads((output/'report.json').read_text())
            rows=report['results']
            summary={'run':run,'iteration':iteration,'physical_gpu':gpu,'functional_passes':report['functional_passes'],
                     'strict_passes':report['strict_passes'],'no_fall':sum(row['checks']['no_fall']for row in rows),
                     'root_xy_rms_mean_m':sum(row['root_xy_rms_m']for row in rows)/len(rows),
                     'seat_force_N':[row['seat_force_N']for row in rows],
                     'output':str(output/'report.json')}
            with(ROOT/'residual_checkpoint_screens.jsonl').open('a')as stream:stream.write(json.dumps(summary)+'\n')
            print(json.dumps(summary),flush=True)
            continue
        if all((ROOT/run/'training_receipt.json').exists()or(ROOT/run/'early_stop_receipt.json').exists()for run in RUNS):break
        time.sleep(5)


if __name__=='__main__':main()
