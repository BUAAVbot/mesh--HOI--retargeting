"""Launch a bounded, recorded experiment matrix on idle GPUs only."""
from pathlib import Path
import argparse,json,os,signal,subprocess,time


def main():
    p=argparse.ArgumentParser();p.add_argument('--reference-fast',required=True);p.add_argument('--reference-slow',required=True)
    p.add_argument('--tag',required=True);p.add_argument('--iterations',type=int,default=600);p.add_argument('--supersede-pid',type=int);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];os.chdir(root)
    logs=root/'outputs/rgb145_pipeline/experiment_logs';logs.mkdir(exist_ok=True)
    if a.supersede_pid:
        proc=Path('/proc')/str(a.supersede_pid)/'cmdline'
        if proc.exists():
            argv=proc.read_bytes().decode().split('\0')
            if 'scripts/rgb145_train_ppo.py'not in argv or '--output'not in argv or argv[argv.index('--output')+1]!='outputs/rgb145_pipeline/ppo_sonic_residual_v2':raise ValueError('Refuse to signal unrelated process')
            old=root/'outputs/rgb145_pipeline/ppo_sonic_residual_v2'
            (old/'superseded.json').write_text(json.dumps({'reason':'Replaced by source-contact, relaxed-wrist and isolated-reset corrected experiment matrix','pid':a.supersede_pid,'argv':argv,'last_metrics':json.loads((old/'metrics.jsonl').read_text().splitlines()[-1])},indent=2)+'\n')
            os.kill(a.supersede_pid,signal.SIGINT)
            for _ in range(30):
                if not proc.exists():break
                time.sleep(.2)
    # GPU, method, pace, residual range, std, learning rate, start fraction, strict early termination.
    configs=[(2,True,'fast',1.6,.04,1e-4,.4,True),(3,True,'fast',1.6,.08,3e-4,.5,True),
             (4,True,'slow',1.6,.04,1e-4,.3,True),(5,False,'fast',.8,.08,1e-4,.5,True),
             (6,False,'slow',.8,.06,3e-4,.5,True),(7,True,'fast',1.6,.04,3e-4,.3,False)]
    gpu_raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    gpu_state={int(parts[0]):[int(v)for v in parts[1:]]for line in gpu_raw.splitlines()if len(parts:=[x.strip()for x in line.split(',')])==3}
    jobs=[]
    for gpu,sonic,pace,scale,std,lr,start,strict in configs:
        output=f'outputs/rgb145_pipeline/{a.tag}_gpu{gpu}'
        if Path(output).exists():raise FileExistsError(output)
        memory,util=gpu_state[gpu]
        if memory>1500 or util>10:
            jobs.append({'gpu':gpu,'status':'skipped_busy','memory_MiB':memory,'utilization_percent':util});continue
        ref=a.reference_fast if pace=='fast'else a.reference_slow
        cmd=['/data/tita_one/DDT_mjlab/.venv/bin/python','scripts/rgb145_train_ppo.py','--model','outputs/rgb145_pipeline/physics_sonic_v2_full/model.xml','--reference',ref,'--output',output,'--num-envs','512','--iterations',str(a.iterations),'--save-every','50','--action-scale',str(scale),'--initial-std',str(std),'--learning-rate',str(lr),'--start-probability',str(start),'--seed',str(145317+gpu)]
        if sonic:cmd.append('--sonic-base')
        if strict:cmd.append('--strict-training')
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_GL='egl',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')
        log=logs/f'{a.tag}_gpu{gpu}.log'
        with log.open('w')as handle:child=subprocess.Popen(cmd,env=env,stdout=handle,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        jobs.append({'gpu':gpu,'status':'launched','pid':child.pid,'output':output,'log':str(log),'argv':cmd,'sonic_base':sonic,'pace':pace,'strict_training':strict})
    record={'schema':'rgb145.idle_gpu_experiment_matrix.v1','tag':a.tag,'jobs':jobs,'no_physics_success_claim':True}
    (logs/f'{a.tag}.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)


if __name__=='__main__':main()
