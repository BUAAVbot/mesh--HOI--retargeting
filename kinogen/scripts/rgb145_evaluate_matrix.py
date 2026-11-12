"""Run bounded concurrent native acceptance evaluations of matrix checkpoints."""
from pathlib import Path
import argparse,json,os,subprocess,time


def main():
    p=argparse.ArgumentParser();p.add_argument('--matrix',type=Path,required=True);p.add_argument('--iteration',type=int,required=True);p.add_argument('--gpu',default='0');p.add_argument('--concurrency',type=int,default=3);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];os.chdir(root);matrix=json.loads(a.matrix.read_text())
    tasks=[];records=[]
    for job in matrix['jobs']:
        if job['status']!='launched':continue
        base=Path(job['output']);checkpoint=base/f'checkpoint_{a.iteration:06d}.pt';output=base.with_name(base.name+f'_eval{a.iteration:04d}')
        if not checkpoint.exists():records.append({'training':str(base),'status':'checkpoint_pending'});continue
        if output.exists():records.append({'training':str(base),'status':'already_evaluated','output':str(output)});continue
        argv=job['argv'];ref=argv[argv.index('--reference')+1];model=argv[argv.index('--model')+1]
        cmd=['/data/tita_one/DDT_mjlab/.venv/bin/python','scripts/rgb145_physics.py','--model',model,'--reference',ref,'--task-policy',str(checkpoint),'--output',str(output),'--finger-driver-rotor-kg-m2','0','--finger-follower-regularization-kg-m2','0']
        tasks.append((cmd,base,output))
    running=[]
    while tasks or running:
        while tasks and len(running)<a.concurrency:
            cmd,base,out=tasks.pop(0);log=out.with_suffix('.log');handle=log.open('w')
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=a.gpu,MUJOCO_GL='egl',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
            child=subprocess.Popen(cmd,stdout=handle,stderr=subprocess.STDOUT,env=env,stdin=subprocess.DEVNULL)
            running.append((child,handle,base,out));print(json.dumps({'launched_evaluation':str(out),'pid':child.pid}),flush=True)
        for row in running[:]:
            child,handle,base,out=row
            if child.poll()is None:continue
            handle.close();running.remove(row);report=out/'report.json'
            record={'training':str(base),'output':str(out),'return_code':child.returncode}
            if report.exists():
                r=json.loads(report.read_text());record.update(status=r['status'],simulation_passed=r['simulation_passed'],duration_s=r['duration_s'],source_covered_s=r['source_time_covered_s'],root_xy_max_m=r['root_horizontal_error_m_max'],root_angle_max_rad=r['root_orientation_error_rad_max'],checks=r['checks'])
            else:record['status']='evaluation_error'
            records.append(record);print(json.dumps(record),flush=True)
        if running:time.sleep(1.)
    path=a.matrix.with_name(a.matrix.stem+f'_native_eval{a.iteration:04d}.json');path.write_text(json.dumps(records,indent=2)+'\n')


if __name__=='__main__':main()
