"""Reproduce a frozen 923-frame/six-object trial, audit, data export and videos.

An executed rollout is not automatically an accepted task. Failed task quality
remains explicit in the manifest and is never exported as accepted training data.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    root=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--sonic-config',type=Path,required=True)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--render-on-failure',action='store_true')
    p.add_argument('--diagnostic-export',action='store_true')
    a=p.parse_args()
    if os.name!='posix':p.error('Use the SSH server; local computer only previews outputs')
    if a.run_dir.exists():raise ValueError('Use a fresh reproduction directory')
    a.run_dir.mkdir(parents=True);trial=a.run_dir/'physics';steps=[]
    env=os.environ.copy();env.setdefault('CUDA_VISIBLE_DEVICES','1');env.setdefault('OPENBLAS_NUM_THREADS','1')
    env.setdefault('MUJOCO_GL','egl');env.setdefault('MUJOCO_EGL_DEVICE_ID','2')
    manifest={'schema':'fourth.full_pipeline_run.v1','status':'running','stages':steps,
        'inputs':{k:{'path':str(v.resolve()),'sha256':sha(v)}for k,v in [('model',a.model),('reference',a.reference),('sonic_config',a.sonic_config)]}}
    def save():
        (a.run_dir/'pipeline_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    def run(name,script,args):
        command=[sys.executable,str(root/'scripts'/script),*map(str,args)]
        log=a.run_dir/(name+'.log')
        with log.open('w',encoding='utf-8')as f:
            result=subprocess.run(command,env=env,cwd=root,stdout=f,stderr=subprocess.STDOUT)
        steps.append({'stage':name,'command':command,'returncode':result.returncode,'log':str(log),'log_sha256':sha(log)})
        save();print(json.dumps({'stage':name,'returncode':result.returncode,'log':str(log)}),flush=True)
        return result.returncode
    save()
    run('physics','fourth_full_physics.py',['--model',a.model,'--reference',a.reference,'--controller','sonic','--sonic-config',a.sonic_config,'--output-dir',trial])
    if not(trial/'report.json').exists():
        manifest['status']='failed_before_rollout_report';save();return 2
    report=json.loads((trial/'report.json').read_text())
    auditcode=run('audit','audit_fourth_full_physics.py',['--trial-dir',trial,'--require-six-objects','--require-full-coverage'])
    audited=auditcode==0
    quality=bool(report['simulation_passed'])
    manifest['physical_task_passed']=quality;manifest['full_coverage_replay_passed']=audited
    if audited and(quality or a.diagnostic_export):
        args=['--trial-dir',trial,'--output-dir',a.run_dir/'hoi']
        if not quality:args.append('--allow-failed-diagnostic')
        run('export','export_fourth_full_hoi.py',args)
    if audited and(quality or a.render_on_failure):
        for view in ['whole','table']:
            run('render_'+view,'render_fourth_full_sequence.py',['--model',report['model']['path'],'--reference',a.reference,'--trial-dir',trial,'--view',view,'--output',a.run_dir/('full923_'+view+'.mp4')])
    manifest['status']='accepted_full_sequence'if audited and quality and all(x['returncode']==0 for x in steps)else'full_sequence_requires_quality_work'
    manifest['artifacts']={str(f.relative_to(a.run_dir)):{'bytes':f.stat().st_size,'sha256':sha(f)}
        for f in sorted(a.run_dir.rglob('*'))if f.is_file()and f.name!='pipeline_manifest.json'}
    save();print(json.dumps({'status':manifest['status'],'manifest':str(a.run_dir/'pipeline_manifest.json')}),flush=True)
    return 0 if manifest['status']=='accepted_full_sequence'else 3


if __name__=='__main__':raise SystemExit(main())
