"""Disclosed zero-update task-expert transfer screen to diagnose reference compatibility."""
from __future__ import annotations
import hashlib,json,os,subprocess
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import numpy as np,torch
    root=Path('/data/KinoGen/outputs/sonic_expert_20261028')
    out=root/'prior_transfer_screen';out.mkdir(exist_ok=True)
    oldpath=Path('/data/KinoGen/outputs/two_sitting_pipeline/training_native_v1/checkpoint_000450.pt')
    old=torch.load(oldpath,map_location='cpu',weights_only=False)
    model=root/'calibration/two_vita_expert.mjb';rows=[]
    for method in ('custom_ik_contact','hoi_retarget','umr_postprocessed','custom_ik'):
        reference=root/'references'/(method+'_transfer_time1p5.npz')
        if not reference.exists():
            values=dict(np.load(root/'references'/(method+'.npz'),allow_pickle=False))
            values['time_s']=values['time_s']*1.5;np.savez_compressed(reference,**values)
        ck=dict(old);ck.update(reference_path=str(reference),reference_sha256=sha(reference),
            model_path=str(model),model_sha256=sha(model),iteration=0,
            method='Zero-update transfer of previous task RL expert to different retarget prior; no new training',
            warm_transfer_source={'checkpoint':str(oldpath),'sha256':sha(oldpath),
                'old_reference_sha256':old['reference_sha256'],'old_training_iteration':old['iteration']})
        checkpoint=out/(method+'_zero_update.pt')
        if not checkpoint.exists():torch.save(ck,checkpoint)
        trial=out/method
        if not(trial/'report.json').exists():
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
            cmd=['/root/vita02a-mjlab/.venv310/bin/python','/data/KinoGen/scripts/evaluate_expert_batch.py',
                '--checkpoint',str(checkpoint),'--output',str(trial),'--num-envs','8']
            with(out/(method+'.log')).open('w')as stream:
                result=subprocess.run(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError('Transfer screen execution failed: '+method)
        report=json.loads((trial/'report.json').read_text())
        row={'method':method,'functional_passes':report['functional_passes'],
             'no_fall_worlds':sum(x['checks']['no_fall']for x in report['results']),
             'mean_root_xy_rms_m':sum(x['root_xy_rms_m']for x in report['results'])/len(report['results']),
             'checkpoint':str(checkpoint),'reference':str(reference),'num_envs':8}
        rows.append(row);print(json.dumps(row),flush=True)
    (out/'summary.json').write_text(json.dumps({'status':'zero_update_transfer_diagnostic',
        'old_expert_weights_retrained':False,'time_scale':1.5,'results':rows},indent=2)+'\n')


if __name__=='__main__':main()
