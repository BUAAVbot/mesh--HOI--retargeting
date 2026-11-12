"""Frozen Gallery08 failure-window comparison of official SBTO and DSMS cores."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import mujoco as mj
import numpy as np

from common import configure_position_pd, load_canonical, sha256, write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--dsms-root',type=Path,required=True)
    parser.add_argument('--dsms-python',type=Path,required=True)
    parser.add_argument('--frames',type=int,default=81)
    parser.add_argument('--methods',nargs='+',choices=['sbto','dsms'],default=['sbto','dsms'])
    args=parser.parse_args()
    root=args.output.resolve()
    root.mkdir(parents=True,exist_ok=True)
    old=Path('/data/yihang.bai/experiments/stage2_latest_gallery08_20261102')
    source=old/'contracts/canonical_contact2.npz'
    model=old/'preflight_contact_hdmi4999/compiled_model.mjb'
    gates=Path('/data/yihang.bai/training/hdmi_vita_mjlab/outputs/gallery08_p01_trajopt_20261030_2205/training_tools/gallery08_gates.v1.json')
    code=Path(__file__).resolve().parent
    original=load_canonical(source)
    count=len(original['qpos_ref'])
    if not 3 <= args.frames <= count:
        raise ValueError('Window frame count outside source')
    reference=root/'reference.npz'
    arrays={k:(v[:args.frames].copy() if v.ndim and len(v)==count else v.copy())
            for k,v in original.items()}
    for k in ('qvel_ref','qvel'):arrays[k][0]=0.
    if reference.exists():
        with np.load(reference,allow_pickle=False) as saved:
            if any(not np.array_equal(saved[k],v,equal_nan=v.dtype.kind in 'fc') for k,v in arrays.items()):
                raise ValueError('Existing experiment uses a different reference')
    else:
        np.savez_compressed(reference,**arrays)
    manifest=dict(source=str(source),source_sha256=sha256(source),
        reference_sha256=sha256(reference),model=str(model),model_sha256=sha256(model),
        gates=str(gates),gates_sha256=sha256(gates),source_frame_range=[0,args.frames-1],
        initialization='Same source initial q; zero velocity for both methods, matching prior strict replay',
        physics_dt=.005,source_dt=.02,methods=['DynaRetarget/SBTO','Shooting for Contact/DSMS'])
    write_json(root/'experiment.json',manifest)
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
    base=Path('/data/yihang.bai/experiments/stage2_same_track_20261102')
    env['LD_LIBRARY_PATH']=str(base/'envs/prime_core/lib')+':'+env.get('LD_LIBRARY_PATH','')

    def replay(method,controls,folder):
        command=[sys.executable,str(code/'replay_validate.py'),'--method',method,
            '--model',str(model),'--reference',str(reference),'--controls',str(controls),
            '--gates',str(gates),'--output',str(folder)]
        subprocess.run(command,env=env,check=True,stdout=(root/(folder.name+'.log')).open('w'))
        return json.loads((folder/'report.json').read_text())

    m=mj.MjModel.from_binary_path(str(model)); configure_position_pd(m,mj)
    adrs=m.jnt_qposadr[m.actuator_trnid[:,0]]
    controls=np.repeat(arrays['qpos_ref'][:-1,adrs],4,axis=0)
    np.savez_compressed(root/'baseline_controls.npz',pd_target=controls)
    replay('sbto',root/'baseline_controls.npz',root/'baseline_replay')
    commands={
        'sbto':[str(base/'envs/sbto_core/bin/python'),str(code/'sbto_adapter.py'),'--sbto-root',str(base/'repos/sbto'),
            '--reference',str(reference),'--model',str(model),'--output',str(root/'sbto'),
            '--samples','128','--max-iterations-per-knot','4','--step-knots','20',
            '--threads','8','--seed','0','--root-xy-weight','800','--root-z-weight','1500',
            '--root-orientation-weight','800','--contact-aware'],
        'dsms':[str(args.dsms_python),str(code/'dsms_adapter.py'),'--dsms-root',str(args.dsms_root),
            '--reference',str(reference),'--model',str(model),'--output',str(root/'dsms'),
            '--iterations','150','--threads','8']}
    processes={}
    for method,command in commands.items():
        if method not in args.methods:continue
        stream=(root/(method+'.log')).open('w')
        processes[method]=(subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT),stream,time.monotonic())
    comparison=root/'comparison.json'
    results=json.loads(comparison.read_text()) if comparison.exists() else {}
    while processes:
        for method,(process,stream,started) in list(processes.items()):
            if process.poll() is None:continue
            stream.close()
            row=dict(exit_code=process.returncode,elapsed_s=time.monotonic()-started)
            if process.returncode==0:
                reports=[replay(method,root/method/'controls.npz',root/(method+'_replay_'+str(i))) for i in (1,2)]
                row.update(passed=all(r['passed'] for r in reports),reports=reports,
                    repeated_record_hash_equal=reports[0]['physical_record_sha256']==reports[1]['physical_record_sha256'])
            results[method]=row
            write_json(root/'comparison.json',results)
            print(json.dumps(dict(method=method,**row)),flush=True)
            del processes[method]
        if processes:time.sleep(2)


if __name__=='__main__':main()
