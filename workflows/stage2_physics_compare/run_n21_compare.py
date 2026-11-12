"""Compare the latest N21 motion on the new stairs using one frozen replay contract."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from common import load_canonical,sha256,write_json
from sbto_adapter import resample_reference


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.output.resolve();code=Path(__file__).resolve().parent
    ref=root/'input/reference_raw.npz';model=root/'input/compiled_model.mjb'
    arrays=load_canonical(ref)
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
    base=Path('/data/yihang.bai/experiments/stage2_same_track_20261102')
    gallery=Path('/data/yihang.bai/experiments/gallery08_dynamics_window_20261105')
    env['LD_LIBRARY_PATH']=str(base/'envs/prime_core/lib')+':'+env.get('LD_LIBRARY_PATH','')
    gates=root/'gates.json'
    write_json(gates,dict(schema='n21-new-stairs-dynamics-gates-v1',frozen_before_optimization=True,
        maximum=dict(root_max_m=.25,root_rmse_m=.1,final_root_error_m=.1,
            ee_z_max_m=.25,body_relative_max_m=.35,body_relative_rmse_m=.15,
            root_orientation_rmse_rad=.25,final_root_orientation_rad=.3,
            body_orientation_rmse_rad=.35,joint_rmse_rad=.3,joint_error_max_rad=1.,
            penetration_max_m=.005,self_penetration_max_m=.005,
            joint_limit_excess_max_rad=.02,effort_ratio_max=1.001,
            dynamics_residual_max_abs=1e-5,stationary_foot_slip_p95_m_s=.05),
        minimum=dict(foot_vertical_support_fraction=.8),
        contact_contract='Old phase timing retained as a declared prior; geometric anchors and forces recomputed on new stairs'))
    write_json(root/'experiment.json',dict(reference_sha256=sha256(ref),model_sha256=sha256(model),
        gates_sha256=sha256(gates),source_frames=209,source_dt=1/30,physics_dt=1/300,
        source_poses_unchanged=True,common_initial_state='Latest N21 first pose and zero velocity',
        candidate_geometry=True,claim_real_geometry_validated=False))

    def replay(method,control,tag):
        outputs=[]
        for repeat in (1,2):
            output=root/(tag+'_replay_'+str(repeat))
            command=[sys.executable,str(code/'replay_validate.py'),'--method',method,
                '--model',str(model),'--reference',str(ref),'--controls',str(control),
                '--gates',str(gates),'--output',str(output),'--sim-dt',str(1/300)]
            with (root/(output.name+'.log')).open('w') as stream:
                subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
            outputs.append(json.loads((output/'report.json').read_text()))
        assert outputs[0]['physical_record_sha256']==outputs[1]['physical_record_sha256']
        return dict(passed=all(r['passed'] for r in outputs),replay=outputs[0],repeated_record_hash_equal=True)

    _,_,_,controls=resample_reference(arrays['qpos_ref'],arrays['qvel_ref'],arrays['ctrl_ref'],
        30.,1/300,np.arange(7,35))
    np.savez_compressed(root/'baseline_controls.npz',pd_target=controls,sim_dt=1/300)
    results={'raw_pd':replay('sbto',root/'baseline_controls.npz','raw_pd')}
    write_json(root/'comparison.json',results)
    commands={
        'sbto':[str(base/'envs/sbto_core/bin/python'),'-u',str(code/'sbto_adapter.py'),
            '--sbto-root',str(base/'repos/sbto'),'--reference',str(ref),'--model',str(model),
            '--output',str(root/'sbto'),'--sim-dt',str(1/300),'--samples','64',
            '--max-iterations-per-knot','2','--step-knots','50','--threads','8','--seed','0',
            '--root-xy-weight','1600','--root-z-weight','3000','--root-orientation-weight','800',
            '--joint-weight','50','--terminal-root-scale','10','--contact-aware'],
        'dsms':[str(gallery/'env-dsms/bin/python'),'-u',str(code/'dsms_adapter.py'),
            '--dsms-root',str(gallery/'dsms-source'),'--reference',str(ref),'--model',str(model),
            '--output',str(root/'dsms'),'--sim-dt',str(1/300),'--node-dt',str(1/30),
            '--iterations','100','--threads','8']}
    jobs={}
    for method,command in commands.items():
        write_json(root/(method+'_command.json'),dict(command=command))
        stream=(root/(method+'.log')).open('w')
        jobs[method]=(subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT),stream,time.monotonic())
    while jobs:
        for method,(job,stream,started) in list(jobs.items()):
            if job.poll() is None:continue
            stream.close();row=dict(exit_code=job.returncode,wall_s=time.monotonic()-started)
            if job.returncode==0:row.update(replay(method,root/method/'controls.npz',method))
            results[method]=row;write_json(root/'comparison.json',results)
            print(json.dumps(dict(method=method,**row)),flush=True)
            del jobs[method]
        if jobs:time.sleep(2)


if __name__=='__main__':main()
