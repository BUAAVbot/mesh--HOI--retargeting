"""Retry the same DSMS window from a continuous, dynamically consistent PD seed."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import mujoco as mj
import numpy as np

from common import configure_position_pd, load_canonical, write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,required=True)
    args=parser.parse_args()
    root=args.experiment
    old=root/'window_0_80'
    manifest=json.loads((old/'experiment.json').read_text())
    reference=old/'reference.npz'
    ref=load_canonical(reference)
    model=mj.MjModel.from_binary_path(manifest['model'])
    configure_position_pd(model,mj)
    model.opt.timestep=.005
    n=len(ref['qpos_ref']);steps=4*(n-1)
    adrs=model.jnt_qposadr[model.actuator_trnid[:,0]]
    controls=np.column_stack([np.interp(np.arange(steps)*.005,np.arange(n)*.02,
        ref['qpos_ref'][:,a]) for a in adrs])
    sys.path.insert(0,str(root/'dsms-source'))
    from src.spline import SplineConfig, make_spline
    spline=make_spline(SplineConfig(M=n,spline_type='linear'),steps,model.nu)
    controls=spline.evaluate(spline.fit(controls))
    np.testing.assert_allclose(controls,spline.evaluate(spline.fit(controls)),atol=1e-12,rtol=0)

    def rollout():
        data=mj.MjData(model)
        data.qpos[:]=ref['qpos_ref'][0];data.qvel[:]=ref['qvel_ref'][0]
        states=[np.r_[data.qpos,data.qvel]]
        for i,u in enumerate(controls,1):
            data.ctrl[:]=u;mj.mj_step(model,data)
            states.append(np.r_[data.qpos,data.qvel])
        return np.array(states)
    nodes=rollout()
    np.testing.assert_array_equal(nodes,rollout())
    seed=root/'fine_physical_seed.npz'
    np.savez_compressed(seed,nodes=nodes,pd_target=controls)
    output=root/'dsms_step_nodes'
    code=Path(__file__).resolve().parent
    base=Path('/data/yihang.bai/experiments/stage2_same_track_20261102')
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
    env['LD_LIBRARY_PATH']=str(base/'envs/prime_core/lib')+':'+env.get('LD_LIBRARY_PATH','')
    command=[str(root/'env-dsms/bin/python'),'-u',str(code/'dsms_adapter.py'),
        '--dsms-root',str(root/'dsms-source'),'--reference',str(reference),
        '--model',manifest['model'],'--output',str(output),'--warmstart',str(seed),
        '--iterations','150','--threads','8','--node-dt','.005']
    with (root/'dsms_step_nodes.log').open('w') as stream:
        subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    reports=[]
    for i in (1,2):
        replay=root/('dsms_step_nodes_replay_'+str(i))
        command=[sys.executable,str(code/'replay_validate.py'),'--method','dsms',
            '--model',manifest['model'],'--reference',str(reference),
            '--controls',str(output/'controls.npz'),'--gates',manifest['gates'],'--output',str(replay)]
        with (root/(replay.name+'.log')).open('w') as stream:
            subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        reports.append(json.loads((replay/'report.json').read_text()))
    write_json(root/'dsms_step_nodes_comparison.json',dict(passed=all(r['passed'] for r in reports),
        repeated_record_hash_equal=reports[0]['physical_record_sha256']==reports[1]['physical_record_sha256'],reports=reports))


if __name__=='__main__':main()
