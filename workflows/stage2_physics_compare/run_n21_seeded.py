"""Replay contact-projected targets and retry both optimizers with recorded seeds."""
import json
import os
from pathlib import Path
import subprocess
import sys

import mujoco as mj
import numpy as np

from common import load_canonical,write_json,sha256
from sbto_adapter import resample_reference


def main():
    root=Path(sys.argv[1]).resolve();code=Path(__file__).resolve().parent
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    source=load_canonical(root/'input/reference_raw.npz')
    projected=load_canonical(root/'contact_projection/reference.npz')
    np.testing.assert_array_equal(projected['qpos_ref'][0],source['qpos_ref'][0])
    np.testing.assert_array_equal(projected['source_times_s'],source['source_times_s'])
    _,_,_,controls=resample_reference(projected['qpos_ref'],projected['qvel_ref'],projected['ctrl_ref'],
        30.,1/300,np.arange(7,35))
    out=root/'contact_pd';out.mkdir(exist_ok=True)
    np.savez_compressed(out/'controls.npz',pd_target=controls,sim_dt=1/300)
    reports=[]
    for repeat in (1,2):
        folder=root/('contact_pd_replay_'+str(repeat))
        command=[sys.executable,str(code/'replay_validate.py'),'--method','sbto',
            '--reference',str(root/'input/reference_raw.npz'),'--model',str(root/'input/compiled_model.mjb'),
            '--controls',str(out/'controls.npz'),'--gates',str(root/'gates.json'),
            '--output',str(folder),'--sim-dt',str(1/300)]
        with (root/(folder.name+'.log')).open('w') as log:
            subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        reports.append(json.loads((folder/'report.json').read_text()))
    assert reports[0]['physical_record_sha256']==reports[1]['physical_record_sha256']
    write_json(root/'contact_pd_comparison.json',dict(method='contact SQP + PD',
        passed=reports[0]['passed'],replay=reports[0],repeated_record_hash_equal=True))

    # Use an exactly representable DSMS spline, then generate its native continuous states.
    dsms=json.loads((root/'dsms_fine_command.json').read_text())['command']
    upstream=Path(dsms[dsms.index('--dsms-root')+1]);sys.path.insert(0,str(upstream))
    from src.spline import LinearSpline,SplineConfig
    spline=LinearSpline(SplineConfig(M=len(source['qpos_ref']),spline_type='linear'),len(controls),28)
    seed_controls=spline.evaluate(source['ctrl_ref'])
    np.testing.assert_allclose(spline.evaluate(spline.fit(seed_controls)),seed_controls,atol=1e-12)
    model=mj.MjModel.from_binary_path(str(root/'input/compiled_model.mjb'));data=mj.MjData(model)
    data.qpos[:]=source['qpos_ref'][0];data.qvel[:]=source['qvel_ref'][0];mj.mj_forward(model,data)
    nodes=[np.r_[data.qpos.copy(),data.qvel.copy()]]
    for u in seed_controls:
        data.ctrl[:]=u;mj.mj_step(model,data)
        nodes.append(np.r_[data.qpos.copy(),data.qvel.copy()])
    warm=root/'dsms_physical_seed.npz'
    np.savez_compressed(warm,nodes=np.array(nodes),pd_target=seed_controls)
    write_json(root/'dsms_physical_seed.json',dict(model_sha256=sha256(root/'input/compiled_model.mjb'),
        reference_sha256=sha256(root/'input/reference_raw.npz'),seed_sha256=sha256(warm),
        source='continuous PD rollout of exactly representable original-reference control spline',
        initial_state_preserved=True,physics_steps=len(seed_controls),no_resets=True))

    sbto=json.loads((root/'sbto_fast_command.json').read_text())['command']
    sbto[sbto.index('--step-knots')+1]='100'
    sbto+=['--warmstart-controls',str(out/'controls.npz')]
    dsms+=['--warmstart',str(warm)]
    for tag,method,command in [('sbto_contact_seed','sbto',sbto),('dsms_physical_seed','dsms',dsms)]:
        command[command.index('--output')+1]=str(root/tag)
        spec=root/(tag+'_command.json');write_json(spec,dict(tag=tag,method=method,command=command))
        with (root/(tag+'_driver.log')).open('w') as log:
            job=subprocess.Popen([sys.executable,'-u',str(code/'run_n21_method.py'),str(spec)],
                env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(tag=tag,pid=job.pid)),flush=True)


if __name__=='__main__':main()
