"""Grow a continuous N21 dynamics optimization only after its prefix passes replay."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from common import load_canonical,sha256,write_json
from sbto_adapter import resample_reference


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--feedforward',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    input_dir=args.input.resolve();code=Path(__file__).resolve().parent
    model=input_dir/'compiled_model.mjb';source=load_canonical(input_dir/'reference_raw.npz')
    spec=json.loads((input_dir/'input_manifest.json').read_text())
    assert sha256(model)==spec['model_sha256'] and sha256(input_dir/'reference_raw.npz')==spec['reference_sha256']
    gates=json.loads(Path('/data/yihang.bai/experiments/n21_dynamics_compare_20261105/gates.json').read_text())
    gates['contact_contract']='Latest paired retarget phases and stationary labels; unchanged source clock and initial state'
    write_json(root/'gates.json',gates)
    write_json(root/'experiment.json',dict(input=str(input_dir),source_sha256=spec['source_sha256'],
        reference_sha256=spec['reference_sha256'],model_sha256=spec['model_sha256'],gates_sha256=sha256(root/'gates.json'),
        source_frames=209,source_fps=30,physics_dt=1/300,initial_state='exact latest qpos and qvel, no settling overwrite',
        no_midrollout_reset=True,source_clock_preserved=True))
    _,_,_,raw=resample_reference(source['qpos_ref'],source['qvel_ref'],source['ctrl_ref'],30.,1/300,np.arange(7,35))
    with np.load(args.feedforward,allow_pickle=False) as z:feedforward=z['pd_target'].copy()
    assert raw.shape==feedforward.shape==(2080,28)
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
    base=Path('/data/yihang.bai/experiments/stage2_same_track_20261102')
    accepted_controls=None;stages=[]

    def replay(folder,reference,controls):
        reports=[]
        for repeat in (1,2):
            target=folder/('replay_'+str(repeat))
            command=[sys.executable,str(code/'replay_validate.py'),'--method','sbto','--reference',str(reference),
                '--model',str(model),'--controls',str(controls),'--gates',str(root/'gates.json'),
                '--output',str(target),'--sim-dt',str(1/300)]
            with (folder/('replay_'+str(repeat)+'.log')).open('w') as log:
                subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            reports.append(json.loads((target/'report.json').read_text()))
        assert reports[0]['physical_record_sha256']==reports[1]['physical_record_sha256']
        row=dict(folder=str(folder),passed=reports[0]['passed'],metrics=reports[0]['metrics'],
            failed_gates=[k for k,v in reports[0]['checks'].items() if not v],
            repeated_record_hash_equal=True,physical_record_sha256=reports[0]['physical_record_sha256'])
        write_json(folder/'comparison.json',row)
        print(json.dumps(dict(folder=str(folder),passed=row['passed'],root_rmse_m=row['metrics']['root_rmse_m'],
            failed_gates=row['failed_gates'])),flush=True)
        return row

    def rank(row):return (not row['passed'],len(row['failed_gates']),row['metrics']['root_rmse_m'])

    for frames in (25,49,97,209):
        stage=root/('prefix_'+str(frames-1).zfill(3));stage.mkdir()
        reference=stage/'reference.npz'
        window={k:(v[:frames].copy() if v.ndim and v.shape[0]==209 else v.copy()) for k,v in source.items()}
        np.testing.assert_array_equal(window['qpos_ref'],source['qpos_ref'][:frames])
        np.testing.assert_array_equal(window['qvel_ref'][0],source['qvel_ref'][0])
        np.savez_compressed(reference,**window)
        count=10*(frames-1);candidates=[]
        for name,controls in [('raw_pd',raw[:count]),('inverse_dynamics_pd',feedforward[:count])]:
            folder=stage/name;folder.mkdir();path=folder/'controls.npz'
            np.savez_compressed(path,pd_target=controls,sim_dt=1/300)
            candidates.append(replay(folder,reference,path))
        best=min(candidates,key=rank)
        warm=feedforward[:count].copy()
        if accepted_controls is not None:warm[:len(accepted_controls)]=accepted_controls
        else:
            with np.load(Path(best['folder'])/'controls.npz',allow_pickle=False) as z:warm=z['pd_target'].copy()
        warm_path=stage/'warmstart.npz';np.savez_compressed(warm_path,pd_target=warm,sim_dt=1/300)
        for attempt in (0,1):
            folder=stage/('sbto_'+str(attempt));folder.mkdir()
            command=[str(base/'envs/sbto_core/bin/python'),'-u',str(code/'sbto_adapter.py'),
                '--sbto-root',str(base/'repos/sbto'),'--reference',str(reference),'--model',str(model),
                '--output',str(folder),'--sim-dt',str(1/300),'--samples',str(96 if attempt==0 else 128),
                '--max-iterations-per-knot',str(4 if attempt==0 else 6),'--step-knots','40',
                '--threads','8','--seed',str(attempt),'--sigma',str(.04 if attempt==0 else .015),
                '--root-xy-weight','1600','--root-z-weight','3000','--root-orientation-weight','800',
                '--joint-weight','50','--terminal-root-scale','20','--contact-aware',
                '--contact-tolerance','.0015','--penetration-weight','100000000',
                '--warmstart-controls',str(warm_path)]
            write_json(folder/'command.json',dict(command=command,reference_sha256=sha256(reference),
                model_sha256=sha256(model),adapter_sha256=sha256(code/'sbto_adapter.py')))
            with (folder/'optimizer.log').open('w') as log:
                subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            candidates.append(replay(folder,reference,folder/'controls.npz'))
            best=min(candidates,key=rank)
            if best['passed']:break
            with np.load(Path(best['folder'])/'controls.npz',allow_pickle=False) as z:warm=z['pd_target'].copy()
            np.savez_compressed(warm_path,pd_target=warm,sim_dt=1/300)
        stages.append(dict(frames=frames,source_duration_s=(frames-1)/30,candidates=candidates,best=best))
        write_json(root/'progress.json',dict(stages=stages,full_clip_passed=frames==209 and best['passed']))
        if not best['passed']:
            print(json.dumps(dict(stopped_at_source_frame=frames-1,reason='prefix not accepted; extension withheld')),flush=True)
            break
        with np.load(Path(best['folder'])/'controls.npz',allow_pickle=False) as z:accepted_controls=z['pd_target'].copy()
    write_json(root/'result.json',dict(source_sha256=spec['source_sha256'],stages=stages,
        full_clip_passed=stages[-1]['frames']==209 and stages[-1]['best']['passed']))


if __name__=='__main__':main()
