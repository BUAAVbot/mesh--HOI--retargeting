"""Grow overlapping optimized prefixes, each verified by continuous native replay."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import mujoco as mj
import numpy as np

from audit_dynamics_feasibility import required_force
from common import classify_robot_static_bodies,configure_position_pd,load_canonical,sha256,write_json
from replay_validate import contact_loads
from sbto_adapter import resample_reference


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--first-controls',type=Path,required=True)
    parser.add_argument('--first-reference',type=Path,required=True)
    parser.add_argument('--evaluations',type=int,default=40)
    args=parser.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    inputs=args.input.resolve();code=Path(__file__).resolve().parent
    model_path=inputs/'compiled_model.mjb';model=mj.MjModel.from_binary_path(str(model_path))
    configure_position_pd(model,mj);ref=load_canonical(inputs/'reference_raw.npz');spec=json.loads((inputs/'input_manifest.json').read_text())
    assert sha256(model_path)==spec['model_sha256'] and sha256(inputs/'reference_raw.npz')==spec['reference_sha256']
    gates=json.loads(Path('/data/yihang.bai/experiments/n21_dynamics_compare_20261105/gates.json').read_text())
    gates['contact_contract']='Latest retarget phases and stationary labels; original clock; actual dynamic state handoff'
    write_json(root/'gates.json',gates)
    write_json(root/'experiment.json',dict(source_sha256=spec['source_sha256'],reference_sha256=spec['reference_sha256'],
        model_sha256=spec['model_sha256'],gates_sha256=sha256(root/'gates.json'),
        block_source_frames=49,prefix_increment_intervals=24,physics_dt=1/300,initial_state='exact current latest qpos/qvel',
        no_midrollout_reset=True,actual_joint_limits_and_motor_effort_caps_unchanged=True))
    _,_,_,raw=resample_reference(ref['qpos_ref'],ref['qvel_ref'],ref['ctrl_ref'],30.,1/300,np.arange(7,35))
    with np.load(args.first_controls,allow_pickle=False) as z:first=z['pd_target'].copy()
    old=load_canonical(args.first_reference)
    _,_,_,old_raw=resample_reference(old['qpos_ref'],old['qvel_ref'],old['ctrl_ref'],30.,1/300,np.arange(7,35))
    assert first.shape==old_raw.shape
    first=raw[:len(first)]+first-old_raw
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
    robot,_,_=classify_robot_static_bodies(model,mj)
    feet=[mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,side+'_ankle_roll_link') for side in ('left','right')]
    windows=[];accepted=None;started=time.monotonic()

    def validate(folder,control,reference):
        rows=[]
        for repeat in (1,2):
            target=folder/('replay_'+str(repeat))
            command=[sys.executable,str(code/'replay_validate.py'),'--method','sbto',
                '--reference',str(reference),'--model',str(model_path),'--controls',str(control),
                '--gates',str(root/'gates.json'),'--output',str(target),'--sim-dt',str(1/300)]
            with (folder/('replay_'+str(repeat)+'.log')).open('w') as log:
                subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            rows.append(json.loads((target/'report.json').read_text()))
        assert rows[0]['physical_record_sha256']==rows[1]['physical_record_sha256']
        return dict(passed=rows[0]['passed'],report=rows[0],repeated_record_hash_equal=True)

    baseline=root/'raw_pd';baseline.mkdir();np.savez_compressed(baseline/'controls.npz',pd_target=raw,sim_dt=1/300)
    write_json(baseline/'comparison.json',validate(baseline,baseline/'controls.npz',inputs/'reference_raw.npz'))

    for stop in (*range(25,209,24),209):
        start=max(0,stop-49);count=(stop-start-1)*10
        folder=root/('prefix_'+str(stop-1).zfill(3));folder.mkdir()
        data=mj.MjData(model);data.qpos[:]=ref['qpos_ref'][0];data.qvel[:]=ref['qvel_ref'][0];mj.mj_forward(model,data)
        q=[data.qpos.copy()];v=[data.qvel.copy()];tau=[];acceleration=[];foot_forces=[];residual=[]

        def advance(us):
            actual=[]
            for u in us:
                data.ctrl[:]=u;mj.mj_step(model,data)
                assert not np.any(data.xfrc_applied) and not np.any(data.qfrc_applied)
                assert np.isclose(data.time,len(q)/300,atol=1e-9,rtol=0)
                q.append(data.qpos.copy());v.append(data.qvel.copy());actual.append(np.r_[data.qpos,data.qvel])
                tau.append(data.actuator_force.copy());acceleration.append(data.qacc.copy())
                foot_forces.append(contact_loads(model,data,robot,feet)[1])
                residual.append(required_force(model,data,data.qacc)-data.qfrc_actuator-data.qfrc_constraint)
            return actual

        fixed=accepted[:start*10] if accepted is not None else np.empty((0,28))
        advance(fixed)
        arrays={k:(z[start:stop].copy() if z.ndim and z.shape[0]==209 else z.copy()) for k,z in ref.items()}
        np.savez_compressed(folder/'reference.npz',**arrays)
        # This packet initializes optimizer copies; the physical data below is never overwritten.
        np.savez_compressed(folder/'initial_state.npz',qpos=data.qpos.copy(),qvel=data.qvel.copy(),
            time_s=data.time,qacc_warmstart=data.qacc_warmstart.copy(),model_sha256=np.array(sha256(model_path)))
        if accepted is None:seed=first[:count].copy()
        else:
            seed=raw[start*10:start*10+count].copy()
            overlap=accepted[start*10:];seed[:len(overlap)]=overlap
            last_offset=accepted[-1]-raw[len(accepted)-1]
            seed[len(overlap):]+=last_offset
        assert seed.shape==(count,28) and np.isfinite(seed).all()
        seed=np.clip(seed,model.actuator_ctrlrange[:,0],model.actuator_ctrlrange[:,1])
        np.savez_compressed(folder/'seed.npz',pd_target=seed,sim_dt=1/300)
        command=[sys.executable,'-u',str(code/'shoot_pd_controls.py'),'--reference',str(folder/'reference.npz'),
            '--model',str(model_path),'--seed',str(folder/'seed.npz'),'--output',str(folder/'optimized'),
            '--initial-state',str(folder/'initial_state.npz'),'--knots',str(5 if count<=240 else 7),
            '--evaluations',str(args.evaluations),'--finite-difference-step','.002']
        write_json(folder/'command.json',dict(command=command))
        with (folder/'optimizer.log').open('w') as log:
            subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        with np.load(folder/'optimized/controls.npz',allow_pickle=False) as z:optimized=z['pd_target'].copy()
        with np.load(folder/'optimized/native_shooting_record.npz',allow_pickle=False) as z:predicted=z['states'].copy()
        actual=advance(optimized)
        np.testing.assert_allclose(actual,predicted[1:],atol=1e-8,rtol=1e-10)
        all_controls=np.vstack([fixed,optimized])
        np.savez_compressed(folder/'controls.npz',pd_target=all_controls,sim_dt=1/300)
        committed_frames=len(q[::10]);prefix={k:(z[:committed_frames].copy() if z.ndim and z.shape[0]==209 else z.copy()) for k,z in ref.items()}
        np.savez_compressed(folder/'committed_reference.npz',**prefix)
        check=validate(folder,folder/'controls.npz',folder/'committed_reference.npz')
        with np.load(folder/'replay_1/physical_record.npz',allow_pickle=False) as z:
            np.testing.assert_allclose(np.array(q)[::10],z['qpos'],atol=1e-8,rtol=1e-10)
            np.testing.assert_allclose(np.array(v)[::10],z['qvel'],atol=1e-8,rtol=1e-10)
        item=dict(source_start_frame=start,committed_stop_frame=stop-1,
            target_stop_frame=stop-1,passed_prefix=check['passed'],
            root_rmse_m=check['report']['metrics']['root_rmse_m'],
            failed_gates=[k for k,z in check['report']['checks'].items() if not z],
            replay_record_sha256=check['report']['physical_record_sha256'])
        windows.append(item);write_json(root/'progress.json',dict(windows=windows,wall_s=time.monotonic()-started))
        np.savez_compressed(folder/'rollout_300hz.npz',time_s=np.arange(len(q))/300,qpos=np.array(q),qvel=np.array(v),
            qacc=np.array(acceleration),actuator_torque_Nm=np.array(tau),foot_force_world_N=np.array(foot_forces),
            dynamics_residual=np.array(residual),reset_count=0,state_writes_after_initialization=0)
        print(json.dumps(item),flush=True)
        if check['passed']:
            accepted=all_controls.copy()
            np.savez_compressed(root/'controls.npz',pd_target=accepted,sim_dt=1/300)
        if not check['passed']:
            print('Stopped after failed continuous prefix; no later source frame is claimed.',flush=True);break
    write_json(root/'result.json',dict(source_sha256=spec['source_sha256'],model_sha256=sha256(model_path),
        committed_source_frames=1+len(accepted)//10 if accepted is not None else 0,
        full_clip_passed=accepted is not None and len(accepted)==2080 and windows[-1]['passed_prefix'],
        windows=windows,wall_s=time.monotonic()-started))


if __name__=='__main__':main()
