"""Measured q-reference transfer to frozen SONIC; no policy training/distillation."""
from pathlib import Path
import hashlib,json,os,subprocess,time
import numpy as np

ROOT=Path('/data/KinoGen/outputs/sonic_expert_20261028')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    trial=ROOT/'terminal_refined_full_v4';out=ROOT/'sonic_refined_reference_transfer';out.mkdir(exist_ok=True)
    z=dict(np.load(trial/'trace.npz',allow_pickle=False));old=dict(np.load(ROOT/'references/phc_time_scale1.5.npz',allow_pickle=False))
    end=8.55;t=np.r_[np.arange(0,end,.02),end];ids=np.rint(t/.005).astype(int)
    if not np.allclose(z['time'][ids],t,atol=2e-12,rtol=0):raise ValueError('Actual physics samples do not match requested50Hz grid')
    q=np.stack([np.concatenate([z['qpos'][ids,rq:rq+7],z['qpos'][ids][:,z['joint_qadr'][actor]]],axis=1)
                for actor,rq in enumerate(z['robot_root_qadr'])],axis=1)
    previous=np.maximum(ids-1,0);report=json.loads((trial/'report.json').read_text())
    contact=z['seat_contact'][previous];force=z['seat_support_force_N'][previous]
    expected=contact&(force>=.10*np.asarray(report['robot_weight_N']))
    # Recorded foot_contact is per robot; recover left/right support from real contact records.
    feet=np.zeros((len(z['ctrl']),2,2),bool)
    for row,interval in enumerate(z['contact_interval_index']):
        for robot_side in (0,1):
            body=str(z['body_names'][z['contact_body_ids'][row,robot_side]])
            othergeom=str(z['geom_names'][z['contact_geom_ids'][row,1-robot_side]]).lower()
            upward=z['contact_force_world_on_geom2_N'][row,2]*(1 if robot_side==1 else -1)
            if not any(x in othergeom for x in ('floor','ground'))or upward<=.5:continue
            for actor,prefix in enumerate(('p1_','p2_')):
                for foot,side in enumerate(('left','right')):
                    if body.startswith(prefix)and side in body and any(x in body for x in ('ankle','foot')):feet[interval,actor,foot]=True
    ref=out/'cem_actual_q_reference_50hz.npz'
    np.savez_compressed(ref,qpos=q,time_s=t,source_time_s=z['source_time_s'][ids],dof_names=z['joint_names'],
        frame_ids=ids,source_hand_valid=np.zeros((len(t),2,2),bool),source_repaired_frame_mask=np.zeros(len(t),bool),
        foot_support_expected=feet[previous],seat_support_expected=expected,
        actual_seat_contact=contact,actual_seat_support_force_N=force,contact_inferred=np.asarray(False),
        source_trace_sha256=np.asarray(sha(trial/'trace.npz')),source_physics_indices=ids)
    initial=trial/'initial_state.npz';bags=dict(np.load(initial,allow_pickle=False))
    # Some source initial-state files use state rather than initial_state. Preserve exact vector/spec.
    vector=bags.get('initial_state',bags.get('state'))
    if not np.array_equal(vector,z['initial_state']):raise ValueError('Initial integration vector differs from validated source')
    init=out/'common_initial_state.npz'
    if not init.exists():np.savez_compressed(init,initial_state=vector,state=vector,state_spec=bags['state_spec'])
    elif not np.array_equal(np.load(init,allow_pickle=False)['initial_state'],vector):raise ValueError('Frozen common initial state changed')
    manifest={'schema':'sonic_expert.refined_q_reference_transfer.v1','purpose':'Measured functional CEM physical q-reference to unchanged frozen SONIC; not policy distillation or PPO success',
        'source_trace':str(trial/'trace.npz'),'source_trace_sha256':sha(trial/'trace.npz'),'refined_reference':str(ref),'refined_reference_sha256':sha(ref),
        'shape':list(q.shape),'time_grid':'0..8.54 by0.02s, plus exact8.55 endpoint; source9.56s final hold excluded',
        'end_s':end,'source_clock_end_s':float(z['source_time_s'][ids[-1]]),'final_hold_s':1.,'requested_total_s':9.55,
        'contact_labels':'Measured actual contact and >=10%weight upward force at preceding physical interval; no source coarse contact copied',
        'q_mapping':'For each actor actual free root7 followed by52 named joint_qadr in trace joint_names order; actual qvel not used as command',
        'same_model_sha256':sha(ROOT/'calibration/two_vita_expert.mjb'),'common_initial_state_sha256':sha(init),
        'unchanged_sonic_weights':True,'learned_residual_exact_zero':True,'learning_performed':False,'not_original_10_baselines':True,
        'hold_rounding':'50Hz policy cycles with4 physics substeps round9.55s to9.56s identically forA/B',
        'source_script_sha256':sha(__file__)}
    (out/'reference_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
    gpu0=next(line for line in gpu.splitlines()if line.split(',')[0].strip()=='0')
    if int(gpu0.split(',')[1].strip().split()[0])>100:raise RuntimeError('GPU0 is not idle; references prepared without launching')
    for tag,reference in [('A_phc_coarse',ROOT/'references/phc_time_scale1.5.npz'),('B_cem_refined',ref)]:
        output=out/tag
        if(output/'report.json').exists():continue
        if output.exists():raise FileExistsError(output)
        command=['/root/vita02a-mjlab/.venv310/bin/python','/data/KinoGen/scripts/residual_seeded_eval_200hz.py',
            '--model',str(ROOT/'calibration/two_vita_expert.mjb'),'--proxy-model','/data/KinoGen/outputs/two_sitting_pipeline/assets_deploy_v3/vita_single_proxy.xml',
            '--reference',str(reference),'--checkpoint',str(ROOT/'expert_phc_slow2_seeded_zerohead/initial.pt'),
            '--output',str(output),'--device','cuda:0','--zero-residual','--initial-state',str(init),'--hold-last-seconds','1']
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',PYTHONPATH='/data/KinoGen/scripts:/data/KinoGen/scripts/direct_proprio_expert_20261028')
        began=time.monotonic()
        with(out/(tag+'.log')).open('w')as log:result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        receipt={'command':command,'physical_gpu':0,'gpu0_idle_probe':gpu0,'returncode':result.returncode,'wall_s':time.monotonic()-began}
        (out/(tag+'.receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n')
        if result.returncode:raise RuntimeError(tag+' evaluation failed; inspectlog')
        r=json.loads((output/'report.json').read_text());print(json.dumps({'tag':tag,'functional':r['functional_task_success'],'strict':r['success'],'simulated_s':r['simulated_seconds']}),flush=True)
if __name__=='__main__':main()
