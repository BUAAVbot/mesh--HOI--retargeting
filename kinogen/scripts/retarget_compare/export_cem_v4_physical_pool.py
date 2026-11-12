"""Audit a functional CEM motor plan and export its actual physical reset states.

The planned demonstration is not a trained PPO expert. Reference disagreement
is kept as a label; actual physical support, not a coarse video contact label,
determines the seed contact/quiet validation.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/data/KinoGen/outputs/sonic_expert_20261028'))
    a=p.parse_args()
    import mujoco as mj,numpy as np
    from two_sitting_physics import independent_replay,interp
    root=a.root;trial=root/'terminal_refined_full_v4';out=root/'cem_v4_physical_state_pool';out.mkdir(exist_ok=True)
    report=json.loads((trial/'report.json').read_text());z=dict(np.load(trial/'trace.npz',allow_pickle=False))
    candidate_path=root/'terminal_motor_search_v4/candidate.npz';candidate=dict(np.load(candidate_path,allow_pickle=False))
    metadata=json.loads(str(candidate['metadata_json']))
    if not report['functional_task_success']or not report['independent_replay']['passed']:raise ValueError('CEM source has not passed physical acceptance')
    model_path=root/'calibration/two_vita_expert.mjb'
    if sha(model_path)!=sha(trial/'model.mjb'):raise ValueError('CEM source physical model differs')
    if not np.array_equal(candidate['ctrl'],z['ctrl']):raise ValueError('CEM source candidate/trace controls differ')
    old=dict(np.load(root/'prior_physical_state_pool/candidate156_actual_states_controls.npz',allow_pickle=False))
    changed=np.flatnonzero(np.any(z['ctrl']!=old['ctrl'],axis=1));first=int(changed[0])if len(changed)else len(z['ctrl'])
    prefix={'control_steps_identical':first,'prefix_duration_s':first*.005,
        'qpos_bitwise_identical_through_prefix':bool(np.array_equal(z['qpos'][:first+1],old['qpos'][:first+1])),
        'qvel_bitwise_identical_through_prefix':bool(np.array_equal(z['qvel'][:first+1],old['qvel'][:first+1])),
        'initial_integration_bitwise_identical':bool(np.array_equal(z['initial_state'],old['initial_state']))}
    if not all(prefix[key]for key in ('qpos_bitwise_identical_through_prefix','qvel_bitwise_identical_through_prefix','initial_integration_bitwise_identical')):
        raise ValueError('The inherited walking prefix is not bitwise reproduced')
    model=mj.MjModel.from_binary_path(str(model_path));spec=mj.mjtState(int(z['state_spec']))
    replay=independent_replay(model_path,z['initial_state'],spec,z['ctrl'],z['qpos'],z['qvel'])
    if not replay['passed']:raise ValueError('Independent current-model CEM replay failed')
    raw=dict(np.load(report['inputs']['reference'],allow_pickle=False))
    desired=np.asarray([interp(raw,t)for t in z['time']]);roots=np.asarray(z['robot_root_qadr'],int);rv=np.asarray(z['robot_root_vadr'],int)
    cols=np.asarray([i for i,n in enumerate(z['joint_names'])if not str(n).startswith(('lh_','rh_','neck_'))])
    qadr=np.asarray(z['joint_qadr']);body_aids=np.asarray(z['actuator_ids'])[:,cols].reshape(-1)
    q,v=z['qpos'],z['qvel'];xy=np.stack([np.linalg.norm(q[:,rq:rq+2]-desired[:,i,:2],axis=-1)for i,rq in enumerate(roots)],1)
    ze=np.stack([abs(q[:,rq+2]-desired[:,i,2])for i,rq in enumerate(roots)],1)
    oe=np.stack([2*np.arccos(np.clip(abs(np.sum(q[:,rq+3:rq+7]*desired[:,i,3:7],axis=-1)),0,1))for i,rq in enumerate(roots)],1)
    je=np.stack([np.sqrt(np.mean((q[:,qadr[i,cols]]-desired[:,i,7+cols])**2,-1))for i in range(2)],1)
    height=q[:,roots+2];tilt=np.stack([np.arccos(np.clip(1-2*(q[:,rq+4]**2+q[:,rq+5]**2),-1,1))for rq in roots],1)
    linear=np.stack([np.linalg.norm(v[:,vadr:vadr+3],axis=-1)for vadr in rv],1)
    angular=np.stack([np.linalg.norm(v[:,vadr+3:vadr+6],axis=-1)for vadr in rv],1)
    pen=np.zeros(len(z['ctrl']));np.maximum.at(pen,z['contact_interval_index'],np.maximum(-z['contact_signed_distance_m'],0))
    weight=np.asarray(report['robot_weight_N']);window=40;records=[];selected=[]
    for index in range(0,len(q)-window,4):
        stop=index+window;fraction=z['seat_contact'][index:stop].mean(0);force=z['seat_support_force_N'][index:stop].mean(0)
        seated=fraction>=.5
        stable=bool(np.isfinite(q[index:stop+1]).all()and np.isfinite(v[index:stop+1]).all()and height[index:stop+1].min()>.45
                    and tilt[index:stop+1].max()<.65 and pen[index:stop].max()<.025)
        load=bool(np.all(~seated|(force>=.10*weight)))
        quiet=bool(np.all(~seated|((linear[index:stop+1].max(0)<.20)&(angular[index:stop+1].max(0)<.4))))
        valid=stable and load and quiet
        if valid:selected.append(index)
        records.append({'state_index':index,'actual_time_s':float(z['time'][index]),'source_time_s':float(z['source_time_s'][index]),
            'future_physical_stability_pass':stable,'actual_seated_actor_mask':seated.tolist(),'future_load_pass':load,'future_quiet_pass':quiet,
            'valid':valid,'phc_root_xy_max_m':float(xy[index].max()),'phc_root_z_max_m':float(ze[index].max()),
            'phc_root_orientation_max_rad':float(oe[index].max()),'phc_body_joint_rms_max_rad':float(je[index].max()),
            'future_penetration_max_m':float(pen[index:stop].max()),'future_seat_force_mean_N':force.tolist()})
    ids=np.asarray(selected,int);chosen=[row for row in records if row['valid']]
    file=out/'terminal_refined_v4_actual_time_physical_contact_valid_pool.npz'
    np.savez_compressed(file,q=q[ids],v=v[ids],ctrl=z['ctrl'][np.minimum(ids,len(z['ctrl'])-1)],phase=z['time'][ids],
        previous=np.zeros((len(ids),56)),body_target=z['ctrl'][np.maximum(ids-1,0)][:,body_aids],
        captured_iteration=np.full(len(ids),-2,dtype=np.int64),root_xy_m=xy[ids].max(-1),
        penetration_m=np.asarray([row['future_penetration_max_m']for row in chosen]),
        actual_time_s=z['time'][ids],source_phase_s=z['source_time_s'][ids],state_index=ids,
        phc_root_z_m=ze[ids].max(-1),phc_root_orientation_rad=oe[ids].max(-1),phc_joint_rms_rad=je[ids].max(-1),
        future_actual_seated_actor_mask=np.asarray([row['actual_seated_actor_mask']for row in chosen],bool),
        future_seat_force_mean_N=np.asarray([row['future_seat_force_mean_N']for row in chosen]),
        is_cem_physical_positive_prior=np.ones(len(ids),bool),is_ppo_expert_success=np.zeros(len(ids),bool))
    (out/'per_candidate_quality.json').write_text(json.dumps(records,indent=2)+'\n')
    changed_aids=np.flatnonzero(np.any(z['ctrl']!=old['ctrl'],axis=0));changed_names=[mj.mj_id2name(model,mj.mjtObj.mjOBJ_ACTUATOR,int(i))for i in changed_aids]
    receipt={'schema':'sonic_expert.cem_actual_physical_seed.v1','status':'functional_cem_plan_seed_ready_not_ppo_expert',
        'source_trial':str(trial),'source_trace_sha256':sha(trial/'trace.npz'),'source_report_sha256':sha(trial/'report.json'),
        'source_candidate':str(candidate_path),'source_candidate_sha256':sha(candidate_path),'candidate_metadata':metadata,
        'source_functional_pass':True,'source_strict_reference_pass':False,'new_ppo_expert_success_claim':False,
        'model_sha256':sha(model_path),'inherited_old_walking_prefix':prefix,'changed_actuator_names':changed_names,
        'max_motor_offset_from_old_rad':float(np.max(abs(z['ctrl']-old['ctrl']))),'full_independent_replay':replay,
        'pool':{'path':str(file),'sha256':sha(file),'states':len(ids),'all_candidates':len(records),
            'actual_phase_coverage_s':[float(z['time'][ids].min()),float(z['time'][ids].max())],
            'q_v_ctrl_unchanged':True,'actual_qvel_scaled':False,'phase_is_actual_execution_time':True,
            'phase_contract':'phase=real source CEM execution time, not compressed source clock; nominal already includes historical delays.',
            'validation':'Future0.2s actual no-fall/rootTilt<0.65/penetration<25mm; actors physically seated for>=half-window require>=10%weight and root speed<0.20m/s, angular speed<0.4rad/s. PHC pose/path errors are labels, not validity gates.',
            'controller_history_contract':'previous new-feedback action=zero; body_target stores previous real full motor packet, not feedback offset. Consumer must reset its own feedback/delay history tozero, not treat this full motor packet as feedback.'},
        'physical_metrics':{key:report[key]for key in ('simulated_seconds','physics_steps','final_root_xy_error_m','max_root_xy_error_m',
            'max_root_z_error_m','max_root_orientation_error_rad','max_penetration_m','final_seat_vertical_force_mean_N',
            'final_root_linear_speed_max_m_s','final_root_angular_speed_max_rad_s')},'audit_source_sha256':sha(__file__)}
    (out/'report.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'pool':str(file),'states':len(ids),'phase':receipt['pool']['actual_phase_coverage_s'],'prefix':prefix,
        'changed_actuators':changed_names,'replay_passed':replay['passed']}),flush=True)


if __name__=='__main__':main()
