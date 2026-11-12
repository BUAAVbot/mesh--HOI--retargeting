"""CPU-only independent audit of direct-v2 observations and positive BC labels."""
from __future__ import annotations
import argparse,hashlib,json,os,shutil,sys
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES']=''


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/data/KinoGen/outputs/sonic_expert_20261028'))
    p.add_argument('--source',type=Path,default=Path('/data/KinoGen/scripts/direct_proprio_seat_expert_20261028'))
    p.add_argument('--bc',type=Path,default=Path('/data/KinoGen/outputs/sonic_expert_20261028/direct_proprio_seat_positive_bc'))
    p.add_argument('--tag',default='direct_proprio_v2')
    a=p.parse_args()
    sys.path.insert(0,str(a.source))
    import mujoco as mj,numpy as np,torch
    from two_sitting_ppo_env import DualReference
    from direct_proprio import augment,CONTRACT
    torch.set_num_threads(2)
    out=a.root/'audit'/a.tag;out.mkdir(parents=True,exist_ok=True)
    names=('direct_proprio.py','bootstrap_positive_proprio.py','train_sugar_expert_20261028.py',
           'paired_physical_eval.py','evaluate_proprio_batch.py','sugar_expert_env.py')
    source_hashes={name:sha(a.source/name)for name in names}
    snapshot=out/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for name in names:shutil.copy2(a.source/name,snapshot/name)
    model=mj.MjModel.from_binary_path(str(a.root/'calibration/two_vita_expert.mjb'))
    ref=DualReference(model,a.root/'references/phc_time_scale1.5.npz','cpu')
    old=dict(np.load(a.root/'prior_physical_state_pool/candidate156_actual_states_controls.npz',allow_pickle=False))
    dataset=dict(np.load(a.bc/'offline_positive_observations.npz',allow_pickle=False))
    rows=dataset['actual_state_indices'];times=dataset['actual_time_s'];x=dataset['observations'];y=dataset['normalized_motor_offsets']
    q=old['qpos'][rows];v=old['qvel'][rows];scratch=mj.MjData(model);positions=[];cvel=[];rotations=[]
    for qq,vv in zip(q,v):
        scratch.qpos[:]=qq;scratch.qvel[:]=vv
        mj.mj_kinematics(model,scratch);mj.mj_comPos(model,scratch);mj.mj_comVel(model,scratch)
        positions.append(scratch.xpos.copy());cvel.append(scratch.cvel.copy())
        rotations.append(scratch.xmat[[mp['base']for mp in ref.maps]].reshape(2,3,3).copy())
    t=lambda value:torch.as_tensor(np.asarray(value),dtype=torch.float32)
    qt,vt,cvt,xpt,phase=t(q),t(v),t(cvel),t(positions),t(times)
    augmented=augment(model,ref,qt,vt,phase,cvt).numpy()
    target=ref.sample(phase)[0].numpy()
    previous=torch.zeros((len(rows),56))
    if len(rows)>1:previous[1:]=torch.atanh(t(y[:-1]))
    reconstructed=torch.cat([ref.observations(qt,vt,xpt,phase,previous),t(augmented)],-1).numpy()
    # Independent matrix-based expected146 channels, rather than reusing rotate_inverse.
    manual=[];seat_extra=[];spec=json.loads(Path('/data/KinoGen/outputs/two_sitting_pipeline/assets_deploy_v3/support_geometry.json').read_text())
    seats=np.asarray([[*seat['center_xy_m'],seat['top_z_m']]for seat in spec['seats']])
    support=ref.support_at(phase)[1].numpy();max_root_body_rotation=0.
    for actor,(rq,rv)in enumerate(zip(ref.root_qadr,ref.root_vadr)):
        actual=[];command=[]
        for qq,tt in zip(qt.numpy(),target):
            R=np.zeros(9);quaternion=qq[rq+3:rq+7].astype(float);quaternion/=np.linalg.norm(quaternion);mj.mju_quat2Mat(R,quaternion);actual.append(R.reshape(3,3))
            R=np.zeros(9);quaternion=tt[rq+3:rq+7].astype(float);quaternion/=np.linalg.norm(quaternion);mj.mju_quat2Mat(R,quaternion);command.append(R.reshape(3,3))
        actual=np.asarray(actual);command=np.asarray(command)
        max_root_body_rotation=max(max_root_body_rotation,float(np.max(abs(actual-np.asarray(rotations)[:,actor]))))
        inv=lambda R,value:np.einsum('nji,nj->ni',R,value)
        ids=ref.qa[actor*28:(actor+1)*28]
        gravity=np.tile([0.,0.,-1.],(len(rows),1))
        manual.extend([qt.numpy()[:,ids],target[:,ids],inv(actual,gravity),inv(command,gravity),
                       .25*inv(actual,vt.numpy()[:,rv:rv+3]),
                       .25*inv(actual,cvt.numpy()[:,ref.maps[actor]['base'],:3]),qt.numpy()[:,rq+2:rq+3]])
        chosen=np.sum((target[:,None,rq:rq+2]-seats[None,:,:2])**2,-1).argmin(-1)
        seat_extra.extend([inv(actual,seats[chosen]-qt.numpy()[:,rq:rq+3]),support[:,actor:actor+1]])
    manual=np.concatenate([np.clip(np.concatenate(manual,-1),-10,10),*seat_extra],-1)
    expected_label=(old['ctrl'][np.minimum(rows+3,len(old['ctrl'])-1)][:,ref.body_aids]-target[:,ref.qa])/2.2
    y_expected=np.clip(expected_label,-.98,.98)
    phase_all=t(old['time'][:-1]);target_all=ref.sample(phase_all)[0].numpy()
    raw_delta=old['ctrl'][:,ref.body_aids]-target_all[:,ref.qa]
    max_raw=float(np.max(abs(raw_delta)));maxloc=np.unravel_index(np.abs(raw_delta).argmax(),raw_delta.shape)
    body_names=[ref.names[col]for col in ref.body_cols]
    sampled_max=float(np.max(abs(old['ctrl'][np.minimum(rows+3,len(old['ctrl'])-1)][:,ref.body_aids]-target[:,ref.qa])))
    nearest=np.rint(np.arange(len(rows))*.02/.005).astype(int)
    target_packets=old['ctrl'][np.minimum(rows+3,len(old['ctrl'])-1)][:,ref.body_aids]
    nearest_packets=old['ctrl'][np.minimum(nearest+3,len(old['ctrl'])-1)][:,ref.body_aids]
    mapping_checks=[]
    for actor,prefix in enumerate(('p1_','p2_')):
        for column,name in enumerate(body_names):
            aid=int(ref.body_aids[actor*28+column]);jid=int(ref.jids[actor*28+column])
            mapping_checks.append(mj.mj_id2name(model,mj.mjtObj.mjOBJ_ACTUATOR,aid)==prefix+'motor_'+name and
                                  mj.mj_id2name(model,mj.mjtObj.mjOBJ_JOINT,jid)==prefix+name and model.actuator_trnid[aid,0]==jid)
    source_text={name:(a.source/name).read_text()for name in names}
    reuse={'training_imports_DirectProprioEnv':'from direct_proprio import DirectProprioEnv as RefinerEnv'in source_text['train_sugar_expert_20261028.py'],
           'screen_imports_same_DirectProprioEnv':'from direct_proprio import DirectProprioEnv as NativeTracking'in source_text['evaluate_proprio_batch.py'],
           'full_eval_imports_same_augment':'from direct_proprio import augment'in source_text['paired_physical_eval.py'],
           'full_eval_checks_v2_contract':"checkpoint.get('expert_observation_contract')=='direct_proprio_seat_v2'"in source_text['paired_physical_eval.py'],
           'offline_calls_same_augment':'augment(env.model,env.ref,q,v,phase,cv)'in source_text['bootstrap_positive_proprio.py'],
           'training_class_calls_same_augment':'augment(self.model,self.ref,self.q,self.v,self.phase,self.cvel)'in source_text['direct_proprio.py']}
    changed={name:sha(a.source/name)for name in names}
    checks={'no_gpu_visible':not torch.cuda.is_available(),'extra_dimension146':augmented.shape[1]==146,
        'dataset_total_dimension481':x.shape[1]==481,'manual_matrix_features_agree':float(np.max(abs(augmented-manual)))<1e-5,
        'stored_extra_features_agree':float(np.max(abs(x[:,-146:]-augmented)))<2e-5,
        'stored_full_observation_agrees':float(np.max(abs(x-reconstructed)))<2e-5,
        'bodyroot_quaternion_matches_actual_base_rotation':max_root_body_rotation<1e-5,
        'actor_joint_actuator_name_order_aligned':all(mapping_checks),
        'labels_match_actual_delayed_motor_packets':float(np.max(abs(y-y_expected)))<2e-6,
        'all_recorded_body_targets_fit_2p2rad_bias':max_raw<2.2*.98,
        'sampled_labels_not_clipped':bool(np.max(abs(expected_label))<.98),
        'train_screen_full_and_offline_reuse_augment':all(reuse.values()),
        'source_files_unchanged_during_audit':changed==source_hashes}
    report={'schema':'sonic_expert.direct_proprio_v2_independent_audit.v1','passed':all(checks.values()),
        'checks':checks,'source_hashes':source_hashes,'source_directory':str(a.source),'observation_contract':CONTRACT,
        'samples_checked':len(rows),'per_actor_base_dimension':69,'two_actor_base_dimension':138,'two_actor_seat_dimension':8,
        'extra_dimension':146,'total_dimension':int(x.shape[1]),
        'feature_order':'P1 actual q28, command q28, actual g3, command g3, actual linear body velocity3*0.25, actual angular body velocity3*0.25, actual root height1; P2 same; P1/P2 seat relative root-frame xyz3+expected-contact1',
        'max_manual_feature_absolute_error':float(np.max(abs(augmented-manual))),
        'max_stored_extra_feature_absolute_error':float(np.max(abs(x[:,-146:]-augmented))),
        'max_stored_full_observation_absolute_error':float(np.max(abs(x-reconstructed))),
        'max_label_absolute_error':float(np.max(abs(y-y_expected))),
        'all_recorded_body_motor_target_count':int(raw_delta.size),'all_body_target_max_abs_offset_rad':max_raw,
        'sampled_target_max_abs_offset_rad':sampled_max,'max_offset_control_interval':int(maxloc[0]),
        'max_offset_actor':int(maxloc[1]//28+1),'max_offset_joint':body_names[maxloc[1]%28],
        'effective_2p2rad_target_capacity_rad':2.2*.98,
        'old_1p5rad_capacity_clipped_fraction':float(np.mean(abs(raw_delta)>1.5*.98)),
        'timestamp_alignment_note':{'planned_policy_dt_s':.02,'physics_dt_s':.005,
            'state_row_mod4_counts':dict(zip(map(str,np.unique(rows%4)),map(int,np.unique(rows%4,return_counts=True)[1]))),
            'ceil_search_vs_nearest_row_mismatch_count':int(np.sum(rows!=nearest)),
            'maximum_state_time_lag_s':float(np.max(abs(times-np.arange(len(rows))*.02))),
            'resulting_body_packet_difference_max_rad':float(np.max(abs(target_packets-nearest_packets))),
            'interpretation':'Searchsorted-left can choose a state5ms after the intended20ms boundary. Actual timestamp/phase is recorded consistently; the original3-physics-step body delay makes compared body motor packets identical here. Nearest-boundary indexing is preferable for exact50Hz BC sampling.'},
        'shared_function_call_checks':reuse,
        'scope':'CPU read-only feature/label/source-interface verification. No training, GPU usage or new task-success claim.',
        'audit_source_sha256':sha(__file__)}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key]for key in ('passed','samples_checked','max_manual_feature_absolute_error',
        'max_stored_full_observation_absolute_error','max_label_absolute_error','all_body_target_max_abs_offset_rad')}),flush=True)
    print(json.dumps(checks),flush=True)


if __name__=='__main__':main()
