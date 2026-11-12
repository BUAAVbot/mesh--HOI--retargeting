"""Consolidate verified dual-Vita functional demos without promoting tracking.

This is a read-only evidence audit, except for its fresh output directory.
No policy, physics, optimization, rendering or training job is launched.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime,timezone,timedelta

IDS=(240,169,218,156,146)
CHOSEN=156


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def require(value,message):
    if not value:raise ValueError(message)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/data/KinoGen'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve()
    if a.output.exists():raise FileExistsError('Use a fresh audit output directory: '+str(a.output))
    base=root/'outputs/two_sitting_pipeline';searchdir=base/'search_native450_settle_v2'
    searchpath=searchdir/'summary.json';search=read(searchpath)
    reports={i:read(base/f'physics_selected450_candidate{i}/report.json')for i in IDS}
    chosen=reports[CHOSEN];chosen_dir=base/f'physics_selected450_candidate{CHOSEN}'
    total=int(search['total_attempts']);attempts=search['all_attempts']
    require(total==256 and len(attempts)==256,'Expected the complete256-attempt search evidence')
    require(len({int(row['candidate_id'])for row in attempts})==256,'Duplicate/missing search candidate identifiers')
    preliminary={int(row['candidate_id'])for row in attempts if row.get('preliminary_selected')}
    require(preliminary==set(IDS),'Selected five reports do not equal the five preliminary selected search attempts')
    hashes={};reference_hashes={};model_hashes={};evidence=[]
    for candidate,r in reports.items():
        require(r.get('functional_physical_demo_pass')is True,f'Candidate{candidate} has no verified functional pass')
        require(all(r['functional_task_checks'].values()),f'Candidate{candidate} functional checks contain a failure')
        require(r.get('source_tracking_pass')is False and r.get('success')is False,f'Unexpected strict tracking outcome for candidate{candidate}')
        require(r.get('termination')=='completed'and int(r['physics_steps'])==1912,f'Candidate{candidate} incomplete replay')
        require(abs(float(r['simulated_seconds'])-9.56)<1e-8,f'Candidate{candidate} duration mismatch')
        for field in ('state_override_after_reset_count','external_force_steps','root_actuators','mocap_bodies'):
            require(r.get(field)==0,f'Candidate{candidate} lacks zero {field} evidence')
        replay=r['independent_replay'];consistency=r['candidate_policy_state_consistency']
        require(replay.get('passed')is True and replay['max_qpos_absolute_error']==0 and replay['max_qvel_absolute_error']==0,f'Candidate{candidate} control-only replay mismatch')
        require(consistency.get('passed')is True and consistency.get('complete')is True and consistency['max_qpos_absolute_error']==0 and consistency['max_qvel_absolute_error']==0,f'Candidate{candidate} policy-state consistency mismatch')
        hashes[candidate]=r['controller']['checkpoint_sha256'];reference_hashes[candidate]=r['inputs']['reference_sha256'];model_hashes[candidate]=r['inputs']['model_sha256']
        path=base/f'physics_selected450_candidate{candidate}/report.json'
        evidence.append({'candidate_id':candidate,'report_path':str(path),'report_sha256':digest(path),
                         'functional_physical_demo_pass':True,'strict_source_tracking_pass':False,
                         'physics_steps':r['physics_steps'],'simulated_seconds':r['simulated_seconds'],
                         'max_root_xy_error_m':r['max_root_xy_error_m'],'max_penetration_m':r['max_penetration_m'],
                         'independent_replay':replay,'candidate_policy_state_consistency':consistency})
    require(len(set(hashes.values()))==1,'Multiple residual checkpoints were mixed')
    require(len(set(reference_hashes.values()))==1 and len(set(model_hashes.values()))==1,'Multiple reference or physical models were mixed')
    checkpoint_hash=next(iter(hashes.values()))
    require(checkpoint_hash==search['checkpoint_sha256'],'Search/checkpoint provenance mismatch')
    require(int(search['checkpoint_iteration'])==450 and int(search['seed'])==174115,'Unexpected checkpoint iteration or sampling seed')
    noise=search['noise_schedule']
    require(noise['motion_std_scale']==1. and noise['tail_std_scale']==0. and noise['effective_tail_seconds']==1.,'Unexpected search noise schedule')
    require(abs(float(noise['tail_start_execution_seconds'])-8.55)<1e-8,'Noise tail must start after original reference action')

    def resolve(value):
        path=Path(value)
        if path.is_file():return path.resolve()
        normalized=str(value).replace('\\','/')
        marker='/KinoGen/'
        if marker in normalized:
            path=root/normalized.split(marker,1)[1]
            if path.is_file():return path.resolve()
        return None

    cache={}
    def existing(path,expected=None):
        path=Path(path)
        if not path.is_file():return None
        key=str(path.resolve())
        if key not in cache:cache[key]={'path':key,'bytes':path.stat().st_size,'sha256':digest(path)}
        item=dict(cache[key])
        if expected is not None:
            require(item['sha256']==expected,'Artifact SHA256 mismatch: '+str(path));item['recorded_hash_verified']=True
        return item

    checkpoint_path=resolve(chosen['controller']['checkpoint'])
    require(checkpoint_path is not None,'Chosen residual checkpoint is missing')
    checkpoint_artifact=existing(checkpoint_path,checkpoint_hash)
    reference_path=resolve(chosen['inputs']['reference'])
    require(reference_path is not None,'Frozen task reference is missing')
    reference_artifact=existing(reference_path,chosen['inputs']['reference_sha256'])
    chosen_artifacts={}
    for name,item in chosen['artifacts'].items():
        path=resolve(item['path'])
        require(path is not None,'Chosen replay artifact missing: '+name)
        chosen_artifacts[name]=existing(path,item['sha256'])
    candidate_info=chosen['candidate_provenance'];candidate_path=resolve(candidate_info['candidate_path'])
    require(candidate_path is not None,'Original selected search control sequence is missing')
    selected_candidate=existing(candidate_path,candidate_info['candidate_sha256'])
    sourcepath=root/'data/two_sitting_people/rgb_174114_sim_handoff/validation.json'
    source=read(sourcepath);require(source.get('frames')==172,'Source frame count is not172')
    require(abs(float(chosen['source_duration_seconds'])-5.7)<1e-8,'Source timeline extent changed')
    training_manifest=base/'training_native_v1/manifest.json';training_receipt=base/'training_native_v1/training_receipt.json'
    manifest,receipt=read(training_manifest),read(training_receipt)
    require(manifest['args']['backend']=='native','Residual native-training backend provenance missing')
    require(manifest['args']['num_envs']==128 and manifest.get('actors')==2 and manifest.get('action_dim')==56,'Training articulation/action provenance differs')
    overview=existing(root/'previews/two_sitting_final_functional.mp4')
    egodir=root/'previews/two_sitting_final_ego';delivery=base/'delivery_final_candidate156'
    ego_files=[existing(path)for path in sorted(egodir.rglob('*'))if path.is_file()]if egodir.is_dir()else[]
    delivery_files=[existing(path)for path in sorted(delivery.rglob('*'))if path.is_file()]if delivery.is_dir()else[]
    hoi_files=[item for item in delivery_files if 'hoi'in Path(item['path']).name.lower()and Path(item['path']).suffix=='.npz']
    ego_videos=[item for item in ego_files if Path(item['path']).suffix=='.mp4']
    missing=[]
    if overview is None:missing.append('final_candidate156_overview_video')
    if not hoi_files:missing.append('final_candidate156_HOI_NPZ')
    if len(ego_videos)<2:missing.append('two_final_candidate156_ego_videos')
    observed_ids=[]
    for path in base.glob('physics_selected450_candidate*/report.json'):
        try:observed_ids.append(int(path.parent.name.removeprefix('physics_selected450_candidate')))
        except ValueError:pass
    independent_count=len(IDS);functional_count=sum(r['functional_physical_demo_pass']is True for r in reports.values())
    strict_count=sum(r['source_tracking_pass']is True for r in reports.values())
    statistics={'search_batch':'search_native450_settle_v2','total_attempts':total,'preliminary_selected_count':len(preliminary),
                'independently_checked_count':independent_count,'independently_checked_candidate_ids':list(IDS),
                'functional_passed_count':functional_count,'functional_confirmed_fraction_of_this_batch':functional_count/total,
                'functional_confirmed_percent_of_this_batch':100*functional_count/total,
                'strict_source_tracking_pass_count':strict_count,'strict_count_scope':'Only the five selected independently checked rollouts.',
                'other_rollouts_not_independently_checked_count':total-independent_count,
                'observed_selected_report_ids':sorted(observed_ids),'robustness_or_generalization_claim':False,
                'disclosure':'5/256 is the confirmed functional-demo yield of this selected batch, not an unbiased or robust policy success-rate estimate. The other251 attempts are not counted as independent acceptance passes or independent strict-tracking failures.'}
    metric_keys=('physics_steps','simulated_seconds','source_duration_seconds','execution_reference_duration_seconds',
                 'max_root_xy_error_m','max_root_z_error_m','max_root_orientation_error_rad','max_penetration_m',
                 'max_interperson_penetration_m','final_0p2s_seat_contact_fraction','final_root_linear_speed_max_m_s',
                 'final_root_angular_speed_max_rad_s','final_seat_vertical_force_mean_N','robot_weight_N',
                 'root_height_min_m','root_global_tilt_max_rad','exchanged_distance_m','final_root_xy_error_m','contact_records')
    final={'schema':'two_sitting_verified_final.v1','status':'functional_demo_verified_strict_source_tracking_not_passed',
           'audited_at_asia_shanghai':datetime.now(timezone(timedelta(hours=8))).isoformat(),
           'functional_physical_demo_pass':True,'strict_source_tracking_pass':False,'original_strict_success_flag':chosen['success'],
           'statistics':statistics,'chosen_candidate_id':CHOSEN,'chosen_quality':{key:chosen[key]for key in metric_keys},
           'chosen_independent_replay':chosen['independent_replay'],'chosen_candidate_policy_state_consistency':chosen['candidate_policy_state_consistency'],
           'physical_integrity':{key:chosen[key]for key in('state_override_after_reset_count','external_force_steps','root_actuators','mocap_bodies')},
           'source':{'frames':172,'last_pts_seconds':5.7,'validation':existing(sourcepath),'reference':reference_artifact},
           'fixed_residual_policy':{'checkpoint_iteration':450,'checkpoint':checkpoint_artifact,'one_fixed_checkpoint_for_all256_attempts':True,
                                    'search_seed':174115,'noise_schedule':noise,'policy_weights_updated_during_search':False},
           'training':{'manifest':existing(training_manifest),'receipt':existing(training_receipt),
                       'method':manifest['method'],'physics':manifest['physics'],'optimizer_updates_in_completed_native_run':receipt['optimizer_updates'],
                       'completed_run_final_checkpoint':receipt['checkpoint'],'completed_run_final_checkpoint_sha256':receipt['sha256'],
                       'completed_run_final_is_selected450':receipt['sha256']==checkpoint_hash,
                       'disclosure':'The completed native-training receipt describes the whole resumed run. Candidate156 uses the earlier fixed450 checkpoint, not the run final checkpoint.'},
           'independent_evidence':evidence,'search_summary':existing(searchpath),'selected_control_candidate':selected_candidate,
           'chosen_replay_artifacts':chosen_artifacts,
           'delivery_inventory':{'overview_video':overview,'ego_files':ego_files,'hoi_files':hoi_files,'delivery_files':delivery_files,
                                 'complete_at_audit':not missing,'missing_at_audit':missing,
                                 'scope':'Only explicit final candidate156 destinations; old native400/reference videos excluded.'},
           'strict_tracking_limitation':{'required_max_root_xy_error_m':.25,'observed_max_root_xy_error_m':chosen['max_root_xy_error_m'],
                                        'passed':False,'meaning':'Two complete functional seat exchanges are verified. Exact whole-sequence source trajectory replication remains unpassed.'},
           'limitations':chosen.get('limitations',[]),'jobs_launched_by_this_audit':[]}
    feishu=(f'双Vita起身、换位、坐下的真实物理demo已跑通。固定第450轮检查点，{total}次随机尝试初筛5条，5条经独立5ms重放均通过功能验收，本批确认产出占{100*functional_count/total:.6f}%；其余{total-independent_count}条未独立验收，不代表鲁棒成功率。选用156号，连续9.56秒、1912步，无根驱动、外力托举或姿态强写；控制重放及原候选状态对照误差均为0。末端座面承重约418/537N，最大穿透17.3mm。严格源轨迹复现未通过：两人全程根位置最大误差0.506/0.485m，超过0.25m门槛。')
    require(150<=len(feishu)<=300,'Feishu summary must remain150–300 characters')
    a.output.mkdir(parents=True)
    (a.output/'final_status.json').write_text(json.dumps(final,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (a.output/'FEISHU_FINAL.txt').write_text(feishu+'\n',encoding='utf-8')
    (a.output/'source_two_sitting_final_audit.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'status':final['status'],'functional_confirmed':f'{functional_count}/{total}','strict_pass_among_checked':strict_count,
                      'chosen_candidate':CHOSEN,'deliverables_complete_at_audit':not missing,'missing_at_audit':missing,'output':str(a.output)},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
