"""Build a reproducible progress delivery; never select/promote an RL checkpoint."""
from __future__ import annotations
import argparse,datetime,hashlib,html,json,re,shutil,zipfile
from pathlib import Path

def read(p,default=None):
    try:return json.loads(Path(p).read_text(encoding='utf-8'))
    except(OSError,ValueError):return default
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/data/KinoGen/outputs/sonic_expert_20261028'))
    p.add_argument('--output',type=Path,default=Path('/data/KinoGen/deliveries/sonic_expert_progress_20261029'))
    p.add_argument('--include-model',action='store_true');p.add_argument('--zip',action='store_true')
    a=p.parse_args();root=a.root.resolve();out=a.output.resolve()
    final=read(root/'expert_final_selection.json')
    prototype_final=bool(final and final.get('status')=='prototype_nominal_pass_no_robustness_gain')
    if out==root or root in out.parents:raise ValueError('Delivery must be outside experiment root to avoid duplicate evaluation discovery')
    out.mkdir(parents=True,exist_ok=True);source_files={};missing=[]
    def include(path,mandatory=True,destination=None):
        path=Path(path)
        if not path.is_file():
            if mandatory:missing.append(str(path))
            return
        target=destination or path.relative_to(root)
        source_files[str(target).replace('\\','/')]=path
    for path in(root/'report').rglob('*'):
        if path.is_file():include(path)
    for n in ['sonic_comparison.json','benchmark_protocol.json','terminal_refined_full_v4/report.json',
        'terminal_refined_full_v4/trace.npz','terminal_refined_full_v4/initial_state.npz','terminal_motor_search_v4/candidate.npz',
        'references/phc.npz','references/phc_time_scale1.5.npz','references/phc_transfer_time1p5.npz',
        'heldout_seed990013/initial_states.npz','heldout_seed990013/initial_states.receipt.json',
        'heldout_seed990013/old_vs_cem_zero_feedback_pair_audit.json','audit/video_delivery_quality.json',
        'cem_v4_physical_state_pool/full_integration_export_receipt.json']:
        include(root/n)
    for name in ['sonic_all_methods','terminal_refined_full_v4','expert_control_v1']:
        for suffix in ['.mp4','.manifest.json','.render.json','_contactsheet.jpg','_final.jpg']:
            include(root/'media'/(name+suffix),mandatory=suffix=='.mp4')
    for row in read(root/'sonic_comparison.json',{})['results']:
        for suffix in ['.mp4','.render.json','_contactsheet.jpg']:include(root/'media'/(row['method']+suffix),mandatory=False)
    # Preserve the relative links in the full report, including fixed32 core traces.
    report_text=(root/'report/REPORT.zh-CN.md').read_text(encoding='utf-8')
    for target in re.findall(r'\]\(([^)]+)\)',report_text):
        if target.startswith(('http:','https:','#')):continue
        path=(root/'report'/target).resolve()
        if path==root or root in path.parents:include(path)
        else:raise ValueError('Report link escapes source experiment: '+target)
    for path in(root/'terminal_refined_full_v4/source_snapshot').glob('*.py'):include(path)
    for path in(root/'audit').rglob('*.json'):
        if path.name!='final_prototype_delivery_quality.json':include(path,mandatory=False)
    for path in(root/'audit/feedback_local_gain_checkpoint50/source_snapshot').glob('*.py'):include(path)
    for folder in ['metrics_corrections','audit/quaternion_metric_source_before_fix','audit/physical_prior_training_objective']:
        for path in(root/folder).rglob('*'):
            if path.is_file():include(path)
    for folder in ['closed_loop_gain_cem','closed_loop_root_gain_cem']:
        for path in(root/folder).rglob('*'):
            if path.is_file():include(path)
    scripts=Path('/data/KinoGen/scripts')
    for name in ['rgb_near_physics.py','two_sitting_physics.py','two_sitting_ppo_env.py','two_sitting_native_ppo_env.py',
        'two_sitting_eval_ppo.py','two_sitting_replay_candidate.py','two_sitting_batch_sonic.py','fourth_sonic_controller.py',
        'build_sonic_expert_report.py','build_sonic_expert_delivery.py','prepare_heldout_initial_states.py',
        'validate_sonic_expert_delivery.py',
        'benchmark_heldout_feedforward.py','audit_prior_state_pool.py','export_prior_task_seeds.py',
        'train_feedback_gain_expert.py','train_root_feedback_gain_expert.py','kinematic_root_feedback_basis.py',
        'export_cem_v4_physical_pool.py','export_cem_full_integration_pool.py','sonic_refined_reference_transfer.py',
        'audit_refined_reference_transfer.py','audit_feedback_local_gain.py','plot_feedback_local_gain.py']:
        include(scripts/name,destination=Path('code/current_support')/name)
    for path in(root/'terminal_refined_full_v4/source_snapshot').glob('*.py'):
        include(path,destination=Path('code/frozen_replay')/path.name)
    include(scripts/'rgb_near_physics.py',destination=Path('code/frozen_replay/rgb_near_physics.py'))
    for name in ['two_sitting_replay_candidate.py','two_sitting_physics.py','two_sitting_ppo_env.py','two_sitting_eval_ppo.py','rgb_near_physics.py']:
        include(scripts/name,destination=Path('code/corrected_replay')/name)
    include(scripts/'correct_quaternion_metric_overlays.py',destination=Path('code/current_support/correct_quaternion_metric_overlays.py'))
    model=root/'calibration/two_vita_expert.mjb';model_hash=read(root/'calibration/report.json',{})['expert_sha256']
    if a.include_model:include(model,destination=Path('models/two_vita_expert.mjb'))
    # Parent-authored final record fixes the checkpoint before heldout evaluation.
    # Copy declared weights/source/prior/input contracts, never choose a checkpoint here.
    selected_artifacts=[]
    for entry in(final or{}).get('delivery_artifacts',[]):
        entry={'path':entry,'role':'declared_artifact'}if isinstance(entry,str)else entry
        source=Path(entry['path']);role=entry.get('role','declared_artifact')
        if not source.exists():missing.append(str(source));continue
        candidates=[source]if source.is_file()else sorted(x for x in source.rglob('*')if x.is_file()and'__pycache__'not in x.parts)
        for path in candidates:
            if path.suffix=='.mjb'and not a.include_model and'model'not in role:continue
            if root in path.parents:target=path.relative_to(root)
            else:target=Path('code/final')/role/(path.relative_to(source.parent)if source.is_file()else Path(source.name)/path.relative_to(source))
            include(path,destination=target);selected_artifacts.append({'role':role,'source':str(path),'delivery_path':str(target).replace('\\','/')})
    if prototype_final:
        suffixes={Path(x['delivery_path']).suffix for x in selected_artifacts}
        for label,options in [('PPO weights',{'.pt','.pth'}),('source code',{'.py'}),('prior/export data',{'.npz'}),
                              ('evaluation/input contract',{'.json'}),('prototype video',{'.mp4'})]:
            if not suffixes.intersection(options):missing.append('Final selection delivery_artifacts missing actual '+label)
    for target,source in source_files.items():
        dest=out/target;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
    if prototype_final:
        expected=final.get('selected_checkpoint_sha256');checkpoint_copy=out/Path(final['selected_checkpoint']).relative_to(root)
        if not checkpoint_copy.exists()or(expected and sha(checkpoint_copy)!=expected):missing.append('Selected PPO checkpoint missing or SHA mismatch')
    baseline=read(root/'sonic_comparison.json');cem=read(root/'terminal_refined_full_v4/report.json')
    actor=baseline['results'][0]['report']['controller']['base']['actors'][0]['original_actor']
    held=read(root/'heldout_seed990013/cem_v4_feedforward_baseline/summary.json')
    if final:
        shutil.copy2(root/'expert_final_selection.json',out/'expert_final_selection.json')
    resources=[{'role':'physical_model','source':str(model),'sha256':model_hash,'bytes':model.stat().st_size,
        'included':a.include_model,'delivery_path':'models/two_vita_expert.mjb'if a.include_model else None,
        'needed_for':'Accepted motor-plan CPU replay; frozen binary includes model geometry'},
        {'role':'original_sonic_weights','source':actor['policy_path'],'sha256':actor['policy_sha256'],'included':False,
        'needed_for':'SONIC inference only; not required to replay accepted recorded motor controls'},
        {'role':'physical_runtime','source':'/root/vita02a-mjlab/.venv310/bin/python','included':False,
        'versions':'Python3.10; MuJoCo3.5.0; Torch2.7.1; NumPy','needed_for':'Existing evaluator/replayer imports'},
        {'role':'original_sonic_fork','source':'/data/yunhao.xia/GR00T-WholeBodyControl','included':False,
        'source_hashes':actor.get('source_code_sha256',{}),'needed_for':'Complete SONIC inference replication'},
        {'role':'single_actor_proxy','source':'/data/KinoGen/outputs/two_sitting_pipeline/assets_deploy_v3/vita_single_proxy.xml',
        'included':False,'needed_for':'Frozen SONIC inference; proxy XML may reference external assets'}]
    gaps=[{'item':'Selected newly trained PPO expert','status':'pending; this builder does not select checkpoints'},
        {'item':'Selected expert full200Hz functional acceptance and independent replay','status':'pending unless supplied and reviewed in expert_final_selection.json'},
        {'item':'Same32 heldout paired expert vs CEM zero-feedback comparison','status':'CEM baseline1/32 ready; final expert comparison pending'},
        {'item':'New PPO expert positive dataset/export receipt','status':'pending valid expert acceptance; CEM positive prior exists but is not PPO success'},
        {'item':'Newly accepted PPO expert video','status':'pending; current successful video is CEM motor plan'},
        {'item':'Distillation into SONIC/generalist','status':'not performed; reference-command transfer A/B failed'},
        {'item':'Portable simulation model/runtime/SONIC dependencies','status':'see external_resources; review bundle alone is not a clean-machine SONIC install'}]
    completed_items=[]
    if prototype_final:
        completed_items=['Predeclared physical200 PPO weights and source','Nominal full200Hz functional acceptance and independent control replay',
            'Same32 heldout evaluation against CEM zero-feedback baseline','Prototype video','Training trajectory/action export','Physical prior and input contract']
        gaps=[{'item':'Heldout robustness gain','status':'not achieved; no new passed cases beyond inherited nominalcase0'},
            {'item':'Strict source tracking','status':'not passed; corrected orientation and other source metrics retained'},
            {'item':'SONIC/generalist distillation','status':'not performed'},
            {'item':'Standalone runtime and original SONIC package','status':'external resources are declared; this is a prototype delivery'}]
    replay_model='"$DELIVERY/models/two_vita_expert.mjb"'if a.include_model else str(model)
    command='''#!/usr/bin/env bash
set -euo pipefail
DELIVERY="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${KINOGEN_PYTHON:-/root/vita02a-mjlab/.venv310/bin/python}"
: "${RUN_OUTPUT:?Set RUN_OUTPUT to a new, nonexistent output directory}"
export PYTHONPATH="$DELIVERY/code/corrected_replay${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" "$DELIVERY/code/corrected_replay/two_sitting_replay_candidate.py" \\
 --model MODEL_PLACEHOLDER \\
 --candidate "$DELIVERY/terminal_motor_search_v4/candidate.npz" \\
 --reference "$DELIVERY/references/phc_transfer_time1p5.npz" \\
 --output "$RUN_OUTPUT"
'''.replace('MODEL_PLACEHOLDER',replay_model)
    (out/'commands').mkdir(exist_ok=True);(out/'commands/replay_accepted_cem_nominal.sh').write_text(command,encoding='utf-8')
    (out/'external_resources.json').write_text(json.dumps(resources,ensure_ascii=False,indent=2)+'\n')
    readme='''# 双Vita SONIC筛选与专项反馈专家：阶段交付

这是可审阅的阶段成果，尚不宣称新PPO专家训练成功。

- SONIC总览：10组冻结策略配置，功能0/10，严格0/10；PHC为相对较优专家参考。
- 可落座视频：历史实际walking控制＋CPU CEM末段腿部电机计划，当前PHC功能通过、严格跟踪未过，**不是PPO专家**。
- 同32初态CEM零反馈基准：功能1/32；其余31个微扰早期失稳，未证明鲁棒控制。
- 物理实际q参考交给原SONIC的A/B都失败；未进行策略蒸馏。

先打开 `index.html` 看视频，再读 `report/REPORT.zh-CN.md` 和 `delivery_manifest.json`。

已接受物理示范复放：为 `RUN_OUTPUT` 设置全新目录，然后在服务器执行 `bash commands/replay_accepted_cem_nominal.sh`。
它只复放保存的104路电机命令，不运行SONIC或PPO策略；原模型、初态、控制及参考SHA保持锁定。命令未在打包时执行。
复放命令使用修正后最短弧姿态插值；原接受报告与原源码另外保留。25条指标修正overlay未改变strict/functional成功结论，所有朝向解释以overlay为准。
默认交付不复制347MB模型和SONIC权重；缺项与准确来源见 `external_resources.json`。构建时加 `--include-model` 可包含独立物理复放模型。

待补齐：最终PPO选择、全量物理验收、同32初态配对收益、已验收专家数据及视频；SONIC蒸馏属于后续阶段。
'''
    (out/'README.md').write_text(readme,encoding='utf-8')
    if prototype_final:
        readme=readme.replace('# 双Vita SONIC筛选与专项反馈专家：阶段交付','# 双Vita SONIC筛选与PPO原型交付')
        readme=readme.replace('这是可审阅的阶段成果，尚不宣称新PPO专家训练成功。','这是预先指定physical200的PPO原型交付：已完成名义full200Hz与独立重放、固定32初态评估；不宣称新增鲁棒能力。')
        readme=readme.replace('待补齐：最终PPO选择、全量物理验收、同32初态配对收益、已验收专家数据及视频；SONIC蒸馏属于后续阶段。','已完成：PPO200权重、源码、先验、输入合同、名义全量和32评估、原型视频及训练导出。未达标：32初态无新增通过、严格源跟踪未过；尚未做SONIC蒸馏。')
        (out/'README.md').write_text(readme,encoding='utf-8')
    doc='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>双Vita阶段交付</title><style>body{max-width:1200px;margin:30px auto;padding:0 20px;font:16px/1.7 system-ui;color:#183044}video,img{width:100%}.note{background:#fff0d7;padding:15px}h2{margin-top:35px}a{color:#176a9a}</style><h1>双Vita：SONIC筛选与专项反馈专家</h1><p class="note">阶段交付：SONIC功能0/10。以下落座正例是CEM电机计划，尚非新PPO专家成功；同32初态零反馈仅1/32。严格源跟踪未过。</p><p><a href="report/report.html">完整中文报告</a> · <a href="report/REPORT.zh-CN.md">报告文件</a> · <a href="delivery_manifest.json">交付与缺项清单</a> · <a href="external_resources.json">模型、权重及环境来源</a></p><h2>冻结SONIC：全部10组物理执行</h2><video controls preload="metadata" src="media/sonic_all_methods.mp4"></video><h2>CEM名义计划：双人换位落座，功能通过、严格未过</h2><p>历史真实walking前缀＋CPU末段电机搜索；这条视频不代表PPO专家或鲁棒成功率。</p><video controls preload="metadata" src="media/terminal_refined_full_v4.mp4"></video><h2>原始训练专家：失败对照</h2><video controls preload="metadata" src="media/expert_control_v1.mp4"></video><h2>冻结SONIC严格与功能指标</h2><img src="report/baseline_metrics.png"><h2>复放与后续</h2><p>复放命令：<code>RUN_OUTPUT=/新的输出目录 bash commands/replay_accepted_cem_nominal.sh</code>。先检查外部模型及运行环境。新专家的最终验收、同32初态对照和SONIC蒸馏仍需补齐。</p></html>'''
    (out/'index.html').write_text(doc,encoding='utf-8')
    if prototype_final:
        doc=doc.replace('双Vita：SONIC筛选与专项反馈专家','双Vita：SONIC筛选与PPO原型交付')
        doc=doc.replace('阶段交付：SONIC功能0/10。以下落座正例是CEM电机计划，尚非新PPO专家成功；同32初态零反馈仅1/32。严格源跟踪未过。','原型交付：PPO physical200已完成名义full200Hz、独立重放与固定32初态评估；32初态仍仅继承case0，相对CEM零反馈无新增通过，严格源跟踪未过。')
        doc=doc.replace('新专家的最终验收、同32初态对照和SONIC蒸馏仍需补齐。','PPO权重、源码、先验、输入合同、全量／32评估、视频及训练导出已完成。未达标的是32初态无新增通过、严格源跟踪未过；SONIC蒸馏尚未做。')
        videos=[x for x in selected_artifacts if x['role']in ['video','prototype_video','final_video']or x['delivery_path'].endswith('.mp4')]
        if videos:doc=doc.replace('<h2>复放与后续</h2>','<h2>预先指定PPO physical200原型</h2><p>名义功能继承同物理先验；32初态无新增通过。</p><video controls preload="metadata" src="'+html.escape(videos[0]['delivery_path'])+'"></video><h2>复放与后续</h2>')
        (out/'index.html').write_text(doc,encoding='utf-8')
    manifest={'schema':'sonic_expert.progress_delivery.v1','status':final.get('status')if prototype_final else'reviewable_progress_not_selected_ppo_expert',
        'generated_at':datetime.datetime.now().astimezone().isoformat(),'source_experiment':str(root),'output':str(out),
        'baseline':{'configurations':10,'functional':0,'strict':0,'selected_reference':baseline['selected_method']},
        'accepted_physical_demo':{'kind':'historical_motor_FF_plus_CPU_CEM_not_PPO','functional':cem['functional_task_success'],
            'strict':cem['success'],'independent_replay':cem['independent_replay'],'candidate_sha256':sha(root/'terminal_motor_search_v4/candidate.npz')},
        'heldout_cem_zero_feedback':held['references'],'final_selection_record_present':bool(final),
        'orientation_metric_authority':'metrics_corrections/quaternion_shortest_arc/correction_summary.json; original report values preserved as raw evidence, not final orientation conclusions',
        'quaternion_correction_summary':read(root/'metrics_corrections/quaternion_shortest_arc/correction_summary.json'),
        'builder_did_not_select_checkpoint':True,'external_resources':resources,'remaining_delivery_items':gaps,'missing_required_inputs':missing,
        'completed_prototype_items':completed_items,'declared_selected_artifacts_copied':selected_artifacts,'final_selection_record':final,
        'builder_source_sha256':sha(__file__),'files':[]}
    for path in sorted(out.rglob('*')):
        if path.is_file()and path.name not in ['delivery_manifest.json','delivery_validation.json']:
            manifest['files'].append({'path':path.relative_to(out).as_posix(),'bytes':path.stat().st_size,'sha256':sha(path)})
    (out/'delivery_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    broken=[];checked=0
    for path in [out/'report/REPORT.zh-CN.md',out/'report/report.html',out/'index.html']:
        text=path.read_text(encoding='utf-8');targets=re.findall(r'\]\(([^)]+)\)',text)+re.findall(r'(?:src|href)="([^\"]+)"',text)
        for target in targets:
            if target.startswith(('http:','https:','#')):continue
            checked+=1
            if not(path.parent/target).resolve().exists():broken.append({'source':str(path.relative_to(out)),'target':target})
    validation={'status':'passed'if not broken and not missing else'incomplete','report_links_checked':checked,'broken_links':broken,
        'missing_required_inputs':missing,'artifact_checks_do_not_establish_new_rl_gain':True,'artifact_files':len(manifest['files'])}
    (out/'delivery_validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
    if a.zip:
        archive=out.with_suffix('.zip')
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=3)as z:
            for path in sorted(out.rglob('*')):
                if path.is_file():z.write(path,path.relative_to(out))
        (archive.with_suffix('.zip.receipt.json')).write_text(json.dumps({'path':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive)},indent=2)+'\n')
    print(json.dumps({'output':str(out),**validation},ensure_ascii=False))
    if missing or broken:raise RuntimeError('Delivery inputs or links incomplete; inspectdelivery_validation.json')
if __name__=='__main__':main()
