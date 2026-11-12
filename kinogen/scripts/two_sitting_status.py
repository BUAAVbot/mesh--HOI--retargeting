"""Summarize recorded two-Vita trials without running or claiming a new trial.

Only report/metadata JSON and existence of related files are inspected. The
candidate is the lowest worst-actor XY-error trial among full-duration runs
whose height/orientation, independent replay and no-force checks passed.
Candidate selection does not change physical acceptance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import time


def read_json(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        return json.load(f)


def finite(value):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def numbers(value):
    if isinstance(value, (list, tuple)):
        return [finite(x) for x in value]
    v = finite(value)
    return [] if v is None else [v]


def maximum(value):
    valid = [v for v in numbers(value) if v is not None]
    return max(valid) if valid else None


def receipt(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {'path': str(path), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def summarize(path):
    r = read_json(path)
    checks = r.get('checks', {})
    explicit_pass = r.get('simulation_passed', r.get('success')) is True
    failed = [key for key, value in checks.items() if value is not True]
    # An explicit successful flag cannot override contradictory quality checks.
    accepted = explicit_pass and bool(checks) and not failed
    eligibility = {
        'full_source_duration': checks.get('full_source_duration') is True,
        'completed_requested_duration': checks.get('completed_requested_duration') is True,
        'normal_termination': r.get('termination') in ('completed', 'requested_duration'),
        'root_height_check': checks.get('root_z_error_below_0p18m', checks.get('both_root_height_tracking')) is True,
        'root_orientation_check': checks.get('root_orientation_error_below_0p65rad', checks.get('both_root_orientation_tracking')) is True,
        'independent_control_only_replay': checks.get('independent_control_only_replay', checks.get('independent_replay')) is True,
        'no_state_overrides_or_external_forces': checks.get('no_state_overrides_or_external_forces') is True,
    }
    xy = numbers(r.get('max_root_xy_error_m', r.get('max_root_horizontal_error_m')))
    xy_max = maximum(xy)
    eligibility['finite_xy_metric'] = len(xy) == 2 and all(v is not None for v in xy)
    controller = r.get('controller') or {}
    actors = controller.get('actors', []) if isinstance(controller, dict) else []
    inference = [a.get('inference_count') for a in actors if isinstance(a, dict)]
    return {
        'trial': path.parent.name, 'report': receipt(path),
        'status': r.get('status'), 'termination': r.get('termination'),
        'explicit_success_flag': explicit_pass, 'simulation_passed': accepted,
        'success_flag_contradicts_checks': explicit_pass and not accepted,
        'simulated_seconds': finite(r.get('simulated_seconds', r.get('duration_s'))),
        'requested_seconds': finite(r.get('requested_seconds')),
        'reference_duration_seconds': finite(r.get('source_duration_seconds', r.get('source_duration_s'))),
        'actual_source_time_range_s': r.get('actual_source_time_range_s'),
        'actual_reference_time_range_s': r.get('actual_reference_time_range_s'),
        'physics_steps': r.get('physics_steps'),
        'max_root_xy_error_m': xy, 'worst_actor_root_xy_error_m': xy_max,
        'max_root_z_error_m': numbers(r.get('max_root_z_error_m')),
        'max_root_orientation_error_rad': numbers(r.get('max_root_orientation_error_rad')),
        'max_penetration_m': finite(r.get('max_penetration_m')),
        'max_interperson_penetration_m': finite(r.get('max_interperson_penetration_m')),
        'final_0p2s_seat_contact_fraction': numbers(r.get('final_0p2s_seat_contact_fraction')),
        'contact_records': r.get('contact_records'),
        'state_override_after_reset_count': r.get('state_override_after_reset_count'),
        'external_force_steps': r.get('external_force_steps'),
        'root_actuators': r.get('root_actuators'), 'mocap_bodies': r.get('mocap_bodies'),
        'controller_mode': controller.get('mode') if isinstance(controller, dict) else None,
        'controller': {key: controller[key] for key in (
            'mode', 'method', 'policy', 'policy_sha256', 'training_iteration',
            'privileged_root_state_observation', 'action_scale_rad') if key in controller} if isinstance(controller, dict) else {},
        'controller_inference_counts': inference,
        'checks': checks, 'failed_checks': failed,
        'independent_replay': r.get('independent_replay'),
        'candidate_eligibility': eligibility,
        'eligible_complete_stable_candidate': all(eligibility.values()),
        'inputs': r.get('inputs'),
        'artifacts': r.get('artifacts'),
        'limitations': r.get('limitations', []),
    }


def fmt_pair(value, scale=1., suffix='', digits=3):
    if not value:
        return '未记录'
    return '/'.join('未记录' if v is None else f'{v*scale:.{digits}f}' for v in value)+suffix


def fmt(value, scale=1., suffix='', digits=3):
    return '未记录' if value is None else f'{value*scale:.{digits}f}'+suffix


def training_status(directory, trials, errors):
    """Read optimization progress separately from reset-free physical acceptance."""
    manifest_path = directory/'manifest.json'
    receipt_path = directory/'training_receipt.json'
    metrics_path = directory/'metrics.jsonl'
    manifest, completion, latest = {}, {}, None
    invalid_lines = 0
    metric_rows = 0
    for path, target in ((manifest_path, manifest), (receipt_path, completion)):
        if path.is_file():
            try:
                target.update(read_json(path))
            except (OSError, ValueError, TypeError) as e:
                errors.append({'path': str(path), 'error': f'{type(e).__name__}: {e}'})
    if metrics_path.is_file():
        # A concurrently appended final line can be incomplete. Retain the last
        # complete record and count skipped records rather than invent progress.
        with metrics_path.open(encoding='utf-8-sig') as stream:
            for line in stream:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError('Metric row must be an object')
                    latest = row
                    metric_rows += 1
                except (ValueError, TypeError):
                    invalid_lines += 1
    args = manifest.get('args', {})
    iteration = None if latest is None else latest.get('iteration')
    age = max(0., time.time()-metrics_path.stat().st_mtime) if metrics_path.exists() else None
    updates_finished = completion.get('status') in (
        'ppo_updates_finished_pending_independent_physics_validation',
        'finished_pending_native_validation')
    matched = []
    for trial in trials:
        inputs = trial.get('inputs') or {}
        controller = trial.get('controller') or {}
        policy = controller.get('policy') or inputs.get('checkpoint')
        # The native report binds its specific checkpoint separately from the
        # training rollout statistics. Never infer physical success from reward.
        path_match = bool(policy) and Path(policy).parent.name == directory.name
        hash_match = bool(completion.get('sha256')) and controller.get('policy_sha256') == completion.get('sha256')
        if path_match or hash_match:
            matched.append({'trial': trial['trial'], 'simulation_passed': trial['simulation_passed'],
                            'policy': policy, 'policy_sha256': controller.get('policy_sha256'),
                            'training_iteration': controller.get('training_iteration'),
                            'failed_checks': trial['failed_checks']})
    native_passed = [t['trial'] for t in matched if t['simulation_passed']]
    if native_passed:
        state = 'native_physics_validation_passed_for_listed_checkpoints'
        state_cn = '列出的策略检查点已有独立原生物理验收通过记录'
    elif updates_finished:
        state = 'finished_pending_native_validation'
        state_cn = 'PPO更新完成，尚无通过的独立原生物理验收'
    elif latest is not None and age is not None and age < 300.:
        state = 'training_in_progress_recent_metrics'
        state_cn = '训练日志正在近期更新；尚无完成回执，未核对进程存活'
    elif latest is not None:
        state = 'no_completion_receipt_running_state_unconfirmed'
        state_cn = '已有训练进度但未见完成回执；日志较久未更新，运行状态待确认'
    elif manifest:
        state = 'initialized_no_metrics_yet'
        state_cn = '已写入训练配置，尚未发现完整训练指标'
    else:
        state = 'training_artifacts_incomplete'
        state_cn = '训练记录不完整，不能确认已开始更新'
    return {
        'training': directory.name, 'directory': str(directory.resolve()),
        'status': state, 'status_zh_CN': state_cn,
        'method': manifest.get('method'), 'physics': manifest.get('physics'),
        'actors': manifest.get('actors'), 'action_dim': manifest.get('action_dim'),
        'num_envs': args.get('num_envs'), 'requested_iterations_this_run': args.get('iterations'),
        'rollout_steps': args.get('rollout'), 'epochs': args.get('epochs'),
        'minibatches': args.get('minibatches'), 'save_every_iterations': args.get('save_every'),
        'sonic_base': args.get('sonic_base'), 'strict_training': args.get('strict_training'),
        'privileged_observations': manifest.get('privileged_observations'),
        'inputs': manifest.get('inputs'),
        'iteration': iteration, 'transitions': None if latest is None else latest.get('transitions'),
        'from_start_successes_during_training': None if latest is None else latest.get('from_start_successes'),
        'from_start_episodes_during_training': None if latest is None else latest.get('from_start_episodes'),
        'max_from_start_source_time_s_during_training': None if latest is None else latest.get('max_from_start_source_time_s'),
        'latest_training_metrics': latest, 'complete_metric_rows': metric_rows,
        'skipped_incomplete_or_invalid_metric_lines': invalid_lines,
        'metrics_last_modified_utc': datetime.fromtimestamp(metrics_path.stat().st_mtime, timezone.utc).isoformat() if metrics_path.exists() else None,
        'metrics_age_seconds_at_summary': age,
        'optimizer_updates_finished': updates_finished,
        'optimizer_updates': completion.get('optimizer_updates'),
        'completion_receipt': completion if receipt_path.is_file() else None,
        'manifest_evidence': receipt(manifest_path) if manifest_path.is_file() else None,
        'completion_receipt_evidence': receipt(receipt_path) if receipt_path.is_file() else None,
        'metrics_path': str(metrics_path.resolve()) if metrics_path.is_file() else None,
        'matched_native_physics_evaluations': matched,
        'independently_accepted_checkpoint_trials': native_passed,
        'training_metrics_are_validation_success': False,
        'interpretation': 'Iteration/transitions/reward/from_start_successes are optimization logs with episode resets; acceptance comes only from corresponding independent physics reports.',
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    root = a.root.resolve()
    pipeline = root if root.name == 'two_sitting_pipeline' else root/'outputs/two_sitting_pipeline'
    if not pipeline.is_dir():
        p.error('Missing pipeline directory: '+str(pipeline))
    trials, invalid = [], []
    for path in sorted(pipeline.glob('physics*/report.json')):
        try:
            trials.append(summarize(path))
        except (OSError, ValueError, TypeError, KeyError) as e:
            invalid.append({'path': str(path.resolve()), 'error': f'{type(e).__name__}: {e}'})
    trainings = [training_status(d, trials, invalid) for d in sorted(pipeline.glob('training*')) if d.is_dir()]
    candidates = [t for t in trials if t['eligible_complete_stable_candidate']]
    selected = min(candidates, key=lambda t: (t['worst_actor_root_xy_error_m'], t['trial'])) if candidates else None
    accepted = [t['trial'] for t in trials if t['simulation_passed']]
    assets = []
    for path in sorted(pipeline.glob('assets*/assets_report.json')):
        try:
            r = read_json(path)
            assets.append({'report': receipt(path), 'dynamic_free_roots': r.get('dynamic_free_roots'),
                           'dynamic_objects': r.get('dynamic_objects'), 'furniture_fixed': r.get('furniture_fixed'),
                           'same_robot_self_collision': r.get('same_robot_self_collision'),
                           'inter_robot_collision': r.get('inter_robot_collision'),
                           'robot_mass_inertia': r.get('robot_mass_inertia'),
                           'explicit_ankle_pitch_bounds_rad': r.get('explicit_ankle_pitch_bounds_rad'),
                           'support_geometry': r.get('support_geometry'), 'outputs': r.get('outputs')})
        except (OSError, ValueError, TypeError) as e:
            invalid.append({'path': str(path), 'error': f'{type(e).__name__}: {e}'})
    references = []
    for path in sorted(pipeline.glob('reference*.json')):
        try:
            r = read_json(path)
            if not path.with_suffix('.npz').is_file():
                continue
            references.append({'report': receipt(path), 'frames': r.get('frames'), 'actors': r.get('actors'),
                               'source_duration_s': r.get('source_duration_s'),
                               'execution_duration_s': r.get('execution_duration_s'), 'time_scale': r.get('time_scale'),
                               'explicit_repair': r.get('explicit_repair'),
                               'max_static_penetration_m': r.get('max_static_penetration_m'),
                               'median_body_rms_m': r.get('median_body_rms_m'),
                               'source_hand_observed': r.get('source_hand_observed'),
                               'seat_xy_adaptation_enabled': r.get('seat_xy_adaptation_enabled'),
                               'continuous_foot_support_inference_enabled': r.get('continuous_foot_support_inference_enabled')})
        except (OSError, ValueError, TypeError) as e:
            invalid.append({'path': str(path), 'error': f'{type(e).__name__}: {e}'})
    deliveries = []
    for path in sorted(pipeline.glob('delivery*/hoi_motion_physics.metadata.json')):
        try:
            r = read_json(path)
            dataset = path.parent/'hoi_motion_physics.npz'
            deliveries.append({'metadata': receipt(path), 'dataset_path': str(dataset.resolve()),
                               'dataset_exists': dataset.is_file(), 'dataset_bytes': dataset.stat().st_size if dataset.is_file() else None,
                               'status': r.get('status'), 'simulation_passed': r.get('simulation_passed') is True,
                               'state_frames': r.get('state_frames'), 'physics_intervals': r.get('physics_intervals'),
                               'contact_points': r.get('contact_points'), 'physics_duration_s': r.get('physics_duration_s'),
                               'input_trace': r.get('input_trace'), 'input_reference': r.get('input_reference')})
        except (OSError, ValueError, TypeError) as e:
            invalid.append({'path': str(path), 'error': f'{type(e).__name__}: {e}'})
    have_ref = any(r['actors'] == 2 and r['frames'] for r in references)
    have_assets = any(r['dynamic_free_roots'] == 2 for r in assets)
    have_export = any(d['dataset_exists'] for d in deliveries)
    modules = [
        {'module': '接触感知运动重定向', 'state': '已生成双Vita完整运动学参考；接触为推断、动力学验收另计' if have_ref else '未找到完整双Vita参考证据',
         'complete_kinematic_reference': bool(have_ref), 'evidence': [r['report']['path'] for r in references]},
        {'module': '仿真资产构建与物性标定', 'state': '双Vita、地面及两凳碰撞体已构建；机器人惯性继承模型，场景物性为假定值' if have_assets else '未找到双Vita动力学资产报告',
         'engineering_assets_built': have_assets, 'measured_system_identification_complete': False,
         'evidence': [r['report']['path'] for r in assets]},
        {'module': '物理运动模仿与交互重仿真', 'state': '已有完整任务通过记录' if accepted else '已实际执行；完整换座并坐稳的任务质量尚未通过' if trials else '未找到物理执行报告',
         'executed_trials': len(trials), 'accepted_trials': accepted},
        {'module': '物理交互标注与质量检验', 'state': '接触图、仿真接触力及关节力矩已有诊断导出；质量标签随试验保留' if have_export else '已有物理日志，尚未找到统一HOI导出' if trials else '未找到导出证据',
         'diagnostic_export_available': have_export, 'evidence': [d['metadata']['path'] for d in deliveries]},
        {'module': 'HOI motion + physics最终产物', 'state': '存在通过试验，发布仍应绑定对应冻结模型/参考/验收' if accepted else '可交付诊断视频及数据；尚不能标为完整成功样本',
         'full_task_accepted': bool(accepted)},
    ]
    missing = [
        '若尚无通过试验：仍须修复根轨迹跟踪、机器人间碰撞及最终凳面支撑，以实际验收报告为准。',
        '现场实测尺度/重力、摩擦和接触刚度等参数未标定；现有力和力矩是基于模型假定的仿真输出。',
        '凳子当前固定在世界中；做此任务不要求凳子动态质量/惯量，若以后让凳子移动才需增加这些参数。',
        '机器人质量惯量继承已有模型，不能等同于本次场景实测系统辨识。',
        '原包手指为默认姿态，无独立MANO结果；本任务以起身、行走和坐下为主，MANO不是跑通的必需依赖。',
    ]
    now = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
    result = {'schema': 'two_sitting_status.v1', 'generated_at_asia_shanghai': now,
              'pipeline_root': str(pipeline), 'trial_count': len(trials), 'accepted_trials': accepted,
              'simulation_passed': bool(accepted), 'selected_candidate': selected,
              'candidate_selection_rule': 'From full-duration normal completion with root height/orientation, independent replay and no-force checks true, minimize max(actor1_xy_error, actor2_xy_error). This is ranking only, not success.',
              'candidate_count': len(candidates), 'trials': trials, 'trainings': trainings, 'references': references,
              'assets': assets, 'deliveries': deliveries, 'diagram_modules': modules,
              'missing_or_unvalidated': missing, 'unreadable_reports': invalid,
              'read_only_workflow': 'No model execution, rollout, training or acceptance flag modification performed.'}
    lines = [f'双人换座 → 双Vita 仿真进展（{now[:16].replace("T", " ")}）', '',
             '结论：'+('已有完整任务通过的冻结试验记录。' if accepted else '资产、双机重定向与真实动力学执行已接通；完整换座并坐稳尚未通过质量验收。'),
             f'已记录 {len(trials)} 轮物理试验；通过 {len(accepted)} 轮。']
    if trainings:
        lines += ['', '真实PPO训练：']
        for train in trainings:
            lines += [f'- {train["training"]}：{train["status_zh_CN"]}。',
                      f'  并行环境 {train["num_envs"]}，双机器人合计策略输出 {train["action_dim"]} 维；已记录 iteration={train["iteration"]}，transitions={train["transitions"]}。',
                      f'  训练中从起点完成计数/起点回合数={train["from_start_successes_during_training"]}/{train["from_start_episodes_during_training"]}；该计数含训练回合重置，不是独立仿真成功率。']
            if train['optimizer_updates_finished']:
                lines += [f'  已有更新完成回执，optimizer_updates={train["optimizer_updates"]}；仍以对应原生MuJoCo无中途重置验收为准。']
            if train['matched_native_physics_evaluations']:
                lines += ['  对应原生评估：'+'；'.join(t['trial']+('通过' if t['simulation_passed'] else '未通过') for t in train['matched_native_physics_evaluations'])+'。']
        lines += ['训练的iteration、transitions、reward与from_start_successes均只代表训练进度；不得替代完整换座、坐稳和独立控制重放验收。']
    if selected:
        lines += ['', f'当前候选：{selected["trial"]}（完整执行且高度/姿态检查通过；候选不等于任务成功）。',
                  f'实际执行 {fmt(selected["simulated_seconds"], suffix="秒")}，{selected["physics_steps"]} 个物理步。',
                  '两机最大根XY误差：'+fmt_pair(selected['max_root_xy_error_m'], suffix='米')+'。',
                  '最大穿透：'+fmt(selected['max_penetration_m'], 1000., '毫米')+'；双机间最大穿透：'+fmt(selected['max_interperson_penetration_m'], 1000., '毫米')+'。',
                  '末0.2秒两机真实凳面接触占比：'+fmt_pair(selected['final_0p2s_seat_contact_fraction'], 100., '%', 1)+'。',
                  '该候选验收：'+('通过。' if selected['simulation_passed'] else '未通过。未通过项：'+ '、'.join(selected['failed_checks'])+'。')]
    else:
        lines += ['', '当前没有同时满足完整时长、高度/姿态、独立重放及无外力检查的候选；不选择部分片段冒充完整成功。']
    lines += ['', '对照流程图：']
    lines += [f'{i+1}. {m["module"]}：{m["state"]}。' for i, m in enumerate(modules)]
    lines += ['', '仍缺/仍需验收：']+[f'- {item}' for item in missing]
    if accepted:
        lines += ['', '通过试验：'+'、'.join(accepted)+'。']
    if deliveries:
        lines += ['', '现有统一HOI导出：']+[f'- {Path(d["metadata"]["path"]).parent.name}：{d["physics_intervals"]}物理步，{d["contact_points"]}接触点，验收'+('通过' if d['simulation_passed'] else '未通过/诊断')+'。' for d in deliveries]
    lines += ['', '全部试验：']
    for t in trials:
        lines.append(f'- {t["trial"]}：{fmt(t["simulated_seconds"], suffix="秒")}；根XY最大误差{fmt_pair(t["max_root_xy_error_m"], suffix="米")}；穿透{fmt(t["max_penetration_m"], 1000., "毫米")}；'+('通过' if t['simulation_passed'] else '未通过')+f'，终止原因={t["termination"]}。')
    lines += ['', '说明：这是基于各试验实际报告的汇总；运动学参考、物理稳定和完整任务通过分别记录。所有接触力/力矩均为仿真值。']
    a.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = a.output_dir/'two_sitting_status.json'
    txt_path = a.output_dir/'FEISHU_TWO_SITTING_STATUS.zh-CN.txt'
    if json_path.exists() or txt_path.exists():
        raise FileExistsError('Use a new status output directory; existing reports are preserved.')
    json_path.write_text(json.dumps(clean(result), indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    txt_path.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'status_json': str(json_path.resolve()), 'feishu_text': str(txt_path.resolve()),
                      'selected_candidate': None if selected is None else selected['trial'],
                      'selected_candidate_passed': False if selected is None else selected['simulation_passed'],
                      'accepted_trials': accepted, 'trial_count': len(trials)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
