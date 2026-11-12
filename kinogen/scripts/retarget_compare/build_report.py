"""Build traceable figures and a Chinese report from the common evaluation."""
from __future__ import annotations
import argparse
import html
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/data/KinoGen/outputs/retarget_compare_20261028'))
    args = parser.parse_args()
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    try:
        from retarget_compare_evaluate import METHODS
    except ModuleNotFoundError:
        from evaluate import METHODS

    data = json.loads((args.root / 'comparison_metrics.json').read_text(encoding='utf-8'))
    rows = data['results']
    primary = [row for row in rows if not row['method'].startswith('umr_')]
    variants = [row for row in rows if row not in primary]
    timing = {
        'gmr': '无训练；求解 7.10 s',
        'custom_ik': '无训练；求解 5.78 s',
        'custom_ik_contact': '无训练；求解 7.69 s',
        'umr': '构建 19.31 s＋训练 32.20 s＋双人重定向 173.52 s；合计 225.02 s',
        'omniretarget': '无训练；求解 66.89 s',
        'hoi_retarget': '无训练；窗口求解 41.14 s（不含 Stage1 GMR 7.10 s）',
        'phc': '无策略训练；逐片拟合 43.74 s（每人 500 次迭代）',
    }
    report_dir = args.root / 'report'
    report_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.facecolor': 'white'})
    labels = [row['label'].replace(' / Vita adapter', '').replace('Custom', 'Our') for row in rows]
    colors = ['#ce7440' if row['method'] in ('custom_ik_contact', 'umr_contact_adapted') else
              '#7a71aa' if row['method'] == 'umr_refined' else '#4c88ae' for row in rows]
    specs = [('mean_bone_direction_error_deg', 'Bone direction error (degrees)', 1),
             ('root_xy_rmse_m', 'Root XY RMSE (cm)', 100),
             ('max_penetration_m', 'Max enabled-pair penetration (mm)', 1000),
             ('seat_endpoint_absolute_distance_mean_m', 'Endpoint hip-seat absolute distance (mm)', 1000),
             ('stance_foot_speed_mean_m_s', 'Ankle XY speed in inferred stance (cm/s)', 100),
             ('body_joint_acceleration_rms_rad_s2', 'Body joint acceleration RMS (rad/s2)', 1)]
    fig, axes = plt.subplots(2, 3, figsize=(19, max(10., len(rows) * 1.25)))
    for axis, (key, title, scale) in zip(axes.flat, specs):
        values = [row['metrics'][key] * scale for row in rows]
        axis.barh(np.arange(len(rows)), values, color=colors)
        axis.set_yticks(np.arange(len(rows)), labels, fontsize=8)
        axis.invert_yaxis()
        axis.set_xlabel(title, fontsize=10)
        axis.grid(axis='x', alpha=.2)
        for index, value in enumerate(values):
            axis.text(value + max(values) * .015, index, f'{value:.2f}', va='center', fontsize=8)
        axis.set_xlim(0, max(values) * 1.24 if max(values) else 1)
    fig.suptitle('Two-person retargeting: common172frames / Vita52 / unchanged scene\nLower is better; local adapters, not original-paper benchmark scores', fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, .95])
    fig.savefig(report_dir / 'metric_comparison.png', dpi=180)
    plt.close(fig)

    if all('peak_body_joint_interval_speed_rad_s' in row['metrics'] for row in rows):
        fig, axis = plt.subplots(figsize=(12, max(5., len(rows) * .55)))
        values = [row['metrics']['peak_body_joint_interval_speed_rad_s'] for row in rows]
        axis.barh(np.arange(len(rows)), values, color=colors)
        axis.set_yticks(np.arange(len(rows)), labels)
        axis.invert_yaxis()
        axis.set_xlabel('Peak adjacent-frame body-joint rate (rad/s)')
        axis.set_title('Actual interval rate: reveals jumps hidden by central averaging\nNot an assumed hardware velocity limit')
        axis.grid(axis='x', alpha=.2)
        for i, value in enumerate(values):
            axis.text(value + .2, i, f'{value:.2f}', va='center')
        axis.set_xlim(0, max(values) * 1.13)
        fig.tight_layout()
        fig.savefig(report_dir / 'peak_interval_rates.png', dpi=180)
        plt.close(fig)

    protocol = json.loads((args.root / 'shared_input/source_manifest.json').read_text(encoding='utf-8'))
    with np.load(args.root / 'shared_input/common_observations.npz', allow_pickle=False) as source_data:
        source = source_data['repaired_source_joints_zup'].copy()
    support_path = Path(protocol['support_geometry'])
    if not support_path.exists() and '/outputs/' in protocol['support_geometry']:
        support_path = args.root.parent / protocol['support_geometry'].split('/outputs/', 1)[1]
    support = json.loads(support_path.read_text(encoding='utf-8'))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for actor, axis in enumerate(axes):
        axis.plot(source[:, actor, 0, 0], source[:, actor, 0, 1], 'k--', lw=2.5, label='Source pelvis')
        for row in primary:
            with np.load(args.root / row['method'] / 'evaluation_arrays.npz', allow_pickle=False) as archive:
                qpos = archive['qpos']
                axis.plot(qpos[:, actor, 0], qpos[:, actor, 1], lw=1.3, label=row['label'].replace(' / Vita adapter', ''))
        for seat in support['seats']:
            polygon = np.asarray(seat['xy_polygon_m'])
            axis.fill(polygon[:, 0], polygon[:, 1], color='#b8c6d0', alpha=.4)
        axis.set_title(f'Person{actor + 1}: root XY / fixed seat footprints')
        axis.set_xlabel('WorldX (m)')
        axis.set_ylabel('WorldY (m)')
        axis.set_aspect('equal', adjustable='box')
        axis.grid(alpha=.2)
    axes[1].legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8)
    fig.tight_layout()
    fig.savefig(report_dir / 'root_paths.png', dpi=180, bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for row in primary:
        with np.load(args.root / row['method'] / 'evaluation_arrays.npz', allow_pickle=False) as archive:
            for actor, axis in enumerate(axes):
                axis.plot(archive['time_s'], archive['signed_hip_seat_distance_m'][:, actor] * 100,
                          label=row['label'].replace(' / Vita adapter', ''))
    for actor, axis in enumerate(axes):
        axis.axhline(0, color='black', lw=.8)
        axis.axhspan(-2, 2, color='#69a583', alpha=.16, label='Geometric +/-2cm band')
        axis.set_title(f'Person{actor + 1}: hip proxy to source-nearest seat (not load bearing)')
        axis.set_ylabel('Signed distance (cm)')
        axis.grid(alpha=.2)
    axes[-1].set_xlabel('Original source time (s)')
    axes[0].legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8)
    fig.tight_layout()
    fig.savefig(report_dir / 'seat_distance_time.png', dpi=180, bbox_inches='tight')
    plt.close(fig)

    def table(input_rows, include_timing=False):
        result = ['| 方法 | 骨方向误差° | 根XY RMSE cm | 最大穿透mm | 首尾髋—凳距离mm | 支撑段踝XY速度cm/s | 关节加速度RMS rad/s² |' + (' 本次训练／求解耗时 |' if include_timing else ''),
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: |' + (' ---: |' if include_timing else '')]
        for row in input_rows:
            m = row['metrics']
            suffix = f" {timing[row['method']]} |" if include_timing else ''
            result.append(f"| {row['label']} | {m['mean_bone_direction_error_deg']:.2f} | {m['root_xy_rmse_m'] * 100:.2f} | {m['max_penetration_m'] * 1000:.2f} | {m['seat_endpoint_absolute_distance_mean_m'] * 1000:.2f} | {m['stance_foot_speed_mean_m_s'] * 100:.2f} | {m['body_joint_acceleration_rms_rad_s2']:.2f} |{suffix}")
        return result

    intermimic_path = args.root / 'intermimic/report.json'
    intermimic = json.loads(intermimic_path.read_text(encoding='utf-8')) if intermimic_path.exists() else {'status': 'probe_pending'}
    lines = ['# 双人起身、换位、坐下：重定向方法对比', '',
             '实验日期：2026-10-28（北京时间）。范围为重定向、几何/接触参考与时序质量；本轮没有执行SONIC或认定物理任务通过。', '',
             '## 已产出什么', '',
             f'- {len(primary)}组主对比参考，每组两人各172帧、原始30Hz、最后PTS5.7秒；另有{len(variants)}组明确标注的UMR配置变体。',
             '- [全片同相机对比视频](../media/overview/comparison.mp4)、[关键帧图](../media/overview/comparison_contactsheet.jpg)、各方法独立视频和参考NPZ。',
             '- 同一Vita52模型、固定凳地代理、共同原始坐标/米制尺度和坏帧修复，已保存来源、配置、命令、SHA与运行日志。', '',
             '## 先看实验结论', '',
             '在这个重建片段和本次Vita适配配置下，人体姿态/路径贴合、座面几何、支撑时段运动和平滑度存在明显取舍，不能由一项误差或视频外观直接宣布方法胜出。', '',
             '- GMR可快速得到完整关节参考，但不主动解决本片段的凳面适配。',
             '- PHC此处运行的是官方运动学拟合循环，姿态与平滑度表现需与座面穿透一起看；没有运行PHC控制策略。',
             '- OmniRetarget执行了官方交互网格约束优化；本噪声源在默认1cm/s阈值下没有检测到锁足帧，不能据求解成功宣称脚已锁住。',
             '- 当前Holosoma官方核心使用单次SQP增量信赖域与软平滑，未执行论文所述跨帧硬速度界；相邻帧waist_yaw峰值约31.76rad/s，需要额外连续性/控制可行性检查。[源码与dt审计](../omniretarget/velocity_constraint_audit.json)。',
             '- HOI-Retarget在同一GMR初始参考上运行官方窗口NLP；Pinocchio后端替换为已对MuJoCo核验的Vita CasADi FK，接触标签是显式推断，未运行stockPinocchio链路。',
             '- UMR真实构建/学习了4096点、500轮对应关系，随后执行两人全片；原生返回码0不等于通过共同场景检查。阈值与接触覆盖的消融单独保留。',
             '- 自建接触适配IK贴近座面、支撑段踝运动较小，但根部路径相对人体发生明显调整，报告同时保留未加该适配的IK。', '',
             '## 共同输入与公平性', '',
             '- 输入为two_sitting_people中的172帧双人SMPL-X/世界网格与场景；保留同一世界原点和整体scale=1。',
             '- 所有方法共同修复第二人63—69帧：62/70为锚，姿态SO3插值，平移/网格/关节点线性插值；处理有效不代表恢复了真实观测。',
             '- GPU0逐帧SMPL-X解码审计：未修复337帧的模型重解码与提供的网格/关节最大差异分别3.59/3.35微米；7个修复帧因姿态插值与世界网格线性插值不同，最大差异约5厘米。修复帧应单列，不能作为人体重建真值。[审计记录](../audit/shared_smplx_gpu0/report.json)。',
             '- 不重新拟合或放大座椅，不按方法改变源时长；形体比例、初始化、接触推断与优化预算分别披露。',
             '- 源手指hand_valid全部为false；本片段不支持比较手指重建/抓握精度。手部默认或表面适配不能当真实手部观测。',
             '- GMR/PHC/自建IK此例主要拟合身体并保留默认辅助通道；UMR原生完整52自由度表面优化会改变手/颈，不把这些无有效手部观测的输出称为手指精度更高。',
             '- 每人主要独立重定向；共享模型再检查双机碰撞，不把它称为完整联合多人HOI优化。', '',
             '## 主对比结果', '', *table(primary, include_timing=True), '',
             '以上数值均越小越好，但各列代表不同目标，不提供任意加权总分。骨方向误差减弱形体长度差异，未缩放的相对关节位置误差另存JSON。', '',
             '耗时均取本次双人各172帧实验的运行回执，不含依赖安装、模型下载、视频渲染和人工检查。UMR的32.20秒是可复用的500轮对应关系训练；同一机器人后续片段可直接复用对应槽。HOI的41.14秒仅为窗口优化，完整链路若计入本次Stage1 GMR则约48.25秒。各方法并行方式和计算资源不同，因此这些数字是本次工程记录，不是官方速度基准。', '',
             '**距离与接触的含义：**髋—凳距离是碰撞代理到最近固定凳子的有符号几何距离，首/尾各7帧汇总，接近侧面也可能距离小，不能当落座或承重通过；“支撑段踝XY速度”使用统一源踝高度/速度推断时段，不能当真实接触点滑移。穿透只覆盖共享模型启用的碰撞对，不是完整独立自碰撞证书。', '',
             '## 配置消融', '', *table(variants), '',
             'UMR refined与inferred-contact变体使用同一学习缓存；增强优化预算/推断阈值明确记录，主表仍保留原始UMR。PHC另保留seed1/2的初值敏感性，主结果始终是seed0零关节初值。', '',
             '另完整运行UMR原包ground与整段SQP后处理，使用泛化到Vita52的原目标/约束。原10轮迭代达到上限、converged=false；改善平滑不等于求解收敛或接触质量通过。该变体单列，未替换原生UMR。', '',
             '## InterMimic单列', '',
             f"接入探针状态：`{intermimic.get('status', 'unknown')}`。完整证据见 [InterMimic探针](../intermimic/report.json)。",
             'InterMimic动态策略/retarget-as-tracking与单纯运动学拟合不同；只有实际完成相同Vita输入/输出协议才计入主表。缺少对应公开入口或Vita控制合同不以其他方法轨迹代替。', '',
             '## 图与视频', '',
             '![共同指标](metric_comparison.png)', '',
             '![根部路径](root_paths.png)', '',
             '![髋与座面距离](seat_distance_time.png)', '',
             '![相邻帧速度峰值](peak_interval_rates.png)', '',
             '相邻帧速度直接使用角度差/原始dt，避免中央差分把来回跳变平均掉；不据此假定未知的硬件速度界。', '',
             '- [总对比视频](../media/overview/comparison.mp4)',
             '- [关键帧图](../media/overview/comparison_contactsheet.jpg)',
             '- [逐方法量化数据](../comparison_metrics.json)',
             '- [统一输入/模型/修复清单](../shared_input/source_manifest.json)',
             '- [PHC三初值对照](../phc_initialization_sensitivity/summary.json)',
             '- [UMR对应学习记录](../umr/learning_receipt.json)', '',
             '- [GPU0人体参数与网格审计](../audit/shared_smplx_gpu0/report.json)',
             '- [GPU0一致性图](../audit/shared_smplx_gpu0/parameter_mesh_consistency.png)',
             '- [UMR后处理消融视频](../media/overview/umr_ablation.mp4)',
             '- [UMR真实对应学习图](../umr/diagnostics/umr_correspondence_and_training.png)',
             '- [PHC收敛和初值对照图](../phc_initialization_sensitivity/phc_loss_curves.png)',
             '- [InterMimic实际接入状态图](../intermimic/diagnostics/intermimic_availability.png)', '',
             '## 耗时与计算资源', '',
             'GMR和自建IK主体为CPU求解；OmniRetarget的原生约束优化、HOI-Retarget的窗口NLP同样主要在CPU执行。UMR对应学习使用GPU3，PHC拟合及额外初值对照使用GPU6；GPU2运行InterMimic的Isaac Sim启动/接口探针，GPU0运行共享SMPL-X全片解码审计。视频渲染使用GPU3。各卡用途与结果分别留有运行证据。', '',
             'UMR一次性对应构建/训练和逐片重定向分开计时。各方法预算、初始化、加载和并行方式不同，本次耗时是运行记录，不是官方速度基准。各方法run_receipt/report/solver_receipts保留完整命令。', '',
             '## 后续接口', '',
             '本轮候选用于下一阶段SONIC/其他控制器验证。先审查限位、瞬时速度、参考跳变、源接触标签与几何，再进入实际动力学；最终仍需分别验收功能完成、源轨迹精度和重复执行。当前运动学视频不替代此前9.56秒物理演示，也不新增其成功率结论。', '',
             '## 原始实现与署名', '',
             '- [GMR](https://github.com/YanjieZe/GMR)',
             '- [UMR源码说明](https://hanyang9.github.io/UMR/)',
             '- [OmniRetarget论文](https://arxiv.org/html/2509.26633v3)与[官方Holosoma实现](https://github.com/amazon-far/holosoma)',
             '- [HOI-Retarget](https://github.com/leggedrobotics/hoi-retarget)',
             '- [PHC](https://github.com/ZhengyiLuo/PHC)',
             '- [InterMimic](https://github.com/Sirui-Xu/InterMimic)', '',
             '本报告是单片段、本地Vita适配和已披露配置的工程对比，不是上述原论文全部基准/实机复现。', '']
    (report_dir / 'REPORT.zh-CN.md').write_text('\n'.join(lines), encoding='utf-8')
    header = ['方法', '骨方向误差°', '根XY RMSE cm', '穿透mm', '首尾髋—凳距离mm', '支撑段踝XY速度cm/s', '关节加速度RMS rad/s²', '本次训练／求解耗时']
    table_html = '<table><thead><tr>' + ''.join(f'<th>{html.escape(h)}</th>' for h in header) + '</tr></thead><tbody>'
    for row in rows:
        m = row['metrics']
        values = [row['label'], f"{m['mean_bone_direction_error_deg']:.2f}", f"{m['root_xy_rmse_m'] * 100:.2f}",
                  f"{m['max_penetration_m'] * 1000:.2f}", f"{m['seat_endpoint_absolute_distance_mean_m'] * 1000:.2f}",
                  f"{m['stance_foot_speed_mean_m_s'] * 100:.2f}", f"{m['body_joint_acceleration_rms_rad_s2']:.2f}",
                  timing.get(row['method'], 'UMR变体；复用原生对应训练')]
        table_html += '<tr>' + ''.join(f'<td>{html.escape(v)}</td>' for v in values) + '</tr>'
    table_html += '</tbody></table>'
    document = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>双人重定向对比</title><style>body{max-width:1250px;margin:32px auto;padding:0 20px;font:16px/1.7 system-ui,sans-serif;color:#172d40}video,img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:8px;border:1px solid #ccd8e1}th{background:#eaf1f6}a{color:#1266a0}.note{background:#fff1d6;padding:16px}pre{white-space:pre-wrap;background:#f2f6f9;padding:20px;font:14px/1.8 system-ui}</style><h1>双人起身、换位、坐下：重定向对比</h1><p>2026-10-28 · 两人各172帧 · 同一Vita52/凳地场景 · 原始时间</p><p class="note">本轮为运动学重定向比较。视频不代表SONIC或物理任务通过；手指无有效观测。InterMimic接入状态单列于完整报告。</p><video controls src="../media/overview/comparison.mp4"></video><h2>统一指标</h2>' + table_html + '<p>髋—凳几何接近不等于承重；支撑段踝速度不等于真实接触滑移。各方法为已披露的Vita适配。</p><img src="metric_comparison.png"><img src="root_paths.png"><img src="seat_distance_time.png"><img src="peak_interval_rates.png"><h2>关键帧</h2><img src="../media/overview/comparison_contactsheet.jpg"><h2>完整说明</h2><pre>' + html.escape('\n'.join(lines)) + '</pre></html>'
    (report_dir / 'report.html').write_text(document, encoding='utf-8')
    print(json.dumps({'report': str(report_dir), 'primary_outputs': len(primary), 'variants': len(variants), 'intermimic_status': intermimic.get('status')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
