"""Collect the full N21 comparison and render only independently replayed motion."""
import os
os.environ['MUJOCO_GL']='osmesa'
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco as mj
import numpy as np
from PIL import Image,ImageDraw

from common import write_json,sha256


def main():
    root=Path(sys.argv[1]).resolve()
    experiment=json.loads((root/'experiment.json').read_text())
    assert sha256(root/'gates.json')==experiment['gates_sha256']
    assert sha256(root/'input/compiled_model.mjb')==experiment['model_sha256']
    assert sha256(root/'input/reference_raw.npz')==experiment['reference_sha256']
    manifest=json.loads((root/'input/input_manifest.json').read_text())
    ref=np.load(root/'input/reference_raw.npz',allow_pickle=False)
    time=ref['source_times_s']-ref['source_times_s'][0]
    variants=[('raw_pd','Original targets + PD'),('contact_pd','Contact SQP + PD'),
        ('sbto_fast','SBTO / original seed'),('sbto_contact_seed','SBTO / contact seed'),
        ('sbto_tracking','SBTO / tracking only'),
        ('dsms_fine','DSMS / reference seed'),('dsms_physical_seed','DSMS / physical seed')]
    rows=[];records={}
    for tag,name in variants:
        reports=[]
        for repeat in (1,2):
            folder=root/(tag+'_replay_'+str(repeat));r=json.loads((folder/'report.json').read_text())
            assert r['model_sha256']==experiment['model_sha256']
            assert r['reference_sha256']==experiment['reference_sha256']
            assert sha256(folder/'physical_record.npz')==r['physical_record_sha256']
            assert r['passed']==all(r['checks'].values()) and len(r['checks'])==18
            reports.append(r)
        assert reports[0]['physical_record_sha256']==reports[1]['physical_record_sha256']
        record=np.load(root/(tag+'_replay_1')/'physical_record.npz',allow_pickle=False)
        assert record['qpos'].shape==ref['qpos_ref'].shape and np.isfinite(record['qpos']).all()
        assert record['qvel'].shape==ref['qvel_ref'].shape and np.isfinite(record['qvel']).all()
        np.testing.assert_array_equal(record['qpos'][0],ref['qpos_ref'][0])
        np.testing.assert_array_equal(record['qvel'][0],ref['qvel_ref'][0])
        assert int(record['reset_count'][0])==int(record['state_writes_after_initialization'][0])==0
        np.testing.assert_allclose(record['times'],time,atol=1e-12)
        records[tag]=record
        bad=np.flatnonzero(np.linalg.norm(record['qpos'][:,:3]-ref['qpos_ref'][:,:3],axis=1)>.25)
        solver_path=root/tag/('result.json' if tag.startswith('dsms') else 'manifest.json')
        solver=json.loads(solver_path.read_text()) if solver_path.exists() else None
        rows.append(dict(tag=tag,name=name,passed=reports[0]['passed'],metrics=reports[0]['metrics'],
            passed_gates=sum(reports[0]['checks'].values()),gate_count=len(reports[0]['checks']),
            failed_gates=[k for k,v in reports[0]['checks'].items() if not v],
            repeated_record_hash_equal=True,physical_record_sha256=reports[0]['physical_record_sha256'],
            first_root_error_over_25cm_s=float(time[bad[0]]) if len(bad) else None,solver=solver))
    best=min((r for r in rows if r['tag'].startswith('sbto')),key=lambda r:(not r['passed'],r['metrics']['root_rmse_m']))
    audits={tag:json.loads((root/tag/'report.json').read_text()) for tag in ('audit_raw','audit_contact')}
    summary=dict(experiment=experiment,input=manifest,results=rows,audits=audits,best_sbto_tag=best['tag'],
        any_full_clip_passed=any(r['passed'] for r in rows),
        coarse_dsms_rejection='Objective directional gradient check failed before solve; see dsms.log',
        candidate_geometry_is_calibrated=False)
    write_json(root/'summary.json',summary)
    fig,axes=plt.subplots(4,1,figsize=(11,10),sharex=True)
    for row in rows:
        rec=records[row['tag']];label=row['name']
        axes[0].plot(time,np.linalg.norm(rec['qpos'][:,:3]-ref['qpos_ref'][:,:3],axis=1),label=label)
        axes[1].plot(time,rec['qpos'][:,2],label=label)
        axes[2].plot(time[1:],rec['physics_foot_force_world_N'][:,:,:,2].sum(axis=-1).mean(axis=1)/
            json.loads((root/(row['tag']+'_replay_1')/'report.json').read_text())['robot_weight_N'],label=label)
        axes[3].plot(time[1:],rec['physics_penetration_m'].max(axis=1)*1000,label=label)
    axes[0].axhline(.25,color='k',linestyle='--');axes[1].plot(time,ref['qpos_ref'][:,2],'k--',label='Kinematic target')
    axes[2].axhline(.1,color='k',linestyle='--');axes[3].axhline(5,color='k',linestyle='--')
    for ax,label in zip(axes,['Root error (m)','Root height (m)','Vertical foot force / weight','Scene penetration (mm)']):
        ax.set_ylabel(label);ax.grid(alpha=.2);ax.legend(fontsize=7,ncol=2)
    axes[-1].set_xlabel('Original N21 time (s)');fig.tight_layout();fig.savefig(root/'comparison.png',dpi=150);plt.close(fig)
    model=mj.MjModel.from_binary_path(str(root/'input/compiled_model.mjb'))
    model.vis.headlight.active=1;model.vis.headlight.ambient[:]=.5
    model.vis.headlight.diffuse[:]=.8;model.vis.headlight.specular[:]=.2
    for g in range(model.ngeom):
        if (model.geom(g).name or '').startswith('n21_new_'):model.geom_rgba[g]=[.54,.46,.34,1.]
    renderer=mj.Renderer(model,height=320,width=480);data=mj.MjData(model)
    camera=mj.MjvCamera();camera.lookat[:]=[4.15,.1,1.]
    camera.distance=4.5;camera.azimuth=115.;camera.elevation=-17.
    chosen=[None,*rows]
    preview=[]
    with imageio.get_writer(root/'comparison.mp4',fps=30,macro_block_size=1) as writer:
        for frame in range(len(time)):
            panels=[]
            for row in chosen:
                data.qpos[:]=ref['qpos_ref'][frame] if row is None else records[row['tag']]['qpos'][frame]
                data.qvel[:]=0.;mj.mj_forward(model,data);renderer.update_scene(data,camera=camera)
                panel=Image.fromarray(renderer.render().copy());draw=ImageDraw.Draw(panel)
                draw.rectangle((0,0,480,43),fill=(15,20,28))
                label='Latest N21 / kinematic only' if row is None else row['name']+' / '+('PASS' if row['passed'] else 'FAIL')
                draw.text((9,5),label,fill='white');draw.text((9,23),f'Frame {frame:03d}/208 | {time[frame]:.2f}s | 1x',fill='white')
                panels.append(np.asarray(panel))
            canvas=np.concatenate([np.concatenate(panels[:4],axis=1),np.concatenate(panels[4:],axis=1)],axis=0)
            writer.append_data(canvas)
            if frame in (0,104,208):preview.append(canvas)
    renderer.close();imageio.imwrite(root/'preview.jpg',np.concatenate(preview,axis=0))
    lines=['# N21 最新轨迹 × 新楼梯碰撞：动力学轨迹优化对比','',
        '使用 209 帧、30 Hz、6.933 秒完整最新轨迹；物理步长 1/300 秒。qpos 保留最新目标，qvel 由 qpos 按原始时钟重算。所有方法使用相同首帧、零初速度、PD 参数和碰撞模型；每个输出执行两次连续无重置独立回放，物理记录哈希一致。',
        '',f'原始轨迹 SHA256：`{manifest["source_sha256"]}`。',
        f'新碰撞 SHA256：`{manifest["geometry_sha256"]}`；200 个凸三角柱，原有孔洞保留。',
        '', '冻结物理模型为 `input/compiled_model.mjb`，控制量语义为关节 PD 位置目标。`input/scene_model.xml` 是加入 PD 参数前的中间 motor 模型，不能直接配合这些位置目标回放。',
        '', '**结果：'+('至少一个方法通过完整回放。' if summary['any_full_clip_passed'] else '本轮没有方法通过完整片段的全部 18 项门槛。')+'**', '',
        '| 方法 | Root RMSE m | 终点误差 m | 场景穿透 mm | 自穿透 mm | 足部垂直支撑 % | 通过门槛 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        m=r['metrics'];lines.append(f'| {r["name"]} | {m["root_rmse_m"]:.4f} | {m["final_root_error_m"]:.4f} | {m["penetration_max_m"]*1000:.2f} | {m["self_penetration_max_m"]*1000:.2f} | {m["foot_vertical_support_fraction"]*100:.1f} | {r["passed_gates"]}/{r["gate_count"]} |')
    lines+=['','首次根位置误差超过 25 cm：']
    for r in rows:
        onset=r['first_root_error_over_25cm_s']
        lines.append(f'- {r["name"]}：'+('未超差' if onset is None else f'{onset:.3f} s'))
    lines+=['','静态接触 SQP 只修复运动学目标，并非动力学解：最大穿透由 36.22 mm 降至 9.05 mm，站立足间隙绝对值 P95 由 26.56 mm 降至 6.78 mm；首帧、根旋转和上肢保持，最大根位置改动 39.31 mm、腿关节改动 0.234 rad。其 PD 回放另行计入上表。',
        '',f'修正后的逐帧动力学审计：原始目标 {audits["audit_raw"]["feasible_frames"]}/209 帧通过，接触修正目标 {audits["audit_contact"]["feasible_frames"]}/209 帧通过；声明支撑却没有原生接触流形的左右足样本由 {audits["audit_raw"]["declared_without_manifold_samples"]} 降至 {audits["audit_contact"]["declared_without_manifold_samples"]}。这是固定 q/v/a、仅允许声明足部接触的受限筛查，不是连续回放证书或一般不可行性证明。',
        '', 'SBTO 使用官方 CEM + incremental 核心，接触/自碰撞代价通过原生 MuJoCo 计算；保守 AABB 筛选仅减少远处碰撞体距离查询，随机几何测试验证筛选前后距离一致。两种初值各 64 samples，每阶段 2 iterations（末阶段 4）；原初值 50 步控制结点、接触初值 100 步结点，seed 0。预算和参数不同，结果是本例适配比较，不是同预算算法排名。',
        '', 'SBTO tracking-only 消融仍启用同一真实碰撞和摩擦，移除额外接触代价；128 samples，每阶段 8 iterations（末阶段 16）、50 步结点、seed 0。最终仍按同一套 300 Hz 接触与穿透门槛独立验收。',
        '', '`comparison.json` 保留最早的适配尝试：未加速 SBTO 因距离查询过慢中止，随后用等价 AABB 筛选重跑；粗节点 DSMS 被梯度自检拒绝。最终七组可回放比较以 `summary.json` 和各 `*_comparison.json` 为准，初期适配记录不算完成的结果。',
        '', 'DSMS 使用上游 multiple shooting 和 G1 tracking objective；首个 30 Hz 节点版本被梯度自检拦住，未求解。随后两次使用 300 Hz 节点、209 个线性控制样条点、75 次迭代预算、1200 CPU 秒上限，分别以参考轨迹、真实连续 PD 轨迹作初值；均通过目标梯度自检。上表只展示其控制量重新进行连续物理回放的结果，不能用 shooting 节点拼接代替验收。']
    for row in rows:
        if row['tag'].startswith('dsms'):
            s=row['solver'];lines.append(f'\n- {row["name"]}：{s["solver_message"]}；shooting 最大缺陷 `{s["initial_constraint_max_abs"]:.3g} → {s["constraint_max_abs"]:.3g}`。')
    lines+=['','验收见 `gates.json`，数值和全部失败项见 `summary.json`，控制量位于各方法 `controls.npz`，真实 30 Hz 回放轨迹位于各方法 `*_replay_1/physical_record.npz`。`comparison.mp4` 全程按原时间轴播放；首格为最新运动学参考，其余七格对应表内全部方法。图和视频由真实回放状态渲染。',
        '', '**限制：**楼梯“五个等高踏步”仍是 RGB 结构假设，未经实物尺寸标定；原接触相位和 stationary 标签作为先验保留，未重新推断。DSMS 为保持冻结 MJB 使用 MuJoCo 3.5，而上游环境要求 >=3.8；这里只能称 Vita 适配实验。未验证扰动鲁棒性和真机，不应把失败候选作为可执行轨迹。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps([dict(tag=r['tag'],passed=r['passed'],root_rmse_m=r['metrics']['root_rmse_m']) for r in rows]),flush=True)


if __name__=='__main__':main()
