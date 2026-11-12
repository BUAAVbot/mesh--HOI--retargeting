"""Build an evidence-first SONIC/specialist report; never promote training reward to success."""
from __future__ import annotations
import argparse, datetime, hashlib, html, json, math, re, shutil
from pathlib import Path


LABELS={'phc':'PHC fitting','gmr':'GMR','custom_ik':'Custom IK','custom_ik_contact':'Custom IK + contact',
        'omniretarget':'OmniRetarget','hoi_retarget':'HOI-Retarget','umr':'UMR',
        'umr_refined':'UMR refinement','umr_contact_adapted':'UMR inferred contact',
        'umr_postprocessed':'UMR ground/SQP'}


def read(path,default=None):
    try:return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError,ValueError):return default


def lines(path):
    result=[]
    try:
        for line in Path(path).read_text(encoding='utf-8').splitlines():
            try:result.append(json.loads(line))
            except ValueError:pass
    except OSError:pass
    return result


def clean(value):
    if isinstance(value,float)and not math.isfinite(value):return None
    if isinstance(value,dict):return {str(k):clean(v)for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v)for v in value]
    return value


def num(value,decimals=2):
    try:return f'{float(value):.{decimals}f}'if math.isfinite(float(value))else'未完成'
    except (TypeError,ValueError):return'—'


def relative(root,path):return Path(path).resolve().relative_to(root.resolve()).as_posix()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/data/KinoGen/outputs/sonic_expert_20261028'))
    a=p.parse_args();root=a.root.resolve();out=root/'report';out.mkdir(exist_ok=True)
    baseline=read(root/'sonic_comparison.json');protocol=read(root/'benchmark_protocol.json')
    if not baseline or not protocol:raise RuntimeError('The complete SONIC baseline is required')
    rows=baseline['results'];selected=baseline['selected_method']
    audit=read(root/'audit/baseline_zero_residual.json',{})
    final_selection=read(root/'expert_final_selection.json')
    prototype_final=bool(final_selection and final_selection.get('status')=='prototype_nominal_pass_no_robustness_gain')
    prior_transfer=read(root/'prior_transfer_screen/summary.json')
    prior_physics=read(root/'prior_physical_state_pool/report.json')
    prior_seed_export=read(root/'prior_physical_state_pool/task_seed_export_report.json')
    proprio_audit=read(root/'audit/direct_proprio_v2/report.json')
    heldout_ff=read(root/'heldout_seed990013/feedforward_baseline/summary.json')
    heldout_cem=read(root/'heldout_seed990013/cem_v4_feedforward_baseline/summary.json')
    cem_plan=read(root/'terminal_refined_full_v4/report.json')
    cem_seed=read(root/'cem_v4_physical_state_pool/report.json')
    cem_fullstate=read(root/'cem_v4_physical_state_pool/full_integration_export_receipt.json')
    sonic_transfer=read(root/'sonic_refined_reference_transfer/comparison.json')
    feedback_gain_audit=read(root/'audit/feedback_local_gain_checkpoint50/mathematical_scope_and_summary.json')
    v5_zero=read(root/'audit/direct_feedforward_v5_centered_zero/report.json')
    v5_training_done=read(root/'audit/direct_feedforward_v5_source_zero_training_done/report.json')
    residual_index=read(root/'residual_expert_experiment_index.json')
    capacity_kl_equivalence=read(root/'capacity_kl_initial_equivalence.json')
    residual_kl_update=read(out/'residual_kl_update_diagnostic.json')
    quaternion_correction=read(root/'metrics_corrections/quaternion_shortest_arc/correction_summary.json')
    physical_prior_training_objective=read(root/'audit/physical_prior_training_objective/report.json')
    closed_loop_gain=read(root/'closed_loop_gain_cem/report.json')
    closed_loop_root_gain=read(root/'closed_loop_root_gain_cem/report.json')
    direct_final32=read(root/'heldout_seed990013/v5_physical_final200/summary.json')
    heldout_initial_validity=read(root/'heldout_seed990013/initial_physical_validity_audit.json')
    residual_final=read(root/'residual_capacity_kl_final_conclusion.json')
    prototype_export=read(root/'ppo_prototype_execution_data_v2/metadata.json')
    phc_orientation_note='四元数度量审计发现：原全量评估的插值未先连续化相邻四元数符号，训练DualReference则做了符号修正。原报告约3rad的姿态峰值不能直接解释为真实朝向错误；原始报告保留，等待correction overlay。'
    if quaternion_correction:
        corrected_phc=next(row for row in quaternion_correction['records']if row['trial']=='sonic_baseline/phc')
        values=corrected_phc['corrected_max_root_orientation_error_rad']
        phc_orientation_note=f'四元数最短弧插值修正后，PHC两人最大姿态误差为{values[0]:.3f}/{values[1]:.3f}rad，均通过0.65rad门槛；原3.096rad峰值是符号插值伪尖峰，不能解释为真实朝向错误。严格源跟踪仍有XY/Z等超限，功能换座仍未通过。所有朝向结论以correction overlay为准，原始报告保持不变。'
    contact_schedule=read(root/'physical_contact_schedule_diagnostic.json')
    contact_equivalence=read(root/'refined_contact_initial_equivalence.json')
    training=[]
    for manifest_path in sorted(root.glob('*/manifest.json')):
        directory=manifest_path.parent;metrics=lines(directory/'metrics.jsonl')
        if len(metrics)<3:continue
        manifest=read(manifest_path,{})
        training.append({'name':directory.name,'manifest':manifest,'metrics':metrics,'latest':metrics[-1],
            'receipt':read(directory/'training_receipt.json'),
            'sampling_std_audit':read(directory/'sampling_std_audit.json'),
            'state_pool':read(directory/'state_pool_report.json',{}),
            'manifest_path':relative(root,manifest_path)})
    screens=[];native=[]
    for report_path in sorted(root.rglob('report.json')):
        if out in report_path.parents or 'sonic_baseline'in report_path.parts or 'prior_transfer_screen'in report_path.parts:continue
        data=read(report_path,{})
        if 'functional_passes'in data and 'num_envs'in data:
            screens.append({'path':relative(root,report_path),'data':data})
        elif 'physics_steps'in data and 'controller'in data:
            native.append({'path':relative(root,report_path),'data':data})
    import numpy as np,matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for run in training:
        manifest=run['manifest'];args=manifest.get('args',{})
        run['time_scale']=args.get('time_scale',1.)
        reference=manifest.get('reference',args.get('reference'))
        if reference and Path(reference).is_file():
            try:
                with np.load(reference,allow_pickle=False)as archive:
                    clock=archive['time_s'];source_clock=archive['source_time_s']
                    run['time_scale']=float((clock[-1]-clock[0])/(source_clock[-1]-source_clock[0]))
            except (KeyError,ValueError,OSError):pass
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    if sonic_transfer:
        transfer_root=root/'sonic_refined_reference_transfer';fig,axes=plt.subplots(1,2,figsize=(13,6))
        references=[('PHC command',root/'references/phc_time_scale1.5.npz','#8d91a1','--'),
            ('CEM actual command',transfer_root/'cem_actual_q_reference_50hz.npz','#279b70',':')]
        for label,path,color,style in references:
            with np.load(path,allow_pickle=False)as archive:
                q=archive['qpos']
                for actor,axis in enumerate(axes):axis.plot(q[:,actor,0],q[:,actor,1],style,color=color,lw=2,label=label)
        for tag,label,color in [('A_phc_coarse','A: SONIC + PHC','#397fb4'),('B_cem_refined','B: SONIC + CEM q','#c76f4d')]:
            with np.load(transfer_root/tag/'trace.npz',allow_pickle=False)as archive:
                q=archive['qpos'];roots=archive['robot_root_qadr']
                for actor,axis in enumerate(axes):
                    rq=roots[actor];axis.plot(q[:,rq],q[:,rq+1],color=color,label=label);axis.scatter(q[-1,rq],q[-1,rq+1],color=color,marker='x',s=70)
        for actor,axis in enumerate(axes):axis.set(title=f'P{actor+1} actual root trajectory',xlabel='World X (m)',ylabel='World Y (m)');axis.axis('equal');axis.grid(alpha=.2);axis.legend(fontsize=8)
        fig.suptitle('Same actual initial state, unchanged SONIC, zero learned residual | both task failures\nA:9.56s complete; B:7.56s early fall; cross=end of recorded rollout')
        fig.tight_layout(rect=[0,0,1,.90]);fig.savefig(out/'refined_sonic_transfer_paths.png',dpi=160);plt.close(fig)
    labels=[LABELS.get(row['method'],row['method'])for row in rows]
    colors=['#347ca4'if row['no_fall']else'#c87d66'for row in rows]
    fig,axes=plt.subplots(2,2,figsize=(16,12))
    specs=[('Root XY tracking RMS (cm)',[row['root_xy_rms_m']*100 for row in rows],None),
           ('Maximum physical contact penetration (mm)',[row['report']['max_penetration_m']*1000 for row in rows],25),
           ('Largest final root XY error across two robots (cm)',[max(row['report']['final_root_xy_error_m'])*100 for row in rows],30)]
    for axis,(title,values,threshold)in zip(axes.flat,specs):
        axis.barh(range(len(rows)),values,color=colors);axis.set_yticks(range(len(rows)),labels);axis.invert_yaxis()
        axis.set_xlabel(title);axis.grid(axis='x',alpha=.2)
        if threshold is not None:axis.axvline(threshold,color='#5f805d',ls='--',label=f'{threshold:g} threshold');axis.legend(fontsize=8)
        for i,value in enumerate(values):axis.text(value+max(values)*.015,i,f'{value:.1f}',va='center',fontsize=8)
        axis.set_xlim(0,max(values)*1.15)
    axis=axes.flat[-1];y=np.arange(len(rows));load=[]
    for actor in (0,1):
        value=[row['report']['final_seat_vertical_force_mean_N'][actor]/row['report']['robot_weight_N'][actor]*100 for row in rows]
        axis.barh(y+(actor-.5)*.32,value,height=.32,label=f'P{actor+1}',color=('#3a86b9','#e49548')[actor]);load.extend(value)
    axis.set_yticks(y,labels);axis.invert_yaxis();axis.set_xlabel('Final0.2s actual upward seat load (% body weight)')
    axis.axvline(10,color='#5f805d',ls='--',label='10% acceptance minimum');axis.legend(fontsize=8);axis.grid(axis='x',alpha=.2)
    axis.set_xlim(0,max(12,max(load)*1.25))
    fig.suptitle('Frozen SONIC: ALL10 configurations | original5.7s +1s final hold\nActual native physics; blue=no fall, orange=fall; functional pass0/10',fontsize=14)
    fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(out/'baseline_metrics.png',dpi=175);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(15,10))
    for run in training:
        m=run['metrics'];x=[row['iteration']for row in m];name=run['name'].replace('expert_phc_','')
        for axis,key in zip(axes.flat,('reward','training_start_successes','pool_states','std')):
            values=np.asarray([row.get(key,np.nan)for row in m],float)
            if key=='std':
                audited=(run.get('sampling_std_audit')or{}).get('actual_sampling_std_mean')
                values=np.asarray([row.get('actual_sampling_std_mean',audited if audited is not None else row.get('std',np.nan))for row in m],float)
            if key=='reward'and len(values)>15:
                line=axis.plot(x,values,lw=.6,alpha=.16)[0]
                smooth=np.convolve(values,np.ones(15)/15,mode='valid')
                axis.plot(x[14:],smooth,label=name,lw=1.5,color=line.get_color())
            else:axis.plot(x,values,label=name,lw=1.5)
    for axis,title in zip(axes.flat,('Training reward (not task acceptance)','Provisional training starts counted successful',
                                   'Validated reset-pool states','Exploration std: actual if audited, otherwise raw parameter')):
        axis.set_title(title);axis.set_xlabel('PPO iteration');axis.grid(alpha=.2)
    if training:axes[0,0].legend(fontsize=8)
    if training and all(max(row.get('training_start_successes',0)for row in run['metrics'])==0 for run in training):
        axes[0,1].set_ylim(0,1)
        axes[0,1].text(.5,.5,'No provisional successes recorded',transform=axes[0,1].transAxes,ha='center',color='#7b5550')
    fig.suptitle('Privileged specialist training | reset curriculum and metrics are not full-episode validation',fontsize=13)
    fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(out/'training_progress.png',dpi=175);plt.close(fig)
    source_root=Path('/data/KinoGen/dependencies/SUGAR-20261028')
    source_manifest=read(source_root/'source_manifest.json',{})
    if source_root.exists():
        dest=out/'sugar_source';dest.mkdir(exist_ok=True)
        for name in ('source_manifest.json','download_receipt.json'):
            if(source_root/name).exists():shutil.copy2(source_root/name,dest/name)
        for path in (source_root/'source').rglob('*'):
            if path.is_file():
                target=dest/'source'/path.relative_to(source_root/'source');target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    def link(path,label=None):
        return f'[{label or Path(path).name}](../{path})'
    generated=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')
    functional=sum(row['report']['functional_task_success']for row in rows)
    strict=sum(row['report']['success']for row in rows)
    nofall=sum(row['no_fall']for row in rows)
    quick_points=[
        f'冻结SONIC已跑完10组真实物理对比，功能{functional}/10、严格跟踪{strict}/10；PHC以根XY RMS约0.463m选为后续参考。',
        '已有双Vita起身—换位—落座视频：名义初态下功能及独立控制重放通过，严格源跟踪未过。它来自历史真实walking控制＋CPU CEM末段优化，不是新PPO专家训练成功。',
        '鲁棒性仍未达标：CEM零反馈在固定32初态中仅1/32通过；专项RL与真实q/v关节／根反馈对照须完成同初态配对验收，不能把继承的名义case0算成RL增益。',
        '本轮按原型交付视频、报告、轨迹、复放命令和SHA审计。四元数度量已修正；尚未把专家数据蒸馏进SONIC，也没有实机结论。']
    feishu_copy='已完成双人坐下任务的10组重定向轨迹SONIC物理对比，并产出双Vita起身、换位、落座的可演示原型。当前落座正例由历史真实控制前缀和CPU CEM末段电机优化获得，功能检查及独立重放通过，严格源跟踪尚未通过；不能表述为新RL专家已训练成功。固定32个初态的CEM零反馈基准仅1例通过，微扰鲁棒性仍是主要缺口。专项RL及真实状态反馈对照继续以同32初态验收，最终结果按原型汇报；视频、数据、复放命令和审计已整理，SONIC蒸馏尚未开展。'
    if prototype_final:
        quick_points[2]='已训练并交付预先指定的physical第200轮PPO原型：名义full200Hz功能与独立重放通过；固定32初态评估仍仅继承case0、相对CEM基准无新增通过。未使用heldout择优。'
        quick_points[3]='PPO权重、源码、物理先验、输入合同、全量／32评估、视频与训练导出均随原型交付；严格源跟踪未过、微扰鲁棒性无新增收益、SONIC蒸馏未做。四元数度量修正未改变成功结论。'
        feishu_copy='本轮已完成10组轨迹的SONIC真实物理对比，并交付双Vita起身—换位—落座原型。预先指定的physical第200轮PPO权重已完成名义200Hz功能验收、独立控制重放和固定32初态评估，代码、先验、输入合同、视频及训练导出已整理。需要明确：落座能力继承历史真实控制和CEM物理先验，PPO在32初态上仍仅1例通过，相对同先验零反馈没有新增通过；严格源跟踪尚未通过，不能表述为鲁棒RL专家已训练成功。两路真实状态闭环增益搜索也未新增功能通过，SONIC蒸馏尚未开展。'
    text=['# 双Vita：SONIC筛选与专项PPO原型交付'if prototype_final else'# 双Vita：冻结SONIC筛选与专项RL Expert', '',
        '实验起始日期：2026-10-28。报告生成时间：'+generated+'。本报告只对保存的实际运行证据下结论。', '',
        '## 先看这四点','',*[str(i+1)+'. '+point for i,point in enumerate(quick_points)],'',
        '## 飞书可复制结论','', '> '+feishu_copy,'',
        '## 详细证据与范围', '',
        f'冻结SONIC已经实际运行全部10组重定向配置。功能换座通过 **{functional}/10**，严格源参考跟踪通过 **{strict}/10**；{nofall}组全程保持不摔倒，但这些组最终均未形成合格双人座面承重。',
        f'按预先声明的选择规则，选定 **{LABELS.get(selected,selected)}** 作为专项专家参考：实际根XY RMS为{rows[0]["root_xy_rms_m"]:.3f}m。这个选择表示本次物理筛选中的相对较优起点，不表示它已完成换座。', '',
        '- [冻结SONIC全片物理对比视频](../media/sonic_all_methods.mp4)',
        '- [逐组实际指标](../sonic_comparison.json)、[共同协议和参考SHA](../benchmark_protocol.json)',
        '- [基线零残差独立审计](../audit/baseline_zero_residual.json)', '']
    if cem_plan:
        text.extend(['**新增一条当前PHC参考下功能通过的物理示范**：历史真实walking控制＋CPU CEM末段腿部电机计划，9.56秒完整积分和独立重放通过，严格源跟踪未过。它是专项反馈RL的正先验，**尚不是已训练通过的PPO专家**。',''])
    if final_selection:
        text.extend(['最终专家选择/验收记录已保存为 [expert_final_selection.json](../expert_final_selection.json)。记录状态：`'+str(final_selection.get('status','unknown'))+'`。',
                     '最终控制验收、未见初态测试和数据导出分别以下方证据为准，不能从选择文件名推定通过。',''])
        if prototype_final:
            text+=['预先指定physical200权重、源码、输入合同、名义full200Hz、固定32评估、原型视频和数据导出均已完成；未达标的是32初态无新增通过、严格源跟踪未过，SONIC蒸馏未做。',
                '- [PPO原型输入合同](../ppo_prototype_input_contract.json)',
                '- [PPO原型视频：名义能力继承先验](../media/ppo_feedback_prototype.mp4)']
            if prototype_export:text+=[f"- [训练执行数据导出回执](../ppo_prototype_execution_data_v2/metadata.json)：{prototype_export['policy_samples']}个50Hz样本，保留请求phase和实际积分时钟、实际q/世界根速度/电机标签/接触力；learned_rl_expert_success_claim=false。导出不等于蒸馏。"]
            text+=['']
    else:text.extend(['**专项专家仍在训练/评估；当前没有已完成最终验收的专家选择记录。** 训练更新、BC拟合误差与checkpoint筛查均单列，不能写成完整任务成功。',''])
    text+=['## 各阶段实际做到什么', '',
        '| 阶段 | 本次已有证据 | 结论范围 |','| --- | --- | --- |',
        '| 运动学重定向 | 六类方法、10组配置，双人各172帧 | 已在前一轮完成；不等于控制可执行 |',
        f'| 冻结SONIC | 同一模型、原始30Hz时钟、5.7秒＋1秒末端保持，10组实际积分 | 功能{functional}/10、严格跟踪{strict}/10 |',
        '| 直接专项专家 | 特权PPO、实际接触反馈、渐进状态池；SONIC不参与其执行 | '+('physical200原型已完成名义全量／32评估，无新增通过；严格未过'if prototype_final else'各检查点需独立闭环验收')+' |',
        '| 冻结SONIC残差专家 | 另一个任务PPO网络输出关节目标残差，SONIC权重锁定 | 属于本项目工程方案，不能称为SUGAR原版Refiner |',
        '| CEM电机计划正示范 | 历史实际walking前缀＋CPU末段电机参数搜索；当前PHC功能通过 | 开环名义计划，严格跟踪未过；不是PPO专家 |',
        '| BC／DAgger稳定性初始化 | 采集SONIC实际执行，拟合其电机目标；包含学生状态查询 | 仅初始化对照，拟合好不代表闭环稳定或任务成功 |',
        '| 专家数据交给通用控制器 | 只有通过功能及独立控制重放的轨迹才可作为正样本导出 | 尚未执行对SONIC的蒸馏；不能声称通用策略已变强 |','',
        '## 冻结SONIC物理结果', '',
        '| 参考配置 | 执行秒数 | 不摔 | 根XY RMS cm | 最终最大XY误差cm | 最大穿透mm | 末段P1/P2座面向上力N | 功能／严格 |',
        '| --- | ---: | --- | ---: | ---: | ---: | --- | --- |']
    for row in rows:
        r=row['report'];f=r['final_seat_vertical_force_mean_N']
        text.append(f"| {LABELS.get(row['method'],row['method'])} | {r['simulated_seconds']:.2f} | {'是'if row['no_fall']else'否'} | {row['root_xy_rms_m']*100:.2f} | {max(r['final_root_xy_error_m'])*100:.2f} | {r['max_penetration_m']*1000:.2f} | {f[0]:.2f}/{f[1]:.2f} | {'通过'if r['functional_task_success']else'未过'}/{'通过'if r['success']else'未过'} |")
    text+=['','![冻结SONIC指标](baseline_metrics.png)','',
        '根XY RMS计算的是整个已记录物理执行相对该方法参考的误差；早停样例的误差覆盖时间较短，因此同时列出执行时长和早停状态。最终座面承重使用实际接触向上力，不以几何距离小或视觉上坐在凳边代替。',
        '本组SONIC基线均直接从各自运动学参考的第0帧姿态、零速度初始化，未先做物理落座。这是相同初始化算法，但不同方法的实际初始关节/接触状态不同，比较同时包含参考初态质量的影响。后续若改用旧演示的实际承重初态，必须另做同初态SONIC与专家配对，把初始化修正收益和RL收益分开。',
        phc_orientation_note,'',
        '## 专家训练与状态池记录','',
        '| 运行 | 架构 | 最新迭代／本次转移 | 源时间倍率 | 池状态／池重置 | 训练完成回执 |',
        '| --- | --- | --- | ---: | --- | --- |']
    for run in training:
        m=run['manifest'];last=run['latest'];args=m.get('args',{});pool=run['state_pool']
        text.append(f"| {run['name']} | {'冻结SONIC＋专项残差'if args.get('sonic_base')else'直接专家'} | {last.get('iteration','—')} / {last.get('transitions_this_run','—')} | {num(run['time_scale'],1)} | {last.get('pool_states','—')} / {last.get('pool_resets','—')} | {'有，仍需验收'if run['receipt']else'尚无完成回执'} |")
    text+=['','![训练过程](training_progress.png)','',
        '训练采样会使用参考状态初始化和经过验证的实际中间状态，并在失败后重置；训练平均误差、奖励和临时成功计数不属于从原始初态连续执行的成功率。奖励图的实线为15条记录移动平均、淡线为原值。直接专家、减速配置和冻结SONIC残差专家分别保留；时间放慢不作为原始时长通过。','',
        '原始直接专家每路600×64×32约122.88万次状态转移。SUGAR论文记录的30000×4096×24约29.49亿次转移，单路规模约为本次原始配置的2400倍；机器人、数据和优化实现也不同。本地短预算未通过不能据此认定SUGAR方法失败。','',
        '采样幅度审计发现，原v4_tiny/v4_small虽然原始std参数为0.005/0.01，共享分布函数将实际std下限夹到0.030197，两路已停止并保留为该探索幅度对照。隔离修正分支将实际下限设为0.001，逐次记录actual_sampling_std_mean和实际反馈幅度；不能按旧分支名声称已验证0.005探索。std图优先实际审计值/实际日志，无此证据时显示原始参数。','',
        '状态池候选来自真实MuJoCo积分后的状态，并经过后续10个控制步（0.2秒）窗口筛查；参考状态初始化单独计数，不称为物理验证状态。原始SONIC残差专家在池重置时还需披露其历史观测重置合同。','',
        '## 检查点闭环筛查','',
        '| 检查点证据 | 迭代 | 世界数 | 功能／严格通过 | 不摔世界 | 数值告警世界 | 级别 |',
        '| --- | ---: | ---: | --- | ---: | ---: | --- |']
    for item in screens:
        r=item['data'];cases=r.get('results',[]);nofall_cases=sum(bool(case.get('checks',{}).get('no_fall'))for case in cases)
        text.append(f"| {link(item['path'],Path(item['path']).parent.name)} | {r.get('iteration','—')} | {r.get('num_envs','—')} | {r.get('functional_passes',0)}/{r.get('strict_passes',0)} | {nofall_cases} | {r.get('numerical_warning_worlds','未记录')} | 50Hz筛查 |")
    text+=['','此表是固定检查点的闭环筛查，包含未扰动初态及明确声明的关节/根位置小扰动。筛查接触与状态为50Hz采样；最终选中策略仍需200Hz控制记录和独立控制重放。MuJoCo数值告警/恢复须判为失败，不以恢复后的有限状态冒充连续成功。','',
        '## 全量物理验收记录','',
        '| 证据 | 控制器 | 物理步 | 功能／严格 | 独立控制重放 |','| --- | --- | ---: | --- | --- |']
    for item in native:
        r=item['data'];c=r.get('controller',{})
        method=r.get('candidate_provenance',{}).get('metadata',{}).get('method',c.get('method','—'))
        text.append(f"| {link(item['path'],Path(item['path']).parent.name)} | {method} | {r.get('physics_steps','—')} | {'通过'if r.get('functional_task_success')else'未过'}/{'通过'if r.get('success')else'未过'} | {'通过'if r.get('independent_replay',{}).get('passed')else'未通过/未记录'} |")
    if not native:text.append('| 暂无新专家最终验收报告 | — | — | 未确认 | 未确认 |')
    text+=['','功能检查同时要求全片完成、双人起身/换位、实际落座承重、末段稳定、穿透及无摔倒；严格检查再要求参考位置/姿态/关节误差。记录控制的独立重放验证积分可复现性，不等于重新运行闭环策略，也不等于泛化率。','',
        ]
    if quaternion_correction:
        text+=['## 四元数度量修正覆盖层：不重跑物理、不改原证据','',
            f"已对{len(quaternion_correction['records'])}条完整trace按相邻四元数最短弧重新计算姿态误差，保留已连续片段的原算术。6个antipodal中点/边界回归均通过。严格成功变化{len(quaternion_correction.get('strict_success_changes',[]))}条、功能成功变化{quaternion_correction.get('functional_success_changes',0)}条。实际q/v、ctrl、接触和sourceclock未改变。",
            '| 物理执行 | 原最大姿态P1/P2 rad | 修正后P1/P2 rad |','| --- | --- | --- |']
        for row in quaternion_correction['records']:
            if row['trial']in ['sonic_baseline/phc','expert_control_v1','terminal_refined_full_v4','sonic_refined_reference_transfer/A_phc_coarse','sonic_refined_reference_transfer/B_cem_refined']:
                oldori=row['old_max_root_orientation_error_rad'];newori=row['corrected_max_root_orientation_error_rad']
                text.append(f"| {row['trial']} | {oldori[0]:.3f}/{oldori[1]:.3f} | {newori[0]:.3f}/{newori[1]:.3f} |")
        text+=['','PHC原始SONIC基线的朝向门槛实际通过；CEM示范约0.771/0.894rad仍超过严格0.65rad，因此不能把修正写成严格源复现通过。',
            '- [全部修正覆盖层、原始SHA和回归结果](../metrics_corrections/quaternion_shortest_arc/correction_summary.json)']
        if physical_prior_training_objective:
            text+=['- [实际CEM状态对原训练门槛的独立审计](../audit/physical_prior_training_objective/report.json)：按真实累积policy clock和DualReference，1913态/20ms边界无source-relative失败done；“旧训练门槛中途截断这条正示范”的猜测已被实测推翻。']
        text+=['修正前源码快照及修正脚本一并保存在交付中，原raw报告可追溯；不使用原姿态伪峰值解释控制失败。','']
    if cem_plan:
        prefix=(cem_seed or{}).get('inherited_old_walking_prefix',{})
        text+=['## CEM电机计划功能正示范与反馈RL边界','',
            'terminal_refined_full_v4在当前PHC time1.5参考下通过全部功能门槛：两台Vita起身、换位、实际落座承重、末段稳定、无摔倒及穿透限制，执行1912个200Hz物理步（9.56秒）。独立控制重放qpos/qvel误差均0；严格源参考位置/朝向/关节跟踪未通过。',
            f"末端两人XY误差分别为{cem_plan['final_root_xy_error_m'][0]:.4f}/{cem_plan['final_root_xy_error_m'][1]:.4f}m，座面向上力{cem_plan['final_seat_vertical_force_mean_N'][0]:.2f}/{cem_plan['final_seat_vertical_force_mean_N'][1]:.2f}N，最大穿透{cem_plan['max_penetration_m']*1000:.2f}mm。",
            f"前{prefix.get('control_steps_identical','—')}个控制步、约{prefix.get('prefix_duration_s','—')}秒及对应q/v逐字节继承旧实际示范；新搜索只修改两人腿部电机目标，未写root、未施加外力。该正例不能归因于PPO训练或冻结SONIC独立完成新PHC轨迹。",
            '299个未来0.2秒实际物理有效种已导出，phase为真实执行时钟，q/v/ctrl不缩放。PHC路径误差保留为标签；新反馈动作和队列历史置零，不把完整名义电机包当成反馈动作。',
            '- [新CEM完整物理验收](../terminal_refined_full_v4/report.json)',
            '- [继承前缀、修改电机与物理种审计](../cem_v4_physical_state_pool/report.json)']
        if cem_fullstate:
            checks=cem_fullstate.get('continuation_checks',[])
            text+=[f"另保存299×1237完整mjSTATE_INTEGRATION。{len(checks)}个中间时刻恢复后继续原ctrl执行0.2秒，q/v与原轨迹逐字节相同；仅q/v恢复出现最高{max(row['qv_only']['max_qpos_absolute_error']for row in checks):.3g}位置/{max(row['qv_only']['max_qvel_absolute_error']for row in checks):.3g}速度差。这是有限迭代数值状态的区别，尚不能据此断言长期必然失稳。",
                '- [完整积分状态与continuation审计](../cem_v4_physical_state_pool/full_integration_export_receipt.json)']
        text+=['新反馈RL必须相对同一CEM名义计划的零反馈控制做配对比较；不能只对旧FF比较，把CEM计划改进记为RL收益。','']
    if proprio_audit:
        clock_note=proprio_audit.get('timestamp_alignment_note',{})
        text+=['## 直接专家v2观测与正BC接口审计','',
            'v2增加146维：每人69维实际/命令身体关节、实际/命令重力方向、实际机体系根线/角速度和高度，再加双人共8维座位相对位置/期望接触。总观测481维；训练、BC、筛查与200Hz执行共用同一个augment函数。',
            f"只读CPU核查{proprio_audit.get('samples_checked','—')}个正BC样本通过：独立旋转矩阵重建feature最大差{proprio_audit.get('max_manual_feature_absolute_error',0):.3g}，完整离线观测最大差{proprio_audit.get('max_stored_full_observation_absolute_error',0):.3g}，电机label最大差{proprio_audit.get('max_label_absolute_error',0):.3g}。两人的关节/执行器顺序正确。",
            f"全部旧实际body控制目标相对PHC的最大偏置{proprio_audit.get('all_body_target_max_abs_offset_rad',0):.4f}rad，小于2.2rad偏置经label0.98裁剪后的有效容量2.156rad，未发生目标范围裁剪。接口对齐通过不能替代BC闭环或专项任务验收。",
            f"该审计版本的取样边界差异为{clock_note.get('ceil_search_vs_nearest_row_mismatch_count','未记录')}/{proprio_audit.get('samples_checked','—')}条，最大状态时刻差{clock_note.get('maximum_state_time_lag_s',0)*1000:.3f}ms；实际timestamp/phase有记录，对应body target包最大差{clock_note.get('resulting_body_packet_difference_max_rad',0):.3g}rad。后续nearest边界版本与本审计SHA版本应分别记录。",
            '[v2观测、label与源码SHA审计](../audit/direct_proprio_v2/report.json)。','']
    if prior_transfer:
        text+=['## 旧专家零更新迁移诊断','',
            '这是旧任务专家零更新迁移，未重新训练；执行时间统一为源时间的1.5倍。它不属于冻结SONIC主表，也不是本次新专家训练成功。',
            '| 新参考 | 世界数 | 功能通过 | 不摔世界 | 根XY RMS均值m |','| --- | ---: | ---: | ---: | ---: |']
        for row in prior_transfer.get('results',[]):
            text.append(f"| {LABELS.get(row['method'],row['method'])} | {row['num_envs']} | {row['functional_passes']} | {row['no_fall_worlds']} | {row['mean_root_xy_rms_m']:.3f} |")
        text+=['','[旧专家零更新诊断数据](../prior_transfer_screen/summary.json)。','']
    if prior_physics:
        replay=prior_physics.get('old_controls_replayed_in_current_model',{})
        text+=['## 旧成功物理演示只作为初始化来源','',
            '旧candidate156来自此前自建IK＋SONIC/PPO换座实验，是筛选过的功能物理演示，严格源跟踪失败；不能归为本次PHC专家的新成功。该旧轨迹只有在模型和坐标合同匹配后，才可提供实际初态与状态池种子。',
            f"本次核对{prior_physics.get('model_arrays_checked','—')}个模型数组、{prior_physics.get('model_scalars_checked','—')}个标量及物理选项，模型参数相同：`{prior_physics.get('model_parameters_identical')}`。旧1912个控制包在当前冻结模型重放通过：`{replay.get('passed')}`，qpos/qvel最大误差分别{replay.get('max_qpos_absolute_error','—')}/{replay.get('max_qvel_absolute_error','—')}。",
            '旧actual轨迹对当前PHC参考的全段根XY RMS约0.305m，末段关节姿态RMS约0.631rad。物理有效状态不保证PHC轨迹精度，种子文件同时保留这些误差标签。',
            '- [模型、重放、PHC对齐及严格86种审计](../prior_physical_state_pool/report.json)',
            '- [旧真实初态和完整actual数据导出回执](../prior_physical_state_pool/initial_export_receipt.json)']
        if prior_seed_export:
            text+=['- [接触有效种和task-only种导出回执](../prior_physical_state_pool/task_seed_export_report.json)',
                '实际接触有效种有288个，其中41个位于末段：保留未来0.2秒无摔、穿透、座面承重和安静门槛，去掉PHC姿态/路径门槛；另保留有放宽PHC门槛的task-only版本。469个候选全部连同有效/无效mask保存，未过门槛者不标有效。',
                '各NPZ的q、v、真实ctrl不缩放或改写；phase按source1x/1.5x/2x分别映射，旧末段保持映射参考末相位。新专家动作历史明确置零，原SONIC历史须由消费者重置。这些初始化数据不能证明新控制器会保持旧成功轨迹。']
        text+=['']
    if contact_schedule:
        text+=['## 用实际承重精修接触时段','',
            '从旧candidate156实际轨迹提取接触/向上承重标签，保留原粗标签和来源SHA；该精修参考的qpos、时间逐元素未变。P1粗标签末段从137帧开始，持续实际承重从156帧开始；P2粗持续段从108帧开始，持续实际承重从134帧开始。按2倍源时长映射，差约1.27秒和1.73秒。',
            '这说明粗接触先验会过早要求承重，符合使用物理执行数据修正先验的工程动机；标签仍来自旧已知演示，不能当新PHC策略的实测真值或任务通过。',
            '- [接触时段与来源审计](../physical_contact_schedule_diagnostic.json)']
        if(out/'physical_contact_schedule.png').exists():text+=['','![接触时段对照](physical_contact_schedule.png)','']
        if contact_equivalence:text+=['- [粗／精标签零反馈初态等价性](../refined_contact_initial_equivalence.json)：同实际初态和扰动下，qpos/qvel/rootXY/座力数组最大差均为0。']
        text+=['修正标签后性能也可能退化；筛查表保留各迭代结果，不能选一个较好检查点就宣布标签修正已解决任务。','']
    if heldout_ff:
        phc=heldout_ff['references']['phc_time1p5'];old=heldout_ff['references']['old_original_reference']
        text+=['## 固定32初态的旧实际FF零反馈基准','',
            'heldout seed990013仅用于评估，禁止训练。case0是逐字节不变的旧完整INTEGRATION初态；其余31例在reset时对双人根XY各加±1cm、28身体关节各加±0.01rad并按限位裁剪，实际qvel及其他状态保留。后续反馈专家必须使用同一文件做成对比较。',
            '| 验收参考 | 功能计数 | 严格计数 | 不摔 |','| --- | ---: | ---: | ---: |',
            f"| 当前PHC，time1.5 | {phc['functional_passes']}/32 | {phc['strict_passes']}/32 | {phc['no_fall_cases']}/32 |",
            f"| 旧原reference_v3_deploy_slow | {old['functional_passes']}/32 | {old['strict_passes']}/32 | {old['no_fall_cases']}/32 |",'',
            '旧1912包200Hz实际控制按原时钟直接执行，没有再加名义动作delay。case0重现9.56秒旧演示，qpos/qvel对旧trace误差均0；它按旧参考功能通过，但按PHC末端XY<0.30m判据未过。其余31个小扰动初态约1.5秒跌至既有root<0.15m终止条件，独立控制重放全部通过。',
            '验收直接执行现有replay评估器提取的原指标/判据AST，未放宽门槛。此结果表明旧选定开环控制对初态敏感，不能把它的单次名义重放当稳健策略；新RL反馈的收益需由同32初态比较证明。各例RMS覆盖的执行前缀长短不同，不可直接按平均RMS排胜负。',
            '- [固定初态文件与SHA](../heldout_seed990013/initial_states.receipt.json)',
            '- [完整32例双参考验收](../heldout_seed990013/feedforward_baseline/summary.json)',
            '- [全部200Hz核心状态及valid mask](../heldout_seed990013/feedforward_baseline/all_cases200hz_core.npz)',
            '这些是限定初态集合的配对计数，包含一个固定名义case；不解释为任意环境或任意动作的成功概率。','']
    if heldout_cem:
        phc=heldout_cem['references']['phc_time1p5']
        text+=['## 同32初态：新CEM名义计划零反馈对照','',
            f"同seed990013、同完整初态文件、同200Hz物理及原验收门槛，新CEM名义计划的PHC功能计数 **{phc['functional_passes']}/32**，严格 **{phc['strict_passes']}/32**，不摔 **{phc['no_fall_cases']}/32**。case0完整重现新CEM正例；其余31个微扰仍早期倒地。全部控制记录再次独立重放通过。",
            '新CEM在未扰动名义初态上把PHC功能计数从旧FF的0提升到1，但没有增加小扰动下稳定的case数。新PPO反馈的比较基准是这一行，而非旧FF的0/32。',
            '- [新CEM固定32初态双参考验收](../heldout_seed990013/cem_v4_feedforward_baseline/summary.json)',
            '- [新CEM全部200Hz核心状态](../heldout_seed990013/cem_v4_feedforward_baseline/all_cases200hz_core.npz)','']
        if heldout_initial_validity:
            text+=[f"独立初态审计显示32/32起点均通过既有穿透/高度/倾斜门槛：最大初始穿透{heldout_initial_validity['max_initial_penetration_m']*1000:.3f}mm、最低根高度{heldout_initial_validity['min_initial_root_height_m']:.4f}m。因此不能把后续失败归因为这些初态本来就越过验收阈值；该审计未静置、未改初态，也不能证明动态可恢复。",
                '- [固定32初态物理有效性](../heldout_seed990013/initial_physical_validity_audit.json)','']
    if direct_final32:
        text+=['## 预先指定physical200：同32初态最终对照','',
            '该PPO200检查点在heldout评价前预先指定，source-final200只作对照，不用heldout择优。固定32初态只用于评价，训练种子与数据分离。',
            '| 控制方案 | 功能通过 | 相对CEM新增通过 | 来源边界 |','| --- | ---: | ---: | --- |',
            '| CEM名义计划，零反馈 | 1/32 | 基准 | 仅case0 |',
            f"| PPO physical final200 | {direct_final32['functional_passes']}/{direct_final32['num_cases']} | {direct_final32.get('incremental_functional_passes',0)} | 名义能力继承物理先验 |"]
        for label,record in [('实际q/v关节增益CEM',closed_loop_gain),('实际根＋关节增益CEM',closed_loop_root_gain)]:
            if record:
                held=record['heldout'];text.append(f"| {label} | {held['functional_passes']}/{len(held['episodes'])} | {held['functional_passes']-1} | 参数训练与heldout分离，非PPO |")
        text+=['',f"PPO final200严格通过{direct_final32['strict_passes']}/32，独立控制重放{direct_final32['independent_replay_passes']}/32。case0相对CEM prior的q/v/ctrl最大差均0，学习反馈动作最大值0；这说明名义能力被保留，**不能算RL新增技能或鲁棒性收益**。",
            '- [PPO200固定32初态完整评价](../heldout_seed990013/v5_physical_final200/summary.json)','']
    if sonic_transfer:
        text+=['## 把物理精修实际q参考交给冻结SONIC的实测','',
            '这是参考命令层面的接入验证，没有更新SONIC权重或执行蒸馏。A使用PHC time1.5参考；B将已过功能验收CEM轨迹的实际qpos映射为双人59维参考，按50Hz取0..8.54秒并加精确8.55秒端点，保留原sourceclock与实际接触/承重标签。旧末段保持未重复算入动作段；两组都额外保持1秒，200Hz完整目标为9.56秒。',
            '两组使用相同完整CEM actual初态、同冻结MJB与new_ft15225 SONIC；残差记录全0，旧专家checkpoint仅提供架构尺寸。逐项SHA、初态逐字节一致性与零残差审计均通过。',
            '| 参考命令 | 执行秒数 | 功能／严格 | 不摔 | 末段P1/P2座力N | 根XY RMS m |','| --- | ---: | --- | --- | --- | ---: |']
        for row in sonic_transfer['results']:
            refkey='PHC'if row['tag'].startswith('A_')else'CEM_actual';force=row['final_seat_vertical_force_mean_N']
            text.append(f"| {row['tag']} | {row['simulated_seconds']:.2f} | {'通过'if row['functional_task_success']else'未过'}/{'通过'if row['success']else'未过'} | {'是'if row['functional_task_checks']['no_fall']else'否'} | {force[0]:.2f}/{force[1]:.2f} | {row['source_tracking_cross_references'][refkey]['root_xy_rms_m']:.3f} |")
        text+=['','A全时长未摔，但未换座/承重；B在7.56秒因P2根高度低于0.15m早停。B的RMS覆盖较短执行前缀且参考不同，不能据较小数值宣布改善。两组独立控制重放误差均0。',
            '**当前结果：可执行电机计划的实际关节轨迹，直接作为命令交给原SONIC，仍未迁移成功。** 能执行该计划和现有通用控制器能跟踪这条关节命令是两项不同能力；后续仍须真正的策略/数据蒸馏或反馈专家训练。本对照不加入原10组排名，不算PPO专家成功。',
            '![实际参考迁移的世界轨迹](refined_sonic_transfer_paths.png)',
            '- [参考schema、来源与SHA](../sonic_refined_reference_transfer/reference_manifest.json)',
            '- [两组200Hz结果与独立配对审计](../sonic_refined_reference_transfer/comparison.json)','']
    if feedback_gain_audit:
        text+=['## CPU虚拟扰动：反馈门控的局部增益风险','',
            '此诊断固定tiny_actualstd第50轮网络，对CEM真实名义状态t=0.5/3/8/9秒，分别在根XY、身体关节q、根线速度和关节速度施加±1e-5/1e-4/1e-3虚拟扰动。共148个状态、72组正负配对，只在CPU推理；未积分物理、未训练、未改控制器。所有新策略历史固定为零。',
            '现门控约为`时间门控 × 归一化状态误差范数 × policy_offset`，四个精确名义状态的门控均0。用`odd=(u(+ε)-u(-ε))/2`提取方向性响应，`even=(u(+ε)+u(-ε))/2-u(0)`提取同向偏置。',
            '| P1根X，实际时刻s | gated odd阶数 | gated even阶数 | 未乘范数的policy odd阶数 | ε=1e-3时even/odd |','| ---: | ---: | ---: | ---: | ---: |']
        for row in feedback_gain_audit['rootX_phase_summaries']:
            text.append(f"| {row['phase_s']:.1f} | {row['gated_odd_log_slope_last_decade']:.3f} | {row['gated_even_log_slope_last_decade']:.3f} | {row['ungated_policy_odd_log_slope_last_decade']:.3f} | {row['even_over_odd_at1e3']:.1f} |")
        text+=['','表中阶数使用ε=1e-4→1e-3的十倍缩放。16组phase×位置/关节q方向中，13组gated odd阶数为1.9~2.1；其余含浮点分辨率/非光滑影响，未强行解释为二阶。',
            '数学上，若局部`h(e)=h0+He+O(e²)`且范数门控`g(e)≈c|e|`，则领先偏置`c|e|h0`为even，而方向性项`c|e|He`是二阶。实测支持这项结构风险，不能证明它是所有任务失败的唯一原因，也不是闭环稳定性证明。',
            '后续可独立测试以相同phase/history构造`h(obs(actual q,v))-h(obs(nominal q,v))`，去掉误差范数乘子并保留独立验收的动作边界。它在名义状态输出0且可保留一阶方向响应；本诊断没有实现该控制器，仍需真正物理配对验证。',
            '- [全部56电机分量、门控、正负分解及源码SHA](../audit/feedback_local_gain_checkpoint50/report.json)',
            '- [数学范围与数值汇总](../audit/feedback_local_gain_checkpoint50/mathematical_scope_and_summary.json)',
            '![虚拟局部反馈阶数](../audit/feedback_local_gain_checkpoint50/feedback_local_gain.png)','']
    if v5_zero:
        text+=['## v5居中反馈与KL更新：接口审计阶段','',
            'v5已实现同phase/history下的actual与完整counterfactual观测相减；反事实同时包含名义q/v、FK位置/速度、PHC命令、座位与零历史。随机网络权重下，1912物理步名义轨迹的观测差、策略mean、q/v/实际ctrl误差均0。这证明名义不变性接口通过，不是已训练专家或扰动鲁棒性成功。',
            '- [v5名义零反馈全量审计](../audit/direct_feedforward_v5_centered_zero/report.json)']
        if v5_training_done:
            text+=['真实训练门槛配对审计中，CEM零反馈轨迹首次done是9.560029秒的正常timeout，first_training_failed为空；没有证据支持“PHC门槛中途拒绝正示范”的先前猜测。物理计划奖励属于另一个工程对照。',
                '- [同CEM轨迹的真实训练done审计](../audit/direct_feedforward_v5_source_zero_training_done/report.json)']
        if residual_index:
            text+=['- [冻结SONIC残差各支路与筛查索引](../residual_expert_experiment_index.json)：已保存的筛查均未功能通过，CEM正示范另列。']
        if capacity_kl_equivalence:
            text+=['残差KL对照的全部actor/critic/normalizer初始张量与原capacity对照逐字节相同，零输出层、模型、参考、动作尺度相同；仅学习率与声明的KL更新规则改变。首minibatch曾测得全56维KL约1.2716并实际回滚模型和Adam后降LR，这说明更新尺度敏感，但不表示任务已解决。',
                '- [KL对照初始等价性](../capacity_kl_initial_equivalence.json)']
        if residual_kl_update:
            text+=[f"优化器诊断截至第{residual_kl_update['through_iteration']}轮：{residual_kl_update['accepted_steps']}次更新接受、{residual_kl_update['rollbacks']}次回滚，最大接受KL为{residual_kl_update['max_accepted_kl']:.5f}。第200轮独立n8筛查仍0功能、8不摔；优化器更新受控不能等同于换座技能成功。",
                '![KL更新尺度诊断](residual_kl_update_diagnostic.png)']
        if residual_final:
            r=residual_final['final_selection_screen']
            text+=[f"残差KL路线最终400轮：{residual_final['accepted_optimizer_updates']}次接受更新、{residual_final['rolled_back_proposals']}次回滚、数值fault为{residual_final['numeric_fault_episodes']}；最终n8功能{r['functional_passes']}/8、不摔{r['no_fall']}/8，根XY RMS {r['root_xy_rms_mean_m']:.3f}m，两人座面力仍全0。更新稳定性已纠正，但该冻结SONIC＋PPO方案仍未完成双人换座。",
                '- [残差KL400最终负结论](../residual_capacity_kl_final_conclusion.json)']
        text+=['']
    if closed_loop_gain or closed_loop_root_gain:
        text+=['## 真实状态闭环增益CEM：独立负结果','',
            '这两路搜索的是实际q/v误差反馈增益，电机反馈随当前物理状态改变；与前面的末段开环电机计划CEM不同，也不是PPO。root反馈使用私有scratch计算运动学映射，不写活体root或外力。训练初态与固定heldout32分离，后者只在搜索后评价。',
            '| 反馈参数搜索 | 训练功能 | heldout32功能 | 名义不变性 | 结论 |','| --- | --- | --- | --- | --- |']
        for label,record,folder in [('关节q/v增益',closed_loop_gain,'closed_loop_gain_cem'),('根状态＋关节增益',closed_loop_root_gain,'closed_loop_root_gain_cem')]:
            if record:
                train=record.get('best',{});hold=record.get('heldout',{});inv=record.get('nominal_invariance',{})
                text.append(f"| {label} | {train.get('functional_passes','—')}/{len(train.get('episodes',[]))} | {hold.get('functional_passes','—')}/{len(hold.get('episodes',[]))} | q/v/u最大差{inv.get('max_q_vs_nominal','—')}/{inv.get('max_v_vs_nominal','—')}/{inv.get('max_control_vs_nominal','—')} | 继承case0不能算新专家增益 |")
                text.append('- '+link(folder+'/report.json',label+'完整结果'))
            else:text.append(f'| {label} | 搜索中 | 尚未最终评价 | 待最终记录 | 不根据存活时长称任务成功 |')
        text+=['','关节反馈搜索12代×32候选×8训练初态，最终训练0/8、heldout1/32，仅保留原名义case0。较长存活时长属于未达标诊断指标，不等于完成起身换座落座任务。',
            '[负结果与边界索引](negative_result_index.json)。','']
    text+=['## SUGAR提供的依据与本次工程差异','',
        '[SUGAR原论文](https://arxiv.org/html/2605.20373v1)提出：把运动学交互先验交给特权RL Refiner，用跟踪、交互、正则目标在物理中精修，再把实际执行的技能蒸馏到Command Tracker与Generator。其渐进状态池保存并验证实际中间状态。论文没有实现“SONIC失败检测→自动训练专家→自动更新SONIC”的既定闭环。','',
        '| 项目 | SUGAR公开方法／实现 | 本次双Vita方案 |','| --- | --- | --- |',
        '| 机器人与引擎 | G1、Isaac Sim5.1／IsaacLab2.3 | 两台Vita52、MuJoCo3.5，CPU原生物理＋GPU策略 |',
        '| Refiner | 特权PPO与交互先验精修 | 自建直接专家；另测冻结SONIC残差专家，均需独立验收 |',
        '| 初始化 | 参考初始化与渐进实际状态池 | 本地改写的0.2秒候选验证及分桶池，具体阈值在源码／回执 |',
        '| 奖励 | 跟踪、交互一致性、运动正则 | 复用本项目跟踪并加座面相对位置和实际承重；非原奖励逐项原样复现 |',
        '| BC与蒸馏 | Refiner→Tracker，之后学习Generator | BC／DAgger是SONIC→独立专家的额外稳定性初始化；未完成反向蒸馏 |',
        '| 随机化与实机 | 论文含物性随机化、扰动与真实机器人 | 这里只做所列初始化扰动；没有完整随机化或实机结论 |','',
        '官方源仓库固定提交：`'+str(source_manifest.get('commit','未记录'))+'`。本次只保存关键奖励、命令、Refiner配置和LICENSE等六个文件；[下载SHA清单](sugar_source/download_receipt.json)不是完整环境安装证明。','',
        '## 可复核的边界','',
        '- 旧任务PPO检查点在冻结SONIC基线中只用于加载维数/元数据；动作被硬置零。真正执行的是两个固定new_ft15225 SONIC actor，逐帧零残差与推理次数见独立审计。',
        '- 专家使用与PHC基线逐字节一致的冻结MJB，电机/转子/关节和物性数组复载误差为0；[模型合同](../calibration/report.json)。',
        '- 源手指hand_valid全为false，本任务不能评价手部重建或抓握精度。',
        '- 场景碰撞和物性含重建估计/工程假设；没有实测系统辨识或真实机器人结论。',
        '- 若没有经过功能及独立重放验收的专家轨迹，不导出“成功专家正样本”；训练数据和失败诊断仍可以保存并标明。',
        '- 精修动作或专家数据尚未蒸馏进SONIC；原通用权重保持不变。','']
    text_string='\n'.join(text)
    (out/'REPORT.zh-CN.md').write_text(text_string,encoding='utf-8')
    summary={'generated_at_asia_shanghai':generated,'baseline_count':len(rows),'baseline_functional_passes':functional,
        'four_point_summary':quick_points,'feishu_copy_draft_not_sent':feishu_copy,
        'baseline_strict_passes':strict,'baseline_no_fall':nofall,'selected_reference':selected,
        'independent_baseline_audit_passed':audit.get('passed'),
        'training_runs':[{'name':run['name'],'latest':run['latest'],'receipt':run['receipt'],'state_pool':run['state_pool'],'sampling_std_audit':run['sampling_std_audit']}for run in training],
        'checkpoint_screens':screens,'native_evaluations':native,'final_selection':final_selection,
        'prior_zero_update_transfer_diagnostic':prior_transfer,'old_physical_prior_model_audit':prior_physics,
        'old_physical_prior_seed_export':prior_seed_export,
        'direct_proprio_v2_interface_audit':proprio_audit,
        'paired_heldout_zero_feedback_baseline':heldout_ff,'physical_contact_schedule':contact_schedule,
        'cem_motorplan_physical_demonstration_not_ppo':cem_plan,'cem_physical_seed_audit':cem_seed,
        'cem_full_integration_state_audit':cem_fullstate,'paired_heldout_cem_zero_feedback_baseline':heldout_cem,
        'frozen_sonic_refined_reference_transfer_not_distillation':sonic_transfer,
        'cpu_virtual_feedback_local_gain_audit_not_rollout':feedback_gain_audit,
        'v5_nominal_invariance_interface_audit':v5_zero,'v5_actual_training_done_audit':v5_training_done,
        'residual_owned_experiment_index':residual_index,'capacity_kl_initial_equivalence':capacity_kl_equivalence,
        'residual_kl_optimizer_diagnostic_not_task_acceptance':residual_kl_update,
        'raw_quaternion_peak_interpretation_suspended_pending_correction_overlay':not bool(quaternion_correction),
        'quaternion_shortest_arc_metric_correction_overlay':quaternion_correction,
        'orientation_metric_authority':'Use quaternion_shortest_arc_metric_correction_overlay for corrected orientation conclusions; native/baseline original report objects remain raw evidence.',
        'actual_physical_prior_training_objective_audit':physical_prior_training_objective,
        'closed_loop_joint_gain_cem_not_ppo':closed_loop_gain,'closed_loop_root_gain_cem_not_ppo':closed_loop_root_gain,
        'predeclared_physical200_heldout32':direct_final32,'heldout32_initial_physical_validity':heldout_initial_validity,
        'residual_kl400_final_negative_result':residual_final,'prototype_execution_data_export_not_distillation':prototype_export,
        'refined_contact_initial_equivalence':contact_equivalence,
        'explicitly_no_generalist_distillation_completed':True}
    (out/'report_data.json').write_text(json.dumps(clean(summary),ensure_ascii=False,indent=2)+'\n')
    negative_index={'schema':'sonic_expert.failure_and_boundary_index.v1','baseline':{'functional_passes':functional,'configurations':len(rows)},
        'frozen_sonic_residual_index':residual_index,'v5_checkpoint_screens':[x for x in screens if 'v5'in x['path']],
        'closed_loop_joint_gain_cem':closed_loop_gain,'closed_loop_root_gain_cem':closed_loop_root_gain,
        'not_a_new_ppo_success':'CEM nominal physical plan and inherited unperturbedcase0 are separate from learned feedback or generalization gains.',
        'final_selection_record':final_selection}
    (out/'negative_result_index.json').write_text(json.dumps(clean(negative_index),ensure_ascii=False,indent=2)+'\n')
    table_rows=[]
    for row in rows:
        r=row['report'];table_rows.append('<tr>'+''.join('<td>'+html.escape(str(value))+'</td>'for value in
            (LABELS.get(row['method'],row['method']),num(row['root_xy_rms_m']*100),num(r['max_penetration_m']*1000),
             '是'if row['no_fall']else'否','通过'if r['functional_task_success']else'未过'))+'</tr>')
    document='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>双Vita SONIC与专项专家</title>
<style>body{max-width:1280px;margin:32px auto;padding:0 22px;font:16px/1.7 system-ui;color:#173047}img,video{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #ccd7df;padding:8px;text-align:left}th{background:#e9f1f7}.note{background:#fff1d6;padding:15px}pre{white-space:pre-wrap;font:14px/1.85 system-ui;background:#f2f6f9;padding:20px}a{color:#12649a}</style>
<h1>双Vita：冻结SONIC与专项RL Expert</h1>'''
    document+=f'<p class="note">冻结SONIC：功能{functional}/10，严格跟踪{strict}/10。PHC选为本次专家参考。训练奖励、拟合误差和50Hz筛查不替代最终物理验收。</p>'
    document+='<ol>'+''.join('<li>'+html.escape(point)+'</li>'for point in quick_points)+'</ol><h2>飞书可复制结论</h2><p>'+html.escape(feishu_copy)+'</p>'
    if cem_plan:document+='<p class="note">CEM名义物理示范：PHC功能通过、严格源跟踪未过；固定32初态零反馈通过1/32。'+('PPO200已完成名义全量与固定32评估，仍仅继承case0、无新增通过。'if prototype_final else'PPO专项反馈专家仍需独立配对验收。')+'</p>'
    document+='<p><a href="REPORT.zh-CN.md">中文报告文件</a> · <a href="../sonic_comparison.json">冻结SONIC数据</a> · <a href="../audit/baseline_zero_residual.json">零残差审计</a></p>'
    document+='<video controls src="../media/sonic_all_methods.mp4"></video><h2>冻结SONIC结果</h2><table><tr><th>参考</th><th>根XY RMS cm</th><th>穿透mm</th><th>不摔</th><th>功能</th></tr>'+''.join(table_rows)+'</table>'
    document+='<img src="baseline_metrics.png"><h2>专家训练过程</h2><img src="training_progress.png"><h2>完整报告</h2><pre>'+html.escape(text_string)+'</pre></html>'
    (out/'report.html').write_text(document,encoding='utf-8')
    targets=re.findall(r'\]\(([^)]+)\)',text_string)+re.findall(r'(?:src|href)="([^"]+)"',document)
    checked=[]
    for target in targets:
        if target.startswith(('http:','https:','#')):continue
        checked.append({'target':target,'exists':(out/target).resolve().exists()})
    validation={'checked_links':len(checked),'broken_links':[row['target']for row in checked if not row['exists']],
        'results':checked,'report_builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'report_link_validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
    if validation['broken_links']:raise RuntimeError('Broken report links: '+str(validation['broken_links']))
    print(json.dumps({'report':str(out),'baseline_count':len(rows),'training_runs':len(training),
        'checkpoint_screens':len(screens),'native_evaluations':len(native),'final_selection_present':bool(final_selection),
        'broken_report_links':len(validation['broken_links'])},ensure_ascii=False))


if __name__=='__main__':main()
