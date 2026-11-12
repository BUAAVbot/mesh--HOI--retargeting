# 多 SIM → 三方法重定向 → SONIC → RL 专家批处理

## 当前可用部分

- `inspect_cases.py`：检查解包后的 SIM 交接目录，或直接只读扫描 ZIP/TAR 原包，核对 manifest、timeline、人体文件、人数、帧数和压缩包 SHA-256；拒绝不安全成员路径和归档链接。已有双人坐下 172 帧、单人台阶 350 帧均通过识别，结果见 `known_cases.json` 和 `known_archives.json`。原包扫描暂不支持 7z。
- `compile_jobs.py`：把案件清单和两类任务的命令配方扩成固定依赖图：每例 `prepare → {GMR, UMR, OmniRetarget} → HOI-Retarget → evaluate → SONIC`，全部合格案例再汇入**一个多轨迹专家训练作业**。配方缺阶段或标记不可用时直接拒绝生成可运行计划。
- `rank_candidates.py`：对三种方法使用同一份运动学评估报告，以关节限位、最大穿透、关节速度作探索性硬门槛，再按任务接触、根轨迹和骨架误差排序；只输出“送 SONIC 的候选”，不宣称物理成功。旧双人坐下对比结果见 `known_shortlist.json`：这组门槛下只有 HOI-Retarget 进入物理筛选。门槛需在更多样本上校准。
- `dag_runner.py`：最多 8 个并发作业，管理 GPU 0–7 的独占分配；CPU 求解器申请 0 卡，UMR/SONIC 可按作业申请 1 卡，分布式专家可申请 8 卡。每步保存日志、退出码和输出回执；声明的输出文件必须真实存在且非空；`--resume` 只有在命令和声明输入的 SHA-256 未变时复用。`selftest.py` 已验证依赖、1/8 卡分配与断点恢复。

## 尚未打通的接口

目前**不能把 10 多个新 SIM 直接投入后宣称自动完成专家训练**：

1. 双人对比预处理写死 172 帧以及 P02 的 63–69 帧修复；UMR 适配器写死 `rgb_174114_sim_handoff`、两个座椅和 172 帧。新片必须按真实有效帧动态修复。
2. OmniRetarget、HOI-Retarget 当前 Vita 适配器按两人和座椅建约束，单人台阶需要独立的接触/地形配置；不能把台阶伪装成座椅任务。
3. 当前评分器写死双人 172 帧。候选轨迹应先按时间、关节限位、足/场景穿透、运动连续性筛选，再让前两名过 SONIC 物理评估，以功能成功和稳定性确定最终训练参考。单纯运动学最低误差不等于物理最好。
4. 双人 SONIC benchmark 固定了原案例目录和资产路径；单人台阶使用另一个控制入口。两类任务都需可复用的输入、模型、检查点、输出参数合同。
5. 服务器 mjlab 目前只完成 Vita28 **单人**两轮 PPO 烟测；HDMI 的特权老师/学生算法及双 Vita 同场训练尚未迁移，不能把烟测检查点称为专家。

## 推荐执行方式

将新压缩包放到 [incoming](../../incoming/README.md)。先用 `inspect_cases.py` 逐包核验 SHA 与包内路径，解包到独立只读输入目录，再生成可运行的案件清单。每个 SIM 的三种方法应使用同一源帧、同一 Vita 模型、同一场景几何和同一评估口径。八卡并行是**跨作业调度**，不是让主要使用 CPU 的 OmniRetarget/HOI-Retarget 虚占 GPU；分布式 RL 需要训练器实际支持多卡同步。

计划与执行命令：

```bash
python inspect_cases.py /path/to/sim_a /path/to/sim_b --output cases.json
python compile_jobs.py --inventory cases.json --recipes recipes.json \
  --output-root /data/KinoGen/outputs/batch_sim_expert --jobs jobs.json
python dag_runner.py jobs.json                    # 仅打印计划
python dag_runner.py jobs.json --run --resume --gpus 0,1,2,3,4,5,6,7
```

`recipes.json` 必须在上述五项接口解除案例硬编码、完成各阶段验收后填写；当前目录故意不提供可误启动的伪配方。每种任务需定义七个案例阶段，另定义 `_global_expert`；每个阶段配方需包含 `argv`（无 shell 的参数数组）、`outputs`（验收文件）、可选 `inputs`（SHA 校验）、`cwd`、`env` 和 `gpus`。一次处理 10 个案例将生成 71 个依赖作业。若某案例失败，其余案例的筛选可继续，但汇总专家训练不会误用不完整数据而自动启动。
