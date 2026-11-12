# G1 与 Vita 机器人资产、控制器及验证结果

盘点日期：2026-10-28。这里按实际模型和控制合同区分各条机器人路线。已有模型、运动学回放、真实策略推理、物理任务通过和完整论文复现是不同状态；以各自冻结报告为证据。

## 1. 模型与关节合同

| 模型路线 | 关节与控制量 | 本地资产与说明 |
| --- | --- | --- |
| GRAIL 使用的 G1 手部资产 | 43 关节资产，应单独维护名称、顺序及动作映射；不能直接套用下面 ROHand 的 41 控制量合同 | [GRAIL](../projects/grail/README.md)、[SONIC 内 G1 模型](../projects/grail/imports/SONIC/gear_sonic_deploy/g1/g1_29dof_with_hand.xml) |
| G1＋双五指 ROHand | 29 身体控制量＋双手 12 独立输入＝**41 独立控制量**。每手 6 输入经厂家非线性连杆公式展开为 25 运动关节；共 79 展开运动关节、89 links | [组合 URDF](../projects/kinogen/data/robots/g1_rohand_five_finger/g1_rohand.urdf)、[五指模型说明](../projects/kinogen/docs/G1_FIVE_FINGER.zh-CN.md) |
| G1 抱箱训练派生模型 | 保持 41 策略输出，将双手展开的 50 个关节用位置伺服近似。接触时实际关节不一定严格满足目标连杆耦合；不是真实闭链手的完整动力学 | [训练资产目录](../projects/kinogen/data/robots/g1_box_training)、[训练说明](../projects/kinogen/docs/G1_BOX_TRAINING.zh-CN.md) |
| Vita02a 原始五指模型 | **52 可动关节＝28 身体＋2 颈部＋22 手指**。原始轨迹按自身 dof_names 映射，不能套用 G1 索引 | [Vita URDF](../projects/kinogen/data/vita_demo_20261026/robot/vita02a/urdf/vita02a_linkerhand_meshcoll_relative.urdf)、[模型与轨迹输入](../projects/kinogen/data/vita_demo_20261026) |
| HDMI-Vita鞋版/非鞋版快照 | 两份URDF均为**28可动关节**，主要身体和腕关节；其Isaac Lab控制配置不是当前52关节五指手模型 | [HDMI Vita URDF](../projects/hdmi/active_adaptation/assets/vita02a/urdf/vita02a_shoes.urdf)、[模型配置](../projects/hdmi/active_adaptation/assets/vita.py)、[差异与缺项](HDMI_MODULE.zh-CN.md) |
| Vita＋新版 LinkerHand O6 | 每手 **11 物理关节、6 独立驱动、5 被动 mimic 关节**。单 Vita 保留 52 通道接口，但 10 个被动通道直接驱动为零；双 Vita 共 104 通道，其中 84 个有直接驱动 | [O6 原始资产](../projects/kinogen/data/linkerhand_o6_06)、[O6 接入报告](../projects/kinogen/reports/O6_06_HAND_INTEGRATION_2026-10-27.md) |

ROHand 和 O6 是两种不同的手。ROHand 的前五个独立输入是米制滑块，最后一个为弧度旋转；O6 的关节角度使用弧度。O6 也需按分支选择冻结合同：长椅 O6_v4 使用标定 JSON 的二次耦合与 SDK 表，后续六器皿 SONIC/O6 v2 按线性 URDF mimic 合同执行，不能将两套公式叠加。不能互换控制量或简单按相同通道数拼接。

**后续 Vita 训练模型约定（2026-10-29）：**默认使用 HDMI-Vita 的 28 可动关节模型；仅在任务明确需要手指抓取等 O6 能力时，切换到 Vita＋O6 模型，并同步适配动作、观测、碰撞与奖励。HDMI 鞋版和非鞋版按任务场景确定，训练与评估必须使用同一版本并记录版本。已有 Vita＋O6 结果仍按原模型解释，不能直接当作 HDMI 训练结果。

## 2. G1：模型、重建、训练与技能扩展

### 五指组合模型与视频重建

- 构建入口：[build_g1_five_finger.py](../projects/kinogen/scripts/build_g1_five_finger.py)。耦合入口：[user_rohand_coupling.py](../projects/kinogen/scripts/user_rohand_coupling.py)。
- 组合树、左右五指张合、厂家输入展开和原身体 FK 检查已完成。[模型报告](../projects/kinogen/reports/user-g1-five-finger.json)。
- 用户双视角抱箱视频的 121 帧重建已形成 V1–V4 候选。V4 修复可见性与腕壳问题，仍有源数据缺测、条件帧和非严格零穿透，不能当作全片可靠的物理抓取。[V4 说明](../projects/kinogen/docs/USER_VIDEO_RECONSTRUCTION_V4.zh-CN.md)、[独立审计](../projects/kinogen/reports/user-v4-independent-audit.json)。
- 预览：[五指张合](../projects/kinogen/previews/user_g1_five_finger_demo.mp4)、[V4 完整网格](../projects/kinogen/previews/user_v4_full_mesh.mp4)。

### 抱箱 PPO：已通过限定任务的物理验收

入口为 [run_g1_box.sh](../projects/kinogen/scripts/run_g1_box.sh)、[train_g1_box.py](../projects/kinogen/scripts/train_g1_box.py)、[g1_box_physics.py](../projects/kinogen/scripts/g1_box_physics.py)；独立验收为 [audit_g1_box_policy.py](../projects/kinogen/scripts/audit_g1_box_policy.py)。

- 真实 Isaac／PhysX 训练完成 **300 轮、4,800 次优化器更新、2,457,600 次环境转移**。
- 固定 1 kg 箱体、±1 cm 初始位置扰动下，32 个预定独立评估回合 **32/32 通过**；末端连续满足抱稳条件的最短时间 4.92 秒。
- 活动段无机器人／箱体状态强写、无外力托举、无箱体附着。动力学模型为浮动基座；手部为展开伺服近似且关闭自碰撞。
- 方法是 **IK 闭手／抬升参考＋PPO 残差**；未验证任意重量、负重行走、真实闭链手或实机，也不等于 KinoGen 官方完整复现。

证据：[最终验收](../projects/kinogen/reports/g1-box-policy-final-audit.json)、[训练来源](../projects/kinogen/reports/g1-box-training-provenance.json)、[训练说明](../projects/kinogen/docs/G1_BOX_TRAINING.zh-CN.md)、[实际物理视频](../projects/kinogen/previews/g1_box_policy_final.mp4)。权重在服务器 `/data/KinoGen/outputs/g1_box_training/train_001/final.pt`，本地报告绑定其 SHA。

### 双手技能 DAG：已完成代表性转运与放置

[本地说明](../projects/kinogen/previews/g1_box_skill_dag/README.md)保存完整来源、控制和数据约定。服务器工程为 `/data/yihang.bai/HumanoidMimicGen/bimanual_dag/`。

- 12 节点、20 依赖边：左右抓取、抬升、转运、落台、释放、退出。双手同步门控。
- 抓取和抬升复用原 PPO actor；转运／放置由新增双臂 Jacobian IK 实现，**没有新增训练**。
- 两个完整代表样例为 **20.30 秒、27.34 秒**；主样例独立动作重放 1015 步，箱体与根节点位置误差为 0。
- 当前范围是同一桌面的短距离站立转运；未验收负重步行、全局避障、厘米级精确放置或统计泛化成功率。

查看：[视频与数据目录](../projects/kinogen/previews/g1_box_skill_dag)。

## 3. Vita：原始回放、SONIC 和物理任务

### 17 条轨迹：运动学交付完成

[prepare_vita_demo_inputs.py](../projects/kinogen/scripts/prepare_vita_demo_inputs.py)、[render_vita_motion_demos.py](../projects/kinogen/scripts/render_vita_motion_demos.py)、[audit_vita_motion_demos.py](../projects/kinogen/scripts/audit_vita_motion_demos.py)处理原始轨迹。

**17 段、6415 帧、原始 30 FPS** 的 52 关节回放全部导出并解码验证。这里没有配套物体、场景、接触力或策略驱动成功证据。

查看：[视频索引](../projects/kinogen/previews/vita17_index.html)、[输入审计](../projects/kinogen/reports/vita17-input-audit.json)、[视频审计](../projects/kinogen/reports/vita17-video-audit.json)、[方法说明](../projects/kinogen/docs/VITA17_DEMOS.zh-CN.md)。

### 新 SONIC 15225：真实策略推理和站立对照已通过

权重为 [model_step_015225.pt](../projects/kinogen/data/vita_demo_20261026/checkpoints/new_ft15225/model_step_015225.pt)。主接口为 [fourth_sonic_controller.py](../projects/kinogen/scripts/fourth_sonic_controller.py)，配置见 [fourth_sonic_newft15225.json](../projects/kinogen/configs/fourth_sonic_newft15225.json)。测试入口为 [fourth_sonic_smoke.py](../projects/kinogen/scripts/fourth_sonic_smoke.py)、[fourth_sonic_default_stand.py](../projects/kinogen/scripts/fourth_sonic_default_stand.py)。

- 原 FSQ Actor 权重严格加载，已找到近源 Vita 训练资产、关节映射、观察历史、延迟及执行器合同并实现迁移。
- **5 秒、1000 个物理步、250 次真实推理**的默认站立对照通过：根水平漂移 5.22 mm、根下降 4.87 mm、最大穿透 7.30 mm。
- 28 身体动作使用 SONIC，颈部／手部辅助关节使用参考 PD。MuJoCo 原生隐式 PD 近似 Isaac 驱动；并联踝电机空间投影等原引擎细节尚非严格等价。
- 精确重放使用冻结 `model.mjb`；同目录 XML 是审阅副本，数值精度不足以代替二进制作逐步精确复现。

证据：[最新合同说明](../projects/kinogen/reports/FOURTH_SONIC_CONTRACT_2026-10-27.md)、[5 秒报告](../projects/kinogen/outputs/fourth_sonic_newft_default_stand_5s/report.json)、[冻结模型](../projects/kinogen/outputs/fourth_sonic_newft_default_stand_5s/model.mjb)。旧 [vita17-policy-tracking-feasibility.json](../projects/kinogen/reports/vita17-policy-tracking-feasibility.json) 中的跌倒和缺件结论保留为历史诊断，不能覆盖上述新证据。

### 单碗 v12：PD 短物理闭环已通过

源第 590–620 帧物体轨迹与时间保持不变，机器人采用站立参考和明确记录的手部接触适配。运行 [fourth_vita_physics.py](../projects/kinogen/scripts/fourth_vita_physics.py)，使用原生 PD，而非 SONIC。

- 2 秒、400 步，12 项物理质量门槛全部通过，独立动力学重放误差为 0。
- 原物体轨迹最大位置误差 8.85 cm、最大穿透 15.38 mm；手碗接触 1.635 秒、法向冲量 1.410 N·s。
- 导出关节运动、物体 6D 轨迹、接触图、仿真接触力和关节驱动力矩。物性是假定参数，力不是传感器真值。

查看：[完整说明](../projects/kinogen/reports/FOURTH_VITA_DEMO_RESULT_2026-10-27.md)、[冻结试验](../projects/kinogen/outputs/fourth_vita_demo/physics_source_slide_v12_final)、[HOI 数据](../projects/kinogen/outputs/fourth_vita_demo/hoi_v12_final)、[结果包](../projects/kinogen/deliveries/fourth_vita_v12_results_20261027.zip)、[视频](../projects/kinogen/previews/fourth_vita_physics_v12.mp4)。

### 六器皿全序列：已执行，质量未通过

管线入口为 [run_fourth_full_pipeline.py](../projects/kinogen/scripts/run_fourth_full_pipeline.py)，模块包括 [全序列参考](../projects/kinogen/scripts/fourth_full_reference.py)、[六物体资产](../projects/kinogen/scripts/fourth_six_object_assets.py)、[物理积分](../projects/kinogen/scripts/fourth_full_physics.py)、[接触反馈](../projects/kinogen/scripts/fourth_contact_feedback.py)、[wrench 检查](../projects/kinogen/scripts/fourth_contact_wrench.py)、[独立审计](../projects/kinogen/scripts/audit_fourth_full_physics.py)、[统一输出](../projects/kinogen/scripts/export_fourth_full_hoi.py)。

[完整 SONIC 静态基座试验报告](../projects/kinogen/outputs/fourth_full_sequence/physics_sonic15225_stationary_v1/report.json)已记录连续 **30.74 秒、6148 步、923 源帧**的真实动力学执行，无活动段 state writes 或外力。机器人基本稳定，但 `status=failed_full_sequence_quality`、`simulation_passed=false`。后续 O6 接触控制、mimic 保护、两指力闭环及重力 wrench 诊断已实现，仍未形成完整六器皿承重搬运通过证据。

### 单碟与RGB145：新增局部物理进展

2026-10-28补充另外两条独立分支：

| 分支 | 当前物理证据 | 界限与入口 |
| --- | --- | --- |
| 单碟桌沿离桌/持稳 | v7原手/原薄碟最低碰撞点离桌24.82mm，独立1ms控制重放误差0；v8高度窗达34.34mm/1.93秒，但后续滑落、稳定承重失败 | 桌沿预设初态、自由根PD＋接触反力前馈，不是原视频全程/SONIC成功；[v7审计](../projects/kinogen/reports/single_plate_microstep_v7_independent_hold_audit.json)、[v8高度窗](../projects/kinogen/reports/single_plate_microstep_v8_probe/independent_height_windows.json)、[台账](../projects/kinogen/reports/fourth_solver_experiment_ledger_v2.json) |
| RGB145单Vita台阶/落座 | 单人350帧，全程运动学预览；源前3.9997秒物理连续前缀通过，独立控制重放误差0 | 最新试验只覆盖源4.319秒后安全停；左脚上阶和末端落座未完成。使用SONIC/WBC/MPC/PPO比较；[前缀审计](../projects/kinogen/outputs/rgb145_pipeline/mpc_temporal_v20d/physics_v21d_knots3_h80/prefix_quality_audit.json)、[物理视频节选](../projects/kinogen/outputs/rgb145_pipeline/rgb145_seat_step_physics_excerpt.mp4) |

以上两个分支、下文的新双Vita换座、已停止的双人长椅/鞋分别记录，不合并成功判据。单碟v9/v10本地脚本与fixture已有，本次未核验其最终执行结果；加厚唇缘分支是显式改变几何的对照，不计原几何通过。

### 双Vita换座：功能通过，严格源轨迹未通过

新输入为 [two_sitting_people.zip](../archives/raw_handoffs/two_sitting_people/two_sitting_people.zip)：双人172帧，两人从凳子起身、交叉换位再落座。与下节已停止的398帧长椅/鞋样本不同。输入含SMPL-X人体网格和世界姿态，但默认手型不是可靠MANO重建；当前任务不依赖抓物。

- 两台浮动基座Vita52共同在MuJoCo世界积分，`nq=118/nv=116/nu=104`；SONIC控制两台各28身体通道，PPO学习56维身体残差，颈/手部辅助通道另按参考控制。
- 已完成参考坏帧修复、坐标转换、脚支撑/落座适配、批量SONIC接口对照、CPU FP64原生物理训练及候选独立重放。选用固定450检查点，并非最终750检查点。
- 本批256次采样筛出5条，5条全部通过独立5ms功能验收；选156号完成9.56秒、1912步的起身/换位/落座。末端座面竖直载荷约418/537N，无根驱动、活动段状态强写或外力托举，记录电机控制独立重放误差0。
- 严格源轨迹跟踪失败：全程根XY最大误差0.506/0.485米，门槛0.25米；最大穿透17.3mm。其余251次未逐条独立验收，不能据筛选比例宣称鲁棒性；重放未独立重跑闭环策略。
- 已导出实际状态、控制、接触图、仿真接触力/力矩、两路虚拟头部视角与相机参数。物性未实测，Ego相机未标定，凳子固定于世界，无独立动态物体轨迹。

代码：[资产](../projects/kinogen/scripts/two_sitting_assets.py)、[重定向](../projects/kinogen/scripts/two_sitting_retarget.py)、[批量SONIC](../projects/kinogen/scripts/two_sitting_batch_sonic.py)、[原生PPO环境](../projects/kinogen/scripts/two_sitting_native_ppo_env.py)、[训练](../projects/kinogen/scripts/two_sitting_train_ppo.py)、[候选搜索](../projects/kinogen/scripts/two_sitting_search_native.py)、[独立重放](../projects/kinogen/scripts/two_sitting_replay_candidate.py)。

结果：[最终审计](../projects/kinogen/outputs/two_sitting_pipeline/final_20261027/final_status.json)、[实际物理视频](../projects/kinogen/previews/two_sitting_final_functional.mp4)、[HOI元数据](../projects/kinogen/outputs/two_sitting_pipeline/delivery_final_candidate156/hoi_motion_physics.metadata.json)、[视频与数据包](../projects/kinogen/deliveries/two_vita_seat_exchange_demo_20261027.zip)。该ZIP未包含约332MiB冻结MJB；精确重放需服务器 `/data/KinoGen/outputs/two_sitting_pipeline/physics_selected450_candidate156/model.mjb`，使用MuJoCo **3.5.0**，模型SHA保存在包和审计内。

## 4. 新 O6 资产与双人长椅／鞋项目

`06.zip` 已形成可复用的 O6 资产接入模块：逐节碰撞体、左右网格、拇指零位、范围、六驱动／五被动耦合、SDK 标定表。已完成 212 项输入／输出可重复 SHA、模型编译、被动通道零直接驱动力及部分手内碰撞检查。

入口：[rgb_near_hand_collision_variant.py](../projects/kinogen/scripts/rgb_near_hand_collision_variant.py)、[rgb_near_o6_hand_variant.py](../projects/kinogen/scripts/rgb_near_o6_hand_variant.py)、[rgb_near_o6_interface_audit.py](../projects/kinogen/scripts/rgb_near_o6_interface_audit.py)、[rgb_near_o6_build_receipt.py](../projects/kinogen/scripts/rgb_near_o6_build_receipt.py)、[o6_calibration.py](../projects/kinogen/scripts/o6_calibration.py)。[接入说明](../projects/kinogen/reports/O6_06_HAND_INTEGRATION_2026-10-27.md)记录真实标定／镜像迁移和仿真假设的区别。

双 Vita 长椅／鞋管线包含 [重定向](../projects/kinogen/scripts/rgb_near_retarget.py)、[资产](../projects/kinogen/scripts/rgb_near_sim_assets.py)、[物理执行](../projects/kinogen/scripts/rgb_near_physics.py)、[任务反馈](../projects/kinogen/scripts/rgb_near_task_feedback.py)、[审计](../projects/kinogen/scripts/audit_rgb_near_physics.py)、[输出](../projects/kinogen/scripts/export_rgb_near_hoi.py)。

**此样例已由用户弃用并停止，完整 398 帧物理任务未通过。**最后一轮实际执行 12.25 秒、覆盖约 10 秒源时间；静态接触或短前缀不是完整任务成功。旧 native-GJK 零距离拇指接触断言已被独立推翻并撤回。模型、源码、失败报告和接触日志保留供复用，不自动恢复仿真。

状态证据：[停止说明](../projects/kinogen/reports/rgb_near_task_stopped_2026-10-27.json)、[进程及最后试验状态](../projects/kinogen/outputs/rgb_near_pipeline/PHYSICS_STOPPED_STATUS.json)、[GJK 审计](../projects/kinogen/reports/RGB_NEAR_GJK_DISTANCE_AUDIT_2026-10-27.md)。

## 5. 其他物理技能结果及运行依赖

[HumanoidMimicGen 导航＋按钮扩展](../projects/kinogen/previews/humanoidmimicgen_report_v2)保存 6 个代表性真实物理成功及独立重放结果，最大状态误差为 0、活动段状态注入为 0。其范围是无遮挡闭环接近＋实时物体相对手臂 IK，没有全局障碍规划、统计鲁棒性或完整论文复现结论。[数据验证](../projects/kinogen/previews/humanoidmimicgen_report_v2/data-verification.json)。应按该任务本身的机器人及控制合同记录，不能仅因视频中同为人形机器人就和其他资产共享关节数组。

所有训练、模型推理和动力学执行按现有约定在 `yz-ai-4090-01.bce` 上完成，本地用于代码、文件校验及视频预览。

| 运行内容 | 已使用的服务器位置／环境 |
| --- | --- |
| G1 抱箱训练和评估 | `/data/KinoGen`；只读复用 `/data/grail-envs/tracking45`，Isaac Sim 4.5／Isaac Lab 2.1／Torch 2.5.1 |
| Vita／器皿／长椅动力学 | `/data/KinoGen`；`/data/grail-envs/retarget/bin/python`，MuJoCo 3.7.0；SONIC 另依赖已核验近源代码 `/data/yunhao.xia/GR00T-WholeBodyControl` |
| 双Vita换座最终训练/重放/渲染 | `/root/vita02a-mjlab/.venv310/bin/python`，MuJoCo **3.5.0**；原生CPU FP64动力学＋GPU批量策略推理；该冻结MJB不能直接改用上行3.7环境加载 |
| G1 技能 DAG | `/data/yihang.bai/HumanoidMimicGen/bimanual_dag`；只读复用 tracking45 |
| 按钮导航扩展 | `/data/yihang.bai/HumanoidMimicGen/results/navigation_report_002`，本地保留视频及验收副本 |

旧脚本中的 GPU 号是历史运行设置，重跑前应检查当时资源及配置。冻结结果应使用新输出目录复测，避免覆盖原权重、模型、参考和审计证据。

新加入UMR带Vita02A双人换座预设，但单体要求 `nq=35/nv=34`（自由根＋28身体关节），与这里Vita52的 `nq=59/nv=58`不同。它是候选重定向模块，尚未接入SONIC/PPO；外部Vita资产及283个LFS实际资源仍需补齐，详见[UMR盘点](UMR_MODULE.zh-CN.md)。
