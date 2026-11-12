# KinoGen 公开组件运行状态

## 当前状态导航（2026-10-27）

整套工作区已整理为[总入口](../README.zh-CN.md)、[流程模块地图](../docs/PIPELINE_MODULES.zh-CN.md)、[论文复现清单](../docs/PAPER_REPRODUCTIONS.zh-CN.md)与[G1/Vita专项索引](../docs/ROBOTS_G1_VITA.zh-CN.md)。下面各日期段落是历史运行快照；新结果不会回写早期失败报告。

- **G1**：抱箱PPO限定任务32/32通过；另有GRAIL楼梯微调新种子100/100、苹果拾取3/8、双手技能DAG两个完整样例及HumanoidMimicGen方法适配样例，分别有独立合同和证据。
- **Vita**：17条/6415帧运动学视频；单碗v12原生PD物理短片段通过；新SONIC真实推理站稳5秒，另完成923帧30.74秒固定站位基线。六物体承重搬运仍未通过，最新接触与静力证据见[SONIC合同](reports/FOURTH_SONIC_CONTRACT_2026-10-27.md)。早期“只有前向/配置缺失”的诊断不能替代当前状态。
- **长椅/鞋**：双Vita及O6手部资产/接口已有实现，完整物理失败；样本已按用户要求停止并放弃。[停止记录](reports/rgb_near_task_stopped_2026-10-27.json)。
- **视觉增强**：RoboRender真实推理已完成，纯生成有一致性失败；当前通过的是明确标注的几何保护混合方案，见[说明](../roborender-tools/README.apple-consistency-fix.zh-CN.md)。

## Vita02a 17 段轨迹演示（2026-10-26）

用户提供的 17 段 Vita02a 轨迹已全部按原始 30 FPS 导出，总计 6415 帧。每段显示两个全身视角和左右五指独立近景，逐帧保留根部位姿及 52 个关节；本地已取回全部 17 段 MP4 与 17 张九宫格关键帧图，并逐文件核对 SHA。[播放全部 17 段](previews/vita17_index.html)、[输入审计](reports/vita17-input-audit.json)、[视频审计](reports/vita17-video-audit.json)、[本地交付回执](reports/vita17-local-delivery.json)、[方法和限制](docs/VITA17_DEMOS.zh-CN.md)。

这些是运动学参考回放，不含箱体/场景、策略驱动和物理接触。新 Vita 检查点严格加载及真实前向计算通过；独立 MuJoCo 试验约 0.98 秒后跌倒，不能认定动力学跟踪成功。[物理试验和缺件清单](reports/vita17-policy-tracking-feasibility.json)。

## G1 双五指抱箱策略训练（2026-10-21，仿真验收通过）

已完成真实 GPU PhysX + PPO 训练，服务器物理 GPU 7、256 并行环境、300轮、4,800次优化器更新和2,457,600次环境转移，浮动 G1 与双五指 ROHand 抱起1 kg动态箱体。本地不训练。权重为 `yz-ai-4090-01.bce:/data/KinoGen/outputs/g1_box_training/train_001/final.pt`。

GPU 6上原定32个独立种子全部通过：初始桌面张手不预抓、真实双手接触、箱底离桌至少10 cm、末端连续保持最短4.92 s、站立且活动期零状态强写/零外力托举。已保留29/32、30/32失败版本及启动预热修复对照。最终视频301帧全部解码，并绑定实际轨迹与检查点来源。[本地抱箱视频](previews/g1_box_policy_final.mp4)、[独立最终验收](reports/g1-box-policy-final-audit.json)、[本地文件校验](reports/g1-box-local-delivery.json)。

该结果仅覆盖固定1 kg箱体和±1 cm位置扰动，方法为IK闭手/抬升参考加PPO残差。模型的50个展开伺服关节是12个手输入的近似动力学实现，自碰撞关闭，并非真实闭链或实机验证；也不等于KinoGen官方完整论文复现。[任务、运行和验收说明](docs/G1_BOX_TRAINING.zh-CN.md)。

## 双五指可见性与腕壳修复（2026-10-21）

第四阶段已导出完整 121 帧、24 FPS 的 G1 双五指三维候选。完整场景原手网格检查中，右手有可用源图观测的 58 帧，缺失轮廓由 9 帧降至 0；左手 79 个可用源图域没有新增缺失。原关节位置、正速度限制和厂家 12→50 耦合保持约束。第 70 帧原约 25.292 mm 的腕壳穿透与第 59 帧条件初值约 7.121 mm 的腕壳穿透已修复。

全片最大手箱采样穿透约 0.235 mm，非手部约 4.898 mm，原 5 mm 必要采样检查没有超界帧；这不是整机器人严格零碰撞。第 56、57 帧仍保留条件穿透，第 59 帧手离箱约 14.4 mm。身体条件标记为 116/121，箱体严格有效仍为 56/121，所有手指三维观测标记保持未知，未恢复经物理验证的双手抓取。

最终独立验收通过当前必要采样、原关节/速度、厂家耦合、所审计图像门槛、动画/FK、四个 GLB 和两段 MP4 全帧解码检查。完整物理重建成功标记仍为 false。最新本地预览：[五指完整场景](previews/user_v4_full_mesh.mp4)、[双视角叠加](previews/user_v4_full_overlay.mp4)。[第四阶段说明](docs/USER_VIDEO_RECONSTRUCTION_V4.zh-CN.md)、[总报告](reports/user-reconstruction-v4-summary.json)、[独立验收](reports/user-v4-independent-audit.json)、[本地文件校验](reports/user-v4-local-delivery.json)。四个 GLB 保留在服务器 `/data/KinoGen/outputs/user_reconstruction_v4/`；不执行本地训练或模型推理。

## 地面与双五指联合修正（2026-10-21）

第三阶段已完成全片 121 帧地面、手腕与五指优化，并真实联合修复第 26 帧身体/手指冲突和第 54 帧穿箱。独立 CPU XML/FK 与原始网格样本检查确认：超过 5 mm 的穿地失败由 120 帧降至 0，手箱穿透失败由 1 帧降至 0；原身体关节位置、正速度上限和厂家手指耦合检查通过。没有移动地板、修改冻结相机/箱体或给失败帧填入插值。

这仍是条件三维候选：箱体仅 56/121 帧严格有效，第 26、54 帧手部轮廓留出仍略高于 4.5 px，左手遮挡和部分离箱状态不能称为可靠双手抓取。新的二维拟合未升级为三维可观测性；未执行动力学、物理策略或实机验证。

完整场景检查进一步发现第 54 帧新手部被箱体挡住，与原 RGB 的清晰五指不符；右手共有 9 个可见轮廓缺失帧。本阶段完整保留为**可见性质量失败的诊断版本**，后续修复使用独立第四阶段文件。[可见性失败证据](reports/user-v3-visible-hand-diagnostic.json)。

最新本地预览：[完整五指场景](previews/user_v3_full_mesh.mp4)、[双视角叠加](previews/user_v3_full_overlay.mp4)。[第三阶段说明与复核命令](docs/USER_VIDEO_RECONSTRUCTION_V3.zh-CN.md)、[最终总报告](reports/user-reconstruction-v3-summary.json)、[独立审计](reports/user-v3-independent-audit.json)、[本地文件校验回执](reports/user-v3-local-delivery.json)。四个完整/静态 GLB 留在服务器 `/data/KinoGen/outputs/user_reconstruction_v3/`，本地不运行 GPU 或训练环境。

## G1 双五指与全身重建候选（2026-10-20）

已将重建模型更换为 G1 身体和左右五指 ROHand：身体 29 个控制量、双手 12 个厂商耦合控制量，共 41 个独立控制量。生成 URDF 的 89 链接组合树及左右五指已检查，原身体 FK 保持一致。见[模型、控制单位和张合预览](docs/G1_FIVE_FINGER.zh-CN.md)。

全 121 帧身体与五指手部优化、接触与速度修正、完整关节网格和箱体三维候选已导出，包含可播放全片的动画 GLB 和两段完整 MP4；身体全片通过当前图像检查，原关节位置与明确给出的正速度上限通过。独立脚部点验证的重投影误差中位数由 0.5405 px 降至 0.1798 px。

数据契约和几何质量分开判定：箱体仅 56/121 帧通过严格几何检查，手部图像有效侧数为左 8 / 右 11，遮挡和外观不一致保留不确定标记。第 54 帧仍有 83.94 mm 手部穿箱；其避碰候选偏离原图，未强行采纳。地面审计发现脚底最深穿地约 30 mm；未恢复可靠的全片双手接触。因此这些是可播放、可检查的三维候选，不是通过物理验收的动作。[第二阶段结果、验收报告和命令](docs/USER_VIDEO_RECONSTRUCTION_V2.zh-CN.md)。

最新本地预览：[五指完整场景](previews/user_v2_contact_full_mesh.mp4)、[双视角图像叠加](previews/user_v2_contact_full_overlay.mp4)；[完整混合单位轨迹](outputs/user_reconstruction_v2/full_trajectory_contact.npz)。按本地只预览视频的要求，121 帧动画 GLB 和三个新静态 GLB 保留在服务器 `/data/KinoGen/outputs/user_reconstruction_v2/`，分别为 `scene_animated_contact.glb` 和 `scene_contact_frame_000/060/120.glb`，已通过服务器文件 SHA 与格式验收。见[本地回传与服务器文件清单](reports/user-local-delivery-v2-contact.json)。

## 用户视频全片处理与三维候选（2026-10-20）

用户双视角抬箱视频的全部 121 帧已在物理 GPU 6 完成 SAM3 分割、TAPNext++ 跟踪和 VDA Large 深度。CPU 导出了 37 个脚/踝共同点的稀疏三维观测、箱体六自由度候选及第 0/60/120 帧彩色点云。最终验收通过 13 个数据文件与 5 个完整预览视频的一致性检查。

第一阶段箱体仅 64/121 帧通过严格几何检查，其余保留无效标记；相对深度的静态条件标定不能证明动态物体的米制精度。水印已作为无效观测处理，源视频未修改。该阶段尚未包含全身/手部关节优化，后续结果见上方第二阶段。[第一阶段完整结果与命令](docs/USER_VIDEO_RECONSTRUCTION.zh-CN.md)、[总报告](reports/user-reconstruction-summary.json)。

2026-10-20 北京时间 17:18:05，服务器 `yz-ai-4090-01.bce:/data/KinoGen` 的物理 GPU 6 完成一键组件验收，状态为 `component_flow_passed`。没有执行训练或视频生成 API。本地 Windows 只保留工具、报告和预览视频。

| 组件 | 实际运行结果 |
| --- | --- |
| Kimodo G1-RP 空文本/约束分支 | 生成 90 帧、30 FPS 的空文本约束动作。目标平滑根节点前进 1 m，平面终点误差 0.01572 m；未执行后处理或物理跟踪 |
| Kimodo G1-RP 原版文字条件 | Llama3 + MNTP 合并 + supervised 适配器，真实特征 `[2,1,4096]` 全部有限；同 seed 42 的两条描述各生成 90 帧、30 FPS，关节数组不同、视频重新解码通过 |
| SAM3 | 同一示例图 image/eager 分割通过，hand 返回 3 个实例，person 返回 1 个实例；实例数不代表准确率评估 |
| SAM3 → TAPNext++ | 从实际手部掩码采样 9 个点，跟踪 12 帧，输出 `[12,9,2]`，97 个可见观测；输入图像与掩码 SHA256 一致性检查通过 |
| WiLoR + MANO | 首帧双手重建，顶点 `[2,778,3]`，模型权重无缺失项 |
| Video Depth Anything Large | 12 帧相对深度 `[12,360,480]`，严格加载检查点通过；Small 另有独立通过记录，深度均非米制 |
| FoundationPose | 官方 refiner/scorer 严格加载、官方预处理和 GPU 前向通过；使用合成 RGB/XYZ 裁剪，未执行真实物体注册 |

验收证据：[本地收集的总报告](reports/server-components/reports/component-flow-latest.json)。各组件报告与服务器运行目录一同列在总报告中，本地副本保留相同相对目录，位于 `reports/server-components/`。

## 原版文字条件接入（2026-10-20 北京时间 19:34）

服务器原版 Llama3 的 14 个文件共 16,069,769,314 字节，全部 SHA256 通过。已从同一底座实际加载两个 LLM2Vec 适配器，每级 448 个适配张量与检查点一致。两条描述的特征余弦相似度约 0.62186，最大元素差 6.53125；相同扩散种子生成的关节数组最大差约 3.61267 m。这些检查证明文字条件实际影响输出，不是语义精度或物理质量指标。

[实际推理报告](reports/server-components/reports/kimodo-text-latest.json)、[下载验收](reports/server-components/reports/llama3-modelscope-server.json)、[文字接入说明](docs/KIMODO_TEXT.zh-CN.md)。预览：[向前走](previews/kimodo_g1_text_1_20261020T113407900253Z.mp4)、[双臂上举](previews/kimodo_g1_text_2_20261020T113407900253Z.mp4)。

单任务调用：

```bash
cd /data/KinoGen
export KINOGEN_CUDA_DEVICES=7
source scripts/activate-server.sh kimodo-text
python scripts/smoke_kimodo_text.py --prompt "A person lifts a box with both hands."
```

该示例只生成身体先验，不创建箱子或保证手物接触。编码后会卸载 8B，再用本次真实特征生成动作，以适配 24 GiB 单卡。

## 复测

VS Code Remote SSH 连接服务器后运行：

```bash
cd /data/KinoGen
KINOGEN_CUDA_DEVICES=6 bash scripts/run_component_flow.sh
```

可改用物理 GPU 7。入口不会使用 GPU 0–5，不安装软件、不调用付费服务。用法与失败判定见 [组件流程](docs/COMPONENT_FLOW.zh-CN.md)。

服务器组件环境为 `.venvs/kimodo`、`.venvs/vision`、`.venvs/sam3`，只读复用已有 GRAIL 的基础包，在项目环境覆盖版本。激活脚本配置进程级 cuDNN 9.6，避开共享目录内 cu13 库冲突；未修改共享环境或驱动。SAM3 图像兼容模式已实测，但 Python 3.10 / Torch 2.5.1 低于上游 README 推荐版本，未验证 SAM3 完整视频跟踪。

## 本地预览

视频已从服务器校验下载到 `previews/`：

- [G1 约束动作](previews/kimodo_g1_kinematic_20261020T091709115959Z.mp4)
- [SAM3 手部分割](previews/sam3_hand_smoke.mp4)
- [SAM3 掩码驱动的点跟踪](previews/tapnextpp_smoke.mp4)
- [WiLoR 双手](previews/wilor_smoke.mp4)
- [相对深度](previews/video_depth_smoke.mp4)

视觉测试来自 WiLoR 官方静态图片及其人工平移序列，没有真实肢体运动或接触过程；WiLoR 与 SAM3 视频只保持首帧结果。G1 动作与视觉测试是独立输入，没有接入视频生成服务。

## 剩余工作与边界

现有 Kimodo 检查点的 Llama3 + LLM2Vec 不能直接换成 Qwen；原版底座与两级适配器现已部署，文字条件分支已通过实际生成，空文本/约束分支也可继续使用。[评估依据](docs/QWEN_COMPATIBILITY.zh-CN.md)。

SAM3 和 MANO 已部署并实测。Llama3 下载、离线路径绑定和文字条件推理均已完成，见 docs/KIMODO_TEXT.zh-CN.md。用户选择免费网页图生视频、手动导出，输入及提示词见 data/video_generation/；Seedance / Ray API 暂不接入。KinoGen 官方集成代码、确切版本与任务资产尚不齐全。本项目已运行身体、五指手部和箱体候选优化，物理策略跟踪未执行，不能把组件或数据契约通过等同于完整论文复现。
