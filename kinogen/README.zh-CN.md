# KinoGen：组件、G1/Vita 任务与物理重建工作区

> **统一入口（2026-10-27）**：[整个工作区总览](../README.zh-CN.md) · [流程模块地图](../docs/PIPELINE_MODULES.zh-CN.md) · [论文复现清单](../docs/PAPER_REPRODUCTIONS.zh-CN.md) · [G1 / Vita 模型与控制器](../docs/ROBOTS_G1_VITA.zh-CN.md)。下文保留早期准备记录；当前任务状态优先使用总索引和具体冻结报告。

最新已核对：G1抱箱PPO通过限定任务32/32；GRAIL楼梯微调通过新种子100/100；Vita单碗PD短片段通过；Vita新SONIC已有真实推理、稳定站立和923帧全长固定站位基线，但六器皿承重搬运未通过。双Vita长椅/鞋样本已按用户要求停止并放弃，模型、碰撞和审计工具保留。详见上述专项索引。

公开组件核验日期：2026-10-20（北京时间）；抱箱训练更新于2026-10-21。服务器已配置公开组件环境，完成真实模型推理与动作预览，并开始独立 G1 抱箱物理策略训练。**按前述核验，当前工作尚非完整论文复现。** 视频生成由用户在网页完成。

最新结果见 [运行状态](RUNTIME_STATUS.zh-CN.md)，服务器复测入口见 [组件流程](docs/COMPONENT_FLOW.zh-CN.md)。本地只下载和预览视频，GPU 推理在 `yz-ai-4090-01.bce:/data/KinoGen` 执行。

2026-10-21 已按用户要求完成 **G1 双五指抱箱 PPO 训练与仿真验收**，使用服务器既有 Isaac Sim 4.5 / Isaac Lab 2.1，完成300轮、4,800次优化器更新。固定1 kg箱体、±1 cm位置扰动下32/32回合通过，末端连续抱稳最短4.92 s。[实际物理轨迹视频](previews/g1_box_policy_final.mp4)、[完整验收](reports/g1-box-policy-final-audit.json)、[方法、复测和边界](docs/G1_BOX_TRAINING.zh-CN.md)。这是IK参考加PPO残差的独立仿真基线，尚非官方完整复现或实机成功。

用户生成的双视角抬箱视频已处理全部 121 帧：SAM3 分割、TAPNext++ 跟踪、VDA Large 深度、脚部稀疏三角化、箱体六自由度候选与三时刻点云均已导出。水印仅排除观测，原视频保留。[第一阶段结果](docs/USER_VIDEO_RECONSTRUCTION.zh-CN.md)。第二阶段完成身体与五指候选和完整动画，[保留原阶段结果](docs/USER_VIDEO_RECONSTRUCTION_V2.zh-CN.md)。第三阶段地面与五指修复保留为[可见性质量失败的诊断版本](docs/USER_VIDEO_RECONSTRUCTION_V3.zh-CN.md)。最新第四阶段修复了 9 个右手缺失轮廓和腕壳超界穿透，仍保留微小穿透、缺测和条件状态；[完整说明及预览](docs/USER_VIDEO_RECONSTRUCTION_V4.zh-CN.md)、[最新总报告](reports/user-reconstruction-v4-summary.json)。

重建目标现已按用户要求换成 **G1 + 左右五指 ROHand**。组合 URDF、官方连杆耦合、单位和实际张合预览见 [五指模型说明](docs/G1_FIVE_FINGER.zh-CN.md)。

新增原版文字条件路径：Llama3 底座约 16.07 GB、14 个文件已完整部署并通过 SHA256，原版 LLM2Vec 两级适配器与 Kimodo 的真实文字条件采样已通过。使用专用 `.venvs/kimodo-text` 环境，避免与其他组件的 Transformers 版本混用；[启动及离线重建说明](docs/KIMODO_TEXT.zh-CN.md)。视频生成由用户在网页完成。

官方来源：

- [项目页](https://kinogen-humanoid.github.io/)：2026-10-20 保存的核验快照显示 `Code coming soon`；本索引整理未重新核验上游发布状态。
- [GitHub 账号仓库列表](https://github.com/KinoGen-humanoid?tab=repositories)：核验时仅有项目展示网站。
- [论文 v1](https://arxiv.org/abs/2610.20552v1)、[全文](https://arxiv.org/html/2610.20552v1)。下载副本与 SHA256 记录见 `sources/manifest.json`。

## 已配置内容

- `.venv`：本地 Python 3.12 资料与诊断工具环境，版本见 `requirements-tools.lock.txt`。服务器组件环境及实际版本见运行报告；这些版本不是 KinoGen 官方锁定版本。
- `configs/reproduction.json`：论文明确披露的参数、运行阶段、待补充资产；不是官方可执行配置。
- `scripts/fetch_sources.py`：手动更新官方网页、论文和公开仓库清单，并记录下载时间、哈希及代码发布提示。
- `scripts/check_setup.py`：检查算法仓库、硬件、输入与权重，生成 `reports/preflight.json`。缺少前置条件时退出码为 2，不把工具自检成功当成复现成功。
- `dependencies/`：官方依赖源码及厂商基础资产。每个快照的提交号、文件范围与未下载内容见 `manifests/sources/`。
- `weights/`：公开权重、配套配置和统计数据；`.part`、`.chunks` 是未完成断点，不可作为模型使用。
- [账号与授权清单](ACCESS_REQUIREMENTS.zh-CN.md)：SAM 3、Llama 3 底座、MANO、生成视频服务及 Isaac Sim 附加条款。
- [准备状态](PREPARATION_STATUS.zh-CN.md)：由本地验证脚本生成，区分完整文件和未完成传输。

## 本机使用

从本目录打开 PowerShell：

```powershell
# 已创建 .venv 后可直接运行
.\.venv\Scripts\python.exe .\scripts\check_setup.py

# 网络可用时，手动重新核验发布状态
.\.venv\Scripts\python.exe .\scripts\fetch_sources.py

# 核验已下载的公开权重和源码，生成准备状态报告
.\.venv\Scripts\python.exe .\scripts\verify_downloads.py

# 通过 SSH 核验服务器文件，并保存服务器报告到本地 reports/
.\.venv\Scripts\python.exe .\scripts\check_server_preparation.py

# 网络可用时补齐公开权重（自动利用断点）
$env:KINOGEN_DOWNLOAD_WORKERS = '4'
.\.venv\Scripts\python.exe .\scripts\prepare_dependencies.py weights
.\.venv\Scripts\python.exe .\scripts\prepare_dependencies.py weights --plan configs/downloads-extra.json
.\.venv\Scripts\python.exe .\scripts\fetch_foundationpose.py

# 恢复逐文件源码下载；每个文件均核对官方 Git blob 哈希
.\.venv\Scripts\python.exe .\scripts\fetch_source_files.py kimodo wilor sam3 foundationpose dno-reference rohand-urdf unitree-ros isaacsim

# Isaac Sim 已保存完整 ZIP，按固定提交校验并展开，随后补齐公开 LFS 素材
.\.venv\Scripts\python.exe .\scripts\stage_isaacsim_archive.py
.\.venv\Scripts\python.exe .\scripts\fetch_isaacsim_lfs.py

# 需要重新创建工具环境时，指定已有 Python 3.12
.\scripts\setup_tools.ps1 -Python 'C:\path\to\python.exe'
```

## 服务器计算与本地视频预览

计算目标为 VS Code Remote SSH 配置中的 `yz-ai-4090-01.bce`，项目目录 `/data/KinoGen`。已探测到 8 张 RTX 4090（每张约 24 GiB）、驱动 550.163.01 和 CUDA Toolkit 12.4。具体部署与校验回执见 `manifests/deployment/`；部署过程不启动训练。

服务器准备脚本 `scripts/setup-server.sh` 创建工具环境和 GPU 基线；`scripts/setup-component-envs.sh` 从已校验的离线 wheelhouse 创建 `.venvs/kimodo`、`.venvs/vision`、`.venvs/sam3`。组件环境通过 system-site-packages 只读复用已有 GRAIL 的 Torch 2.5.1+cu121，并在项目环境覆盖所需 Python 包。已有 GRAIL 环境未改写。它们不是完全独立的环境，也不是官方 KinoGen 运行时。

已完成 Kimodo、WiLoR + MANO、SAM3 图像分割、TAPNext++、Video Depth Anything 和 FoundationPose 网络推理。具体输入、检查项和结果以组件报告为准。后续已复用服务器既有 Isaac Sim 4.5 / Isaac Lab 2.1 完成G1抱箱训练，并使用MuJoCo完成Vita单碗、SONIC站立及其他物理试验；各任务环境与边界见[G1/Vita索引](../docs/ROBOTS_G1_VITA.zh-CN.md)。

VS Code 连接该 SSH 主机后打开 `/data/KinoGen`，可使用项目中的解释器设置。服务器终端激活方式：

```bash
cd /data/KinoGen
source scripts/activate-server.sh tools
# 单组件推理前激活相应环境：kimodo / vision / sam3
KINOGEN_CUDA_DEVICES=6 source scripts/activate-server.sh vision
# 或直接运行完整组件验收入口
KINOGEN_CUDA_DEVICES=6 bash scripts/run_component_flow.sh
```

激活脚本同时设置进程级 cuDNN 9.6 库，绕开共享环境中的 cu13 库冲突；没有修改全局驱动或共享环境文件。仅选择 VS Code 的 Python 解释器不会自动应用这些环境变量，推理应通过该脚本启动。默认使用物理 GPU 6，可改为 7。

本地只预览视频，不安装训练、CUDA、MuJoCo 或 Isaac Sim 运行环境。服务器已导出视频至 `outputs/previews/`，本地运行：

```powershell
# 查看服务器已导出的视频
.\.venv\Scripts\python.exe .\scripts\pull_preview.py --list
# 下载、校验并用本机默认播放器打开
.\.venv\Scripts\python.exe .\scripts\pull_preview.py tapnextpp_smoke.mp4 --open
```

视频保存在本地 `previews/`，已有 G1 约束动作、分割、点跟踪、手部和相对深度预览。视觉输入为官方静态示例及其平移序列，不能作为真实任务动作重建结果。

## 完整复现仍需补齐

1. 官方算法仓库、提交号、许可、依赖锁定、运行命令和任务配置。
2. 任务特定的场景/物体碰撞、相机与初始状态标定，以及各控制器的资产合同。G1+ROHand组合、Vita52和新版O6资产均已有实现；应按任务选择并核验，不能再笼统列为缺机器人模型。
3. 论文涉及的生成服务及先验模型访问条件和权重，详见配置中的阶段列表。视频服务可能需要账号与费用，当前未调用。
4. 官方任务的完整输入、评价与配置对齐。当前已有多个自建/适配任务的训练及验收配置，覆盖范围见[论文清单](../docs/PAPER_REPRODUCTIONS.zh-CN.md)，不能自动外推为官方完整复现。

代码发布后，将算法仓库放在 `upstream/`，确认 `configs/reproduction.json` 中已填的公共候选权重版本，并补齐其余数据路径。检查器只检查文件存在，最终仍须按官方说明验证权重版本、模型加载、最小任务和指标。

公开模型准备的是上游可获取版本，不能认定为论文所用的确切 checkpoint。Kimodo 同时保留 G1-RP/G1-SEED；深度模型准备相对深度 Large 与 Small；TAPNext++ 采用标准 256 checkpoint，512 备选单列 `configs/downloads-optional.json`；FoundationPose 分开保存原始 PyTorch 和官方 ONNX 导出。DNO 是优化方法参考代码，不另下载与此任务无关的人体 MDM 模型。

Unitree 快照选择了 `robots/g1_description` 与根目录说明。ROHand 原厂URDF保留在依赖目录，另已完成项目的G1双五指组合与训练派生模型；这不自动等价于论文特定资产。[组合和控制单位](docs/G1_FIVE_FINGER.zh-CN.md)。Kimodo 30 fps、生成视频24 fps及物理控制周期需要显式对齐，不能直接按相同帧号拼接。
