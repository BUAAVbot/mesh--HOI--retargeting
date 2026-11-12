# 可重复的公开组件运行流程

日期：2026-10-20。计算在 `yz-ai-4090-01.bce:/data/KinoGen` 执行，本地 Windows 仅拉取和预览视频。此入口串行验证已公开的组件，并验证 SAM3 掩码进入 TAPNext++ 的实际数据流；它不是 KinoGen 官方端到端程序。

## 当前验证情况

| 组件 | 已实际验证 | 结果范围 |
|---|---|---|
| Kimodo G1-RP | 空文本无条件采样，90 帧、30 FPS；官方 Root2D 起终点约束采样 | 目标前进 1 m，已测平面终点误差约 0.01572 m。动作是运动学结果，未做物理跟踪验证 |
| WiLoR + 用户提供的 MANO | 官方示例图首帧，两只手重建 | 手部顶点形状 `[2,778,3]`；未验证视频手部时序 |
| TAPNext++ | 静态图片平移形成的 12 帧输入，9 个查询点 | SAM3 掩码内选点并跟踪通过，形状 `[12,9,2]`，97 个可见观测；掩码 SHA256 串联校验通过 |
| Video Depth Anything Large / Small | 两个公开检查点均已完成 12 帧实际推理 | 相对深度形状均为 `[12,360,480]`，不能据此宣称获得米制场景深度 |
| FoundationPose | 官方 refiner/scorer 严格加载、官方预处理和真实网络前向 | 输入是明确标注的合成 RGB/XYZ 裁剪；未执行真实物体 registration |
| SAM3 | 独立环境中 image/eager 兼容模式实际推理通过；`hand` 3 个实例，`person` 1 个实例 | 首帧分割与 SAM3→TAPNext++ 串联通过；完整视频 tracker 未测 |

本次一键流程于 2026-10-20 09:18:05 UTC（北京时间 17:18:05）通过，运行目录 `outputs/component_flow/20261020T091703Z-3501415/`。一键入口默认使用 **VDA Large**（`vitl`），Large 与 Small 的独立组件测试均已通过。默认推理尺寸 `--depth-input-size 280` 是便于组件演示的设置；具体变体及尺寸尚未被 KinoGen 官方集成代码确认。

## 服务器执行

在 VS Code Remote SSH 连接该服务器后：

```bash
cd /data/KinoGen
bash -n scripts/run_component_flow.sh
KINOGEN_CUDA_DEVICES=6 bash scripts/run_component_flow.sh
```

也可选物理 GPU 7。脚本默认 GPU 6，并拒绝使用物理 GPU 0–5；不会继承其他任务遗留的 `CUDA_VISIBLE_DEVICES=0`。各组件串行执行，每步用独立子 shell 激活本项目环境，避免混用 Python 包。脚本不执行安装、下载、训练、付费 API 或授权操作。

若只需复测已验证的 Small 深度模型：

```bash
KINOGEN_CUDA_DEVICES=7 KINOGEN_DEPTH_ENCODER=vits bash scripts/run_component_flow.sh
```

执行前必须已有 `.venvs/kimodo`、`.venvs/vision`、`.venvs/sam3`、本项目权重及 `data/smoke_inputs/test1.jpg`。其中测试图是 WiLoR 官方示例，非用户实拍任务素材。

## 执行顺序与失败判定

1. Kimodo 使用空文本和官方运动学约束生成 1 m 前进动作，输出真实动作数组及骨架预览。
2. FoundationPose 使用官方检查点完成严格加载、合成 crop 的官方 RGB/XYZ 归一化及两个网络前向。
3. SAM3 对同一官方示例图首帧运行 `hand` 和 `person` 提示，保存实例数组、合并掩码和静态叠加预览。
4. WiLoR + MANO 重建首帧双手；VDA 估计平移序列的相对深度；TAPNext++ 从实际 SAM3 `hand_mask.png` 内选择 9 个点并跟踪 12 帧。

脚本检查 SAM3 手部掩码至少包含 9 个前景像素；TAPNext++ 还会检查缩放后前景数量。空掩码或任一组件失败会使流程退出，不会用规则网格冒充语义查询点。最终验收比较掩码 SHA-256，并确认 SAM3 和视觉序列来自同一图像。

**SAM3 运行采用明确开启的兼容模式**：`--allow-unsupported-runtime`。目前环境 Python 3.10 / PyTorch 2.5.1 低于该快照 README 列出的 Python ≥3.12、PyTorch ≥2.7、CUDA ≥12.6；本次 image/eager 图像分割已实际通过，不代表完整视频 tracker 或所有官方功能均兼容。每次仅将实际完成并产生有限输出的推理记作通过，报告保留版本信息。[官方运行要求](https://github.com/facebookresearch/sam3#installation)

## 报告和预览

- 每次独立日志和报告：`outputs/component_flow/<UTC时间>-<进程号>/`。
- 完整组件验收通过后才写入该次 `flow.json`，并更新 `reports/component-flow-latest.json`。
- 中间模型数组和预览：`outputs/vision_smoke/`、`outputs/sam3_smoke/`、`outputs/previews/`。
- `reports/sam3-smoke.json`、`reports/vision-smoke-all.json` 是最近一次单组件报告；历史验收请查看每次运行目录中的副本。失败运行不会刷新“最近成功”的 flow 报告，应结合 `checked_at` 和本次日志判断。

本地 PowerShell 可列出服务器预览并下载指定文件：

```powershell
cd D:\workspace\SyntheticRobotDemonstrations\KinoGen
.\.venv\Scripts\python.exe scripts\pull_preview.py --list
.\.venv\Scripts\python.exe scripts\pull_preview.py tapnextpp_smoke.mp4 --open
```

下载后位于本地 `KinoGen/previews/`。不同运行的通用视觉预览文件名相同；若本地已有内容不同的同名文件，先重命名本地旧视频再拉取，下载器会拒绝覆盖。Kimodo 预览文件名含时间戳，可在列表中选择。

## 结果的适用范围

视觉输入是官方静态示例图向右及轻微向下平移形成的序列。它没有真实肢体运动、遮挡变化或接触过程；不能用于宣称真实动作视频重建或论文精度。WiLoR 和 SAM3 仅使用首帧；SAM3 预览把静态分割结果保持三秒，不代表 SAM3 视频跟踪。

Kimodo 的新生成动作与视觉测试图是各自独立的输入，当前没有把该动作送入视频生成接口。Seedance / Ray API 未配置，KinoGen 官方集成尚未发布，手物体优化、任务相机标定、完整机器人手部装配及物理策略跟踪不在此组件验收范围。

本一键入口的 Kimodo 阶段仍使用空文本和根节点约束。原版 Llama3 + 两级 LLM2Vec 已另外部署，并在专用 `kimodo-text` 环境完成文字条件采样；使用入口见 [文字接入说明](KIMODO_TEXT.zh-CN.md)。Qwen 系列不能直接替换现有检查点的文本表示，具体原因见 [Qwen 兼容性评估](QWEN_COMPATIBILITY.zh-CN.md)。
