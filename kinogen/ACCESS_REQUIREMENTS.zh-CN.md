# 账号、授权与上游缺口

核验日期：2026-10-20。SAM 3、Llama 3 与 KinoGen 发布状态的本次复核记录见 `sources/access-check-2026-10-20.json`。没有登录用户账号、接受协议、购买服务或调用付费推理。

后续准备进展（2026-10-20）：用户已自行注册、激活相关人体/手部模型账号并下载官方文件。SMPL、SMPL-X、MANO 必要模型已解压部署到本地项目和 `yz-ai-4090-01.bce` 服务器；MANO 右手现位于 `weights/wilor/mano_data/MANO_RIGHT.pkl`，已绑定准备配置。全部压缩包 CRC、部署文件 SHA256 和服务器四个模型的 CPU 加载/前向检查通过，证据见 `reports/licensed-models-deployment.json`、`reports/licensed-models-server.json`。用户反馈 SAM 3、Llama 3 已申请，尚未核实其审批结果。

SAM 3 后续进展（2026-10-20）：已按用户指定的 [ModelScope facebook/sam3](https://www.modelscope.cn/models/facebook/sam3) 下载固定版本 `96f3e1b404ba14f2cfac60ee6ae87c269a7b7923`，服务器主权重位于 `/data/KinoGen/weights/sam3/sam3.pt`。大小 3,450,062,241 字节，SHA256 与原始 Hugging Face `sam3.pt` 一致；配套 10 个文件哈希、PyTorch ZIP CRC、CPU 安全读取 1465 个张量均通过。记录见 `reports/sam3-modelscope-server.json`。主权重保存在服务器，本地保留配套文件、下载断点和回执。独立 `.venvs/sam3` 环境的 image/eager 图像分割兼容模式已实际通过：官方 WiLoR 示例图首帧检测到 `hand` 3 个实例、`person` 1 个实例；报告见 `reports/sam3-smoke.json`。环境 Python 3.10 / PyTorch 2.5.1 低于上游推荐版本，完整视频 tracker 未测，SAM3→TAPNext++ 掩码选点串联已通过本次一键组件验收，见 RUNTIME_STATUS.zh-CN.md。使用已部署 checkpoint 无需再从 Hugging Face 下载该文件。

组件进展（2026-10-20）：WiLoR 已使用用户提供的 MANO 实际重建双手，输出顶点形状 `[2,778,3]`；Video Depth Anything Large 和 Small 均已完成 12 帧推理，输出相对深度形状 `[12,360,480]`。Llama 3 底座仍仅影响 Kimodo 的文字语义条件分支，不阻止已验证的空文本无条件采样或仅运动学约束采样。Qwen 系列不能直接替换该检查点训练时使用的 Llama3 + LLM2Vec 表示，见 [兼容性评估](docs/QWEN_COMPATIBILITY.zh-CN.md)。

| 项目 | 需要补充的内容 | 官方入口 | 核验依据 |
| --- | --- | --- | --- |
| SAM 3 权重 | 已通过用户指定的 ModelScope 来源部署，首帧图像分割推理通过；若选择从 Hugging Face 重新下载，仍需相应账号批准与 token | [ModelScope](https://www.modelscope.cn/models/facebook/sam3)、[HF 模型页](https://huggingface.co/facebook/sam3) | 已部署文件与原始 `sam3.pt` 的 SHA256 一致；下载见 `reports/sam3-modelscope-server.json`，图像兼容模式推理见 `reports/sam3-smoke.json`；完整视频 tracker 未测 |
| Kimodo 文本编码器的 Llama 3 底座 | 已从同一原版 ModelScope 镜像完整部署，14 个文件 SHA256 通过，原始两级 LLM2Vec + Kimodo 文字条件推理通过；当前运行无需 HF token，适用许可随文件保留 | [ModelScope 原版](https://www.modelscope.cn/models/LLM-Research/Meta-Llama-3-8B-Instruct) | 下载验收见 reports/server-components/reports/llama3-modelscope-server.json；真实编码与动作见 reports/server-components/reports/kimodo-text-latest.json，使用方法见 docs/KIMODO_TEXT.zh-CN.md |
| MANO 右手模型 | 用户已自行注册并下载，部署及 WiLoR 双手前向已通过；后续使用仍遵循适用许可 | [注册](https://mano.is.tue.mpg.de/register.php)、[许可](https://mano.is.tue.mpg.de/license.html) | [WiLoR 官方安装说明](https://github.com/rolpotamias/WiLoR#installation)要求单独获取 MANO。已部署文件及模型前向见 `reports/licensed-models-deployment.json`、`reports/licensed-models-server.json`；未替用户接受条款 |
| Seedance 2.0 视频生成 | 官方服务账号、模型开通、API Key 和可用额度 | [字节官方模型页](https://seed.bytedance.com/en/seedance2_0)、[火山引擎文档](https://docs.volcengine.com/docs/ark/seedance-2-0?lang=zh) | 服务访问条件见官方文档；未找到可下载的官方本地权重 |
| Ray 3.2 关键帧视频再生成 | Luma 开发者账号、API Key、模型权限及额度 | [Luma API](https://lumalabs.ai/api)、[Ray 3.2](https://lumalabs.ai/news/introducing-ray-3-2) | 作为托管服务使用，未找到可下载的官方本地权重 |
| Isaac Sim 二进制运行时与 Kit 组件 | 安装运行时前确认适用的 Omniverse Kit 与附加材料条款；若使用 NGC 容器，按所选渠道配置访问 | [源码](https://github.com/isaac-sim/IsaacSim)、[许可说明](https://docs.isaacsim.omniverse.nvidia.com/latest/common/legal.html) | 公开源码及其 Git LFS 下载不需要账号；运行时与附加组件许可另计。本任务未安装运行时，尚未定位到现有可复用安装，也没有替用户接受 EULA |

## 不属于账号问题的缺口

- KinoGen 算法仓库、集成代码、优化超参数、固定的依赖版本和具体 checkpoint 选型尚待官方发布。
- 论文的标定相机、初始状态、任务场景/物体、G1 与 ROHand 的组装版本和训练策略未随项目公开。厂商公开 URDF 只能作为基础资产，不能宣称与论文完全一致。
- 下载过程中出现了跨 GitHub、Hugging Face、Google Drive 的 TLS 重置及连接超时。这些是网络传输问题；下载清单和报告分别记录，不列为授权缺失。

## 公开下载仍有许可条件

Kimodo G1 权重随附 NVIDIA 模型许可；Video Depth Anything Large 标注 CC-BY-NC-4.0，Small 标注 Apache-2.0；WiLoR 官方仓库说明模型采用 CC-BY-NC-ND，并依赖 MANO、Ultralytics 的各自许可。FoundationPose 原始代码和数据采用 NVIDIA Source Code License。公开可下载不等于所有用途均无限制，使用时以下载文件中的具体许可为准。

后续提供访问权限时，请在本机配置凭证，不要把 token 写进仓库、配置 JSON 或聊天消息。

视频生成方案更新（2026-10-20）：用户选择免费网页图生视频并手动导出，不需要现在配置 Seedance / Ray API 账号或付费额度。参考输入及提示词在 data/video_generation/；需保留网页原始输出、双视角布局与实际生成参数。此方式不保证复现论文所用服务的效果；多关键帧修正仍取决于所选网页的功能。

原版文字条件后续进展（2026-10-20）：Llama3 底座及配置/tokenizer已在服务器完整校验；专用 kimodo-text 环境的双向注意力、两级适配器加载、真实编码和90帧动作采样均通过。Llama3 当前已不再是本项目组件运行的缺失项。
