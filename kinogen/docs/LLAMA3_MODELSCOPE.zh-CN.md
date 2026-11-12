# ModelScope 的 Llama 3 底座核对

核验日期：2026-10-20。结论：**ModelScope 有 Kimodo 所需的原始 `Meta-Llama-3-8B-Instruct` Transformers 模型镜像，可作为同模型的下载来源；无需改用 Qwen。后续已完成服务器 14 个文件的完整 SHA256 校验，并实测原版 LLM2Vec→Kimodo 文字条件采样；[实际接入说明](KIMODO_TEXT.zh-CN.md)。以下保留最初的镜像核对过程与取证范围。**

来源是 [ModelScope：LLM-Research/Meta-Llama-3-8B-Instruct](https://www.modelscope.cn/models/LLM-Research/Meta-Llama-3-8B-Instruct)。本次固定 revision 为 `e9f7e7d3fa08b550ea228e38bb5501d35be92c0d`。选择的是根目录四个 `model-0000*-of-00004.safetensors`，不是量化版、GGUF、Llama 3.1/3.2，也不是 `original/consolidated.00.pth` 那套重复的原生格式。

## 确认了什么

1. ModelScope 固定版本文件清单可匿名读取。四个 Transformers 权重分片共 **16,060,556,376 字节，约 14.958 GiB**；包含配置、tokenizer、说明与许可的根目录 14 个文件合计 16,069,769,314 字节。
2. 四个分片在 ModelScope 发布的大小及 SHA-256，与 [Meta 原始 Hugging Face 模型](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)公开 API 元数据逐项一致。HF 核对版本为 `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`。这属于两端发布元数据一致性核对；尚未完整下载分片，自然也尚未独立计算整文件哈希。
3. 每个大分片只进行 `Range: bytes=0-15` 只读请求，四次均收到 HTTP 206、精确匹配的 `Content-Range` 和 16 字节响应。大权重实际读取合计 **64 字节**，证明本次网络可匿名按范围读取这些文件；没有下载完整模型。
4. 小文件 `config.json`、`tokenizer_config.json`、分片索引、LICENSE、USE_POLICY 已读出并核对文件 SHA-256。完整 filetree、tokenizer 发布大小/哈希、Range 结果及原始 HF 元数据保存在 `reports/llama3-modelscope-check.json`。

分片发布信息：

| 文件 | 字节数 | SHA-256（ModelScope 与原始 HF 发布值一致） |
|---|---:|---|
| `model-00001-of-00004.safetensors` | 4,976,698,672 | `d8cf9c4d0dd972e1a2131bfe656235ee98221679711a3beef6d46dadf0f20b5c` |
| `model-00002-of-00004.safetensors` | 4,999,802,720 | `8d4782b4a69ef03845159ce1a15e272aadaaf134dc138d68f616098e8531729c` |
| `model-00003-of-00004.safetensors` | 4,915,916,176 | `3acdd690e65c24f42a24581b8467af98bd3ca357444580f8012aacd2bd607921` |
| `model-00004-of-00004.safetensors` | 1,168,138,808 | `67e9ad31c8c32abf3a55ee7fc7217b3ecb35fd3c74d98a5bd233e0e4d6964f46` |

## 为什么适用于已有 LLM2Vec 适配器

已部署的 MNTP 适配器 `adapter_config.json` 声明底座是 `meta-llama/Meta-Llama-3-8B-Instruct`。其配套配置与镜像配置的九项架构参数一致：hidden size 4096、intermediate size 14336、32 层、32 个注意力头、8 个 KV 头、词表 128256、Llama 模型类型、RoPE theta 500000、上下文长度 8192。

镜像是同一底座的另一个文件来源，而不是用另一套语义表示替换动作模型输入。已部署的 MNTP 及 supervised 两个适配器仍应按原 LLM2Vec 流程叠加；仅运行普通 Llama 聊天模型不会得到 Kimodo 训练时的文本嵌入。[LLM2Vec 官方实现](https://github.com/McGill-NLP/llm2vec)、[官方 supervised 适配器](https://huggingface.co/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised)

后续完成条件是：下载固定 revision 的四个分片及必要配置/tokenizer并逐文件校验；在项目工作配置中把底座解析到本地目录，防止适配器加载器仍自动请求 gated HF 仓库；实际加载原底座及两级适配器，验证有限的 `[B,1,4096]` 文本特征，再验证文字条件的 Kimodo 采样。上述“可作为同模型来源”的结论不能代替这一步实际运行验收。

## 许可与后续状态

匿名 HTTP 可读取不等于没有许可条件。该镜像随附 Meta Llama 3 Community License 与使用政策，后续使用仍遵循这些条款；从镜像获取也不会自动确认用户的 Hugging Face 审批状态。本次没有登录账号、提交审批、点击接受协议或启动模型推理。[Meta 官方许可](https://github.com/meta-llama/llama3/blob/main/LICENSE)

当前应把缺口分开看：

- **文字条件接入已完成**：底座完整部署、LLM2Vec 本地绑定、实际编码与 Kimodo 动作采样均已通过。使用专用 kimodo-text 环境，原版双向编码与两级适配器均验证；语义准确率和物理有效性未作定量评估。
- **视频输入与生成阶段**：公开视频生成 API 未配置。若用户在网页生成视频后手动导出，可以把文件作为后续输入素材；这与完成论文中自动化视频生成集成不同。
- **完整论文流程**：KinoGen 官方集成尚未发布，任务相机、物体几何/初始状态、具体手物体优化与物理跟踪仍需对应代码和资产。公开组件自检不能代替这些环节。

本文件只核对 Llama 3 下载来源。其余组件运行情况以最新组件流程报告及 `docs/COMPONENT_FLOW.zh-CN.md` 为准。
