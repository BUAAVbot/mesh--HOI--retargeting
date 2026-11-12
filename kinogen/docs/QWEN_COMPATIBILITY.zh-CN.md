# Kimodo 的 Llama 3 与 Qwen 替换评估

检查日期：2026-10-20。检查对象是本目录固定的 Kimodo 源码提交 `e1c219c99b4b94ba48be1fb6a1cbc64ecee4c954`，以及已下载的 G1-RP / G1-SEED 模型。本文不表示 KinoGen 官方完整流程已运行。

后续运行进展（2026-10-20）：原版 Llama3 已从核验一致的 ModelScope 镜像部署；原版 LLM2Vec 与 Kimodo 文字条件采样已实测通过，无需采用 Qwen 替换路线。原版入口见 [文字接入说明](KIMODO_TEXT.zh-CN.md)。下文对 Qwen 表示空间的兼容性结论保持适用。

## 结论

**Qwen 可以作为新方案的文本编码器，但不能直接替换已发布 Kimodo G1 权重所用的 Llama 3 + LLM2Vec。** 下载 Qwen 后改模型名称、补齐维度，均不足以恢复原模型的文本控制。应继续运行无文本条件或仅运动学约束的公开 Kimodo 分支；文本驱动的原版结果仍需原编码器，或者另做经过验证的表示对齐/模型微调。

这与 Qwen 的语言能力强弱无关：预训练动作网络已经针对特定编码空间学习了输入投影。换编码器后，即便输出尺寸相同，坐标及分布也不相同。这是根据公开接口与检查点作出的工程判断，不是厂商发布的兼容性承诺。

## 本地证据

- `dependencies/kimodo/kimodo/model/load_model.py` 的 `TEXT_ENCODER_PRESETS` 只有 `llm2vec`，指定 `McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp`、`...-mntp-supervised` 和 `llm_dim=4096`。
- `weights/kimodo-g1-rp-v1/config.yaml` 与 SEED 配置要求 `llm_shape: [1, 4096]`。wrapper 返回的批量特征形状是 `[B, 1, 4096]`，另有文本长度/掩码。
- 直接读取 RP safetensors 头部可见两组已经训练的投影：`denoiser.backbone.body_model.embed_text.weight`、`denoiser.backbone.root_model.embed_text.weight`，形状均为 `[1024, 4096]`。本次仅读取文件头，没有加载或执行模型。
- 已下载 MNTP 的 `adapter_config.json` 明确指定 `base_model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct`。这两个 LoRA 不是任意底座通用的权重。
- 本地 vendored LLM2Vec 的 `_get_model_class` 虽保留 `Qwen2Config` 分支，但其导入的 `models/bidirectional_qwen2.py` 并不在该 Kimodo 快照中。因此还存在源码层面的缺失；不能仅凭函数中的分支声称能直接使用 Qwen。

## 候选比较

| 候选 | 已核实的输出/架构事实 | 对既有 Kimodo 的结论 |
|---|---|---|
| Qwen2-7B-Instruct | 官方配置 hidden_size 为 3584，Qwen2 架构 | 输入维度与 4096 不一致；原 Llama LoRA 也不匹配。须改接口并重新对齐/训练 |
| Qwen2.5-7B-Instruct | 官方配置 hidden_size 同为 3584 | 同上。补零或未经训练的线性层不能恢复文本语义兼容性 |
| Qwen3-Embedding-8B | 官方提供最长 4096 维句向量；示例使用末 token 池化与归一化 | 尺寸可以相同，但不是 Llama3-LLM2Vec 的表示空间；仍不能直接替换 |
| LLM2Vec + Qwen2 | LLM2Vec 上游明确支持 Qwen2 架构 | 支持构建编码器，不代表 Kimodo 动作检查点已训练适配它 |

Qwen3-Embedding 的事实来自 [Qwen 官方模型卡](https://huggingface.co/Qwen/Qwen3-Embedding-8B)。Qwen2 / Qwen2.5 的尺寸分别见 [Qwen2 官方配置](https://huggingface.co/Qwen/Qwen2-7B-Instruct/blob/main/config.json)、[Qwen2.5 官方配置](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json)。LLM2Vec 的架构支持与训练方式见 [LLM2Vec 官方仓库](https://github.com/McGill-NLP/llm2vec)。

若以后研究 Qwen 替换，可用配对的原 LLM2Vec / Qwen 文本嵌入学习映射，并在动作条件响应上验证，或用对应文本—动作数据微调 Kimodo 条件网络。前者仍需要能合法生成原编码器嵌入，后者需要数据和训练；当前环境准备不自动启动这些训练。

## 无需 Llama 下载即可执行的真实分支

官方 `Kimodo._generate` 已实现空文本逻辑：对于空字符串，将文本特征清零、文本长度改为 0，并产生全 False 文本掩码；CFG 的无文本分支也使用零特征与无效文本掩码。可以保留官方动作网络、扩散采样器及运动学后处理，仅在独立适配脚本中省掉“先加载 Llama 再丢弃空文本编码”的额外步骤。

实施方式：

1. 用 `load_model(..., text_encoder=empty_only_encoder)` 注入一个只允许空字符串的编码器。这个参数在官方加载器中会跳过实际文本编码器的实例化。
2. 编码器返回 `[B,1,4096]` 全零特征以及每条长度 0。对任何非空输入立即报错，禁止伪装为文本控制。
3. 通过公开 `model(prompts="", ...)` 接口运行，或直接传入 `_generate` 的零特征及全 False 文本掩码。后者是内部接口，可能随源码版本变化。
4. 无约束时标记为 **unconditional / no-text motion generation**；有约束时标记为 **kinematic-constraint-only generation**。保存真实动作 NPZ，随后渲染视频；说明它是 Kimodo 子流程验证。

这里的零向量是官方定义的“无文本”条件，不是用随机向量假冒文本嵌入。`cfg_type="nocfg"` 只表示不执行 CFG 混合，本身并不会自动禁用文字；必须同时使用空文本/全无效文本掩码。

固定版本依据：[官方模型加载器](https://github.com/nv-tlabs/kimodo/blob/e1c219c99b4b94ba48be1fb6a1cbc64ecee4c954/kimodo/model/load_model.py)、[官方采样与空字符串处理](https://github.com/nv-tlabs/kimodo/blob/e1c219c99b4b94ba48be1fb6a1cbc64ecee4c954/kimodo/model/kimodo_model.py)、[官方 CFG 分支](https://github.com/nv-tlabs/kimodo/blob/e1c219c99b4b94ba48be1fb6a1cbc64ecee4c954/kimodo/model/cfg.py)。

## 其他可能路径的检查结果

- 默认文本编码服务地址是 `http://127.0.0.1:9550/`，对应自行启动的 LLM2Vec 服务，不是可免除模型权限的公共托管接口。
- Kimodo 有磁盘嵌入缓存逻辑，但本次下载的官方快照未提供可直接使用的文本嵌入缓存。不能把已有动作示例当作当前新采样结果。
- 标准文本流程的 gated 依赖仍由 [NVIDIA 官方安装说明](https://research.nvidia.com/labs/sil/projects/kimodo/docs/getting_started/installation.html)明确列出。本文没有下载 Qwen、接受任何新许可，或修改依赖快照。
