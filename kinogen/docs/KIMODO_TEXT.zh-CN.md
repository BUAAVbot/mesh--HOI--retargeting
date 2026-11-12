# 原版 Llama 3 接入 Kimodo 的文字条件验证

日期：2026-10-20。底座采用 ModelScope 上同一原版 `Meta-Llama-3-8B-Instruct` 的 Transformers 镜像，使用项目已下载的 MNTP 和 supervised 两级 LLM2Vec 适配器。不会将 Qwen 或随机向量替代文字条件，也不会修改原模型或依赖源码快照。

## 已完成的实际验收

2026-10-20 已在服务器物理 GPU 7 完成默认双提示词流程，状态为 `real_text_conditioned_component_passed`。底座加载没有缺失或形状不匹配的权重，仅忽略编码器不使用的 `lm_head.weight`；MNTP 和 supervised 各 448 个适配器张量均与公开检查点逐项一致。

真实文本特征形状 `[2,1,4096]`、全部有限；两条特征最大绝对差 6.53125、余弦相似度约 0.621857。两条提示词使用同一随机种子 42，生成关节数组的最大绝对差约 3.612672 m，通过“文字变化实际影响采样”的数据流判定；这个差异不是动作准确率指标。

两段视频均为 90 帧、30 FPS，写出后重新解码验证帧数通过：

- `outputs/previews/kimodo_g1_text_1_20261020T113407900253Z.mp4`：`A person walks forward.`
- `outputs/previews/kimodo_g1_text_2_20261020T113407900253Z.mp4`：`A person raises both arms above their head.`

该次真实特征、动作数组及报告保存在 `outputs/kimodo_text/20261020T113407900253Z/`。下面的环境和脚本可重复执行；最新状态仍以本次或后续运行报告为准。

## 兼容环境

专用服务器环境为 `/data/KinoGen/.venvs/kimodo-text`，继承项目 Kimodo 基线所需的 Torch 和动作生成依赖，仅在本项目环境覆盖以下包：

| 包 | 版本 |
|---|---|
| transformers | 4.51.3 |
| peft | 0.15.2 |
| tokenizers | 0.21.4 |
| huggingface-hub | 0.36.0 |

现有 `.venvs/kimodo` 和共享 GRAIL 环境保持原状。上述 wheel 从官方 PyPI 获取并按其公布的 SHA-256 核验，回执在 `downloads/kimodo-text-wheels/receipt.json`。

环境可由以下入口重复创建或核对（要求该服务器上的项目 Kimodo 基线和 wheel 文件已准备好）：

```bash
cd /data/KinoGen
KINOGEN_CUDA_DEVICES=7 bash scripts/setup-kimodo-text.sh
```

该脚本验证 wheel 回执，使用 `requirements-kimodo-text.lock.txt` 的四个版本与固定 wheel 哈希离线安装，再执行小模型双向性预检，写出 `reports/kimodo-text-environment.json`。重复执行只核对或更新本项目 `kimodo-text` overlay；不会在共享 GRAIL 或 `.venvs/kimodo` 中安装包。此 lock 仅固定四个兼容覆盖包，其余依赖仍来自已有基线，并非整套环境的独立锁文件。

采用独立环境有实际依据：原基线 transformers 5.1.0 的 `LlamaModel.forward` 不再调用该 Kimodo 快照 `LlamaBiModel._update_causal_mask`，因此仅成功导入类不等于仍在做双向编码。小型随机初始化 CPU 模型的检查中，改变末 token 后，5.1.0 的首 token 输出差为 0；专用 4.51.3 环境下差约 0.305135，确认未来 token 能影响此前位置。此小模型只检查注意力接口，不向实际动作模型提供任何特征。

## 运行

在服务器执行，显式选择物理 GPU 7（或 GPU 6）：

```bash
cd /data/KinoGen
export KINOGEN_CUDA_DEVICES=7
source scripts/activate-server.sh kimodo-text
python scripts/smoke_kimodo_text.py --preflight
python scripts/smoke_kimodo_text.py
```

完整运行的前置条件是原版四个底座分片已经完整下载并通过哈希核验；`--preflight` 只做小模型 API/双向注意力检查，不加载 8B 权重。运行时强制离线、限制单一可见物理 GPU 6 或 7，避免 LLM2Vec 根据多 GPU 数量启动多进程。

默认对两条不同文字分别生成 90 帧、30 FPS 的真实 Kimodo 动作：

```text
A person walks forward.
A person raises both arms above their head.
```

正常单任务可以只提供一条非空描述，例如：

```bash
python scripts/smoke_kimodo_text.py --prompt 'A robot lifts a box with both hands.'
```

单条描述会正常执行真实编码及动作采样，报告标记 `comparison_not_applicable_single_prompt`，不产生不同描述的特征比较或余弦相似度结论。上述提示词只是输入示例；本动作组件不创建箱子几何、约束接触或验证搬运成功。

也可重复传入 `--prompt`，提供多条不同且非空的描述：

```bash
python scripts/smoke_kimodo_text.py \
  --prompt 'A person walks forward.' \
  --prompt 'A person sits down.'
```

## 本地路径与编码顺序

`configs/kimodo-text-runtime.json` 固定底座身份、镜像 revision、分片及适配器 SHA-256、本地目录和编码设置。脚本先逐文件校验，再执行：

1. 直接从本地原版底座目录创建官方 vendored `LlamaBiModel`，不通过适配器内的远端底座名称自动解析；要求底座权重无缺失/不匹配，仅允许编码器不使用的 `lm_head.weight`。
2. 用本地 MNTP 适配器加载后合并，再加载本地 supervised 适配器；两个适配器均逐张量检查加载值与公开检查点一致。
3. 使用 MNTP 配套 tokenizer，保留原模型身份引导的 user header 格式、左侧 padding、双向注意力、mean pooling、最大长度 512、文档长度 400 和跳过指令的官方编码逻辑。实际编码 dtype 为 bfloat16，内部 batch size 固定为 1。
4. 保存实际编码获得的 `[B,1,4096]` 特征、原文字、长度及模型来源回执。始终检查全部有限；两条及以上描述时额外要求特征不同并记录余弦相似度，单条则明确标记比较不适用。不人为改变或随机填充特征。
5. 释放整个 8B 编码器并清理 GPU 缓存，再加载 Kimodo G1-RP，用本次真实缓存采样。缓存编码器对不在本次编码集合中的文字直接报错。

将编码与动作生成分成两个显存阶段，是为了在 24 GiB 单卡上运行；缓存中的特征仍来自原版模型和两个原适配器。脚本使用 eager 注意力实现，具体设置写入回执，不声称与原论文底层数值逐位一致。

## 输出与验收范围

- `reports/kimodo-text-preflight.json`：小模型兼容性检查，不能视为真实文字编码通过。
- `outputs/kimodo_text/<时间戳>/llm2vec_features.npz`：本次真实编码缓存。
- 同目录 `report.json`、动作 NPZ：实际来源、加载和采样证据。
- `outputs/previews/kimodo_g1_text_<编号>_<时间戳>.mp4`：真实生成动作的双视角骨架预览，写出后重新解码检查帧数。
- `reports/kimodo-text-latest.json`：最近一次运行状态；只有 `real_text_conditioned_component_passed` 才表示原版文字编码与动作采样均完成。

多条描述使用相同扩散随机种子，记录其实际动作差异，方便人工比较；单条不记录不存在的跨描述比较。有限的真实特征、有效动作和视频只证明文字条件的数据流实际执行；尚未完成文字语义准确率的定量评估，也未做物理稳定性验证、手物体交互、视频生成或完整 KinoGen 集成。

本次接入仍遵循 Meta Llama 3 与各适配器的许可，见 [Meta 官方许可](https://github.com/meta-llama/llama3/blob/main/LICENSE)和 [LLM2Vec 官方实现](https://github.com/McGill-NLP/llm2vec)。同模型镜像核对见 [ModelScope 核验](LLAMA3_MODELSCOPE.zh-CN.md)。
