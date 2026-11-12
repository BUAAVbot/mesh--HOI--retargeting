# HDMI-Vita 接入盘点

更新：2026-10-29。用户提供的 `HDMI-vita-main.zip` 已完整归档至 [原包](../archives/source_snapshots/hdmi/HDMI-vita-main.zip)，源码及随包资产解包至 [projects/hdmi](../projects/hdmi)。原包SHA256为 `cd6ad62911e970d8d40d11439d8c1929abb5f70749e5b29ffceee1d4932bbfda`；436个文件、解包总计190,056,035字节，ZIP CRC及提取后436份逐文件SHA256复核通过。[逐文件SHA清单](../manifests/hdmi_source_inventory.json)、[集成审计](../manifests/hdmi_integration_audit.json)。本次只清点、静态编译与归档，没有安装或运行HDMI训练。

## 在现有链路中的位置

`视频/外部SMPL-X → 场景和人体重建 → GMR/UMR/IK等机器人重定向 → **HDMI动作格式与Isaac Lab追踪策略训练** → 物理rollout/接触与质量检查 → HOI数据`。

HDMI是控制/物理模仿环节的另一套框架，不替代三维重建、重定向或本工作区已有的SONIC/PPO物理结果。其[主README](../projects/hdmi/README.md)描述Isaac Sim 4.5、Isaac Lab 2.2和PPO/ROA；[算法代码](../projects/hdmi/active_adaptation/learning/ppo/ppo_roa.py)、[训练/播放入口](../projects/hdmi/scripts/train.py)、[Vita数据转换器](../projects/hdmi/scripts/vita/umr_to_motion.py)均已随包提供。

| 接口 | 包内已有 | 对我们当前结果的关系 |
| --- | --- | --- |
| 机器人 | [Vita配置](../projects/hdmi/active_adaptation/assets/vita.py)、鞋版/非鞋版URDF及网格 | 两份URDF均为**28个可动关节**，动作和驱动主要针对身体/腕；我们现有双Vita参考是**52关节、五指手**。需显式按关节名选择28维身体投影及手部处理方案，不能直接复用qpos。 |
| 场景/任务 | [双人换座P01](../projects/hdmi/cfg/task/Vita/seatswap_p01.yaml)、[P02](../projects/hdmi/cfg/task/Vita/seatswap_p02.yaml)、RGB145台阶配置 | 每人一个独立追踪任务；配置不是双机器人同场交互策略。场景文件从`data/motion/vita/.../scene.json`读取。 |
| 动作输入 | [转换器](../projects/hdmi/scripts/vita/umr_to_motion.py)读取UMR `qpos`、`robot_joint_names`，生成`motion.npz`、`meta.json`、`scene.json`；[加载器](../projects/hdmi/active_adaptation/utils/motion.py)按名称映射并重采样 | 与现有[UMR双人172帧结果](../workflows/vita_retarget_comparison/README.md)有明确可对接点，但要先适配28/52维、场景代理、原始时间、坐标及接触定义。转换器默认MJCF是原作者机器绝对路径，可通过`--robot_xml`覆盖。 |
| 训练/输出 | Hydra任务配置、PPO/ROA、播放/rollout导出、接触检查与渲染脚本 | 包内**没有Vita策略检查点**，也没有双人换座的`motion.npz/meta.json/scene.json`。现有9.56秒SONIC换座样例与HDMI无关。 |

## 当前缺项与验证边界

1. `projects/hdmi/data/motion/vita/`整个目录未随ZIP提供；换座P01/P02及台阶配置引用的数据均不存在，需要从现有UMR结果生成并核对场景。包内脚本说明也明确要求重新生成。
2. 原Vita转换器默认MJCF、部分渲染脚本及USD缓存目录写有原作者绝对路径；迁移到`yz-ai-4090-01.bce`需用显式参数/环境变量或隔离适配，不修改原包证据。
3. README建议Isaac Lab 2.2.0；现有已记录的GRAIL跟踪环境是Isaac Lab 2.1系。必须单独验证版本/API，不把“服务器有Isaac Sim”当HDMI可直接启动。
4. [Vita脚本说明](../projects/hdmi/scripts/vita/README.md)提及`train_and_export.sh`，但该文件不在ZIP内；可用已有`train.py`/`play.py`重新编排，不能照抄该命令。
5. 主README示例引用的`cfg/algo/ppo_roa_train.yaml`及`cfg/task/G1/G1Track-walk.yaml`也未随ZIP提供；`cfg/train.yaml`默认配置由此不能直接组成一个完整训练启动。需要审查并补齐Hydra配置，不能只按README命令运行。
6. 本次静态检查：96个Python文件全部可编译；这只排除语法错误，**不证明依赖可安装、物理任务通过或原论文复现**。

因此当前状态是：**源码、Vita28资产与训练配置已纳入工作区；独立mjlab接入烟测已运行，但HDMI原始老师—学生算法尚未运行，Vita52五指手训练接入仍待实现。** 其原始G1样例动作和G1资产保留在项目内部，不混入现有GRAIL/KinoGen已验收结果。

## mjlab迁移准备（2026-10-29新增）

[迁移目录](../workflows/hdmi_vita_tracking/mjlab/README.md)中已生成HDMI鞋版Vita28碰撞MJCF，并将本工作区UMR、OmniRetarget、HOI-Retarget的双人轨迹共六条投影为Vita28。2026-10-29经授权主机网络使用现成SSH别名连接服务器，在`.venv310`完成MuJoCo编译、六条轨迹的50Hz重采样和目标刚体FK、单人20步物理烟测、UMR P01两轮PPO以及50帧独立回放。**这是训练入口烟测，不是任务成功：**回放出现5次终止，UMR P01仍有右脚最低约20.4mm穿地，OmniRetarget P01最大关节速度约27.43rad/s。当前配置采用mjlab内置跟踪PPO基线，HDMI的`ppo_roa.py`老师隐变量／学生蒸馏尚未迁移；双人同场物理训练也未完成。
