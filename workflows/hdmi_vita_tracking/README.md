# HDMI-Vita → mjlab 迁移工作区（2026-10-29）

## 状态边界

本目录已把 HDMI 鞋版 Vita28 URDF 转换为无视觉网格的碰撞 MJCF（28 可动关节、47 碰撞体），把现有双人 **UMR、OmniRetarget、HOI-Retarget** 的 Vita52 轨迹各投影为 Vita28 的 172 帧输入（共六条），并实现 HDMI NPZ → **目标 MuJoCo 模型顺序** 的重新正运动学转换器。`task.py` 在 mjlab 的内置跟踪任务上配置 Vita28、50 Hz、身体跟踪、双座位静态实体和不扰动的播放配置。原 HDMI 源码保留在 `projects/hdmi`，没有改写。

**已完成服务器模型编译、六条轨迹转换、单人物理烟测，并用 4096 环境三阶段 PPO 训练通过 UMR P01 单人完整轨迹验收。** 最优 `model_1847.pt` 的确定性回放为 361 帧、零终止，根部最大位置误差 0.0614 m；视频、权重和指标位于 `results/umr_p01_full_trajectory_success/precision_best/`。三阶段配方已固化为 `three_stage_train.py` 与 `three_stage_template.json`，说明见 `THREE_STAGE_TEMPLATE.zh-CN.md`。本结果是 mjlab 原生 RSL-RL 专项轨迹专家，不是 HDMI 官方 `ppo_roa.py` 老师隐变量编码／学生残差蒸馏实现；随机外力、质量、摩擦与观测噪声下的鲁棒性仍需另行验收。2026-10-29 已在授权主机网络中通过现成 SSH 别名 `yz-ai-4090-01.bce` 连接服务器；实际环境为 `/root/vita02a-mjlab/.venv310`（MuJoCo 3.5.0、mjlab 1.2、PyTorch 2.7.1+cu126）。

## 已有视频与准备产物

| 关卡 | 可核对输入／输出 | 当前视频 |
| --- | --- | --- |
| 上游仿真／重建参考 | `projects/kinogen/outputs/retarget_compare_20261028` | `media/overview/source.mp4` |
| UMR | 同一源、双人 172 帧 | `media/overview/umr.mp4` |
| OmniRetarget | 同一源、双人 172 帧 | `media/overview/omniretarget.mp4` |
| HOI-Retarget | 同一源、双人 172 帧 | `media/overview/hoi_retarget.mp4` |
| mjlab 物理烟测 | 20 步环境验收、两轮 UMR P01 PPO、独立 50 帧回放 | `media/umr_person_01_ppo_smoke.mp4`，**5 次终止／50 帧，未成功** |
| mjlab 长训策略 | UMR P01，4096 环境×320 轮；31,457,280 步 | `results/umr_p01_4096x320/best_model_319.mp4`，**7 次终止／361 帧，未通过** |
| mjlab 专项轨迹专家 | UMR P01，4096 环境三阶段训练；361 帧确定性回放 | `results/umr_p01_full_trajectory_success/precision_best/full_trajectory_precision_best.mp4`，**零终止名义验收通过** |
| 组装总视频 | 只在五段均核验后生成 | **不存在** |

上游 `source.mp4` 是重建参考可视化；它不是上游“仿真成功”的证明。三种重定向视频也是运动学可视化，物理通过须以后续 mjlab rollout 为准。

## 服务器验收记录（2026-10-29）

- Vita28 MJCF 在 MuJoCo 3.5.0 编译为 `nq=35`、`nv=34`、55 刚体、47 碰撞几何、2 IMU 传感器；六条轨迹各为 361 帧、50 Hz，关节与刚体顺序合同均通过。`task.py` 将 `nconmax` 提到 256、`njmax` 提到 4096 后，GPU 0 的单人环境 20 步可运行。
- UMR P01 的两轮、32 环境 PPO 烟测生成了 `model_0.pt` 和 `model_1.pt`，但平均 episode 仅约 12.65 步；独立 50 帧物理视频中出现 5 次终止。该视频用于展示当前失败状态，不能作任务完成演示。
- [4096环境早期对比测试](results/umr_p01_4096x17/REPORT.zh-CN.md)使用同一UMR P01轨迹完成17轮、1,671,168个转移；稳态约5.4万步／秒，训练循环33秒，端到端98.68秒。固定初态361帧回放出现41次终止，平均奖励0.0290，仍未通过物理跟踪验收。该记录验证的是大并行训练入口和吞吐，不是HDMI专家成功。
- [4096环境320轮长训](results/umr_p01_4096x320/REPORT.zh-CN.md)共采样31,457,280个转移，端到端634.50秒；第50/100/150/200/250/300/319轮固定初态回放终止次数依次为39/14/11/9/7/7/7。当前最佳为`model_319.pt`，但361帧仍有7次终止，只能称长训策略原型，不能称已验收专家。
- 输入仍需物理清洗：UMR P01 转换报告右脚最低穿地约 20.4 mm，9 帧超过 5 mm；OmniRetarget P01 的最大关节速度约 27.43 rad/s，明显不适合原样训练。双人同场动力学和 HDMI 老师—学生训练仍未建立。

## 服务器接通后的固定次序

在项目根目录运行以下命令。`PY` 应指向服务器已安装 `mujoco/numpy/mjlab` 的 Python；不要在 Windows 本地做训练。现有服务器工作区为 `/data/yihang.bai/training/hdmi_vita_mjlab`，`PY=/root/vita02a-mjlab/.venv310/bin/python`。`prepare_all.py` 已在服务器为六条轨迹生成 50 Hz、361 帧的 NPZ，回执见 `data/server_prepare_all_receipt.json`；`smoke_env.py`、`train_smoke.py`、`record_rollout.py` 已逐项运行。`train_smoke.py` 自带 mjlab 1.2 的 `registry_name` 兼容层，避免改写安装包。当前 50 帧视频统计见 `media/umr_person_01_ppo_smoke.json`。

```bash
PY=/path/to/mjlab/.venv/bin/python
TASK=workflows/hdmi_vita_tracking/mjlab
URDF=projects/hdmi/active_adaptation/assets/vita02a/urdf/vita02a_shoes.urdf

$PY "$TASK/build_vita_mjcf.py" --urdf "$URDF" --output "$TASK/assets/vita28_shoes.xml"
$PY "$TASK/support_to_scene.py" --source projects/kinogen/outputs/two_sitting_pipeline/assets/support_geometry.json --output "$TASK/data/seats_source.json"

# 本地已生成 3 方法×2 人的投影；在服务器重做并核对 SHA、帧数、关节名。
for method in umr omniretarget hoi_retarget; do
  for person in 01 02; do
    case "$method" in
      umr) input="projects/kinogen/outputs/retarget_compare_20261028/umr/postprocessed/smoothing_v1/results/person_${person}_smoothed.npz"; person_arg="" ;;
      *) input="projects/kinogen/outputs/retarget_compare_20261028/${method}/reference.npz"; person_arg="--person-index $((10#$person-1))" ;;
    esac
    $PY "$TASK/project_vita52_to_vita28.py" --source "$input" $person_arg \
      --urdf "$URDF" --output "$TASK/data/${method}_person_${person}_vita28.npz"
    $PY projects/hdmi/scripts/vita/umr_to_motion.py \
      --umr_npz "$TASK/data/${method}_person_${person}_vita28.npz" \
      --robot_xml "$TASK/assets/vita28_shoes.xml" \
      --scene_json "$TASK/data/seats_source.json" \
      --out_dir "$TASK/data/hdmi_${method}_person_${person}"
    $PY "$TASK/prepare_motion.py" \
      --source "$TASK/data/hdmi_${method}_person_${person}" \
      --mjcf "$TASK/assets/vita28_shoes.xml" \
      --output "$TASK/data/mjlab_${method}_person_${person}.npz"
  done
done
```

`prepare_motion.py` 必须使用最终的 mjlab MJCF 重算全部刚体 FK：mjlab 官方文档明确警告，Isaac 与 MuJoCo 刚体索引顺序不同，直接复制 HDMI 的 body 数组会静默训练错误。输出同名 JSON 回执包含关节／刚体顺序和根位置误差。`scene.json` 的箱体与 `mjlab_person_*.npz` 共享 HDMI 转换后的 XY 偏移。

现有 UMR 两人轨迹均为 **30 Hz**；HDMI 转换器保留原帧率，而 mjlab 任务控制频率是 **50 Hz**。`prepare_motion.py` 默认把位置／四元数重采样到 50 Hz，并在新时间轴上重算关节与根速度。不得把 30 Hz 数组直接当 50 Hz 训练。重采样函数已用匀速／恒姿态合成输入检查时长和速度，真实轨迹仍需服务器 MuJoCo FK 验收。

在 mjlab 源码检出根目录下，先用 `install_task.py --mjlab-root /path/to/mjlab` 复制本任务为 `src/mjlab/tasks/hdmi_vita28`；它不会覆盖已有目录。设置 `HDMI_VITA_MOTION` 和 `HDMI_VITA_SCENE` 为某一条 `mjlab_<方法>_person_<人>.npz` 和对应 `hdmi_<方法>_person_<人>/scene.json` 的**绝对路径**。然后运行 `uv run play Mjlab-Tracking-Flat-HDMI-Vita28 --agent zero` 检查模型、座位、动作和碰撞；再以少量环境做第一轮短训练。六条参考应分别启动、分别保存配置／日志／视频。录制时用 `uv run play ... --checkpoint-file ... --video` 或该版本的录制参数；具体 CLI 以已安装版本 `--help` 为准。

## 必须逐关通过的检查

1. `MjModel.from_xml_path` 编译；28 运动关节、47 碰撞体、1 个自由根；关节名、质量、限位与 URDF 静态清单一致。当前生成器省略视觉网格，碰撞形状和惯量来自 HDMI 鞋版 URDF；mjlab 配置已按 HDMI 分组设置身体／腕部 PD、动作缩放，并用投影增益近似踝部。**并联踝电机的完整力矩映射、HDMI 动作延迟和软限位仍须逐项对拍**，不能称动力学等价。
2. P01/P02 轨迹各 172 源帧、30 Hz，投影后 `qpos` 宽 35，最终 `joint_pos` 宽 28、50 Hz；时间戳、坐标、四元数和 FK 误差合格。HDMI 转换器默认加起末 padding，因此最终帧数会大于 172，需在回执里记录并检查总时长不变。
3. 场景两座位与根位置同系；静态姿态下检查足—地、臀—座接触、初始穿透。单人任务看另一人是否需要作为动态障碍；当前任务未加入第二个机器人，因此**不能宣称双人同场换座 RL 完成**。
4. `--agent zero` 与随机动作各录短视频、保存 MuJoCo 状态和碰撞报告；训练后老师 checkpoint、独立物理重放视频和固定初态成功率需要逐项保存。不能用奖励曲线代替任务成功率。
5. 真正 HDMI 老师→学生迁移还需把 `projects/hdmi/active_adaptation/learning/ppo/ppo_roa.py` 的特权编码器、适应模块、残差动作蒸馏接到 mjlab 观测和训练 runner；HDMI 的未来参考步 `[1,2,8,16,32]`、分组奖励和动作延迟也须在 mjlab 中对齐，再验证老师／学生同初态对照。当前 RSL-RL 基线只用来先验收物理与动作合同。

官方依据：[mjlab 训练与任务注册](https://mujocolab.github.io/mjlab/main/source/training/rsl_rl.html)、[mjlab 运动模仿数据要求](https://mujocolab.github.io/mjlab/v1.6.0/source/training/motion_imitation.html)。
