# Retargeting tools baseline

## 范围与完整性

此基线将 UMR 核心扩展和已运行过的双 Vita02A 换座处理代码一起纳入版本控制。
双人链路不再依赖仓库外 `data_umr/work/` 中的 Python 脚本，也不硬编码某台机器的资产路径。
它是**代码与入口完整、外部资产另行提供**的运动学基线，不是所有数据集的数据工程快照。

包含：

- 单条/批量重定向、共享 neutral 对应关系、`slot_name` 兼容及长文件名 sidecar。
  浮动 MJCF 随结果保存，不再向原机器人资产目录写入生成文件。
- 多物体接触/碰撞、批量 HOI 单条求解器分流、物体元数据读取。
- 启用碰撞掩码/显式 pair 的地面约束，以及不膨胀接触缓冲区的全 pair 查询。
- 源有效性导出、SOMA CPU 分块 LBS 与精确 no-grad 身份缓存。
- 双人输入准备、双凳拟合、独立重定向、接触修正、整段时序平滑、碰撞代理检查及视频。
- 无私有模型/数据依赖的回归测试。

不包含原始 ZIP、SMPL-X 权重、私有 Vita02A 机器人资源、已有对应关系缓存、日志、
动作输出和视频。生成文件默认放在已忽略的 `output/`。
仓库外的全量数据调度、训练数据打包、发布及数据集下载工具不属于此基线。
上游不包含的 BVH-to-SMPL-X 转换器也没有因此补齐，仍见主 README 的来源支持说明。

## 安装与外部资产

两套安装配置不能混装：

1. 上游通用安装：主 README 中的 PyTorch 和 `requirements-umr.txt`。
   使用双人视频工具时另装 `requirements-preview.txt`。
2. 本次验收运行环境的参考配置：先安装适合机器的 `torch==2.7.0`
   （验收使用 CUDA 12.8 构建），再安装 `requirements-baseline.txt`。
   实际验收解释器为 Python 3.11.15，核心版本为 NumPy 2.4.6、
   MuJoCo 3.12.0、SciPy 1.15.3、Clarabel 0.11.1。
   这份文件记录观察到的版本，不是全部传递依赖和外部资产的可复现锁文件。
   本次没有重新创建干净环境验证安装。

参考配置用于**提供已有对应关系**的运行方式，不含未在此环境安装的可选
`cholespy`/`obj2mjcf` 学习和资产转换依赖。需要重新学习对应关系时，参考上游完整安装；
本次不声称验证了重新训练或所有旧版依赖组合。

需要用户自行提供：

- 场景重建 ZIP，目录结构见下文。
- Vita02A MJCF 及其引用的 mesh。该预设要求单机器人 `nq=35, nv=34`、
  `waist_yaw_link`、对应肩肘关节及足底 collision geom 命名。
  配置在 `robot_configs/humanoid_retarget_vita02a_example.json`。
- 合法获得的 neutral SMPL-X 模型目录，与对应关系的源模板一致。
- 可选：兼容的 `correspondence_slots_final.npz`。省略 `--slots` 时入口安排
  build/train，仅对共享 neutral 模板学习一次。

新克隆还需安装 Git LFS 并下载仓库跟踪的 LFS 资源。
私有模型或用户数据不应放进本次代码提交。

## 一条命令

从仓库根目录运行，路径按本机资产位置填写：

```bash
python tools/two_person_baseline/run.py \
  --work-dir output/two_person_baseline \
  --input-zip /path/to/two_sitting_people.zip \
  --robot-xml /path/to/vita02a_gmr.xml \
  --smplx-model-dir /path/to/smplx_models \
  --slots /path/to/correspondence_slots_final.npz
```

`--dry-run` 只打印命令，不创建目录或修改文件。`--help` 不加载模型。
可选 `--archive-prefix` 改 ZIP 内根目录，`--seat-centers X1 Y1 X2 Y2`
改 Z-up 世界坐标下两张凳子的拟合搜索中心。
入口固定使用当前 Python、仓库路径和工作目录，不依赖执行时的 shell 当前目录。
Linux 无显示器渲染默认使用 EGL；需要时设置 `MUJOCO_EGL_DEVICE_ID`。
OMP/MKL/OpenBLAS 线程数未配置时默认设为 2。

首次运行只初始化空目录。后续只需 `--work-dir` 和 `--stage`，
资产路径会保存于该目录的 `baseline_settings.json`。
已有阶段输出默认拒绝覆盖；仅在明确需要重算时使用 `--overwrite`。
更换输入或资产路径请使用新工作目录；成功准备后会用 ZIP 哈希阻止原地更换输入内容。
不要把工作目录指向之前的交付目录。工作目录若位于仓库内，必须在 `output/` 下。

### 分阶段

| 阶段 | 操作 |
| --- | --- |
| `prepare` | 读取同步时间轴，转换 SMPL-X，拟合两个静态座椅，生成初版/强化配置 |
| `correspondence` | 复用显式提供的缓存，或调用上游 `build`、`train` |
| `retarget` | 对两人分别运行强化配置，保留共享世界坐标和缩放 |
| `scene` | 创建强化结果的双机器人场景，不渲染 |
| `ground` | 足锚 QP、有限根平移修正，创建接触修正后的场景 |
| `smooth` | 整段稀疏 SQP，记录迭代状态、是否收敛及前后运动指标 |
| `render` | 接触修正视频、平滑视频、正常/半速对比，以及质量报告 |
| `all` | 默认，依次执行以上阶段 |

例如完成准备后：

```bash
python tools/two_person_baseline/run.py --work-dir output/two_person_baseline --stage correspondence
python tools/two_person_baseline/run.py --work-dir output/two_person_baseline --stage retarget
```

`all` 直接使用强化重定向作为后处理输入，不额外计算早期初版视频。
如需早期初版，可手动将 `configs/person_01.json`、`person_02.json`
交给 `scripts/humanoid_retarget_pipeline.py --stage retarget --skip-view`。
子阶段脚本通过 `UMR_BASELINE_WORKDIR` 读取工作区，建议统一使用入口而非直接调用。

## 输入契约与预设边界

默认 ZIP 根目录 `rgb_174114_sim_handoff/`：

```text
timeline.npz
cameras_world.npz
input.mp4
scene/pi3_mesh.npz
human/person_01/smplx_params_world.npz
human/person_01/smplx_meshes_world.npz
human/person_02/smplx_params_world.npz
human/person_02/smplx_meshes_world.npz
```

要求：至少 4 帧，双方全帧有效，连续零起点帧号，均匀且从零开始的时间戳，
同步 RGB 视频，每人固定 betas，neutral SMPL-X，`model_scale=1`。
参数包含 `global_orient/body_pose/left_hand_pose/right_hand_pose/transl/betas/valid`。
源为 Y-up、近似米制，场景和人体共用同一坐标转换。
帧数和帧率从时间轴读取，不再固定为 172/30。
源网格和相机保留作审计输入，不用于重新估计人体、深度或相机。

这是**双 Vita02A 走动换座预设**，不是任意多人/机器人/物体的自动适配器：
凳面搜索高度约 0.3–0.8 米，碰撞代理为轴对齐实体箱，默认相机与场景裁剪针对本片段。
假设没有跳跃，至少一脚支撑；片头、片尾约定坐姿双支撑。
速度上限按帧率换算，但滤波窗口与优化权重仍需针对新动作重新调参。
其他机器人或任务应使用通用 UMR 入口，或显式修改本预设。

## 输出与验收边界

推荐最终输出：

- `smoothing_v1/results/person_0{1,2}_smoothed.npz`
- `smoothing_v1/preview/dual_vita02a_seat_exchange_smoothed.mp4`
- `smoothing_v1/preview/before_after_smoothing.mp4`
- `smoothing_v1/smoothing_report.json`
- `smoothing_v1/preview/quality_report.json`、`collision_distances.npz`
- `smoothing_v1/preview/video_validation.json`

中间结果保留在 `results/`；逐帧接触修正可能引入抖动，不应替代最终时序平滑结果。
平滑保持帧数与正常速度，对比视频另有明确标注的半速段。
根旋转采用旋转空间滤波，时序 SQP 约束关节速度、关节限位、地面/凳子和近距离双机碰撞。
两人原始重定向仍为独立求解，不是完整联合多人 HOI 优化。
渲染地面为 `z=-0.008m`，支撑脚的代理间隙目标约 11mm；距离查询在 0.5m 截断。

原 172 帧、30 Hz 案例中，平滑后的最大关节单帧变化约 7°，
两人关节加速度 RMS 相较逐帧接触修正降低约 94%/95%，碰撞代理检查通过。
该案例的有限次 SQP **未达到迭代收敛阈值**；通过输出检查不代表全局最优或任意输入都成功。
报告中的 `converged`、求解状态和最终质量指标应一起查看。

这不是动力学 rollout：没有验证接触力、力矩、平衡、控制器跟踪或真机安全。
地面/凳子与机器人碰撞基于代理，不是完整视觉 mesh、自碰撞或完整环境的精确证明。
RGB 重建本身不是实测几何。不能直接将输出称为可上机轨迹。

## 测试与验证记录

在已安装运行依赖的环境中，无需私有资产运行：

```bash
python -m unittest discover -s tests -v
```

19 项测试覆盖多物体第二物体碰撞、地面掩码/显式 pair、QP 可行性拒绝、长文件名、
有效帧时间轴、SOMA LBS 等价性与缓存失效，以及入口计划/工作目录保护。
实际输入准备另检验了 float32 时间戳精度问题，采用整段时间轴拟合帧率避免差分误差累积。

本次提交验证使用已有合法资产和独立的忽略工作目录。
验证范围包括真实 ZIP 准备、两人各 4 帧求解，以及从此前已验证的 172 帧强化结果重放
接触修正、平滑和完整视频验证。没有重新训练对应关系、全量重算 172 帧重定向、
重跑所有批量数据集或重新安装干净环境。具体结果见本地验收日志。

## 保留的运行控制

- `UMR_QP_NUMERICAL_RETRY=1` 显式启用同一 QP 的数值重试，仍检查原始约束/KKT，
  不删除约束或将不可行结果升级。诊断捕获为单独 opt-in 模式。
- 机器人/物体硬约束失败后退回软约束，只有显式开启
  `robot_object_hard_failure_fallback_to_soft` 才生效；双人预设开启并记录计数，
  因此必须检查最终碰撞报告，不能只看求解器退出码。
- SOMA CPU 默认 `UMR_SOMA_CPU_LBS=vertex_chunked`、
  `UMR_SOMA_IDENTITY_CACHE=last_input`，可分别设为 `original` / `off`。
- `bones_pause_gate.py` / `bones_shard_gate.py` 保留原有特定路径作业的暂停/远端所有权控制，
  依赖仓库旁 `data_umr/work/` 中的控制文件，不是通用数据准备工具。
  只有命中特定 BONES 配置路径才启用，不参与这里的双人链路。
