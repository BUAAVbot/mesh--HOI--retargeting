# 交接说明

## 必须外部提供的资产

每个案例至少需要以下清单及 SHA-256：

1. 原视频、FPS、有效帧和相机参数。
2. SMPL-X/人体 Mesh、物体 Mesh 与 6D 轨迹、场景 Mesh。
3. 世界/相机/人体/物体坐标变换和单位。
4. 机器人 URDF/MJCF、关节顺序、执行器和限位。
5. 质量、摩擦、PD、仿真步长和策略周期。
6. 模型权重与训练检查点的下载位置。

## 每阶段交付物

| 阶段 | 最小交付物 | 通过条件 |
| --- | --- | --- |
| 视频与 Mesh | 视频、相机、人体/物体/场景 Mesh | 时间、尺度、坐标和 valid 可追溯 |
| HOI | 人体状态、物体轨迹、接触窗口、场景描述 | 无静默坏帧，接触类型明确 |
| Retargeting | 机器人 root/joint/body FK 与元数据 | 限位、穿透、自碰撞、速度门禁通过 |
| 动力学 | 实际 state/control/contact/torque | 连续重放，无中途 reset 或 root 强写 |
| 交付 | 报告、配置、SHA、对照视频 | 独立机器可复现同一审计结论 |

## 推荐入口

- UMR：`umr/scripts/humanoid_retarget_pipeline_hsi_hoi.py`
- HDMI 转换：`hdmi_adapters/vita/umr_to_motion.py`
- 批处理：`workflows/batch_sim_expert/inspect_cases.py`、`compile_jobs.py`、`dag_runner.py`
- 动力学审计：`workflows/stage2_physics_compare/replay_validate.py`

成功、失败和仅运动学结果必须分别标记；仿真接触力不能当成传感器真值。
