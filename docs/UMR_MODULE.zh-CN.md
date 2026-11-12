# UMR 新增模块与接入位置

更新：2026-10-28（北京时间）。原包清点与本次服务器实测分开记录：原包确有缺失的LFS资源，但已针对双人换座建立隔离的Vita52适配，真实训练对应关系并运行全片运动学重定向。[实验入口](../workflows/vita_retarget_comparison/README.md)。

## 收到的内容

- 原包：[UMR-main.zip](../archives/source_snapshots/umr/UMR-main.zip)，30,509,629字节。
- SHA256：`a2cfb5f26080b7bec0da200eaf88e1fa51db68ba40668b688a6bffc4b6007ab5`。
- 解包：[UMR-main](../projects/umr)，411个文件，ZIP CRC通过。
- 其中**283个文件是Git LFS指针**，包括大量机器人mesh、样例动作和SMPL-X分区资源；占位文件不含实际资产数据。[逐文件SHA及占位清单](../manifests/umr_source_inventory.json)。

## 它负责哪一环

包内名称为 **Unified Motion Retargeting for Humanoids with Learned Point Cloud Correspondence**。通过学习人体表面与机器人表面的对应关系，将人体运动重定向为机器人关节轨迹，并处理场景、物体接触与运动学约束。

```text
已有视频/外部SMPL-X重建 + 场景/物体几何与轨迹
  → UMR表面对应学习/复用缓存
  → 接触与场景约束下的重定向
  → 足支撑修正/时序平滑/碰撞代理检查
  → 对齐Vita52或G1的关节、时间和控制合同
  → 现有SONIC/PD/PPO残差 → 真正动力学验收/HOI标签
```

UMR位于系统图中的“接触感知运动重定向”。它可作为现有GMR/自建IK的候选方案，不替代三维重建、SONIC、动力学积分或稳定性验收。使用MuJoCo显示和查询距离，也不代表已做动力学rollout。

## 已包含的模块

| 内容 | 本地入口 | 当前能确认的范围 |
| --- | --- | --- |
| 表面对应学习 | [学习脚本](../projects/umr/scripts/train_correspondence_template_residual_ae.py)、[训练数据构建](../projects/umr/scripts/build_correspondence_ae_dataset.py) | 原包无缓存；本次4096点、500轮真实学习，缓存和训练回执已保存于[实验结果](../projects/kinogen/outputs/retarget_compare_20261028/umr/learning_receipt.json) |
| 单条与批量重定向 | [单条入口](../projects/umr/scripts/humanoid_retarget_pipeline.py)、[批量入口](../projects/umr/scripts/humanoid_retarget_pipeline_batch.py) | 本次经隔离Vita52适配对双人各172帧实际执行；非原28关节配置原样运行 |
| 场景/物体约束 | [HSI/HOI入口](../projects/umr/scripts/humanoid_retarget_pipeline_hsi_hoi.py) | GRAIL/标准HOI等适配；输入仍需真实mesh、物体轨迹和合法模型 |
| G1/Vita等配置 | [G1示例](../projects/umr/robot_configs/humanoid_retarget_unitree_g1_example.json)、[Vita示例](../projects/umr/robot_configs/humanoid_retarget_vita02a_example.json) | 配置存在不代表机器人资源齐全；Vita私有MJCF/mesh需外部提供 |
| 双Vita换座处理 | [统一入口](../projects/umr/tools/two_person_baseline/run.py)、[基线指南](../projects/umr/docs/retargeting_baseline.md) | 原包未附结果；本次生成原始、精修、推断接触及原包ground/SQP后处理四组轨迹与[对比视频](../projects/kinogen/outputs/retarget_compare_20261028/media/overview/umr_ablation.mp4) |

## 与我们现有双Vita结果的关系

UMR双人预设默认读取 `rgb_174114_sim_handoff/`，对应我们现有的 [two_sitting_people.zip](../archives/raw_handoffs/two_sitting_people/two_sitting_people.zip)。**原始输入已具备，但机器人模型合同不同**：

| 项目 | UMR双人预设 | 已有物理演示 |
| --- | --- | --- |
| 单机器人状态维数 | `nq=35, nv=34`，自由根＋28身体关节 | `nq=59, nv=58`，自由根＋Vita52关节 |
| 主要目标 | 运动学轨迹、接触修正、平滑、代理碰撞检查 | 实际控制、力/力矩、功能完成和源轨迹检查 |
| 验证状态 | 本次Vita52隔离适配已跑两人各172帧；原始/消融由共同场景评估。后处理SQP达到迭代上限未收敛；未进物理 | 已有9.56秒选定功能样例，严格源轨迹跟踪失败；不是UMR输入 |

本次已对齐Vita52关节、源30Hz时间、坐标和整体scale，并保存碰撞/座面/踝速度评估。源手部有效标记全为false：UMR虽优化完整52关节，不可据此声称手指准确。下一步若要接控制链，仍须重新检查初态、速度、执行器/延迟和支撑接触；不能认为换参考会自动改善物理任务。

本次[后处理报告](../projects/kinogen/outputs/retarget_compare_20261028/umr/postprocessed/report.json)同样记录原包ground/SQP步骤到迭代上限未收敛。平滑/凳面距离的变化已用统一评估器测量，但未做接触力/力矩、平衡或控制跟踪验证。原包指南中的历史结果也不能算作本次物理验收。

## 当前剩余边界

1. 原ZIP的283个Git LFS占位文件仍不能视为完整通用资产；本次仅恢复/替换了Vita52双人片段所需资源，其他机器人/任务需逐项检查。
2. 已授权neutral SMPL-X和现有Vita52模型在服务器隔离环境接入；其路径与SHA保存在源清单/学习回执，未随精简交付ZIP复制授权模型和大体积检查点。
3. 原28身体关节预设仍不等于Vita52合同。本次使用显式Vita映射、全52维输出和同一场景评估，后处理尚未收敛。
4. 本次轨迹尚未接入SONIC/PPO或独立动力学重放；功能、严格源跟踪、稳定性及接触力学均待另行验收。

目前状态：**双人Vita52片段已真实学习对应关系并完成运动学对比；原通用源码包资源仍不完整，物理链未验收。**
