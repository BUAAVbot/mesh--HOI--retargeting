# G1 改为双五指 ROHand

已按用户要求将重建目标改为 **G1 身体 + 左右五指 ROHand**。生成模型保存在 `data/robots/g1_rohand_five_finger/g1_rohand.urdf`，场景配置为 `data/video_generation/candidate_scene/scene_rohand.json`。原始 Unitree 和 ROHand 源码/网格未修改。

左右手各有拇指、食指、中指、无名指和小指。每只 ROHand 只有 **6 个独立控制量**，通过厂商公开的非线性连杆公式展开为 25 个 URDF 运动关节；双手是 12 个控制量、50 个展开关节。加上 G1 的 29 个身体关节，总计 41 个独立控制量，不能把 79 个展开运动关节误称为 79 个独立自由度。

## 位置与单位

每侧控制顺序为：拇指屈曲滑块、食指滑块、中指滑块、无名指滑块、小指滑块、拇指根旋转。前五个单位是米，最后一个是弧度。拇指滑块范围 0–0.010 m，其余四个 0–0.019 m，拇指根 0–1.57 rad。

`scripts/user_rohand_coupling.py` 直接只读调用厂商的 `dependencies/rohand-urdf/scripts/FingerMathURDF.py`，没有把连杆关节分别自由优化。完整原控制范围各采样 1001 点的计算均有限，但采样检查不是形式化连续性证明。

新 URDF 中手部名称带 `left_hand_ro_` / `right_hand_ro_` 前缀。`scene_rohand.json` 的 `robot.joint_positions_rad` 仅包含 29 个身体弧度关节；`robot.joint_positions` 包含全部展开运动关节，单位由 `robot.joint_units` 明确标识。消费轨迹时也必须读取 `joint_units`，不能把滑块米数当作弧度。

## 腕部安装

保留 G1 原来的 29 关节链，将原三指手整棵子树移除，在 `left_wrist_yaw_link` 和 `right_wrist_yaw_link` 添加固定适配关节：

| 手 | 平移 XYZ（m） | RPY（rad） |
| --- | --- | --- |
| 左 | 0.0415, 0.003, 0 | π/2, 0, π |
| 右 | 0.0415, -0.003, 0 | -π/2, 0, π |

该朝向将 ROHand 的 -X 指尖方向对齐 G1 的 +X，将食指侧朝向 +Z，并使两手屈曲分别朝向内侧。89 个链接的组合树与左右五个指尖均已验证；换手前后原身体链接的 FK 最大差约 1.11×10⁻¹⁶。

这些适配件是本项目建立的虚拟模型安装关系，尚未验证真实机械连接、载荷或动力学，也不声称与论文作者装配细节完全相同。ROHand 的掌长、连杆外形仍可能与生成视频中的手有所不同。

## 预览和重建

- [左右五指张合图](../previews/user_g1_five_finger_hands.jpg)
- [实际连杆张合动画](../previews/user_g1_five_finger_demo.mp4)：演示六控制驱动五指，不是已拟合的抓取动作。
- [组合模型验收报告](../reports/user-g1-five-finger.json)

服务器重新构建模型和演示：

```bash
cd /data/KinoGen
export KINOGEN_CUDA_DEVICES=7
source scripts/activate-server.sh vision
python scripts/build_g1_five_finger.py
python scripts/export_user_full_scene.py --assembly-preview
```

几何构建使用 CPU；预览使用已协调的物理 GPU 7。没有启动训练、安装本地 CUDA 或改写共享环境。此前的三指参考图与 v1 结果保留为历史证据；新全身/手部优化与网格导出读取五指模型，不把旧参考图当作五指模型的像素真值。
