# 用户双视角视频：全身、五指手与箱体 v2

本流程在服务器 `yz-ai-4090-01.bce:/data/KinoGen` 上计算，本地只预览视频和三维导出。输入仍是同一段 121 帧、24 FPS、1280×720 的 AI 生成视频，左右各 640×720；源视频 SHA256 为 `035c5852f9580cbbf5d612e2ad93a7f80065426a34ce355fb87db43876b20d6b`。不改原视频、不训练，也不调用付费视频 API。

这是透明的项目自实现、已知机器人模型与箱体约束下的三维候选拟合。官方完整 KinoGen 重建集成未发布；本结果不等同论文完整复现、真实物理标定或硬件可执行动作。

## 当前验收状态

最终 contact 版本已完成真实 121 帧 CPU 总验收：来源合同、官方耦合、位置限位、独立 XML FK、全部导出文件 SHA 与两段视频完整性通过；原 G1 明确给出的正速度上限也全部通过。**物理质量仍未通过**：第54帧右手最大穿箱83.938 mm；原始机器人网格仍有120/121帧穿地超过5 mm，最深29.995 mm。保留严格无效标志，不能称为物理可执行或双手接触恢复成功。

独立 baseline 也保留：左腕 yaw 最大35.664 rad/s，超过原URDF的22 rad/s；原始52个手部visual的全部去重顶点与三角中心最大穿箱97.906 mm，顶点超过5 mm比例中位数10.282%，三角中心10.303%（旧简化渲染网格为97.672 mm/8.148%）。contact同口径的顶点/三角中心穿透比例中位数都降为0，但第54帧失败没有被插值或抹掉。

| 部分 | 已知状态 |
|---|---|
| 相机 | 静态桌面/地面真实匹配与空间留出检查；左视图留出 RMS 1.782→0.977 px，右视图独立样本不足，保留原相机。另有未参与相机拟合的脚点审计。物理标定仍未证明。 |
| 身体 | 完整121帧图像候选有效；contact接受57个右臂分支，基座/腿/腰逐值不动。399个新分支角明确为未知误差NaN且图像支持为false；最终2262/3509个身体角有支持，1247个未观测。 |
| 活跃箱体 | 固定尺寸 0.28×0.34×0.24 m，121/121 可见轮廓有效，56/121 通过原严格几何门槛，65 帧仍保留失败标志。相机与箱体冻结。 |
| 独立箱体 v3 候选 | 完整 121 帧，严格有效 59，所有帧验证像素误差不高于冻结 v2；联合优化未收敛，平滑度反而较差，未替换活跃箱体。 |
| 双侧五指手 | 最终contact完整121帧实际优化，接受15个侧候选和9个yaw修正；严格有效侧左7/右1，两侧同时严格有效0帧。342/6050个手部状态有支持、5708个不确定；控制量支持81/1452、不确定1371。较低观测计数如实保留。 |
| 完整 GLB、骨架与视频 | contact两段视频各121帧、三个静态GLB和完整121帧动画GLB均验收。89链接独立XML FK最大误差5.55e-16，动画回读变换误差8.18e-8；物理质量仍失败。 |

## 模型和数据合同

机器人是 G1 身体加双侧 OYMotion ROHand 五指手，模型为 `data/robots/g1_rohand_five_finger/g1_rohand.urdf`，SHA256 `71bc895263e495a3ea01bbb9626458ac5a15a81387bf6f9a8afaa1e700bff7b6`。腕部安装适配属于本项目组装，尚未经过实物验证。

每手只有 6 个独立控制量：5 个滑块位移（m）和 1 个拇指根部角（rad）。双手 12 个控制量通过未修改的 `dependencies/rohand-urdf/scripts/FingerMathURDF.py` 展开为 50 个混合单位 URDF 状态。完整 `joint_positions[121,79]` 是 **29 个身体角 + 50 个耦合手部状态**；不能称为“79 个独立自由度”。身体 29 个角与手部 12 个控制量合计 41 个独立关节控制量，浮动基座位姿另存。

世界坐标为米、Z 向上、X 为机器人前方、Y 为机器人左侧。相机使用 OpenCV 的 u 右、v 下、z 前约定，`P_refined = H @ K @ [R|t]`。静态相机微调没有使用动态箱体或身体拟合残差。

标准输入均在 `outputs/user_reconstruction_v2/`：

- `camera_refinement.npz`：固定双视图投影、K/R/t/H、逐帧相机有效性。
- `body_trajectory.npz`：浮动基座、29 个 rad 角、89 链接变换、`candidate_valid` 与 `joint_image_supported`。身体文件内的指部变换只是模型默认姿态。
- `box_trajectory.npz`：固定尺寸箱体中心、xyzw 四元数、刚体变换、图像和严格几何有效性。严格无效帧的 valid-only 坐标为 NaN。
- `hand_trajectory_wrist_refined.npz`：12 个控制量、50 个耦合状态、`observed/uncertain`、接触冲突及显式身体修正。只允许左右 `wrist_yaw_joint` 改变，其他 27 个身体角逐值保持不变；未接受修正的侧，其 yaw 也保持原值。

有限数组只用于保留可渲染候选，不能表示该关节已被观测。身体、手部、接触、箱体各自的有效性和不确定性必须保留。

## 服务器运行

默认只做 CPU 验收，不占 GPU：

```bash
cd /data/KinoGen
bash scripts/run_user_reconstruction_v2.sh preflight
bash scripts/run_user_reconstruction_v2.sh verify
```

需要重新计算时，`body` 从完整图像拟合开始，再按实际图像修复 59–120 帧；`hand` 在冻结身体上拟合双侧五指和显式 yaw；`export` 使用已协调的物理 GPU7 渲染。`all` 顺序执行这三步并验收。所有计算都在服务器进行，默认 `verify` 与 `preflight` 为 CPU；0–5 号 GPU 不在流程中使用。

```bash
bash scripts/run_user_reconstruction_v2.sh body
bash scripts/run_user_reconstruction_v2.sh hand
bash scripts/run_user_reconstruction_v2.sh export
bash scripts/run_user_reconstruction_v2.sh verify
# 或全部重算：bash scripts/run_user_reconstruction_v2.sh all
```

重算身体会使旧手部和完整场景的来源 SHA 失效，须继续重算后续阶段。总入口不会重新下载模型或覆盖 v1 输入，也不会自动把箱体 v3 候选替换为活跃箱体。

`all` 只重算身体、手部和场景导出，**复用本次冻结的相机与箱体**，不重新执行相机或箱体估计。`preflight` 和最终验收都强制相机 SHA `813f64ca29c7199cb033c020e6d9b8ada1bc2c77313946baf53585fe50247b5a`、箱体 SHA `a9c5b1f274997f86382f24ba09a67abe84ae4231370888f05f254a2d49c415fa`。身体/手部/导出允许重新生成，但其相互来源图必须全部使用当前实际 SHA。另行估计相机或箱体属于新运行，不能直接混入此冻结交付。

## 验收与预览

`verify_user_reconstruction_v2.py` 检查源视频与所有来源 SHA、121 帧时间轴、相机一致性、身体与手部限位、每个控制量的原始执行器范围、全部官方耦合值，以及显式 yaw 之外的身体角不变。它使用独立 XML FK 重算全部帧的组合骨架，检查实际五指网格与骨架，再完整解码两段视频。它同时检查缺失观测与不确定性没有被有限参数或导出步骤抹掉。

最终导出位置：

- `outputs/user_reconstruction_v2/full_trajectory.npz`：组合后的 79 个混合单位状态、12 个手部控制量、基座、骨架和各部分有效性。
- `outputs/user_reconstruction_v2/scene_frame_000.glb`、`060.glb`、`120.glb`：首、中、末帧真实模型网格。
- `outputs/previews/user_v2_full_mesh.mp4`：独立视角完整机器人与箱体网格。
- `outputs/previews/user_v2_full_overlay.mp4`：投影回两幅源图的模型轮廓。
- `outputs/previews/user_reconstruction_v2_review.jpg`：首、中、末帧验收拼图。
- `reports/user-reconstruction-v2-summary.json`：CPU 验收结果，保留 `strict_metric_reconstruction_valid=false`。

baseline 的手部观测/不确定性、穿透与腕速度失败已记录在总验收报告。脚本只对身体29关节中原URDF明确给出正速度上限的项判定超速；ROHand被动continuous关节缺失上限、执行器velocity=0占位均列为未指定，不能误判手部不可运动。12个原执行器的位置范围与全部官方耦合仍逐值严格检查。

合同通过但物理质量失败时，Python验收器返回非零状态3，并保持 `strict_metric_reconstruction_valid=false`。即使以后速度与穿透必要检查通过，也没有进行动力学、力矩、完整自碰撞或实机验证。

## 独立速度与接触修正

修正不覆盖 baseline。输入为 `body_trajectory_contact.npz`（C）和 `hand_trajectory_contact.npz`，父级为冻结身体 D 与独立 `hand_trajectory_velocity_refined.npz`。C只允许14个手臂角改变；基座、腿部和腰部相对D逐值保持。C内的指部变换使用显式父级H_velocity的固定12控制量/50耦合状态，不是D的q0指角，也不是C新估计出的手指运动。newH的非yaw身体角须逐值等于C，所有新增来源SHA必须可追溯。

```bash
source scripts/activate-server.sh vision
CUDA_VISIBLE_DEVICES= python scripts/verify_user_reconstruction_v2.py --run-variant contact --preflight
# root导出完成后：
CUDA_VISIBLE_DEVICES= python scripts/verify_user_reconstruction_v2.py --run-variant contact
```

也可显式提供 `--body-file` 和 `--hand-file`，但不能跳过所选来源图。contact 导出使用 `full_trajectory_contact.npz`、`scene_contact_frame_000/060/120.glb`、`user_v2_contact_full_mesh.mp4` 与 `user_v2_contact_full_overlay.mp4`；独立报告为 `reports/user-reconstruction-v2-summary-contact.json`，审查拼图为 `outputs/previews/user_reconstruction_v2_contact_review.jpg`。最终C SHA为下方所列159cabb...，newH SHA为 `4f8f7c16df560d785c515fa8462c7de7243e93e1211c351ed6028b20a93ca9a1`，都来自完整121帧实际优化。

最终独立CPU检查确认：左右腕yaw最大分别为21.999999990与20.410499120 rad/s，全部正G1速度界通过；原始52个手部visual的顶点最大穿箱83.938172 mm，三角中心最大83.650998 mm，超过5 mm的失败帧只有54。双方最近表面距离逐帧保留，仍不证明双手真实接触。箱体仍仅56/121帧严格有效，手部严格有效没有被接触或有限候选值提升。完整质量失败时验收器返回非零3；该状态是如实验收结果，不能改门槛制造通过。

接触验收除了核对渲染报告，还直接用原URDF的52个手部visual（各自scale与origin）以及已独立XML FK核对的骨架，CPU重算全部去重顶点和每个三角中心的箱体SDF。它同时记录每侧最近表面样本距离；零穿透不能证明两只手真的接触箱体，更不能证明动作能抬起箱子。原始样本检查也不是连续三角形相交或实物接触证明。

完整修正阶段入口（依赖已准备的baseline D及冻结相机/箱体，服务器执行）：

```bash
bash scripts/run_user_reconstruction_v2.sh velocity-hand
bash scripts/run_user_reconstruction_v2.sh contact-body
bash scripts/run_user_reconstruction_v2.sh contact-hand
bash scripts/run_user_reconstruction_v2.sh contact-export
bash scripts/run_user_reconstruction_v2.sh contact-verify
# 或顺序执行：bash scripts/run_user_reconstruction_v2.sh contact-all
```

`contact-body` 先运行全121帧，然后按真实SHA归档第一遍NPZ及匹配报告，仅对报告中仍有碰撞失败的帧再搜索一次（实际手部ROI训练边缘权重0.65，原留出与碰撞门槛不变），最后运行只补元数据的finalize步骤。此次实际重试帧为32、47、54、57、62、78、79，第一遍归档为 `body_trajectory_contact_firstpass_e1a3fa8447f8.npz` 与 `user-arm-contact_firstpass_e1a3fa8447f8.json`；最终C SHA为 `159cabb128e6f2d2f1f182e747c77f51ab6b25c79cfc22df9e49a1de46fb6f66`。新运行根据自身报告选择失败帧，并保存自己的SHA归档，不复用旧结果。随后newH绑定最终C的SHA。左手近表面三关键帧诊断不进入此默认完整流程，也不覆盖C。`contact-all` 的最终质量未过会返回非零，不能以文件存在或优化器退出来表示通过。

地面另做独立检查：以已知场景地板z=0、5mm阈值，用原机器人visual全部去重顶点及已验证FK重算最低z；另保留简化renderer的最低z曲线并注明来源。baseline与contact原始网格最低z范围均约[-29.995,-3.535]mm，120帧超过5mm；baseline简化网格范围约[-29.94,-3.52]mm。C固定基座/腿/腰，没有做feet/floor适配或物理地面标定；即使手部速度和多数帧穿箱改善，也不能将整体物理质量宣称通过。

动画导出还包括 `scene_animated_contact.glb`（baseline若有则为`scene_animated.glb`）：全部121个实际24FPS时刻的每link/箱体刚体TRS，带固定Z-up→glTF Y-up根变换。导出器逐节点回读全部帧变换并用通用glTF加载器检查，验收器核对其来源报告与文件SHA。帧间插值只用于预览，不表示物理仿真或79个独立自由度。旧baseline不强制补动画文件；contact最终交付须有对应动画与audit。

## 本地交付

按用户“本地只预览视频”的要求，本地保留五指URDF、完整NPZ轨迹、视频、拼图和报告；四个新的GLB保留在服务器 `/data/KinoGen/outputs/user_reconstruction_v2/`。服务器CPU验收已逐个核验这些GLB的SHA，不能把服务器文件误称为已下载到本地。Windows的 `scripts/verify_local_reconstruction_delivery.py --run-variant contact --preview-only` 仅使用标准库，实际校验17个本地文件、列出4个服务器GLB，状态为 `local_preview_delivery_sha256_passed`。完整清单为 `reports/user-local-delivery-v2-contact.json`；此状态只表示交付完整性，不提升几何或物理质量。
