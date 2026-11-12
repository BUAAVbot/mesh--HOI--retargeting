# G1 双五指三维重建第三阶段

计算位于 `yz-ai-4090-01.bce:/data/KinoGen`，Windows 仅收集报告、轨迹与视频预览。第三阶段不覆盖第一、第二阶段的源视频、分割、跟踪、深度、相机、箱体和轨迹快照。

## 来源与实际优化

机器人仍为 G1 的 29 个身体关节和左右五指 ROHand 的 12 个独立控制量。厂家连杆公式展开为 50 个混合单位的指部状态；连同身体共 79 个状态、89 个链接，不是 79 个独立自由度。虚拟腕部安装座和 ROHand 外形尚未验证与生成视频完全一致。

地面阶段以冻结的第二阶段身体 C、Hcontact 实际腕角和手指控制为初值；基座、腿和手臂共同优化。训练使用实际视频身体轮廓、近似人工/LK 肩肘腕点和 TAP 脚部材料点，脚底使用原始足部网格。每第五个 TAP 物理点 ID、固定空间分块轮廓留出均保持一致，不按候选残差重配点或删除坏点。留出参与候选选择，属于诊断验证，不能当成新的独立测量。

地面完整 121 帧首轮通过 116 帧；原门槛下的一次局部重试提高到 120 帧。第 26 帧的固定手指候选虽然能消除穿地，却使真实手腕点偏差超标，未采用。独立 XML FK 审计全部 121 帧，误差最大 `5.55e-16`；固定材料点投影检查通过。初始地面结果及失败搜索均保留，最终结果以以下总报告为准。

第 54 帧随后在新地面轨迹上重新求解手臂和手指。没有把旧世界坐标的角度差直接叠加到新身体上，也没有对失败帧填入插值。全片手部进一步对每侧真实腕部三轴和六个厂家控制量作一次图像优化，140.39 秒完成；83 个帧侧被选择，其中 40 个严格通过图像指标、43 个仅条件改善。缺测或不通过选择时精确保留同帧父级状态。

最后第 26 帧的 47 变量联合试验在首初值、两轮外循环、32.60 秒内找到通过原地面、身体、腕、TAP 与脚部轮廓门槛的候选。双脚最低约 +0.483 / +0.412 mm，真实腕误差约 0.505 / 0.362 px；其他 120 帧的实际身体、基座、手指控制逐值不动。其右手新域留出由 5.740 降至 4.606 px，仍高于 4.5，图像严格标记保持 false；左手该帧缺测，且距箱约 72 mm，不能宣称双手抓住箱体。第 54 帧同样保留约 4.604 px 的略超门槛结果。

## 图像与几何指标

手部新观测域由源 RGB 的灰色金属像素、SAM 机器人连通区域、实际腕掌 ROI 构建。模型投影不参与挑选源像素。优化保留远距离梯度，最终图像验收沿用原加权训练指标 `<4 px` 和每轮廓 12 点代理的空间留出均值 `<4.5 px`；16 点代理、未加权误差与完整可见网格指标分别披露，不混用数值定义。

完整场景的 z-buffer 另行识别实际可见手部，在相同 RGB 观测域内对比冻结第二阶段与最终第三阶段的可见掩码 IoU 和外部轮廓距离。此指标用于发现代理拟合与实际外观的差异，不暗中替换原验收门槛。原机器人完整网格顶点、手部每个三角中心另做 CPU 几何复核，与渲染简化网格独立。

地板仍为场景原有 `z=0`，相机和 `.28×.34×.24 m` 箱体保持冻结。脚踝表面的跟踪点不是脚底点，不能据其高度移动地板。箱体仍仅 56/121 帧严格有效；双视角生成的一致性、尺度标定和遮挡可观测性仍有不确定性。

最终必要几何检查的超过 5 mm 穿地帧由 120 降为 0，手箱穿透帧由 1 降为 0；全片手部图像有效侧数由左 8 / 右 11 提高至左 20 / 右 25，其余不强行标为有效。原身体 29 关节的明确正速度限制全部通过。ROHand 原 URDF 的速度/力矩字段为 0 占位，不能当成已测定的硬件参数；指部检查限于位移/角度界和厂家连杆耦合，不宣称验证了手部动力学。

**完整可见网格检查发现了重要失败，第三阶段不能视为图像重建完成。** 第 54 帧原 RGB 域包含 1588 个可见灰手像素，模型却被箱体遮挡，可见像素为 0。全片右手共有 9 个缺失可见轮廓的帧（32、33、53、54、56、57、59、61、62），其中 54 是本轮避碰新增的问题，其余 8 帧第二阶段也已存在。原简化轮廓代理未考虑这一遮挡，零穿透并不足以恢复画面。[独立可见性失败说明](../reports/user-v3-visible-hand-diagnostic.json)与完整渲染图保留该证据，后续遮挡修复使用独立第四阶段文件，不覆盖本阶段。

## 最终文件与验收入口

最终统计见 [第三阶段总报告](../reports/user-reconstruction-v3-summary.json)；完整独立检查见 [CPU 独立审计](../reports/user-v3-independent-audit.json)；逐阶段来源见 [最终合并来源](../reports/user-v3-canonical-provenance.json)。最终渲染及可见手部对比见 [网格与动画报告](../reports/user-full-scene-v3.json)。

本地预览为 [完整五指场景](../previews/user_v3_full_mesh.mp4) 和 [原视频双视角叠加](../previews/user_v3_full_overlay.mp4)，均为 121 帧、24 FPS。逐帧字幕显示箱体严格状态、手部图像状态、手箱穿透和原始网格最低地面高度；失败不会因多数帧通过而隐藏。

服务器 `outputs/user_reconstruction_v3/` 保存最终 `body_trajectory.npz`、`hand_trajectory.npz`、`full_trajectory.npz`，以及 `scene_animated.glb` 和三个 `scene_frame_000/060/120.glb`。动画含全部实际 121 个时刻的刚体链接变换；帧间插值只用于预览。四个 GLB 留在服务器，本地只回传预览与小型数据、报告，文件完整性见 [本地交付回执](../reports/user-v3-local-delivery.json)。

最终 canonical 身体吸收实际手腕角，按最终厂家控制量重算 89 链接 FK；手部绑定实际身体 SHA。原阶段文件仍为只读父级，来源哈希没有循环。新增身体角和改变腕部姿态的同侧手部观测支持均保守清为未知，不把良好的二维轮廓拟合升级为三维观测。

服务器复核已生成的最终结果：

```bash
cd /data/KinoGen
source scripts/activate-server.sh vision
CUDA_VISIBLE_DEVICES= python scripts/verify_user_reconstruction_v3.py \
  --body-file outputs/user_reconstruction_v3/body_trajectory.npz \
  --hand-file outputs/user_reconstruction_v3/hand_trajectory.npz \
  --full-file outputs/user_reconstruction_v3/full_trajectory.npz \
  --scene-report reports/user-full-scene-v3.json \
  --provenance-report reports/user-v3-canonical-provenance.json
python scripts/summarize_user_reconstruction_v3.py
```

质量未通过时独立验收退出码为 3，并如实写报告。`strict_metric_reconstruction_valid=false` 和 `physical_hardware_validated=false` 始终保留；零样本穿透不等于连续三角面无碰撞、双手可靠抓取或动力学可行。未执行物理策略跟踪、力矩验证、实机或官方完整 KinoGen 算法验证。
