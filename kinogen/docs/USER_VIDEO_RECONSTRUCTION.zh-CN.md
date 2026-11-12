# 用户双视角抬箱视频：分割、跟踪、深度与三维候选

2026-10-20，已在 `yz-ai-4090-01.bce:/data/KinoGen` 完成这个视频的全部 121 帧处理。GPU 推理只使用物理 GPU 6；几何拟合、验收和点云导出使用 CPU。本地不安装 Torch/CUDA，不运行训练或视频生成 API。

这次跑通的是公开视觉组件与自行实现的条件三维基线。它产生可检查的点云、脚部稀疏三维点和箱体轨迹，**尚未完成全身/手部关节优化，也不是完整 KinoGen 论文复现或可直接训练的运动参考**。

## 输入与水印

- 输入：`data/video_generation/candidate_scene/generated.mp4`。
- SHA256：`035c5852f9580cbbf5d612e2ad93a7f80065426a34ce355fb87db43876b20d6b`。
- H.264，1280×720，24 FPS，121 帧，约 5.04 秒；左右视图分别为 640×720。
- 原视频保留不变；处理数组保留原始 RGB。水印采取观测排除，不使用生成式修补。
- 最终水印方案为 v3：从第 120 帧纯色背景提取实际标志，用高通模板和多尺度相关定位，再加 6 像素边界；67 帧有候选水印，27 帧被标记为低置信候选。平均排除面积约 0.94%。弱水印未检出不等于保证不存在水印。
- SAM、深度有效区域与最终水印掩码逐像素一致；跟踪的有效点避开水印。早期错误掩码结果已弃用，最终所有几何结果绑定 v3 输入哈希。

## 实际结果

| 步骤 | 完成情况 | 可用范围 |
| --- | --- | --- |
| SAM3 | 121 帧×2 视图，机器人和箱子均有有效分割，共 242 个视图观测/类别 | 逐帧 image/eager 推理；没有冒充 SAM3 视频跟踪 |
| TAPNext++ | 71,107 个有效二维点观测；两视图初始独立查询分别 378、243 个 | 模型可见性、分割与水印过滤；共同三维点还要求正反向跟踪一致 |
| 稀疏双视角 | 37 个共同脚/踝表面点；4,394/4,477 次三维观测通过检查；重投影中位数 0.537 px | 所有帧均有有效点；仅脚部，不代表全身三维重建 |
| Video Depth Anything Large | 全部 121×2 帧、518 输入尺度，原始深度全部有限 | 原始输出为相对深度，不能直接当米 |
| 条件深度尺度 | 每视图全片固定逆深度映射，以已知桌面/地面做空间分组留出验证 | 左/右中位误差 6.07/5.49 cm，P90 20.25/13.46 cm；不能推导机器人/箱体同精度 |
| 已知尺寸箱体拟合 | 每帧双视角轮廓约束、六自由度姿态；121 帧图像拟合通过，64 帧严格几何检查通过 | 其余 57 帧保留无效标记，严格有效位置数组为 NaN；没有强制预设抬升距离 |
| 同时刻点云 | 第 0、60、120 帧，每帧融合两个视图，约 2.1 万点/帧 | 彩色 PLY；未将运动序列堆成静态物体，未做可信表面网格认证 |

箱体的已知尺寸为 0.28×0.34×0.24 m。候选最大抬升约 14 cm，独立单视图反解中心的分歧中位数约 9.1 cm。轮廓拟合很好不能证明物理三维正确；该分歧也不是可校准的误差置信区间。深度拟合中右视图桌面存在约 -11.3 cm 的中位偏差，点云中还能看到两个视图的箱体表面错位。

相机使用参考场景的物理参数及实际测得的参考图到生成视频的图像变换 `P_video=H@P`。全片静态背景漂移已检查，但生成视频与参考相机的兼容性仍是条件假设。源机器人为公开 G1 三指手模型，并非论文 ROHand 装配。生成视频首帧已改变手臂姿态，因此不能把所有参考 q0 表面点直接当三维真值；独立视图采样点也从未按数组索引冒充左右对应点。

## 文件

服务器完整结果位于 `/data/KinoGen/outputs/user_reconstruction/`：

- `input.npz`：原始双视角 RGB、最终水印无效区域、帧号和时间戳。
- `segmentation.npz`：机器人/箱体掩码、分数、原始掩码 QA 副本和有效区域。
- `tracking.npz`：逐帧二维点、可见性、有效标记及正反向误差。
- `sparse_stereo.npz`：共同脚部点三维位置、有效标记、重投影和极线误差。
- `depth.npz`：原始相对深度。
- `metric_depth.npz`：条件相机 Z 深度候选及支持/有效区域。大数组保留服务器。
- `geometry.npz`：相机、参考几何、初始配准、共同点身份、静态表面与相机漂移诊断。
- `box_trajectory.npz/json/ply`：六自由度箱体候选、图像/严格有效层与不确定性诊断。消费者必须检查 `valid`，不能忽略无效帧。
- `cloud_frame_000.ply`、`cloud_frame_060.ply`、`cloud_frame_120.ply`：三个时刻的同帧融合候选点云，本地有校验副本。

本地查看：

- [分割](../previews/user_segmentation.mp4)、[点跟踪](../previews/user_tracking.mp4)、[相对深度](../previews/user_depth.mp4)、[箱体叠加](../previews/user_box_reconstruction.mp4)。箱体视频的 1280×840 尺寸包含底部 120 像素诊断面板。
- [条件深度](../outputs/previews/user_metric_depth.mp4)、[三时刻点云图](../outputs/previews/user_metric_pointcloud.jpg)、[首中末帧总览](../previews/user_reconstruction_review.jpg)、[水印观测排除检查](../previews/user_watermark_review.png)。
- [总验收报告](../reports/user-reconstruction-summary.json)、[箱体逐帧报告](../reports/user-box-reconstruction.json)、[深度留出验证](../reports/user-metric-depth.json)。

## 重跑与验收

在 VS Code Remote SSH 的服务器终端运行。入口只适用于上述 SHA 锁定的视频，遇到不同输入会拒绝沿用水印和标定。

```bash
cd /data/KinoGen
# 默认仅 CPU 验收，不重新跑模型
bash scripts/run_user_reconstruction.sh verify

# 重新生成全部中间结果；先确认物理 GPU 6 空闲
bash scripts/run_user_reconstruction.sh all
```

`all` 按顺序执行水印准备、SAM3、参考几何与相机漂移检查、共同点初始化、VDA、TAPNext++、箱体拟合、深度尺度/点云及最终验收。不会安装软件或修改共享环境。最终验收重新核对来源哈希、掩码一致性、跟踪避开水印、三角化阈值、箱体无效帧传播、13 个数据文件和 5 个视频的完整解码，生成 `reports/user-reconstruction-summary.json`。

Windows 回传小型结果时，使用带分块/完整 SHA 检查的 SSH 工具。大深度数组继续保留服务器：

```powershell
.\.venv\Scripts\python.exe .\scripts\pull_reconstruction_artifacts.py outputs/previews/user_tracking.mp4
```

尚未完成的论文步骤是全身/手部先验联合优化、可靠的稠密双视角对应、接触及物理一致性优化和策略跟踪。本次未将这些步骤标为通过，也没有把轮廓基线标为 FoundationPose 注册。
