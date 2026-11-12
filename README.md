# 视频 → Mesh → HOI → Retargeting → 动力学轨迹

本仓库交接从单目/多目视频到机器人动力学轨迹的工程代码。大型数据、模型权重、授权人体模型和实验输出不进入 Git。

## 流程

```text
视频与相机
  → 人体/物体/场景 Mesh
  → 统一 HOI 数据与接触窗口
  → GMR / HOI-Retarget / OmniRetarget / UMR
  → 运动学门禁
  → HDMI / SONIC / MuJoCo / Isaac 动力学
  → 独立重放、审计和轨迹导出
```

## 目录

- `kinogen/`：视频、Mesh、HOI、机器人资产适配、动力学与数据导出脚本。
- `umr/`：UMR 表面对应学习、场景感知重定向、配置与测试。
- `hdmi_adapters/`：UMR/HOI 到 HDMI motion 数据的 Vita 适配器和检查工具。
- `workflows/batch_sim_expert/`：案例检查、任务 DAG、候选排序和批处理自检。
- `workflows/stage2_physics_compare/`：SPIDER/SBTO/PRIME 等动力学修复与严格重放。
- `workflows/hdmi_vita_tracking/`：Vita/mjlab 轨迹准备、动力学记录和轨迹优化。
- `docs/`：接口、机器人合同、UMR、HDMI 和 Stage 2 说明。

## 最小检查

```bash
python workflows/batch_sim_expert/selftest.py
python -m pytest -q umr/tests
python -m pytest -q workflows/stage2_physics_compare/test_common.py
```

完整任务必须另外提供视频、相机、SMPL-X/MANO、物体与场景 Mesh、机器人模型、检查点和配置。见 [交接说明](docs/HANDOFF.zh-CN.md)。

## 当前边界

- UMR 输出首先是运动学参考，不代表动力学成功。
- 批处理已有双人座椅等案例适配，接新任务仍需明确人数、有效帧和场景合同。
- 独立重放必须禁止活动段 root/state 强写和外力托举。
- `projects/hdmi` 上游框架未整体复制；这里只交接本项目的 Vita 数据适配与验收代码。

## 第三方

`umr/LICENSE` 为 UMR 的 MIT 许可证。HDMI、Isaac Lab、MuJoCo、SMPL-X、MANO 等依赖须按各自上游许可证和下载方式获取。
