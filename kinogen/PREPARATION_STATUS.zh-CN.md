# KinoGen 依赖准备状态

核验时间（UTC）：2026-10-20T01:14:41.394271+00:00

本报告仅核对下载计划文件的大小与哈希，本校验脚本不执行推理。另行完成的模型推理及组件验收见 RUNTIME_STATUS.zh-CN.md；此处清单不含后续单独部署的 SAM3、MANO 等授权模型。

## 公开权重

| 组件 | 完整文件 / 计划文件 | 完整文件大小（MiB） |
| --- | ---: | ---: |
| Kimodo | 20 / 20 | 2163.3 |
| Video Depth Anything | 2 / 2 | 1467.1 |
| FoundationPose ONNX export (not original PyTorch checkpoint) | 3 / 3 | 126.0 |
| WiLoR | 5 / 5 | 2497.3 |
| TAPNext++ | 1 / 1 | 2415.0 |
| Kimodo text encoder MNTP adapter | 9 / 9 | 168.8 |
| Kimodo text encoder supervised adapter | 3 / 3 | 160.1 |
| Video Depth Anything small public alternative | 2 / 2 | 111.0 |
| FoundationPose original PyTorch | 4 / 4 | 246.5 |

## 官方源码

| 组件 | 状态 | 固定提交 |
| --- | --- | --- |
| sam3 | verified_source_snapshot | 7f473cfd983d5eb43040ff608f66bff0d7bdcb53 |
| tapnet | verified_source_snapshot | 730cda1c730877cfedbe01bf87fb1cadb78a565d |
| video-depth-anything | verified_source_snapshot | 4f5ae23172ba60fd7bc11ef671cca678842c7072 |
| kimodo | verified_source_snapshot | e1c219c99b4b94ba48be1fb6a1cbc64ecee4c954 |
| wilor | verified_source_snapshot | fcb911312a38fa8badd30d9656a167485d61b8f9 |
| foundationpose | verified_source_snapshot | a1b694b83e633c2cb6115b9063d940a687759392 |
| dno-reference | verified_source_snapshot | 8f84dc832d3057063c5adaa3c283c0b0a0d3a2bc |
| isaacsim | verified_source_snapshot | 87c95274f50836068b44eeb030bf62aa2108bdf8 |
| unitree-ros | verified_source_snapshot | 5994d4faef0a9cadd3287f8de0199a67eeb2a259 |
| rohand-urdf | verified_source_snapshot | 4d0157f8ba732a8332300619697a646ef7f6c36a |
| pytorch3d | verified_source_snapshot | 88e182f989c80836f4bd744e0d9cb1852762ce01 |
| nvdiffrast | verified_source_snapshot | 253ac4fcea7de5f396371124af597e6cc957bfae |
| chumpy | verified_source_snapshot | 580566eafc9ac68b2614b64d6f7aaa84eebb70da |

逐文件源码快照会另列省略的大媒体、子模块、符号链接或 LFS；完整范围见 `manifests/sources/`。厂商基础模型与论文中的机器人组装并不等价。

isaacsim：已补齐 1099 个 LFS 文件，未补齐 0 个。

## 未完成传输

公开权重计划中的文件已全部完成。

账号、人工批准、许可和托管视频服务见 [访问条件](ACCESS_REQUIREMENTS.zh-CN.md)。网络中断与授权缺失分别记录。

全部逐文件校验结果见 `reports/dependency_inventory.json`。KinoGen 官方集成与版本选型仍待发布。
