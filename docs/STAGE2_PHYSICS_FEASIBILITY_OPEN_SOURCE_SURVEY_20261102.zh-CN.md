# Stage 2：Physics Feasibility Refiner 开源方案调研（2026-11-02）

## 结论

当前项目不应再让 RL 专家直接吸收不满足物理合同的 Stage 1 轨迹。建议把 Stage 2 实现为两层：

1. **MuJoCo 物理采样修复层**：优先接入 SPIDER，随后接入 DynaRetarget/SBTO，得到在同一仿真器中可开环滚动的状态和控制轨迹；
2. **显式动力学证书层**：以 Crocoddyl/Aligator 为主，参考 PRIME 的平滑接触模型，联合求解状态、关节力矩和接触力，并输出约束残差及拒绝原因。

单一方法很难同时兼顾现有 HDMI/MJLab 接入速度、长时域 HOI、未知接触模式和严格的力/力矩证书。最快且风险最低的路线是先用 SPIDER/SBTO 解决“能否在 MuJoCo 中执行”，再用显式优化器解决“是否满足可审计的物理合同”。

## 推荐排序

### 1. SPIDER：最适合立刻接入现有 HDMI/MJLab 链路

- 论文：SPIDER: Scalable Physics-Informed Dexterous Retargeting
- 代码：https://github.com/facebookresearch/spider
- 许可：CC BY-NC 4.0，研究使用方便，但商用需要额外确认。
- 与当前项目的契合点：
  - 已提供 HDMI workflow，后端明确使用 MjLab；
  - 支持 MuJoCo Warp、Genesis、IsaacGym，支持 humanoid 和 HOI；
  - 输入是运动学参考、物体运动和场景，输出物理 rollout 下的控制与状态轨迹；
  - 使用大规模并行采样与 virtual contact guidance，专门处理错误接触模式、悬空和接触漂移；
  - 可直接在服务器 GPU 上并行。
- 局限：
  - 主要依赖仿真 rollout 和代价函数，不天然提供显式接触力/动力学等式残差证书；
  - 对长时域、延迟收益的 HOI 可能短视；DynaRetarget 论文专门报告了固定短时域采样在部分长任务上的失败。

适用任务：①②③的落脚/接触切换，④⑤⑧的起始稳定，⑨的首段全身/手腕状态修复。⑦可以先作为 infeasible 检测样本。

### 2. DynaRetarget / SBTO：最适合长时域 HOI 物理修复

- 论文：DynaRetarget: Dynamically-Feasible Retargeting using Sampling-Based Trajectory Optimization
- 代码：https://github.com/Atarilab/sbto
- 许可风险：公开仓库当前没有明确 LICENSE 文件；可用于内部研究验证，但不能默认具有再分发或商用权限。
- 与当前项目的契合点：
  - 官方示例直接读取 OmniRetarget 数据；
  - 使用 MuJoCo 3.3.7、完整机器人碰撞模型和并行 rollout；
  - 优化变量是 PD target 控制序列，输出可执行的状态/控制轨迹；
  - 目标同时跟踪 q、v、躯干、脚、手和对象，并惩罚非预期碰撞；
  - 逐步扩展优化时域，避免短 MPC 只顾眼前，因此更适合搬运、翻越、长时域接触任务；
  - 论文明确对 OmniRetarget 后的轨迹进行动态精炼，并用于后续 RL。
- 局限：
  - 官方模型和配置以 G1/OmniRetarget 数据为主，需要增加 Vita28 的 MJCF、关节顺序、执行器、PD、场景及 NPZ adapter；
  - 当前 Python MuJoCo 版主要使用 CPU 并行 rollout，八张 GPU 不能像 MJX/MuJoCo Warp 那样直接吃满；适合九任务并行，而不是单任务八卡；
  - 仍然更像“同仿真器中的可执行性证明”，不是显式力/力矩可行域证书。

适用任务：带对象或长时域后果的轨迹，尤其⑦ Train vid2，以及未来拿碟子、抬箱、双人交互。

### 3. PRIME：最接近“轨迹＋接触力＋参数”的显式 Refiner

- 论文：PRIME: Physically-consistent Robotic Inertial and Motion Estimation for Legged and Humanoid Robots
- 代码：https://github.com/well-robotics/PRIME
- 许可：BSD-3-Clause。
- 能力：
  - 联合优化机器人轨迹、摩擦接触力和惯性参数；
  - 使用可微的 Anitescu 平滑互补接触模型；
  - 基于 Crocoddyl/FDDP；
  - 自带 Unitree G1 真人和仿真实验、URDF/SRDF/MJCF、结果和可视化。
- 与当前项目的差距：
  - PRIME 的原始问题是从 q/v/u 测量和执行器命令做 full-information estimation；我们只有 Stage 1 的 q/v 和场景，通常没有真实 u；
  - 需要把控制输入也设为待优化变量，并加入参考跟踪、Vita28 力矩限制和任务完成约束；
  - 工程接入比 SPIDER/SBTO 大，但它最适合作为显式物理证书层的参考实现。

### 4. Crocoddyl / Aligator：正式 Stage 2 求解后端

- Crocoddyl：https://github.com/loco-3d/crocoddyl ，BSD-3-Clause。
- Aligator：https://github.com/Simple-Robotics/aligator ，BSD-2-Clause。
- 能力：
  - 浮动基多体系统、接触序列、关节力矩和接触力联合优化；
  - 可加摩擦锥、wrench cone、关节/速度/力矩限制、终端约束；
  - Pinocchio 提供解析刚体动力学与导数；
  - Crocoddyl 有多接触 FDDP 和逆动力学 OCP；Aligator 更适合一般的显式约束 ProxDDP。
- 推荐用法：
  - Stage 1 已能提供接触标签时，先固定接触 schedule，避免一开始就做最难的 contact-implicit；
  - 用 Stage 1 轨迹 warm start，依次求“运动学平滑 → 逆动力学 → 全 kinodynamic TO”；
  - 对接触标签不确定的局部窗口，再引入 PRIME 的平滑互补接触或交给 SPIDER/SBTO 搜索接触模式。
- 代价：需要维护 Vita28 的 URDF/Pinocchio 与 MJCF/MuJoCo 双模型一致性，并把场景代理转换为接触面/碰撞约束。

### 5. cuRoboV2：适合作为高速预处理器，不宜单独承担 Stage 2

- 代码：https://github.com/NVlabs/curobo
- 许可：Apache-2.0。
- 优点：GPU 加速、高自由度 humanoid、B-spline 平滑、逆动力学、力矩限制、ESDF 场景碰撞和自碰撞。
- 边界：它擅长平滑、碰撞和逆动力学约束，但不是面向浮动基多接触平衡的完整 humanoid contact dynamics solver。
- 推荐位置：Stage 1.5 或 Stage 2 warm start，用于去除速度/加速度尖峰、关节限位、自碰撞和场景穿透，再交给 Crocoddyl/Aligator 或采样物理 refiner。

### 6. MuJoCo MPC：方便做局部基线和验证器

- 代码：https://github.com/google-deepmind/mujoco_mpc
- 优点：直接使用 MJCF/MuJoCo；提供 iLQG、梯度下降、CEM、Predictive Sampling，并有 humanoid mocap tracking 示例。
- 边界：短时域 MPC 容易在长时域 HOI 中短视，硬约束和显式可行性证书不如 Aligator/Crocoddyl。
- 推荐位置：用于首个失败窗口的快速局部实验，以及验证 Stage 2 cost 是否合理，不作为最终唯一后端。

## 暂不建议投入的项目

- **CITO**：思想匹配 contact-implicit TO，但依赖旧 MuJoCo 2.00、ROS catkin、SQOPT/SNOPT，仓库也明确仍在开发；迁移成本高。
- **FRDV**：论文的 contact-implicit iLQR 很匹配，但官网明确说明计划发布的只有运动学部分，动态验证代码不发布。
- **KDMR / SPARK / Direct Dynamic Retargeting**：论文思想非常贴合，分别覆盖多接触全身 TO、渐进式 kinodynamic TO 和直接物理 MPC；当前没有可直接使用的正式代码发布，适合跟踪而非作为近期工程依赖。

## 建议的 Stage 2 数学合同

### 输入

- `q_ref, v_ref`：Stage 1 轨迹；
- `contact_schedule`：脚、手、物体、椅子/台阶接触标签及置信度；
- Vita28 MJCF/URDF、质量/惯量、执行器力矩、关节/速度限制、PD 参数；
- 场景和物体碰撞几何、摩擦、质量；
- 任务约束：物体轨迹、末端目标、最终姿态。

### 优化变量

- `q_t, v_t, a_t`；
- 关节力矩 `tau_t`；
- 接触力/力矩 `lambda_t`；
- 物体状态和接触 wrench；
- 必要时优化时间步或接触切换时刻。

### 硬约束

- 全刚体动力学：`M(q)a + h(q,v) = S^T tau + J_c^T lambda`；
- 单边接触、摩擦锥、接触互补或固定接触 schedule；
- 接触期间的无穿透、近似无滑移；
- 关节位置、速度、加速度和力矩限制；
- 自碰撞和机器人—场景穿透限制；
- 质心、ZMP/捕获点或 centroidal momentum dynamics；
- 物体刚体动力学与手/物接触一致性。

### 软目标

- 跟踪 Stage 1 的根、身体、脚、手和物体语义；
- 最小化 jerk、力矩变化、接触力变化和足滑；
- 优先保留高置信接触，允许低置信接触重规划。

### 输出与拒绝合同

- `q, v, a, tau, contact_wrench, object_state`；
- 每类约束最大/均方残差；
- 同一 MuJoCo 模型中的无重置开环/闭环 replay；
- 明确状态：`pass / repaired / replanned / rejected`；
- 拒绝原因例如：`torque_limit`、`friction_cone`、`missing_support`、`collision`、`contact_mode_infeasible`、`task_goal_unreachable`。

## 对九任务的实验顺序

1. **先做 SPIDER Vita28 adapter**：选③ Gallery08 作为接触切换样例、⑨ Train vid1 作为首帧腕部样例。
2. **接 SBTO Vita28 adapter**：先跑③，再跑长时域⑦；⑦应允许输出 `rejected`，验证拒绝合同而不是强行生成轨迹。
3. **实现显式 inverse-dynamics audit**：不改轨迹，只计算每帧所需 `tau/lambda` 和动力学残差，立即筛掉不可行参考。
4. **接 Aligator/Crocoddyl 固定接触 TO**：从④ Gallery14 的起始静态支撑窗口开始，再扩展到①②③的接触切换窗口。
5. **全片验证后再进入 HDMI 专家训练**：Stage 2 未通过的轨迹不得进入 RL 长训。
