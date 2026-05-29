# G1 MimicKit Deployment Design

## Goal

在 RoboJuDo 中从零构建一套可复用的 MimicKit 部署框架，提供两个统一入口：

- `g1_mimickit`
- `g1_mimickit_real`

这两个入口都采用“稳定 locomotion warmup -> 自动切入 MimicKit 动作跟踪 -> 可随时回退到 locomotion”的部署方式。部署框架必须支持把新的 MimicKit 训练结果导入 RoboJuDo，并在不改 Python 代码的前提下完成后续替换使用。

本设计以用户当前资产为首个验证对象：

- `MimicKit/data/models/qkf_model.pt`
- `MimicKit/data/motions/hkf_mimic.pkl`
- `MimicKit/data/logs/lo/`

`MimicKit/data/motions/hkf.pkl` 作为训练相关原始 motion 资产保留，不作为运行时必需输入。

## Scope

### In Scope

- 在 RoboJuDo 中新建 MimicKit 部署 runtime，不复用现有失败尝试代码。
- 导入 MimicKit 训练产物并生成 RoboJuDo 可运行 bundle。
- 在 RoboJuDo 中实现 MimicKit observation 构建、checkpoint 解析、motion 播放和 actor 推理。
- 接入 G1 的 sim2sim 与 sim2real 入口配置。
- 使用 `RlLocoMimicPipeline` 实现 warmup、自动切换、手动回退和 real safety guard。
- 对 sim2real 增加显式准入检查，避免在状态源不满足条件时错误切入 MimicKit。
- 为当前 `qkf/hkf_mimic/lo` 资产完成首轮验证。

### Out of Scope

- 不修改 MimicKit 训练代码。
- 不在本次范围内支持所有 MimicKit 算法。首版只面向 DeepMimic-style tracking policy。
- 不承诺所有 MimicKit bundle 都可直接真机部署。sim2real 是否允许切入取决于 bundle 的观测需求与 real 环境可提供的状态质量。
- 不为旧的 `robojudo/mimickit_runtime` 或旧 `mimickit_policy` 做兼容保留。

## Key Findings

### MimicKit Deployability

MimicKit 不是“原理上只能存在于仿真中”。论文和源码都明确把 Engine 视为可替换后端，理论上可以接入其他 simulator 或真实机器人。真正的部署前提是：部署端必须提供与训练时同构的 observation，并在坐标系、关节顺序、normalizer 和 action 语义上严格对齐。

### Sim2Sim

`sim2sim` 是明确可行的。RoboJuDo 的 MuJoCo 环境能提供 `dof_pos/dof_vel/base_quat/base_ang_vel/base_pos/base_lin_vel`，也能通过 FK 提供 `torso_pos/torso_quat` 和 key body poses，这足以构造 MimicKit tracking policy 所需的部署观测。

### Sim2Real

`sim2real` 不是原理上不可能，但不是无条件成立。若某个 MimicKit bundle 的观测依赖高质量 `root position` 和 `root linear velocity`，则 real 部署必须拥有稳定 odometry 或外部定位来源。RoboJuDo 当前的 real 环境已支持：

- `UNITREE` odometry
- `ZED` odometry
- IMU-derived base orientation and angular velocity
- FK-derived torso pose

因此本框架应当支持 sim2real，但必须带显式准入判断；不满足状态条件时只能停留在 locomotion，不得切入 MimicKit。

## Architecture

部署框架分为四层。

### 1. MimicKit Bundle Layer

所有 MimicKit 训练结果先被导入为 RoboJuDo 标准 bundle。运行时只认 bundle，不直接依赖 MimicKit 原始训练目录。

建议目录结构：

```text
assets/mimickit/g1/<bundle_name>/
  policy.pt
  motion.pkl
  char.xml
  env_config.yaml
  agent_config.yaml
  engine_config.yaml
  deploy_meta.yaml
```

其中 `deploy_meta.yaml` 由导入脚本自动生成，用于固化部署期所需元信息。

### 2. MimicKit Deploy Runtime

在 RoboJuDo 中新建独立目录：

```text
robojudo/mimickit_deploy/
```

该目录仅承载部署逻辑，不引用旧失败实现。内部职责拆分如下：

- `bundle.py`
  - 加载 bundle
  - 校验文件完整性
  - 暴露统一路径与元数据接口
- `checkpoint.py`
  - 从 `.pt` 中动态提取 actor 权重
  - 提取 `obs_norm`
  - 提取 `action_norm`
  - 不依赖固定网络宽度名
- `motion.py`
  - 读取 MimicKit `motion.pkl`
  - 按时间插值取帧
  - 处理 `loop_mode`、`phase`
- `char_model.py`
  - 解析 `char.xml`
  - 获取 DFS joint/body 顺序
  - 完成 dof 与 joint rotation 转换
  - 提供 FK
- `obs_builder.py`
  - 构建 MimicKit 部署 observation
  - 输出 `obs_raw` 和 `obs_norm`
  - 逻辑对齐 MimicKit `compute_deepmimic_obs`
- `actor.py`
  - 基于提取出的线性层做前向
  - 输出 `action_net` 与反归一化后的 action
- `validator.py`
  - 执行运行前检查
  - 评估 bundle 是否可切入 sim2sim/sim2real

### 3. RoboJuDo Policy Layer

新增或重写：

```text
robojudo/policy/mimickit_policy.py
```

职责：

- 从 RoboJuDo `env_data` 收集机器人状态
- 调用 `obs_builder.py`
- 调用 `actor.py`
- 按 MimicKit 的 `zero_center_action` 语义生成 `action_delta`
- 驱动 motion timeline
- 在 motion 结束或异常时输出 callback，通知 pipeline 回退

首版 G1 MimicKit 部署的 canonical root state contract 固定为：

- `root_pos <- env_data.base_pos`
- `root_rot <- env_data.base_quat`
- `root_lin_vel <- env_data.base_lin_vel`
- `root_ang_vel <- env_data.base_ang_vel`

不以 `torso_pos/torso_quat` 作为首版 root state 主来源。`torso_*` 与 `fk_info` 仅用于未来扩展、调试可视化或额外 sanity check，不作为首版 MimicKit observation 的硬依赖。

这里必须严格保证以下对齐：

- observation 维度与顺序
- quaternion 表达与坐标系
- `quat_to_tan_norm`
- `global_obs` / local obs
- `root_height_obs`
- `enable_phase_obs`
- `enable_tar_obs`
- `tar_obs_steps`
- `key_bodies`
- `zero_center_action`
- actor observation normalization
- action normalization 与 clipping
- joint DFS 顺序与 XML 对齐

### 4. Pipeline Entry Layer

新增配置文件：

```text
robojudo/config/g1/policy/g1_mimickit_policy_cfg.py
robojudo/config/g1/g1_mimickit_cfg.py
```

同时必须更新 G1 配置导入注册路径，保证 `run_pipeline.py -c ...` 能解析到新入口。实现阶段至少要修改一处现有 G1 config import 聚合文件，使新模块被导入并完成 `cfg_registry.register`。

最终对外暴露两个入口：

- `g1_mimickit`
- `g1_mimickit_real`

两个入口都基于 `RlLocoMimicPipeline`，采用：

- `loco_policy = G1LocoModePolicyCfg()`
- `mimic_policies = [G1MimicKitPolicyCfg(bundle_name="default")]`

默认 warmup：

- `warmup_steps = 100`
- `warmup_to_mimic = True`
- `warmup_mimic_idx = 0`

`g1_mimickit_real` 额外开启：

- `do_safety_check = True`
- 默认 `odometry_type = "UNITREE"`

现有仓库中的旧 MimicKit 配置与入口不作为首版支持对象。实现阶段必须明确完成以下其一：

- 删除旧 MimicKit 入口的注册
- 或将旧入口重写为显式 deprecated alias，并在日志中提示仅支持新入口

最终要求是：仓库中对外唯一受支持的 MimicKit 运行入口是 `g1_mimickit` 与 `g1_mimickit_real`。

`G1MimicKitPolicyCfg` 的 v1 契约固定为 bundle-first API：

- 必须提供 `bundle_name`
- 可选提供 `bundle_root`

v1 不再以 `model_pt_file`、`motion_file`、`char_file` 作为主配置入口。若为了过渡期保留旧字段，只能作为 deprecated alias，且解析优先级必须低于 bundle lookup，并在日志中明确提示不受长期支持。

## Bundle Contract

### Capability vs Runtime Admission

本设计明确区分两个概念：

- `bundle capability`
  - 导入期静态结论
  - 仅基于 bundle 文件、自描述元信息和 RoboJuDo 环境契约判断
- `runtime admission`
  - 运行期动态结论
  - 依赖当前 env 状态、odometry 有效性、数值健康度和切换时机

导入脚本不允许宣称“当前真机可切入 MimicKit”，它只能输出：

- bundle 是否有效
- bundle 是否具备 `sim2sim` 能力
- bundle 在什么前提下具备 `sim2real` 静态能力
- runtime 还需要满足哪些状态条件

最终是否允许切入 MimicKit，只能由运行期 validator 判定。

### User-Facing Workflow

用户平时只需关心两类核心资产：

- `policy.pt`
- `motion.pkl`

但为了保证部署可靠性，系统内部会要求导入脚本同时读取训练日志目录中的配置文件。运行时不允许只靠手工替换两个文件后直接启动。

### Import Script

新增：

```text
scripts/import_mimickit_bundle.py
```

目标命令形式：

```bash
python scripts/import_mimickit_bundle.py \
  --bundle-name default \
  --policy /path/to/policy.pt \
  --motion /path/to/motion.pkl \
  --log-dir /path/to/log_dir \
  --char /path/to/g1.xml
```

导入脚本职责：

- 默认复制核心资产到 `assets/mimickit/g1/<bundle_name>/`，保证 bundle 导入后自包含
- 可选支持显式 `--link` 模式使用软链接，但不是默认行为
- 若目标 `bundle_name` 已存在，则默认 hard fail；只有显式 `--force` 才允许覆盖
- 读取 `env_config.yaml` / `agent_config.yaml` / `engine_config.yaml`
- 提取并验证 checkpoint 中的 actor 与 normalizer
- 生成 `deploy_meta.yaml`
- 检查 `obs_dim` 和 `action_dim`
- 输出一份导入报告，至少包含：
  - `bundle_valid`
  - `bundle_sim2sim_capable`
  - `bundle_sim2real_capable`
  - `runtime_real_requirements`
  - `why_not`

### deploy_meta.yaml

`deploy_meta.yaml` 至少包含：

- `bundle_name`
- `bundle_format_version`
- `robot`
- `algorithm`
- `runtime_type`
- `obs_dim`
- `action_dim`
- `control_mode`
- `control_freq`
- `sim_freq`
- `global_obs`
- `root_height_obs`
- `enable_phase_obs`
- `enable_tar_obs`
- `tar_obs_steps`
- `num_phase_encoding`
- `key_bodies`
- `zero_center_action`
- `joint_names_dfs`
- `body_names_dfs`
- `requires_root_pos`
- `requires_root_lin_vel`
- `char_file`
- `motion_file`
- `policy_file`

`requires_root_pos` 与 `requires_root_lin_vel` 用于 real 准入检查，不允许在运行期靠猜测。

要求：

- `bundle_format_version` 必须由 importer 写入
- `algorithm` 首版固定为 `deepmimic_tracking`
- `runtime_type` 首版固定为 `g1_mimickit_v1`

运行期 loader 必须在以下情况 hard fail：

- `bundle_format_version` 缺失或主版本不兼容
- `algorithm` 不是首版支持值
- `runtime_type` 不是首版支持值

## Validation Ownership

为避免导入期与运行期行为矛盾，各层职责固定如下：

| Layer | Runs When | Inputs | Hard Fail | Deny Switch | Runtime Fallback |
|-------|-----------|--------|-----------|-------------|------------------|
| `import_mimickit_bundle.py` | bundle 导入时 | 原始 MimicKit 资产 | bundle 文件缺失、checkpoint 无法解析、维度不一致、导入目标冲突、schema 生成失败 | 不适用 | 不适用 |
| `bundle.py` | 启动加载时 | bundle 目录 | 缺文件、schema 不兼容、路径不自洽、`algorithm/runtime_type` 不支持 | 不适用 | 不适用 |
| `checkpoint.py` | 启动加载时 | `policy.pt` | actor 或 normalizer 解析失败 | 不适用 | 不适用 |
| `validator.py` | 启动与每次尝试切入 mimic 前 | bundle meta、env_data | 不适用 | 缺少实时条件时拒绝切换，并给出原因 | 激活后若条件失效则触发回退 |
| `mimickit_policy.py` | observation / action 构建时 | env_data、bundle、motion | 仅对不可恢复的 bundle/logic 错误 hard fail | observation 不可构建时返回 deny/fallback 信号 | 数值异常、motion 结束时回退 |
| `RlLocoMimicPipeline` | 所有切换路径 | validator 与 policy 回调 | 不适用 | 对所有 mimic 激活路径统一执行 gate | 接收回退信号并切回 loco |

切换语义固定：

- 导入期 hard fail 不产生 bundle
- 启动期 hard fail 不允许创建 MimicKit policy
- 切换期 deny 只会阻止切入 MimicKit，不影响 locomotion
- 激活后异常一律优先 fallback to loco

Pipeline integration points 在 v1 必须是显式实现项：

- `RlLocoMimicPipeline.post_step_callback`
  - 处理 warmup auto-switch 触发
- `PolicyInterpManager.switch_to_mimic`
  - 处理手动/自动切换请求
- 任意未来 mimic policy 切换入口
  - 必须复用同一 `request_switch_to_mimic(policy_idx)` gate

要求：

- 不允许在 `switch_to_mimic()` 内直接切入目标 mimic policy
- 必须先经过 shared runtime admission gate
- policy 发出的 fallback callback 必须由 pipeline 统一收敛为 `switch_to_loco()`

## State Contract

`mimickit_policy.py` 对 RoboJuDo `env_data` 的输入契约固定如下：

| Field | Source in RoboJuDo | Frame / Format | Required | Policy Handling |
|------|---------------------|----------------|----------|-----------------|
| `dof_pos` | `env_data.dof_pos` | env joint order, radians | Yes | 先经 `DoFAdapter` 映射到 bundle DFS joint order |
| `dof_vel` | `env_data.dof_vel` | env joint order, rad/s | Yes | 同上 |
| `base_quat` | `env_data.base_quat` | `xyzw`, world/aligned orientation | Yes | 作为 canonical root rotation；obs builder 内负责转换到 MimicKit 需要的表示 |
| `base_ang_vel` | `env_data.base_ang_vel` | world frame, rad/s | Yes | 作为 canonical root angular velocity |
| `base_pos` | `env_data.base_pos` | aligned world position | If `requires_root_pos=true` | 缺失则 deny switch |
| `base_lin_vel` | `env_data.base_lin_vel` | body-local linear velocity | If `requires_root_lin_vel=true` | obs builder 先旋转到 world，再按 `global_obs` 决定是否转 heading-local |
| `torso_pos` | `env_data.torso_pos` | aligned world position | No | 首版不作为 root 主来源，可用于 debug/sanity |
| `torso_quat` | `env_data.torso_quat` | `xyzw` | No | 首版不作为 root 主来源，可用于 debug/sanity |
| `fk_info` | `env_data.fk_info` | body poses in world/aligned frame | No | 首版 observation 不依赖 env FK；key body positions由 char model FK 自行计算 |

关键接口边界：

- env 向 policy 暴露的 quaternion 顺序固定为 `xyzw`
- `base_ang_vel` 视为 world frame
- `base_lin_vel` 视为 body-local velocity
- MimicKit global observation 需要 world-frame linear velocity，因此 obs builder 必须显式完成旋转
- 首版 observation 构建禁止隐式在 `base_*` 与 `torso_*` 之间切换 root source

## Runtime Behavior

### State Machine

统一状态机如下：

1. `prepare`
   - 将机器人拉到稳定 locomotion 初始姿态

2. `LOCO_WARMUP`
   - 先运行稳定 `LocoMode`
   - 不允许立即切入 MimicKit

3. `AUTO_SWITCH_TO_MIMICKIT`
   - warmup 结束
   - 先运行 validator
   - 通过后再插值切入 MimicKit

4. `MIMICKIT_ACTIVE`
   - MimicKit policy 接管
   - 按 motion timeline 推进

5. `FALLBACK_TO_LOCO`
   - 回退到 stable locomotion

### Fallback Conditions

任一情况发生都必须回退到 loco，而不是继续执行 MimicKit：

- motion 播放结束
- 用户手动切回
- 缺少观测字段
- odometry 失效或异常
- 观测出现 NaN / Inf
- actor 输出出现 NaN / Inf
- bundle 与当前机器人配置不匹配
- joint/body 顺序检查失败
- validator 明确拒绝

只有姿态安全检查已经显示机器人危险时，才允许 pipeline 走已有 `do_safety_check` 停机/复位逻辑。

## Activation Gate

所有尝试进入 MimicKit 的路径必须复用同一个 gate，不允许存在绕过 validator 的激活分支。以下路径统一走：

- warmup 后自动切换
- 手动 `[POLICY_MIMIC]`
- 未来多 MimicKit policy 的前后切换

统一入口语义为：

```text
request_switch_to_mimic(policy_idx)
  -> validator.check_runtime_admission(...)
  -> allow: 插值切换
  -> deny: 记录日志并保持 loco
```

因此 `AUTO_SWITCH_TO_MIMICKIT` 只是一个触发方式，不是唯一 gate。

## Sim2Real Admission Rules

切入 MimicKit 前必须全部满足：

- bundle 文件完整
- `deploy_meta` 与 checkpoint 实际维度一致
- 环境控制语义等价于 MimicKit `control_mode=pos`
- 当前机器人 joint names 与 bundle 的 DFS 顺序一致
- `dof_pos/dof_vel/base_quat/base_ang_vel` 可用
- 若 bundle 需要 root position，则 `base_pos` 可用
- 若 bundle 需要 root linear velocity，则 `base_lin_vel` 可用
- 所有输入有限值
- quaternion 可归一化
- base height 在合理范围

不满足时：

- 不允许切入 MimicKit
- 继续停留在 locomotion
- 输出明确拒绝原因

## Implementation Constraints

### Rejected Options

- 不继续沿用旧 `robojudo/mimickit_runtime`
- 不使用“直接单策略 MimicKit 直上”的 `RlPipeline` 入口
- 不新建一套重复的专用 pipeline，除非现有 `RlLocoMimicPipeline` 无法满足需求

### Selected Option

使用现有 `RlLocoMimicPipeline` 作为切换骨架，MimicKit 仅作为 mimic policy 接入。

原因：

- 现成支持 warmup 与回退
- 现成支持手动切换
- 现成支持 sim2real safety check
- 交互模式与 BeyondMimic 一致，用户更容易理解

## File Plan

计划新增或重写的关键文件：

```text
robojudo/mimickit_deploy/__init__.py
robojudo/mimickit_deploy/bundle.py
robojudo/mimickit_deploy/checkpoint.py
robojudo/mimickit_deploy/motion.py
robojudo/mimickit_deploy/char_model.py
robojudo/mimickit_deploy/obs_builder.py
robojudo/mimickit_deploy/actor.py
robojudo/mimickit_deploy/validator.py

robojudo/policy/mimickit_policy.py
robojudo/config/g1/policy/g1_mimickit_policy_cfg.py
robojudo/config/g1/g1_mimickit_cfg.py
robojudo/config/g1/g1_cfg.py or equivalent G1 config import aggregator

robojudo/pipeline/rl_loco_mimic_pipeline.py

scripts/import_mimickit_bundle.py
```

计划新增文档：

```text
docs/mimickit_deploy.md
```

计划新增测试：

```text
tests/test_import_mimickit_bundle.py
tests/test_mimickit_bundle_loader.py
tests/test_mimickit_checkpoint.py
tests/test_mimickit_obs_builder.py
tests/test_mimickit_validator.py
tests/test_g1_mimickit_config.py
```

关键负例必须覆盖：

- bundle 名冲突且未显式 `--force`
- `--force` 覆盖已有 bundle
- `copy` 与 `--link` 两种导入模式
- `deploy_meta.yaml` 缺字段
- `bundle_format_version` 不兼容
- `obs_dim` / `action_dim` 不匹配
- runtime 缺失 `base_pos`
- runtime 缺失 `base_lin_vel`
- runtime 数值异常
- 手动 mimic 激活被 gate 拒绝
- golden-reference observation / action 数值不匹配

## Verification Strategy

### Asset-Level Verification

- bundle 文件齐全
- checkpoint 可解析
- actor 层序列完整
- `obs_dim` / `action_dim` 自洽
- motion 可正常读取
- schema 版本兼容
- `algorithm/runtime_type` 受支持

### Observation-Level Verification

- `obs_raw.shape[0] == obs_norm_mean.shape[0]`
- `obs_norm` 无 NaN / Inf
- 不同开关组合下 observation 仍维度正确

### Golden-Reference Parity Verification

必须至少存在一组 golden-reference 测试，不允许只做 shape-level 验证。

要求：

- 对同一 recorded state / motion time，比较 MimicKit 原始实现与 RoboJuDo deploy runtime 生成的：
  - `obs_raw`
  - `obs_norm`
  - `action_denorm`
- 误差必须在预先定义的数值容差内

推荐以当前用户资产 `qkf_model.pt + hkf_mimic.pkl + lo/` 生成首个 golden reference。

### Sim2Sim Verification

至少验证以下行为：

- `python scripts/run_pipeline.py -c g1_mimickit` 可启动
- warmup 后自动切到 MimicKit
- motion 结束后自动回 loco
- 手动 loco/mimic 切换正常
- 用户当前 bundle `qkf + hkf_mimic + lo` 能跑通

### Sim2Real Verification

至少验证以下行为：

- `python scripts/run_pipeline.py -c g1_mimickit_real` 能完成启动前检查
- odometry 满足条件时允许切入 MimicKit
- 缺少 `base_pos/base_lin_vel` 时明确拒绝切入
- 状态异常时自动保持或切回 loco
- 不做“假通过”
- 手动与自动两条 mimic 激活路径都走同一 gate

## Acceptance Criteria

以下全部成立才视为完成：

- 当前给定 bundle 能在 RoboJuDo 中完成 `sim2sim`
- 存在统一入口 `g1_mimickit`
- 存在统一入口 `g1_mimickit_real`
- `g1_mimickit_real` 带显式准入检查
- 替换 bundle 后无需修改 Python 代码
- 最多只需重新运行一次 bundle 导入脚本
- 所有失败场景优先回退到 locomotion
- `bundle_format_version` / `algorithm` / `runtime_type` 不兼容时能被明确拒绝
- 至少一组 golden-reference 数值对齐测试通过，而不是只验证 shape

## Open Risks

- 不同 MimicKit 训练结果可能使用不同 observation 结构，因此 bundle 导入校验必须足够严格。
- real 环境中的 `base_pos/base_lin_vel` 质量可能不足，导致某些 bundle 只能通过 sim2sim，不应强行宣称可真机部署。
- 当前工作区存在未跟踪的 `unitree_rl_gym/` 目录，后续实现和提交时必须避免误纳入无关内容。
