RoboMimic -> RoboJuDo 迁移复盘

目的
- 记录 locomode 与 motion_tracking 迁移过程、关键改动与验证点
- 便于后续全局迁移复盘与对齐


一、LocoMode 迁移

1. 迁移目标
- 使用 RoboMimicDeploy_G1 的 locomode 推理与观测逻辑
- 在 RoboJuDo 内替换 AMO 作为 mimic0（可与 BeyondMimic 切换）

2. 资产与配置
- 权重：RoboMimicDeploy_G1/policy/loco_mode/model/policy_29dof.pt
- 配置：RoboMimicDeploy_G1/policy/loco_mode/config/LocoMode.yaml
- 落地路径：
  - assets/models/g1/locomode/policy_29dof.pt
  - robojudo/config/g1/policy/g1_locomode_policy_cfg.py

3. 代码迁移
- 新增策略：robojudo/policy/locomode_policy.py
- 新增配置：robojudo/config/g1/policy/g1_locomode_policy_cfg.py
- 注册策略：robojudo/policy/__init__.py
- 接入管线：robojudo/config/g1/g1_loco_mimic_cfg.py 的 g1_locomode_beyondmimic

4. 关键对齐点
- 观测构造顺序：严格对齐 LocoMode.py
- 关节顺序映射：joint2motor_idx 参与重排
- cmd 输入：对齐 cmd_range/cmd_scale/cmd_signs/cmd_deadzone
- 四元数顺序：RoboMimic 用 wxyz，RoboJuDo base_quat 为 xyzw，需转换后再算 gravity
- 动作输出：RoboMimic 输出绝对关节角；RoboJuDo PolicyWrapper 会再加 default_pos

5. 关键修复记录
- 重力方向四元数顺序修正：
  - quat_wxyz = base_quat[[3,0,1,2]]
  - 使用 wxyz 计算 gravity
- 避免默认姿态二次叠加：
  - 改为输出 action_delta = action_reorder - default_angles_reorder

6. 验证要点
- LocoMode 稳定站立与遥控行走
- 与 BeyondMimic 在 g1_locomode_beyondmimic 内平滑切换
- 无“半蹲+后撤+摔倒”问题


二、MotionTracking 迁移

1. 迁移目标
- 迁移 RoboMimicDeploy_G1 的 montion_tracking 模式
- 在 RoboJuDo 中新增 demo2/demo4 两个固定动作

2. 资产与配置
- 权重：
  - RoboMimicDeploy_G1/policy/montion_tracking/model/demo2.onnx
  - RoboMimicDeploy_G1/policy/montion_tracking/model/demo4.onnx
- 动作数据：
  - RoboMimicDeploy_G1/policy/montion_tracking/motion/robot_demo2.npz
  - RoboMimicDeploy_G1/policy/montion_tracking/motion/robot_demo4.npz
- 配置：
  - RoboMimicDeploy_G1/policy/montion_tracking/config/Motion.yaml
- 落地路径：
  - assets/models/g1/motion_tracking/demo2.onnx
  - assets/models/g1/motion_tracking/demo4.onnx
  - assets/motions/g1/motion_tracking/robot_demo2.npz
  - assets/motions/g1/motion_tracking/robot_demo4.npz

3. 代码迁移
- 新增策略：robojudo/policy/motion_tracking_policy.py
- 新增配置：robojudo/config/g1/policy/g1_motion_tracking_policy_cfg.py
- 新增 cfg：robojudo/policy/policy_cfgs.py 的 MotionTrackingPolicyCfg
- 注册策略：robojudo/policy/__init__.py
- 接入管线：g1_locomode_beyondmimic 的 mimic4/mimic5

4. 观测与推理对齐（obs=154）
- obs 结构：
  - motioninput (joint_pos + joint_vel) = 58
  - anchor 旋转 6D = 6
  - ang_vel = 3
  - qj_rel = 29
  - dqj = 29
  - last_action = 29
- 输入：obs + time_step
- 输出：actions
- 动作还原：
  - default_angles_seq + action * action_scale_seq
  - seq -> xml 重排

5. 关键对齐点
- 关节顺序映射：XML ↔ Seq 严格一致
- base_pos/base_quat：使用 torso_link 对齐 MotionTracking 的 anchor
- yaw 对齐：进入模式时对齐 motion 第 0 帧 yaw
- PD 参数：stiffness/damping 与 Motion.yaml 一致

6. 关键修复记录
- 四元数顺序与类型：
  - base_quat xyzw -> wxyz
  - mujoco.mju_quat2Mat 需要 float64
- 默认姿态二次叠加：
  - 返回 action_delta，避免 PolicyWrapper 再加 default_pos
- motion 四元数归一化：
  - 加载后整体归一化
  - 每帧再归一化一次
- 角速度来源：
  - 使用 base_ang_vel 与 RoboMimic deploy_mujoco 保持一致
- 初始姿态：
  - 不再强制插值到动作首帧，保持默认站姿进入

7. 验证要点
- mimic4/demo2 与 mimic5/demo4 均能稳定执行
- 与 AMO、LocoMode、BeyondMimic 切换不互相干扰


三、后续复盘建议
- 迁移任何新策略前，先锁定观测维度与顺序
- 明确 base_quat 顺序与默认姿态是否会被二次叠加
- 逐项对齐 Motion.yaml/LocoMode.yaml 的参数到 cfg
- 用同一初始姿态与同一 yaw 对齐流程验证


四、运行命令与切换流程
- 启动命令：
  - python scripts/run_pipeline.py -c g1_locomode_beyondmimic
- 模式说明：
  - loco：AMO（启动后先 AMO，按配置自动切到 mimic0=LocoMode）
  - mimic0：LocoMode
  - mimic1：BeyondMimic@Dance_wose
  - mimic2：BeyondMimic@Violin
  - mimic3：BeyondMimic@Waltz
  - mimic4：MotionTracking@demo2
  - mimic5：MotionTracking@demo4


五、手柄/键盘操作（与 g1_locomode_beyondmimic 一致）
- 键盘：
  - ] 进入 loco
  - [ 进入 mimic
  - ; 下一个 mimic
  - ' 上一个 mimic
- 手柄（PS5 / Xbox）：
  - PS5：Share 切回 Loco，Options 进入 Mimic，R1 下一项，L1 上一项
  - Xbox：View 切回 Loco，Menu 进入 Mimic，RB 下一项，LB 上一项
