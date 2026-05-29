import logging
import time

import numpy as np
from box import Box

import robojudo.environment
import robojudo.policy
from robojudo.controller import CtrlManager
from robojudo.environment import Environment
from robojudo.pipeline import Pipeline, pipeline_registry
from robojudo.pipeline.pipeline_cfgs import RlPipelineCfg
from robojudo.policy import Policy, PolicyCfg
from robojudo.tools.dof import DoFAdapter
from robojudo.tools.tool_cfgs import DoFConfig
from robojudo.utils.progress import ProgressBar
from robojudo.utils.util_func import get_gravity_orientation

logger = logging.getLogger(__name__)


# [底层原理] PolicyWrapper：对 Policy 进行 DoF 适配封装
# 解决训练环境（IsaacLab）与部署环境（MuJoCo/真机）关节顺序/数量不一致的问题
# obs_adapter：将环境关节顺序重排为 policy 期望的顺序（isaaclab_to_mujoco_reindex）
# actions_adapter：将 policy 输出的动作重排回环境关节顺序
class PolicyWrapper:
    """A wrapper for Policy to handle observation and action adaptation."""

    def __init__(self, cfg_policy: PolicyCfg, env_dof_cfg: DoFConfig, device: str):
        self.env_dof_cfg = env_dof_cfg

        policy_type = cfg_policy.policy_type
        policy_name = policy_type
        if hasattr(cfg_policy, "policy_name"):
            policy_name += "@" + cfg_policy.policy_name  # type: ignore
        # while policy_name in self.policies.keys():
        #     policy_name += "_new"
        self.name = policy_name

        # [工程细节] 动态反射实例化 Policy 类，支持多种 policy 类型无需 if-else
        policy_class: type[Policy] = getattr(robojudo.policy, policy_type)
        self.policy: Policy = policy_class(cfg_policy=cfg_policy, device=device)
        # [Sim-to-Real] DoFAdapter：处理训练环境与部署环境之间的关节名称/顺序差异
        # obs_adapter：env关节顺序 → policy期望顺序（用于构建 obs）
        # actions_adapter：policy输出顺序 → env关节顺序（用于发送控制指令）
        self.obs_adapter = DoFAdapter(env_dof_cfg.joint_names, self.policy.cfg_obs_dof.joint_names)
        self.actions_adapter = DoFAdapter(self.policy.cfg_action_dof.joint_names, env_dof_cfg.joint_names)

    def get_observation(self, env_data: Box, ctrl_data: Box):
        # [工程细节] 在传入 policy 前先对 dof_pos/vel 做关节重排序
        # 确保 policy 看到的关节顺序与训练时一致
        env_data_adapted = env_data.copy()
        env_data_adapted.dof_pos = self.obs_adapter.fit(env_data_adapted.dof_pos)
        env_data_adapted.dof_vel = self.obs_adapter.fit(env_data_adapted.dof_vel)
        return self.policy.get_observation(env_data_adapted, ctrl_data)

    def get_action(self, obs):
        action = self.policy.get_action(obs)
        # [工程细节] policy 输出动作后，通过 actions_adapter 重排回环境关节顺序
        return self.actions_adapter.fit(action)

    def get_pd_target(self, obs):
        action = self.policy.get_action(obs)
        # [底层原理] pd_target = action（增量）+ default_pos（默认姿态）
        # 将相对默认姿态的动作增量转换为绝对关节角度目标，用于 PD 控制器
        pd_target = action + self.policy.default_pos
        # [工程细节] template=env_dof_cfg.default_pos：对于 policy 未覆盖的关节，使用环境默认位置填充
        return self.actions_adapter.fit(pd_target, template=self.env_dof_cfg.default_pos)

    def get_init_dof_pos(self):
        # [工程细节] 获取 policy 的初始关节位置（参考动作第 0 帧），用于 prepare() 阶段的插值目标
        return self.actions_adapter.fit(self.policy.get_init_dof_pos(), template=self.env_dof_cfg.default_pos)

    def __getattr__(self, name):
        """Fallback: delegate other func to the wrapped policy."""
        return getattr(self.policy, name)


@pipeline_registry.register
class RlPipeline(Pipeline):
    cfg: RlPipelineCfg

    def __init__(self, cfg: RlPipelineCfg):
        super().__init__(cfg=cfg)

        # [工程细节] 动态反射实例化环境类（MujocoEnv 或真机环境），统一接口
        env_class: type[Environment] = getattr(robojudo.environment, self.cfg.env.env_type)
        self.env: Environment = env_class(cfg_env=self.cfg.env, device=self.device)

        # [底层原理] CtrlManager 管理所有 Controller（如 BeyondMimicCtrl）
        # 负责在每步从各 Controller 收集参考数据，汇总为 ctrl_data 传给 policy
        self.ctrl_manager = CtrlManager(cfg_ctrls=self.cfg.ctrl, env=self.env, device=self.device)

        # [底层原理] PolicyWrapper 封装 policy，处理 DoF 适配（关节重排序）
        self.policy = PolicyWrapper(
            cfg_policy=self.cfg.policy,
            env_dof_cfg=self.env.dof_cfg,
            device=self.device,
        )

        # [工程细节] 用 policy 的 action DoF 配置覆盖环境的 DoF 配置
        # 确保 PD 增益（stiffness/damping）与训练时使用的参数一致
        self.env.update_dof_cfg(override_cfg=self.policy.cfg_action_dof)
        self.visualizer = self.env.visualizer

        # [面试考点：频率解耦] Policy 推理频率与控制频率的对齐
        # self.freq 为 policy 推理频率（如 50Hz），self.dt = 1/freq 为推理周期
        # 底层 MuJoCo 以 sim_dt（如 0.002s=500Hz）运行，通过 sim_decimation 解耦
        self.freq = self.cfg.policy.freq
        self.dt = 1.0 / self.freq

        self.self_check()
        self.reset()

    def self_check(self):
        # [工程细节] 启动前自检：运行 10 步 dry_run 预热 policy 推理（触发 ONNX JIT 编译）
        # 确保第一次真实推理不会因 JIT 延迟导致帧丢失
        self.env.self_check()
        for _ in range(10):
            self.step(dry_run=True)

    def reset(self):
        # [工程细节] 完整重置：环境状态、policy 历史（last_action/timestep）、controller 时间步
        logger.info("Pipeline reset")
        self.timestep = 0

        self.env.reset()
        # self.env.reborn(init_qpos=[0.2, 0.2, 0.8] + [ 0.707, 0, 0, 0.707]) # FOR SIM DEBUG
        self.policy.reset()
        self.ctrl_manager.reset()

    def safety_check(self):
        # [面试考点：安全机制] 摔倒检测和紧急停止
        # 通过重力方向向量判断机器人是否摔倒：
        # get_gravity_orientation 返回重力在机器人本体系下的方向向量
        # 正常站立时 gravity_ori[2] ≈ -1（重力朝下），摔倒时 |gravity_ori[2]| 减小
        # arccos(-gravity_ori[2]) 计算躯干与竖直方向的夹角，超过 ~57° 判定为摔倒
        if not self.do_safety_check:
            return
        gravity_ori = get_gravity_orientation(self.env.base_quat)
        angle = np.arccos(np.clip(-gravity_ori[2], -1.0, 1.0))
        if abs(angle) > 1.0:  # more than ~57 degrees
            logger.error("Robot fallen! Shutdown for safety.")
            # [Sim-to-Real] 仿真环境支持 reborn（重置到初始状态继续训练）
            # 真机环境只能 shutdown（无法自动复位），需要人工干预
            if hasattr(self.env, "reborn"):
                self.env.reborn()  # pyright: ignore[reportAttributeAccessIssue]
            else:
                self.env.shutdown()

    def post_step_callback(self, env_data, ctrl_data, extras, pd_target):
        self.timestep += 1
        # [工程细节] 从 ctrl_data 中提取命令字符串，驱动状态机
        commands = ctrl_data.get("COMMANDS", [])
        for command in commands:
            match command:
                case "[SHUTDOWN]":
                    # [面试考点：安全机制] 紧急关机命令：由外部触发器（如遥控器急停）发送
                    logger.warning("Emergency shutdown!")
                    self.env.shutdown()
                case "[SIM_REBORN]":
                    # [Sim-to-Real] 仿真专用：重置仿真环境到初始状态，用于自动化测试
                    if hasattr(self.env, "reborn"):
                        logger.warning("Simulation Env reborn!")
                        self.env.reborn()  # pyright: ignore[reportAttributeAccessIssue]

        # [工程细节] 通知各 Controller 和 Policy 完成本步回调（推进时间步、更新状态机）
        self.ctrl_manager.post_step_callback(ctrl_data)

        self.policy.post_step_callback(commands)
        if self.visualizer is not None:
            self.policy.debug_viz(self.visualizer, env_data, ctrl_data, extras)

        # [面试考点：安全机制] 每步结束后执行安全检查，确保机器人姿态在安全范围内
        self.safety_check()
        if self.cfg.debug.log_obs:
            self.debug_logger.log(
                env_data=env_data,
                ctrl_data=ctrl_data,
                extras=extras,
                pd_target=pd_target,
                timestep=self.timestep,
            )

    def step(self, dry_run=False):
        # [底层原理] 主循环单步执行流程：
        # 1. env.update()：从仿真/传感器读取最新状态
        # 2. env.get_data()：将状态打包为 Box 字典（env_data）
        # 3. ctrl_manager.get_ctrl_data()：从各 Controller 获取参考数据（ctrl_data）
        # 4. policy.get_observation()：构建 obs tensor
        # 5. policy.get_pd_target()：推理得到 PD 目标关节角度
        # 6. env.step()：将 PD 目标发送给仿真/真机执行
        self.env.update()
        env_data = self.env.get_data()

        ctrl_data = self.ctrl_manager.get_ctrl_data(env_data)

        commands = ctrl_data.get("COMMANDS", [])
        if len(commands) > 0:
            logger.info(f"{'=' * 10} COMMANDS {'=' * 10}\n{commands}")

        obs, extras = self.policy.get_observation(env_data, ctrl_data)
        # [底层原理] get_pd_target = get_action() + default_pos，得到绝对关节角度目标
        # 内部调用 ONNX 推理，包含 EMA 动作滤波
        pd_target = self.policy.get_pd_target(obs)

        # [工程细节] dry_run=True 时跳过 env.step()，只做推理预热，不发送控制指令
        if not dry_run:
            self.env.step(pd_target, extras.get("hand_pose", None))

        self.post_step_callback(env_data, ctrl_data, extras, pd_target)

    def prepare(self, init_motor_angle=None):
        # [Sim-to-Real] prepare() 阶段：真机部署前将机器人从当前姿态平滑插值到 policy 初始姿态
        # 避免直接跳变到目标姿态导致关节冲击或摔倒
        if init_motor_angle is not None:
            desired_motor_angle = init_motor_angle
        else:
            # [工程细节] 从 policy 获取参考动作第 0 帧的关节位置作为目标姿态
            desired_motor_angle = self.policy.get_init_dof_pos()

        # logger.info(f"{desired_motor_angle=}")
        current_motor_angle = np.array(self.env.dof_pos)
        # logger.info(f"{current_motor_angle=}")

        # [关键参数] traj_len=1000 步，以 policy 频率运行约 20s（50Hz）
        # 前 300 步线性插值（blend_ratio 从 0 到 1），后续保持目标姿态
        traj_len = 1000
        last_step_time = time.time()
        logger.warning("prepare_init")
        pbar = ProgressBar("Prepare", traj_len)

        for t in range(traj_len):
            current_motor_angle = np.array(self.env.dof_pos)

            # [底层原理] 线性插值：blend_ratio 在前 300 步从 0 线性增长到 1
            # action = (1-r)*current + r*desired，平滑过渡到目标姿态
            blend_ratio = np.minimum(t / 300, 1)
            action = (1 - blend_ratio) * current_motor_angle + blend_ratio * desired_motor_angle

            # warm up network
            # [工程细节] 在插值过程中同步预热 policy 网络，确保推理 JIT 编译完成
            self.step(dry_run=True)

            self.env.step(action)

            # [面试考点：频率解耦] prepare 阶段同样需要频率控制，保持与主循环一致的控制频率
            time_diff = last_step_time + self.dt - time.time()
            if time_diff > 0:
                time.sleep(time_diff)
            else:
                logger.error("Warning: frame drop")
            last_step_time = time.time()
            pbar.update()

            # [工程细节] 在 90% 进度时执行 reset()，清除 policy 历史状态
            # 确保正式运行时 last_action 为零，避免 prepare 阶段的动作历史污染推理
            if t == 0.9 * traj_len:
                logger.info(f"{'=' * 10} RESET ZERO POSITION {'=' * 10}")
                self.reset()

        time.sleep(0.01)
        pbar.close()
        logger.warning("prepare_done")


if __name__ == "__main__":
    pass
