import logging
import time

import mujoco
import mujoco_viewer
import numpy as np

from robojudo.environment import Environment, env_registry
from robojudo.environment.env_cfgs import MujocoEnvCfg
from robojudo.environment.utils.mujoco_viz import MujocoVisualizer
# [底层原理] quat_rotate_inverse_np：将世界系速度转换到机器人本体系
# quatToEuler：四元数转欧拉角，用于姿态监控和安全检查
from robojudo.utils.util_func import quat_rotate_inverse_np, quatToEuler

logger = logging.getLogger(__name__)


@env_registry.register
class MujocoEnv(Environment):
    # [底层原理] MuJoCo 仿真环境：与 Isaac Sim 的主要区别：
    # 1. MuJoCo 使用 XML 模型文件，Isaac Sim 使用 USD/URDF
    # 2. MuJoCo 通过 mj_step 推进物理仿真，Isaac Sim 通过 PhysX 引擎
    # 3. MuJoCo 的 data.qpos/qvel 直接暴露状态，Isaac Sim 需要通过 ArticulationView 访问
    # 4. MuJoCo 适合单机器人精细仿真，Isaac Sim 适合大规模并行训练
    cfg_env: MujocoEnvCfg

    def __init__(self, cfg_env: MujocoEnvCfg, device="cpu"):
        super().__init__(cfg_env=cfg_env, device=device)

        self.sim_duration = cfg_env.sim_duration
        # [关键参数] sim_dt：物理仿真步长（如 0.002s = 500Hz），决定仿真精度
        self.sim_dt = cfg_env.sim_dt
        # [关键参数] sim_decimation：每次 policy 推理对应的仿真步数
        # control_dt = sim_dt * sim_decimation（如 0.002 * 10 = 0.02s = 50Hz policy 频率）
        self.sim_decimation = cfg_env.sim_decimation
        # [面试考点：频率解耦] control_dt 是 policy 控制周期，sim_dt 是物理仿真步长
        # 两者通过 sim_decimation 解耦，允许高频物理仿真 + 低频 policy 推理
        self.control_dt = self.sim_dt * self.sim_decimation

        # [底层原理] 从 XML 文件加载 MuJoCo 模型，设置仿真步长
        self.model = mujoco.MjModel.from_xml_path(cfg_env.xml)  # pyright: ignore[reportAttributeAccessIssue]
        self.model.opt.timestep = self.sim_dt
        # [底层原理] MjData 存储仿真状态（qpos/qvel/ctrl 等），每次 mj_step 后更新
        self.data = mujoco.MjData(self.model)  # pyright: ignore[reportAttributeAccessIssue]
        # mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        # [工程细节] 初始化时执行一步仿真，确保 data 中的状态（如接触力）被正确初始化
        mujoco.mj_step(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]

        # [工程细节] MujocoViewer 提供实时可视化窗口，hide_menus 和 diable_key_callbacks 用于简化界面
        self.viewer = mujoco_viewer.MujocoViewer(
            self.model,
            self.data,
            width=1200,
            height=900,
            hide_menus=True,
            diable_key_callbacks=True,
        )
        # [工程细节] 相机参数：distance=3m，elevation=-10°（俯视），azimuth=180°（正面朝向）
        self.viewer.cam.distance = 3.0
        self.viewer.cam.elevation = -10.0
        self.viewer.cam.azimuth = 180.0
        # self.viewer._paused = True

        if cfg_env.visualize_extras:
            self.visualizer = MujocoVisualizer(self.viewer)
        else:
            self.visualizer = None

        self.last_time = time.time()

        # [工程细节] 初始化时调用 update() 获取初始状态，确保 dof_pos/quat 等属性有效
        self.update()  # get initial state

    def reborn(self, init_qpos=None):
        # [Sim-to-Real] reborn()：仿真专用的重置接口，真机环境没有此方法
        # 摔倒后可以直接重置到初始状态继续测试，无需重启仿真
        if init_qpos is not None:
            # [工程细节] qpos[0:7] = [x, y, z, qw, qx, qy, qz]（MuJoCo 浮动基座格式）
            self.data.qpos[0:7] = init_qpos
            self.data.qvel[:] = 0.0
            self.data.ctrl[:] = 0.0
        else:
            # [工程细节] 使用 XML 中定义的 keyframe 0 作为默认初始状态
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)  # pyright: ignore[reportAttributeAccessIssue]
        # [底层原理] mj_forward：重新计算正向运动学和动力学，更新所有派生量（接触、传感器等）
        # 不推进时间，只更新当前状态的所有计算量
        mujoco.mj_forward(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]

    def reset(self):
        if self.born_place_align:  # TODO: merge
            self.born_place_align = False  # disable during reset
            self.update()
            self.born_place_align = True  # enable after reset
            self.set_born_place()
            self.update()

    def set_gains(self, stiffness, damping):
        assert len(stiffness) == self.num_dofs and len(damping) == self.num_dofs
        # [关键参数] PD 增益设置：stiffness（位置增益 Kp）和 damping（速度增益 Kd）
        # 这些参数在训练时确定，部署时必须与训练完全一致，否则控制行为会改变
        self.stiffness = np.asarray(stiffness)
        self.damping = np.asarray(damping)

    def self_check(self):
        pass

    def set_born_place(self, quat: np.ndarray | None = None, pos: np.ndarray | None = None):
        quat_ = self.base_quat if quat is None else quat
        pos_ = self.base_pos if pos is None else pos
        super().set_born_place(quat_, pos_)

    def update(self, simple=False):  # TODO: clean sensors in xml
        """simple: only update dof pos & vel"""
        # [底层原理] MuJoCo qpos 布局（浮动基座）：
        # qpos[0:3]  = 基座位置 (x, y, z)
        # qpos[3:7]  = 基座四元数 (qw, qx, qy, qz)（MuJoCo 格式，w 在前）
        # qpos[7:]   = 各关节角度（按 XML 定义顺序）
        # qvel 布局：qvel[0:3]=线速度, qvel[3:6]=角速度, qvel[6:]=关节速度
        dof_pos = self.data.qpos.astype(np.float32)[-self.num_dofs :]
        dof_vel = self.data.qvel.astype(np.float32)[-self.num_dofs :]

        self._dof_pos = dof_pos.copy()
        self._dof_vel = dof_vel.copy()

        if simple:
            return

        # [工程细节] 四元数格式转换：MuJoCo 使用 [w,x,y,z]，内部统一使用 [x,y,z,w]
        # [[1,2,3,0]] 重排：取 qpos[4,5,6,3] 得到 [x,y,z,w] 格式
        quat = self.data.qpos.astype(np.float32)[3:7][[1, 2, 3, 0]]
        ang_vel = self.data.qvel.astype(np.float32)[3:6]
        base_pos = self.data.qpos.astype(np.float32)[:3]
        lin_vel = self.data.qvel.astype(np.float32)[0:3]

        # [工程细节] born_place_align：将机器人坐标系对齐到出生点，消除初始位置偏差
        # 用于多次 reborn 后保持坐标系一致性
        if self.born_place_align:
            quat, base_pos = self.base_align.align_transform(quat, base_pos)

        # [底层原理] quat_rotate_inverse_np：将世界系线速度旋转到机器人本体系
        # policy 训练时使用本体系速度，部署时需要做同样的坐标变换
        lin_vel = quat_rotate_inverse_np(quat, lin_vel)
        rpy = quatToEuler(quat)

        self._base_rpy = rpy.copy()
        self._base_quat = quat.copy()
        self._base_ang_vel = ang_vel.copy()

        self._base_pos = base_pos.copy()
        self._base_lin_vel = lin_vel.copy()

        # [工程细节] update_with_fk：通过正向运动学计算 torso（躯干）的位姿
        # 当 torso 不是浮动基座根节点时（如有腰部关节），需要 FK 计算真实 torso 位姿
        if self.update_with_fk:
            fk_info = self.fk()
            self._fk_info = fk_info.copy()
            self._torso_ang_vel = fk_info[self._torso_name]["ang_vel"]
            self._torso_quat = fk_info[self._torso_name]["quat"]
            self._torso_pos = fk_info[self._torso_name]["pos"]

    def step(self, pd_target, hand_pose=None):
        # [底层原理] MuJoCo 关节控制接口：使用 PD 控制器将目标关节角度转换为力矩
        # 与 Isaac Sim 的区别：Isaac Sim 直接支持 position_target 模式，MuJoCo 需要手动计算力矩
        assert len(pd_target) == self.num_dofs, "pd_target len should be num_dofs of env"

        if hand_pose is not None:
            logger.info("Hand pose-->", hand_pose)

        # [工程细节] 相机跟随机器人：将 lookat 设置为机器人基座位置，保持机器人在视野中心
        self.viewer.cam.lookat = self.data.qpos.astype(np.float32)[:3]
        if self.viewer.is_alive:
            self.viewer.render()

        # [面试考点：频率解耦] sim_decimation 循环：每次 policy 推理对应多步物理仿真
        # 在每个仿真子步中重新计算力矩（使用最新的 dof_pos/vel），提高控制精度
        for _ in range(self.sim_decimation):
            # [底层原理] PD 控制律：torque = Kp * (target - pos) - Kd * vel
            # 这是标准的 PD 位置控制，Kp=stiffness（弹簧刚度），Kd=damping（阻尼）
            torque = (pd_target - self.dof_pos) * self.stiffness - self.dof_vel * self.damping
            # [面试考点：动作滤波] torque clip：限制力矩在安全范围内，防止关节过载
            # torque_limits 来自 URDF/XML 中的关节力矩限制
            torque = np.clip(torque, -self.torque_limits, self.torque_limits)

            # [底层原理] data.ctrl 是 MuJoCo 的控制输入，对应 XML 中定义的 actuator
            # 这里直接写入力矩（torque control 模式）
            self.data.ctrl = torque

            # [底层原理] mj_step：推进一步物理仿真（sim_dt 时长），更新所有状态
            mujoco.mj_step(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]
            # [工程细节] simple=True：仿真子步中只更新 dof_pos/vel，跳过完整状态更新（节省计算）
            self.update(simple=True)
        # [工程细节] 所有子步完成后执行完整状态更新（包括 quat、lin_vel、FK 等）
        self.update(simple=False)

    def shutdown(self):
        # [工程细节] 关闭可视化窗口，释放 MuJoCo 资源
        # 真机环境的 shutdown() 会发送急停指令，MuJoCo 只需关闭 viewer
        self.viewer.close()


if __name__ == "__main__":
    from robojudo.config.g1.env.g1_mujuco_env_cfg import G1MujocoEnvCfg

    mujoco_env = MujocoEnv(cfg_env=G1MujocoEnvCfg())
    mujoco_env.viewer._paused = False

    while True:
        # mujoco_env.update()
        mujoco_env.step(np.zeros(mujoco_env.num_dofs))
        time.sleep(0.02)
