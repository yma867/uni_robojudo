from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np

from robojudo.controller import Controller, ctrl_registry
from robojudo.controller.ctrl_cfgs import BeyondMimicCtrlCfg
from robojudo.environment import Environment
from robojudo.utils.progress import ProgressBar
from robojudo.utils.rotation import TransformAlignment


# From BeyondMimic
# [底层原理] MotionLoader：从 .npz 文件加载预处理好的参考动作数据
# .npz 文件包含：fps（帧率）、joint_pos/vel（关节状态序列）、body_pos_w/quat_w（身体位姿序列）
# 这些数据由动作捕捉数据或物理仿真轨迹预处理生成，是运动模仿的"教师信号"
class MotionLoader:
    def __init__(self, motion_file: str, body_indexes: Sequence[int], device: str = "cpu"):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file)
        # [关键参数] fps：参考动作的采样频率，必须与 policy 推理频率匹配
        # 若 fps=50 而 policy 以 50Hz 运行，则每步推进 1 帧；若频率不匹配需要插值
        self.fps = data["fps"]
        # [底层原理] joint_pos/vel：shape=(T, num_joints)，T 为总帧数
        # 每帧存储所有关节的目标位置和速度，作为 policy 的运动跟踪目标
        self.joint_pos = data["joint_pos"]
        self.joint_vel = data["joint_vel"]
        self._body_pos_w = data["body_pos_w"]
        self._body_quat_w = data["body_quat_w"]
        self._body_lin_vel_w = data["body_lin_vel_w"]
        self._body_ang_vel_w = data["body_ang_vel_w"]
        # [工程细节] body_indexes：只加载需要的身体部位数据，减少内存占用
        # 通过索引从全身体数据中选取 anchor body 等关键部位
        self._body_indexes = body_indexes
        # [工程细节] hand_pose 为可选字段，用于手部姿态控制（如抓取动作）
        self.hand_pose = data.get("hand_pose", None)
        # self.hand_pose = np.zeros((self.joint_pos.shape[0], 2, 6))  # TODO: dummy for now
        self.time_step_total = self.joint_pos.shape[0]

    @property
    def body_pos_w(self) -> np.ndarray:
        # [工程细节] 通过 body_indexes 切片，只返回配置中指定的身体部位位置数据
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> np.ndarray:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> np.ndarray:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> np.ndarray:
        return self._body_ang_vel_w[:, self._body_indexes]


@ctrl_registry.register
class BeyondMimicCtrl(Controller):
    cfg_ctrl: BeyondMimicCtrlCfg
    env: Environment

    def __init__(self, cfg_ctrl: BeyondMimicCtrlCfg, env, device="cpu"):
        super().__init__(cfg_ctrl=cfg_ctrl, env=env, device=device)
        assert self.env is not None, "Env is required for BeyondMimicCtrl"
        self.override_robot_anchor_pos = self.cfg_ctrl.override_robot_anchor_pos

        motion_file = self.cfg_ctrl.motion_path
        motion_cfg = self.cfg_ctrl.motion_cfg
        # [工程细节] 将配置中的 body_names 映射为 .npz 数据中的索引
        # body_names_all 是 .npz 中所有身体部位的完整列表，body_names 是需要使用的子集
        body_indexes = [motion_cfg.body_names_all.index(name) for name in motion_cfg.body_names]
        # [关键参数] motion_anchor_body_index：anchor body 在 body_names 子集中的索引
        # anchor body 通常选择骨盆或腰部，作为运动跟踪的参考基准点
        self.motion_anchor_body_index = motion_cfg.body_names.index(motion_cfg.anchor_body_name)

        # [底层原理] 加载参考动作文件，只保留需要的身体部位数据
        self.motion = MotionLoader(motion_file, body_indexes, device="cpu")
        # [面试考点：观测历史] 时间步管理：timestep 从 0 开始，每步 +1，直到 time_step_total-1
        self.timestep = 0
        # [工程细节] playing 标志控制时间步是否推进，支持 fade in/out 状态机
        self.playing = False

        # [底层原理] TransformAlignment：将参考动作的世界坐标对齐到机器人初始位置
        # yaw_only=True：只对齐偏航角，保留 pitch/roll；xy_only=True：只对齐水平位置
        self.motion_init_align = TransformAlignment(yaw_only=True, xy_only=True)
        self.reset()

    @property
    def command(self) -> np.ndarray:
        # [底层原理] command 向量 = [joint_pos | joint_vel]，作为 policy obs 中的运动跟踪目标
        # policy 通过比较当前关节状态与 command 的差值来计算跟踪误差
        return np.concatenate([self.joint_pos, self.joint_vel], axis=-1)

    @property
    def joint_pos(self) -> np.ndarray:
        # [底层原理] 按当前时间步从动作表中查询参考关节位置，copy() 防止外部修改影响原始数据
        return self.motion.joint_pos[self.timestep].copy()

    @property
    def joint_vel(self) -> np.ndarray:
        return self.motion.joint_vel[self.timestep].copy()

    @property
    def anchor_pos_w(self) -> np.ndarray:
        # [底层原理] 获取当前帧 anchor body 的世界坐标位置，并通过 motion_init_align 对齐到机器人初始位置
        anchor_pos_w_raw = self.motion.body_pos_w[self.timestep, self.motion_anchor_body_index].copy()
        anchor_pos_w = self.motion_init_align.align_pos(anchor_pos_w_raw)
        return anchor_pos_w

    @property
    def anchor_quat_w(self) -> np.ndarray:
        # [工程细节] .npz 中四元数格式为 [w,x,y,z]，内部统一使用 [x,y,z,w]，需要 [[1,2,3,0]] 重排
        anchor_quat_w_raw = self.motion.body_quat_w[self.timestep, self.motion_anchor_body_index].copy()[[1, 2, 3, 0]]
        return self.motion_init_align.align_quat(anchor_quat_w_raw)

    @property
    def robot_anchor_pos_w(self) -> np.ndarray:
        # [Sim-to-Real] override_robot_anchor_pos：调试时用参考动作位置替代真实机器人位置
        # 真机部署时必须关闭，使用真实的 base_pos，否则 obs 中的位置误差计算会失真
        if self.override_robot_anchor_pos:  # OVERRIDE
            return self.anchor_pos_w
        else:
            base_pos = self.env.torso_pos
            assert base_pos is not None
            return base_pos

    @property
    def robot_anchor_quat_w(self) -> np.ndarray:
        # [底层原理] 机器人 anchor 的朝向直接使用 torso（躯干）的四元数
        # 与参考动作的 anchor_quat_w 做差，得到朝向跟踪误差
        torso_quat = self.env.torso_quat
        assert torso_quat is not None
        return torso_quat

    @property
    def hand_pose(self) -> np.ndarray | None:
        if self.motion.hand_pose is not None:
            hand_pose = self.motion.hand_pose[self.timestep].copy()
            # [工程细节] 兼容两种 hand_pose 存储格式：
            # 1D 格式（flat）：reshape 为 (2, hand_dim) 对应左右手
            # 2D 格式：直接使用
            if len(hand_pose.shape) == 1:
                hand_dim = hand_pose.shape[0] // 2
                hand_pose = hand_pose.reshape(2, hand_dim)
            return hand_pose
        else:
            return None

    def reset(self):
        self.timestep = 0
        self.pbar = ProgressBar(f"BeyondmimicCtrl {self.cfg_ctrl.motion_name}", self.motion.time_step_total)

        # [底层原理] 初始化坐标对齐：以参考动作第 0 帧的 anchor body 位姿为基准
        # 后续所有帧的 anchor 坐标都相对此基准变换，使参考动作"跟随"机器人初始位置
        # align the robot to the motion's starting pose
        init2anchor_pos = self.motion.body_pos_w[0, self.motion_anchor_body_index].copy()
        init2anchor_quat = self.motion.body_quat_w[0, self.motion_anchor_body_index].copy()[[1, 2, 3, 0]]
        # keep yaw only
        # [工程细节] set_base 设置对齐基准，后续 align_pos/align_quat 都相对此基准计算
        self.motion_init_align.set_base(quat=init2anchor_quat, pos=init2anchor_pos)

    def post_step_callback(self, commands: list[str] | None = None):
        # [面试考点：观测历史] 时间步管理：每步 +1，到达末尾时停止（不循环，避免动作突变）
        self.pbar.set(self.timestep)
        if self.timestep < self.motion.time_step_total - 1:
            # [工程细节] playing=False 时时间步冻结，动作保持在当前帧（fade out 效果）
            if self.playing:
                self.timestep += 1

        # [工程细节] 命令驱动的状态机：与 policy 侧的命令处理保持一致
        # [MOTION_RESET]：重置时间步到 0；[MOTION_FADE_IN]：开始播放；[MOTION_FADE_OUT]：暂停
        for command in commands or []:
            match command:
                case "[MOTION_RESET]":
                    self.reset()
                case "[MOTION_FADE_IN]":
                    self.playing = True
                case "[MOTION_FADE_OUT]":
                    self.playing = False

    def get_data(self):
        # [底层原理] 将当前时间步的所有参考数据打包为字典，传递给 policy 的 _get_command()
        # 包含：command（关节目标）、anchor 坐标变换、时间步、手部姿态
        ctrl_data = {
            "command": self.command,
            "joint_pos": self.joint_pos,
            # "joint_vel": self.joint_vel,
            "robot_anchor_pos_w": self.robot_anchor_pos_w,
            "robot_anchor_quat_w": self.robot_anchor_quat_w,
            "anchor_pos_w": self.anchor_pos_w,
            "anchor_quat_w": self.anchor_quat_w,
            "timestep": self.timestep,
            "hand_pose": self.hand_pose,
        }
        return ctrl_data


if __name__ == "__main__":
    # Example usage
    from robojudo.config.g1.ctrl.g1_beyondmimic_ctrl_cfg import G1BeyondmimicCtrlCfg
    from robojudo.config.g1.env.g1_mujuco_env_cfg import G1MujocoEnvCfg
    from robojudo.environment.mujoco_env import MujocoEnv

    env = MujocoEnv(cfg_env=G1MujocoEnvCfg())
    ctrl = BeyondMimicCtrl(cfg_ctrl=G1BeyondmimicCtrlCfg(), env=env)
    print(ctrl.get_data())  # This will print the command tensor
