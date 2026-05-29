import logging

import numpy as np
# [底层原理] onnxruntime 是跨平台推理引擎，支持 CPU/CUDA/TensorRT 多后端
# 训练好的 PyTorch 模型导出为 ONNX 格式后，可在无 PyTorch 依赖的环境（如 Jetson）上高效推理
import onnxruntime as ort

from robojudo.environment.utils.mujoco_viz import MujocoVisualizer
from robojudo.policy import Policy, policy_registry
from robojudo.policy.policy_cfgs import BeyondMimicPolicyCfg
from robojudo.tools.dof import DoFConfig
from robojudo.utils.progress import ProgressBar
from robojudo.utils.rotation import TransformAlignment
from robojudo.utils.util_func import matrix_from_quat, subtract_frame_transforms

logger = logging.getLogger(__name__)


@policy_registry.register
class BeyondMimicPolicy(Policy):
    cfg_policy: BeyondMimicPolicyCfg

    def __init__(self, cfg_policy: BeyondMimicPolicyCfg, device):
        # [底层原理] ONNX 模型加载：onnxruntime 根据 providers 列表优先级自动选择推理后端
        # CPU → CUDA → TensorRT 依次降级，TensorRT 在 Jetson 上有最优推理速度
        # init onnx, override dof cfg if needed
        sess_options = ort.SessionOptions()

        # [工程细节] 当前强制使用 CPU，实际部署时可根据硬件改为 cuda 或 tensorrt
        device = "cpu"
        if device == "cpu":
            providers = ["CPUExecutionProvider"]
        elif device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif device == "tensorrt":
            # Jetson
            # [工程细节] Jetson 平台优先使用 TensorRT，可将推理延迟从 ~10ms 降至 ~2ms
            providers = [
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
        else:
            raise ValueError(f"Unknown device: {device}")

        # [底层原理] 加载 ONNX 模型文件，建立推理 session；模型中可能已 bake 了 obs normalization
        self.session = ort.InferenceSession(cfg_policy.policy_file, sess_options, providers=providers)

        # [工程细节] 动态获取模型输入/输出名称，避免硬编码，兼容不同训练框架导出的 ONNX
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.motion_anchor_body_index = -1

        cfg_policy_new = cfg_policy.model_copy()
        # [工程细节] use_modelmeta_config=True 时从 ONNX 模型元数据中读取关节配置
        # 这样模型文件自包含所有必要参数，部署时无需额外配置文件，降低版本不匹配风险
        if cfg_policy_new.use_modelmeta_config:
            logger.info("[BeyondMimicPolicy] Using modelmeta as config ...")
            modelmeta = self.session.get_modelmeta()  # all str,
            modelmeta_dict = modelmeta.custom_metadata_map

            # dict_keys(['joint_names', 'run_path', 'command_names', 'joint_stiffness', 'joint_damping',
            # 'default_joint_pos', 'action_scale', 'observation_names', 'anchor_body_name', 'body_names'])
            def parse_floats(s):
                return [float(item) for item in s.split(",")]

            def parse_strings(s):
                return [item for item in s.split(",")]

            # [关键参数] 从模型元数据解析关节名称、默认位置、PD 增益
            # 这些参数在训练时确定，部署时必须与训练完全一致，否则会导致控制异常
            dof_config = DoFConfig(
                joint_names=parse_strings(modelmeta_dict["joint_names"]),
                default_pos=parse_floats(modelmeta_dict["default_joint_pos"]),
                stiffness=parse_floats(modelmeta_dict["joint_stiffness"]),
                damping=parse_floats(modelmeta_dict["joint_damping"]),
            )
            # [关键参数] action_scale：将网络输出的归一化动作映射回关节角度空间
            # 通常为每个关节单独设置，反映各关节的运动范围差异
            action_scales = parse_floats(modelmeta_dict["action_scale"])

            # [底层原理] anchor_body：运动参考锚点身体部位（如骨盆/腰部）
            # 用于将参考动作的世界坐标系变换到机器人本体坐标系，实现运动跟踪
            anchor_body_name = modelmeta_dict["anchor_body_name"]
            body_names = parse_strings(modelmeta_dict["body_names"])
            self.motion_anchor_body_index = body_names.index(anchor_body_name)

            # command_names = parse_strings(modelmeta_dict["command_names"])
            # observation_names = parse_strings(modelmeta_dict["observation_names"])

            cfg_policy_new.action_dof = dof_config
            cfg_policy_new.obs_dof = dof_config

            cfg_policy_new.action_scales = action_scales

        super().__init__(cfg_policy=cfg_policy_new, device=device)
        # [关键参数] action_scales 将网络输出（通常在 [-1,1] 附近）缩放到实际关节角度增量（rad）
        self.action_scales = np.asarray(self.cfg_policy.action_scales)
        # [Sim-to-Real] without_state_estimator：真机部署时可能没有可靠的线速度估计
        # 设为 True 时从 obs 中去掉 base_lin_vel，降低对状态估计器的依赖
        self.without_state_estimator = self.cfg_policy.without_state_estimator
        self.override_robot_anchor_pos = self.cfg_policy.override_robot_anchor_pos
        # [工程细节] use_motion_from_model：参考动作直接从 ONNX 模型输出获取（内嵌动作表）
        # 而非从外部 .npz 文件加载，简化部署时的文件依赖
        self.use_motion_from_model = self.cfg_policy.use_motion_from_model

        # [关键参数] max_timestep：动作序列最大帧数，超过后 play_speed 置 0，动作冻结
        self.max_timestep = self.cfg_policy.max_timestep
        self.command = None
        self.reset()

        if self.use_motion_from_model:
            assert self.motion_anchor_body_index >= 0, "motion_anchor_body_index not set"
            assert self.command is not None, "command not initialized"
            command_init = self.command.copy()

            # [底层原理] motion init2anchor alignment：将参考动作的初始帧对齐到世界坐标系原点
            # 提取第 0 帧的 anchor body 位置和朝向，构建坐标变换，后续所有帧都相对此变换
            # motion init2anchor alignment
            anchor_pos_w_init = command_init["body_pos_w"][self.motion_anchor_body_index, :]
            # [工程细节] 四元数格式转换：ONNX 输出 [w,x,y,z]，内部使用 [x,y,z,w]，需要 [[1,2,3,0]] 重排
            anchor_quat_w_init = command_init["body_quat_w"][self.motion_anchor_body_index, :][[1, 2, 3, 0]]

            # [底层原理] yaw_only=True, xy_only=True：只对齐水平面内的偏航角和 XY 位置
            # 保留 Z 轴高度和 pitch/roll，避免将重力方向也变换掉
            self.command_init_align = TransformAlignment(
                quat=anchor_quat_w_init, pos=anchor_pos_w_init, yaw_only=True, xy_only=True
            )

    def _prepare_policy(self):
        # [工程细节] 预热推理：用零向量跑一次前向传播，触发 ONNX Runtime 的 JIT 编译和内存分配
        # 避免第一次真实推理时出现延迟尖峰（对实时控制影响较大）
        obs_shape = self.session.get_inputs()[0].shape  # e.g. [1, 154]
        obs = np.zeros(obs_shape[1], dtype=np.float32)
        self.get_action(obs)

    def reset(self):
        # [面试考点：观测历史] 历史观测的维护方式
        # reset 时将 timestep 归零，last_action 清零（在父类 Policy.reset() 中处理）
        # play_speed 恢复为 1.0，flag_motion_done 清除，确保下一轮动作从头开始
        self.timestep: float = self.cfg_policy.start_timestep
        if self.use_motion_from_model:
            self.pbar = ProgressBar(f"Beyondmimic {self.cfg_policy.policy_name}", self.max_timestep)
        else:
            self.pbar = None
        # [关键参数] play_speed：控制动作时间步推进速率，1.0 为正常速度，0.0 为暂停（fade out 时使用）
        self.play_speed: float = 1.0
        self.flag_motion_done = False
        self._prepare_policy()

    def post_step_callback(self, commands: list[str] | None = None):
        # [底层原理] 每个控制步结束后推进时间步，play_speed 为 0 时时间步冻结（动作暂停）
        self.timestep += 1 * self.play_speed
        if self.pbar:
            self.pbar.set(self.timestep)

        # [工程细节] 动作序列播放完毕后将 play_speed 置 0，防止 timestep 越界
        if 0 < self.max_timestep <= self.timestep:
            self.play_speed = 0.0
            self.flag_motion_done = True

        # [工程细节] 命令驱动的状态机：通过字符串命令控制动作播放状态
        # [MOTION_RESET]：重置到初始帧；[MOTION_FADE_IN]：开始播放；[MOTION_FADE_OUT]：暂停播放
        for command in commands or []:
            match command:
                case "[MOTION_RESET]":
                    self.reset()
                case "[MOTION_FADE_IN]":
                    self.play_speed = 1.0
                case "[MOTION_FADE_OUT]":
                    self.play_speed = 0.0

    def _get_command(self, env_data, ctrl_data):
        if not self.use_motion_from_model:
            # [底层原理] 从外部 BeyondMimicCtrl 获取参考动作数据
            # ctrl_data 中包含当前时间步的关节位置/速度目标和 anchor 坐标变换
            assert "BeyondMimicCtrl" in ctrl_data, "BeyondMimicCtrl not found in ctrl_data"
            command = ctrl_data.get("BeyondMimicCtrl")
            self.command = command
            # print(command.time_steps[0])
            return (
                command.command,
                command.robot_anchor_pos_w,
                command.robot_anchor_quat_w,
                command.anchor_pos_w,
                command.anchor_quat_w,
                command.get("hand_pose", None),
            )
        else:
            # [底层原理] 从 ONNX 模型上一步输出的 command 字典中获取参考动作
            # 模型内嵌了完整的参考动作表，通过 time_step 索引查表
            assert self.command is not None, "command not initialized"
            # print(self.command["time_step"])
            # [工程细节] 将关节位置和速度拼接为 command 向量，作为 policy 的运动跟踪目标
            command = np.concatenate([self.command["joint_pos"], self.command["joint_vel"]], axis=-1)

            anchor_pos_w = self.command["body_pos_w"][self.motion_anchor_body_index, :]
            # [工程细节] 四元数格式转换 [w,x,y,z] → [x,y,z,w]
            anchor_quat_w = self.command["body_quat_w"][self.motion_anchor_body_index, :][[1, 2, 3, 0]]

            # [底层原理] 将参考动作的 anchor 坐标对齐到机器人初始位置，消除绝对坐标偏差
            if self.command_init_align is not None:
                anchor_quat_w, anchor_pos_w = self.command_init_align.align_transform(anchor_quat_w, anchor_pos_w)

            # [Sim-to-Real] override_robot_anchor_pos：调试模式下用参考动作的 anchor 位置替代真实机器人位置
            # 真机部署时应关闭此选项，使用真实的 base_pos 作为 robot_anchor
            if self.override_robot_anchor_pos:  # OVERRIDE
                robot_anchor_pos_w = anchor_pos_w.copy()
            else:
                base_pos = env_data.torso_pos
                robot_anchor_pos_w = base_pos

            robot_anchor_quat_w = env_data.torso_quat

            return command, robot_anchor_pos_w, robot_anchor_quat_w, anchor_pos_w, anchor_quat_w, None

    def get_observation(self, env_data, ctrl_data):
        # [底层原理] 观测构建：从传感器数据组装 policy 输入 obs tensor
        # obs 由多个分量拼接而成，顺序必须与训练时完全一致
        dof_pos = env_data.dof_pos
        dof_vel = env_data.dof_vel
        ang_vel = env_data.base_ang_vel
        lin_vel = env_data.base_lin_vel

        command, robot_anchor_pos_w, robot_anchor_quat_w, anchor_pos_w, anchor_quat_w, hand_pose = self._get_command(
            env_data, ctrl_data
        )

        # [底层原理] subtract_frame_transforms：计算参考 anchor 相对于机器人 anchor 的相对位姿
        # pos：参考动作 anchor 在机器人本体系下的位置偏差（3D）
        # ori：参考动作 anchor 相对机器人的旋转（四元数）
        pos, ori = subtract_frame_transforms(
            robot_anchor_pos_w,
            robot_anchor_quat_w,
            anchor_pos_w,
            anchor_quat_w,
        )
        # [底层原理] 将旋转矩阵的前两列（6D 旋转表示）作为朝向观测
        # 6D 旋转表示比四元数更适合神经网络学习（无奇异性，梯度连续）
        mat = matrix_from_quat(ori)

        # [面试考点：观测历史] 历史观测的维护方式
        # obs_command：参考动作的关节位置+速度目标（来自动作表查表结果）
        obs_command = command
        # obs_motion_anchor_pos_b：参考 anchor 在机器人本体系下的 3D 位置偏差
        obs_motion_anchor_pos_b = pos
        # obs_motion_anchor_ori_b：参考 anchor 相对朝向（旋转矩阵前两列，6 维）
        obs_motion_anchor_ori_b = mat[:, :2].flatten()

        obs_base_lin_vel = lin_vel
        obs_base_ang_vel = ang_vel
        # [关键参数] 关节位置使用相对默认姿态的偏差，而非绝对角度
        # 这样网络输出可以直接解释为对默认姿态的修正量
        obs_joint_pos_rel = dof_pos - self.default_dof_pos
        obs_joint_vel_rel = dof_vel
        # [面试考点：观测历史] last_action 作为历史动作观测，帮助网络感知自身运动状态
        # 避免网络在不知道上一步动作的情况下产生抖动
        obs_last_action = self.last_action

        # [底层原理] obs 拼接顺序：command | anchor_pos | anchor_ori | lin_vel | ang_vel | joint_pos | joint_vel | last_action
        # without_state_estimator 时去掉 anchor_pos 和 lin_vel（这两项依赖状态估计器）
        obs_prop = np.concatenate(
            [
                obs_command,
                obs_motion_anchor_pos_b if not self.without_state_estimator else [],
                obs_motion_anchor_ori_b,
                obs_base_lin_vel if not self.without_state_estimator else [],
                obs_base_ang_vel,
                obs_joint_pos_rel,
                obs_joint_vel_rel,
                obs_last_action,
            ]
        )

        obs = obs_prop
        extras = {
            "pos": pos,
            "ori": ori,
            "robot_anchor_pos_w": robot_anchor_pos_w,
            "robot_anchor_quat_w": robot_anchor_quat_w,
            "anchor_pos_w": anchor_pos_w,
            "anchor_quat_w": anchor_quat_w,
            "command": command,
            "hand_pose": hand_pose,
            # [工程细节] CALLBACK 机制：动作播放完毕时向 pipeline 发送 [MOTION_DONE] 信号
            "CALLBACK": ["[MOTION_DONE]"] if self.flag_motion_done else [],
        }
        return obs, extras

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        # [底层原理] ONNX 推理：输入 obs 和当前时间步，输出动作及参考动作数据
        # time_step 作为额外输入，使模型能够从内嵌的动作表中查询对应帧的参考数据
        ort_inputs = {
            "obs": np.expand_dims(obs, axis=0).astype(np.float32),
            # [关键参数] time_step 用于模型内部查表，必须与 self.timestep 同步
            "time_step": np.expand_dims(np.array([int(self.timestep)]), axis=0).astype(np.float32),
        }

        # [底层原理] 模型同时输出：
        # actions：策略动作（关节角度增量）
        # joint_pos/joint_vel：当前时间步的参考关节状态（来自内嵌动作表）
        # body_pos_w/body_quat_w：参考身体位姿（用于计算 anchor 误差）
        ort_outputs = self.session.run(
            [
                "actions",
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
            ],
            ort_inputs,
        )
        actions: np.ndarray = np.asarray(ort_outputs[0]).squeeze()

        # [面试考点：动作滤波] EMA 低通滤波防止高频振荡
        # action_beta 为 EMA 系数（接近 1 时响应快但抖动大，接近 0 时平滑但滞后）
        # 公式：filtered = (1 - beta) * last_action + beta * new_action
        # 等价于一阶低通滤波器，截止频率 = beta * policy_freq / (2π)
        actions = (1 - self.action_beta) * self.last_action + self.action_beta * actions
        # [面试考点：动作滤波] 保存滤波后的动作作为下一步的历史，同时作为 obs 中的 last_action
        self.last_action = actions.copy()

        # [关键参数] action_scales 将归一化动作映射到实际关节角度空间（单位：rad）
        scaled_actions = actions * self.action_scales

        # [工程细节] 当使用模型内嵌动作表时，将本步的参考数据缓存到 self.command
        # 供下一步 _get_command() 使用，形成"查表→推理→缓存"的闭环
        if self.use_motion_from_model:
            self.command = {
                "time_step": self.timestep,
                "joint_pos": np.asarray(ort_outputs[1]).squeeze(),
                "joint_vel": np.asarray(ort_outputs[2]).squeeze(),
                "body_pos_w": np.asarray(ort_outputs[3]).squeeze(),
                "body_quat_w": np.asarray(ort_outputs[4]).squeeze(),  # as [w, x, y, z]
            }
        return scaled_actions

    def get_init_dof_pos(self) -> np.ndarray:
        """
        Return first frame of the reference motion.
        """
        if self.command is not None:
            joint_pos = self.command["joint_pos"]
            return joint_pos.copy()
        else:
            return self.default_dof_pos.copy()

    def debug_viz(self, visualizer: MujocoVisualizer, env_data, ctrl_data, extras):
        robot_anchor_pos_w = extras["robot_anchor_pos_w"]
        robot_anchor_quat_w = extras["robot_anchor_quat_w"]
        anchor_pos_w = extras["anchor_pos_w"]
        anchor_quat_w = extras["anchor_quat_w"]

        pos = extras["pos"]
        # ori = extras["ori"]

        visualizer.draw_arrow(anchor_pos_w, anchor_quat_w, [0.2, 0, 0], color=[1, 0, 0, 1], scale=2, id=0)
        visualizer.draw_arrow(
            robot_anchor_pos_w,
            robot_anchor_quat_w,
            [0.2, 0, 0],
            color=[0, 1, 0, 1],
            scale=2,
            id=1,
        )
        visualizer.draw_arrow(robot_anchor_pos_w, robot_anchor_quat_w, pos, color=[0, 1, 1, 1], scale=2, id=2)

        torso_pos = env_data["torso_pos"]
        torso_quat = env_data["torso_quat"]

        visualizer.draw_arrow(torso_pos, torso_quat, [0.2, 0, 0], color=[1, 1, 0, 1], scale=2, id=3)
