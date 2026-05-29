from robojudo.policy.policy_cfgs import LocoModePolicyCfg
from robojudo.tools.tool_cfgs import DoFConfig


LOCOMODE_JOINT2MOTOR = [
    0, 6, 12,
    1, 7, 13,
    2, 8, 14,
    3, 9, 15, 22,
    4, 10, 16, 23,
    5, 11, 17, 24,
    18, 25,
    19, 26,
    20, 27,
    21, 28,
]

LOCOMODE_DEFAULT_ANGLES = [
    -0.2, -0.2, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
    0.42, 0.42, 0.35, 0.35,
    -0.23, -0.23, 0.18, -0.18,
    0.0, 0.0, 0.0, 0.0,
    0.87, 0.87,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
]

LOCOMODE_KPS = [
    200, 200, 200,
    150, 150, 200,
    150, 150, 200,
    200, 200, 100, 100,
    20, 20, 100, 100,
    20, 20, 50, 50,
    50, 50,
    40, 40,
    40, 40,
    40, 40,
]

LOCOMODE_KDS = [
    5, 5, 5,
    5, 5, 5,
    5, 5, 5,
    5, 5, 2, 2,
    2, 2, 2, 2,
    2, 2, 2, 2,
    2, 2,
    2, 2,
    2, 2,
    2, 2,
]

LOCOMODE_TAU_LIMIT = [
    88, 88, 88, 139, 50, 50,
    88, 88, 88, 139, 50, 50,
    88, 50, 50,
    25, 25, 25, 25, 5, 5, 5,
    25, 25, 25, 25, 5, 5, 5,
]


def _reorder_by_joint2motor(values: list[float]) -> list[float]:
    reordered = [0.0 for _ in range(len(values))]
    for i, motor_idx in enumerate(LOCOMODE_JOINT2MOTOR):
        reordered[motor_idx] = values[i]
    return reordered


class G1LocoModeDoF(DoFConfig):
    # Mujoco joint order (motor order)
    joint_names: list[str] = [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]

    default_pos: list[float] | None = _reorder_by_joint2motor(LOCOMODE_DEFAULT_ANGLES)
    stiffness: list[float] | None = _reorder_by_joint2motor(LOCOMODE_KPS)
    damping: list[float] | None = _reorder_by_joint2motor(LOCOMODE_KDS)
    torque_limits: list[float] | None = LOCOMODE_TAU_LIMIT


class G1LocoModePolicyCfg(LocoModePolicyCfg):
    robot: str = "g1"

    obs_dof: DoFConfig = G1LocoModeDoF()
    action_dof: DoFConfig = obs_dof

    kps: list[float] = LOCOMODE_KPS
    kds: list[float] = LOCOMODE_KDS
    default_angles: list[float] = LOCOMODE_DEFAULT_ANGLES
    joint2motor_idx: list[int] = LOCOMODE_JOINT2MOTOR
    tau_limit: list[float] = LOCOMODE_TAU_LIMIT

    num_actions: int = 29
    num_obs: int = 96

    ang_vel_scale: float = 1.0
    dof_pos_scale: float = 1.0
    dof_vel_scale: float = 1.0
    action_scale: float = 0.25

    cmd_scale: list[float] = [1.0, 1.0, 1.0]
    cmd_init: list[float] = [0.0, 0.0, 0.0]
    cmd_range: dict[str, list[float]] = {
        "lin_vel_x": [-0.4, 0.7],
        "lin_vel_y": [-0.4, 0.4],
        "ang_vel_z": [-1.57, 1.57],
    }
    cmd_deadzone: float = 0.05
    cmd_signs: list[float] = [1.0, -1.0, -1.0]