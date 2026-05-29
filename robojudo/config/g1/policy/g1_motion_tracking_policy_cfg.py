from robojudo.policy.policy_cfgs import MotionTrackingPolicyCfg
from robojudo.tools.tool_cfgs import DoFConfig


MOTIONTRACKING_KPS = [
    40.17923847,
    99.09842778,
    40.17923847,
    99.09842778,
    28.5012462,
    28.5012462,
    40.17923847,
    99.09842778,
    40.17923847,
    99.09842778,
    28.5012462,
    28.5012462,
    40.17923847,
    28.5012462,
    28.5012462,
    14.2506231,
    14.2506231,
    14.2506231,
    14.2506231,
    14.2506231,
    16.77832748,
    16.77832748,
    14.2506231,
    14.2506231,
    14.2506231,
    14.2506231,
    14.2506231,
    16.77832748,
    16.77832748,
]

MOTIONTRACKING_KDS = [
    2.55788977,
    6.30880185,
    2.55788977,
    6.30880185,
    1.81444569,
    1.81444569,
    2.55788977,
    6.30880185,
    2.55788977,
    6.30880185,
    1.81444569,
    1.81444569,
    2.55788977,
    1.81444569,
    1.81444569,
    0.90722284,
    0.90722284,
    0.90722284,
    0.90722284,
    0.90722284,
    1.0681415,
    1.0681415,
    0.90722284,
    0.90722284,
    0.90722284,
    0.90722284,
    0.90722284,
    1.0681415,
    1.0681415,
]

MOTIONTRACKING_DEFAULT_ANGLES = [
    -0.312,
    0.0,
    0.0,
    0.669,
    -0.363,
    0.0,
    -0.312,
    0.0,
    0.0,
    0.669,
    -0.363,
    0.0,
    0.0,
    0.0,
    0.0,
    0.2,
    0.2,
    0.0,
    0.6,
    0.0,
    0.0,
    0.0,
    0.2,
    -0.2,
    0.0,
    0.6,
    0.0,
    0.0,
    0.0,
]

MOTIONTRACKING_DEFAULT_ANGLES_SEQ = [
    -0.312,
    -0.312,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.669,
    0.669,
    0.2,
    0.2,
    -0.363,
    -0.363,
    0.2,
    -0.2,
    0.0,
    0.0,
    0.0,
    0.0,
    0.6,
    0.6,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
]

MOTIONTRACKING_ACTION_SCALE_SEQ = [
    0.5475464463233948,
    0.5475464463233948,
    0.5475464463233948,
    0.3506614565849304,
    0.3506614565849304,
    0.4385773241519928,
    0.5475464463233948,
    0.5475464463233948,
    0.4385773241519928,
    0.3506614565849304,
    0.3506614565849304,
    0.4385773241519928,
    0.4385773241519928,
    0.4385773241519928,
    0.4385773241519928,
    0.4385773241519928,
    0.4385773241519928,
    0.4385773241519928,
    0.4385773241519928,
    0.4385773241519928,
    0.4385773241519928,
    0.4385773241519928,
    0.4385773241519928,
    0.4385773241519928,
    0.4385773241519928,
    0.07450087368488312,
    0.07450087368488312,
    0.07450087368488312,
    0.07450087368488312,
]


class G1MotionTrackingDoF(DoFConfig):
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

    default_pos: list[float] | None = MOTIONTRACKING_DEFAULT_ANGLES
    stiffness: list[float] | None = MOTIONTRACKING_KPS
    damping: list[float] | None = MOTIONTRACKING_KDS


class G1MotionTrackingPolicyCfg(MotionTrackingPolicyCfg):
    robot: str = "g1"

    policy_name: str = "demo2"
    motion_name: str = "robot_demo2"

    obs_dof: DoFConfig = G1MotionTrackingDoF()
    action_dof: DoFConfig = obs_dof

    action_beta: float = 1.0

    kps: list[float] = MOTIONTRACKING_KPS
    kds: list[float] = MOTIONTRACKING_KDS
    default_angles: list[float] = MOTIONTRACKING_DEFAULT_ANGLES
    default_angles_seq: list[float] = MOTIONTRACKING_DEFAULT_ANGLES_SEQ
    action_scale_seq: list[float] = MOTIONTRACKING_ACTION_SCALE_SEQ
    num_actions: int = 29
    num_obs: int = 154
    motion_length: float = 10.0
