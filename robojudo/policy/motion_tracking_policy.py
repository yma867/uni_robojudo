import logging

import numpy as np
import onnxruntime as ort
import mujoco
from scipy.spatial.transform import Rotation as R

from robojudo.policy import Policy, policy_registry
from robojudo.policy.policy_cfgs import MotionTrackingPolicyCfg

logger = logging.getLogger(__name__)


@policy_registry.register
class MotionTrackingPolicy(Policy):
    """
    Ported from RoboMimicDeploy_G1/policy/montion_tracking/MotionTracking.py
    """

    cfg_policy: MotionTrackingPolicyCfg

    def __init__(self, cfg_policy: MotionTrackingPolicyCfg, device: str = "cpu"):
        sess_options = ort.SessionOptions()
        providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(cfg_policy.policy_file, sess_options, providers=providers)
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]

        # Load motion data
        motion = np.load(cfg_policy.motion_file)
        self.motionpos = motion["body_pos_w"].astype(np.float32)
        self.motionquat = motion["body_quat_w"].astype(np.float32)
        self.motioninputpos = motion["joint_pos"].astype(np.float32)
        self.motioninputvel = motion["joint_vel"].astype(np.float32)
        self.total_frames = self.motioninputpos.shape[0]
        # Normalize motion quaternions to avoid drift from stored data.
        norms = np.linalg.norm(self.motionquat, axis=-1, keepdims=True)
        self.motionquat = self.motionquat / np.clip(norms, 1e-8, None)

        cfg_policy_new = cfg_policy.model_copy()
        super().__init__(cfg_policy=cfg_policy_new, device=device)

        self.num_actions = cfg_policy.num_actions
        self.num_obs = cfg_policy.num_obs

        self.kps = np.asarray(cfg_policy.kps, dtype=np.float32)
        self.kds = np.asarray(cfg_policy.kds, dtype=np.float32)
        self.default_angles_xml = np.asarray(cfg_policy.default_angles, dtype=np.float32)
        self.default_angles_seq = np.asarray(cfg_policy.default_angles_seq, dtype=np.float32)
        self.action_scale_seq = np.asarray(cfg_policy.action_scale_seq, dtype=np.float32)

        self.joint_xml = [
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
        self.joint_seq = [
            "left_hip_pitch_joint",
            "right_hip_pitch_joint",
            "waist_yaw_joint",
            "left_hip_roll_joint",
            "right_hip_roll_joint",
            "waist_roll_joint",
            "left_hip_yaw_joint",
            "right_hip_yaw_joint",
            "waist_pitch_joint",
            "left_knee_joint",
            "right_knee_joint",
            "left_shoulder_pitch_joint",
            "right_shoulder_pitch_joint",
            "left_ankle_pitch_joint",
            "right_ankle_pitch_joint",
            "left_shoulder_roll_joint",
            "right_shoulder_roll_joint",
            "left_ankle_roll_joint",
            "right_ankle_roll_joint",
            "left_shoulder_yaw_joint",
            "right_shoulder_yaw_joint",
            "left_elbow_joint",
            "right_elbow_joint",
            "left_wrist_roll_joint",
            "right_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "right_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_wrist_yaw_joint",
        ]
        self._joint_xml_to_seq = {joint: i for i, joint in enumerate(self.joint_xml)}
        self._joint_seq_to_xml = {joint: i for i, joint in enumerate(self.joint_seq)}

        self.obs = np.zeros(self.num_obs, dtype=np.float32)
        self.action_buffer = np.zeros(self.num_actions, dtype=np.float32)

        self.timestep = 0

        self.world_transform_quat = None
        self.align_pending = True

        # Warmup
        self.get_action(self.obs.copy())
        self.reset()

    def reset(self):
        self.timestep = 0
        self.action_buffer[:] = 0.0
        self.obs[:] = 0.0
        self.motion_done = False
        self.align_pending = True
        self.world_transform_quat = None

    def post_step_callback(self, commands: list[str] | None = None):
        self.timestep += 1
        if self.timestep >= self.total_frames:
            self.motion_done = True

    @staticmethod
    def _quat_conjugate_wxyz(q: np.ndarray) -> np.ndarray:
        return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)

    @staticmethod
    def _quat_multiply_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return np.array([w, x, y, z], dtype=np.float32)

    @staticmethod
    def _subtract_frame_transforms_wxyz(
        pos_a: np.ndarray, quat_a: np.ndarray, pos_b: np.ndarray, quat_b: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        rotm_a = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rotm_a, quat_a.astype(np.float64))
        rotm_a = rotm_a.reshape(3, 3)
        rel_pos = rotm_a.T @ (pos_b - pos_a)
        rel_quat = MotionTrackingPolicy._quat_multiply_wxyz(
            MotionTrackingPolicy._quat_conjugate_wxyz(quat_a), quat_b
        )
        rel_quat = rel_quat / np.linalg.norm(rel_quat)
        return rel_pos.astype(np.float32), rel_quat.astype(np.float32)

    def _update_world_transform(self, base_quat_wxyz: np.ndarray):
        motion_initial_quat = self.motionquat[0, 9, :]  # torso_link in body_names_all

        current_quat_xyzw = np.array(
            [base_quat_wxyz[1], base_quat_wxyz[2], base_quat_wxyz[3], base_quat_wxyz[0]], dtype=np.float32
        )
        motion_quat_xyzw = np.array(
            [motion_initial_quat[1], motion_initial_quat[2], motion_initial_quat[3], motion_initial_quat[0]],
            dtype=np.float32,
        )

        current_r = R.from_quat(current_quat_xyzw)
        motion_r = R.from_quat(motion_quat_xyzw)
        current_yaw = current_r.as_euler("xyz")[2]
        motion_yaw = motion_r.as_euler("xyz")[2]
        yaw_diff = current_yaw - motion_yaw

        yaw_transform_r = R.from_euler("z", yaw_diff)
        world_transform_quat_xyzw = yaw_transform_r.as_quat()
        self.world_transform_quat = np.array(
            [
                world_transform_quat_xyzw[3],
                world_transform_quat_xyzw[0],
                world_transform_quat_xyzw[1],
                world_transform_quat_xyzw[2],
            ],
            dtype=np.float32,
        )
        self.align_pending = False

    def get_observation(self, env_data, ctrl_data):
        qj = env_data.dof_pos
        dqj = env_data.dof_vel
        ang_vel = env_data.base_ang_vel

        base_pos = env_data.torso_pos if env_data.torso_pos is not None else env_data.base_pos
        if base_pos is None:
            base_pos = np.zeros(3, dtype=np.float32)

        base_quat_xyzw = env_data.torso_quat if env_data.torso_quat is not None else env_data.base_quat
        if base_quat_xyzw is None:
            base_quat_xyzw = np.array([0, 0, 0, 1], dtype=np.float32)
        base_quat_wxyz = base_quat_xyzw[[3, 0, 1, 2]]

        if self.align_pending:
            self._update_world_transform(base_quat_wxyz)

        frame_idx = min(self.timestep, self.total_frames - 1)
        motioninput = np.concatenate(
            (self.motioninputpos[frame_idx, :], self.motioninputvel[frame_idx, :]), axis=0
        ).astype(np.float32)

        motionposcurrent = self.motionpos[frame_idx, 9, :]
        motionquatcurrent = self.motionquat[frame_idx, 9, :]
        if self.world_transform_quat is not None:
            motionquatcurrent = self._quat_multiply_wxyz(self.world_transform_quat, motionquatcurrent)
        motionquatcurrent = motionquatcurrent / np.clip(np.linalg.norm(motionquatcurrent), 1e-8, None)

        _, anchor_quat = self._subtract_frame_transforms_wxyz(base_pos, base_quat_wxyz, motionposcurrent, motionquatcurrent)

        anchor_ori = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(anchor_ori, anchor_quat.astype(np.float64))
        anchor_ori = anchor_ori.reshape(3, 3)[:, :2].reshape(-1)

        qj_obs_seq = np.array([qj[self._joint_xml_to_seq[joint]] for joint in self.joint_seq], dtype=np.float32)
        dqj_obs_seq = np.array([dqj[self._joint_xml_to_seq[joint]] for joint in self.joint_seq], dtype=np.float32)

        offset = 0
        self.obs[offset : offset + 58] = motioninput
        offset += 58
        self.obs[offset : offset + 6] = anchor_ori
        offset += 6
        self.obs[offset : offset + 3] = ang_vel
        offset += 3
        self.obs[offset : offset + 29] = qj_obs_seq - self.default_angles_seq
        offset += 29
        self.obs[offset : offset + 29] = dqj_obs_seq
        offset += 29
        self.obs[offset : offset + 29] = self.action_buffer

        extras = {"CALLBACK": ["[MOTION_DONE]"] if self.motion_done else []}
        return self.obs.copy(), extras

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        ort_inputs = {}
        for inp in self.session.get_inputs():
            name = inp.name
            shape = inp.shape
            if name == "obs":
                ort_inputs[name] = np.expand_dims(obs, axis=0).astype(np.float32)
            elif name == "time_step":
                ort_inputs[name] = np.array([[float(self.timestep)]], dtype=np.float32)
            else:
                dims = []
                for s in shape:
                    if isinstance(s, str) or s is None:
                        dims.append(1)
                    else:
                        dims.append(int(s))
                if len(dims) == 0 or dims[0] != 1:
                    dims = [1] + dims[1:]
                ort_inputs[name] = np.zeros(dims, dtype=np.float32)

        ort_outputs = self.session.run(self.output_names, ort_inputs)
        action = np.asarray(ort_outputs[0]).reshape(-1).astype(np.float32)
        self.action_buffer = action.copy()

        target_dof_pos_seq = self.default_angles_seq + action * self.action_scale_seq
        target_dof_pos = np.array(
            [target_dof_pos_seq[self._joint_seq_to_xml[joint]] for joint in self.joint_xml], dtype=np.float32
        )

        action_delta = target_dof_pos - self.default_angles_xml
        return action_delta

    def get_init_dof_pos(self) -> np.ndarray:
        # Keep consistent with RoboMimic: no forced pose to motion frame at entry.
        return self.default_angles_xml.copy()
