import logging

import numpy as np
import torch

from robojudo.policy import Policy, policy_registry
from robojudo.policy.policy_cfgs import LocoModePolicyCfg
from robojudo.utils.util_func import get_gravity_orientation

logger = logging.getLogger(__name__)


def _scale_values(values: np.ndarray, target_ranges: list[list[float]]) -> np.ndarray:
    scaled = []
    for val, (new_min, new_max) in zip(values, target_ranges):
        scaled_val = (val + 1) * (new_max - new_min) / 2 + new_min
        scaled.append(scaled_val)
    return np.array(scaled, dtype=np.float32)


@policy_registry.register
class LocoModePolicy(Policy):
    """
    Ported from RoboMimicDeploy_G1/policy/loco_mode/LocoMode.py
    """

    cfg_policy: LocoModePolicyCfg

    def __init__(self, cfg_policy: LocoModePolicyCfg, device: str = "cpu"):
        self.policy = torch.jit.load(cfg_policy.policy_file, map_location=device)
        super().__init__(cfg_policy=cfg_policy, device=device)

        self.num_actions = cfg_policy.num_actions
        self.num_obs = cfg_policy.num_obs

        self.kps = np.asarray(cfg_policy.kps, dtype=np.float32)
        self.kds = np.asarray(cfg_policy.kds, dtype=np.float32)
        self.default_angles = np.asarray(cfg_policy.default_angles, dtype=np.float32)
        self.joint2motor_idx = np.asarray(cfg_policy.joint2motor_idx, dtype=np.int32)
        self.tau_limit = np.asarray(cfg_policy.tau_limit, dtype=np.float32)

        self.ang_vel_scale = cfg_policy.ang_vel_scale
        self.dof_pos_scale = cfg_policy.dof_pos_scale
        self.dof_vel_scale = cfg_policy.dof_vel_scale
        self.action_scale = cfg_policy.action_scale

        self.cmd_scale = np.asarray(cfg_policy.cmd_scale, dtype=np.float32)
        self.range_velx = np.asarray(cfg_policy.cmd_range["lin_vel_x"], dtype=np.float32)
        self.range_vely = np.asarray(cfg_policy.cmd_range["lin_vel_y"], dtype=np.float32)
        self.range_velz = np.asarray(cfg_policy.cmd_range["ang_vel_z"], dtype=np.float32)

        self.qj_obs = np.zeros(self.num_actions, dtype=np.float32)
        self.dqj_obs = np.zeros(self.num_actions, dtype=np.float32)
        self.cmd = np.asarray(cfg_policy.cmd_init, dtype=np.float32)
        self.obs = np.zeros(self.num_obs, dtype=np.float32)
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.default_angles_reorder = self._reorder_by_joint2motor(self.default_angles)

        for _ in range(50):
            with torch.inference_mode():
                obs_tensor = torch.from_numpy(self.obs.reshape(1, -1)).float()
                self.policy(obs_tensor)

        logger.info("LocoMode policy initializing ...")
        self.reset()

    def reset(self):
        self.timestep = 0
        self.action[:] = 0.0
        self.cmd = np.asarray(self.cfg_policy.cmd_init, dtype=np.float32)

    def post_step_callback(self, commands=None):
        self.timestep += 1

    def _get_joy_cmd(self, ctrl_data) -> np.ndarray:
        deadzone = self.cfg_policy.cmd_deadzone
        signs = np.asarray(self.cfg_policy.cmd_signs, dtype=np.float32)

        def apply_deadzone(val: float) -> float:
            return 0.0 if abs(val) < deadzone else val

        for key in ctrl_data.keys():
            if key in ["JoystickCtrl", "UnitreeCtrl"]:
                axes = ctrl_data[key]["axes"]
                lx = apply_deadzone(axes["LeftX"])
                ly = apply_deadzone(axes["LeftY"])
                rx = apply_deadzone(axes["RightX"])
                cmd = np.array([ly, lx, rx], dtype=np.float32)
                return cmd * signs
        return np.zeros(3, dtype=np.float32)

    @staticmethod
    def _get_gravity_orientation_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
        qw, qx, qy, qz = quat_wxyz
        gravity_orientation = np.zeros(3, dtype=np.float32)
        gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
        gravity_orientation[1] = -2 * (qz * qy + qw * qx)
        gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
        return gravity_orientation

    def _reorder_by_joint2motor(self, values: np.ndarray) -> np.ndarray:
        reordered = np.zeros_like(values)
        for i in range(len(self.joint2motor_idx)):
            motor_idx = self.joint2motor_idx[i]
            reordered[motor_idx] = values[i]
        return reordered

    def get_observation(self, env_data, ctrl_data):
        qj = env_data.dof_pos
        dqj = env_data.dof_vel
        ang_vel = env_data.base_ang_vel
        # LocoMode uses wxyz quaternion in RoboMimicDeploy_G1
        quat_wxyz = env_data.base_quat[[3, 0, 1, 2]]
        gravity_orientation = self._get_gravity_orientation_wxyz(quat_wxyz)

        joycmd = self._get_joy_cmd(ctrl_data)
        self.cmd = _scale_values(joycmd, [self.range_velx, self.range_vely, self.range_velz])

        for i in range(len(self.joint2motor_idx)):
            self.qj_obs[i] = qj[self.joint2motor_idx[i]]
            self.dqj_obs[i] = dqj[self.joint2motor_idx[i]]

        qj_obs = (self.qj_obs - self.default_angles) * self.dof_pos_scale
        dqj_obs = self.dqj_obs * self.dof_vel_scale
        ang_vel = ang_vel * self.ang_vel_scale
        cmd = self.cmd * self.cmd_scale

        self.obs[:3] = ang_vel
        self.obs[3:6] = gravity_orientation
        self.obs[6:9] = cmd
        self.obs[9 : 9 + self.num_actions] = qj_obs
        self.obs[9 + self.num_actions : 9 + self.num_actions * 2] = dqj_obs
        self.obs[9 + self.num_actions * 2 : 9 + self.num_actions * 3] = self.action

        extras = {"command": cmd}
        return self.obs.copy(), extras

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        obs_tensor = torch.from_numpy(obs.reshape(1, -1)).float().clip(-100, 100)
        with torch.inference_mode():
            action = self.policy(obs_tensor).clip(-100, 100).detach().cpu().numpy().squeeze()

        self.action = action.astype(np.float32)

        loco_action = self.action * self.action_scale + self.default_angles
        action_reorder = loco_action.copy()
        for i in range(len(self.joint2motor_idx)):
            motor_idx = self.joint2motor_idx[i]
            action_reorder[motor_idx] = loco_action[i]

        # PolicyWrapper will add default_pos again, so return delta to avoid double offset.
        action_delta = action_reorder - self.default_angles_reorder
        return action_delta.astype(np.float32)
