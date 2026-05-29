from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import yaml

from robojudo.config import ROOT_DIR
from robojudo.policy import Policy, policy_registry
from robojudo.policy.policy_cfgs import MimicKitPolicyCfg
from robojudo.mimickit_runtime import MJCFCharModel, MotionLib
from robojudo.mimickit_runtime.motion_lib import extract_pose_data
from robojudo.mimickit_runtime import torch_util


@policy_registry.register
class MimicKitPolicy(Policy):
    cfg_policy: MimicKitPolicyCfg

    def __init__(self, cfg_policy: MimicKitPolicyCfg, device: str = "cpu"):
        cfg = cfg_policy.model_copy()
        self._load_env_config(cfg)

        self.torch_device = torch.device("cpu")
        self.kin_char_model = MJCFCharModel(self.torch_device)
        self.kin_char_model.load(self.char_file)
        self.key_body_ids = self._resolve_body_ids(self.key_body_names)

        # Verify joint order (important for alignment with training)
        # Joint order is determined by DFS traversal of XML, should match training
        if hasattr(self.kin_char_model, "get_body_names"):
            body_names = self.kin_char_model.get_body_names()
            # Log joint order for verification (joints start from index 1, root is 0)
            joint_names = []
            for j in range(1, self.kin_char_model.get_num_joints()):
                joint = self.kin_char_model.get_joint(j)
                joint_names.append(joint.name)
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"[MimicKitPolicy] Joint order (DFS): {joint_names}")

        self.motion_lib = MotionLib(motion_file=self.motion_file, kin_char_model=self.kin_char_model, device=self.torch_device)
        self.motion_ids = torch.tensor([0], device=self.torch_device, dtype=torch.long)
        self.motion_length = self.motion_lib.get_motion_length(self.motion_ids)[0].item()
        self.motion_loop_mode = self.motion_lib.get_motion_loop_mode(self.motion_ids)[0].item()
        self.motion_dt = float(self.motion_lib._motion_dt[0].item())

        if cfg.model_pt_file:
            self._load_model_pt(cfg.model_pt_file)
        elif cfg.model_npz_file:
            self._load_model_npz(cfg.model_npz_file)
        else:
            raise ValueError("Either model_pt_file or model_npz_file must be provided.")

        if self.action_mean is None or self.action_std is None:
            self.action_mean, self.action_std = self._build_action_norm(
                position_limits=cfg.action_dof.position_limits,
                zero_center_action=self.zero_center_action,
                num_actions=cfg.action_dof.num_dofs,
            )

        # Build action bounds for clipping (matching MimicKit's _apply_action clip logic)
        self.action_bound_low, self.action_bound_high = self._build_action_bounds(
            position_limits=cfg.action_dof.position_limits,
            zero_center_action=self.zero_center_action,
            num_actions=cfg.action_dof.num_dofs,
        )

        super().__init__(cfg_policy=cfg, device=device)
        self.init_dof_pos = self._build_init_dof_pos()

        # ---- debug dump (one-shot) ----
        env_dump = os.getenv("ROBOJUDO_MIMICKIT_DUMP", "")
        self._debug_dump_once = bool(cfg.debug_dump_once) or env_dump == "1"
        self._debug_dump_path = cfg.debug_dump_path or os.getenv("ROBOJUDO_MIMICKIT_DUMP_PATH", "")
        self._debug_dump_done = False
        self._debug_last = {}

        self.reset()

    def _build_init_dof_pos(self) -> np.ndarray:
        """Build initial dof_pos from init_pose (for environment default_pos)"""
        if self.init_pose is not None:
            _, _, joint_dof = extract_pose_data(np.asarray(self.init_pose, dtype=np.float32))
            return np.asarray(joint_dof, dtype=np.float32)

        motion_times = torch.zeros((1,), device=self.torch_device, dtype=torch.float32)
        _, _, _, _, joint_rot, _ = self.motion_lib.calc_motion_frame(self.motion_ids, motion_times)
        dof_pos = self.motion_lib.joint_rot_to_dof(joint_rot).squeeze(0).cpu().numpy()
        return dof_pos.astype(np.float32)

    def _build_motion_first_frame_dof_pos(self) -> np.ndarray:
        """Build dof_pos from motion first frame (t=0) - this is what MimicKit uses for reset"""
        motion_times = torch.zeros((1,), device=self.torch_device, dtype=torch.float32)
        _, _, _, _, joint_rot, _ = self.motion_lib.calc_motion_frame(self.motion_ids, motion_times)
        dof_pos = self.motion_lib.joint_rot_to_dof(joint_rot).squeeze(0).cpu().numpy()
        return dof_pos.astype(np.float32)

    @property
    def default_pose_for_action(self) -> np.ndarray:
        """For zero_center_action=True, this should be the motion first frame pose (ref_dof_pos in MimicKit)"""
        if not hasattr(self, '_motion_first_frame_dof_pos'):
            self._motion_first_frame_dof_pos = self._build_motion_first_frame_dof_pos()
        return self._motion_first_frame_dof_pos

    def _load_env_config(self, cfg: MimicKitPolicyCfg):
        env_cfg = yaml.safe_load(Path(cfg.env_config_path).read_text(encoding="utf-8"))

        self.global_obs = env_cfg["global_obs"] if cfg.global_obs is None else cfg.global_obs
        self.root_height_obs = env_cfg.get("root_height_obs", True) if cfg.root_height_obs is None else cfg.root_height_obs
        self.enable_phase_obs = env_cfg.get("enable_phase_obs", True) if cfg.enable_phase_obs is None else cfg.enable_phase_obs
        self.enable_tar_obs = env_cfg.get("enable_tar_obs", False) if cfg.enable_tar_obs is None else cfg.enable_tar_obs
        self.num_phase_encoding = env_cfg.get("num_phase_encoding", 0) if cfg.num_phase_encoding is None else cfg.num_phase_encoding
        self.tar_obs_steps = env_cfg.get("tar_obs_steps", [1]) if cfg.tar_obs_steps is None else cfg.tar_obs_steps
        self.key_body_names = env_cfg.get("key_bodies", []) if cfg.key_bodies is None else cfg.key_bodies

        self.zero_center_action = env_cfg.get("zero_center_action", False) if cfg.zero_center_action is None else cfg.zero_center_action
        self.obs_clip = cfg.obs_clip
        self.init_pose = env_cfg.get("init_pose", None)

        self.char_file = self._resolve_mimickit_path(cfg.char_file, env_cfg.get("char_file"))
        self.motion_file = self._resolve_mimickit_path(cfg.motion_file, env_cfg.get("motion_file"))

    @staticmethod
    def _resolve_mimickit_path(path_override: str | None, path_from_env: str | None) -> str:
        path = path_override or path_from_env
        if path is None:
            raise ValueError("MimicKit asset path missing.")
        path_obj = Path(path)
        if path_obj.is_absolute():
            return path_obj.as_posix()
        return (ROOT_DIR / path_obj).as_posix()

    def _load_model_pt(self, model_pt_file: str):
        state = torch.load(model_pt_file, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        if not isinstance(state, dict):
            raise ValueError("Unsupported model format: expected a state_dict.")

        actor_keys = [k for k in state.keys() if "actor_layers" in k and "weight" in k]
        if not actor_keys:
            raise ValueError("No actor_layers found in .pt file.")

        self.weights = []
        self.biases = []
        # Extract Linear layer weights (skip ReLU activations in Sequential)
        # Sequential structure: Linear(0) -> ReLU(1) -> Linear(2) -> ReLU(3)
        for key in sorted(actor_keys):
            if ".weight" in key:
                layer_idx = int(key.split(".")[-2])
                w_key = f"_model._actor_layers.{layer_idx}.weight"
                b_key = f"_model._actor_layers.{layer_idx}.bias"
                if w_key in state and b_key in state:
                    self.weights.append(state[w_key].cpu().numpy().astype(np.float32))
                    self.biases.append(state[b_key].cpu().numpy().astype(np.float32))

        action_head_w_key = "_model._action_dist._mean_net.weight"
        action_head_b_key = "_model._action_dist._mean_net.bias"
        if action_head_w_key in state and action_head_b_key in state:
            self.weights.append(state[action_head_w_key].cpu().numpy().astype(np.float32))
            self.biases.append(state[action_head_b_key].cpu().numpy().astype(np.float32))
        else:
            raise ValueError("Missing action_dist._mean_net in .pt file.")

        obs_norm_mean_key = "_obs_norm._mean"
        obs_norm_std_key = "_obs_norm._std"
        if obs_norm_mean_key in state and obs_norm_std_key in state:
            self.obs_mean = state[obs_norm_mean_key].cpu().numpy().astype(np.float32)
            obs_std = state[obs_norm_std_key].cpu().numpy().astype(np.float32)
            self.obs_var = np.square(obs_std)
        else:
            raise ValueError("Missing obs_norm in .pt file.")

        action_norm_mean_key = "_a_norm._mean"
        action_norm_std_key = "_a_norm._std"
        if action_norm_mean_key in state and action_norm_std_key in state:
            self.action_mean = state[action_norm_mean_key].cpu().numpy().astype(np.float32)
            self.action_std = state[action_norm_std_key].cpu().numpy().astype(np.float32)

    def _load_model_npz(self, model_npz_file: str):
        data = np.load(model_npz_file)

        self.weights = []
        self.biases = []
        layer_idx = 0
        while f"w{layer_idx}" in data and f"b{layer_idx}" in data:
            self.weights.append(data[f"w{layer_idx}"].astype(np.float32))
            self.biases.append(data[f"b{layer_idx}"].astype(np.float32))
            layer_idx += 1
        if not self.weights:
            raise ValueError(f"No layers found in MimicKit model file: {model_npz_file}")

        self.action_mean = None
        self.action_std = None

        if "obs_mean" in data:
            self.obs_mean = data["obs_mean"].astype(np.float32)
        elif "running_mean" in data:
            self.obs_mean = data["running_mean"].astype(np.float32)
        else:
            raise ValueError("Missing obs mean in model npz (expected obs_mean or running_mean).")

        if "obs_var" in data:
            self.obs_var = data["obs_var"].astype(np.float32)
        elif "running_var" in data:
            self.obs_var = data["running_var"].astype(np.float32)
        elif "obs_std" in data:
            self.obs_var = np.square(data["obs_std"].astype(np.float32))
        else:
            raise ValueError("Missing obs var/std in model npz (expected obs_var/obs_std/running_var).")

        if "action_mean" in data and "action_std" in data:
            self.action_mean = data["action_mean"].astype(np.float32)
            self.action_std = data["action_std"].astype(np.float32)

    @staticmethod
    def _build_action_norm(
        position_limits: list[list[float]] | None, zero_center_action: bool, num_actions: int
    ):
        if position_limits is None or len(position_limits) == 0:
            mean = np.zeros((num_actions,), dtype=np.float32)
            std = np.ones((num_actions,), dtype=np.float32)
            return mean, std

        limits = np.asarray(position_limits, dtype=np.float32)
        low = limits[:, 0]
        high = limits[:, 1]

        if zero_center_action:
            mid = np.zeros_like(low)
        else:
            mid = 0.5 * (low + high)

        scale = np.maximum(np.abs(high - mid), np.abs(low - mid))
        scale = np.maximum(scale, 1e-6) * 1.4
        return mid, scale

    @staticmethod
    def _build_action_bounds(
        position_limits: list[list[float]] | None, zero_center_action: bool, num_actions: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build action bounds for clipping, matching MimicKit's action space construction.
        In MimicKit, action bounds are computed based on joint limits and zero_center_action.
        """
        if position_limits is None or len(position_limits) == 0:
            # No limits: use very wide bounds (matching MimicKit's behavior for unlimited joints)
            low = np.full((num_actions,), -np.inf, dtype=np.float32)
            high = np.full((num_actions,), np.inf, dtype=np.float32)
            return low, high

        limits = np.asarray(position_limits, dtype=np.float32)
        low = limits[:, 0]
        high = limits[:, 1]

        if zero_center_action:
            mid = np.zeros_like(low)
        else:
            mid = 0.5 * (low + high)

        scale = np.maximum(np.abs(high - mid), np.abs(low - mid))
        scale = np.maximum(scale, 1e-6) * 1.4

        action_bound_low = mid - scale
        action_bound_high = mid + scale

        return action_bound_low, action_bound_high

    def _resolve_body_ids(self, body_names: list[str]) -> list[int]:
        if not body_names:
            return []
        model_body_names = self.kin_char_model.get_body_names()
        body_ids = []
        for name in body_names:
            if name not in model_body_names:
                raise ValueError(f"Body {name} not found in MimicKit model.")
            body_ids.append(model_body_names.index(name))
        return body_ids

    def _get_motion_time(self) -> float:
        return self.motion_time

    def reset(self):
        self.timestep = 0
        self.motion_time = 0.0
        self.play_speed = 1.0
        self.flag_motion_done = False
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        # Reset debug dump flag so we can dump on first observation after reset
        if self._debug_dump_once:
            self._debug_dump_done = False

    def post_step_callback(self, commands: list[str] | None = None):
        self.timestep += 1
        self.motion_time += self.motion_dt * self.play_speed

        if self.motion_loop_mode == 0 and self.motion_time >= self.motion_length:
            self.flag_motion_done = True
            self.play_speed = 0.0

        for command in commands or []:
            match command:
                case "[MOTION_RESET]":
                    self.reset()
                case "[MOTION_FADE_IN]":
                    self.play_speed = 1.0
                case "[MOTION_FADE_OUT]":
                    self.play_speed = 0.0

    def _compute_tar_obs_data(self, motion_times: torch.Tensor):
        tar_root_pos_list = []
        tar_root_rot_list = []
        tar_joint_rot_list = []
        tar_key_pos_list = []

        for step in self.tar_obs_steps:
            tar_time = motion_times + float(step) * self.motion_dt
            root_pos, root_rot, _, _, joint_rot, _ = self.motion_lib.calc_motion_frame(self.motion_ids, tar_time)
            tar_root_pos_list.append(root_pos)
            tar_root_rot_list.append(root_rot)
            tar_joint_rot_list.append(joint_rot)

            if self.key_body_ids:
                body_pos, _ = self.kin_char_model.forward_kinematics(root_pos, root_rot, joint_rot)
                key_pos = body_pos[:, self.key_body_ids, :]
                tar_key_pos_list.append(key_pos)

        tar_root_pos = torch.stack(tar_root_pos_list, dim=1)
        tar_root_rot = torch.stack(tar_root_rot_list, dim=1)
        tar_joint_rot = torch.stack(tar_joint_rot_list, dim=1)

        if self.key_body_ids:
            tar_key_pos = torch.stack(tar_key_pos_list, dim=1)
        else:
            tar_key_pos = torch.zeros([0], device=self.torch_device)

        return tar_root_pos, tar_root_rot, tar_joint_rot, tar_key_pos

    def get_observation(self, env_data, ctrl_data):
        root_pos = env_data.torso_pos if env_data.torso_pos is not None else env_data.base_pos
        root_rot = env_data.torso_quat if env_data.torso_quat is not None else env_data.base_quat
        root_vel = env_data.base_lin_vel if env_data.base_lin_vel is not None else np.zeros(3, dtype=np.float32)
        root_ang_vel = env_data.base_ang_vel

        if root_pos is None:
            root_pos = np.zeros(3, dtype=np.float32)
        if root_rot is None:
            root_rot = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        dof_pos = env_data.dof_pos
        dof_vel = env_data.dof_vel

        root_pos_t = torch.tensor(root_pos, device=self.torch_device, dtype=torch.float32).unsqueeze(0)
        root_rot_t = torch.tensor(root_rot, device=self.torch_device, dtype=torch.float32).unsqueeze(0)
        root_vel_t = torch.tensor(root_vel, device=self.torch_device, dtype=torch.float32).unsqueeze(0)
        root_ang_vel_t = torch.tensor(root_ang_vel, device=self.torch_device, dtype=torch.float32).unsqueeze(0)
        dof_pos_t = torch.tensor(dof_pos, device=self.torch_device, dtype=torch.float32).unsqueeze(0)
        dof_vel_t = torch.tensor(dof_vel, device=self.torch_device, dtype=torch.float32).unsqueeze(0)

        # CRITICAL FIX: RoboJuDo's base_lin_vel is body-local (converted via quat_rotate_inverse_np in MujocoEnv),
        # but MimicKit's IsaacLab engine returns world-frame velocity (obj.data.root_link_vel_w[:, :3]).
        # Convert from body-local to world-frame using root_rot_t to match MimicKit's expectation.
        # In MujocoEnv: lin_vel = quat_rotate_inverse_np(quat, lin_vel_world) -> body-local
        # Reverse: lin_vel_world = quat_rotate(quat, lin_vel_body_local)
        root_vel_t = torch_util.quat_rotate(root_rot_t, root_vel_t)  # Convert body-local to world-frame
        # Note: root_ang_vel is already in world-frame (from data.qvel[3:6] without rotation in MujocoEnv)

        joint_rot_t = self.kin_char_model.dof_to_rot(dof_pos_t)

        if self.key_body_ids:
            body_pos_t, _ = self.kin_char_model.forward_kinematics(root_pos_t, root_rot_t, joint_rot_t)
            key_pos_t = body_pos_t[:, self.key_body_ids, :]
        else:
            key_pos_t = torch.zeros([0], device=self.torch_device)

        motion_times = torch.tensor([self._get_motion_time()], device=self.torch_device, dtype=torch.float32)
        if self.enable_phase_obs:
            motion_phase = self.motion_lib.calc_motion_phase(self.motion_ids, motion_times)
        else:
            motion_phase = torch.zeros([0], device=self.torch_device)

        if self.enable_tar_obs:
            tar_root_pos, tar_root_rot, tar_joint_rot, tar_key_pos = self._compute_tar_obs_data(motion_times)
        else:
            tar_root_pos = torch.zeros([0], device=self.torch_device)
            tar_root_rot = tar_root_pos
            tar_joint_rot = tar_root_pos
            tar_key_pos = tar_root_pos

        # Manually construct observation to ensure exact alignment with training
        # Following motion_tracking success pattern: manual concatenation instead of relying on original functions
        obs_parts = []

        # 1. Char obs (character observation)
        # 1.1 Root height (if enabled)
        if self.root_height_obs:
            root_h = root_pos_t[:, 2:3]  # [batch, 1]
            obs_parts.append(root_h)

        # 1.2 Root rotation (6D tangent-normal)
        if self.global_obs:
            root_rot_obs = torch_util.quat_to_tan_norm(root_rot_t)  # [batch, 6]
        else:
            heading_rot = torch_util.calc_heading_quat_inv(root_rot_t)
            local_root_rot = torch_util.quat_mul(heading_rot, root_rot_t)
            root_rot_obs = torch_util.quat_to_tan_norm(local_root_rot)
        obs_parts.append(root_rot_obs)

        # 1.3 Root velocity
        # root_vel_t is now in world-frame (converted from body-local above)
        if self.global_obs:
            root_vel_obs = root_vel_t  # [batch, 3] - world-frame, matching MimicKit's expectation
        else:
            heading_rot = torch_util.calc_heading_quat_inv(root_rot_t)
            root_vel_obs = torch_util.quat_rotate(heading_rot, root_vel_t)  # Convert to heading-local frame
        obs_parts.append(root_vel_obs)

        # 1.4 Root angular velocity
        # root_ang_vel_t is already in world-frame (from MujocoEnv's data.qvel[3:6])
        if self.global_obs:
            root_ang_vel_obs = root_ang_vel_t  # [batch, 3] - world-frame, matching MimicKit's expectation
        else:
            heading_rot = torch_util.calc_heading_quat_inv(root_rot_t)
            root_ang_vel_obs = torch_util.quat_rotate(heading_rot, root_ang_vel_t)  # Convert to heading-local frame
        obs_parts.append(root_ang_vel_obs)

        # 1.5 Joint rotations (6D tangent-normal)
        joint_rot_flat = torch.reshape(joint_rot_t, [joint_rot_t.shape[0] * joint_rot_t.shape[1], joint_rot_t.shape[2]])
        joint_rot_obs_flat = torch_util.quat_to_tan_norm(joint_rot_flat)
        joint_rot_obs = torch.reshape(
            joint_rot_obs_flat, [joint_rot_t.shape[0], joint_rot_t.shape[1] * joint_rot_obs_flat.shape[-1]]
        )  # [batch, 29*6=174]
        obs_parts.append(joint_rot_obs)

        # 1.6 Joint velocities
        obs_parts.append(dof_vel_t)  # [batch, 29]

        # 1.7 Key body positions (if enabled)
        if len(self.key_body_ids) > 0:
            root_pos_expand = root_pos_t.unsqueeze(-2)  # [batch, 1, 3]
            key_pos_rel = key_pos_t - root_pos_expand  # [batch, num_key_bodies, 3]
            if not self.global_obs:
                heading_rot = torch_util.calc_heading_quat_inv(root_rot_t)
                heading_rot_expand = heading_rot.unsqueeze(-2).repeat(1, key_pos_rel.shape[1], 1)
                heading_rot_flat = heading_rot_expand.reshape(-1, 4)
                key_pos_flat = key_pos_rel.reshape(-1, 3)
                key_pos_flat = torch_util.quat_rotate(heading_rot_flat, key_pos_flat)
                key_pos_rel = key_pos_flat.reshape(key_pos_rel.shape)
            key_pos_flat = torch.reshape(key_pos_rel, [key_pos_rel.shape[0], key_pos_rel.shape[1] * key_pos_rel.shape[2]])
            obs_parts.append(key_pos_flat)  # [batch, num_key_bodies*3]

        # 2. Phase observation (if enabled)
        if self.enable_phase_obs:
            phase_obs = motion_phase.unsqueeze(-1)  # [batch, 1]
            if self.num_phase_encoding > 0:
                angles = torch.arange(
                    1, self.num_phase_encoding + 1, device=phase_obs.device, dtype=phase_obs.dtype
                ) * 2 * torch.pi
                sin_phase = torch.sin(angles * phase_obs)
                cos_phase = torch.cos(angles * phase_obs)
                phase_obs = torch.cat([sin_phase, cos_phase], dim=-1)  # [batch, 2*num_phase_encoding]
            obs_parts.append(phase_obs)

        # 3. Target observations (if enabled)
        if self.enable_tar_obs and len(tar_root_pos) > 0:
            # Compute tar_obs for each step
            tar_obs_list = []
            num_tar_steps = tar_root_pos.shape[1]
            for step_idx in range(num_tar_steps):
                tar_root_pos_step = tar_root_pos[:, step_idx, :]  # [batch, 3]
                tar_root_rot_step = tar_root_rot[:, step_idx, :]  # [batch, 4]
                tar_joint_rot_step = tar_joint_rot[:, step_idx, :, :]  # [batch, num_joints, 4]

                if self.global_obs:
                    ref_root_pos = root_pos_t
                    ref_root_rot = root_rot_t
                else:
                    ref_root_pos = tar_root_pos[:, 0, :]
                    ref_root_rot = tar_root_rot[:, 0, :]

                # Root position relative to reference
                # ref_root_pos is [batch, 3], tar_root_pos_step is [batch, 3]
                tar_root_pos_obs = tar_root_pos_step - ref_root_pos  # [batch, 3]

                # Key positions relative to root
                if len(self.key_body_ids) > 0:
                    tar_key_pos_step = tar_key_pos[:, step_idx, :, :]  # [batch, num_key_bodies, 3]
                    tar_key_pos_rel = tar_key_pos_step - tar_root_pos_step.unsqueeze(-2)
                else:
                    tar_key_pos_rel = torch.zeros([0], device=self.torch_device)

                # Apply heading-local transform if not global
                if not self.global_obs:
                    heading_inv_rot = torch_util.calc_heading_quat_inv(ref_root_rot)
                    # tar_root_pos_obs is [batch, 3], need to rotate it
                    tar_root_pos_obs = torch_util.quat_rotate(heading_inv_rot, tar_root_pos_obs)  # [batch, 3]

                    tar_root_rot_step = torch_util.quat_mul(heading_inv_rot, tar_root_rot_step)

                    if len(self.key_body_ids) > 0:
                        heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2)  # [batch, 1, 4]
                        heading_inv_rot_expand = heading_inv_rot_expand.repeat(1, tar_key_pos_rel.shape[1], 1)  # [batch, num_key_bodies, 4]
                        heading_inv_rot_flat = heading_inv_rot_expand.reshape(-1, 4)
                        tar_key_pos_flat = tar_key_pos_rel.reshape(-1, 3)
                        tar_key_pos_flat = torch_util.quat_rotate(heading_inv_rot_flat, tar_key_pos_flat)
                        tar_key_pos_rel = tar_key_pos_flat.reshape(tar_key_pos_rel.shape)

                # Root height or xy
                # Note: if root_height_obs=True, z component is absolute height, not relative
                # Original code: root_pos_obs[..., 2] = root_pos[..., 2] if root_height_obs else root_pos_obs[..., :2]
                if self.root_height_obs:
                    # Keep all 3 dimensions, but replace z with absolute height
                    tar_root_pos_obs_final = tar_root_pos_obs.clone()  # [batch, 3]
                    tar_root_pos_obs_final[:, 2] = tar_root_pos_step[:, 2]  # Use absolute height
                else:
                    # Only keep xy
                    tar_root_pos_obs_final = tar_root_pos_obs[:, :2]  # [batch, 2]

                # Root rotation (6D)
                tar_root_rot_obs = torch_util.quat_to_tan_norm(tar_root_rot_step)  # [batch, 6]

                # Joint rotations (6D)
                tar_joint_rot_flat = tar_joint_rot_step.reshape(-1, 4)
                tar_joint_rot_obs_flat = torch_util.quat_to_tan_norm(tar_joint_rot_flat)
                tar_joint_rot_obs = tar_joint_rot_obs_flat.reshape(tar_joint_rot_step.shape[0], -1)  # [batch, 29*6]

                # Concatenate tar_obs for this step
                tar_obs_step_parts = [tar_root_pos_obs_final, tar_root_rot_obs, tar_joint_rot_obs]
                if len(self.key_body_ids) > 0:
                    tar_key_pos_flat = tar_key_pos_rel.reshape(tar_key_pos_rel.shape[0], -1)
                    tar_obs_step_parts.append(tar_key_pos_flat)
                tar_obs_step = torch.cat(tar_obs_step_parts, dim=-1)
                tar_obs_list.append(tar_obs_step)

            # Concatenate all tar steps
            tar_obs = torch.cat(tar_obs_list, dim=-1)  # [batch, num_steps * tar_obs_dim]
            obs_parts.append(tar_obs)

        # Concatenate all parts
        obs_t = torch.cat(obs_parts, dim=-1)  # [batch, obs_dim]
        obs = obs_t.squeeze(0).cpu().numpy()

        # Verify observation dimension matches expected
        expected_obs_dim = self.obs_mean.shape[0]
        if obs.shape[0] != expected_obs_dim:
            raise ValueError(
                f"Observation dimension mismatch: got {obs.shape[0]}, expected {expected_obs_dim}. "
                f"Parts: {[p.shape[-1] for p in obs_parts]}"
            )

        obs_norm = (obs - self.obs_mean) / np.sqrt(self.obs_var + 1e-8)
        if self.obs_clip is not None:
            obs_norm = np.clip(obs_norm, -self.obs_clip, self.obs_clip)

        # cache debug tensors for action dump
        if self._debug_dump_once and not self._debug_dump_done:
            self._debug_last = {
                "obs_raw": obs.astype(np.float32),
                "obs_norm": obs_norm.astype(np.float32),
                "obs_raw_first_10": obs[:10].astype(np.float32),
                "obs_norm_first_10": obs_norm[:10].astype(np.float32),
                "env_dof_pos": np.asarray(dof_pos, dtype=np.float32),
                "env_dof_vel": np.asarray(dof_vel, dtype=np.float32),
                "root_pos": np.asarray(root_pos, dtype=np.float32),
                "root_rot": np.asarray(root_rot, dtype=np.float32),
                "root_vel": np.asarray(root_vel, dtype=np.float32),
                "root_ang_vel": np.asarray(root_ang_vel, dtype=np.float32),
                "joint_rot_t_shape": np.asarray(joint_rot_t.shape, dtype=np.int32),
                "meta_global_obs": np.asarray([int(self.global_obs)], dtype=np.int32),
                "meta_root_height_obs": np.asarray([int(self.root_height_obs)], dtype=np.int32),
                "meta_enable_tar_obs": np.asarray([int(self.enable_tar_obs)], dtype=np.int32),
                "meta_enable_phase_obs": np.asarray([int(self.enable_phase_obs)], dtype=np.int32),
                "meta_zero_center_action": np.asarray([int(self.zero_center_action)], dtype=np.int32),
                "meta_motion_dt": np.asarray([float(self.motion_dt)], dtype=np.float32),
                "meta_dt": np.asarray([float(self.dt)], dtype=np.float32),
                "meta_tar_obs_steps": np.asarray(self.tar_obs_steps, dtype=np.int32),
                "meta_key_body_num": np.asarray([int(len(self.key_body_ids))], dtype=np.int32),
                "meta_joint_rot_num": np.asarray([int(joint_rot_t.shape[1])], dtype=np.int32),
                "meta_obs_dim": np.asarray([int(obs.shape[0])], dtype=np.int32),
            }

        extras = {"CALLBACK": ["[MOTION_DONE]"] if self.flag_motion_done else []}
        return obs_norm.astype(np.float32), extras

    def _mlp_forward(self, obs_norm: np.ndarray) -> np.ndarray:
        x = obs_norm.astype(np.float32)
        for idx, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ w.T + b
            if idx < len(self.weights) - 1:
                x = np.maximum(x, 0.0)
        return x.astype(np.float32)

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        action_net = self._mlp_forward(obs)
        action_smoothed = (1 - self.action_beta) * self.last_action + self.action_beta * action_net
        self.last_action = action_smoothed.copy()

        action_denorm = action_smoothed.copy()
        if self.action_mean.size == action_denorm.size:
            action_denorm = action_denorm * self.action_std + self.action_mean

        # Clip action to action bounds (matching MimicKit's _apply_action clip logic)
        # MimicKit clips the action before applying it: clip_action = clip(actions, action_bound_low, action_bound_high)
        action_denorm = np.clip(action_denorm, self.action_bound_low, self.action_bound_high)

        if self.zero_center_action:
            target_joint_pos = action_denorm + self.default_pose_for_action
            action_delta = target_joint_pos - self.default_pos
        else:
            action_delta = action_denorm - self.default_pos

        # dump once (after action computed)
        if self._debug_dump_once and not self._debug_dump_done:
            dump = dict(self._debug_last)
            dump.update(
                {
                    "action_net": action_net.astype(np.float32),
                    "action_smoothed": action_smoothed.astype(np.float32),
                    "action_denorm": action_denorm.astype(np.float32),
                    "action_delta": action_delta.astype(np.float32),
                    "default_pos": np.asarray(self.default_pos, dtype=np.float32),
                    "init_dof_pos": np.asarray(self.init_dof_pos, dtype=np.float32),
                    "default_pose_for_action": np.asarray(self.default_pose_for_action, dtype=np.float32),
                    "target_joint_pos": (action_denorm + self.default_pose_for_action).astype(np.float32) if self.zero_center_action else np.zeros_like(action_denorm),
                }
            )
            out_path = self._debug_dump_path
            if not out_path:
                out_path = (Path.cwd() / "mimickit_dump.npz").as_posix()
            out_path_p = Path(out_path).expanduser()
            out_path_p.parent.mkdir(parents=True, exist_ok=True)
            np.savez(out_path_p.as_posix(), **dump)
            self._debug_dump_done = True

        return action_delta.astype(np.float32)

    def get_init_dof_pos(self) -> np.ndarray:
        return self.init_dof_pos.astype(np.float32)
