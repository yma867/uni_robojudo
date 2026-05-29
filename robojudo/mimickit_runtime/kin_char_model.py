import abc
import enum
import numpy as np
import torch

from . import torch_util


class JointType(enum.Enum):
    ROOT = 0
    HINGE = 1
    SPHERICAL = 2
    FIXED = 3


class Joint:
    def __init__(self, name, joint_type, axis):
        self.name = name
        self.joint_type = joint_type
        self.axis = axis
        self.dof_idx = -1

    def get_dof_dim(self):
        if self.joint_type == JointType.ROOT:
            dof_dim = 0
        elif self.joint_type == JointType.HINGE:
            dof_dim = 1
        elif self.joint_type == JointType.SPHERICAL:
            dof_dim = 3
        elif self.joint_type == JointType.FIXED:
            dof_dim = 0
        else:
            raise ValueError(f"Unsupported joint type: {self.joint_type}")
        return dof_dim

    def get_joint_dof(self, dof):
        dof_idx = self.dof_idx
        dof_dim = self.get_dof_dim()
        j_dof = dof[..., dof_idx : dof_idx + dof_dim]
        return j_dof

    def set_joint_dof(self, j_dof, out_dof):
        dof_idx = self.dof_idx
        dof_dim = self.get_dof_dim()
        out_dof[..., dof_idx : dof_idx + dof_dim] = j_dof

    def dof_to_rot(self, dof):
        rot_shape = list(dof.shape[:-1]) + [4]
        rot = torch.zeros(rot_shape, device=dof.device, dtype=dof.dtype)

        if self.joint_type == JointType.ROOT:
            rot[..., -1] = 1
        elif self.joint_type == JointType.HINGE:
            axis = self.axis
            axis_shape = rot[..., 0:3].shape
            axis = torch.broadcast_to(axis, axis_shape)
            dof = dof.squeeze(-1)
            rot[:] = torch_util.axis_angle_to_quat(axis, dof)
        elif self.joint_type == JointType.SPHERICAL:
            rot[:] = torch_util.exp_map_to_quat(dof)
        elif self.joint_type == JointType.FIXED:
            rot[..., -1] = 1
        else:
            raise ValueError(f"Unsupported joint type: {self.joint_type}")

        return rot

    def rot_to_dof(self, rot):
        if rot.shape[-1] == 3:
            rot = torch_util.exp_map_to_quat(rot)

        dof_dim = self.get_dof_dim()
        dof_shape = list(rot.shape[:-1]) + [dof_dim]
        dof = torch.zeros(dof_shape, device=rot.device, dtype=rot.dtype)

        if self.joint_type == JointType.ROOT:
            pass
        elif self.joint_type == JointType.HINGE:
            j_axis = self.axis
            angle = torch_util.quat_twist_angle(rot, j_axis)
            dof[:] = angle.unsqueeze(-1)
        elif self.joint_type == JointType.SPHERICAL:
            dof[:] = torch_util.quat_to_exp_map(rot)
        elif self.joint_type == JointType.FIXED:
            pass
        else:
            raise ValueError(f"Unsupported joint type: {self.joint_type}")

        return dof


class KinCharModel:
    def __init__(self, device):
        self._device = device

    def init(self, body_names, parent_indices, local_translation, local_rotation, joints):
        num_bodies = len(body_names)
        if not (len(parent_indices) == num_bodies and len(local_translation) == num_bodies):
            raise ValueError("Body arrays size mismatch.")
        if not (len(local_rotation) == num_bodies and len(joints) == num_bodies):
            raise ValueError("Body arrays size mismatch.")

        self._body_names = body_names
        self._parent_indices = torch.tensor(parent_indices, device=self._device, dtype=torch.long)
        self._local_translation = torch.tensor(np.array(local_translation), device=self._device, dtype=torch.float32)
        self._local_rotation = torch.tensor(np.array(local_rotation), device=self._device, dtype=torch.float32)
        self._joints = joints

        self._dof_size = self._label_dof_indices(self._joints)
        self._name_body_map = self._build_name_body_map()

    @abc.abstractmethod
    def load(self, char_file):
        raise NotImplementedError

    @abc.abstractmethod
    def save(self, output_file):
        raise NotImplementedError

    def get_body_names(self):
        return self._body_names

    def get_joint(self, j):
        if j <= 0:
            raise ValueError("Joint index must be > 0")
        return self._joints[j]

    def get_parent_id(self, j):
        return self._parent_indices[j]

    def get_dof_size(self):
        return self._dof_size

    def get_joint_dof_idx(self, j):
        return self.get_joint(j).dof_idx

    def get_joint_dof_dim(self, j):
        return self.get_joint(j).get_dof_dim()

    def get_num_joints(self):
        return len(self._joints)

    def dof_to_rot(self, dof):
        num_joints = self.get_num_joints()
        rot_shape = list(dof.shape[:-1]) + [num_joints - 1, 4]
        joint_rot = torch.zeros(rot_shape, device=dof.device, dtype=dof.dtype)

        for j in range(1, num_joints):
            joint = self.get_joint(j)
            j_dof = joint.get_joint_dof(dof)
            j_rot = joint.dof_to_rot(j_dof)
            joint_rot[..., j - 1, :] = j_rot

        return joint_rot

    def rot_to_dof(self, rot):
        dof_shape = list(rot.shape[:-2]) + [self._dof_size]
        dof = torch.zeros(dof_shape, device=rot.device, dtype=rot.dtype)

        num_joints = self.get_num_joints()
        for j in range(1, num_joints):
            joint = self.get_joint(j)
            j_dof_dim = joint.get_dof_dim()
            if j_dof_dim > 0:
                j_rot = rot[..., j - 1, :]
                j_dof = joint.rot_to_dof(j_rot)
                joint.set_joint_dof(j_dof, dof)

        return dof

    def forward_kinematics(self, root_pos, root_rot, joint_rot):
        num_joints = self.get_num_joints()
        body_pos = [None] * num_joints
        body_rot = [None] * num_joints

        body_pos[0] = root_pos
        body_rot[0] = root_rot

        for j in range(1, num_joints):
            j_rot = joint_rot[..., j - 1, :]
            local_trans = self._local_translation[j]
            local_rot = self._local_rotation[j]
            parent_idx = self._parent_indices[j]

            parent_pos = body_pos[parent_idx]
            parent_rot = body_rot[parent_idx]

            local_trans = torch.broadcast_to(local_trans, parent_pos.shape)
            local_rot = torch.broadcast_to(local_rot, parent_rot.shape)

            world_trans = torch_util.quat_rotate(parent_rot, local_trans)

            curr_pos = parent_pos + world_trans
            curr_rot = torch_util.quat_mul(local_rot, j_rot)

            body_pos[j] = curr_pos
            body_rot[j] = curr_rot

        body_pos = torch.stack(body_pos, dim=-2)
        body_rot = torch.stack(body_rot, dim=-2)
        return body_pos, body_rot

    def _build_name_body_map(self):
        name_body_map = {}
        for i, name in enumerate(self._body_names):
            name_body_map[name] = i
        return name_body_map

    def _label_dof_indices(self, joints):
        dof_idx = 0
        for curr_joint in joints:
            if curr_joint is not None:
                dof_dim = curr_joint.get_dof_dim()
                curr_joint.dof_idx = dof_idx
                dof_idx += dof_dim
        return dof_idx
