import numpy as np
import torch
import xml.etree.ElementTree as ET

from . import kin_char_model
from . import torch_util


class MJCFCharModel(kin_char_model.KinCharModel):
    def __init__(self, device):
        super().__init__(device)

    def load(self, char_file):
        tree = ET.parse(char_file)
        xml_doc_root = tree.getroot()
        xml_world_body = xml_doc_root.find("worldbody")
        if xml_world_body is None:
            raise ValueError("Missing worldbody in MJCF.")

        xml_body_root = xml_world_body.find("body")
        if xml_body_root is None:
            raise ValueError("Missing root body in MJCF.")

        body_names = []
        parent_indices = []
        local_translation = []
        local_rotation = []
        joints = []

        default_joint_type = self._parse_default_joint_type(xml_doc_root)

        def _add_xml_body(xml_node, parent_index, body_index, default_joint_type):
            body_name = xml_node.attrib.get("name")
            pos_data = xml_node.attrib.get("pos")
            if pos_data is None:
                pos = np.array([0.0, 0.0, 0.0])
            else:
                pos = np.fromstring(pos_data, dtype=float, sep=" ")

            rot_data = xml_node.attrib.get("quat")
            if rot_data is None:
                rot = np.array([0.0, 0.0, 0.0, 1.0])
            else:
                rot = np.fromstring(rot_data, dtype=float, sep=" ")
                rot_w = rot[..., 0].copy()
                rot[..., 0:3] = rot[..., 1:]
                rot[..., -1] = rot_w

            if body_index == 0:
                curr_joint = self._build_root_joint()
            else:
                joint_data = xml_node.findall("joint")
                curr_joint = self._parse_joint(body_name, joint_data, default_joint_type)

            body_names.append(body_name)
            parent_indices.append(parent_index)
            local_translation.append(pos)
            local_rotation.append(rot)
            joints.append(curr_joint)

            curr_index = body_index
            body_index += 1
            for child_body in xml_node.findall("body"):
                body_index = _add_xml_body(child_body, curr_index, body_index, default_joint_type)

            return body_index

        _add_xml_body(xml_body_root, -1, 0, default_joint_type)

        self.init(
            body_names=body_names,
            parent_indices=parent_indices,
            local_translation=local_translation,
            local_rotation=local_rotation,
            joints=joints,
        )

    def save(self, output_file):
        raise NotImplementedError("Save not required for runtime.")

    def _parse_joint(self, body_name, xml_joint_data, default_joint_type):
        num_joints = len(xml_joint_data)

        if num_joints == 0:
            joint = self._parse_fixed_joint(body_name)
        elif num_joints == 3:
            joint = self._parse_sphere_joint(xml_joint_data, default_joint_type)
        elif num_joints == 1:
            joint_type_str = xml_joint_data[0].attrib.get("type")
            if joint_type_str is None:
                joint_type_str = default_joint_type

            if joint_type_str == "hinge":
                joint = self._parse_hinge_joint(xml_joint_data[0])
            else:
                raise ValueError(f"Unsupported joint type: {joint_type_str}")
        else:
            raise ValueError("Series joints are not supported.")

        return joint

    def _parse_hinge_joint(self, xml_joint_data):
        joint_name = xml_joint_data.attrib.get("name")

        joint_pos_data = xml_joint_data.attrib.get("pos")
        if joint_pos_data is not None:
            joint_pos = np.fromstring(joint_pos_data, dtype=float, sep=" ")
            if np.any(joint_pos):
                raise ValueError("Joint offsets are not supported")

        joint_axis = np.fromstring(xml_joint_data.attrib.get("axis"), dtype=float, sep=" ")
        joint_axis = torch.tensor(joint_axis, device=self._device, dtype=torch.float32)

        joint = kin_char_model.Joint(
            name=joint_name,
            joint_type=kin_char_model.JointType.HINGE,
            axis=joint_axis,
        )
        return joint

    def _parse_sphere_joint(self, xml_joint_data, default_joint_type):
        num_joints = len(xml_joint_data)
        if num_joints != 3:
            raise ValueError("Sphere joint should have three hinges.")

        is_spherical = True
        for joint_data in xml_joint_data:
            joint_type_str = joint_data.attrib.get("type")
            if joint_type_str is None:
                joint_type_str = default_joint_type

            joint_pos_data = joint_data.attrib.get("pos")
            if joint_pos_data is not None:
                joint_pos = np.fromstring(joint_pos_data, dtype=float, sep=" ")
                if np.any(joint_pos):
                    raise ValueError("Joint offsets are not supported")

            if joint_type_str != "hinge":
                is_spherical = False
                break

        if is_spherical:
            joint_name = xml_joint_data[0].attrib.get("name")
            joint_name = joint_name[: joint_name.rfind("_")]
            joint = kin_char_model.Joint(
                name=joint_name,
                joint_type=kin_char_model.JointType.SPHERICAL,
                axis=None,
            )
        else:
            raise ValueError("Invalid format for a spherical joint")

        return joint

    def _parse_fixed_joint(self, body_name):
        joint = kin_char_model.Joint(
            name=body_name,
            joint_type=kin_char_model.JointType.FIXED,
            axis=None,
        )

        return joint

    def _build_root_joint(self):
        axis = torch.tensor([0.0, 0.0, 1.0], device=self._device, dtype=torch.float32)
        joint = kin_char_model.Joint("root", kin_char_model.JointType.ROOT, axis)
        return joint

    def _parse_default_joint_type(self, xml_node):
        joint_type_str = "hinge"

        default_data = xml_node.find("default")
        if default_data is not None:
            default_joint = default_data.find("joint")
            if default_joint is not None:
                joint_type_str = default_joint.attrib.get("type", joint_type_str)

        return joint_type_str
