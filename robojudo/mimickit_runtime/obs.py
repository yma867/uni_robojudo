import torch

from . import torch_util


@torch.jit.script
def convert_to_local_body_pos(root_rot, body_pos):
    heading_inv_rot = torch_util.calc_heading_quat_inv(root_rot)
    heading_rot_expand = heading_inv_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, body_pos.shape[1], 1))
    flat_heading_rot_expand = heading_rot_expand.reshape(
        heading_rot_expand.shape[0] * heading_rot_expand.shape[1], heading_rot_expand.shape[2]
    )
    flat_body_pos = body_pos.reshape(body_pos.shape[0] * body_pos.shape[1], body_pos.shape[2])
    flat_local_body_pos = torch_util.quat_rotate(flat_heading_rot_expand, flat_body_pos)
    local_body_pos = flat_local_body_pos.reshape(body_pos.shape[0], body_pos.shape[1], body_pos.shape[2])

    return local_body_pos


@torch.jit.script
def convert_to_local_root_body_pos(root_rot, body_pos):
    root_inv_rot = torch_util.quat_conjugate(root_rot)
    root_rot_expand = root_inv_rot.unsqueeze(-2)
    root_rot_expand = root_rot_expand.repeat((1, body_pos.shape[1], 1))
    flat_root_rot_expand = root_rot_expand.reshape(
        root_rot_expand.shape[0] * root_rot_expand.shape[1], root_rot_expand.shape[2]
    )
    flat_body_pos = body_pos.reshape(body_pos.shape[0] * body_pos.shape[1], body_pos.shape[2])
    flat_local_body_pos = torch_util.quat_rotate(flat_root_rot_expand, flat_body_pos)
    local_body_pos = flat_local_body_pos.reshape(body_pos.shape[0], body_pos.shape[1], body_pos.shape[2])

    return local_body_pos


@torch.jit.script
def compute_char_obs(root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, key_pos, global_obs, root_height_obs):
    # type: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool, bool) -> torch.Tensor
    heading_rot = torch_util.calc_heading_quat_inv(root_rot)

    if global_obs:
        root_rot_obs = torch_util.quat_to_tan_norm(root_rot)
        root_vel_obs = root_vel
        root_ang_vel_obs = root_ang_vel
    else:
        local_root_rot = torch_util.quat_mul(heading_rot, root_rot)
        root_rot_obs = torch_util.quat_to_tan_norm(local_root_rot)
        root_vel_obs = torch_util.quat_rotate(heading_rot, root_vel)
        root_ang_vel_obs = torch_util.quat_rotate(heading_rot, root_ang_vel)

    joint_rot_flat = torch.reshape(joint_rot, [joint_rot.shape[0] * joint_rot.shape[1], joint_rot.shape[2]])
    joint_rot_obs_flat = torch_util.quat_to_tan_norm(joint_rot_flat)
    joint_rot_obs = torch.reshape(
        joint_rot_obs_flat, [joint_rot.shape[0], joint_rot.shape[1] * joint_rot_obs_flat.shape[-1]]
    )

    obs = [root_rot_obs, root_vel_obs, root_ang_vel_obs, joint_rot_obs, dof_vel]

    if len(key_pos) > 0:
        root_pos_expand = root_pos.unsqueeze(-2)
        key_pos = key_pos - root_pos_expand
        if not global_obs:
            key_pos = convert_to_local_body_pos(root_rot, key_pos)

        key_pos_flat = torch.reshape(key_pos, [key_pos.shape[0], key_pos.shape[1] * key_pos.shape[2]])
        obs = obs + [key_pos_flat]

    if root_height_obs:
        root_h = root_pos[:, 2:3]
        obs = [root_h] + obs

    obs = torch.cat(obs, dim=-1)
    return obs


@torch.jit.script
def compute_tar_obs(ref_root_pos, ref_root_rot, root_pos, root_rot, joint_rot, key_pos, global_obs, root_height_obs):
    # type: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool, bool) -> torch.Tensor
    ref_root_pos = ref_root_pos.unsqueeze(-2)
    root_pos_obs = root_pos - ref_root_pos

    if len(key_pos) > 0:
        key_pos = key_pos - root_pos.unsqueeze(-2)

    if not global_obs:
        heading_inv_rot = torch_util.calc_heading_quat_inv(ref_root_rot)
        heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2)
        heading_inv_rot_expand = heading_inv_rot_expand.repeat((1, root_pos.shape[1], 1))
        heading_inv_rot_flat = heading_inv_rot_expand.reshape(
            (heading_inv_rot_expand.shape[0] * heading_inv_rot_expand.shape[1], heading_inv_rot_expand.shape[2])
        )
        root_pos_obs_flat = torch.reshape(root_pos_obs, [root_pos_obs.shape[0] * root_pos_obs.shape[1], root_pos_obs.shape[2]])
        root_pos_obs_flat = torch_util.quat_rotate(heading_inv_rot_flat, root_pos_obs_flat)
        root_pos_obs = torch.reshape(root_pos_obs_flat, root_pos.shape)

        root_rot = torch_util.quat_mul(heading_inv_rot_expand, root_rot)

        if len(key_pos) > 0:
            heading_inv_rot_expand = heading_inv_rot_expand.unsqueeze(-2)
            heading_inv_rot_expand = heading_inv_rot_expand.repeat((1, 1, key_pos.shape[2], 1))
            heading_inv_rot_flat = heading_inv_rot_expand.reshape(
                (
                    heading_inv_rot_expand.shape[0]
                    * heading_inv_rot_expand.shape[1]
                    * heading_inv_rot_expand.shape[2],
                    heading_inv_rot_expand.shape[3],
                )
            )
            key_pos_flat = key_pos.reshape((key_pos.shape[0] * key_pos.shape[1] * key_pos.shape[2], key_pos.shape[3]))
            key_pos_flat = torch_util.quat_rotate(heading_inv_rot_flat, key_pos_flat)
            key_pos = key_pos_flat.reshape(key_pos.shape)

    if root_height_obs:
        root_pos_obs[..., 2] = root_pos[..., 2]
    else:
        root_pos_obs = root_pos_obs[..., :2]

    root_rot_flat = torch.reshape(root_rot, [root_rot.shape[0] * root_rot.shape[1], root_rot.shape[2]])
    root_rot_obs_flat = torch_util.quat_to_tan_norm(root_rot_flat)
    root_rot_obs = torch.reshape(
        root_rot_obs_flat, [root_rot.shape[0], root_rot.shape[1], root_rot_obs_flat.shape[-1]]
    )

    joint_rot_flat = torch.reshape(joint_rot, [joint_rot.shape[0] * joint_rot.shape[1] * joint_rot.shape[2], joint_rot.shape[3]])
    joint_rot_obs_flat = torch_util.quat_to_tan_norm(joint_rot_flat)
    joint_rot_obs = torch.reshape(
        joint_rot_obs_flat, [joint_rot.shape[0], joint_rot.shape[1], joint_rot.shape[2] * joint_rot_obs_flat.shape[-1]]
    )

    obs = [root_pos_obs, root_rot_obs, joint_rot_obs]
    if len(key_pos) > 0:
        key_pos = torch.reshape(key_pos, [key_pos.shape[0], key_pos.shape[1], key_pos.shape[2] * key_pos.shape[3]])
        obs.append(key_pos)

    obs = torch.cat(obs, dim=-1)

    return obs


@torch.jit.script
def compute_phase_obs(phase, num_phase_encoding):
    # type: (torch.Tensor, int) -> torch.Tensor
    phase = phase.unsqueeze(-1)
    if num_phase_encoding > 0:
        angles = torch.arange(1, num_phase_encoding + 1, device=phase.device, dtype=phase.dtype) * 2 * torch.pi
        sin_phase = torch.sin(angles * phase)
        cos_phase = torch.cos(angles * phase)
        phase_obs = torch.cat([sin_phase, cos_phase], dim=-1)
    else:
        phase_obs = phase
    return phase_obs


@torch.jit.script
def compute_deepmimic_obs(
    root_pos,
    root_rot,
    root_vel,
    root_ang_vel,
    joint_rot,
    dof_vel,
    key_pos,
    global_obs,
    root_height_obs,
    phase,
    num_phase_encoding,
    enable_phase_obs,
    enable_tar_obs,
    tar_root_pos,
    tar_root_rot,
    tar_joint_rot,
    tar_key_pos,
):
    # type: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool, bool, torch.Tensor, int, bool, bool, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
    char_obs = compute_char_obs(
        root_pos=root_pos,
        root_rot=root_rot,
        root_vel=root_vel,
        root_ang_vel=root_ang_vel,
        joint_rot=joint_rot,
        dof_vel=dof_vel,
        key_pos=key_pos,
        global_obs=global_obs,
        root_height_obs=root_height_obs,
    )
    obs = [char_obs]

    if enable_phase_obs:
        phase_obs = compute_phase_obs(phase=phase, num_phase_encoding=num_phase_encoding)
        obs.append(phase_obs)

    if enable_tar_obs:
        if global_obs:
            ref_root_pos = root_pos
            ref_root_rot = root_rot
        else:
            ref_root_pos = tar_root_pos[..., 0, :]
            ref_root_rot = tar_root_rot[..., 0, :]

        tar_obs = compute_tar_obs(
            ref_root_pos=ref_root_pos,
            ref_root_rot=ref_root_rot,
            root_pos=tar_root_pos,
            root_rot=tar_root_rot,
            joint_rot=tar_joint_rot,
            key_pos=tar_key_pos,
            global_obs=global_obs,
            root_height_obs=root_height_obs,
        )

        tar_obs = torch.reshape(tar_obs, [tar_obs.shape[0], tar_obs.shape[1] * tar_obs.shape[2]])
        obs.append(tar_obs)

    obs = torch.cat(obs, dim=-1)
    return obs
