"""Helpers for mimic playback progress display in loco-mimic pipelines."""

from __future__ import annotations

import numpy as np

from robojudo.pipeline.rl_pipeline import PolicyWrapper


def mimic_label(wrapper: PolicyWrapper) -> str:
    policy = wrapper.policy
    cfg = getattr(policy, "cfg_policy", None)
    if cfg is not None and hasattr(cfg, "policy_name"):
        name = cfg.policy_name
        if name:
            return str(name)
    if "@" in wrapper.name:
        return wrapper.name.split("@", 1)[1]
    return wrapper.name


def mimic_menu_label(wrapper: PolicyWrapper) -> str:
    """Human-readable name for loco-mimic menu (matches dance-style short names)."""
    policy = wrapper.policy
    cls = policy.__class__.__name__
    cfg = policy.cfg_policy
    match cls:
        case "LocoModePolicy":
            return "LocoMode"
        case "MotionTrackingPolicy":
            return str(cfg.policy_name)
        case _:
            return mimic_label(wrapper)


def mimic_progress_spec(wrapper: PolicyWrapper) -> tuple[str, float] | None:
    """Return (label, total). total <= 0 means continuous playback indicator."""
    policy = wrapper.policy
    label = mimic_label(wrapper)
    cls = policy.__class__.__name__

    match cls:
        case "BeyondMimicPolicy":
            if not getattr(policy, "use_motion_from_model", True):
                return None
            max_timestep = getattr(policy, "max_timestep", -1)
            return (label, float(max_timestep) if max_timestep > 0 else 0.0)
        case "LocoModePolicy":
            return (label, 0.0)
        case "MotionTrackingPolicy":
            return (label, float(policy.total_frames))
        case "MimicKitPolicy":
            if policy.motion_loop_mode == 0:
                steps = int(np.ceil(policy.motion_length / policy.motion_dt))
                return (label, float(max(steps, 1)))
            return (label, 0.0)
        case "AsapPolicy":
            return (label, float(policy.motion_length_s))
        case _:
            if hasattr(policy, "max_timestep") and policy.max_timestep > 0:
                return (label, float(policy.max_timestep))
            if hasattr(policy, "total_frames"):
                return (label, float(policy.total_frames))
            if hasattr(policy, "timestep"):
                return (label, 0.0)
            return None


def mimic_progress_value(policy) -> float:
    cls = policy.__class__.__name__
    match cls:
        case "MimicKitPolicy":
            if policy.motion_dt > 0:
                return policy.motion_time / policy.motion_dt
            return float(policy.timestep)
        case "AsapPolicy":
            return policy.timestep * policy.dt
        case _:
            return float(getattr(policy, "timestep", 0))
