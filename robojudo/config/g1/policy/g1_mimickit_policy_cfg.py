from robojudo.config import ASSETS_DIR
from robojudo.config.g1.env.g1_env_cfg import G1_29DoF
from robojudo.policy.policy_cfgs import MimicKitPolicyCfg
from robojudo.tools.tool_cfgs import DoFConfig


class G1MimicKitPolicyCfg(MimicKitPolicyCfg):
    robot: str = "g1"

    obs_dof: DoFConfig = G1_29DoF()
    action_dof: DoFConfig = obs_dof

    policy_name: str = "g1_mimickit"
    motion_name: str = "hkf_mimic"

    env_config_path: str = (ASSETS_DIR / "robots/g1/mimickit/configs/env_config.yaml").as_posix()
    agent_config_path: str = (ASSETS_DIR / "robots/g1/mimickit/configs/agent_config.yaml").as_posix()
    char_file: str = (ASSETS_DIR / "robots/g1/mimickit/g1.xml").as_posix()
    motion_file: str = (ASSETS_DIR / "motions/g1/mimickit/hkf_mimic.pkl").as_posix()
    model_pt_file: str = (ASSETS_DIR / "models/g1/mimickit/qkf_model.pt").as_posix()

    action_beta: float = 1.0
