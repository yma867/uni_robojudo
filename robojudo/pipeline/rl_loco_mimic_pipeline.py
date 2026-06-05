import logging
from collections.abc import Callable
from enum import Enum, auto

import numpy as np
from tqdm import tqdm

import robojudo.environment
from robojudo.controller import CtrlManager
from robojudo.environment import Environment
from robojudo.pipeline import Pipeline, pipeline_registry
from robojudo.pipeline.pipeline_cfgs import RlLocoMimicPipelineCfg
from robojudo.pipeline.rl_multi_policy_pipeline import PolicyManager, RlMultiPolicyPipeline
from robojudo.pipeline.rl_pipeline import PolicyWrapper
from robojudo.policy import PolicyCfg
from robojudo.utils.mimic_progress import mimic_label, mimic_menu_label, mimic_progress_spec, mimic_progress_value
from robojudo.utils.progress import ProgressBar
from robojudo.utils.terminal_style import dim, key, mark, strip_ansi, white

logger = logging.getLogger(__name__)


class PolicyInterpManager(PolicyManager):
    class InterpState(Enum):
        IDLE = auto()
        START = auto()
        IN_PROGRESS = auto()
        END = auto()

    DURATIONS_LOCO_MIMIC = [0, 75, 25]  # [start, in-progress, end] in steps
    DURATIONS_MIMIC_LOCO = [25, 75, 0]  # [start, in-progress, end] in steps

    def __init__(
        self,
        cfg_policy_loco: PolicyCfg,
        cfg_policies: list[PolicyCfg],
        env: Environment,
        loco_dof_pos: np.ndarray | None = None,
        device: str = "cpu",
    ):
        cfg_policies_all = [cfg_policy_loco] + cfg_policies
        super().__init__(cfg_policies_all, env, device)

        self.policy_loco_id = 0
        self.policy_mimic_num = len(cfg_policies)
        assert self.policy_mimic_num > 0, "At least one mimic policy is required for switching."
        self.policy_mimic_ids = list(range(1, self.policy_mimic_num + 1))
        self.policy_mimic_idx = 0

        # Interpolation variables
        self.interp_state = self.InterpState.IDLE
        self.interp_timestep = 0
        self.interp_durations = [20, 40, 20]  # [start, in-progress, end] in steps
        self.interp_pbar = None
        self.interp_callback_start = None
        self.interp_callback_end = None

        self.loco_dof_pos = loco_dof_pos if loco_dof_pos is not None else self.env.default_pos.copy()
        self.override_dof_pos = self.loco_dof_pos.copy()
        self._mimic_pbar: ProgressBar | None = None

    @property
    def is_sim(self) -> bool:
        return self.env.cfg_env.is_sim

    def _deploy_badge(self) -> str:
        return white("[仿真 Sim]") if self.is_sim else white("[真机 Real]")

    def _return_key_hint(self) -> str:
        return key("]") if self.is_sim else key("Select")

    def _amo_locomotion_lines(self) -> list[str]:
        if self.is_sim:
            return [
                "AMO 行走（Sim：需手柄，键盘不能走）：",
                f"  左摇杆 {key('↑↓')} = 前进/后退    左摇杆 {key('←→')} = 左右平移",
                f"  右摇杆 {key('←→')} = 转向          {key('Y')} = 切换步态/模式",
            ]
        return [
            "AMO 行走（Unitree 遥控器）：",
            f"  左摇杆 {key('↑↓')} = 前进/后退    左摇杆 {key('←→')} = 左右平移",
            f"  右摇杆 {key('←→')} = 转向          {key('Y')} = 切换步态/模式",
        ]

    def _mode_switch_lines(self) -> list[str]:
        if self.is_sim:
            return [
                "模式切换（Sim 键盘）：",
                "  "
                + key(";")
                + " / "
                + key("'")
                + " = 上一个/下一个动作    "
                + key("[")
                + " = 开始当前动作",
                f"  {key(']')} = 打断 Mimic / 回到 AMO / 显示菜单",
                dim(f"  {key('i')} = 仅重置仿真姿态（不回 AMO）    退出：终端 Ctrl+C"),
            ]
        return [
            "模式切换（Unitree 遥控器）：",
            f"  {key('R1')} / {key('L1')} = 上一个/下一个动作    {key('Start')} = 开始当前动作",
            f"  {key('Select')} = 打断 Mimic / 回到 AMO    {key('A')} = 急停",
        ]

    def _mimic_interrupt_lines(self) -> list[str]:
        if self.is_sim:
            return [
                "打断并回到 AMO：",
                f"  Sim 键盘：{key(']')}（与 {key('[')} 同一排）",
                "",
                dim("以下键不会回到 AMO："),
                f"  {key('i')} = 只重置仿真里机器人位置，动作继续播放",
            ]
        return [
            "打断并回到 AMO：",
            f"  Unitree 遥控器：{key('Select')}",
            f"  急停：{key('A')}",
        ]

    def _close_mimic_progress(self) -> None:
        if self._mimic_pbar is not None:
            self._mimic_pbar.close()
            self._mimic_pbar = None

    def _open_mimic_progress(self, policy_id: int) -> None:
        self._close_mimic_progress()
        if policy_id == self.policy_loco_id:
            return
        wrapper = self.policy_by_id(policy_id)
        spec = mimic_progress_spec(wrapper)
        if spec is None:
            return
        label, total = spec
        display = mimic_menu_label(wrapper)
        self._mimic_pbar = ProgressBar(f"Mimic {display}", total)
        self._mimic_pbar.set(mimic_progress_value(wrapper.policy))

    def _update_mimic_progress(self) -> None:
        if self._mimic_pbar is None or self.current_policy_id == self.policy_loco_id:
            return
        wrapper = self.policy_by_id(self.current_policy_id)
        self._mimic_pbar.set(mimic_progress_value(wrapper.policy))

    def set_policy(self, policy_id: int):
        if self.current_policy_id != policy_id:
            self._close_mimic_progress()
        super().set_policy(policy_id)
        if policy_id != self.policy_loco_id:
            self._open_mimic_progress(policy_id)
            self.log_mimic_panel(policy_id)

    def log_mimic_panel(self, policy_id: int) -> None:
        """Print mimic playback panel with interrupt hints."""
        wrapper = self.policy_by_id(policy_id)
        name = mimic_menu_label(wrapper)
        spec = mimic_progress_spec(wrapper)
        if spec is not None and spec[1] > 0:
            duration_hint = f"共 {int(spec[1])} 步，播完自动回 AMO"
        else:
            duration_hint = "无限时长，播完需手动回 AMO"

        sep = dim("=" * 42)
        lines = [
            f"{sep} {self._deploy_badge()} {white('Mimic 播放中：' + name)} {sep}",
            f"{mark('>>>')} {white('[MIMIC]')} {duration_hint}",
            "",
            *self._mimic_interrupt_lines(),
        ]
        for line in lines:
            tqdm.write(line)
        logger.info(strip_ansi(f"Mimic panel: {name}"))

    def _mimic_policy_label(self, mimic_idx: int) -> str:
        policy_id = self.policy_mimic_ids[mimic_idx]
        return self.policy_by_id(policy_id).name

    def _mimic_run_mode(self, mimic_idx: int) -> str:
        wrapper = self.policy_by_id(self.policy_mimic_ids[mimic_idx])
        policy = wrapper.policy
        match policy.__class__.__name__:
            case "LocoModePolicy":
                return "continuous | manual return (]/Select)"
            case "BeyondMimicPolicy":
                if policy.max_timestep > 0:
                    return f"finite {policy.max_timestep} steps | auto return when done"
                return "continuous | manual return (]/Select)"
            case "MotionTrackingPolicy":
                return f"finite {policy.total_frames} frames | auto return when done"
            case _:
                return "manual return (]/Select)"

    def _mimic_short_label(self, mimic_idx: int) -> str:
        policy_id = self.policy_mimic_ids[mimic_idx]
        return mimic_menu_label(self.policy_by_id(policy_id))

    def _mimic_run_mode_cn(self, mimic_idx: int) -> str:
        wrapper = self.policy_by_id(self.policy_mimic_ids[mimic_idx])
        policy = wrapper.policy
        ret = self._return_key_hint()
        match policy.__class__.__name__:
            case "LocoModePolicy":
                return f"无限时长，需手动按 {ret} 回到 AMO"
            case "BeyondMimicPolicy":
                if policy.max_timestep > 0:
                    return f"有限 {policy.max_timestep} 步，播完自动回 AMO"
                return f"无限时长，需手动按 {ret} 回到 AMO"
            case "MotionTrackingPolicy":
                return f"有限 {policy.total_frames} 帧，播完自动回 AMO"
            case _:
                return f"需手动按 {ret} 回到 AMO"

    def log_amo_panel(self, context: str = "active") -> None:
        """Print AMO status panel: current mode, mimic menu, and controls."""
        titles = {
            "startup": "AMO 就绪",
            "active": "AMO 模式",
            "returned": "已回到 AMO",
            "selected": "AMO 模式（已选动作）",
        }
        hints = {
            "startup": "当前在 AMO：可站立或用手柄/遥控器行走",
            "active": "当前在 AMO：按 " + self._return_key_hint() + " 可随时重新显示本菜单",
            "returned": "刚从 Mimic 动作回到 AMO",
            "selected": "仍在 AMO：按 " + (key("[") if self.is_sim else key("Start")) + " 开始下面选中的动作",
        }
        title = titles.get(context, "AMO 模式")
        hint = hints.get(context, hints["active"])

        sep = dim("=" * 42)
        select_hint = (
            key(";") + " / " + key("'") + " 切换，" + key("[") + " 开始"
            if self.is_sim
            else key("R1") + " / " + key("L1") + " 切换，" + key("Start") + " 开始"
        )
        lines = [
            f"{sep} {self._deploy_badge()} {white(title)} {sep}",
            f"{mark('>>>')} {white('[AMO]')} {hint}",
            "",
            f"可选 Mimic 动作（{select_hint}）：",
        ]
        for idx in range(self.policy_mimic_num):
            marker = mark(">>>") if idx == self.policy_mimic_idx else "   "
            name = self._mimic_short_label(idx)
            lines.append(f"{marker} [{idx}] {name}")
            lines.append(f"       {self._mimic_run_mode_cn(idx)}")
        lines.extend([""] + self._amo_locomotion_lines() + [""] + self._mode_switch_lines())
        for line in lines:
            tqdm.write(line)
        logger.info(strip_ansi(f"AMO panel: {title}"))

    def log_mimic_menu(self, title: str = "Mimic selection") -> None:
        context_map = {
            "Ready in AMO": "startup",
            "Back in AMO": "returned",
            "Selected mimic (AMO mode)": "selected",
        }
        self.log_amo_panel(context_map.get(title, "active"))

    def _interpolate_init(
        self,
        get_target_pos: Callable[[], np.ndarray],
        durations: list[int],
        callback_start=None,
        callback_end=None,
    ):
        self.interp_get_target_pos = get_target_pos
        self.interp_durations = durations
        self.interp_callback_start = callback_start
        self.interp_callback_end = callback_end
        self.interp_pbar = ProgressBar("Interpolation", durations[1])

        self.interp_state = self.InterpState.START
        # Starting Tasks
        self.timer.add(self._interpolate_start, delay_steps=durations[0])
        # Ending Tasks
        self.timer.add(self._interpolate_end, delay_steps=sum(durations) + 1)

    def _interpolate_start(self):
        if self.interp_state != self.InterpState.START:
            return
        if self.interp_callback_start is not None:
            self.interp_callback_start()
            self.interp_callback_start = None

        self.interp_start_pos = self.env.dof_pos.copy()
        self.interp_target_pos = self.interp_get_target_pos()
        self.interp_timestep = 0
        self.interp_state = self.InterpState.IN_PROGRESS

        # logger.debug("Interpolation started.")

    def _interpolate_end(self):
        if self.interp_state != self.InterpState.END:
            return
        self.override_dof_pos = self.interp_target_pos.copy()
        if self.interp_pbar:
            self.interp_pbar.close()
            self.interp_pbar = None
        if self.interp_callback_end is not None:
            self.interp_callback_end()
            self.interp_callback_end = None
        self.interp_state = self.InterpState.IDLE

        # logger.debug("Interpolation ended.")

    def _interpolate_step(self):
        if self.interp_state != self.InterpState.IN_PROGRESS:
            return

        if self.interp_pbar:
            self.interp_pbar.set(self.interp_timestep)

        progress = self.interp_timestep / self.interp_durations[1]
        alpha = min(progress, 1.0)
        self.override_dof_pos = (1 - alpha) * self.interp_start_pos + alpha * self.interp_target_pos

        if self.interp_timestep < self.interp_durations[1]:
            self.interp_timestep += 1
        else:
            self.interp_state = self.InterpState.END

    def step(self, env_data, ctrl_data):
        super().step(env_data, ctrl_data)
        self._interpolate_step()
        self._update_mimic_progress()

    def toggle_mimic_policy(self, delta: int):
        # only switch mimic policy if current policy is locomotion
        if self.current_policy_id != self.policy_loco_id:
            logger.warning("Cannot switch mimic policy when policy is mimic.")
            return

        self.policy_mimic_idx = (self.policy_mimic_idx + delta) % self.policy_mimic_num
        self.log_mimic_menu("Selected mimic (AMO mode)")

    def switch_to_loco(self):
        if self.current_policy_id == self.policy_loco_id and self.interp_state == self.InterpState.IDLE:
            self.log_amo_panel("active")
            return
        if self.current_policy_id != self.policy_loco_id:
            self.policy_by_id(self.policy_loco_id).reset()
            self.warmup_policy_indices.add(self.policy_loco_id)
        self._interpolate_init(
            get_target_pos=lambda: self.loco_dof_pos,
            durations=self.DURATIONS_MIMIC_LOCO,
            callback_start=lambda: self.set_policy(self.policy_loco_id),
            callback_end=lambda: self.log_mimic_menu("Back in AMO"),
        )

    def switch_to_mimic(self):
        if self.current_policy_id != self.policy_loco_id:
            logger.warning("Already in mimic policy. Press ]/Select to return to AMO first.")
            return
        policy_mimic_id = self.policy_mimic_ids[self.policy_mimic_idx]
        logger.warning(
            f"========== Start mimic [{self.policy_mimic_idx}] "
            f"{self._mimic_short_label(self.policy_mimic_idx)} =========="
        )
        self.policy_by_id(policy_mimic_id).reset()
        self.warmup_policy_indices.add(policy_mimic_id)
        self._interpolate_init(
            get_target_pos=lambda: self.policy_by_id(policy_mimic_id).get_init_dof_pos(),
            durations=self.DURATIONS_LOCO_MIMIC,
            callback_end=lambda: self.set_policy(policy_mimic_id),
        )


@pipeline_registry.register
class RlLocoMimicPipeline(RlMultiPolicyPipeline):
    cfg: RlLocoMimicPipelineCfg

    @property
    def policy(self) -> PolicyWrapper:
        return self.policy_manager.policy

    def __init__(self, cfg: RlLocoMimicPipelineCfg):
        # Skip RlMultiPolicyPipeline initialization
        Pipeline.__init__(self, cfg=cfg)

        env_class: type[Environment] = getattr(robojudo.environment, self.cfg.env.env_type)
        self.env: Environment = env_class(cfg_env=self.cfg.env, device=self.device)

        self.ctrl_manager = CtrlManager(cfg_ctrls=self.cfg.ctrl, env=self.env, device=self.device)

        # upper body override
        self.num_upper_body_dof = self.cfg.upper_dof_num
        if upper_dof_pos_default := self.cfg.upper_dof_pos_default:
            loco_dof_pos = self.env.default_pos.copy()
            loco_dof_pos[-self.num_upper_body_dof :] = upper_dof_pos_default
            self.loco_dof_pos = loco_dof_pos
        else:
            self.loco_dof_pos = self.env.default_pos
        if override_dof_indices := self.cfg.upper_dof_override_indices:
            self.override_dof_indices = override_dof_indices
        else:
            self.override_dof_indices = list(range(-self.num_upper_body_dof, 0))

        self.policy_manager = PolicyInterpManager(
            cfg_policy_loco=self.cfg.loco_policy,
            cfg_policies=self.cfg.mimic_policies,
            env=self.env,
            loco_dof_pos=self.loco_dof_pos,
            device=self.device,
        )
        self.env.update_dof_cfg(override_cfg=self.policy.cfg_action_dof)
        self.visualizer = self.env.visualizer

        self.freq = self.cfg.loco_policy.freq
        self.dt = 1.0 / self.freq

        self.policy_locomotion_mimic_flag = 0  # 0: locomotion, 1: mimic
        self._auto_switch_done = False
        self._warmup_steps = max(self.cfg.warmup_steps, 0)
        self._warmup_to_mimic = self.cfg.warmup_to_mimic
        self._warmup_mimic_idx = max(self.cfg.warmup_mimic_idx, 0)

        self.self_check()
        self.reset()
        self.policy_manager.log_amo_panel("startup")

    def post_step_callback(self, env_data, ctrl_data, extras, pd_target):
        self.timestep += 1

        commands = ctrl_data.get("COMMANDS", [])

        # Handle policy CALLBACK
        for callback in extras.get("CALLBACK", []):
            match callback:
                case "[MOTION_DONE]":
                    if self.policy_locomotion_mimic_flag == 1:
                        mimic_name = self.policy_manager._mimic_short_label(self.policy_manager.policy_mimic_idx)
                        logger.warning(f"========== Mimic done: {mimic_name} → auto return to AMO ==========")
                        commands.append("[POLICY_LOCO]")

        for command in commands:
            match command:
                case "[SHUTDOWN]":
                    logger.warning("Emergency shutdown!")
                    self.env.shutdown()
                case "[SIM_REBORN]":
                    if hasattr(self.env, "reborn"):
                        logger.warning("Simulation Env reborn!")
                        self.env.reborn()  # pyright: ignore[reportAttributeAccessIssue]
                case cmd if cmd.startswith("[POLICY_SWITCH]"):
                    switch_target = cmd.split(",")[1]
                    if switch_target == "NEXT":
                        self.policy_manager.toggle_mimic_policy(1)
                    elif switch_target == "LAST":
                        self.policy_manager.toggle_mimic_policy(-1)
                case "[POLICY_LOCO]":
                    self.policy_locomotion_mimic_flag = 0
                    self.policy_manager.switch_to_loco()
                case "[POLICY_MIMIC]":
                    self.policy_locomotion_mimic_flag = 1
                    self.policy_manager.switch_to_mimic()

        # Auto switch after warmup (e.g., AMO -> LocoMode)
        if (
            not self._auto_switch_done
            and self._warmup_to_mimic
            and self._warmup_steps > 0
            and self.timestep >= self._warmup_steps
            and self.policy_manager.current_policy_id == self.policy_manager.policy_loco_id
        ):
            self._auto_switch_done = True
            self.policy_manager.policy_mimic_idx = self._warmup_mimic_idx % self.policy_manager.policy_mimic_num
            self.policy_locomotion_mimic_flag = 1
            logger.info(
                f"Warmup done ({self._warmup_steps} steps). Auto switch to mimic idx={self.policy_manager.policy_mimic_idx}."
            )
            self.policy_manager.switch_to_mimic()

        self.ctrl_manager.post_step_callback(ctrl_data)

        self.policy.post_step_callback(commands)
        if self.visualizer is not None:
            self.policy.debug_viz(self.visualizer, env_data, ctrl_data, extras)

        # # Handle policy switch after step to avoid mid-step change
        self.policy_manager.step(env_data, ctrl_data)

        self.safety_check()
        if self.cfg.debug.log_obs:
            self.debug_logger.log(
                env_data=env_data,
                ctrl_data=ctrl_data,
                extras=extras,
                pd_target=pd_target,
                timestep=self.timestep,
            )

    def step(self, dry_run=False):
        self.env.update()
        env_data = self.env.get_data()
        ctrl_data = self.ctrl_manager.get_ctrl_data(env_data)

        commands = ctrl_data.get("COMMANDS", [])
        if len(commands) > 0:
            logger.info(f"{'=' * 10} COMMANDS {'=' * 10}\n{commands}")

        if self.policy_manager.current_policy_id == self.policy_manager.policy_loco_id:
            ctrl_data["ref_dof_pos"] = self.policy.obs_adapter.fit(self.policy_manager.override_dof_pos)

        obs, extras = self.policy.get_observation(env_data, ctrl_data)

        pd_target = self.policy.get_pd_target(obs)

        if self.policy_manager.current_policy_id == self.policy_manager.policy_loco_id:
            pd_target[self.override_dof_indices] = self.policy_manager.override_dof_pos[self.override_dof_indices]

        if not dry_run:
            self.env.step(pd_target, extras.get("hand_pose", None))
            # logger.debug(pd_target)

        self.post_step_callback(env_data, ctrl_data, extras, pd_target)

    def prepare(self):
        init_motor_angle = self.loco_dof_pos.copy()
        super().prepare(init_motor_angle=init_motor_angle)


if __name__ == "__main__":
    pass
