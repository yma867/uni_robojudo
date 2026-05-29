import time
from queue import Empty, Queue

from robojudo.controller import Controller, ctrl_registry
from robojudo.controller.ctrl_cfgs import JoystickCtrlCfg
from robojudo.controller.utils.joystick import JoystickThread


@ctrl_registry.register
class JoystickCtrl(Controller):
    cfg_ctrl: JoystickCtrlCfg

    def __init__(self, cfg_ctrl: JoystickCtrlCfg, env=None, device="cpu"):
        super().__init__(cfg_ctrl=cfg_ctrl, env=env, device=device)

        self.state_queue = Queue(maxsize=2)  # for axes
        self.event_queue = Queue(maxsize=100)  # for button/dpad events
        self.joystick_thread = JoystickThread(self.state_queue, self.event_queue)
        self.joystick_thread.start()

        self.axes_names = self.joystick_thread.config["axis_config"]["axis_map"].keys()
        self.reset()

    def reset(self):
        self.combination_init_buttons = self.cfg_ctrl.combination_init_buttons
        self.onhold_buttons = set()
        while not self.state_queue.empty():
            try:
                self.state_queue.get_nowait()
            except Empty:
                break

        while not self.event_queue.empty():
            try:
                self.event_queue.get_nowait()
            except Empty:
                break

        self.last_state = {
            "type": "axes",
            "axes": {name: 0.0 for name in self.axes_names},
            "timestamp": time.time(),
        }

    def get_state(self):
        try:
            state = self.state_queue.get_nowait()
            self.last_state = state.copy()
        except Empty:
            state = self.last_state

        return state

    def get_events(self):
        events = []
        while not self.event_queue.empty():
            try:
                event = self.event_queue.get_nowait()
                events.append(event)
            except Empty:
                break
        return events

    def get_data(self):
        state = self.get_state()
        events = self.get_events()

        return {
            "axes": state["axes"],
            "button_event": events,
        }

    def process_triggers(self, ctrl_data):
        commands = []
        if len(self.triggers) == 0:
            return ctrl_data, commands

        # Normalize button naming across controller families:
        # - DualSense uses PS5 physical semantics (×/○/□/△, L1/R1, Share/Options).
        # - Many configs historically used Xbox-like semantics (A/B/X/Y, LB/RB, Back/Start).
        #
        # To keep configs PS5-native (as the user expects) while remaining backward compatible,
        # we try trigger lookups using both the original name and its known aliases.
        alias_pairs = {
            # Face buttons
            "×": "A",
            "○": "B",
            "□": "X",
            "△": "Y",
            # Shoulders
            "L1": "LB",
            "R1": "RB",
            # Menu buttons
            "Share": "Back",
            "Options": "Start",
        }
        alias_map: dict[str, set[str]] = {}
        for a, b in alias_pairs.items():
            alias_map.setdefault(a, set()).add(b)
            alias_map.setdefault(b, set()).add(a)

        def name_variants(name: str) -> list[str]:
            variants = [name]
            for alt in sorted(alias_map.get(name, set())):
                if alt not in variants:
                    variants.append(alt)
            return variants

        def lookup_trigger(key: str) -> str | None:
            return self.triggers.get(key, None)

        for event in ctrl_data["button_event"]:
            if event["type"] == "button":
                if event["name"] in self.combination_init_buttons:
                    if event["pressed"]:
                        self.onhold_buttons.add(event["name"])
                    else:
                        self.onhold_buttons.discard(event["name"])
                else:
                    if event["pressed"]:
                        command = None
                        if len(self.onhold_buttons) == 0:
                            for n in name_variants(event["name"]):
                                command = lookup_trigger(n)
                                if command is not None:
                                    break
                        else:
                            onhold_sorted = sorted(list(self.onhold_buttons))
                            onhold_variants = [name_variants(n) for n in onhold_sorted]
                            event_variants = name_variants(event["name"])

                            # Generate small cartesian product of variants (onhold size is tiny in practice).
                            def gen_onhold(idx: int, acc: list[str]):
                                if idx == len(onhold_variants):
                                    yield acc
                                    return
                                for v in onhold_variants[idx]:
                                    yield from gen_onhold(idx + 1, acc + [v])

                            for onhold_choice in gen_onhold(0, []):
                                for ev_name in event_variants:
                                    event_combination = "+".join(onhold_choice + [ev_name])
                                    command = lookup_trigger(event_combination)
                                    if command is not None:
                                        break
                                if command is not None:
                                    break
                        if command is not None:
                            commands.append(command)
                            # remove event after triggered
                            ctrl_data["button_event"].remove(event)

        return ctrl_data, commands


if __name__ == "__main__":
    joystick_ctrl = JoystickCtrl(
        cfg_ctrl=JoystickCtrlCfg(
            triggers={
                "×": "[TEST_CROSS]",
                "○": "[TEST_CIRCLE]",
                "L1+Left": "[TEST_L1_Left]",
                "R1+Right": "[TEST_R1_Right]",
                "L1+R1+×": "[TEST_L1_R1_CROSS]",
            },
        )
    )
    for _ in range(10000):
        ctrl_data = joystick_ctrl.get_data()
        ctrl_data, commands = joystick_ctrl.process_triggers(ctrl_data)
        print(ctrl_data)
        print(commands)
        print("================================")
        time.sleep(0.3)
    exit()
