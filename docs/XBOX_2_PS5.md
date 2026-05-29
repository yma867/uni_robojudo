# Xbox 手柄 ↔ PS5 DualSense 手柄按键映射（用于改代码）

本项目 sim2sim 的 `JoystickCtrl` 支持 **PS5 DualSense**：当检测到 DualSense 时，会在输入事件里使用 **PS5 物理按键语义**（`×/○/□/△`, `L1/R1`, `Share/Options` 等），而不是 ABXY。

> 兼容性：为了兼容旧配置，触发器仍支持别名（例如 `×` ↔ `A`, `R1` ↔ `RB`）。但建议配置里优先写 PS5 语义，避免混淆。

## 1) 物理按键名称对照（最常用）

| 逻辑含义（很多项目用） | Xbox 物理按键 | PS5 物理按键 |
|---|---|---|
| A | A | Cross（×）|
| B | B | Circle（○）|
| X | X | Square（□）|
| Y | Y | Triangle（△）|
| LB / L1 | LB | L1 |
| RB / R1 | RB | R1 |
| Back / Select | Back | Share |
| Start | Start | Options |
| Home | Xbox 键 | PS 键 |
| L3 | 左摇杆按下 | 左摇杆按下 |
| R3 | 右摇杆按下 | 右摇杆按下 |

## 2) pygame/SDL 下常见 button id（注意：不同驱动可能不同）

### 2.1 “标准布局”（很多项目直接用 A/B/X/Y 抽象）

这套对应本项目历史的 “Xbox-like” 命名风格（见 `robojudo/controller/utils/joystick.py` 的默认映射）：

| 抽象按钮 | button id | PS5 物理按键 | Xbox 物理按键 |
|---|---:|---|---|
| A | 0 | Cross（×） | A |
| B | 1 | Circle（○） | B |
| X | 2 | Square（□） | X |
| Y | 3 | Triangle（△） | Y |
| L1 | 4 | L1 | LB |
| R1 | 5 | R1 | RB |
| SELECT | 6 | Share | Back |
| START | 7 | Options | Start |

### 2.2 本项目 PS5Controller（实测 DualSense）

本项目 DualSense（PS5）命名风格的 button id（实测，见 `robojudo/controller/utils/joystick.py` 的 DualSense profile）：

| PS5 物理按键 | button id |
|---|---:|
| Cross（×） | 0 |
| Circle（○） | 1 |
| Triangle（△） | 2 |
| Square（□） | 3 |
| L1 | 4 |
| R1 | 5 |
| L2 | 6 |
| R2 | 7 |
| Share | 8 |
| Options | 9 |
| PS | 10 |
| L3 | 11 |
| R3 | 12 |

## 3) pygame/SDL 下常见 axis id（摇杆/扳机）

### 3.1 PS5 DualSense（本项目实测）

来自 `robojudo/controller/utils/joystick.py` 的 DualSense profile：

| 功能 | axis id |
|---|---:|
| 左摇杆 X（左右） | 0 |
| 左摇杆 Y（上下） | 1 |
| L2 扳机 | 2 |
| 右摇杆 X（左右） | 3 |
| 右摇杆 Y（上下） | 4 |
| R2 扳机 | 5 |

### 3.2 Xbox（常见参考，可能因驱动不同而变）

| 功能 | 常见 axis id |
|---|---:|
| 左摇杆 X | 0 |
| 左摇杆 Y | 1 |
| LT | 2 |
| 右摇杆 X | 3 |
| 右摇杆 Y | 4 |
| RT | 5 |

## 4) 最快验证方法

用本项目的测试脚本看你机器上的真实 button/axis：

```bash
python tools/joystick_test.py
```


