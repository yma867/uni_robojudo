# locomode_beyondmimic 使用说明

## 目的
本模式用于在 RoboJuDo 中运行：
- **AMO 预热**（启动后稳定站立）
- **LocoMode 行走**（来自 HKUST-GZ Liucheng 的 LocoMode）
- **BeyondMimic 动作**（Dance_wose / Violin / Waltz）

本模式强调在稳定站立后再切入 LocoMode，从而获得更稳的行走表现。

## 启动方式
在项目根目录执行：
```
python scripts/run_pipeline.py -c g1_locomode_beyondmimic

python scripts/run_pipeline.py -c g1_locomode_beyondmimic_real
```

## 策略结构（当前实现）
- **Loco（按 `]`）**：AMO
- **Mimic 列表（按 `[` 进入后用 `;` / `'` 切换）**：
  - 0: LocoModePolicy@policy_29dof
  - 1: BeyondMimic Dance_wose
  - 2: BeyondMimic Violin
  - 3: BeyondMimic Waltz

## 自动切换逻辑
启动后会：

1. 先进入 AMO
2. 约 2 秒后自动切换到 **LocoMode**

当前参数位置：`robojudo/config/g1/g1_loco_mimic_cfg.py`  
```
warmup_steps: 100  # 50Hz 下约 2 秒
warmup_to_mimic: True
warmup_mimic_idx: 0  # LocoMode
```

## 切换操作
### 键盘
- `]`：切回 Loco（AMO）
- `[`：进入 Mimic
- `;`：下一个 Mimic
- `'`：上一个 Mimic
- `i`：重置仿真
- `o`：退出程序

### 手柄操作说明（PS5 / Xbox）
本项目对 PS5 与 Xbox 名称做了别名映射，功能一致。下面按实体按键说明。

PS5 DualSense：
- `Share`：切回 Loco（AMO）
- `Options`：进入 Mimic
- `R1`：下一个 Mimic
- `L1`：上一个 Mimic

Xbox：
- `View`：切回 Loco（AMO）
- `Menu`：进入 Mimic
- `RB`：下一个 Mimic
- `LB`：上一个 Mimic

摇杆行走控制（两者一致）：
- 左摇杆前后：前进/后退
- 左摇杆左右：侧移
- 右摇杆左右：转向

## 常见切换路径
### AMO → LocoMode
- 等自动 2 秒切换，或  
- 按 `[` 进入 Mimic，再用 `;` / `'` 切到 idx=0

### AMO → BeyondMimic
- 按 `[` 进入 Mimic  
- 用 `;` / `'` 切到 idx=1/2/3

### BeyondMimic → AMO
- 按 `]` 或 `Share`

### BeyondMimic → LocoMode
- **不要按 `]`**  
- 直接用 `;` / `'` 切到 idx=0

## 行走控制（LocoMode）
手柄输入：
- 左摇杆前后：前进/后退
- 左摇杆左右：侧移
- 右摇杆左右：转向

unitree 遥控器操作逻辑：
select是从mimic切换到amo；
start是从amo切换mimic

R1是在amo里，切换mimicpolicy，+1
L1是在amo里，切换mimicpolicy，-1

0: LocoModePolicy
1: Dance_wose
2: Violin
3: Waltz
4: demo2
5: demo4
6: demo5
7: demo6
8: demo7
9: demo8
10: newyear1 (新增)

## 真机部署 - 网卡配置

在真机上运行时，如果提示网口名称错误，需要设置正确的网卡名称。使用以下命令查看网卡名称：

```bash
ip a
# 或者
ip link show
```

找到连接到机器人的网卡名称（通常是 `eth0`、`eno1`、`enp13s0` 等），然后在配置文件中修改 `net_if` 参数：

配置文件位置：`robojudo/config/g1/g1_loco_mimic_cfg.py` 中的 `g1_locomode_beyondmimic_real` 类

```python
env: G1RealEnvCfg = G1RealEnvCfg(
    unitree=G1UnitreeCfg(
        net_if="eth0",  # 修改为你的网卡名称
    ),
)
```

