# Fix OMP perfmance issue on ARM platform (Jetson)
# [工程细节] ARM 平台（如 Jetson）上 OpenMP 多线程会导致推理延迟抖动，强制单线程避免性能问题
import os
import platform

# [工程细节] 仅在 aarch64 架构（Jetson/ARM）上限制 OMP 线程数，x86 不受影响
if platform.machine().startswith("aarch64"):
    os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import logging
import time

import robojudo.pipeline
from robojudo.config.config_manager import ConfigManager
from robojudo.pipeline.pipeline_cfgs import RlPipelineCfg
from robojudo.pipeline.rl_pipeline import RlPipeline

logger = logging.getLogger("robojudo")


def parse_args():
    # [工程细节] 通过 -c/--config 参数选择机器人配置（如 g1、h1 等），支持多机器人复用同一套代码
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="g1",
        help="Name of the config class to use",
    )
    args = parser.parse_args()
    return args


def main():
    # [底层原理] 部署入口：解析配置 → 实例化 Pipeline → 执行主循环
    # 整体流程：ConfigManager 加载 YAML/Python 配置 → 动态反射获取 Pipeline 类 → 调用 prepare() 初始化姿态 → 主循环 step()
    args = parse_args()
    logger.info(f"Using config: {args.config}")
    # [工程细节] ConfigManager 根据 config_name 动态加载对应机器人的完整配置树（env/policy/ctrl 三层配置）
    config_manager = ConfigManager(config_name=args.config)

    cfg: RlPipelineCfg = config_manager.get_cfg()

    pipeline_type = cfg.pipeline_type

    # [工程细节] 使用 getattr 动态反射获取 Pipeline 类，支持不同 pipeline 类型（RlPipeline 等）无需 if-else
    pipeline_class: type[RlPipeline] = getattr(robojudo.pipeline, pipeline_type)
    logger.info(f"Using pipeline: {pipeline_type} -> {pipeline_class}")

    pipeline = pipeline_class(cfg=cfg)

    # [Sim-to-Real] 真机部署时需要 prepare() 阶段：将机器人从当前姿态平滑插值到策略初始姿态，避免突变导致摔倒
    # 仿真环境跳过此步骤，因为可以直接 reset 到目标初始状态
    if not cfg.env.is_sim:
        pipeline.prepare()

    # [面试考点：频率解耦] Policy 推理频率与控制频率的对齐
    # 主循环以 pipeline.dt（= 1/policy_freq）为目标周期运行
    # 底层 MuJoCo/真机控制器以更高频率（sim_dt * decimation）执行 PD 控制
    # 两者通过 decimation 解耦：policy 每推理一次，底层执行 sim_decimation 步物理仿真
    while True:
        time_start = time.time()
        pipeline.step()
        time_end = time.time()
        time_diff = time_end - time_start

        # keep the pipeline running at the desired frequency
        # [面试考点：频率解耦] 用 time.sleep 补齐剩余时间，确保主循环严格以 policy 推理频率运行
        # time_diff 此时变为"剩余可睡眠时间" = 目标周期 - 实际耗时
        if not cfg.run_fullspeed:
            time_diff = pipeline.dt - time_diff
            if time_diff > 0:
                # [工程细节] 正常情况：推理耗时 < 目标周期，sleep 补齐剩余时间
                time.sleep(time_diff)
            else:
                # [面试考点：安全机制] 帧丢失检测：推理耗时超过目标周期（time_diff < 0）
                # 真机上帧丢失意味着控制指令延迟，可能导致不稳定，需要记录并在严重时紧急停止
                if not cfg.env.is_sim:
                    logger.error(f"Warning: frame drop -> {time_diff}")
                    # [面试考点：安全机制] 摔倒检测和紧急停止
                    # 帧丢失超过 200ms（time_diff < -0.2）视为严重异常，触发紧急关机
                    # 等待 10s 是为了让机器人安全停止后再退出进程
                    if time_diff < -0.2:
                        logger.critical("Exiting due to excessive frame drop")
                        pipeline.env.shutdown()
                        time.sleep(10)
                        break


# [工程细节] 标准 Python 入口保护，确保脚本只在直接运行时执行 main()，被 import 时不触发
if __name__ == "__main__":
    main()
