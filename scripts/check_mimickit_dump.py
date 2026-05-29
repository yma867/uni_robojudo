#!/usr/bin/env python3
"""对比 MimicKit dump 数据与标准答案"""
import numpy as np
import json
from pathlib import Path

def load_standard_answer():
    """加载标准答案"""
    std_path = Path("/home/lyz/Desktop/code/MimicKit/DEBUG_STANDARD_ANSWER.json")
    if std_path.exists():
        with open(std_path, "r") as f:
            return json.load(f)
    return None

def compare_dump(dump_path="/tmp/mimickit_dump.npz"):
    """对比 dump 文件与标准答案"""
    dump_path = Path(dump_path)
    if not dump_path.exists():
        print(f"❌ Dump file not found: {dump_path}")
        return
    
    std = load_standard_answer()
    if std is None:
        print("⚠️  Standard answer not found, only showing dump data")
    
    data = np.load(dump_path)
    
    print("=" * 80)
    print("MimicKit Debug Dump Comparison")
    print("=" * 80)
    
    # 1. 观测维度检查
    obs_raw = data.get("obs_raw")
    if obs_raw is not None:
        print(f"\n1. 观测维度: {obs_raw.shape}")
        if std:
            expected_dim = std["observation_data"]["raw_obs_dimension"]
            if obs_raw.shape[0] == expected_dim:
                print(f"   ✅ 维度正确: {expected_dim}")
            else:
                print(f"   ❌ 维度错误: 期望 {expected_dim}, 实际 {obs_raw.shape[0]}")
        
        # 前10维对比
        print(f"\n2. 观测前10维 (raw):")
        print(f"   实际: {obs_raw[:10]}")
        if std:
            expected_first_10 = std["observation_data"]["raw_obs_first_10"]
            print(f"   期望: {expected_first_10}")
            diff = np.abs(obs_raw[:10] - np.array(expected_first_10))
            print(f"   差异: {diff}")
            max_diff_idx = np.argmax(diff)
            print(f"   最大差异在索引 {max_diff_idx}: {diff[max_diff_idx]}")
    
    # 3. 网络动作输出
    action_net = data.get("action_net")
    if action_net is not None:
        print(f"\n3. 网络动作输出 (前5维):")
        print(f"   实际: {action_net[:5]}")
        if std:
            expected_action = std["action_data"]["raw_network_action"]
            print(f"   期望: {expected_action[:5]}")
            diff = np.abs(action_net - np.array(expected_action))
            print(f"   最大差异: {np.max(diff):.6f} (索引 {np.argmax(diff)})")
    
    # 4. Default Pose 检查
    default_pose = data.get("default_pose_for_action")
    if default_pose is not None:
        print(f"\n4. Default Pose (用于动作计算):")
        print(f"   实际: {default_pose}")
        if std:
            expected_default = std["action_data"]["init_dof_pos_default_pose"]
            print(f"   期望: {expected_default}")
            diff = np.abs(default_pose - np.array(expected_default))
            print(f"   差异: {diff}")
            if np.any(diff > 0.01):
                print(f"   ⚠️  发现差异！最大差异: {np.max(diff):.6f} (索引 {np.argmax(diff)})")
            else:
                print(f"   ✅ Default Pose 匹配")
    
    # 5. 最终目标关节位置
    target_joint_pos = data.get("target_joint_pos")
    if target_joint_pos is not None:
        print(f"\n5. 最终目标关节位置 (前5维):")
        print(f"   实际: {target_joint_pos[:5]}")
        if std:
            # 根据公式计算期望值
            expected_action = np.array(std["action_data"]["raw_network_action"])
            expected_default = np.array(std["action_data"]["init_dof_pos_default_pose"])
            expected_target = expected_action + expected_default
            print(f"   期望: {expected_target[:5]}")
            diff = np.abs(target_joint_pos - expected_target)
            print(f"   最大差异: {np.max(diff):.6f} (索引 {np.argmax(diff)})")
    
    # 6. 元数据检查
    print(f"\n6. 配置元数据:")
    for key in ["meta_global_obs", "meta_zero_center_action", "meta_obs_dim"]:
        if key in data:
            print(f"   {key}: {data[key]}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    import sys
    dump_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/mimickit_dump.npz"
    compare_dump(dump_path)
