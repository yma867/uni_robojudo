"""
Probe embedded motion length of a BeyondMimic ONNX model.

The ONNX stores reference joint_pos indexed by time_step. After the clip ends,
joint_pos typically repeats (plateau). This script finds that last changing frame.

Usage:
  python scripts/probe_beyondmimic_length.py Dance_wose
  python scripts/probe_beyondmimic_length.py --all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "models" / "g1" / "beyondmimic"


def probe_motion_length(onnx_path: Path, max_scan: int = 20000, diff_threshold: float = 1e-3) -> int:
    """Return the last frame index where reference joint_pos still changes."""
    sess = ort.InferenceSession(onnx_path.as_posix(), providers=["CPUExecutionProvider"])
    obs = np.zeros((1, sess.get_inputs()[0].shape[-1]), dtype=np.float32)

    def joint_pos(t: int) -> np.ndarray:
        return sess.run(
            ["joint_pos"],
            {"obs": obs, "time_step": np.array([[t]], dtype=np.float32)},
        )[0].flatten()

    last_active = 0
    for t in range(max_scan):
        if np.max(np.abs(joint_pos(t + 1) - joint_pos(t))) > diff_threshold:
            last_active = t + 1
    return last_active


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe BeyondMimic ONNX motion length")
    parser.add_argument("policy_name", nargs="?", help="e.g. Dance_wose")
    parser.add_argument("--all", action="store_true", help="probe all .onnx in beyondmimic folder")
    parser.add_argument("--freq", type=float, default=50.0, help="control frequency for seconds estimate")
    args = parser.parse_args()

    if args.all:
        names = sorted(p.stem for p in ASSETS.glob("*.onnx"))
    else:
        if not args.policy_name:
            parser.error("provide policy_name or use --all")
        names = [args.policy_name]

    print(f"{'policy':<16} {'max_timestep':>12} {'seconds@50Hz':>12}")
    print("-" * 44)
    for name in names:
        path = ASSETS / f"{name}.onnx"
        if not path.exists():
            print(f"{name:<16} {'MISSING':>12}")
            continue
        steps = probe_motion_length(path)
        seconds = steps / args.freq
        print(f"{name:<16} {steps:>12} {seconds:>12.1f}")
        print(f"  → config: G1BeyondMimicPolicyCfg(policy_name=\"{name}\", max_timestep={steps})")


if __name__ == "__main__":
    main()
