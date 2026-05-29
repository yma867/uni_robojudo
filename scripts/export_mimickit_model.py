import argparse
import re
from pathlib import Path

import numpy as np
import torch


def _load_state_dict(pt_path: Path) -> dict:
    obj = torch.load(pt_path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise ValueError("Unsupported model format: expected a state_dict.")
    return obj


def _find_unique_key(keys: list[str], contains: list[str], suffixes: list[str]) -> str:
    candidates = [k for k in keys if all(s in k for s in contains)]
    candidates = [k for k in candidates if any(k.endswith(sfx) for sfx in suffixes)]
    if len(candidates) != 1:
        raise ValueError(f"Expected single match for {contains}+{suffixes}, got {candidates}")
    return candidates[0]


def _extract_actor_layers(state: dict) -> tuple[list[np.ndarray], list[np.ndarray]]:
    weight_keys = []
    bias_keys = []
    pattern = re.compile(r".*actor_layers\.(\d+)\.weight$")

    for key in state.keys():
        match = pattern.match(key)
        if match:
            weight_keys.append((int(match.group(1)), key))

    if not weight_keys:
        raise ValueError("No actor layers found in state dict.")

    weight_keys.sort(key=lambda x: x[0])
    for idx, w_key in weight_keys:
        b_key = w_key.replace(".weight", ".bias")
        if b_key not in state:
            raise ValueError(f"Missing bias for {w_key}")
        bias_keys.append((idx, b_key))

    weights = [state[k].cpu().numpy().astype(np.float32) for _, k in weight_keys]
    biases = [state[k].cpu().numpy().astype(np.float32) for _, k in bias_keys]
    return weights, biases


def _extract_action_head(state: dict) -> tuple[np.ndarray, np.ndarray]:
    keys = list(state.keys())
    w_key = _find_unique_key(keys, ["action_dist", "mean_net"], [".weight"])
    b_key = _find_unique_key(keys, ["action_dist", "mean_net"], [".bias"])
    return state[w_key].cpu().numpy().astype(np.float32), state[b_key].cpu().numpy().astype(np.float32)


def _extract_obs_norm(state: dict) -> tuple[np.ndarray, np.ndarray]:
    keys = list(state.keys())
    mean_key = _find_unique_key(keys, ["obs_norm"], ["_mean", ".mean", "._mean"])
    std_key = _find_unique_key(keys, ["obs_norm"], ["_std", ".std", "._std"])
    obs_mean = state[mean_key].cpu().numpy().astype(np.float32)
    obs_std = state[std_key].cpu().numpy().astype(np.float32)
    return obs_mean, obs_std


def _extract_action_norm(state: dict) -> tuple[np.ndarray, np.ndarray] | None:
    if "_a_norm._mean" in state and "_a_norm._std" in state:
        mean = state["_a_norm._mean"].cpu().numpy().astype(np.float32)
        std = state["_a_norm._std"].cpu().numpy().astype(np.float32)
        return mean, std
    return None


def export_model(pt_path: Path, out_path: Path):
    state = _load_state_dict(pt_path)
    weights, biases = _extract_actor_layers(state)
    head_w, head_b = _extract_action_head(state)
    obs_mean, obs_std = _extract_obs_norm(state)
    action_norm = _extract_action_norm(state)

    weights.append(head_w)
    biases.append(head_b)

    data = {
        "obs_mean": obs_mean,
        "obs_var": np.square(obs_std),
    }
    if action_norm is not None:
        data["action_mean"] = action_norm[0]
        data["action_std"] = action_norm[1]
    for i, (w, b) in enumerate(zip(weights, biases)):
        data[f"w{i}"] = w
        data[f"b{i}"] = b

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **data)


def main():
    parser = argparse.ArgumentParser(description="Export MimicKit actor + obs normalizer to npz.")
    parser.add_argument("--pt", required=True, help="Path to MimicKit .pt model")
    parser.add_argument("--out", required=True, help="Output .npz path")
    args = parser.parse_args()

    pt_path = Path(args.pt).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    export_model(pt_path, out_path)
    print(f"Exported: {out_path}")


if __name__ == "__main__":
    main()
