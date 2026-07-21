#!/usr/bin/env python3
import argparse
import json
import os
import random
import sys
from pathlib import Path

import dill
import hydra
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_frozen_replay_197 import (  # noqa: E402
    load_policy,
    load_policy_input_npz,
    post_process_action,
    raw_obs_to_model_obs,
    relative_actions_to_absolute_actions,
)
from diffusion_policy.common.pytorch_util import dict_apply  # noqa: E402


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def adjacent_quat_angles_deg(final_action):
    arr = np.asarray(final_action, dtype=np.float64)
    out = {}
    for name, sl in (("left", slice(3, 7)), ("right", slice(11, 15))):
        q = arr[:, sl]
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
        signed_dots = np.sum(q[:-1] * q[1:], axis=1)
        physical_dots = np.clip(np.abs(signed_dots), 0.0, 1.0)
        signed_dots_clipped = np.clip(signed_dots, -1.0, 1.0)
        out[name] = {
            "physical_deg": np.degrees(2.0 * np.arccos(physical_dots)),
            "signed_deg": np.degrees(2.0 * np.arccos(signed_dots_clipped)),
            "signed_dots": signed_dots,
        }
    return out


def summarize_angles(values):
    arr = np.asarray(values, dtype=np.float64)
    bins = [0, 1, 2, 5, 10, 30, 60, 90, 120, 150, 170, 175, 179, 180.0001]
    hist, edges = np.histogram(arr, bins=bins)
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)) if arr.size else None,
        "mean": float(np.mean(arr)) if arr.size else None,
        "p50": float(np.percentile(arr, 50)) if arr.size else None,
        "p90": float(np.percentile(arr, 90)) if arr.size else None,
        "p95": float(np.percentile(arr, 95)) if arr.size else None,
        "p99": float(np.percentile(arr, 99)) if arr.size else None,
        "max": float(np.max(arr)) if arr.size else None,
        "near_0_le_2deg": int(np.sum(arr <= 2.0)),
        "near_0_le_5deg": int(np.sum(arr <= 5.0)),
        "large_ge_30deg": int(np.sum(arr >= 30.0)),
        "large_ge_90deg": int(np.sum(arr >= 90.0)),
        "near_180_ge_170deg": int(np.sum(arr >= 170.0)),
        "near_180_ge_175deg": int(np.sum(arr >= 175.0)),
        "hist_bins_deg": bins,
        "hist_counts": hist.astype(int).tolist(),
        "hist_labels": [f"[{edges[i]}, {edges[i + 1]})" for i in range(len(hist))],
    }


def run_one(name, ckpt_path, frozen_input, runs, seed, num_inference_steps, device):
    policy, action_representation, cfg = load_policy(ckpt_path, num_inference_steps, device)
    model_obs, absolute_batch = raw_obs_to_model_obs(
        frozen_input,
        action_representation=action_representation,
        use_relative_action=True,
    )
    obs_tensor = dict_apply(model_obs, lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
    base_abs = np.concatenate(
        [absolute_batch["left_robot_tcp_pose"][-1], absolute_batch["right_robot_tcp_pose"][-1]]
    )
    out = {
        "ckpt_path": str(ckpt_path),
        "action_representation": action_representation,
        "task_name": str(cfg.task.name),
        "phases": {},
    }
    for phase, phase_seed in (("random_denoise", None), ("fixed_seed_denoise", seed)):
        phase_out = {"left": [], "right": [], "left_signed": [], "right_signed": [], "runs": []}
        for i in range(runs):
            if phase_seed is not None:
                set_all_seeds(phase_seed)
            with torch.no_grad():
                result = policy.predict_action(obs_tensor)
            raw = result.get("action", result.get("action_pred"))[0].detach().cpu().numpy()
            abs_action = relative_actions_to_absolute_actions(
                raw,
                base_absolute_action=base_abs,
                action_representation=action_representation,
            )
            final = post_process_action(abs_action)
            angles = adjacent_quat_angles_deg(final)
            run_record = {"iteration": i}
            for arm in ("left", "right"):
                phys = angles[arm]["physical_deg"]
                signed = angles[arm]["signed_deg"]
                phase_out[arm].extend(phys.tolist())
                phase_out[f"{arm}_signed"].extend(signed.tolist())
                run_record[arm] = {
                    "max_physical_deg": float(np.max(phys)),
                    "max_signed_deg": float(np.max(signed)),
                    "near_180_physical_count": int(np.sum(phys >= 170.0)),
                    "small_physical_count": int(np.sum(phys <= 5.0)),
                }
            phase_out["runs"].append(run_record)
            print(
                f"{name} {phase} iter={i:02d} "
                f"Lmax={run_record['left']['max_physical_deg']:.2f} "
                f"Rmax={run_record['right']['max_physical_deg']:.2f}",
                flush=True,
            )
        out["phases"][phase] = {
            "left_physical": summarize_angles(phase_out["left"]),
            "right_physical": summarize_angles(phase_out["right"]),
            "left_signed_no_abs_dot": summarize_angles(phase_out["left_signed"]),
            "right_signed_no_abs_dot": summarize_angles(phase_out["right_signed"]),
            "runs": phase_out["runs"],
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--ckpt", action="append", nargs=2, metavar=("NAME", "PATH"), required=True)
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    frozen_input, obs_time = load_policy_input_npz(args.input)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = {
        "obs_time": obs_time,
        "runs": args.runs,
        "seed": args.seed,
        "num_inference_steps": args.num_inference_steps,
        "angle_definition": "physical_deg = 2*acos(abs(dot(q_t,q_t+1))); signed_no_abs_dot keeps q/-q sign flips visible",
        "ckpts": {},
    }
    for name, ckpt_path in args.ckpt:
        result["ckpts"][name] = run_one(
            name=name,
            ckpt_path=Path(ckpt_path),
            frozen_input=frozen_input,
            runs=args.runs,
            seed=args.seed,
            num_inference_steps=args.num_inference_steps,
            device=device,
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
