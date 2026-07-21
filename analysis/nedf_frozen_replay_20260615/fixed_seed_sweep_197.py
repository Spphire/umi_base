#!/usr/bin/env python3
import argparse
import json
import os
import random
import sys
from pathlib import Path

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
    quat_angle_metrics,
    raw_obs_to_model_obs,
    relative_actions_to_absolute_actions,
    step_metrics,
)
from diffusion_policy.common.pytorch_util import dict_apply  # noqa: E402


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_seed(policy, obs_tensor, base_abs, action_representation, seed):
    set_all_seeds(seed)
    with torch.no_grad():
        result = policy.predict_action(obs_tensor)
    raw = result.get("action", result.get("action_pred"))[0].detach().cpu().numpy()
    abs_action = relative_actions_to_absolute_actions(
        raw,
        base_absolute_action=base_abs,
        action_representation=action_representation,
    )
    final = post_process_action(abs_action)
    q = quat_angle_metrics(final)
    return {
        "seed": int(seed),
        "raw_max_step_l2": step_metrics(raw)["max_step_l2"],
        "final_max_step_l2": step_metrics(final)["max_step_l2"],
        "left_max_deg": q["left"]["max_angle_deg"],
        "right_max_deg": q["right"]["max_angle_deg"],
        "max_deg": max(q["left"]["max_angle_deg"], q["right"]["max_angle_deg"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--repeat-bad", type=int, default=5)
    parser.add_argument("--threshold-deg", type=float, default=90.0)
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
        "start": args.start,
        "count": args.count,
        "threshold_deg": args.threshold_deg,
        "num_inference_steps": args.num_inference_steps,
        "ckpts": {},
    }
    seeds = list(range(args.start, args.start + args.count))
    for name, ckpt_path in args.ckpt:
        policy, action_representation, cfg = load_policy(Path(ckpt_path), args.num_inference_steps, device)
        model_obs, absolute_batch = raw_obs_to_model_obs(
            frozen_input,
            action_representation=action_representation,
            use_relative_action=True,
        )
        obs_tensor = dict_apply(model_obs, lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
        base_abs = np.concatenate(
            [absolute_batch["left_robot_tcp_pose"][-1], absolute_batch["right_robot_tcp_pose"][-1]]
        )
        rows = []
        bad = []
        for seed in seeds:
            row = run_seed(policy, obs_tensor, base_abs, action_representation, seed)
            rows.append(row)
            if row["max_deg"] >= args.threshold_deg:
                bad.append(row)
            print(
                f"{name} seed={seed} max_deg={row['max_deg']:.2f} "
                f"raw_l2={row['raw_max_step_l2']:.3f} final_l2={row['final_max_step_l2']:.3f}",
                flush=True,
            )
        repeats = {}
        for row in bad[:5]:
            seed = row["seed"]
            repeats[str(seed)] = [
                run_seed(policy, obs_tensor, base_abs, action_representation, seed)
                for _ in range(args.repeat_bad)
            ]
        result["ckpts"][name] = {
            "ckpt_path": str(ckpt_path),
            "task_name": str(cfg.task.name),
            "action_representation": action_representation,
            "rows": rows,
            "bad_seeds": bad,
            "bad_seed_repeats": repeats,
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
