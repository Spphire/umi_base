#!/usr/bin/env python3
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import dill
import hydra
import numpy as np
import torch
import transforms3d as t3d
from scipy.spatial.transform import Rotation as R

from diffusion_policy.common.action_utils import (
    absolute_actions_to_relative_actions,
    relative_actions_to_absolute_actions,
)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.space_utils import ortho6d_to_rotation_matrix


def load_policy_input_npz(path):
    with np.load(path, allow_pickle=False) as data:
        manifest = json.loads(str(data["__manifest__"].item()))
        policy_input = {}
        for key, spec in manifest["keys"].items():
            if spec["type"] == "list":
                policy_input[key] = [np.array(data[name], copy=True) for name in spec["items"]]
            else:
                policy_input[key] = np.array(data[spec["item"]], copy=True)
        return policy_input, int(manifest["obs_time"])


def center_crop_and_resize_image(image, target_size=(224, 224)):
    h, w = image.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    cropped = image[y0 : y0 + side, x0 : x0 + side]
    return cv2.resize(cropped, target_size)


def process_quaternion(quat, conversion_type):
    if conversion_type == "f2l":
        return [quat[1], quat[2], quat[3], quat[0]]
    if conversion_type == "l2f":
        return [quat[3], quat[0], quat[1], quat[2]]
    raise ValueError(conversion_type)


def quaternion_to_pose9d(pose):
    pose = np.asarray(pose)
    pos = pose[..., :3]
    quat = pose[..., 3:]
    quat = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    rot_mats = R.from_quat(quat).as_matrix()
    rot6 = rot_mats[..., :, :2].reshape(pose.shape[:-1] + (6,), order="F")
    return np.concatenate([pos, rot6], axis=-1)


def raw_obs_to_model_obs(raw_obs, action_representation, use_relative_action=True):
    batch = {}
    for idx, camera_name in enumerate(("IPhoneCameraDevice_0", "IPhoneCameraDevice_1")):
        frames = [np.asarray(frame) for frame in raw_obs[camera_name]]
        processed = [center_crop_and_resize_image(frame, (224, 224)) for frame in frames]
        stacked = np.stack(processed)
        key = "left_wrist_img" if idx == 0 else "right_wrist_img"
        batch[key] = (stacked / 255.0).transpose(0, 3, 1, 2).astype(np.float32)

    robot_obs = np.asarray(raw_obs["RizonRobot_2"])
    if robot_obs.ndim == 1:
        robot_obs = robot_obs[None, :]
    left_tcp_w_grip = robot_obs[:, 14:22]
    right_tcp_w_grip = robot_obs[:, 36:44]

    left_pose = np.concatenate(
        [
            left_tcp_w_grip[:, :3],
            np.array([process_quaternion(q, "f2l") for q in left_tcp_w_grip[:, 3:7]]),
        ],
        axis=-1,
    )
    right_pose = np.concatenate(
        [
            right_tcp_w_grip[:, :3],
            np.array([process_quaternion(q, "f2l") for q in right_tcp_w_grip[:, 3:7]]),
        ],
        axis=-1,
    )
    batch["left_robot_tcp_pose"] = quaternion_to_pose9d(left_pose)
    batch["right_robot_tcp_pose"] = quaternion_to_pose9d(right_pose)
    batch["left_robot_gripper_width"] = left_tcp_w_grip[:, 7:8]
    batch["right_robot_gripper_width"] = right_tcp_w_grip[:, 7:8]

    absolute_batch = {k: np.array(v, copy=True) for k, v in batch.items()}
    if use_relative_action:
        for key in ("left_robot_tcp_pose", "right_robot_tcp_pose"):
            base = batch[key][-1].copy()
            batch[key] = absolute_actions_to_relative_actions(
                batch[key],
                base_absolute_action=base,
                action_representation=action_representation,
            )

    model_obs = dict(batch)
    for key in (
        "left_robot_gripper_width",
        "right_robot_gripper_width",
        "left_robot_tcp_pose",
        "right_robot_tcp_pose",
    ):
        model_obs.pop(key, None)
    return model_obs, absolute_batch


def post_process_action(action_abs):
    if action_abs.shape[-1] != 20:
        raise ValueError(f"expected 20D dual-arm action, got {action_abs.shape}")
    left_rot = ortho6d_to_rotation_matrix(action_abs[:, 3:9])
    right_rot = ortho6d_to_rotation_matrix(action_abs[:, 12:18])
    left_quat = np.array([process_quaternion(t3d.quaternions.mat2quat(m), "f2l") for m in left_rot])
    right_quat = np.array([process_quaternion(t3d.quaternions.mat2quat(m), "f2l") for m in right_rot])
    left = np.concatenate([action_abs[:, :3], left_quat, action_abs[:, 18:19]], axis=1)
    right = np.concatenate([action_abs[:, 9:12], right_quat, action_abs[:, 19:20]], axis=1)
    return np.concatenate([left, right], axis=1)


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def step_metrics(action):
    arr = np.asarray(action, dtype=np.float64)
    diffs = np.diff(arr, axis=0)
    l2 = np.linalg.norm(diffs, axis=1)
    max_abs = np.max(np.abs(diffs), axis=1)
    max_l2_idx = int(np.argmax(l2))
    max_abs_idx = int(np.argmax(max_abs))
    return {
        "shape": list(arr.shape),
        "max_step_l2": float(l2[max_l2_idx]),
        "max_step_l2_index": max_l2_idx,
        "mean_step_l2": float(np.mean(l2)),
        "max_step_abs": float(max_abs[max_abs_idx]),
        "max_step_abs_index": max_abs_idx,
        "max_step_abs_dim": int(np.argmax(np.abs(diffs[max_abs_idx]))),
        "max_jump_before": arr[max_l2_idx].tolist(),
        "max_jump_after": arr[max_l2_idx + 1].tolist(),
    }


def quat_angle_metrics(final_action):
    arr = np.asarray(final_action, dtype=np.float64)
    out = {}
    for name, sl in (("left", slice(3, 7)), ("right", slice(11, 15))):
        q = arr[:, sl]
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
        dots = np.sum(q[:-1] * q[1:], axis=1)
        dots = np.clip(np.abs(dots), 0.0, 1.0)
        angles = 2.0 * np.arccos(dots)
        idx = int(np.argmax(angles))
        out[name] = {
            "max_angle_rad": float(angles[idx]),
            "max_angle_deg": float(np.degrees(angles[idx])),
            "max_angle_index": idx,
            "mean_angle_deg": float(np.degrees(np.mean(angles))),
        }
    return out


def output_delta_metrics(current, reference):
    if reference is None:
        return {}
    diff = np.asarray(current, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    return {
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "l2": float(np.linalg.norm(diff.reshape(-1))),
    }


def load_policy(ckpt_path, num_inference_steps, device):
    with open(ckpt_path, "rb") as f:
        payload = torch.load(f, map_location="cpu", pickle_module=dill)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=["optimizer", "lr_scheduler"], include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema and hasattr(workspace, "ema_model") else workspace.model
    policy.num_inference_steps = num_inference_steps
    policy.eval().to(device)
    action_representation = (
        cfg.task.dataset.action_representation
        if "action_representation" in cfg.task.dataset
        else "relative"
    )
    if action_representation == "only-y-train":
        action_representation = "only-y-inference"
    return policy, action_representation, cfg


def run_ckpt(name, ckpt_path, frozen_input, runs, seed, num_inference_steps, output_dir, device):
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
    out_path = output_dir / f"{name}_frozen_replay.jsonl"
    if out_path.exists():
        out_path.unlink()

    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "event": "meta",
            "ckpt_path": str(ckpt_path),
            "action_representation": action_representation,
            "num_inference_steps": num_inference_steps,
            "runs": runs,
            "seed": seed,
            "task_name": str(cfg.task.name),
        }, ensure_ascii=True) + "\n")

    summary = []
    for phase, phase_seed in (("random_denoise", None), ("fixed_seed_denoise", seed)):
        first_final = None
        prev_final = None
        for i in range(runs):
            if phase_seed is not None:
                set_all_seeds(phase_seed)
            start = time.perf_counter()
            with torch.no_grad():
                result = policy.predict_action(obs_tensor)
            elapsed = time.perf_counter() - start
            raw = result.get("action", result.get("action_pred"))[0].detach().cpu().numpy()
            abs_action = relative_actions_to_absolute_actions(
                raw,
                base_absolute_action=base_abs,
                action_representation=action_representation,
            )
            final = post_process_action(abs_action)
            payload = {
                "event": "frozen_policy_sampling",
                "phase": phase,
                "iteration": i,
                "elapsed_s": elapsed,
                "raw_metrics": step_metrics(raw),
                "absolute_metrics": step_metrics(abs_action),
                "final_metrics": step_metrics(final),
                "quat_angle_metrics": quat_angle_metrics(final),
                "delta_from_first_final": output_delta_metrics(final, first_final),
                "delta_from_previous_final": output_delta_metrics(final, prev_final),
            }
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=True) + "\n")
            first_final = final.copy() if first_final is None else first_final
            prev_final = final.copy()
            line = (
                f"{name} {phase} iter={i:02d} elapsed={elapsed:.3f}s "
                f"raw_l2={payload['raw_metrics']['max_step_l2']:.4f} "
                f"final_l2={payload['final_metrics']['max_step_l2']:.4f} "
                f"Ldeg={payload['quat_angle_metrics']['left']['max_angle_deg']:.2f} "
                f"Rdeg={payload['quat_angle_metrics']['right']['max_angle_deg']:.2f}"
            )
            print(line, flush=True)
            summary.append(payload)
    return out_path, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--ckpt", action="append", nargs=2, metavar=("NAME", "PATH"), required=True)
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen_input, obs_time = load_policy_input_npz(args.input)
    print(f"loaded frozen input obs_time={obs_time} keys={sorted(frozen_input)}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} cuda_count={torch.cuda.device_count()}", flush=True)

    all_summaries = {}
    for name, ckpt_path in args.ckpt:
        out_path, rows = run_ckpt(
            name=name,
            ckpt_path=Path(ckpt_path),
            frozen_input=frozen_input,
            runs=args.runs,
            seed=args.seed,
            num_inference_steps=args.num_inference_steps,
            output_dir=output_dir,
            device=device,
        )
        all_summaries[name] = {"jsonl": str(out_path), "rows": rows}

    compact = {}
    for name, data in all_summaries.items():
        compact[name] = {}
        for phase in ("random_denoise", "fixed_seed_denoise"):
            rows = [r for r in data["rows"] if r["phase"] == phase]
            compact[name][phase] = {
                "max_raw_step_l2": max(r["raw_metrics"]["max_step_l2"] for r in rows),
                "max_final_step_l2": max(r["final_metrics"]["max_step_l2"] for r in rows),
                "max_left_quat_angle_deg": max(r["quat_angle_metrics"]["left"]["max_angle_deg"] for r in rows),
                "max_right_quat_angle_deg": max(r["quat_angle_metrics"]["right"]["max_angle_deg"] for r in rows),
                "max_delta_from_first_final": max(
                    (r["delta_from_first_final"] or {"max_abs": 0.0})["max_abs"] for r in rows
                ),
            }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(compact, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(compact, indent=2, ensure_ascii=True), flush=True)
    print(f"summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
