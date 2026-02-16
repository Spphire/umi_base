"""
Pick-and-place sign accuracy evaluation (single hand, single head)

- Read each episode from a zarr replay buffer
- Detect the gripper-closing window (width decreasing to minimum)
- Compare the sign of action X between policy prediction and GT within that window
"""
import os
import pathlib
import pickle
import dill
import argparse
import csv
from typing import Optional, Tuple, List

import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import hydra

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.action_utils import (
    absolute_actions_to_relative_actions,
    get_inter_gripper_actions,
)

# Avoid OpenMP duplicate load issue (Windows common, harmless on macOS)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def load_policy(ckpt_path: str, cfg_yaml_path: Optional[str] = None):
    """Load policy checkpoint (workspace-based)."""
    ckpt_path = pathlib.Path(ckpt_path).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading policy from: {ckpt_path}")
    payload = torch.load(ckpt_path.open("rb"), pickle_module=dill, map_location="cpu")

    if cfg_yaml_path is not None:
        cfg_yaml_path = pathlib.Path(cfg_yaml_path).expanduser().resolve()
        if not cfg_yaml_path.is_file():
            raise FileNotFoundError(f"Config yaml not found: {cfg_yaml_path}")
        yaml_cfg = OmegaConf.load(str(cfg_yaml_path))
        base_cfg = OmegaConf.create(payload["cfg"])
        OmegaConf.set_struct(base_cfg, False)
        cfg = OmegaConf.merge(base_cfg, yaml_cfg)
        print(f"  Using cfg merged from yaml: {cfg_yaml_path}")
    else:
        cfg = payload["cfg"]

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.model
    if cfg.training.use_ema and hasattr(workspace, "ema_model"):
        policy = workspace.ema_model
        print("  Using EMA model")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.eval().to(device)
    print(f"  Policy loaded on device: {device}")
    return policy, cfg, device


def load_replay_buffer(dataset_path: str):
    dataset_path = pathlib.Path(dataset_path).expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    print(f"Loading ReplayBuffer from: {dataset_path}")
    replay_buffer = ReplayBuffer.create_from_path(str(dataset_path), mode="r")
    print(f"  Loaded! Total episodes: {replay_buffer.n_episodes}")
    return replay_buffer


def preprocess_image(img, target_size=224, is_wrist=True):
    """Center-crop (wrist) or pad (eye) then resize and normalize to [0, 1]."""
    h, w = img.shape[:2]
    if is_wrist:
        if h > w:
            start = (h - w) // 2
            img_square = img[start:start + w, :]
        else:
            start = (w - h) // 2
            img_square = img[:, start:start + h]
    else:
        if h > w:
            pad_left = (h - w) // 2
            pad_right = h - w - pad_left
            img_square = np.pad(img, ((0, 0), (pad_left, pad_right), (0, 0)), mode="constant", constant_values=0)
        else:
            pad_top = (w - h) // 2
            pad_bottom = w - h - pad_top
            img_square = np.pad(img, ((pad_top, pad_bottom), (0, 0), (0, 0)), mode="constant", constant_values=0)

    img_resized = cv2.resize(img_square, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    return img_resized.astype(np.float32) / 255.0

def select_gripper_width_key(episode_data: dict, expected_keys: Optional[set] = None) -> str:
    candidates = [k for k in episode_data.keys() if "gripper_width" in k.lower()]
    if expected_keys:
        filtered = [k for k in candidates if k in expected_keys]
        if filtered:
            candidates = filtered
    candidates = [k for k in candidates if "right" not in k.lower()]
    if not candidates:
        raise KeyError("No gripper_width key found in episode data")
    if "left_robot_gripper_width" in candidates:
        return "left_robot_gripper_width"
    return candidates[0]


def find_closing_window(
    gripper_width: np.ndarray,
    close_eps: float = 1e-4,
    min_len: int = 5,
    smooth_window: int = 5,
    min_drop: float = 0.0,
) -> Optional[Tuple[int, int]]:
    w = np.asarray(gripper_width).squeeze()
    if w.ndim != 1 or len(w) < 2:
        return None

    if smooth_window > 1:
        kernel = np.ones(smooth_window, dtype=np.float32) / float(smooth_window)
        w_smooth = np.convolve(w, kernel, mode="same")
    else:
        w_smooth = w

    dw = np.diff(w_smooth)
    dec = dw < -close_eps

    segments: List[Tuple[int, int]] = []
    start = None
    for i, flag in enumerate(dec):
        if flag and start is None:
            start = i
        if (not flag) and start is not None:
            segments.append((start, i - 1))
            start = None
    if start is not None:
        segments.append((start, len(dec) - 1))

    if not segments:
        return None

    # filter by minimum drop
    filtered_segments: List[Tuple[int, int, float]] = []
    for s, e in segments:
        end_idx = min(e + 1, len(w_smooth) - 1)
        drop = float(w_smooth[s] - w_smooth[end_idx])
        if drop >= min_drop:
            filtered_segments.append((s, e, drop))

    if not filtered_segments:
        return None

    min_idx = int(np.argmin(w_smooth))
    target = max(0, min_idx - 1)

    def segment_score(seg):
        s, e, drop = seg
        if s <= target <= e:
            return (0, -drop, abs(e - target))
        return (1, -drop, abs(e - target))

    filtered_segments.sort(key=segment_score)
    seg_start, seg_end, _ = filtered_segments[0]

    window_start = seg_start
    window_end = seg_end + 1
    window_end = min(window_end, len(w) - 1)

    if window_end - window_start + 1 < min_len:
        window_end = min(len(w) - 1, max(window_end, min_idx))
        window_start = max(0, window_end - min_len + 1)

    return window_start, window_end


def build_obs_window(
    episode_data: dict,
    t: int,
    n_obs_steps: int,
    img_keys: List[str],
    lowdim_keys: List[str],
    action_representation: str,
    relative_tcp_obs_for_relative_action: bool,
):
    episode_length = len(episode_data["action"])
    obs_imgs = {}
    obs_lowdim = {}

    for obs_t in range(t, min(t + n_obs_steps, episode_length)):
        for img_key in img_keys:
            img = episode_data[img_key][obs_t]
            is_wrist = "wrist" in img_key.lower()
            img_processed = preprocess_image(img, target_size=224, is_wrist=is_wrist)
            img_tensor = torch.from_numpy(img_processed).permute(2, 0, 1).float()
            obs_imgs.setdefault(img_key, []).append(img_tensor)

        for key in lowdim_keys:
            if key in episode_data and "wrt" not in key:
                val = np.asarray(episode_data[key][obs_t])
                if val.ndim == 0:
                    val = val.reshape(1)
                obs_lowdim.setdefault(key, []).append(torch.from_numpy(val.astype(np.float32)))

    if not obs_imgs and not obs_lowdim:
        return None

    obs = {k: torch.stack(v, dim=0) for k, v in obs_imgs.items()}
    for key, vals in obs_lowdim.items():
        obs[key] = torch.stack(vals, dim=0)

    # pad to n_obs_steps by repeating last frame
    for key in list(obs.keys()):
        if len(obs[key]) < n_obs_steps:
            obs[key] = torch.cat([obs[key], obs[key][-1:].repeat(n_obs_steps - len(obs[key]), *([1] * (obs[key].ndim - 1)))], dim=0)

    # relative tcp obs if needed
    if relative_tcp_obs_for_relative_action:
        for key in list(obs.keys()):
            if ("robot_tcp_pose" in key) and ("wrt" not in key):
                abs_seq = obs[key].cpu().numpy()
                rel_seq = absolute_actions_to_relative_actions(
                    abs_seq,
                    base_absolute_action=abs_seq[-1],
                    action_representation=action_representation,
                )
                obs[key] = torch.from_numpy(rel_seq.astype(np.float32))

    if len(lowdim_keys) > 0:
        obs_np = {k: v.cpu().numpy() for k, v in obs.items() if k in lowdim_keys}
        inter = get_inter_gripper_actions(obs_np, lowdim_keys)
        for k, v in inter.items():
            if k in lowdim_keys:
                obs[k] = torch.from_numpy(v.astype(np.float32))

    return obs


def predict_actions_for_episode(
    policy,
    episode_data: dict,
    device: torch.device,
    n_obs_steps: int,
    action_representation: str,
    relative_tcp_obs_for_relative_action: bool,
) -> np.ndarray:
    episode_length = len(episode_data["action"])

    img_keys = [k for k in episode_data.keys() if "img" in k.lower()]
    expected_keys = None
    if hasattr(policy, "normalizer") and hasattr(policy.normalizer, "params_dict"):
        expected_keys = set(policy.normalizer.params_dict.keys())
        img_keys = [k for k in img_keys if k in expected_keys]

    lowdim_keys = []
    if expected_keys is not None:
        lowdim_keys = [k for k in expected_keys if ("img" not in k) and (k != "action")]

    n_action_steps = getattr(policy, "n_action_steps", 8)

    pred_actions = []
    t = 0
    while t < episode_length:
        obs = build_obs_window(
            episode_data,
            t,
            n_obs_steps,
            img_keys,
            lowdim_keys,
            action_representation,
            relative_tcp_obs_for_relative_action,
        )
        if obs is None:
            break

        obs = dict_apply(obs, lambda x: x.unsqueeze(0).to(device))
        with torch.no_grad():
            result = policy.predict_action(obs)
        pred_full = result["action_pred"][0].detach().cpu().numpy()
        pred_seq = pred_full[:n_action_steps]

        for i in range(len(pred_seq)):
            if t + i >= episode_length:
                break
            pred_actions.append(pred_seq[i])
        t += n_action_steps

    return np.array(pred_actions)


def predict_actions_chunk_at_index(
    policy,
    episode_data: dict,
    device: torch.device,
    n_obs_steps: int,
    action_representation: str,
    relative_tcp_obs_for_relative_action: bool,
    idx: int,
) -> np.ndarray:
    img_keys = [k for k in episode_data.keys() if "img" in k.lower()]
    expected_keys = None
    if hasattr(policy, "normalizer") and hasattr(policy.normalizer, "params_dict"):
        expected_keys = set(policy.normalizer.params_dict.keys())
        img_keys = [k for k in img_keys if k in expected_keys]

    lowdim_keys = []
    if expected_keys is not None:
        lowdim_keys = [k for k in expected_keys if ("img" not in k) and (k != "action")]

    n_action_steps = getattr(policy, "n_action_steps", 8)
    obs = build_obs_window(
        episode_data,
        idx,
        n_obs_steps,
        img_keys,
        lowdim_keys,
        action_representation,
        relative_tcp_obs_for_relative_action,
    )
    if obs is None:
        return np.empty((0,))

    obs = dict_apply(obs, lambda x: x.unsqueeze(0).to(device))
    with torch.no_grad():
        result = policy.predict_action(obs)
    pred_full = result["action_pred"][0].detach().cpu().numpy()
    pred_seq = pred_full[:n_action_steps]
    return pred_seq


def compute_sign_accuracy(
    pred_chunk: np.ndarray,
    gt_actions: np.ndarray,
    idx: int,
    zero_eps: float = 1e-6,
    relative: bool = False,
) -> Tuple[float, int, int]:
    if pred_chunk is None or len(pred_chunk) == 0:
        return float("nan"), 0, 0

    chunk_end = min(idx + len(pred_chunk) - 1, len(gt_actions) - 1)
    pred_x = pred_chunk[: chunk_end - idx + 1, 0]
    gt_x = gt_actions[idx:chunk_end + 1, 0]
    if relative:
        gt_x -= gt_actions[idx - 1, 0]

    valid = (np.abs(gt_x) > zero_eps) & (np.abs(pred_x) > zero_eps)
    if valid.sum() == 0:
        return float("nan"), 0, 0

    correct = np.sign(pred_x[valid]) == np.sign(gt_x[valid])
    acc = float(correct.mean())
    return acc, int(correct.sum()), int(valid.sum())


def plot_debug_episode(
    episode_idx: int,
    gripper_width: np.ndarray,
    window: Tuple[int, int],
    pred_actions: np.ndarray,
    gt_actions: np.ndarray,
    zero_eps: float,
    chunk_len: int,
    plot_relative: bool,
    output_dir: str,
):
    w = np.asarray(gripper_width).squeeze()
    start, end = window
    end = min(end, len(pred_actions) - 1, len(gt_actions) - 1)
    start = min(start, end)
    idx = end
    chunk_end = min(idx + max(1, int(chunk_len)) - 1, len(pred_actions) - 1, len(gt_actions) - 1)

    pred_x = pred_actions[:, 0]
    gt_x = gt_actions[:, 0]

    # if plot_relative:
    #     pred_x = np.concatenate([[0.0], pred_x[1:] - pred_x[:-1]])
        #gt_x = np.concatenate([[0.0], gt_x[1:] - gt_x[:-1]])

    valid = (np.abs(gt_x) > zero_eps) & (np.abs(pred_x) > zero_eps)
    same = (np.sign(pred_x) == np.sign(gt_x)) & valid

    out_dir = pathlib.Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(18, 10))

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(w, color="black", linewidth=2, label="gripper_width")
    ax1.axvspan(start, end, color="orange", alpha=0.2, label="closing_window")
    ax1.axvline(idx, color="blue", linestyle=":", linewidth=2, label="test_frame")
    ax1.axvspan(idx, chunk_end, color="cyan", alpha=0.15, label="test_chunk")
    ax1.set_title(f"Episode {episode_idx} - Gripper Width with Closing Window")
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Width")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(2, 1, 2)
    t = np.arange(len(gt_x))
    ax2.plot(t, gt_x, color="green", linewidth=2, label="GT X")
    ax2.plot(t, pred_x, color="red", linewidth=2, linestyle="--", label="Pred X")
    ax2.axvspan(start, end, color="orange", alpha=0.2, label="closing_window")
    ax2.axvline(idx, color="blue", linestyle=":", linewidth=2, label="test_frame")
    ax2.axvspan(idx, chunk_end, color="cyan", alpha=0.15, label="test_chunk")

    if len(same) == len(t):
        good_idx = np.where(same)[0]
        bad_idx = np.where(valid & ~same)[0]
        ax2.scatter(good_idx, pred_x[good_idx], color="blue", s=20, label="sign match")
        ax2.scatter(bad_idx, pred_x[bad_idx], color="purple", s=20, label="sign mismatch")

    ax2.set_title("Action X Sign Consistency" + (" (Relative)" if plot_relative else ""))
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Action X")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    save_path = out_dir / f"episode_{episode_idx}_debug.png"
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved debug plot to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Pick-and-place sign accuracy evaluation")
    parser.add_argument("--ckpt", default='/mnt/data/shenyibo/workspace/umi_base/data/outputs/2026.02.15/10.03.41_train_diffusion_unet_timm_q3_mouse_197/checkpoints/latest.ckpt', help="Path to policy checkpoint")
    parser.add_argument("--cfg", default=None, help="Optional cfg yaml path")
    parser.add_argument("--dataset", default='/mnt/data/shenyibo/workspace/umi_base/.cache/q3_mouse_dh/replay_buffer.zarr', help="Path to replay_buffer.zarr")
    parser.add_argument("--n-obs-steps", type=int, default=1, help="Observation history length")
    parser.add_argument("--start-episode", type=int, default=0, help="Start episode index (inclusive)")
    parser.add_argument("--end-episode", type=int, default=None, help="End episode index (exclusive)")
    parser.add_argument("--close-eps", type=float, default=1e-4, help="Threshold for detecting closing slope")
    parser.add_argument("--min-len", type=int, default=5, help="Minimum closing window length")
    parser.add_argument("--smooth-window", type=int, default=5, help="Smoothing window for gripper width")
    parser.add_argument("--min-drop", type=float, default=0.0, help="Minimum drop in gripper width to accept a closing segment")
    parser.add_argument("--zero-eps", type=float, default=1e-6, help="Ignore near-zero actions")
    parser.add_argument("--output-csv", default=None, help="Optional output CSV path")
    parser.add_argument("--debug-episode", type=int, default=93, help="If set, only run this episode and save debug plot")
    parser.add_argument("--debug-dir", default="./debug_sign_accuracy", help="Output directory for debug plots")
    args = parser.parse_args()

    policy, cfg, device = load_policy(args.ckpt, cfg_yaml_path=args.cfg)
    replay_buffer = load_replay_buffer(args.dataset)

    action_representation = "relative"
    relative_tcp_obs_for_relative_action = True
    if cfg is not None and hasattr(cfg, "task") and hasattr(cfg.task, "dataset"):
        ds_cfg = cfg.task.dataset
        if hasattr(ds_cfg, "action_representation"):
            action_representation = ds_cfg.action_representation
        if hasattr(ds_cfg, "relative_tcp_obs_for_relative_action"):
            relative_tcp_obs_for_relative_action = ds_cfg.relative_tcp_obs_for_relative_action

    expected_keys = None
    if hasattr(policy, "normalizer") and hasattr(policy.normalizer, "params_dict"):
        expected_keys = set(policy.normalizer.params_dict.keys())

    total_correct = 0
    total_valid = 0
    episode_results = []

    if args.debug_episode is not None:
        start_ep = max(0, args.debug_episode)
        end_ep = min(start_ep + 1, replay_buffer.n_episodes)
    else:
        start_ep = max(0, args.start_episode)
        end_ep = replay_buffer.n_episodes if args.end_episode is None else min(args.end_episode, replay_buffer.n_episodes)

    for ep_idx in range(start_ep, end_ep):
        episode_data = replay_buffer.get_episode(ep_idx)
        gripper_key = select_gripper_width_key(episode_data, expected_keys)
        gripper_width = episode_data[gripper_key]

        window = find_closing_window(
            gripper_width,
            close_eps=args.close_eps,
            min_len=args.min_len,
            smooth_window=args.smooth_window,
            min_drop=args.min_drop,
        )
        if window is None:
            print(f"Episode {ep_idx}: no closing window found, skipping")
            episode_results.append((ep_idx, None, None, float("nan"), 0, 0))
            continue

        gt_actions = np.asarray(episode_data["action"])

        idx = min(window[1], len(gt_actions) - 1)

        if args.debug_episode is not None:
            pred_actions = predict_actions_for_episode(
                policy,
                episode_data,
                device,
                n_obs_steps=args.n_obs_steps,
                action_representation=action_representation,
                relative_tcp_obs_for_relative_action=relative_tcp_obs_for_relative_action,
            )
            pred_chunk = pred_actions[idx:idx + getattr(policy, "n_action_steps", 8)]
        else:
            pred_chunk = predict_actions_chunk_at_index(
                policy,
                episode_data,
                device,
                n_obs_steps=args.n_obs_steps,
                action_representation=action_representation,
                relative_tcp_obs_for_relative_action=relative_tcp_obs_for_relative_action,
                idx=idx,
            )

        acc, correct, valid = compute_sign_accuracy(
            pred_chunk,
            gt_actions,
            idx,
            zero_eps=args.zero_eps,
            relative=action_representation == "relative"
        )

        total_correct += correct
        total_valid += valid

        chunk_len = len(pred_chunk) if pred_chunk is not None else 0
        chunk_end = min(idx + max(1, int(chunk_len)) - 1, len(gt_actions) - 1)
        print(
            f"Episode {ep_idx}: window={window}, test_chunk=({idx},{chunk_end}), "
            f"valid={valid}, correct={correct}, acc={acc if not np.isnan(acc) else 'nan'}"
        )
        episode_results.append((ep_idx, window[0], window[1], acc, correct, valid))

        if args.debug_episode is not None:
            plot_debug_episode(
                ep_idx,
                gripper_width,
                window,
                pred_actions,
                gt_actions,
                args.zero_eps,
                max(1, int(chunk_len)) if chunk_len > 0 else getattr(policy, "n_action_steps", 8),
                action_representation == "relative",
                args.debug_dir,
            )
            break

    overall_acc = float(total_correct / total_valid) if total_valid > 0 else float("nan")
    print("=" * 60)
    print(f"Overall sign accuracy: {overall_acc if not np.isnan(overall_acc) else 'nan'}")
    print(f"Total valid points: {total_valid}")

    if args.output_csv:
        output_path = pathlib.Path(args.output_csv).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "window_start", "window_end", "acc", "correct", "valid"])
            for row in episode_results:
                writer.writerow(row)
        print(f"Saved CSV to: {output_path}")


if __name__ == "__main__":
    main()
