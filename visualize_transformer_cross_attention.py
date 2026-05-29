import json
import pathlib
import pickle
from typing import Dict, List, Tuple

import cv2
import dill
import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from diffusion_policy.common.action_utils import (
    absolute_actions_to_relative_actions,
    get_inter_gripper_actions,
)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.workspace.base_workspace import BaseWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)


def _save_action_plot(gt: np.ndarray, pred: np.ndarray, path: pathlib.Path):
    num_dims = gt.shape[-1]
    fig, axes = plt.subplots(num_dims, 1, figsize=(8, 2 * num_dims), sharex=True)
    if num_dims == 1:
        axes = [axes]
    for dim in range(num_dims):
        axes[dim].plot(gt[:, dim], label="gt", linewidth=1)
        axes[dim].plot(pred[:, dim], label="pred", linewidth=1)
        axes[dim].set_ylabel(f"dim_{dim}")
        axes[dim].legend(loc="upper right")
    axes[-1].set_xlabel("timestep")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_obs_images(obs: dict, output_dir: pathlib.Path):
    for key, value in obs.items():
        if not torch.is_tensor(value) or value.ndim != 5:
            continue
        img = value[0, -1].detach().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = np.clip(img, 0.0, 1.0)
        plt.imsave(output_dir / f"{key}.png", img)


def _save_attention_plots(summary: dict, output_dir: pathlib.Path):
    aggregate = summary.get("aggregate_mean_by_key", {})
    if len(aggregate) == 0:
        return

    keys = list(aggregate.keys())
    values = [aggregate[key] for key in keys]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(keys, values)
    ax.set_ylabel("mean attention mass")
    ax.set_title("Cross-Attention Mean By Key")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_dir / "cross_attention_mean_by_key.png", dpi=150)
    plt.close(fig)

    per_step = summary.get("per_step", [])
    if len(per_step) == 0:
        return

    step_labels = [str(item["diffusion_timestep"]) for item in per_step]
    fig, ax = plt.subplots(figsize=(10, 4))
    for key in keys:
        series = [item["mean_attention_by_key"].get(key, 0.0) for item in per_step]
        ax.plot(step_labels, series, marker="o", linewidth=1, label=key)
    ax.set_xlabel("diffusion timestep")
    ax.set_ylabel("mean attention mass")
    ax.set_title("Cross-Attention By Diffusion Step")
    ax.legend(loc="best")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_dir / "cross_attention_by_step.png", dpi=150)
    plt.close(fig)



def _preprocess_video_frame(img: np.ndarray, is_wrist: bool, target_size: int = 224) -> np.ndarray:
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
            img_square = np.pad(img, ((0, 0), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=0)
        else:
            pad_top = (w - h) // 2
            pad_bottom = w - h - pad_top
            img_square = np.pad(img, ((pad_top, pad_bottom), (0, 0), (0, 0)), mode='constant', constant_values=0)
    img_resized = cv2.resize(img_square, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    return img_resized.astype(np.float32) / 255.0


def _load_video_frames(video_path: pathlib.Path) -> List[np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f'Failed to open video: {video_path}')
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if len(frames) == 0:
        raise RuntimeError(f'No frames loaded from video: {video_path}')
    return frames


def _find_video_path(video_dir: pathlib.Path, keywords: List[str]) -> pathlib.Path:
    candidates = []
    for path in sorted(video_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {'.mp4', '.mov', '.avi', '.mkv'}:
            continue
        lower_name = path.name.lower()
        if any(keyword in lower_name for keyword in keywords):
            candidates.append(path)
    if len(candidates) == 0:
        raise FileNotFoundError(f'No video found in {video_dir} for keywords {keywords}')
    return candidates[0]


def _load_video_episode_frames(video_dir: pathlib.Path, rgb_obs_keys: List[str]) -> Tuple[Dict[str, np.ndarray], dict]:
    video_dir = video_dir.expanduser().resolve()
    if not video_dir.is_dir():
        raise FileNotFoundError(f'Video directory not found: {video_dir}')

    source_files = dict()
    raw_frames = dict()
    for key in rgb_obs_keys:
        if key == 'left_eye_img':
            video_path = _find_video_path(video_dir, ['head', 'eye'])
            is_wrist = False
        elif key == 'left_wrist_img':
            video_path = _find_video_path(video_dir, ['wrist'])
            is_wrist = True
        else:
            raise KeyError(f'Unsupported video obs key: {key}')
        source_files[key] = str(video_path)
        raw_frames[key] = {
            'frames': _load_video_frames(video_path),
            'is_wrist': is_wrist,
        }

    num_frames = min(len(item['frames']) for item in raw_frames.values())
    processed = dict()
    for key, item in raw_frames.items():
        processed[key] = np.asarray([
            _preprocess_video_frame(frame, is_wrist=item['is_wrist'])
            for frame in item['frames'][:num_frames]
        ], dtype=np.float32)

    metadata = {
        'video_dir': str(video_dir),
        'source_files': source_files,
        'num_frames': int(num_frames),
    }
    return processed, metadata


def _build_video_obs_window(video_frames: Dict[str, np.ndarray], current_idx: int, n_obs_steps: int,
        rgb_obs_keys: List[str]) -> Dict[str, torch.Tensor]:
    obs = dict()
    for key in rgb_obs_keys:
        frames = video_frames[key]
        start_idx = current_idx - n_obs_steps + 1
        if start_idx < 0:
            pad = np.repeat(frames[[0]], repeats=(-start_idx), axis=0)
            window = np.concatenate([pad, frames[:current_idx + 1]], axis=0)
        else:
            window = frames[start_idx:current_idx + 1]
        if window.shape[0] < n_obs_steps:
            pad = np.repeat(window[[0]], repeats=(n_obs_steps - window.shape[0]), axis=0)
            window = np.concatenate([pad, window], axis=0)
        obs[key] = torch.from_numpy(np.moveaxis(window, -1, 1).astype(np.float32))
    return obs


def _save_video_summary(attention_records: List[dict], metadata: dict, output_dir: pathlib.Path):
    payload = {
        'video_dir': metadata['video_dir'],
        'source_files': metadata['source_files'],
        'num_frames': metadata['num_frames'],
        'num_timeline_samples': len(attention_records),
        'anchors': [
            {
                'frame_idx': int(record['entry']['current_rel_idx']),
                'progress': float(record['entry']['progress']),
                'aggregate_mean_by_key': record['summary']['aggregate_mean_by_key'],
            }
            for record in attention_records
        ],
    }
    with (output_dir / 'video_attention_summary.json').open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


def _run_video_attention_analysis(policy: BaseImagePolicy, device: torch.device,
        output_dir: pathlib.Path, video_dir: pathlib.Path, timeline_num_samples: int,
        contact_sheet_num_samples: int, n_obs_steps: int, rgb_obs_keys: List[str]):
    video_frames, metadata = _load_video_episode_frames(video_dir, rgb_obs_keys)
    total_frames = metadata['num_frames']
    entries = [
        {
            'sample_idx': frame_idx,
            'current_abs_idx': frame_idx,
            'current_rel_idx': frame_idx,
            'progress': frame_idx / max(total_frames - 1, 1),
        }
        for frame_idx in range(total_frames)
    ]

    timeline_entries = _select_evenly_spaced_entries(entries, timeline_num_samples)
    contact_entries = _select_evenly_spaced_entries(
        timeline_entries if len(timeline_entries) >= contact_sheet_num_samples else entries,
        contact_sheet_num_samples,
    )
    contact_lookup = {entry['sample_idx']: entry for entry in contact_entries}

    video_output_dir = output_dir / video_dir.name
    video_output_dir.mkdir(parents=True, exist_ok=True)

    attention_records = []
    cached_contact_samples = dict()
    for entry in timeline_entries:
        obs = _build_video_obs_window(
            video_frames=video_frames,
            current_idx=entry['current_rel_idx'],
            n_obs_steps=n_obs_steps,
            rgb_obs_keys=rgb_obs_keys,
        )
        obs_batch = dict_apply(obs, lambda x: x.unsqueeze(0).to(device, non_blocking=True))
        with torch.no_grad():
            result = policy.predict_action(obs_batch, return_attention=True)
        summary = result.get('cross_attention_summary')
        if summary is None:
            raise RuntimeError('cross_attention_summary is None. Model may not support attention capture.')
        attention_records.append({
            'entry': entry,
            'summary': summary,
        })
        if entry['sample_idx'] in contact_lookup:
            cached_contact_samples[entry['sample_idx']] = {
                'entry': entry,
                'obs': dict_apply(obs, lambda x: x.clone()),
            }

    ordered_contact_samples = [cached_contact_samples[entry['sample_idx']] for entry in contact_entries if entry['sample_idx'] in cached_contact_samples]
    _save_episode_attention_heatmap(attention_records, video_output_dir)
    _save_episode_contact_sheet(ordered_contact_samples, video_output_dir)
    _save_video_summary(attention_records, metadata, video_output_dir)

    print(f'Saved video-level cross-attention visualizations to: {video_output_dir}')
    print(f'  Total frames: {total_frames}')
    print(f'  Dense timeline samples: {len(timeline_entries)}')
    print(f'  Contact sheet samples: {len(ordered_contact_samples)}')

def _build_deterministic_dataset_item(dataset, idx: int) -> Dict[str, torch.Tensor]:
    data = dataset.sampler.sample_sequence(idx)
    T_slice = slice(dataset.n_obs_steps)

    obs_dict = dict()
    for key in dataset.rgb_keys:
        img = data[key][T_slice]
        img_normalized = np.moveaxis(img, -1, 1).astype(np.float32) / 255.0
        obs_dict[key] = img_normalized

    for key in dataset.lowdim_keys:
        if 'wrt' not in key:
            obs_dict[key] = data[key][:, :dataset.shape_meta['obs'][key]['shape'][0]][T_slice].astype(np.float32)

    obs_dict.update(get_inter_gripper_actions(obs_dict, dataset.lowdim_keys))
    for key in dataset.lowdim_keys:
        if 'wrt' in key:
            obs_dict[key] = obs_dict[key][:, :dataset.shape_meta['obs'][key]['shape'][0]][T_slice].astype(np.float32)

    action = data['action'][:, :dataset.shape_meta['action']['shape'][0]].astype(np.float32)
    if dataset.n_latency_steps > 0:
        action = action[dataset.n_latency_steps:]

    if dataset.relative_action:
        base_absolute_action = np.concatenate([
            dataset._get_base_obs_from_sample(data, obs_dict, 'left_robot_tcp_pose', T_slice),
            dataset._get_base_obs_from_sample(data, obs_dict, 'right_robot_tcp_pose', T_slice),
            data['left_robot_gripper_width'][:, :1][T_slice].astype(np.float32)[-1]
            if 'left_robot_gripper_width' in data else np.array([]),
            data['right_robot_gripper_width'][:, :1][T_slice].astype(np.float32)[-1]
            if 'right_robot_gripper_width' in data else np.array([]),
        ], axis=-1)
        extra_dim = 1
        if base_absolute_action.shape[-1] + extra_dim == action.shape[-1]:
            action[..., :-extra_dim] = absolute_actions_to_relative_actions(
                action[..., :-extra_dim],
                base_absolute_action=base_absolute_action,
                action_representation=dataset.action_representation,
            )
        else:
            action = absolute_actions_to_relative_actions(
                action,
                base_absolute_action=base_absolute_action,
                action_representation=dataset.action_representation,
            )

        if dataset.relative_tcp_obs_for_relative_action:
            for key in dataset.lowdim_keys:
                if 'tcp_pose' in key and 'wrt' not in key:
                    obs_dict[key] = absolute_actions_to_relative_actions(
                        obs_dict[key],
                        base_absolute_action=obs_dict[key][-1],
                        action_representation=dataset.action_representation,
                    )

    return {
        'obs': dict_apply(obs_dict, torch.from_numpy),
        'action': torch.from_numpy(action),
    }


def _get_episode_bounds(replay_buffer, episode_idx: int) -> Tuple[int, int]:
    episode_ends = replay_buffer.episode_ends[:]
    if episode_idx < 0 or episode_idx >= len(episode_ends):
        raise IndexError(f"episode_idx {episode_idx} out of range [0, {len(episode_ends) - 1}]")
    start_idx = 0 if episode_idx == 0 else int(episode_ends[episode_idx - 1])
    end_idx = int(episode_ends[episode_idx])
    return start_idx, end_idx


def _get_episode_timeline_entries(dataset, episode_idx: int) -> Tuple[int, int, List[dict]]:
    start_idx, end_idx = _get_episode_bounds(dataset.replay_buffer, episode_idx)
    n_obs_steps = int(dataset.n_obs_steps or 1)
    best_by_frame = dict()

    for sample_idx, row in enumerate(dataset.sampler.indices):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = [int(x) for x in row]
        current_offset = max(0, n_obs_steps - 1 - sample_start_idx)
        current_abs_idx = buffer_start_idx + current_offset
        if current_abs_idx < start_idx or current_abs_idx >= end_idx:
            continue
        candidate = {
            'sample_idx': sample_idx,
            'buffer_start_idx': buffer_start_idx,
            'buffer_end_idx': buffer_end_idx,
            'sample_start_idx': sample_start_idx,
            'sample_end_idx': sample_end_idx,
            'current_abs_idx': current_abs_idx,
        }
        prev = best_by_frame.get(current_abs_idx)
        if prev is None or sample_start_idx < prev['sample_start_idx']:
            best_by_frame[current_abs_idx] = candidate

    entries = [best_by_frame[idx] for idx in sorted(best_by_frame.keys())]
    episode_length = max(end_idx - start_idx, 1)
    for entry in entries:
        entry['current_rel_idx'] = entry['current_abs_idx'] - start_idx
        entry['progress'] = entry['current_rel_idx'] / max(episode_length - 1, 1)
    return start_idx, end_idx, entries


def _select_evenly_spaced_entries(entries: List[dict], count: int) -> List[dict]:
    if len(entries) == 0:
        return []
    count = max(1, min(int(count), len(entries)))
    if count >= len(entries):
        return list(entries)

    raw = np.linspace(0, len(entries) - 1, num=count)
    rounded = np.round(raw).astype(int).tolist()
    unique = []
    for idx in rounded:
        if idx not in unique:
            unique.append(idx)
    if len(unique) < count:
        for idx in range(len(entries)):
            if idx not in unique:
                unique.append(idx)
            if len(unique) >= count:
                break
    unique = sorted(unique[:count])
    return [entries[idx] for idx in unique]


def _obs_tensor_to_image(value: torch.Tensor) -> np.ndarray:
    arr = value.detach().cpu().numpy()
    if arr.ndim == 5:
        arr = arr[0, -1]
    elif arr.ndim == 4:
        arr = arr[-1]
    arr = np.transpose(arr, (1, 2, 0))
    return np.clip(arr, 0.0, 1.0)


def _save_episode_contact_sheet(contact_samples: List[dict], output_dir: pathlib.Path):
    if len(contact_samples) == 0:
        return

    rgb_keys = [
        key for key, value in contact_samples[0]['obs'].items()
        if torch.is_tensor(value) and value.ndim == 4
    ]
    if len(rgb_keys) == 0:
        return

    rows = len(rgb_keys)
    cols = len(contact_samples)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.0 * rows), squeeze=False)

    for row_idx, key in enumerate(rgb_keys):
        for col_idx, sample in enumerate(contact_samples):
            ax = axes[row_idx][col_idx]
            ax.imshow(_obs_tensor_to_image(sample['obs'][key]))
            ax.axis('off')
            if row_idx == 0:
                entry = sample['entry']
                ax.set_title(f"t={entry['current_rel_idx']}\n{entry['progress']:.0%}", fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(key, fontsize=10)

    fig.tight_layout()
    fig.savefig(output_dir / 'episode_obs_contact_sheet.png', dpi=150)
    plt.close(fig)


def _save_episode_attention_heatmap(attention_records: List[dict], output_dir: pathlib.Path):
    if len(attention_records) == 0:
        return

    keys = list(attention_records[0]['summary']['aggregate_mean_by_key'].keys())
    x_values = [record['entry']['current_rel_idx'] for record in attention_records]
    matrix = np.asarray([
        [record['summary']['aggregate_mean_by_key'].get(key, 0.0) for record in attention_records]
        for key in keys
    ], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(max(10, len(attention_records) * 0.6), 1.8 + 1.0 * len(keys)))
    im = ax.imshow(matrix, aspect='auto', cmap='viridis')
    ax.set_yticks(np.arange(len(keys)))
    ax.set_yticklabels(keys)
    ax.set_xticks(np.arange(len(x_values)))
    ax.set_xticklabels([str(x) for x in x_values], rotation=45, ha='right')
    ax.set_xlabel('episode timestep')
    ax.set_title('Cross-Attention Aggregate Over Episode Timeline')
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label='attention mass')
    if len(attention_records) <= 24:
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                ax.text(col_idx, row_idx, f"{matrix[row_idx, col_idx]:.2f}",
                        ha='center', va='center', color='white', fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / 'episode_attention_heatmap.png', dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(10, len(attention_records) * 0.6), 4.5))
    for row_idx, key in enumerate(keys):
        ax.plot(x_values, matrix[row_idx], marker='o', linewidth=1.5, label=key)
    ax.set_xlabel('episode timestep')
    ax.set_ylabel('mean attention mass')
    ax.set_title('Cross-Attention Aggregate Over Episode Timeline')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / 'episode_attention_lines.png', dpi=150)
    plt.close(fig)


def _save_episode_trajectory(replay_buffer, start_idx: int, end_idx: int,
        anchor_entries: List[dict], output_dir: pathlib.Path):
    if 'left_robot_tcp_pose' not in replay_buffer:
        return

    poses = replay_buffer['left_robot_tcp_pose'][start_idx:end_idx, :3]
    if len(poses) == 0:
        return

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(poses[:, 0], poses[:, 1], poses[:, 2], linewidth=1.5, color='tab:blue', label='trajectory')
    ax.scatter(poses[0, 0], poses[0, 1], poses[0, 2], color='tab:green', s=40, label='start')
    ax.scatter(poses[-1, 0], poses[-1, 1], poses[-1, 2], color='tab:red', s=40, label='end')

    for rank, entry in enumerate(anchor_entries, start=1):
        rel_idx = entry['current_rel_idx']
        point = poses[rel_idx]
        ax.scatter(point[0], point[1], point[2], color='tab:orange', s=50)
        ax.text(point[0], point[1], point[2], str(rank), fontsize=9)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Episode Left TCP Trajectory')
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(output_dir / 'episode_left_tcp_trajectory.png', dpi=150)
    plt.close(fig)


def _save_episode_summary(attention_records: List[dict], start_idx: int, end_idx: int,
        output_dir: pathlib.Path):
    payload = {
        'episode_start_idx': int(start_idx),
        'episode_end_idx': int(end_idx),
        'episode_length': int(end_idx - start_idx),
        'num_timeline_samples': len(attention_records),
        'anchors': [
            {
                'sample_idx': int(record['entry']['sample_idx']),
                'current_abs_idx': int(record['entry']['current_abs_idx']),
                'current_rel_idx': int(record['entry']['current_rel_idx']),
                'progress': float(record['entry']['progress']),
                'aggregate_mean_by_key': record['summary']['aggregate_mean_by_key'],
            }
            for record in attention_records
        ],
    }
    with (output_dir / 'episode_attention_summary.json').open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


def _run_episode_attention_analysis(policy: BaseImagePolicy, dataset, device: torch.device,
        output_dir: pathlib.Path, episode_idx: int, timeline_num_samples: int,
        contact_sheet_num_samples: int):
    start_idx, end_idx, entries = _get_episode_timeline_entries(dataset, episode_idx)
    if len(entries) == 0:
        raise RuntimeError(f'No sampler entries found for episode {episode_idx}.')

    timeline_entries = _select_evenly_spaced_entries(entries, timeline_num_samples)
    contact_entries = _select_evenly_spaced_entries(
        timeline_entries if len(timeline_entries) >= contact_sheet_num_samples else entries,
        contact_sheet_num_samples,
    )
    contact_lookup = {entry['sample_idx']: entry for entry in contact_entries}

    episode_dir = output_dir / f'episode_{episode_idx:03d}'
    episode_dir.mkdir(parents=True, exist_ok=True)

    attention_records = []
    cached_contact_samples = dict()

    for entry in timeline_entries:
        sample = _build_deterministic_dataset_item(dataset, entry['sample_idx'])
        obs = sample['obs']
        obs_batch = dict_apply(obs, lambda x: x.unsqueeze(0).to(device, non_blocking=True))

        with torch.no_grad():
            result = policy.predict_action(obs_batch, return_attention=True)

        summary = result.get('cross_attention_summary')
        if summary is None:
            raise RuntimeError('cross_attention_summary is None. Model may not support attention capture.')
        attention_records.append({
            'entry': entry,
            'summary': summary,
        })

        if entry['sample_idx'] in contact_lookup:
            cached_contact_samples[entry['sample_idx']] = {
                'entry': entry,
                'obs': dict_apply(obs, lambda x: x.clone()),
            }

    ordered_contact_samples = [cached_contact_samples[entry['sample_idx']] for entry in contact_entries if entry['sample_idx'] in cached_contact_samples]
    _save_episode_attention_heatmap(attention_records, episode_dir)
    _save_episode_contact_sheet(ordered_contact_samples, episode_dir)
    _save_episode_trajectory(dataset.replay_buffer, start_idx, end_idx, contact_entries, episode_dir)
    _save_episode_summary(attention_records, start_idx, end_idx, episode_dir)

    print(f'Saved episode-level cross-attention visualizations to: {episode_dir}')
    print(f'  Episode length: {end_idx - start_idx}')
    print(f'  Dense timeline samples: {len(timeline_entries)}')
    print(f'  Contact sheet samples: {len(ordered_contact_samples)}')


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("diffusion_policy", "config")),
    config_name="train_diffusion_transformer_timm_q3_place_cup_single_frame_workspace",
)
def main(cfg: OmegaConf):
    if "ckpt_path" not in cfg or not cfg.ckpt_path:
        raise ValueError("ckpt_path is required. Example: +ckpt_path=path/to/checkpoints/XXXX.ckpt")

    ckpt_path = pathlib.Path(cfg.ckpt_path).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    payload = torch.load(ckpt_path.open("rb"), pickle_module=dill, map_location="cpu")
    runtime_cfg = OmegaConf.create(payload["cfg"])
    OmegaConf.set_struct(runtime_cfg, False)
    if "eval" in cfg:
        runtime_cfg.eval = cfg.eval

    cls = hydra.utils.get_class(runtime_cfg._target_)
    workspace: BaseWorkspace = cls(runtime_cfg)
    state_dicts = payload.get("state_dicts", {})
    if "model" in state_dicts:
        workspace.model.load_state_dict(state_dicts["model"])
    if "ema_model" in state_dicts and getattr(workspace, "ema_model", None) is not None:
        workspace.ema_model.load_state_dict(state_dicts["ema_model"])

    run_dir = ckpt_path.parent.parent
    normalizer_path = run_dir / "normalizer.pkl"
    if normalizer_path.is_file():
        normalizer = pickle.load(normalizer_path.open("rb"))
        if hasattr(workspace.model, "set_normalizer"):
            workspace.model.set_normalizer(normalizer)
        if getattr(workspace, "ema_model", None) is not None and hasattr(workspace.ema_model, "set_normalizer"):
            workspace.ema_model.set_normalizer(normalizer)

    policy: BaseImagePolicy = workspace.model
    if runtime_cfg.training.use_ema and hasattr(workspace, "ema_model") and workspace.ema_model is not None:
        policy = workspace.ema_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.eval().to(device)

    dataset = hydra.utils.instantiate(runtime_cfg.task.dataset)
    eval_cfg = runtime_cfg.get("eval", {})
    split = str(eval_cfg.get("split", "train")).lower()
    if split == "val":
        dataset = dataset.get_validation_dataset()
    elif split != "train":
        raise ValueError(f"Unsupported eval split: {split}")

    output_dir = pathlib.Path(eval_cfg.get("output_dir", "output_images/transformer_cross_attention"))
    output_dir.mkdir(parents=True, exist_ok=True)

    mode = str(eval_cfg.get("mode", "sample")).lower()
    if mode == 'episode':
        episode_idx = int(eval_cfg.get('episode_idx', 0))
        timeline_num_samples = int(eval_cfg.get('timeline_num_samples', 21))
        contact_sheet_num_samples = int(eval_cfg.get('contact_sheet_num_samples', 5))
        _run_episode_attention_analysis(
            policy=policy,
            dataset=dataset,
            device=device,
            output_dir=output_dir,
            episode_idx=episode_idx,
            timeline_num_samples=timeline_num_samples,
            contact_sheet_num_samples=contact_sheet_num_samples,
        )
        return
    if mode == 'video_episode':
        video_dir = pathlib.Path(eval_cfg.get('video_dir', '')).expanduser().resolve()
        if not str(video_dir):
            raise ValueError('eval.video_dir is required for mode=video_episode')
        timeline_num_samples = int(eval_cfg.get('timeline_num_samples', 31))
        contact_sheet_num_samples = int(eval_cfg.get('contact_sheet_num_samples', 5))
        shape_meta = OmegaConf.to_container(runtime_cfg.shape_meta, resolve=True)
        rgb_obs_keys = [
            key for key, attr in shape_meta['obs'].items()
            if attr.get('type', 'low_dim') == 'rgb'
        ]
        _run_video_attention_analysis(
            policy=policy,
            device=device,
            output_dir=output_dir,
            video_dir=video_dir,
            timeline_num_samples=timeline_num_samples,
            contact_sheet_num_samples=contact_sheet_num_samples,
            n_obs_steps=int(runtime_cfg.n_obs_steps),
            rgb_obs_keys=rgb_obs_keys,
        )
        return

    num_samples = int(eval_cfg.get("num_samples", 10))
    start_index = int(eval_cfg.get("start_index", 0))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    saved = 0
    for sample_idx, batch in enumerate(dataloader):
        if sample_idx < start_index:
            continue
        if saved >= num_samples:
            break

        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
        obs = batch["obs"]
        gt_action = batch["action"]

        with torch.no_grad():
            result = policy.predict_action(obs, return_attention=True)

        pred_action = result["action_pred"]
        attention_summary = result.get("cross_attention_summary")

        sample_dir = output_dir / f"sample_{sample_idx:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        gt = gt_action[0].detach().cpu().numpy()
        pred = pred_action[0].detach().cpu().numpy()
        _save_action_plot(gt, pred, sample_dir / "actions.png")
        _save_obs_images(obs, sample_dir)

        if attention_summary is not None:
            with (sample_dir / "cross_attention_summary.json").open("w", encoding="utf-8") as f:
                json.dump(attention_summary, f, indent=2)
            _save_attention_plots(attention_summary, sample_dir)

        saved += 1

    print(f"Saved transformer cross-attention visualizations to: {output_dir}")


if __name__ == "__main__":
    main()