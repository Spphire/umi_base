"""
可视化policy预测轨迹 vs GT轨迹
- 从zarr选定一个episode
- policy推理输出相对action，累加形成轨迹
- 与GT绝对轨迹对比（起点对齐）
"""
import os
import pathlib
import pickle
import argparse
import json
try:
    import dill
except ImportError:
    dill = pickle
import numpy as np
import torch
import cv2
from omegaconf import OmegaConf
import hydra

# 临时规避OpenMP重复加载问题（Windows常见）
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# 设置matplotlib非交互式backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.action_utils import (
    absolute_actions_to_relative_actions,
    relative_actions_to_absolute_actions,
    get_inter_gripper_actions
)


def load_policy(ckpt_path: str, cfg_yaml_path: str | None = None):
    """加载policy checkpoint（使用workspace）
    cfg_yaml_path: 可选，使用yaml中的cfg替代checkpoint中的cfg。
    默认优先使用 checkpoint 同目录下的 .hydra/config.yaml。
    """
    ckpt_path = pathlib.Path(ckpt_path).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    run_dir = ckpt_path.parent.parent
    print(f"Loading policy from: {ckpt_path}")

    payload = torch.load(ckpt_path.open("rb"), pickle_module=dill, map_location='cpu')

    auto_cfg_yaml_path = run_dir / '.hydra' / 'config.yaml'
    if cfg_yaml_path is None and auto_cfg_yaml_path.is_file():
        cfg_yaml_path = str(auto_cfg_yaml_path)

    if cfg_yaml_path is not None:
        cfg_yaml_path = pathlib.Path(cfg_yaml_path).expanduser().resolve()
        if not cfg_yaml_path.is_file():
            raise FileNotFoundError(f"Config yaml not found: {cfg_yaml_path}")
        yaml_cfg = OmegaConf.load(str(cfg_yaml_path))
        base_cfg = OmegaConf.create(payload['cfg'])
        OmegaConf.set_struct(base_cfg, False)
        cfg = OmegaConf.merge(base_cfg, yaml_cfg)
        print(f"  Using cfg merged from yaml: {cfg_yaml_path}")
    else:
        cfg = payload['cfg']

    OmegaConf.set_struct(cfg, False)
    if (
        hasattr(cfg, 'policy')
        and hasattr(cfg.policy, 'obs_encoder')
        and hasattr(cfg.policy.obs_encoder, 'pretrained')
        and cfg.policy.obs_encoder.pretrained
    ):
        cfg.policy.obs_encoder.pretrained = False
        print("  Disabled obs_encoder.pretrained for offline checkpoint eval")

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=['lr_scheduler'], include_keys=None)

    normalizer_path = run_dir / 'normalizer.pkl'
    if normalizer_path.is_file():
        normalizer = pickle.load(normalizer_path.open('rb'))
        if hasattr(workspace.model, 'set_normalizer'):
            workspace.model.set_normalizer(normalizer)
        if getattr(workspace, 'ema_model', None) is not None and hasattr(workspace.ema_model, 'set_normalizer'):
            workspace.ema_model.set_normalizer(normalizer)
        print(f"  Loaded normalizer from: {normalizer_path}")

    policy = workspace.model
    if cfg.training.use_ema and getattr(workspace, 'ema_model', None) is not None:
        policy = workspace.ema_model
        print("  Using EMA model")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.eval().to(device)

    print(f"  Policy loaded on device: {device}")
    return policy, cfg, device


def infer_dataset_path(cfg, dataset_path: str | None = None):
    candidates = []

    def add_candidate(candidate):
        if not candidate:
            return
        candidate = pathlib.Path(candidate).expanduser()
        if candidate not in candidates:
            candidates.append(candidate)

    add_candidate(dataset_path)
    if hasattr(cfg, 'task') and hasattr(cfg.task, 'dataset') and hasattr(cfg.task.dataset, 'local_files_only'):
        add_candidate(cfg.task.dataset.local_files_only)
    if hasattr(cfg, 'task') and hasattr(cfg.task, 'dataset') and hasattr(cfg.task.dataset, 'dataset_path'):
        add_candidate(cfg.task.dataset.dataset_path)

    task_name = None
    if hasattr(cfg, 'task') and hasattr(cfg.task, 'name'):
        task_name = cfg.task.name

    if task_name:
        workspace_root = pathlib.Path('/mnt/data/shenyibo/workspace')
        add_candidate(workspace_root / task_name / 'replay_buffer.zarr')
        for suffix in ('_100', '_150', '_154'):
            add_candidate(workspace_root / f'{task_name}{suffix}' / 'replay_buffer.zarr')

    for candidate in candidates:
        if candidate.exists():
            resolved = str(candidate.resolve())
            print(f"  Using dataset path: {resolved}")
            return resolved

    tried = '\n'.join(str(p) for p in candidates)
    raise FileNotFoundError(f"Dataset not found. Tried:\n{tried}")


def build_base_absolute_action_from_episode(episode_data, idx: int, action_dim: int):
    """从 episode 原始键中构造用于相对/绝对 action 互转的基准姿态。
    这些键只用于轨迹还原，不会送进 policy infer。
    """
    if action_dim == 10 and 'left_robot_tcp_pose' in episode_data and 'left_robot_gripper_width' in episode_data:
        return np.concatenate([
            np.asarray(episode_data['left_robot_tcp_pose'][idx], dtype=np.float32),
            np.asarray(episode_data['left_robot_gripper_width'][idx], dtype=np.float32).reshape(-1),
        ], axis=-1)

    if action_dim == 19 and all(k in episode_data for k in ('left_robot_tcp_pose', 'left_robot_gripper_width', 'left_eye_tcp_pose')):
        return np.concatenate([
            np.asarray(episode_data['left_robot_tcp_pose'][idx], dtype=np.float32),
            np.asarray(episode_data['left_robot_gripper_width'][idx], dtype=np.float32).reshape(-1),
            np.asarray(episode_data['left_eye_tcp_pose'][idx], dtype=np.float32),
        ], axis=-1)

    raw_action = np.asarray(episode_data['action'][idx], dtype=np.float32).reshape(-1)
    if raw_action.shape[0] < action_dim:
        raise ValueError(
            f'Cannot build base_absolute_action with dim={action_dim}, only raw action dim={raw_action.shape[0]}'
        )
    return raw_action[:action_dim].copy()


def load_replay_buffer(dataset_path: str):
    """??ReplayBuffer"""
    dataset_path = pathlib.Path(dataset_path).expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    print(f"\nLoading ReplayBuffer from: {dataset_path}")
    replay_buffer = ReplayBuffer.create_from_path(str(dataset_path), mode='r')
    print(f"  Loaded! Total episodes: {replay_buffer.n_episodes}")
    
    return replay_buffer

def preprocess_image(img, target_size=224, is_wrist=True):
    """
    预处理图像：
    - wrist图像：center crop到正方形后resize
    - eye图像：pad到正方形后resize
    
    Args:
        img: (H, W, 3) numpy array
        target_size: 目标尺寸 (default: 224)
        is_wrist: 是否是wrist图像（True=crop, False=pad）
    
    Returns:
        img_processed: (target_size, target_size, 3) numpy array, float32, [0, 1]
    """
    h, w = img.shape[:2]
    
    if is_wrist:
        # Wrist图像：center crop到正方形
        if h > w:
            # 高度大，裁剪高度
            start = (h - w) // 2
            img_square = img[start:start+w, :]
        else:
            # 宽度大，裁剪宽度
            start = (w - h) // 2
            img_square = img[:, start:start+h]
    else:
        # Eye图像：pad到正方形
        if h > w:
            # 高度大，左右pad
            pad_left = (h - w) // 2
            pad_right = h - w - pad_left
            img_square = np.pad(img, ((0, 0), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=0)
        else:
            # 宽度大，上下pad
            pad_top = (w - h) // 2
            pad_bottom = w - h - pad_top
            img_square = np.pad(img, ((pad_top, pad_bottom), (0, 0), (0, 0)), mode='constant', constant_values=0)
    
    # Resize到目标尺寸
    img_resized = cv2.resize(img_square, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    
    # 归一化到[0, 1]
    img_processed = img_resized.astype(np.float32) / 255.0
    
    return img_processed


def absolute_to_relative_action(actions):
    """
    将绝对action转换为相对action
    action[t] = state[t+1] (absolute)
    relative_action[t] = action[t] - action[t-1]
    
    Args:
        actions: (T, D) 绝对action
    
    Returns:
        relative_actions: (T, D) 相对action
    """
    relative_actions = np.zeros_like(actions)
    relative_actions[0] = actions[0]  # 第一帧保持不变
    relative_actions[1:] = actions[1:] - actions[:-1]
    return relative_actions


def rollout_policy_on_episode(
    policy,
    replay_buffer,
    episode_idx,
    device,
    n_obs_steps=2,
    action_dim=None,
    action_representation='relative',
    relative_tcp_obs_for_relative_action=True,
    blackout_left_wrist=False,
    rollout_mode='open_loop'
):
    """
    在指定episode上rollout policy
    
    Args:
        blackout_left_wrist: 是否将left_eye_img涂黑（参数名保持向后兼容）
    
    Returns:
        pred_actions_relative: (T, D) 预测的相对action
        gt_actions_absolute: (T, D) GT绝对action
        start_pos: (3,) 起始位置（用于对齐）
    """
    print(f"\nRolling out policy on episode {episode_idx}...")
    if rollout_mode not in ('open_loop', 'teacher_forcing'):
        raise ValueError(f"Unsupported rollout_mode: {rollout_mode}")
    print(f"  Rollout mode: {rollout_mode}")
    
    # ============ 硬编码开关：是否将图像设置为全黑 ============
    BLACKOUT_LEFT_EYE = blackout_left_wrist  # 使用函数参数控制是否黑化left_eye_img
    if BLACKOUT_LEFT_EYE:
        print(f"  [DEBUG] left_eye_img will be blacked out!")
    # ==================================================================
    
    # 获取episode数据
    episode_data = replay_buffer.get_episode(episode_idx)
    episode_length = len(episode_data['action'])
    
    print(f"  Episode length: {episode_length}")
    
    # 检查可用的图像键
    all_img_keys = [k for k in episode_data.keys() if 'img' in k.lower()]
    img_keys = all_img_keys.copy()
    expected_keys = None
    if hasattr(policy, "normalizer") and hasattr(policy.normalizer, "params_dict"):
        expected_keys = set(policy.normalizer.params_dict.keys())
        img_keys = [k for k in img_keys if k in expected_keys]
    print(f"  All image keys in episode: {all_img_keys}")
    print(f"  Available image keys (after policy filter): {img_keys}")
    print(f"  BLACKOUT_LEFT_EYE = {BLACKOUT_LEFT_EYE}")
    if expected_keys is not None:
        print(f"  Expected keys from policy: {sorted(list(expected_keys))}")
    lowdim_keys = []
    if expected_keys is not None:
        lowdim_keys = [k for k in expected_keys if ('img' not in k) and (k != 'action')]
    
    pred_actions_relative = []
    pred_actions_absolute = []
    
    # 获取policy输出的action序列长度
    n_action_steps = 8  # default
    if hasattr(policy, 'n_action_steps'):
        n_action_steps = policy.n_action_steps
    print(f"  Policy n_action_steps: {n_action_steps}")

    # 按n_action_steps chunk推理，保持轨迹连续性
    # 主要思路：第一次从 GT 开始，之后用上一次的推理结果继续
    t = 0
    last_absolute_state = None  # 记录上一次的最后一个绝对位置
    
    while t < episode_length:
        # 收集observation序列 (n_obs_steps个时间步)
        obs_imgs = {}
        obs_lowdim = {}
        
        for obs_t in range(t, min(t + n_obs_steps, episode_length)):
            # 处理每个图像
            for img_key in img_keys:
                img = episode_data[img_key][obs_t]  # (H, W, 3)
                
                # 硬编码开关：将指定图像设置为全黑
                if BLACKOUT_LEFT_EYE and 'left_eye' in img_key.lower():
                    img = np.zeros_like(img, dtype=img.dtype)
                
                # 判断是wrist还是eye
                is_wrist = 'wrist' in img_key.lower()
                img_processed = preprocess_image(img, target_size=224, is_wrist=is_wrist)
                
                # 转换为torch tensor (C, H, W)
                img_tensor = torch.from_numpy(img_processed).permute(2, 0, 1).float()
                
                # 添加到obs字典
                if img_key not in obs_imgs:
                    obs_imgs[img_key] = []
                obs_imgs[img_key].append(img_tensor)

            # 处理低维观测
            for key in lowdim_keys:
                if key in episode_data and 'wrt' not in key:
                    val = episode_data[key][obs_t]
                    val = np.asarray(val)
                    if val.ndim == 0:
                        val = val.reshape(1)
                    if key not in obs_lowdim:
                        obs_lowdim[key] = []
                    obs_lowdim[key].append(torch.from_numpy(val.astype(np.float32)))
        
        if not obs_imgs:
            break

        # Stack时间维度
        obs = {k: torch.stack(v, dim=0) for k, v in obs_imgs.items()}
        for key, vals in obs_lowdim.items():
            obs[key] = torch.stack(vals, dim=0)

        # 不足n_obs_steps时复制最后一个obs
        first_obs_key = next(iter(obs))
        while obs[first_obs_key].shape[0] < n_obs_steps:
            obs = dict_apply(obs, lambda x: torch.cat([x, x[-1:]], dim=0))
            break

        # 仅保留模型真正需要的 lowdim obs 给 policy；
        # 但轨迹还原用的 base pose 仍可直接从 episode_data 的 tcp/gripper 键中读取。
        abs_obs = {k: v.clone() for k, v in obs.items() if k in lowdim_keys}

        use_gt_base = (rollout_mode == 'teacher_forcing') or (last_absolute_state is None)
        if use_gt_base:
            if action_dim is None:
                action_dim = int(episode_data['action'].shape[-1])
            base_idx = min(t + n_obs_steps - 1, episode_length - 1)
            if abs_obs:
                base_absolute_action = np.concatenate([
                    abs_obs['left_robot_tcp_pose'][-1].cpu().numpy() if 'left_robot_tcp_pose' in abs_obs else np.array([]),
                    abs_obs['right_robot_tcp_pose'][-1].cpu().numpy() if 'right_robot_tcp_pose' in abs_obs else np.array([]),
                    abs_obs['left_robot_gripper_width'][-1].cpu().numpy() if 'left_robot_gripper_width' in abs_obs else np.array([]),
                    abs_obs['right_robot_gripper_width'][-1].cpu().numpy() if 'right_robot_gripper_width' in abs_obs else np.array([]),
                ], axis=-1)
            else:
                base_absolute_action = build_base_absolute_action_from_episode(episode_data, base_idx, action_dim)
                print(f"    t={t}: Using GT zarr tcp/gripper keys to build base_absolute_action (dim={action_dim})")
        else:
            base_absolute_action = last_absolute_state
            print(f"    t={t}: Using predicted state from previous inference")

        # 只有模型真的吃 tcp pose 观测时，才对这些键做 relative 化。
        if relative_tcp_obs_for_relative_action and lowdim_keys:
            for key in list(obs.keys()):
                if ('robot_tcp_pose' in key) and ('wrt' not in key):
                    abs_seq = obs[key].cpu().numpy()
                    rel_seq = absolute_actions_to_relative_actions(
                        abs_seq,
                        base_absolute_action=abs_seq[-1],
                        action_representation=action_representation
                    )
                    obs[key] = torch.from_numpy(rel_seq.astype(np.float32))

        # 计算inter-gripper (wrt) keys
        if len(lowdim_keys) > 0:
            obs_np = {k: v.cpu().numpy() for k, v in obs.items() if k in lowdim_keys}
            inter = get_inter_gripper_actions(obs_np, lowdim_keys)
            for k, v in inter.items():
                if k in lowdim_keys:
                    obs[k] = torch.from_numpy(v.astype(np.float32))
        
        # 添加batch维度
        obs = dict_apply(obs, lambda x: x.unsqueeze(0).to(device))
        # print(obs.keys())
        # quit()
        # Policy推理
        with torch.no_grad():
            result = policy.predict_action(obs)
        
        # 获取预测的action序列（相对action应该是(1, 16, D)）
        pred_actions_rel_full = result["action_pred"][0].detach().cpu().numpy()  # (actual_horizon, D)
        
        # 仅在第一次迭代时打印shape
        if t == 0:
            print(f"  First inference action_pred shape: {result['action_pred'].shape}")
            print(f"  After [0]: {pred_actions_rel_full.shape}")
            # 只使用前n_action_steps步
            print(f"  Using only first {n_action_steps} actions")

        # 只使用前n_action_steps个action
        pred_actions_rel_seq = pred_actions_rel_full[:n_action_steps]

        # 对每个action进行处理
        for i, pred_action_rel in enumerate(pred_actions_rel_seq):
            # 检查是否超过episode长度
            if t + i >= episode_length:
                break
            pred_actions_relative.append(pred_action_rel)

            # 还原为绝对action（与dataset逻辑一致）
            pred_action_abs = relative_actions_to_absolute_actions(
                pred_action_rel[None, :],
                base_absolute_action=base_absolute_action,
                action_representation=action_representation
            )[0]
            pred_actions_absolute.append(pred_action_abs)
            
            # 更新最后一个绝对状态，用于下一次迭代
            # 加上当前的相对action（一维不变的部分）
            last_absolute_state = pred_action_abs.copy()

        # 跳n_action_steps帧
        t += n_action_steps
    
    pred_actions_relative = np.array(pred_actions_relative)  # (T, D)
    pred_actions_absolute = np.array(pred_actions_absolute)  # (T, D)

    if len(pred_actions_relative) == 0:
        raise RuntimeError('Policy rollout produced 0 actions. Check obs keys and base pose reconstruction logic.')
    
    # 获取GT绝对action
    gt_actions_absolute = episode_data['action']  # (T, D)
    
    # 获取起始位置（从第一个GT action中提取左臂xyz）
    start_pos = gt_actions_absolute[0, :3]  # (3,)
    
    print(f"  Predicted {len(pred_actions_relative)} actions")
    print(f"  Start position: {start_pos}")
    
    return pred_actions_relative, pred_actions_absolute, gt_actions_absolute, start_pos


def visualize_trajectories(
    pred_actions_absolute,
    pred_actions_relative,
    gt_actions_absolute,
    start_pos,
    episode_idx,
    output_dir
):
    """
    可视化预测轨迹vs GT轨迹
    
    Args:
        pred_actions_absolute: (T, D) 预测的绝对action
        pred_actions_relative: (T, D) 预测的相对action
        gt_actions_absolute: (T, D) GT绝对action
        start_pos: (3,) 起始位置
        episode_idx: episode索引
        output_dir: 输出目录
    """
    print("\nVisualizing trajectories...")
    
    # 提取左臂xyz（绝对）
    pred_xyz_absolute = pred_actions_absolute[:, :3]  # (T, 3)
    gt_xyz_absolute = gt_actions_absolute[:, :3]  # (T, 3)
    
    # 提取相对xyz
    pred_xyz_relative = pred_actions_relative[:, :3]  # (T, 3)
    # GT相对action = GT[t] - GT[t-1]
    gt_xyz_relative = np.zeros_like(gt_xyz_absolute)
    gt_xyz_relative[0] = gt_xyz_absolute[0]  # 第一步保持不变
    gt_xyz_relative[1:] = np.diff(gt_xyz_absolute, axis=0)
    
    # GT轨迹已经是绝对位置
    # 但为了对齐，我们让GT也从相同起点开始
    gt_xyz_aligned = gt_xyz_absolute.copy()
    gt_offset = gt_xyz_absolute[0] - start_pos
    gt_xyz_aligned = gt_xyz_absolute - gt_offset
    
    print(f"  Pred trajectory shape: {pred_xyz_absolute.shape}")
    print(f"  GT trajectory shape: {gt_xyz_aligned.shape}")
    print(f"  Pred range: X=[{pred_xyz_absolute[:, 0].min():.3f}, {pred_xyz_absolute[:, 0].max():.3f}]")
    print(f"  GT range:   X=[{gt_xyz_aligned[:, 0].min():.3f}, {gt_xyz_aligned[:, 0].max():.3f}]")
    
    # 创建输出目录
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 3D轨迹对比
    fig = plt.figure(figsize=(20, 12))
    
    # 3D视图
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    
    # 绘制GT轨迹
    ax1.plot(gt_xyz_aligned[:, 0], gt_xyz_aligned[:, 1], gt_xyz_aligned[:, 2], 
             'g-', linewidth=3, label='GT', alpha=0.7)
    ax1.scatter(gt_xyz_aligned[0, 0], gt_xyz_aligned[0, 1], gt_xyz_aligned[0, 2],
               color='green', s=200, marker='o', edgecolors='darkgreen', linewidths=3, label='Start')
    ax1.scatter(gt_xyz_aligned[-1, 0], gt_xyz_aligned[-1, 1], gt_xyz_aligned[-1, 2],
               color='green', s=200, marker='s', edgecolors='darkgreen', linewidths=3)
    
    # 绘制预测轨迹
    ax1.plot(pred_xyz_absolute[:, 0], pred_xyz_absolute[:, 1], pred_xyz_absolute[:, 2],
             'r--', linewidth=3, label='Predicted', alpha=0.7)
    ax1.scatter(pred_xyz_absolute[-1, 0], pred_xyz_absolute[-1, 1], pred_xyz_absolute[-1, 2],
               color='red', s=200, marker='s', edgecolors='darkred', linewidths=3)
    
    ax1.set_xlabel('X (m)', fontsize=12)
    ax1.set_ylabel('Y (m)', fontsize=12)
    ax1.set_zlabel('Z (m)', fontsize=12)
    ax1.set_title(f'Episode {episode_idx} - 3D Trajectory (Absolute)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # XYZ位置对比（时间序列，绝对）
    ax2 = fig.add_subplot(2, 2, 2)
    timesteps = np.arange(len(pred_xyz_absolute))
    
    # X轴
    ax2.plot(timesteps, gt_xyz_aligned[:, 0], 'g-', linewidth=2.5, label='GT X', alpha=0.7)
    ax2.plot(timesteps, pred_xyz_absolute[:, 0], 'r--', linewidth=2.5, label='Pred X', alpha=0.7)
    
    # Y轴
    ax2.plot(timesteps, gt_xyz_aligned[:, 1], 'g-', linewidth=2, label='GT Y', alpha=0.5)
    ax2.plot(timesteps, pred_xyz_absolute[:, 1], 'r--', linewidth=2, label='Pred Y', alpha=0.5)
    
    # Z轴
    ax2.plot(timesteps, gt_xyz_aligned[:, 2], 'g:', linewidth=2, label='GT Z', alpha=0.5)
    ax2.plot(timesteps, pred_xyz_absolute[:, 2], 'r:', linewidth=2, label='Pred Z', alpha=0.5)
    
    ax2.set_xlabel('Timestep', fontsize=12)
    ax2.set_ylabel('Position (m)', fontsize=12)
    ax2.set_title('Absolute Position Over Time', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10, ncol=2)
    ax2.grid(True, alpha=0.3)
    
    # 相对action对比（XYZ）
    ax3 = fig.add_subplot(2, 2, 3)
    timesteps_rel = np.arange(len(pred_xyz_relative))
    
    # X轴
    ax3.plot(timesteps_rel, gt_xyz_relative[:, 0], 'g-', linewidth=2.5, label='GT ΔX', alpha=0.7)
    ax3.plot(timesteps_rel, pred_xyz_relative[:, 0], 'r--', linewidth=2.5, label='Pred ΔX', alpha=0.7)
    
    # Y轴
    ax3.plot(timesteps_rel, gt_xyz_relative[:, 1], 'g-', linewidth=2, label='GT ΔY', alpha=0.5)
    ax3.plot(timesteps_rel, pred_xyz_relative[:, 1], 'r--', linewidth=2, label='Pred ΔY', alpha=0.5)
    
    # Z轴
    ax3.plot(timesteps_rel, gt_xyz_relative[:, 2], 'g:', linewidth=2, label='GT ΔZ', alpha=0.5)
    ax3.plot(timesteps_rel, pred_xyz_relative[:, 2], 'r:', linewidth=2, label='Pred ΔZ', alpha=0.5)
    
    ax3.set_xlabel('Timestep', fontsize=12)
    ax3.set_ylabel('Relative Position (m)', fontsize=12)
    ax3.set_title('Relative Action (Δ Position) Over Time', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10, ncol=2)
    ax3.grid(True, alpha=0.3)
    
    # 误差对比
    ax4 = fig.add_subplot(2, 2, 4)
    # 绝对轨迹误差
    min_len_abs = min(len(pred_xyz_absolute), len(gt_xyz_aligned))
    error_abs = np.linalg.norm(pred_xyz_absolute[:min_len_abs] - gt_xyz_aligned[:min_len_abs], axis=1)
    ax4.plot(np.arange(len(error_abs)), error_abs, 'b-', linewidth=2.5, label='Absolute Error', alpha=0.8)
    
    # 相对action误差
    min_len_rel = min(len(pred_xyz_relative), len(gt_xyz_relative))
    error_rel = np.linalg.norm(pred_xyz_relative[:min_len_rel] - gt_xyz_relative[:min_len_rel], axis=1)
    ax4.plot(np.arange(len(error_rel)), error_rel, 'm-', linewidth=2.5, label='Relative Action Error', alpha=0.8)
    
    ax4.set_xlabel('Timestep', fontsize=12)
    ax4.set_ylabel('Error (m)', fontsize=12)
    ax4.set_title('Error Comparison', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = output_path / f"episode_{episode_idx}_policy_vs_gt.png"
    fig.savefig(str(save_path), dpi=150, bbox_inches='tight')
    print(f"✓ Saved to: {save_path}")
    plt.close(fig)
    
    # ============ 单独保存相对action XYZ分离图 ============
    fig_rel = plt.figure(figsize=(18, 5))
    timesteps_rel = np.arange(len(pred_xyz_relative))
    
    # X轴
    ax_x = fig_rel.add_subplot(1, 3, 1)
    ax_x.plot(timesteps_rel, gt_xyz_relative[:, 0], 'g-', linewidth=2.5, label='GT', alpha=0.7, marker='o', markersize=3)
    ax_x.plot(timesteps_rel, pred_xyz_relative[:, 0], 'r--', linewidth=2.5, label='Predicted', alpha=0.7, marker='s', markersize=3)
    ax_x.set_xlabel('Timestep', fontsize=12)
    ax_x.set_ylabel('Δ X (m)', fontsize=12)
    ax_x.set_title('Relative Action - X Axis', fontsize=13, fontweight='bold')
    ax_x.legend(fontsize=11)
    ax_x.grid(True, alpha=0.3)
    
    # Y轴
    ax_y = fig_rel.add_subplot(1, 3, 2)
    ax_y.plot(timesteps_rel, gt_xyz_relative[:, 1], 'g-', linewidth=2.5, label='GT', alpha=0.7, marker='o', markersize=3)
    ax_y.plot(timesteps_rel, pred_xyz_relative[:, 1], 'r--', linewidth=2.5, label='Predicted', alpha=0.7, marker='s', markersize=3)
    ax_y.set_xlabel('Timestep', fontsize=12)
    ax_y.set_ylabel('Δ Y (m)', fontsize=12)
    ax_y.set_title('Relative Action - Y Axis', fontsize=13, fontweight='bold')
    ax_y.legend(fontsize=11)
    ax_y.grid(True, alpha=0.3)
    
    # Z轴
    ax_z = fig_rel.add_subplot(1, 3, 3)
    ax_z.plot(timesteps_rel, gt_xyz_relative[:, 2], 'g-', linewidth=2.5, label='GT', alpha=0.7, marker='o', markersize=3)
    ax_z.plot(timesteps_rel, pred_xyz_relative[:, 2], 'r--', linewidth=2.5, label='Predicted', alpha=0.7, marker='s', markersize=3)
    ax_z.set_xlabel('Timestep', fontsize=12)
    ax_z.set_ylabel('Δ Z (m)', fontsize=12)
    ax_z.set_title('Relative Action - Z Axis', fontsize=13, fontweight='bold')
    ax_z.legend(fontsize=11)
    ax_z.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path_rel = output_path / f"episode_{episode_idx}_relative_action_xyz.png"
    fig_rel.savefig(str(save_path_rel), dpi=150, bbox_inches='tight')
    print(f"✓ Saved to: {save_path_rel}")
    plt.close(fig_rel)
    
    # 计算误差统计
    min_len = min(len(pred_xyz_absolute), len(gt_xyz_aligned))
    error = np.linalg.norm(pred_xyz_absolute[:min_len] - gt_xyz_aligned[:min_len], axis=1)
    
    print(f"\n  Error statistics:")
    print(f"    Mean error: {error.mean():.4f} m")
    print(f"    Max error:  {error.max():.4f} m")
    print(f"    Final error: {error[-1]:.4f} m")
    
    return pred_xyz_absolute, gt_xyz_aligned


def parse_episode_spec(spec: str):
    episodes = []
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if ':' in part:
            fields = [x.strip() for x in part.split(':')]
            if len(fields) not in (2, 3):
                raise ValueError(f"Bad episode range: {part}")
            start = int(fields[0])
            stop = int(fields[1])
            step = int(fields[2]) if len(fields) == 3 and fields[2] else 1
            episodes.extend(range(start, stop, step))
        else:
            episodes.append(int(part))
    if not episodes:
        raise ValueError("No episodes selected")
    return episodes


def compute_xyz_error_stats(pred_actions_abs, gt_actions_abs):
    min_len = min(len(pred_actions_abs), len(gt_actions_abs))
    pred = pred_actions_abs[:min_len]
    gt = gt_actions_abs[:min_len]

    groups = {"left_xyz": slice(0, 3)}
    if pred.shape[-1] >= 12 and gt.shape[-1] >= 12:
        groups["right_xyz"] = slice(9, 12)

    stats = {"num_steps": int(min_len)}
    for name, sl in groups.items():
        error = np.linalg.norm(pred[:, sl] - gt[:, sl], axis=1)
        stats[name] = {
            "mean_m": float(error.mean()),
            "max_m": float(error.max()),
            "final_m": float(error[-1]),
        }
    if "right_xyz" in groups:
        both = np.concatenate([pred[:, :3] - gt[:, :3], pred[:, 9:12] - gt[:, 9:12]], axis=1)
        error = np.linalg.norm(both, axis=1)
        stats["both_arms_xyz"] = {
            "mean_m": float(error.mean()),
            "max_m": float(error.max()),
            "final_m": float(error[-1]),
        }
    return stats


def main():
    parser = argparse.ArgumentParser(description="Visualize and lightly evaluate policy rollout vs dataset GT.")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint, usually checkpoints/latest.ckpt")
    parser.add_argument("--cfg", default=None, help="Optional config yaml. Defaults to ckpt run .hydra/config.yaml")
    parser.add_argument("--dataset", default=None, help="Optional replay_buffer.zarr path. Defaults to cfg task.dataset.dataset_path")
    parser.add_argument("--episodes", default="0", help="Episode list/ranges, e.g. '0,3,5' or '0:5'")
    parser.add_argument("--output-dir", default=None, help="Directory for figures and summary.json")
    parser.add_argument("--n-obs-steps", type=int, default=None, help="Override n_obs_steps from cfg")
    parser.add_argument("--rollout-mode", choices=["open_loop", "teacher_forcing"], default="open_loop")
    parser.add_argument("--no-vis", action="store_true", help="Only compute metrics, skip png outputs")
    args = parser.parse_args()

    print("="*60)
    print("Policy vs GT Trajectory Visualization / Light Eval")
    print("="*60)

    policy, cfg, device = load_policy(args.ckpt, cfg_yaml_path=args.cfg)
    dataset_path = infer_dataset_path(cfg, args.dataset)
    replay_buffer = load_replay_buffer(dataset_path)
    episodes = parse_episode_spec(args.episodes)

    output_dir = args.output_dir
    if output_dir is None:
        run_dir = pathlib.Path(args.ckpt).expanduser().resolve().parent.parent
        output_dir = str(run_dir / "policy_vs_gt_eval")

    n_obs_steps = args.n_obs_steps
    if n_obs_steps is None:
        n_obs_steps = int(cfg.n_obs_steps) if hasattr(cfg, "n_obs_steps") else 2

    action_representation = "relative"
    relative_tcp_obs_for_relative_action = True
    if cfg is not None and hasattr(cfg, "task") and hasattr(cfg.task, "dataset"):
        ds_cfg = cfg.task.dataset
        if hasattr(ds_cfg, "action_representation"):
            action_representation = ds_cfg.action_representation
        if hasattr(ds_cfg, "relative_tcp_obs_for_relative_action"):
            relative_tcp_obs_for_relative_action = ds_cfg.relative_tcp_obs_for_relative_action

    action_dim = int(cfg.shape_meta.action.shape[0]) if hasattr(cfg, "shape_meta") else None
    print(f"dataset_path: {dataset_path}")
    print(f"episodes: {episodes}")
    print(f"n_obs_steps: {n_obs_steps}")
    print(f"action_dim: {action_dim}")
    print(f"action_representation: {action_representation}")
    print(f"rollout_mode: {args.rollout_mode}")
    print(f"output_dir: {output_dir}")

    results = []
    for episode_idx in episodes:
        if episode_idx >= replay_buffer.n_episodes:
            raise ValueError(f"Episode index {episode_idx} out of range. Total episodes: {replay_buffer.n_episodes}")

        pred_actions_rel, pred_actions_abs, gt_actions_abs, start_pos = rollout_policy_on_episode(
            policy,
            replay_buffer,
            episode_idx,
            device,
            n_obs_steps=n_obs_steps,
            action_dim=action_dim,
            action_representation=action_representation,
            relative_tcp_obs_for_relative_action=relative_tcp_obs_for_relative_action,
            blackout_left_wrist=False,
            rollout_mode=args.rollout_mode
        )

        stats = compute_xyz_error_stats(pred_actions_abs, gt_actions_abs)
        stats["episode_idx"] = int(episode_idx)
        results.append(stats)

        print("\nEpisode metrics:")
        print(json.dumps(stats, indent=2))

        if not args.no_vis:
            visualize_trajectories(
                pred_actions_abs, pred_actions_rel, gt_actions_abs, start_pos, episode_idx, output_dir
            )

    aggregate = {"num_episodes": len(results), "rollout_mode": args.rollout_mode, "episodes": results}
    for key in ("left_xyz", "right_xyz", "both_arms_xyz"):
        present = [r[key] for r in results if key in r]
        if present:
            aggregate[key] = {
                "mean_of_mean_m": float(np.mean([x["mean_m"] for x in present])),
                "max_of_max_m": float(np.max([x["max_m"] for x in present])),
                "mean_final_m": float(np.mean([x["final_m"] for x in present])),
            }

    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_path = output_path / "summary.json"
    summary_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    print("\nAggregate metrics:")
    print(json.dumps({k: v for k, v in aggregate.items() if k != "episodes"}, indent=2))
    print(f"\nSaved summary to: {summary_path}")
    print("\nAll done!")


if __name__ == "__main__":
    main()
