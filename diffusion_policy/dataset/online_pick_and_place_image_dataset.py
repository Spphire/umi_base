from typing import Dict, List, Optional, Union
import numpy as np
import torch
import os
import copy
from collections import deque
from threadpoolctl import threadpool_limits

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, downsample_mask
from diffusion_policy.common.action_utils import absolute_actions_to_relative_actions, get_inter_gripper_actions
from loguru import logger


class AdaptiveSampler:
    """
    Implements SOP-style adaptive sampling between online and offline data.
    
    Formula: ω_on = exp(α·l̄_on) / (exp(α·l̄_on) + exp(l̄_off))
    where α > 1 is a boost factor to prioritize online data.
    """
    
    def __init__(
        self,
        window_size: int = 200,
        boost_factor: float = 1.5,
        min_online_ratio: float = 0.2,
        max_online_ratio: float = 0.8,
    ):
        self.window_size = window_size
        self.boost_factor = boost_factor
        self.min_online_ratio = min_online_ratio
        self.max_online_ratio = max_online_ratio
        
        self.online_losses: deque = deque(maxlen=window_size)
        self.offline_losses: deque = deque(maxlen=window_size)
        
        self._online_weight = 0.5
    
    def add_loss(self, loss: float, is_online: bool):
        if is_online:
            self.online_losses.append(loss)
        else:
            self.offline_losses.append(loss)
    
    def update_weights(self):
        if len(self.online_losses) < 10 or len(self.offline_losses) < 10:
            return self._online_weight
        
        l_on = np.mean(list(self.online_losses))
        l_off = np.mean(list(self.offline_losses))
        
        exp_on = np.exp(np.clip(self.boost_factor * l_on, -50, 50))
        exp_off = np.exp(np.clip(l_off, -50, 50))
        
        omega_on = exp_on / (exp_on + exp_off)
        omega_on = np.clip(omega_on, self.min_online_ratio, self.max_online_ratio)
        
        self._online_weight = omega_on
        return self._online_weight
    
    @property
    def online_weight(self) -> float:
        return self._online_weight
    
    @property
    def offline_weight(self) -> float:
        return 1.0 - self._online_weight
    
    def get_stats(self) -> Dict:
        return {
            'online_weight': self._online_weight,
            'offline_weight': 1.0 - self._online_weight,
            'online_loss_mean': np.mean(list(self.online_losses)) if self.online_losses else 0.0,
            'offline_loss_mean': np.mean(list(self.offline_losses)) if self.offline_losses else 0.0,
            'online_loss_count': len(self.online_losses),
            'offline_loss_count': len(self.offline_losses),
        }


class OnlinePickAndPlaceImageDataset(BaseImageDataset):
    """
    Dataset supporting online data collection and adaptive sampling between
    online and offline data buffers. Based on SOP paper's design.
    
    Features:
    - Dual buffer: offline (static) + online (dynamically growing)
    - Adaptive sampling: loss-based weight adjustment between buffers
    - Thread-safe data appending
    - val_ratio=0 (no validation split for online training)
    """
    
    def __init__(
        self,
        shape_meta: dict,
        offline_dataset_path: str,
        horizon: int = 1,
        pad_before: int = 0,
        pad_after: int = 0,
        n_obs_steps: Optional[int] = None,
        n_latency_steps: int = 0,
        seed: int = 42,
        max_train_episodes: Optional[int] = None,
        delta_action: bool = False,
        relative_action: bool = False,
        use_quantiles: bool = False,
        action_representation: str = 'relative',
        # Adaptive sampling parameters
        adaptive_window_size: int = 200,
        adaptive_boost_factor: float = 1.5,
        adaptive_min_online_ratio: float = 0.2,
        adaptive_max_online_ratio: float = 0.8,
        # Online buffer settings
        initial_online_weight: float = 0.5,
    ):
        logger.info(f'Initializing OnlinePickAndPlaceImageDataset')
        logger.info(f'  offline_dataset_path: {offline_dataset_path}')
        logger.info(f'  use_quantiles: {use_quantiles}')
        logger.info(f'  action_representation: {action_representation}')
        
        assert os.path.isdir(offline_dataset_path), f"Offline dataset path not found: {offline_dataset_path}"
        
        # Parse shape_meta
        rgb_keys = []
        lowdim_keys = []
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type_ = attr.get('type', 'low_dim')
            if type_ == 'rgb':
                rgb_keys.append(key)
            elif type_ == 'low_dim':
                lowdim_keys.append(key)
        
        # Load offline buffer
        zarr_path = os.path.join(offline_dataset_path)
        zarr_load_keys = rgb_keys + lowdim_keys + ['action']
        zarr_load_keys = list(filter(lambda key: "wrt" not in key, zarr_load_keys))
        
        self.offline_replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=zarr_load_keys)
        logger.info(f'Loaded offline buffer with {self.offline_replay_buffer.n_episodes} episodes, {self.offline_replay_buffer.n_steps} steps')
        
        # Create empty online buffer (numpy backend for dynamic growth)
        self.online_replay_buffer = ReplayBuffer.create_empty_numpy()
        
        # Handle delta action if needed
        if delta_action:
            self._apply_delta_action(self.offline_replay_buffer)
        
        # Setup key_first_k for performance
        key_first_k = {}
        if n_obs_steps is not None:
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps
        
        # Create samplers for both buffers
        # val_ratio=0 for online training
        offline_episode_mask = np.ones(self.offline_replay_buffer.n_episodes, dtype=bool)
        offline_episode_mask = downsample_mask(
            mask=offline_episode_mask,
            max_n=max_train_episodes,
            seed=seed
        )
        
        self.offline_sampler = SequenceSampler(
            replay_buffer=self.offline_replay_buffer,
            sequence_length=horizon + n_latency_steps,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=offline_episode_mask,
            key_first_k=key_first_k,
        )
        
        # Online sampler will be recreated when data is added
        self.online_sampler = None
        
        # Adaptive sampler
        self.adaptive_sampler = AdaptiveSampler(
            window_size=adaptive_window_size,
            boost_factor=adaptive_boost_factor,
            min_online_ratio=adaptive_min_online_ratio,
            max_online_ratio=adaptive_max_online_ratio,
        )
        self.adaptive_sampler._online_weight = initial_online_weight
        
        # Store configuration
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.n_obs_steps = n_obs_steps
        self.horizon = horizon
        self.n_latency_steps = n_latency_steps
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.use_quantiles = use_quantiles
        self.action_representation = action_representation
        self.relative_action = relative_action
        self.delta_action = delta_action
        self.key_first_k = key_first_k
        self.seed = seed
        
        self.relative_tcp_obs_for_relative_action = True
        
        self._online_episodes_count = 0
    
    def _apply_delta_action(self, replay_buffer: ReplayBuffer):
        actions = replay_buffer['action'][:]
        assert actions.shape[1] <= 3
        actions_diff = np.zeros_like(actions)
        episode_ends = replay_buffer.episode_ends[:]
        for i in range(len(episode_ends)):
            start = 0
            if i > 0:
                start = episode_ends[i-1]
            end = episode_ends[i]
            actions_diff[start+1:end] = np.diff(actions[start:end], axis=0)
        replay_buffer['action'][:] = actions_diff
    
    def append_episodes(self, episodes_data: List[Dict[str, np.ndarray]]):
        """
        Append new episodes to the online buffer.
        
        Args:
            episodes_data: List of episode data dicts, each containing
                          arrays for each key with shape (T, ...)
        """
        for episode_data in episodes_data:
            self.online_replay_buffer.add_episode(episode_data, is_dagger=True)
            self._online_episodes_count += 1
        
        # Rebuild online sampler
        if self.online_replay_buffer.n_episodes > 0:
            online_episode_mask = np.ones(self.online_replay_buffer.n_episodes, dtype=bool)
            self.online_sampler = SequenceSampler(
                replay_buffer=self.online_replay_buffer,
                sequence_length=self.horizon + self.n_latency_steps,
                pad_before=self.pad_before,
                pad_after=self.pad_after,
                episode_mask=online_episode_mask,
                key_first_k=self.key_first_k,
            )
        
        logger.info(f'Online buffer updated: {self.online_replay_buffer.n_episodes} episodes, '
                   f'{self.online_replay_buffer.n_steps} steps')
    
    def update_sampling_weights(self, loss: float, is_online: bool):
        """
        Update adaptive sampling weights based on training loss.
        
        Args:
            loss: The training loss value
            is_online: Whether this loss was from online or offline data
        """
        self.adaptive_sampler.add_loss(loss, is_online)
        return self.adaptive_sampler.update_weights()
    
    def get_sampling_stats(self) -> Dict:
        return self.adaptive_sampler.get_stats()
    
    def get_all_actions(self) -> torch.Tensor:
        offline_actions = self.offline_replay_buffer['action'][:, :self.shape_meta['action']['shape'][0]]
        if self.online_replay_buffer.n_steps > 0:
            online_actions = self.online_replay_buffer['action'][:, :self.shape_meta['action']['shape'][0]]
            all_actions = np.concatenate([offline_actions, online_actions], axis=0)
        else:
            all_actions = offline_actions
        return torch.from_numpy(all_actions)
    
    def __len__(self):
        offline_len = len(self.offline_sampler)
        online_len = len(self.online_sampler) if self.online_sampler is not None else 0
        return offline_len + online_len
    
    def sample_batch_indices(self, batch_size: int, replace: bool = False) -> List[tuple]:
        """
        Sample batch indices using adaptive sampling weights.
        
        Args:
            batch_size: Number of samples to draw
            replace: If False, sample without replacement (no duplicates).
                    If True, sample with replacement (may have duplicates).
                    When replace=False and batch_size > available samples,
                    falls back to sampling with replacement.
        
        Returns list of (is_online, idx) tuples.
        """
        offline_len = len(self.offline_sampler)
        online_len = len(self.online_sampler) if self.online_sampler is not None else 0
        
        if online_len == 0:
            # Only offline data available
            if not replace and batch_size <= offline_len:
                indices = np.random.choice(offline_len, size=batch_size, replace=False)
            else:
                indices = np.random.randint(0, offline_len, size=batch_size)
            return [(False, int(idx)) for idx in indices]
        
        online_weight = self.adaptive_sampler.online_weight
        
        n_online = int(batch_size * online_weight)
        n_offline = batch_size - n_online
        
        # Sample online indices
        if n_online > 0:
            if not replace and n_online <= online_len:
                online_indices = np.random.choice(online_len, size=n_online, replace=False)
            else:
                online_indices = np.random.randint(0, online_len, size=n_online)
        else:
            online_indices = np.array([], dtype=np.int64)
        
        # Sample offline indices
        if n_offline > 0:
            if not replace and n_offline <= offline_len:
                offline_indices = np.random.choice(offline_len, size=n_offline, replace=False)
            else:
                offline_indices = np.random.randint(0, offline_len, size=n_offline)
        else:
            offline_indices = np.array([], dtype=np.int64)
        
        batch = [(True, int(idx)) for idx in online_indices] + [(False, int(idx)) for idx in offline_indices]
        np.random.shuffle(batch)
        return batch
    
    def get_item_by_source(self, is_online: bool, idx: int) -> Dict[str, torch.Tensor]:
        """Get item from specific buffer (online or offline)."""
        if is_online and self.online_sampler is not None:
            return self._get_item_from_sampler(self.online_sampler, idx)
        else:
            return self._get_item_from_sampler(self.offline_sampler, idx)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Standard getitem - uses unified indexing across both buffers."""
        offline_len = len(self.offline_sampler)
        
        if idx < offline_len:
            return self._get_item_from_sampler(self.offline_sampler, idx)
        else:
            online_idx = idx - offline_len
            if self.online_sampler is not None and online_idx < len(self.online_sampler):
                return self._get_item_from_sampler(self.online_sampler, online_idx)
            else:
                return self._get_item_from_sampler(self.offline_sampler, idx % offline_len)
    
    def _get_item_from_sampler(self, sampler: SequenceSampler, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = sampler.sample_sequence(idx)
        
        T_slice = slice(self.n_obs_steps)
        
        obs_dict = {}
        for key in self.rgb_keys:
            obs_dict[key] = np.moveaxis(data[key][T_slice], -1, 1).astype(np.float32) / 255.
            if key not in self.rgb_keys:
                del data[key]
        
        for key in self.lowdim_keys:
            if 'wrt' not in key:
                obs_dict[key] = data[key][:, :self.shape_meta['obs'][key]['shape'][0]][T_slice].astype(np.float32)
        
        obs_dict.update(get_inter_gripper_actions(obs_dict, self.lowdim_keys))
        for key in self.lowdim_keys:
            if 'wrt' in key:
                obs_dict[key] = obs_dict[key][:, :self.shape_meta['obs'][key]['shape'][0]][T_slice].astype(np.float32)
        
        action = data['action'][:, :self.shape_meta['action']['shape'][0]].astype(np.float32)
        
        if self.n_latency_steps > 0:
            action = action[self.n_latency_steps:]
        
        if self.relative_action:
            base_absolute_action = np.concatenate([
                obs_dict['left_robot_tcp_pose'][-1] if 'left_robot_tcp_pose' in obs_dict else np.array([]),
                obs_dict['right_robot_tcp_pose'][-1] if 'right_robot_tcp_pose' in obs_dict else np.array([]),
                obs_dict['left_robot_gripper_width'][-1] if 'left_robot_gripper_width' in obs_dict else np.array([]),
                obs_dict['right_robot_gripper_width'][-1] if 'right_robot_gripper_width' in obs_dict else np.array([]),
            ], axis=-1)
            action = absolute_actions_to_relative_actions(
                action, base_absolute_action=base_absolute_action,
                action_representation=self.action_representation
            )
            
            if self.relative_tcp_obs_for_relative_action:
                for key in self.lowdim_keys:
                    if 'robot_tcp_pose' in key and 'wrt' not in key:
                        obs_dict[key] = absolute_actions_to_relative_actions(
                            obs_dict[key],
                            base_absolute_action=obs_dict[key][-1],
                            action_representation=self.action_representation
                        )
        
        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(action)
        }
        
        return torch_data
    
    @property
    def online_episodes_count(self) -> int:
        return self._online_episodes_count
    
    @property
    def offline_episodes_count(self) -> int:
        return self.offline_replay_buffer.n_episodes
    
    @property
    def total_steps(self) -> int:
        return self.offline_replay_buffer.n_steps + self.online_replay_buffer.n_steps