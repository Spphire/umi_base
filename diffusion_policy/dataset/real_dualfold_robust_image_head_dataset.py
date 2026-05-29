from typing import Dict
import os

import numpy as np
import torch
from loguru import logger
from threadpoolctl import threadpool_limits

from diffusion_policy.common.action_utils import absolute_actions_to_relative_actions, get_inter_gripper_actions
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, downsample_mask, get_val_mask
from diffusion_policy.dataset.real_pick_and_place_image_head_dataset import RealPickAndPlaceImageHeadDataset


class RealDualfoldRobustImageHeadDataset(RealPickAndPlaceImageHeadDataset):
    def __init__(
        self,
        shape_meta: dict,
        dataset_path: str,
        horizon=1,
        pad_before=0,
        pad_after=0,
        n_obs_steps=None,
        n_latency_steps=0,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
        delta_action=False,
        relative_action=False,
        use_quantiles=False,
        action_representation='relative',
        dagger_sampling_ratio=1,
        random_mask_head_image=True,
        action_slice=None,
    ):
        logger.info(f'use_quantiles: {use_quantiles}')
        logger.info(f'using action representation: {action_representation}')
        assert os.path.isdir(dataset_path)

        self.random_mask_head_image = random_mask_head_image
        self.action_dim = int(shape_meta['action']['shape'][0])
        self.action_indices = self._resolve_action_indices(action_slice, self.action_dim)
        logger.info(
            f"Randomly masking head image is {'enabled' if self.random_mask_head_image else 'disabled'}."
        )

        rgb_keys = []
        lowdim_keys = []
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            key_type = attr.get('type', 'low_dim')
            if key_type == 'rgb':
                rgb_keys.append(key)
            elif key_type == 'low_dim':
                lowdim_keys.append(key)

        zarr_load_keys = rgb_keys + lowdim_keys + ['action']
        auxiliary_lowdim_keys = []
        if 'left_robot_gripper_width' not in zarr_load_keys and 'left_wrist_img' in zarr_load_keys:
            zarr_load_keys.append('left_robot_gripper_width')
            auxiliary_lowdim_keys.append('left_robot_gripper_width')
        if 'left_robot_tcp_pose' not in zarr_load_keys and 'left_wrist_img' in zarr_load_keys:
            zarr_load_keys.append('left_robot_tcp_pose')
            auxiliary_lowdim_keys.append('left_robot_tcp_pose')
        if 'right_robot_gripper_width' not in zarr_load_keys and 'right_wrist_img' in zarr_load_keys:
            zarr_load_keys.append('right_robot_gripper_width')
            auxiliary_lowdim_keys.append('right_robot_gripper_width')
        if 'right_robot_tcp_pose' not in zarr_load_keys and 'right_wrist_img' in zarr_load_keys:
            zarr_load_keys.append('right_robot_tcp_pose')
            auxiliary_lowdim_keys.append('right_robot_tcp_pose')
        if 'left_eye_tcp_pose' not in zarr_load_keys and 'left_eye_img' in zarr_load_keys:
            zarr_load_keys.append('left_eye_tcp_pose')
            auxiliary_lowdim_keys.append('left_eye_tcp_pose')

        zarr_load_keys = [key for key in zarr_load_keys if 'wrt' not in key]
        replay_buffer = ReplayBuffer.copy_from_path(dataset_path, keys=zarr_load_keys)

        if delta_action:
            actions = replay_buffer['action'][:]
            assert actions.shape[1] <= 3
            actions_diff = np.zeros_like(actions)
            episode_ends = replay_buffer.episode_ends[:]
            for i in range(len(episode_ends)):
                start = 0
                if i > 0:
                    start = episode_ends[i - 1]
                end = episode_ends[i]
                actions_diff[start + 1 : end] = np.diff(actions[start:end], axis=0)
            replay_buffer['action'][:] = actions_diff

        key_first_k = {}
        if n_obs_steps is not None:
            for key in rgb_keys + lowdim_keys + auxiliary_lowdim_keys:
                key_first_k[key] = n_obs_steps

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)

        sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon + n_latency_steps,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k,
            dagger_sampling_ratio=dagger_sampling_ratio,
        )

        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.n_obs_steps = n_obs_steps
        self.val_mask = val_mask
        self.horizon = horizon
        self.n_latency_steps = n_latency_steps
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.use_quantiles = use_quantiles
        self.action_representation = action_representation
        self.relative_action = relative_action
        self.key_first_k = key_first_k
        self.dagger_sampling_ratio = dagger_sampling_ratio
        self.auxiliary_lowdim_keys = auxiliary_lowdim_keys
        self.relative_tcp_obs_for_relative_action = True

        if relative_action:
            logger.info('Relative action is enabled. All actions will be relative to the current frame.')
            if self.action_dim not in (10, 20, 29):
                raise ValueError(f'Unsupported dualfold relative action dim: {self.action_dim}')

    @staticmethod
    def _resolve_action_indices(action_slice, action_dim):
        if action_slice is None:
            return np.arange(action_dim)
        if isinstance(action_slice, str):
            if action_slice == 'left':
                return np.array([*range(9), 18], dtype=np.int64)
            if action_slice == 'right':
                return np.array([*range(9, 18), 19], dtype=np.int64)
            raise ValueError(f'Unsupported action_slice: {action_slice}')
        action_indices = np.array(action_slice, dtype=np.int64)
        if len(action_indices) != action_dim:
            raise ValueError(f'action_slice length {len(action_indices)} does not match action dim {action_dim}')
        return action_indices

    def _build_base_absolute_action(self, data, obs_dict, t_slice):
        if self.action_dim == 10 and np.array_equal(self.action_indices, np.array([*range(9), 18])):
            keys = ['left_robot_tcp_pose', 'left_robot_gripper_width']
        elif self.action_dim == 10 and np.array_equal(self.action_indices, np.array([*range(9, 18), 19])):
            keys = ['right_robot_tcp_pose', 'right_robot_gripper_width']
        else:
            keys = [
                'left_robot_tcp_pose',
                'right_robot_tcp_pose',
                'left_robot_gripper_width',
                'right_robot_gripper_width',
            ]
        if self.action_dim == 29:
            keys.append('left_eye_tcp_pose')

        parts = []
        for key in keys:
            if key.endswith('tcp_pose'):
                parts.append(self._get_base_obs_from_sample(data, obs_dict, key, t_slice))
            else:
                parts.append(data[key][:, :1][t_slice].astype(np.float32)[-1])
        return np.concatenate(parts, axis=-1)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)
        t_slice = slice(self.n_obs_steps)

        obs_dict = {}
        for key in self.rgb_keys:
            img = data[key][t_slice]
            mask_img_flag = False
            if 'eye' in key:
                if np.random.rand() < 0.5 and 'right_eye_img' in data:
                    img = data['right_eye_img'][t_slice]
            elif 'wrist' in key:
                if np.random.rand() < 0.05:
                    mask_img_flag = True
            else:
                raise NotImplementedError(f'Unknown image key: {key}')

            img_normalized = np.moveaxis(img, -1, 1).astype(np.float32) / 255.0
            if mask_img_flag:
                img_normalized = np.zeros_like(img_normalized)
            obs_dict[key] = img_normalized

        for key in self.lowdim_keys:
            if 'wrt' not in key:
                obs_dict[key] = data[key][:, : self.shape_meta['obs'][key]['shape'][0]][t_slice].astype(np.float32)

        obs_dict.update(get_inter_gripper_actions(obs_dict, self.lowdim_keys))
        for key in self.lowdim_keys:
            if 'wrt' in key:
                obs_dict[key] = obs_dict[key][:, : self.shape_meta['obs'][key]['shape'][0]][t_slice].astype(np.float32)

        action = data['action'][:, self.action_indices].astype(np.float32)
        if self.n_latency_steps > 0:
            action = action[self.n_latency_steps :]

        if self.relative_action:
            base_absolute_action = self._build_base_absolute_action(data, obs_dict, t_slice)
            extra_dim = 1
            if base_absolute_action.shape[-1] + extra_dim == action.shape[-1]:
                action[..., :-extra_dim] = absolute_actions_to_relative_actions(
                    action[..., :-extra_dim],
                    base_absolute_action=base_absolute_action,
                    action_representation=self.action_representation,
                )
            elif base_absolute_action.shape[-1] == action.shape[-1]:
                action = absolute_actions_to_relative_actions(
                    action,
                    base_absolute_action=base_absolute_action,
                    action_representation=self.action_representation,
                )
            else:
                raise ValueError(
                    f'Base absolute action dim {base_absolute_action.shape[-1]} does not match action dim {action.shape[-1]}'
                )

            if self.relative_tcp_obs_for_relative_action:
                for key in self.lowdim_keys:
                    if 'tcp_pose' in key and 'wrt' not in key:
                        obs_dict[key] = absolute_actions_to_relative_actions(
                            obs_dict[key],
                            base_absolute_action=obs_dict[key][-1],
                            action_representation=self.action_representation,
                        )

        return {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(action),
        }
