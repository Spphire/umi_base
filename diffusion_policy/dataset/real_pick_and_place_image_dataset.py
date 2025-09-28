from typing import Dict, List
import torch
import numpy as np
import os
from threadpoolctl import threadpool_limits
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.common.normalize_util import (
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats
)
from diffusion_policy.common.action_utils import absolute_actions_to_relative_actions
from loguru import logger
import tqdm
from diffusion_policy.common.normalize_util import get_action_normalizer

class RealPickAndPlaceImageDataset(BaseImageDataset):
    def __init__(self,
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
        ):
        print(dataset_path)
        assert os.path.isdir(dataset_path)

        zarr_path = os.path.join(dataset_path)
        replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['left_robot_tcp_pose', 'left_robot_gripper_width',
                             'action', 'left_wrist_img'])
        
        if delta_action:
            # replace action as relative to previous frame
            actions = replay_buffer['action'][:]
            # support positions only at this time
            assert actions.shape[1] <= 3
            actions_diff = np.zeros_like(actions)
            episode_ends = replay_buffer.episode_ends[:]
            for i in range(len(episode_ends)):
                start = 0
                if i > 0:
                    start = episode_ends[i-1]
                end = episode_ends[i]
                # delta action is the difference between previous desired position and the current
                # it should be scheduled at the previous timestep for the current timestep
                # to ensure consistency with positional mode
                actions_diff[start+1:end] = np.diff(actions[start:end], axis=0)
            replay_buffer['action'][:] = actions_diff

        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)
        
        key_first_k = dict()
        if n_obs_steps is not None:
            # only take first k obs from images
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        sampler = SequenceSampler(
            replay_buffer=replay_buffer, 
            sequence_length=horizon+n_latency_steps,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k)
        
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
        self.relative_action = relative_action
        if relative_action:
            logger.info(
                "Relative action is enabled. All actions will be relative to the current frame.")

        self.relative_tcp_obs_for_relative_action = True

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon+self.n_latency_steps,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=self.val_mask
            )
        val_set.val_mask = ~self.val_mask
        return val_set


    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        if self.relative_action:
            relative_data_dict = {key: list() for key in (self.lowdim_keys + ['action']) if ('robot_tcp_pose' in key and 'wrt' not in key) or 'action' in key}
            for data in tqdm.tqdm(self, leave=False, desc='Calculating relative action/obs for normalizer'):
                for key in relative_data_dict.keys():
                    if key == 'action':
                        relative_data_dict[key].append(data[key])
                    else:
                        relative_data_dict[key].append(data['obs'][key])
            relative_data_dict = dict_apply(relative_data_dict, np.stack)

            # from diffusion_policy.scripts.check_iphone_data_distribution import plot_data
            # plot_data(relative_data_dict['left_robot_tcp_pose'].reshape(-1, 9), image_name="iphone_rel_tcp_pose_distribution")
            # plot_data(relative_data_dict['action'].reshape(-1, 10), image_name="iphone_rel_action_distribution")

        # action
        if self.relative_action:
            action_all = relative_data_dict['action']
        else:
            action_all = self.replay_buffer['action'][:, :self.shape_meta['action']['shape'][0]]

        normalizer['action'] = get_action_normalizer(action_all)

        # obs
        for key in list(set(self.lowdim_keys)):
            if self.relative_action and key in relative_data_dict:
                normalizer[key] = get_action_normalizer(relative_data_dict[key])
            elif 'robot_tcp_pose' in key and 'wrt' not in key:
                normalizer[key] = get_action_normalizer(self.replay_buffer[key][:, :self.shape_meta['obs'][key]['shape'][0]])
            else:
                normalizer[key] = SingleFieldLinearNormalizer.create_fit(
                    self.replay_buffer[key][:, :self.shape_meta['obs'][key]['shape'][0]])


        # image
        for key in list(set(self.rgb_keys)):
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'][:, :self.shape_meta['action']['shape'][0]])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)
        # to save RAM, only return first n_obs_steps of OBS
        # since the rest will be discarded anyway.
        # when self.n_obs_steps is None
        # this slice does nothing (takes all)
        T_slice = slice(self.n_obs_steps)

        obs_dict = dict()
        for key in self.rgb_keys:
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_dict[key] = np.moveaxis(data[key][T_slice],-1,1
                ).astype(np.float32) / 255.
            # T,C,H,W
            # save ram
            if key not in self.rgb_keys:
                del data[key]
        for key in self.lowdim_keys:
            if 'wrt' not in key:
                obs_dict[key] = data[key][:, :self.shape_meta['obs'][key]['shape'][0]][T_slice].astype(np.float32)

        action = data['action'][:, :self.shape_meta['action']['shape'][0]].astype(np.float32)
        # handle latency by dropping first n_latency_steps action
        # observations are already taken care of by T_slice
        if self.n_latency_steps > 0:
            action = action[self.n_latency_steps:]
        if self.relative_action:
            base_absolute_action = np.concatenate([
                obs_dict['left_robot_tcp_pose'][-1] if 'left_robot_tcp_pose' in obs_dict else np.array([]),
                obs_dict['right_robot_tcp_pose'][-1] if 'right_robot_tcp_pose' in obs_dict else np.array([])
            ], axis=-1)
            action = absolute_actions_to_relative_actions(action, base_absolute_action=base_absolute_action)

            if self.relative_tcp_obs_for_relative_action:
                for key in self.lowdim_keys:
                    if 'robot_tcp_pose' in key and 'wrt' not in key:
                        obs_dict[key]  = absolute_actions_to_relative_actions(obs_dict[key], base_absolute_action=base_absolute_action)

        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(action)
        }

        return torch_data

def test():
    import hydra
    from hydra import initialize, compose
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("eval", eval, replace=True)

    with initialize('../config'):
        cfg = hydra.compose('train_diffusion_unet_real_image_workspace',
                            overrides=['task=real_whiteboard_image'])
        OmegaConf.resolve(cfg)
        dataset = hydra.utils.instantiate(cfg.task.dataset)

    from matplotlib import pyplot as plt
    normalizer = dataset.get_normalizer()

    for i in range(len(dataset)):
        data = dataset[i]
        if i > 10:
            break

if __name__ == '__main__':
    test()
