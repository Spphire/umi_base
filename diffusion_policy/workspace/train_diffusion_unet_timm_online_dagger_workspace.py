if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from torch import nn
import torch.optim as optim
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import shutil
import tempfile
import requests
import hashlib
import json
import tarfile
import lz4.frame
import threading
from datetime import timedelta
from typing import Optional, Dict, List
import torch.distributed as dist

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_timm_policy import DiffusionUnetTimmPolicy
from diffusion_policy.dataset.online_pick_and_place_image_dataset import OnlinePickAndPlaceImageDataset
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.model.common.lr_decay import param_groups_lrd
from diffusion_policy.common.checkpoint_sync import CheckpointSyncServer
from diffusion_policy.common.replay_buffer import ReplayBuffer
from post_process_scripts.post_process_data_iphone import convert_data_to_zarr
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from loguru import logger


OmegaConf.register_new_resolver("eval", eval, replace=True)


class OnlineDataFetcher:
    """
    Fetches new episodes from cloud storage (object storage service).
    Similar to CloudPickAndPlaceImageDataset but for incremental fetching.
    """
    
    def __init__(
        self,
        datacloud_endpoint: str,
        identifier: str,
        query_filter: dict,
        use_absolute_action: bool = True,
        action_type: str = 'left_arm_6DOF_gripper_width',
        temporal_downsample_ratio: int = 0,
        use_dino: bool = False,
        episode_clip_head_seconds: float = 0.0,
        episode_clip_tail_seconds: float = 0.0,
    ):
        self.datacloud_endpoint = datacloud_endpoint
        self.identifier = identifier
        self.query_filter = query_filter
        self.use_absolute_action = use_absolute_action
        self.action_type = action_type
        self.temporal_downsample_ratio = temporal_downsample_ratio
        self.use_dino = use_dino
        self.episode_clip_head_seconds = episode_clip_head_seconds
        self.episode_clip_tail_seconds = episode_clip_tail_seconds
        
        self._fetched_uuids = set()
        self._lock = threading.Lock()
    
    def fetch_new_episodes(self) -> Optional[List[Dict[str, np.ndarray]]]:
        """
        Fetch new episodes from cloud storage that haven't been fetched yet.
        Returns list of episode data dicts or None if no new data.
        """
        with self._lock:
            try:
                list_recordings_request = {
                    "identifier": self.identifier,
                    "query_filter": self.query_filter,
                    "limit": 10000,
                    "skip": 0,
                }
                url = f"{self.datacloud_endpoint}/v1/logs"
                response = requests.post(
                    url,
                    json=list_recordings_request,
                    headers={"Content-Type": "application/json"},
                    timeout=30,
                )
                
                if response.status_code != 200:
                    logger.warning(f"Failed to list recordings: {response.text}")
                    return None
                
                records = response.json().get('data', [])
                all_uuids = set(record['uuid'] for record in records)
                new_uuids = all_uuids - self._fetched_uuids
                
                if not new_uuids:
                    return None
                
                logger.info(f"Found {len(new_uuids)} new episodes to fetch")
                
                episodes_data = self._download_and_process(list(new_uuids))
                
                if episodes_data:
                    self._fetched_uuids.update(new_uuids)
                
                return episodes_data
                
            except Exception as e:
                logger.error(f"Error fetching new episodes: {e}")
                return None
    
    def _download_and_process(self, uuids: List[str]) -> Optional[List[Dict[str, np.ndarray]]]:
        """Download and process episodes into replay buffer format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filename = os.path.join(temp_dir, "downloaded_records.tar.lz4")
            
            try:
                data_request = {
                    "identifier": self.identifier,
                    "uuids": uuids,
                }
                response = requests.post(
                    f"{self.datacloud_endpoint}/v1/download_records",
                    json=data_request,
                    stream=True,
                    timeout=300,
                )
                
                if response.status_code != 200:
                    logger.error(f"Failed to download records: {response.text}")
                    return None
                
                with open(filename, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024*1024):
                        if chunk:
                            f.write(chunk)
                
                server_sha256sum = response.headers.get('X-File-SHA256')
                if server_sha256sum:
                    sha256_hash = hashlib.sha256()
                    with open(filename, 'rb') as f:
                        for chunk in iter(lambda: f.read(4096), b""):
                            sha256_hash.update(chunk)
                    file_sha256sum = sha256_hash.hexdigest()
                    if file_sha256sum != server_sha256sum:
                        logger.error(f"SHA256 checksum mismatch")
                        return None
                
                extract_dir = os.path.join(temp_dir, "downloaded_records")
                os.makedirs(extract_dir, exist_ok=True)
                
                with lz4.frame.open(filename, 'rb') as lz4_file:
                    with tarfile.open(fileobj=lz4_file, mode='r|') as tar:
                        tar.extractall(path=extract_dir)
                
                from diffusion_policy.common.data_models import ActionType
                zarr_dir = os.path.join(temp_dir, "zarr_output")
                zarr_path = convert_data_to_zarr(
                    input_dir=extract_dir,
                    output_dir=zarr_dir,
                    temporal_downsample_ratio=self.temporal_downsample_ratio,
                    use_absolute_action=self.use_absolute_action,
                    action_type=ActionType[self.action_type],
                    use_dino=self.use_dino,
                    episode_clip_head_seconds=self.episode_clip_head_seconds,
                    episode_clip_tail_seconds=self.episode_clip_tail_seconds,
                )
                
                if zarr_path and os.path.exists(zarr_path):
                    replay_buffer = ReplayBuffer.copy_from_path(zarr_path)
                    episodes = []
                    for i in range(replay_buffer.n_episodes):
                        episode_data = replay_buffer.get_episode(i, copy=True)
                        episodes.append(episode_data)
                    logger.info(f"Processed {len(episodes)} episodes from downloaded data")
                    return episodes
                
                return None
                
            except Exception as e:
                logger.error(f"Error downloading/processing episodes: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return None
    
    @property
    def fetched_count(self) -> int:
        with self._lock:
            return len(self._fetched_uuids)


def broadcast_episodes(episodes: Optional[List[Dict[str, np.ndarray]]], accelerator) -> Optional[List[Dict[str, np.ndarray]]]:
    """
    Broadcast episode data from main process to all other processes.
    """
    if accelerator.state.num_processes <= 1:
        return episodes
    
    device = accelerator.device
    
    # Broadcast whether there are episodes to send
    has_episodes = torch.tensor([1 if episodes and len(episodes) > 0 else 0], device=device)
    dist.broadcast(has_episodes, src=0)
    
    if has_episodes.item() == 0:
        return None
    
    if accelerator.is_main_process:
        # Serialize episodes
        import io
        import pickle
        buffer = io.BytesIO()
        pickle.dump(episodes, buffer)
        data_bytes = buffer.getvalue()
        data_size = torch.tensor([len(data_bytes)], device=device)
    else:
        data_size = torch.tensor([0], device=device)
    
    # Broadcast size
    dist.broadcast(data_size, src=0)
    size = data_size.item()
    
    if accelerator.is_main_process:
        data_tensor = torch.frombuffer(bytearray(data_bytes), dtype=torch.uint8).to(device)
    else:
        data_tensor = torch.zeros(size, dtype=torch.uint8, device=device)
    
    # Broadcast data
    dist.broadcast(data_tensor, src=0)
    
    if not accelerator.is_main_process:
        import io
        import pickle
        data_bytes = bytes(data_tensor.cpu().numpy())
        buffer = io.BytesIO(data_bytes)
        episodes = pickle.load(buffer)
    
    return episodes


def sample_non_overlapping_batch_indices(
    dataset: 'OnlinePickAndPlaceImageDataset',
    batch_size_per_gpu: int,
    accelerator
) -> List[tuple]:
    """
    Sample batch indices ensuring no overlap across GPUs.
    
    Main process samples (batch_size_per_gpu * num_gpus) unique indices
    without replacement, then each GPU gets a distinct slice.
    
    Returns list of (is_online, idx) tuples for this GPU.
    """
    num_gpus = accelerator.state.num_processes
    total_batch_size = batch_size_per_gpu * num_gpus
    device = accelerator.device
    
    if num_gpus <= 1:
        # Single GPU: sample without replacement
        return dataset.sample_batch_indices(batch_size_per_gpu, replace=False)
    
    # Main process samples all indices without replacement
    if accelerator.is_main_process:
        all_indices = dataset.sample_batch_indices(total_batch_size, replace=False)
        # Convert to tensor: [is_online (0/1), idx]
        indices_tensor = torch.tensor(
            [[1 if is_online else 0, idx] for is_online, idx in all_indices],
            dtype=torch.long, device=device
        )
    else:
        indices_tensor = torch.zeros((total_batch_size, 2), dtype=torch.long, device=device)
    
    # Broadcast all indices to all GPUs
    dist.broadcast(indices_tensor, src=0)
    
    # Each GPU takes its own slice
    rank = accelerator.process_index
    start_idx = rank * batch_size_per_gpu
    end_idx = start_idx + batch_size_per_gpu
    my_indices = indices_tensor[start_idx:end_idx]
    
    return [(bool(row[0].item()), row[1].item()) for row in my_indices]


def aggregate_losses_and_update_weight(
    batch_loss: float,
    n_online: int,
    n_offline: int,
    dataset: 'OnlinePickAndPlaceImageDataset',
    accelerator
) -> Dict[str, float]:
    """
    Aggregate online/offline losses from all GPUs and update sampling weight.
    
    Each GPU reports:
    - Its batch loss
    - Number of online samples in its batch
    - Number of offline samples in its batch
    
    We aggregate across all GPUs to compute the true online/offline loss averages,
    then update the adaptive sampling weight.
    
    Returns dict with aggregated stats.
    """
    device = accelerator.device
    
    # Pack local stats: [online_loss_sum, online_count, offline_loss_sum, offline_count]
    # Loss is weighted by sample count for proper averaging
    online_loss_sum = batch_loss * n_online if n_online > 0 else 0.0
    offline_loss_sum = batch_loss * n_offline if n_offline > 0 else 0.0
    
    local_stats = torch.tensor(
        [online_loss_sum, float(n_online), offline_loss_sum, float(n_offline)],
        dtype=torch.float64, device=device
    )
    
    if accelerator.state.num_processes > 1:
        # Sum across all GPUs
        dist.all_reduce(local_stats, op=dist.ReduceOp.SUM)
    
    total_online_loss_sum = local_stats[0].item()
    total_online_count = local_stats[1].item()
    total_offline_loss_sum = local_stats[2].item()
    total_offline_count = local_stats[3].item()
    
    # Compute average losses
    avg_online_loss = total_online_loss_sum / max(total_online_count, 1)
    avg_offline_loss = total_offline_loss_sum / max(total_offline_count, 1)
    
    # Update adaptive sampler with aggregated losses
    if total_online_count > 0:
        dataset.adaptive_sampler.add_loss(avg_online_loss, is_online=True)
    if total_offline_count > 0:
        dataset.adaptive_sampler.add_loss(avg_offline_loss, is_online=False)
    
    # Update weight (all GPUs do this with same aggregated data, so result is consistent)
    new_weight = dataset.adaptive_sampler.update_weights()
    
    return {
        'online_loss': avg_online_loss,
        'offline_loss': avg_offline_loss,
        'online_count': total_online_count,
        'offline_count': total_offline_count,
        'online_weight': new_weight,
    }


class TrainDiffusionUnetTimmOnlineDaggerWorkspace(BaseWorkspace):
    """
    Online DAgger training workspace implementing SOP-style training loop.
    
    Features:
    - Step-based training loop (no epoch concept)
    - Periodic data fetching from cloud storage
    - Adaptive sampling between online and offline data
    - Checkpoint synchronization to inference server
    - wandb logging of sampling weights and losses
    - Multi-GPU synchronized training
    """
    include_keys = ['global_step', 'adaptive_sampler_state']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model: DiffusionUnetTimmPolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionUnetTimmPolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        if 'timm' in cfg.policy.obs_encoder._target_:
            if cfg.training.layer_decay < 1.0:
                assert not cfg.policy.obs_encoder.use_lora
                assert not cfg.policy.obs_encoder.share_rgb_model
                obs_encorder_param_groups = param_groups_lrd(
                    self.model.obs_encoder,
                    shape_meta=cfg.shape_meta,
                    weight_decay=cfg.optimizer.encoder_weight_decay,
                    no_weight_decay_list=self.model.obs_encoder.no_weight_decay(),
                    layer_decay=cfg.training.layer_decay
                )
                count = 0
                for group in obs_encorder_param_groups:
                    count += len(group['params'])
                if cfg.policy.obs_encoder.feature_aggregation == 'map':
                    obs_encorder_param_groups.extend([{'params': self.model.obs_encoder.attn_pool.parameters()}])
                    for _ in self.model.obs_encoder.attn_pool.parameters():
                        count += 1
                print(f'obs_encorder params: {count}')
                param_groups = [{'params': self.model.model.parameters()}]
                param_groups.extend(obs_encorder_param_groups)
            else:
                obs_encorder_lr = cfg.optimizer.lr
                if cfg.policy.obs_encoder.pretrained and not cfg.policy.obs_encoder.use_lora:
                    obs_encorder_lr *= cfg.training.encoder_lr_coefficient
                    print('==> reduce pretrained obs_encorder\'s lr')
                obs_encorder_params = list()
                for param in self.model.obs_encoder.parameters():
                    if param.requires_grad:
                        obs_encorder_params.append(param)
                print(f'obs_encorder params: {len(obs_encorder_params)}')
                param_groups = [
                    {'params': self.model.model.parameters()},
                    {'params': obs_encorder_params, 'lr': obs_encorder_lr}
                ]
            optimizer_cfg = OmegaConf.to_container(cfg.optimizer, resolve=True)
            optimizer_cfg.pop('_target_')
            if 'encoder_weight_decay' in optimizer_cfg.keys():
                optimizer_cfg.pop('encoder_weight_decay')
            self.optimizer = torch.optim.AdamW(
                params=param_groups,
                **optimizer_cfg
            )
        else:
            optimizer_cfg = OmegaConf.to_container(cfg.optimizer, resolve=True)
            optimizer_cfg.pop('encoder_weight_decay')
            accelerator = Accelerator()
            cuda_count = accelerator.num_processes
            print(f"Number of available CUDA devices: {cuda_count}.")
            print(f"Original learning rate: {optimizer_cfg['lr']}")
            optimizer_cfg['lr'] = optimizer_cfg['lr'] * cuda_count
            print(f"Updated learning rate: {optimizer_cfg['lr']}")
            self.optimizer = hydra.utils.instantiate(
                optimizer_cfg, params=self.model.parameters())

        self.global_step = 0
        self.adaptive_sampler_state = None

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=360000))
        accelerator = Accelerator(log_with='wandb', kwargs_handlers=[kwargs])
        wandb_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
        wandb_cfg.pop('project')
        accelerator.init_trackers(
            project_name=cfg.logging.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": wandb_cfg},
        )

        # Online DAgger requires an initial checkpoint to load normalizer from
        # The normalizer should come from the original trained policy, not recomputed from data
        initial_ckpt_path = cfg.online_training.initial_checkpoint_path
        assert initial_ckpt_path is not None, "Online DAgger requires initial_checkpoint to load normalizer"
        assert os.path.isfile(initial_ckpt_path), f"Initial checkpoint not found: {initial_ckpt_path}"
        
        accelerator.print(f"Loading initial checkpoint from {initial_ckpt_path}")
        # do not load optimizer and pickle payloads 
        if cfg.training.resume:
            self.load_checkpoint(path=initial_ckpt_path)
            self.global_step += 1
        else:
            self.load_checkpoint(path=initial_ckpt_path, exclude_keys=['optimizer'], include_keys=[])

        # Get normalizer from the loaded model
        normalizer = self.model.normalizer
        assert normalizer is not None, "Normalizer not found in initial checkpoint"
        accelerator.print("Normalizer loaded from initial checkpoint")
        
        dataset: OnlinePickAndPlaceImageDataset
        if accelerator.is_main_process:
            dataset = hydra.utils.instantiate(cfg.task.dataset)
        accelerator.wait_for_everyone()
        if not accelerator.is_main_process:
            dataset = hydra.utils.instantiate(cfg.task.dataset)
            
        assert isinstance(dataset, OnlinePickAndPlaceImageDataset)
        
        # Restore adaptive sampler state if resuming
        if self.adaptive_sampler_state is not None:
            dataset.adaptive_sampler.load_state_dict(self.adaptive_sampler_state)
            accelerator.print(f"Restored adaptive sampler state: online_weight={dataset.adaptive_sampler.online_weight:.4f}")

        if accelerator.is_main_process:
            os.makedirs(self.output_dir, exist_ok=True)

        # Normalizer already loaded from checkpoint, just ensure ema_model has it too
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        online_cfg = cfg.online_training
        data_fetcher = OnlineDataFetcher(
            datacloud_endpoint=online_cfg.datacloud_endpoint,
            identifier=online_cfg.identifier,
            query_filter=OmegaConf.to_container(online_cfg.query_filter, resolve=True) if online_cfg.query_filter else {},
            use_absolute_action=online_cfg.get('use_absolute_action', True),
            action_type=online_cfg.get('action_type', 'left_arm_6DOF_gripper_width'),
            temporal_downsample_ratio=online_cfg.get('temporal_downsample_ratio', 0),
            use_dino=online_cfg.get('use_dino', False),
            episode_clip_head_seconds=online_cfg.get('episode_clip_head_seconds', 0.0),
            episode_clip_tail_seconds=online_cfg.get('episode_clip_tail_seconds', 0.0),
        )

        checkpoint_sync = CheckpointSyncServer(
            inference_server_url=online_cfg.inference_server_url,
            timeout=online_cfg.get('sync_timeout', 120),
            max_retries=online_cfg.get('sync_retries', 3),
        )

        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=cfg.training.num_training_steps \
                // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step-1
        )

        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        self.model, self.ema_model, self.optimizer, lr_scheduler = accelerator.prepare(
            self.model, self.ema_model, self.optimizer, lr_scheduler
        )
        
        if accelerator.state.num_processes > 1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                accelerator.unwrap_model(self.model),
                device_ids=[self.model.device],
                find_unused_parameters=True
            )

        device = accelerator.device

        if cfg.training.debug:
            cfg.training.num_training_steps = 20
            online_cfg.fetch_interval = 1
            online_cfg.sync_interval = 5
            cfg.training.checkpoint_every = 1
            cfg.training.log_every = 1
            cfg.training.eval_every = 1

        if cfg.training.freeze_encoder:
            unwrapped_model = accelerator.unwrap_model(self.model)
            unwrapped_model.obs_encoder.eval()
            unwrapped_model.obs_encoder.requires_grad_(False)

        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            with tqdm.tqdm(
                initial=self.global_step,
                total=cfg.training.num_training_steps,
                desc="Training",
                mininterval=cfg.training.tqdm_interval_sec
            ) as pbar:
                while self.global_step < cfg.training.num_training_steps:
                    # === Fetch new episodes (main process only, then broadcast to all) ===
                    new_episodes = None
                    if self.global_step > 0 and self.global_step % online_cfg.fetch_interval == 0:
                        if accelerator.is_main_process:
                            new_episodes = data_fetcher.fetch_new_episodes()
                        
                        new_episodes = broadcast_episodes(new_episodes, accelerator)
                        
                        if new_episodes:
                            dataset.append_episodes(new_episodes)
                            if accelerator.is_main_process:
                                logger.info(f"Added {len(new_episodes)} new episodes at step {self.global_step}")

                    # === Sample non-overlapping batch indices across GPUs ===
                    batch_indices = sample_non_overlapping_batch_indices(
                        dataset=dataset,
                        batch_size_per_gpu=cfg.dataloader.batch_size,
                        accelerator=accelerator,
                    )
                    
                    batch_data = []
                    batch_is_online = []
                    for is_online, idx in batch_indices:
                        item = dataset.get_item_by_source(is_online, idx)
                        batch_data.append(item)
                        batch_is_online.append(is_online)
                    
                    batch = {
                        'obs': dict_apply(
                            {k: torch.stack([d['obs'][k] for d in batch_data]) for k in batch_data[0]['obs'].keys()},
                            lambda x: x.to(device)
                        ),
                        'action': torch.stack([d['action'] for d in batch_data]).to(device)
                    }

                    raw_loss = self.model(batch)
                    loss = raw_loss / cfg.training.gradient_accumulate_every
                    accelerator.backward(loss)

                    if self.global_step % cfg.training.gradient_accumulate_every == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        lr_scheduler.step()
                    
                    if cfg.training.use_ema:
                        ema.step(accelerator.unwrap_model(self.model))

                    raw_loss_cpu = raw_loss.item()
                    pbar.set_postfix(loss=raw_loss_cpu, refresh=False)

                    # === Aggregate losses from all GPUs and update sampling weight ===
                    n_online = sum(batch_is_online)
                    n_offline = len(batch_is_online) - n_online
                    
                    agg_stats = aggregate_losses_and_update_weight(
                        batch_loss=raw_loss_cpu,
                        n_online=n_online,
                        n_offline=n_offline,
                        dataset=dataset,
                        accelerator=accelerator,
                    )

                    # === Logging ===
                    if self.global_step % cfg.training.log_every == 0:
                        sampling_stats = dataset.get_sampling_stats()
                        
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'lr': lr_scheduler.get_last_lr()[0],
                            'online_weight': sampling_stats['online_weight'],
                            'offline_weight': sampling_stats['offline_weight'],
                            'online_loss_mean': sampling_stats['online_loss_mean'],
                            'offline_loss_mean': sampling_stats['offline_loss_mean'],
                            'online_episodes': dataset.online_episodes_count,
                            'offline_episodes': dataset.offline_episodes_count,
                            'total_steps': dataset.total_steps,
                            'fetched_episodes': data_fetcher.fetched_count,
                        }
                        
                        accelerator.log(step_log, step=self.global_step)
                        json_logger.log(step_log)

                    # === Sampling ===
                    if self.global_step > 0 and self.global_step % cfg.training.sample_every == 0:
                        policy = accelerator.unwrap_model(self.model)
                        if cfg.training.use_ema:
                            policy = self.ema_model
                        policy.eval()

                        with torch.no_grad():
                            sampling_log = {}
                            sampling_batch_size = cfg.dataloader.batch_size
                            
                            # Sample from offline data
                            if dataset.offline_sampler is not None and len(dataset.offline_sampler) > 0:
                                offline_indices = dataset.sample_offline_indices(sampling_batch_size)
                                offline_batch_data = [dataset.get_item_by_source(False, idx) for idx in offline_indices]
                                
                                offline_batch = {
                                    'obs': dict_apply(
                                        {k: torch.stack([d['obs'][k] for d in offline_batch_data]) for k in offline_batch_data[0]['obs'].keys()},
                                        lambda x: x.to(device)
                                    ),
                                    'action': torch.stack([d['action'] for d in offline_batch_data]).to(device)
                                }
                                
                                offline_result = policy.predict_action(offline_batch['obs'])
                                offline_pred = offline_result['action_pred']
                                offline_gt = offline_batch['action']
                                
                                all_offline_preds, all_offline_gt = accelerator.gather_for_metrics((offline_pred, offline_gt))
                                
                                if accelerator.is_main_process:
                                    offline_mse = torch.nn.functional.mse_loss(all_offline_preds, all_offline_gt)
                                    sampling_log['train_action_mse_error_offline'] = offline_mse.item()
                            
                            # Sample from online data (if available)
                            if dataset.online_sampler is not None and len(dataset.online_sampler) > 0:
                                online_indices = dataset.sample_online_indices(sampling_batch_size)
                                online_batch_data = [dataset.get_item_by_source(True, idx) for idx in online_indices]
                                
                                online_batch = {
                                    'obs': dict_apply(
                                        {k: torch.stack([d['obs'][k] for d in online_batch_data]) for k in online_batch_data[0]['obs'].keys()},
                                        lambda x: x.to(device)
                                    ),
                                    'action': torch.stack([d['action'] for d in online_batch_data]).to(device)
                                }
                                
                                online_result = policy.predict_action(online_batch['obs'])
                                online_pred = online_result['action_pred']
                                online_gt = online_batch['action']
                                
                                all_online_preds, all_online_gt = accelerator.gather_for_metrics((online_pred, online_gt))
                                
                                if accelerator.is_main_process:
                                    online_mse = torch.nn.functional.mse_loss(all_online_preds, all_online_gt)
                                    sampling_log['train_action_mse_error_online'] = online_mse.item()
                            
                            if accelerator.is_main_process and sampling_log:
                                accelerator.log(sampling_log, step=self.global_step)
                                json_logger.log(sampling_log)

                        policy.train()

                    # === Save checkpoint ===
                    if self.global_step > 0 and self.global_step % cfg.training.checkpoint_every == 0:
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            logger.info(f"Saving checkpoint at step {self.global_step}")
                            model_ddp = self.model
                            self.model = accelerator.unwrap_model(self.model)
                            
                            # Save adaptive sampler state for resume
                            self.adaptive_sampler_state = dataset.adaptive_sampler.state_dict()

                            if cfg.checkpoint.save_last_ckpt:
                                self.save_checkpoint(use_thread=False)
                            if cfg.checkpoint.save_last_snapshot:
                                self.save_snapshot()

                            sampling_stats = dataset.get_sampling_stats()
                            metric_dict = {
                                'global_step': self.global_step,
                                'train_loss': raw_loss_cpu,
                                'online_weight': sampling_stats['online_weight'],
                            }
                            
                            topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                            if topk_ckpt_path is not None:
                                self.save_checkpoint(path=topk_ckpt_path, use_thread=False)

                            self.model = model_ddp
                    
                    # === Checkpoint sync to inference server ===
                    if self.global_step > 0 and self.global_step % online_cfg.sync_interval == 0:
                        if accelerator.is_main_process:
                            logger.info(f"Syncing checkpoint to inference server at step {self.global_step}")
                            latest_ckpt_path = self.get_checkpoint_path()
                            if latest_ckpt_path.is_file():
                                success = checkpoint_sync.push_checkpoint(
                                    checkpoint_path=str(latest_ckpt_path),
                                    workspace_config=cfg.name,
                                    task_config=cfg.task_name,
                                    metadata={
                                        'global_step': self.global_step,
                                        'online_weight': sampling_stats['online_weight'],
                                    }
                                )
                                
                                if success:
                                    logger.info(f"Checkpoint synced to inference server at step {self.global_step}")
                                else:
                                    logger.warning(f"Failed to sync checkpoint at step {self.global_step}")
                        
                    self.global_step += 1
                    pbar.update(1)

        checkpoint_sync.close()
        accelerator.end_training()
