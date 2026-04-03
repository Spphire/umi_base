from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.transformer_for_action_diffusion import TransformerForActionDiffusion
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.vision.transformer_obs_encoder import TransformerObsEncoder


class DiffusionTransformerTimmPolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            obs_encoder: TransformerObsEncoder,
            num_inference_steps=None,
            input_pertub=0.1,
            # arch
            n_layer=7,
            n_head=8,
            n_emb=768,
            p_drop_attn=0.1,
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shapes
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        action_horizon = shape_meta['action']['horizon']
        
        obs_shape = obs_encoder.output_shape()
        assert obs_shape[-1] == n_emb
        obs_tokens = obs_shape[-2]
        
        model = TransformerForActionDiffusion(
            input_dim=action_dim,
            output_dim=action_dim,
            action_horizon=action_horizon,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            max_cond_tokens=obs_tokens+1, # obs tokens + 1 token for time
            p_drop_attn=p_drop_attn
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.input_pertub = input_pertub
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            cond=None, generator=None,
            return_attention=False,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler
        attention_steps = list() if return_attention else None
        if attention_steps is not None and hasattr(model, 'set_capture_cross_attention'):
            model.set_capture_cross_attention(True)
            model.clear_cross_attention_cache()

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, cond)
            if attention_steps is not None and hasattr(model, 'pop_last_cross_attention_weights'):
                timestep = int(t.item()) if torch.is_tensor(t) else int(t)
                attention_steps.append({
                    'diffusion_timestep': timestep,
                    'layers': model.pop_last_cross_attention_weights()
                })

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        if attention_steps is not None and hasattr(model, 'set_capture_cross_attention'):
            model.set_capture_cross_attention(False)
            model.clear_cross_attention_cache()
            return trajectory, attention_steps
        return trajectory


    def _summarize_cross_attention(self, attention_steps):
        if not attention_steps:
            return None
        if not hasattr(self.obs_encoder, 'get_token_slices') or not hasattr(self.obs_encoder, 'get_token_layout'):
            return None

        token_slices = self.obs_encoder.get_token_slices(include_time_token=True)
        token_layout = self.obs_encoder.get_token_layout(include_time_token=True)
        aggregate_by_key = {key: list() for key in token_slices.keys()}
        per_step = list()

        for step in attention_steps:
            layer_maps = list()
            for layer_attn in step.get('layers', list()):
                if layer_attn is None:
                    continue
                layer_map = layer_attn.detach().float().mean(dim=1).mean(dim=1)
                layer_maps.append(layer_map)
            if len(layer_maps) == 0:
                continue

            step_map = torch.stack(layer_maps, dim=0).mean(dim=0).mean(dim=0)
            step_attention = dict()
            for key, token_slice in token_slices.items():
                step_attention[key] = float(step_map[token_slice].sum().item())
                aggregate_by_key[key].append(step_attention[key])
            per_step.append({
                'diffusion_timestep': int(step['diffusion_timestep']),
                'mean_attention_by_key': step_attention
            })

        if len(per_step) == 0:
            return None

        aggregate_mean_by_key = dict()
        for key, values in aggregate_by_key.items():
            aggregate_mean_by_key[key] = float(np.mean(values)) if len(values) > 0 else 0.0

        return {
            'token_layout': token_layout,
            'aggregate_mean_by_key': aggregate_mean_by_key,
            'per_step': per_step
        }

    def predict_action(self, obs_dict: Dict[str, torch.Tensor], return_attention: bool=False) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]
        
        # process input
        obs_tokens = self.obs_encoder(nobs)
        # (B, N, n_emb)
        
        # empty data for action
        cond_data = torch.zeros(size=(B, self.action_horizon, self.action_dim), device=self.device, dtype=self.dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        
        # run sampling
        sample_result = self.conditional_sample(
            condition_data=cond_data, 
            condition_mask=cond_mask,
            cond=obs_tokens,
            return_attention=return_attention,
            **self.kwargs)
        if return_attention:
            nsample, attention_steps = sample_result
        else:
            nsample = sample_result
            attention_steps = None
        
        # unnormalize prediction
        assert nsample.shape == (B, self.action_horizon, self.action_dim)
        action_pred = self.normalizer['action'].unnormalize(nsample)

        result = {
            'action': action_pred,
            'action_pred': action_pred
        }
        if return_attention:
            result['cross_attention_summary'] = self._summarize_cross_attention(attention_steps)
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
            self, 
            lr: float,
            weight_decay: float,
            obs_encoder_lr: float,
            obs_encoder_weight_decay: float,
            betas: Tuple[float, float]
        ) -> torch.optim.Optimizer:
        optim_groups = self.model.get_optim_groups(
            weight_decay=weight_decay)
        
        backbone_params = list()
        other_obs_params = list()
        for key, value in self.obs_encoder.named_parameters():
            if not value.requires_grad:
                continue
            if key.startswith('key_model_map'):
                backbone_params.append(value)
            else:
                other_obs_params.append(value)
        optim_groups.append({
            "params": backbone_params,
            "weight_decay": obs_encoder_weight_decay,
            "lr": obs_encoder_lr # for fine tuning
        })
        optim_groups.append({
            "params": other_obs_params,
            "weight_decay": obs_encoder_weight_decay
        })
        optimizer = torch.optim.AdamW(
            optim_groups, lr=lr, betas=betas
        )
        return optimizer

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        trajectory = nactions
        
        # process input
        obs_tokens = self.obs_encoder(nobs)
        # (B, N, n_emb)
        
        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        # input perturbation by adding additonal noise to alleviate exposure bias
        # reference: https://github.com/forever208/DDPM-IP
        noise_new = noise + self.input_pertub * torch.randn(trajectory.shape, device=trajectory.device)

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (nactions.shape[0],), device=trajectory.device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise_new, timesteps)
        
        # Predict the noise residual
        pred = self.model(
            noisy_trajectory,
            timesteps, 
            cond=obs_tokens
        )

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()

        return loss

    def forward(self, batch):
        return self.compute_loss(batch)
