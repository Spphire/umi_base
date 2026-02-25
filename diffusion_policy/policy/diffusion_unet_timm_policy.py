from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from loguru import logger

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.timm_obs_encoder import TimmObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply


class DiffusionUnetTimmPolicy(BaseImagePolicy):
    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: DDPMScheduler,
        obs_encoder: TimmObsEncoder,
        num_inference_steps=None,
        obs_as_global_cond=True,
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
        input_pertub=0.1,
        inpaint_fixed_action_prefix=False,
        train_diffusion_n_samples=1,
        # parameters passed to step
        **kwargs,
    ):
        super().__init__()

        # parse shapes
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        action_horizon = shape_meta["action"]["horizon"]
        # get feature dim
        obs_feature_dim = np.prod(obs_encoder.output_shape())

        # create diffusion model
        assert obs_as_global_cond
        input_dim = action_dim
        global_cond_dim = obs_feature_dim

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon  # used for training
        self.obs_as_global_cond = obs_as_global_cond
        self.input_pertub = input_pertub
        self.inpaint_fixed_action_prefix = inpaint_fixed_action_prefix
        self.train_diffusion_n_samples = int(train_diffusion_n_samples)
        self.kwargs = kwargs

        # [eval_flexiv compability]
        self.horizon = action_horizon
        # []

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    # ========= inference  ============
    def conditional_sample(
        self,
        condition_data,
        condition_mask,
        local_cond=None,
        global_cond=None,
        generator=None,
        # keyword arguments to scheduler.step
        **kwargs,
    ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(
                trajectory, t, local_cond=local_cond, global_cond=global_cond
            )

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, generator=generator, **kwargs
            ).prev_sample

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory

    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        fixed_action_prefix: torch.Tensor = None,
        debug=None,
    ) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        fixed_action_prefix: unnormalized action prefix
        result: must include "action" key
        """
        assert "past_action" not in obs_dict  # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]

        # condition through global feature
        global_cond = self.obs_encoder(nobs)
        # [debug, !!!]
        # global_cond = torch.from_numpy(np.load("ignore-/0825-dino-inference/1/global_cond_0.npy")).to(self.device)

        # Removed debug code that saves tensor data to hardcoded paths.

        # empty data for action
        cond_data = torch.zeros(
            size=(B, self.action_horizon, self.action_dim),
            device=self.device,
            dtype=self.dtype,
        )
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        if fixed_action_prefix is not None and self.inpaint_fixed_action_prefix:
            n_fixed_steps = fixed_action_prefix.shape[1]
            cond_data[:, :n_fixed_steps] = fixed_action_prefix
            cond_mask[:, :n_fixed_steps] = True
            cond_data = self.normalizer["action"].normalize(cond_data)

        # run sampling
        nsample = self.conditional_sample(
            condition_data=cond_data,
            condition_mask=cond_mask,
            local_cond=None,
            global_cond=global_cond,
            **self.kwargs,
        )

        # unnormalize prediction
        assert nsample.shape == (B, self.action_horizon, self.action_dim)
        action_pred = self.normalizer["action"].unnormalize(nsample)

        result = {"action": action_pred, "action_pred": action_pred}
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch, return_per_sample_loss=False):
        def _debug_tensor(name, t):
            if not torch.is_tensor(t):
                return
            if torch.isfinite(t).all():
                return
            t_cpu = t.detach().float().cpu()
            print(
                f"[NaN Debug][{name}] min={t_cpu.min().item():.6f} max={t_cpu.max().item():.6f} "
                f"finite={torch.isfinite(t_cpu).all().item()} shape={tuple(t_cpu.shape)}",
                flush=True
            )
            raise RuntimeError(f"NaN/Inf detected in {name}")

        # normalize input
        assert "valid_mask" not in batch
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        # check normalized inputs
        if isinstance(nobs, dict):
            for k, v in nobs.items():
                _debug_tensor(f"nobs[{k}]", v)
        _debug_tensor("nactions", nactions)

        original_batch_size = nactions.shape[0]
        assert self.obs_as_global_cond
        global_cond = self.obs_encoder(nobs)
        _debug_tensor("global_cond", global_cond)

        # train on multiple diffusion samples per obs
        if self.train_diffusion_n_samples != 1:
            # repeat obs features and actions multiple times along the batch dimension
            # each sample will later have a different noise sample, effecty training
            # more diffusion steps per each obs encoder forward pass
            global_cond = torch.repeat_interleave(
                global_cond, repeats=self.train_diffusion_n_samples, dim=0
            )
            nactions = torch.repeat_interleave(
                nactions, repeats=self.train_diffusion_n_samples, dim=0
            )

        trajectory = nactions
        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        _debug_tensor("noise", noise)
        # input perturbation by adding additonal noise to alleviate exposure bias
        # reference: https://github.com/forever208/DDPM-IP
        noise_new = noise + self.input_pertub * torch.randn(
            trajectory.shape, device=trajectory.device
        )
        _debug_tensor("noise_new", noise_new)

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (nactions.shape[0],),
            device=trajectory.device,
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise_new, timesteps
        )
        _debug_tensor("noisy_trajectory", noisy_trajectory)

        # Predict the noise residual
        pred = self.model(
            noisy_trajectory, timesteps, local_cond=None, global_cond=global_cond
        )
        _debug_tensor("pred", pred)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        elif pred_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(trajectory, noise, timesteps)
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")
        _debug_tensor("target", target)

        # loss = F.mse_loss(pred, target, reduction="none")
        # loss = loss.type(loss.dtype)
        # _debug_tensor("loss", loss)
        # # per-sample loss: shape [batch_size] (or [batch_size * n_samples] if repeated)
        # per_sample_loss = reduce(loss, "b ... -> b (...)", "mean").mean(dim=-1)

        # 前10维做MSE，后面做分类loss
        mse_dim = 10
        mse_loss = F.mse_loss(pred[..., :mse_dim], target[..., :mse_dim], reduction="none")
        mse_loss = mse_loss.type(mse_loss.dtype)
        _debug_tensor("mse_loss", mse_loss)
        mse_per_sample = reduce(mse_loss, "b ... -> b (...)", "mean").mean(dim=-1)

        if pred.shape[-1] > mse_dim:
            class_logits = pred[..., mse_dim:].mean(dim=-2)
            target_class = target[..., mse_dim:].mean(dim=-2).argmax(dim=-1)
            ce_loss = F.cross_entropy(class_logits, target_class, reduction="none")
            ce_loss = ce_loss.type(ce_loss.dtype)
            _debug_tensor("ce_loss", ce_loss)
            ce_per_sample = reduce(ce_loss, "b ... -> b (...)", "mean").mean(dim=-1)
        else:
            ce_per_sample = torch.zeros_like(mse_per_sample)

        # 合并loss（可加权，这里简单相加）
        per_sample_loss = mse_per_sample + ce_per_sample*0.01

        if self.train_diffusion_n_samples != 1:
            # average across repeated samples for each original batch item
            per_sample_loss = per_sample_loss.view(
                original_batch_size, self.train_diffusion_n_samples
            ).mean(dim=-1)

        if return_per_sample_loss:
            return per_sample_loss

        return per_sample_loss.mean()

    def forward(self, batch, return_per_sample_loss=False):
        return self.compute_loss(batch, return_per_sample_loss=return_per_sample_loss)
