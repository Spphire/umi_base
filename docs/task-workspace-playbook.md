# Task + Workspace Playbook

Last updated: 2026-03-18

## 1. Common Combinations

## 1.1 Primary combo (recommended default)

- Task: `q3_hang_cup`
- Workspace: `train_diffusion_unet_timm_single_frame_workspace`
- Command:
  - `python train.py --config-name train_diffusion_unet_timm_single_frame_workspace task=q3_hang_cup`

## 1.2 Mouse task combo

- Task: `q3_mouse` or `q3_mouse_384x288`
- Workspace: `train_diffusion_unet_timm_single_frame_workspace`
- Commands:
  - `python train.py --config-name train_diffusion_unet_timm_single_frame_workspace task=q3_mouse`
  - `python train.py --config-name train_diffusion_unet_timm_single_frame_workspace task=q3_mouse_384x288`

## 1.3 Secondary combo (occasional)

- Task: `q3_mouse` or `q3_hang_cup`
- Workspace: `train_diffusion_transformer_timm_single_frame_workspace`
- Command:
  - `python train.py --config-name train_diffusion_transformer_timm_single_frame_workspace task=q3_hang_cup`

## 1.4 HOMMI (custom Transformer)

- Default task: `q3_mouse` (override as needed)
- Workspace: `HOMMI`
- Command:
  - `python train.py --config-name HOMMI task=q3_mouse`

## 2. Config Reference

## 2.1 Task configs

- `q3_mouse.yaml`
  - `identifier: q3_mouse_195`
  - `image_shape: [3, 224, 224]`
  - `local_files_only: /mnt/data/users/shenyibo/umi_base/.cache/q3_mouse_dh_justresize_rawgamma/replay_buffer.zarr`
- `q3_mouse_384x288.yaml`
  - `identifier: q3_mouse_195`
  - `image_shape: [3, 288, 384]`
  - `local_files_only: /mnt/data/users/shenyibo/umi_base/.cache/q3_mouse_dh_384x288/replay_buffer.zarr`
- `q3_hang_cup.yaml`
  - `identifier: q3_hang_cup`
  - `image_shape: [3, 224, 224]`
  - `local_files_only: /mnt/data/users/shenyibo/umi_base/.cache/q3_hang_cup/replay_buffer.zarr`

## 2.2 Workspace configs (single-frame)

- `train_diffusion_unet_timm_single_frame_workspace.yaml`
  - `_target_`: `TrainDiffusionUnetTimmWorkspace`
  - `n_obs_steps: 1`
  - `optimizer.lr: 3.0e-4`
  - `obs_encoder.model_name: vit_base_patch16_dinov3.lvd1689m`
- `train_diffusion_transformer_timm_single_frame_workspace.yaml`
  - `_target_`: `TrainDiffusionTransformerTimmWorkspace`
  - `n_obs_steps: 1`
  - `optimizer.lr: 7.5e-5`
  - `obs_encoder.model_name: vit_base_patch16_dinov3.lvd1689m`
- `HOMMI.yaml`
  - `_target_`: `TrainDiffusionTransformerTimmWorkspace`
  - default `task: q3_mouse`
  - per-key `feature_aggregation` map (`eye/head` using `attn_pool`)

## 3. Daily Command Templates

## 3.1 Switch task only

- `python train.py --config-name train_diffusion_unet_timm_single_frame_workspace task=q3_hang_cup`
- `python train.py --config-name train_diffusion_unet_timm_single_frame_workspace task=q3_mouse`

## 3.2 Switch workspace (UNet -> Transformer)

- `python train.py --config-name train_diffusion_transformer_timm_single_frame_workspace task=q3_hang_cup`

## 3.3 Use Makefile defaults (current: q3_hang_cup + UNet single-frame)

- Train:
  - `make train`
- Multi-GPU train:
  - `make train_acc_amp`

## 4. Pre-run Checklist

1. Verify `task.dataset.local_files_only` exists.
2. Verify `task.dataset.identifier` matches target data.
3. Verify `image_shape` matches data post-processing output.
4. Verify `action_type` stays `head_6DOF_left_arm_6DOF_gripper_width`.
5. Verify output path under `data/outputs/${date}/...`.

## 5. Known Risks

1. UNet workspace rollout still has temporary hard-coded runner behavior.
2. Dataset wrist mask probability is hard-coded and should be tracked when changed.
3. HOMMI and baseline Transformer single-frame are not equivalent because aggregation logic differs.
