# Project Analysis (Training and Deployment)

Last updated: 2026-03-18

## 1. Scope

`umi_base` is a Hydra-driven robot imitation learning codebase with a diffusion policy stack.
The main loop is:

1. Data collection and post-processing (`record_data.py`, `post_process_data.py`)
2. Policy training (`train.py` + `diffusion_policy/config/*.yaml`)
3. Real-robot inference/evaluation (`eval_real_robot_flexiv.py` + `task.env_runner`)

Current `main` (`ba8ef9f`) includes:

- `diffusion_policy/config/task/q3_hang_cup.yaml`
- `diffusion_policy/config/task/q3_mouse_384x288.yaml`
- `diffusion_policy/config/HOMMI.yaml`

## 2. Entry and Config Resolution

## 2.1 Training Entry

- Script entry: `train.py`
- Typical triggers: `Makefile` targets `train`, `train_acc`, `train_acc_amp`
- Runtime override style: `python train.py --config-name ${WKSPACE} task=${TASK}`

Resolution chain:

1. `Makefile` passes `WKSPACE` and `TASK`
2. Hydra loads `diffusion_policy/config/${WKSPACE}.yaml`
3. Workspace config `defaults` then loads `task: ${TASK}`
4. `cfg._target_` selects the workspace class (UNet or Transformer)
5. Workspace instantiates `cfg.policy`, `cfg.task.dataset`, `cfg.task.env_runner`

## 2.2 Current Makefile Defaults (2026-03-18)

- `TASK := q3_hang_cup`
- `WKSPACE := train_diffusion_unet_timm_single_frame_workspace`

This matches the intended workflow: mostly UNet single-frame, occasionally Transformer single-frame.

## 3. Layered Modules

## 3.1 Config Layer (`diffusion_policy/config`)

- `train_diffusion_unet_timm_single_frame_workspace.yaml`
- `train_diffusion_transformer_timm_single_frame_workspace.yaml`
- `HOMMI.yaml` (custom Transformer config)
- `task/q3_mouse*.yaml`, `task/q3_hang_cup.yaml`

## 3.2 Workspace Layer (`diffusion_policy/workspace`)

- `train_diffusion_unet_timm_workspace.py`
- `train_diffusion_transformer_timm_workspace.py`

Responsibilities:
training loop, optimizer/lr scheduler, checkpointing, logging, validation, rollout.

## 3.3 Dataset Layer (`diffusion_policy/dataset`)

- `cloud_pick_and_place_image_head_dataset.py`
- `real_pick_and_place_image_head_dataset.py`

`q3_*` tasks are served by cloud-record-based zarr datasets, with `local_files_only` for cache-first loading.

## 3.4 Runner Layer (`diffusion_policy/env_runner`)

- `timm_image_runner.py` for real robot single-arm inference (timing, buffering, action conversion)
- `robomimic_runner.py` for robomimic simulation evaluation

## 4. Current Task/Workspace Characteristics

## 4.1 Shared Pattern for `q3_mouse` and `q3_hang_cup`

- Observation keys: `left_wrist_img`, `left_eye_img`, `left_robot_tcp_pose`
- Action shape: 10D (`head_6DOF_left_arm_6DOF_gripper_width`)
- Dataset class: `CloudPickAndPlaceImageHeadDataset`
- Runner class: `TimmImageRunner`
- `n_obs_steps` is typically `1` (single-frame setup)

## 4.2 Task Differences

- `q3_mouse.yaml`
  - `identifier: q3_mouse_195`
  - local zarr cache: `.cache/q3_mouse_dh_justresize_rawgamma/replay_buffer.zarr`
- `q3_hang_cup.yaml`
  - `identifier: q3_hang_cup`
  - local zarr cache: `.cache/q3_hang_cup/replay_buffer.zarr`
  - task name: `q3_hang_cup_5%maskwrist` (aligned with 5% wrist masking behavior)

## 4.3 Single-Frame Workspace Differences

- UNet single-frame (primary)
  - config: `train_diffusion_unet_timm_single_frame_workspace.yaml`
  - policy: `DiffusionUnetTimmPolicy`
  - learning rate: `3e-4`
- Transformer single-frame (secondary)
  - config: `train_diffusion_transformer_timm_single_frame_workspace.yaml`
  - policy: `DiffusionTransformerTimmPolicy`
  - learning rate: `7.5e-5`
- HOMMI (custom Transformer)
  - config: `HOMMI.yaml`
  - feature: per-key `feature_aggregation` map (for example `eye/head -> attn_pool`)

## 5. Code Notes and Risks

1. `transformer_obs_encoder.py` supports per-key aggregation and caches attention maps for visualization.
2. `real_pick_and_place_image_head_dataset.py` includes hard-coded wrist image masking probability (`0.05`).
3. `train_diffusion_unet_timm_workspace.py` still hard-codes `RealPushTImageRunner` for rollout; this may bypass `task.env_runner` behavior and should be reviewed before rollout-sensitive experiments.

## 6. Suggested Next Documentation Items

1. Add a "best ckpt + hyperparameter snapshot" section for `q3_mouse` and `q3_hang_cup`.
2. Track `query_filter` history when data selection logic changes.
3. Add a dedicated HOMMI page with rationale and measured gain versus baseline Transformer single-frame.
