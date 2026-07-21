#!/usr/bin/env bash
set -euo pipefail

cd /mnt/workspace/shenyibo/diffusion_policy
source .venv4041/bin/activate
export HYDRA_FULL_ERROR=1
export HF_HUB_OFFLINE=1

exec .venv4041/bin/python -m accelerate.commands.launch \
  --config_file accelerate/8gpu-amp.yaml \
  train.py \
  --config-name train_diffusion_transformer_timm_dualfold_2frame_30horizon_workspace_sharergb \
  task=dualfold_3view_20action \
  task.dataset.dataset_path=/mnt/workspace/shenyibo/diffusion_policy/.cache/dualfold_pink_0608_0610_recovery0708_full_consis232_zarr/zarr/replay_buffer.zarr \
  exp_name=dualfold_pink_0608_0610_recovery0708_full_consis232_3view
