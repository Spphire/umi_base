INFERENCE_SERVER_URL=http://192.168.1.242:8080

# TASK_NAME=single_frame_blocksv2_100_rel
# OFFLINE_DATASET_PATH=/home/wendi/Desktop/dataset/blocksv2_100/replay_buffer.zarr
# DAGGER_IDENTIFIER=blocksv2_dagger_0109
# INITIAL_CHECKPOINT_PATH=/home/wendi/Desktop/dp_ckpt/2025.12.31/11.34.10_train_diffusion_unet_timm_single_frame_blocksv2_100_rel/checkpoints/latest.ckpt
# RESUME=False

# TASK_NAME=single_frame_pourmsg_100_rel
# OFFLINE_DATASET_PATH=/home/wendi/Desktop/dataset/pourmsg_100/replay_buffer.zarr
# DAGGER_IDENTIFIER=pourmsg_dagger_0110
# INITIAL_CHECKPOINT_PATH=/home/wendi/Desktop/dp_ckpt/2025.12.31/01.10.33_train_diffusion_unet_timm_single_frame_pourmsg_100_rel/checkpoints/latest.ckpt
# RESUME=False

TASK_NAME=single_frame_towelv2_100_rel
OFFLINE_DATASET_PATH=/home/wendi/Desktop/dataset/towelv2_100/replay_buffer.zarr
DAGGER_IDENTIFIER=towelv2_dagger_online_0116
INITIAL_CHECKPOINT_PATH=/home/wendi/Desktop/dp_ckpt/2026.01.15/19.35.00_train_diffusion_unet_timm_single_frame_towelv2_100_rel_dinov2/checkpoints/latest.ckpt
RESUME=False

export HYDRA_FULL_ERROR=1 && \
	python train.py \
	--config-name train_diffusion_unet_timm_online_dagger_single_frame_workspace \
	task.name=${TASK_NAME} \
	task=real_pick_and_place_dino \
	task.dataset.action_representation=relative \
	task.dataset.offline_dataset_path=${OFFLINE_DATASET_PATH} \
	online_training.identifier=${DAGGER_IDENTIFIER} \
    online_training.initial_checkpoint_path=${INITIAL_CHECKPOINT_PATH} \
	online_training.inference_server_url=${INFERENCE_SERVER_URL} \
	training.resume=${RESUME} \
	logging.mode=online