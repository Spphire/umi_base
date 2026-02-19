SHELL := /bin/bash

IMAGE_NAME := umi_base_devel

# PREPARE_VENV := . ../real_env/venv/bin/activate
PREPARE_ROS := source /opt/ros/humble/setup.bash

#  && export ROS_DOMAIN_ID=192.168.2.223

# teleop config

#TASK := q3_shop_bagging_0202
#TASK := q3_shop_bagging_0207_250
#TASK := q3_shop_bagging_0207_150_nogripperinput
#TASK := q3_block_100_nogripperinput
TASK := q3_mouse


WKSPACE := timm_resume_nomaskwrist
#WKSPACE := train_diffusion_unet_timm_single_frame_workspace
#WKSPACE := train_diffusion_transformer_timm_single_frame_workspace
# DATASET_PATH := /root/umi_base_devel/data/pick_and_place_coffee_iphone_collector_zarr_clip

# record config
SAVE_BASE_DIR := /mnt/data/shenyibo/workspace/umi_base/data
SAVE_FILE_DIR := ${TASK}
# SAVE_FILE_DIR := test
# SAVE_FILE_NAME := trial1.pkl

PROJECT_BASE_DIR = /mnt/data/shenyibo/workspace/umi_base
PROJECT_NAME = umi_base_devel

teleop.launch_camera:
	${PREPARE_VENV} && \
	${PREPARE_ROS} && \
	python camera_node_launcher.py \
	task=${TASK}

teleop.launch_robot:
	${PREPARE_VENV} && \
	${PREPARE_ROS} && \
	python teleop.py \
	task=${TASK}

teleop.start_record:
	${PREPARE_VENV} && \
	${PREPARE_ROS} && \
	python record_data.py \
	    --save_base_dir ${SAVE_BASE_DIR} \
	    --save_file_dir ${SAVE_FILE_DIR} \
		--save_to_disk
	    # --save_file_name ${SAVE_FILE_NAME} \ # auto numbered while not specified

teleop.post_process_iphone:
	${PREPARE_VENV} && \
	${PREPARE_ROS} && \
	python post_process_data_iphone.py \
	--tag ${SAVE_FILE_DIR}

teleop.post_process:
	${PREPARE_VENV} && \
	${PREPARE_ROS} && \
	python post_process_data.py \
	--tag ${SAVE_FILE_DIR}

train:
	export HYDRA_FULL_ERROR=1 && \
	python train.py \
	--config-name ${WKSPACE} \
	task=${TASK} \

train_acc:
	export HYDRA_FULL_ERROR=1 && \
	accelerate launch --config_file accelerate/4gpu.yaml train.py \
	--config-name ${WKSPACE} \
	task=${TASK}

train_acc_amp:
	export HYDRA_FULL_ERROR=1 && \
	accelerate launch --config_file accelerate/4gpu-amp.yaml train.py \
	--config-name ${WKSPACE} \
	task=${TASK}
 	#task.dataset.local_files_only=/mnt/workspace/shenyibo/data/umi_base/packsnackq3_1-24/replay_buffer.zarr \
	#task.name=single_frame_packsnackq3_100

train_acc8:
	export HYDRA_FULL_ERROR=1 && \
	accelerate launch --config_file accelerate/8gpu.yaml train.py \
	--config-name ${WKSPACE} \
	task=${TASK}

train_acc8_amp:
	export HYDRA_FULL_ERROR=1 && \
	accelerate launch --config_file accelerate/8gpu-amp.yaml train.py \
	--config-name ${WKSPACE} \
	task=${TASK}

train.data:
	export HYDRA_FULL_ERROR=1 && \
	python train.py \
	--config-name ${WKSPACE} \
	task=${TASK}

train_acc_amp_remote:
	export HYDRA_FULL_ERROR=1 && \
	accelerate launch --config_file accelerate/4gpu-amp.yaml train.py \
	--config-name ${WKSPACE} \
	task=${TASK} \
	task.dataset.identifier=fold_towel
	
eval.launch_record:
	${PREPARE_ROS} && \
 	python camera_recorder.py --output-dir /home/fangyuan/lt_ws/umi_video

eval.launch_camera:
	${PREPARE_ROS} && \
	python camera_node_launcher.py \
	task=${TASK}

eval.launch_fake_camera:
	${PREPARE_ROS} && \
	python fake_camera_launcher.py \
	task=${TASK}

eval.launch_robot:
	${PREPARE_ROS} && \
	python teleop.py \
	task=${TASK}

eval.replay:
	${PREPARE_ROS} && \
	export HYDRA_FULL_ERROR=1 && \
	python replay.py \
	--config-name ${WKSPACE} \
	task=q3_shop_bagging_0202_replay

eval.draw_circle:
	${PREPARE_ROS} && \
	export HYDRA_FULL_ERROR=1 && \
	python draw_circle.py \
	--config-name ${WKSPACE} \
	task=draw_circle

eval.inference:
	${PREPARE_ROS} && \
	export HYDRA_FULL_ERROR=1 && \
	python eval_real_robot_flexiv.py \
	--config-name ${WKSPACE} \
	task=${TASK} \
	+task.env_runner.output_dir='data/outputs/$(shell date +%Y.%m.%d)/$(shell date +%H.%M.%S)_${TASK}_inference_video' \
    +ckpt_path='/mnt/data/shenyibo/workspace/umi_base/data/outputs/2026.02.03/10.09.10_train_diffusion_unet_timm_q3_shop_bagging_0202_100/checkpoints/1190.ckpt'


test.cloud_dataset:
	${PREPARE_ROS} && \
	python -m tests.test_cloud_dataset

utils.check_iphone_data:
	${PREPARE_ROS} && \
	python -m diffusion_policy.scripts.check_iphone_data

train.dataset:
	${PREPARE_ROS} && \
	export HYDRA_FULL_ERROR=1 && \
	python scripts/generate_offline_dataset_from_datacloud.py

post_process.vr:
	python -m post_process_scripts.post_process_data_vr

diffusion_policy.dataset.head:
	python -m diffusion_policy.dataset.real_pick_and_place_image_head_dataset

#python -m diffusion_policy.dataset.real_pick_and_place_image_dataset