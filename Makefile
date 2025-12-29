SHELL := /bin/bash

IMAGE_NAME := umi_base_devel

# PREPARE_VENV := . ../real_env/venv/bin/activate
PREPARE_ROS := source /opt/ros/humble/setup.bash

#  && export ROS_DOMAIN_ID=192.168.2.223

# teleop config
TASK := real_pick_and_place_dino
# TASK := real_pick_and_place_dino_bimanual
# TASK := real_pick_and_place_image_iphone
# TASK := pick_no_fisheye
# TASK := real_pick_and_place_image
# TASK := real_pick_and_place_gr00t
# TASK := real_pick_and_place_pi0
# TASK := single_arm_iphone_teleop
# TASK := single_arm_one_realsense_30fps
# TASK := bimanual_one_realsense_rgb_left_30fps

# workspace config , have to be consistent with the task
# WKSPACE := train_diffusion_unet_real_image_workspace
# WKSPACE := train_diffusion_unet_timm_workspace
WKSPACE := train_diffusion_unet_timm_single_frame_workspace
# DATASET_PATH := /root/umi_base_devel/data/pick_and_place_coffee_iphone_collector_zarr_clip

# record config
SAVE_BASE_DIR := /home/fangyuan/Documents/umi_base
SAVE_BASE_DIR := /root/umi_base_devel/data
SAVE_FILE_DIR := ${TASK}
# SAVE_FILE_DIR := test
# SAVE_FILE_NAME := trial1.pkl

PROJECT_BASE_DIR = /home/fangyuan/Documents/umi_base
PROJECT_NAME = umi_base_devel

# docker.build:
# 	docker build -t ${IMAGE_NAME}:latest -f docker/Dockerfile .

# # docker.run:
# # 	@if [ -z "$$(docker ps -q -f name=teleop)" ]; then \
# # 		docker run -d \
# # 			--runtime=nvidia \
# # 			--gpus all \
# # 			--network host \
# # 			--privileged \
# # 			-v ${PROJECT_BASE_DIR}:/root/${PROJECT_NAME}\
# # 			-v ${SAVE_BASE_DIR}:/root/record_data \
# # 			-w /root/${PROJECT_NAME} \
# # 			--name teleop \
# # 			--shm-size 32G \
# # 			${IMAGE_NAME}:latest \real_world_env
# # 			tail -f /dev/null; \
# # 	fi && \
# # 	docker exec -it teleop bash

# docker.run:
# 	docker run -d \
# 		--runtime=nvidia \
# 		--gpus all \
# 		--network host \
# 		--privileged \
# 		-v ${PROJECT_BASE_DIR}:/root/${PROJECT_NAME}\
# 		-v ${SAVE_BASE_DIR}:/root/record_data \
# 		-w /root/${PROJECT_NAME} \
# 		--name teleop_fyzhou \
# 		--shm-size 32G \
# 		${IMAGE_NAME}:latest \
# 		tail -f /dev/null; \
# 	docker exec -it teleop_fyzhou bash

# docker.remove:
# 	docker rm teleop_fyzhou

# docker.clean:
# 	docker image rm ${IMAGE_NAME}:latest

# docker.all: docker.build docker.run

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
	task=${TASK} \
	+task.dataset.local_files_only=/home/wendi/Desktop/openpi/data/pourmsg_100/replay_buffer.zarr \
	task.name=single_frame_pourmsg_100

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
	task=replay_cloud_data

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
    +ckpt_path='/home/fangyuan/Documents/GitHub/julyfun/umi_base/data/outputs/2025.11.11/22.20.46_train_diffusion_unet_timm_single_right_arm_pick_and_place_s1_image_only/checkpoints/latest.ckpt'


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
