SHELL := /bin/bash

IMAGE_NAME := umi_base_devel

# PREPARE_VENV := . ../real_env/venv/bin/activate
PREPARE_ROS := source /opt/ros/humble/setup.bash

#  && export ROS_DOMAIN_ID=192.168.2.223

# teleop config
TASK := dino_test
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
WKSPACE := train_diffusion_unet_timm_workspace
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
	${PREPARE_ROS} && \
	export HYDRA_FULL_ERROR=1 && \
	python train.py \
	--config-name ${WKSPACE} \
	task=${TASK} \

eval.launch_camera:
	${PREPARE_ROS} && \
	python camera_node_launcher.py \
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

eval.inference:
	${PREPARE_ROS} && \
	export HYDRA_FULL_ERROR=1 && \
	python eval_real_robot_flexiv.py \
	--config-name ${WKSPACE} \
	task=${TASK} \
	+task.env_runner.output_dir='data/outputs/$(shell date +%Y.%m.%d)/$(shell date +%H.%M.%S)_${TASK}_inference_video' \
	+ckpt_path='/home/fangyuan/Documents/GitHub/julyfun/umi_base/data/outputs/fold-towel/latest2.ckpt'
# 	+ckpt_path='/home/fangyuan/Documents/GitHub/julyfun/umi_base/.cache/umi_base/wood_9.1-small_finger-iphone-ble-100-train9.3/checkpoints/latest.ckpt'
# 	+ckpt_path='/home/fangyuan/Documents/GitHub/julyfun/umi_base/.cache/umi_base/dino_test-pp_wo_8.13-8.29/checkpoints/latest.ckpt'

test.cloud_dataset:
	${PREPARE_VENV} && \
	${PREPARE_ROS} && \
	python -m tests.test_cloud_dataset

utils.check_iphone_data:
	${PREPARE_VENV} && \
	${PREPARE_ROS} && \
	python -m diffusion_policy.scripts.check_iphone_data
