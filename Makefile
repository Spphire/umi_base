SHELL := /bin/bash

IMAGE_NAME := umi_base_devel

PREPARE_VENV := . ../real_env/venv/bin/activate
PREPARE_ROS := source /opt/ros/humble/setup.bash && export ROS_DOMAIN_ID=192.168.2.223

# teleop config
TASK := real_pick_and_place_pi0
# TASK := single_arm_one_realsense_30fps
# TASK := bimanual_one_realsense_rgb_left_30fps

# workspace config , have to be consistent with the task
WKSPACE := train_diffusion_unet_real_image_workspace

# record config
SAVE_BASE_DIR := /home/wangyi/umi_base/data
SAVE_FILE_DIR := ${TASK}
SAVE_FILE_NAME := trial50.pkl

PROJECT_BASE_DIR = /home/wangyi/umi_base
PROJECT_NAME = umi_base_devel

docker.build:
	docker build -t ${IMAGE_NAME}:latest -f docker/Dockerfile .

docker.run:
	@if [ -z "$$(docker ps -q -f name=teleop)" ]; then \
		docker run -d \
			--runtime=nvidia \
			--gpus all \
			--network host \
			--privileged \
			-v ${PROJECT_BASE_DIR}:/root/${PROJECT_NAME}\
			-v ${SAVE_BASE_DIR}:/root/record_data \
			-w /root/${PROJECT_NAME} \
			--name teleop \
			--shm-size 32G \
			${IMAGE_NAME}:latest \
			tail -f /dev/null; \
	fi && \
	docker exec -it teleop bash

docker.remove:
	docker rm teleop

docker.clean:
	docker image rm ${IMAGE_NAME}:latest

docker.all: docker.build docker.run

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
	    --save_base_dir /root/record_data \
	    --save_file_dir ${SAVE_FILE_DIR} \
	    --save_file_name ${SAVE_FILE_NAME} \
	    --save_to_disk

teleop.post_process:
	${PREPARE_VENV} && \
	${PREPARE_ROS} && \
	python post_process_data.py \
	--tag ${SAVE_FILE_DIR}

train:
	${PREPARE_VENV} && \
	export HYDRA_FULL_ERROR=1 && \
	python train.py \
	--config-name ${WKSPACE} \
	task=${TASK}

eval.launch_camera:
	${PREPARE_VENV} && \
	${PREPARE_ROS} && \
	python camera_node_launcher.py \
	task=${TASK}

eval.launch_robot:
	${PREPARE_VENV} && \
	${PREPARE_ROS} && \
	python teleop.py \
	task=${TASK}

eval.inference_pi0:
	${PREPARE_VENV} && \
	${PREPARE_ROS} && \
	export HYDRA_FULL_ERROR=1 && \
	python eval_real_robot_flexiv_pi0.py \
	--config-name ${WKSPACE} \
	task=${TASK} \
	+task.env_runner.output_dir=data/outputs/$(shell date +%Y.%m.%d)/$(shell date +%H.%M.%S)_${TASK}_inference_vedio \

eval.inference:
	${PREPARE_VENV} && \
	${PREPARE_ROS} && \
	export HYDRA_FULL_ERROR=1 && \
	python eval_real_robot_flexiv.py \
	--config-name ${WKSPACE} \
	task=${TASK} \
	+task.env_runner.output_dir=data/outputs/$(shell date +%Y.%m.%d)/$(shell date +%H.%M.%S)_${TASK}_inference_vedio \
	+ckpt_path=data/outputs/2025.03.23/03.18.07_train_diffusion_unet_image_single_right_arm_pick_and_place_s1_pi0/checkpoints/latest.ckpt
