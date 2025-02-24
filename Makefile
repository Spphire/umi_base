SHELL := /bin/bash

IMAGE_NAME := umi_base_devel

PREPARE_VENV := . ../real_env/venv/bin/activate
PREPARE_ROS := source /opt/ros/humble/setup.bash

# teleop config
TASK := bimanual_one_realsense_rgb_left_30fps

# record config
SAVE_BASE_DIR := /home/wangyi/umi_base/record_data
SAVE_FILE_DIR := test
SAVE_FILE_NAME := trial1.pkl

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