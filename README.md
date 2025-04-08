# Data collection-Model training-Real robot evaluation Workflow Guidance

## Data collection (optional)
If VR teleopration used, refer to [VR usage]().

## Model training
If original diffusion policy used, create your workspace and task configuration with ```diffusion_policy/config/train_diffusion_unet_real_image_workspace.yaml``` and ```diffusion_policy/config/task/real_pick_and_place_image.yaml``` taken as templates.

Use your config in ```Makefile``` and run below command inside your docker container:
```shell
make train
```
## Real robot evaluation
Modify your config params related to ```env_runner``` to make sure observations got and preprocessed, models loaded correctly.
Use your config in ```Makefile``` and run below command inside your docker container:
```shell
(optional, when realsense used) make eval.launch_camera
make eval.launch_robot
make eval.inference
```