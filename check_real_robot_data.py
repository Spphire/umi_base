import zarr
import open3d as o3d
import cv2
import numpy as np
import tqdm

def check_push_t():
    zarr_path = 'data/pusht_real/real_pusht_20230105/replay_buffer.zarr'
    zarr_file = zarr.open(zarr_path)
    print(zarr_file.tree())
    data_file = zarr_file['data']
    robot_eef_pose = data_file['robot_eef_pose'][:]
    action = data_file['action'][:]
    stage = data_file['stage'][:]

    num_samples = action.shape[0]
    for i in tqdm.tqdm(range(num_samples)):
        print(f'action {i}:')
        print(action[i])

        print(f'stage {i}:')
        print(stage[i])

        print(f'robot_eef_pose {i}:')
        print(robot_eef_pose[i])

def check_sweep_cube():
    zarr_path = 'data/sweep_nuts_zarr/replay_buffer.zarr'
    zarr_file = zarr.open(zarr_path)
    print(zarr_file.tree())
    data_file = zarr_file['data']
    state = data_file['state'][:]
    action = data_file['action'][:]

    num_samples = action.shape[0]
    for i in tqdm.tqdm(range(num_samples)):
        print(f'state {i}:')
        print(state[i])
        print(f'action {i}:')
        print(action[i])

def check_sweep_nuts():
    zarr_path = 'data/sweep_nuts_v2_zarr/replay_buffer.zarr'
    zarr_file = zarr.open(zarr_path)
    print(zarr_file.tree())
    data_file = zarr_file['data']
    state = data_file['state'][:]
    action = data_file['action'][:]

    num_samples = action.shape[0]
    for i in tqdm.tqdm(range(num_samples)):
        print(f'state {i}:')
        print(state[i])
        print(f'action {i}:')
        print(action[i])

def check_reshape_rope():
    zarr_path = 'data/reshape_rope_zarr/replay_buffer.zarr'
    zarr_file = zarr.open(zarr_path)
    print(zarr_file.tree())
    data_file = zarr_file['data']
    state = data_file['state'][:]
    action = data_file['action'][:]

    num_samples = action.shape[0]
    for i in tqdm.tqdm(range(num_samples)):
        print(f'state {i}:')
        print(state[i])
        print(f'action {i}:')
        print(action[i])

def check_pick_and_place():
    zarr_path = 'data/pick_and_place_v4_zarr/replay_buffer.zarr'
    zarr_file = zarr.open(zarr_path)
    print(zarr_file.tree())
    data_file = zarr_file['data']
    state = data_file['state'][:]
    action = data_file['action'][:]

    num_samples = action.shape[0]
    for i in tqdm.tqdm(range(num_samples)):
        print(f'state {i}:')
        print(state[i])
        print(f'action {i}:')
        print(action[i])
if __name__ == '__main__':
    # check_sweep_cube()
    # check_sweep_nuts()
    # check_reshape_rope()
    check_pick_and_place()