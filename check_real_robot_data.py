import zarr
import open3d as o3d
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
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
    zarr_path = '/root/umi_base_devel/data/real_pick_and_place_pi0_zarr/replay_buffer.zarr'
    zarr_file = zarr.open(zarr_path)
    print(zarr_file.tree())
    data_file = zarr_file['data']
    # state = data_file['state'][:]
    action = data_file['action'][:]
    left_wrist_images = data_file['left_wrist_img'][:]
    external_images = data_file['external_img'][:]

    rot_6d = action[:, 3:9]  # 形状为 (100, 6)
    
    # 将6D向量转换为旋转矩阵
    def six_d_to_rot_matrix(six_d):
        """
        将6D向量转换为3x3的旋转矩阵。
        根据 "On the Continuity of Rotation Representations in Neural Networks" 论文的方法。
        """
        a = six_d[:3]
        b = six_d[3:]
        
        # 正规化第一个向量
        a_norm = a / np.linalg.norm(a)
        
        # 去除b在a方向的分量，并正规化
        b_proj = b - np.dot(b, a_norm) * a_norm
        b_norm = b_proj / np.linalg.norm(b_proj)
        
        # 计算第三个向量作为a和b的叉积
        c = np.cross(a_norm, b_norm)
        
        # 构建旋转矩阵
        rot_matrix = np.stack([a_norm, b_norm, c], axis=1)  # 形状为 (3,3)
        return rot_matrix
    
    # rot_matrices = []
    # for i in range(rot_6d.shape[0]):
    #     rot_matrix = six_d_to_rot_matrix(rot_6d[i])
    #     rot_matrices.append(rot_matrix)
    
    # rot_matrices = np.array(rot_matrices)  # 形状为 (100, 3, 3)
    
    # # 将旋转矩阵转换为欧拉角 (roll, pitch, yaw)
    # # 假设使用 'xyz' 顺序，即 roll (x), pitch (y), yaw (z)
    # rotations = R.from_matrix(rot_matrices)
    # euler_angles = rotations.as_euler('xyz', degrees=False)  # 形状为 (100, 3)
    
    # # 构建新的数组
    # # 保留原数组中未被替换的部分 (假设是第0,1,2,9列)
    # other_columns = action[:, [0, 1, 2, 9]]  # 形状为 (100, 4)

    # new_action = np.hstack([other_columns, euler_angles])

    # min_vals = np.min(new_action, axis=0)
    # max_vals = np.max(new_action, axis=0)
    # print('min_vals:', min_vals)
    # print('max_vals:', max_vals)

    num_samples = action.shape[0]

    scale_factor = 30
    
    for i in tqdm.tqdm(range(int(num_samples/scale_factor))):
        # print(f'state {i}:')
        # print(state[i])
        i = i * scale_factor
        cv2.imwrite(f'image_{i}', images[i])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # print(f'action {i}:')
        # print(action[i])
if __name__ == '__main__':
    # check_sweep_cube()
    # check_sweep_nuts()
    # check_reshape_rope()
    check_pick_and_place()