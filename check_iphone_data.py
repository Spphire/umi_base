import zarr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.transform import Rotation as R

def detect_jumps(data, threshold=0.1):
    """
    检测数据中的跳变
    使用z-score方法检测异常值
    """
    # 计算差分
    diff = np.diff(data)
    # 找出超过阈值且小于3的点
    jumps = np.where((np.abs(diff) > threshold) & (np.abs(diff) < 3))[0] + 1
    return jumps

def analyze_zarr_data(zarr_path, key_name='action'):
    """
    分析zarr数据中的跳变并可视化
    
    参数:
    zarr_path: zarr文件路径
    key_name: 要分析的数据键名 (例如 'action', 'state' 等)
    """
    # 读取zarr数据
    zarr_file = zarr.open(zarr_path)
    data_file = zarr_file['data']
    action = data_file[key_name][:]

    rot_6d = action[:, 3:9]
    
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
    
    rot_matrices = []
    for i in range(rot_6d.shape[0]):
        rot_matrix = six_d_to_rot_matrix(rot_6d[i])
        rot_matrices.append(rot_matrix)
    
    rot_matrices = np.array(rot_matrices)

    rotations = R.from_matrix(rot_matrices)
    euler_angles = rotations.as_euler('xyz', degrees=False)

    other_columns = action[:, [0, 1, 2, 9]]
    data = np.hstack([other_columns, euler_angles]) # (x,y,z,gripper_width,roll,pitch,yaw)

    
    # 获取episode分割点
    if 'meta' in zarr_file and 'episode_ends' in zarr_file['meta']:
        episode_ends = zarr_file['meta']['episode_ends'][:]
    else:
        print("未找到episode_ends信息，将整个数据视为单个episode")
        episode_ends = [len(data)]
    
    # 为每个维度创建一个图
    n_dims = data.shape[1]
    n_episodes = len(episode_ends)
    
    # 设置图表风格
    sns.set_style("whitegrid")
    plt.figure(figsize=(15, 4 * n_dims))
    
    # 记录每个维度的跳变统计
    jump_stats = {}
    
    # 为每个维度创建子图
    for dim in range(n_dims):
        plt.subplot(n_dims, 1, dim + 1)
        
        start_idx = 0
        dim_jumps = []
        
        # 绘制每个episode的数据
        for ep_idx, end_idx in enumerate(episode_ends):
            episode_data = data[start_idx:end_idx, dim]
            time_steps = np.arange(start_idx, end_idx)
            
            # 绘制数据线
            plt.plot(time_steps, episode_data, 
                    label=f'Episode {ep_idx}', alpha=0.7)
            
            # 检测跳变
            jumps = detect_jumps(episode_data)
            dim_jumps.extend(jumps + start_idx)
            
            # 标记跳变点
            if len(jumps) > 0:
                plt.scatter(jumps + start_idx, 
                          episode_data[jumps], 
                          color='red', 
                          marker='x', 
                          s=100, 
                          label='Jumps' if ep_idx == 0 else '')
            
            # 添加垂直分隔线标示episode边界，并标注episode id
            plt.axvline(x=end_idx, color='black', linestyle='--', alpha=0.3)
            plt.text(end_idx, plt.ylim()[1], f'{ep_idx}', color='black', 
                     fontsize=6, ha='left', va='top', alpha=0.6, rotation=0)
            
            start_idx = end_idx
        
        jump_stats[f'dim_{dim}'] = len(dim_jumps)
        
        dim_names = ['x', 'y', 'z', 'gripper_width', 'roll', 'pitch', 'yaw']
        plt.title(f'Dimension {dim} ({dim_names[dim]}) Values Over Time\n'
              f'Total jumps detected: {len(dim_jumps)}')
        plt.xlabel('Time steps')
        plt.ylabel(f'{key_name}[{dim}]')
        # plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{key_name}_analysis.png')
    plt.close()
    
    # 打印统计信息
    print("\n跳变检测统计:")
    for dim, count in jump_stats.items():
        print(f"{dim}: 检测到 {count} 个跳变")

if __name__ == '__main__':
    # 示例使用
    zarr_path = '/root/umi_base_devel/data/pick_and_place_coffee_iphone_collector_zarr/replay_buffer.zarr'
    
    # 分析action数据
    analyze_zarr_data(zarr_path, 'action')
    
    # 如果还需要分析其他数据，可以取消下面的注释
    # analyze_zarr_data(zarr_path, 'state')