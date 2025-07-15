import zarr
import numpy as np
import matplotlib.pyplot as plt
import os
# import debugpy
# debugpy.listen(5679)
# print("Waiting for debugger attach.")
# debugpy.wait_for_client()

def main():
    zarr_path = "/home/fangyuan/Documents/umi_base/.cache/cloud_pick_and_place_image_dataset/6adadfda53b00c3f/replay_buffer.zarr"

    zarr_data = zarr.open(zarr_path, mode='r')

    imgs = zarr_data['data']['left_wrist_img']
    action = zarr_data['data']['action']
    left_robot_tcp_pose = zarr_data['data']['left_robot_tcp_pose']
    left_robot_gripper_width = zarr_data['data']['left_robot_gripper_width']

    
    # 获取action数据
    action_data = np.array(action)
    left_robot_gripper_width_data = np.array(left_robot_gripper_width)
    all_data = np.concatenate((action_data, left_robot_gripper_width_data), axis=1)
    
    plot_data(all_data)

def plot_data(all_data, image_name="iphone_action_distribution"):
    debug_dir = "./.debug"
    os.makedirs(debug_dir, exist_ok=True)
    print(f"Action shape: {all_data.shape}")
    print(f"Action dimensions: {all_data.shape[-1]}")
    
    # 创建子图，根据action维度数量
    n_dims = all_data.shape[-1]
    n_cols = 3  # 每行3个子图
    n_rows = (n_dims + n_cols - 1) // n_cols  # 向上取整
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    fig.suptitle('Action Distribution by Dimension', fontsize=16)
    
    # 如果只有一行，确保axes是二维数组
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # 为每个维度绘制直方图
    for dim in range(n_dims):
        row = dim // n_cols
        col = dim % n_cols
        
        # 获取当前维度的数据
        dim_data = all_data[:, dim]
        
        axes[row, col].hist(dim_data, bins=50, alpha=0.7, edgecolor='black')
        axes[row, col].set_title(f'Dimension {dim}')
        axes[row, col].set_xlabel('Value')
        axes[row, col].set_ylabel('Frequency')
        axes[row, col].grid(True, alpha=0.3)
        
        # 添加统计信息
        mean_val = np.mean(dim_data)
        std_val = np.std(dim_data)
        axes[row, col].axvline(mean_val, color='red', linestyle='--', 
                              label=f'Mean: {mean_val:.3f}')
        axes[row, col].legend()
        
        print(f"Dimension {dim}: mean={mean_val:.4f}, std={std_val:.4f}, "
              f"min={np.min(dim_data):.4f}, max={np.max(dim_data):.4f}")
    
    # 隐藏多余的子图
    for dim in range(n_dims, n_rows * n_cols):
        row = dim // n_cols
        col = dim % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(debug_dir, f"{image_name}.png")
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"Action distribution histogram saved to: {output_path}")
    
    plt.show()

if __name__ == '__main__':
    main()