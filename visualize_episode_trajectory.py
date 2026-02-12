"""
可视化指定episode的action轨迹（xyz绝对位置）
重要：action = state[t+1]，所以action包含绝对位置而非相对位移
"""
import pathlib
import numpy as np
import zarr

# 设置matplotlib非交互式backend（必须在import pyplot之前）
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from diffusion_policy.common.replay_buffer import ReplayBuffer


def load_replay_buffer(dataset_path: str) -> ReplayBuffer:
    """从zarr数据集加载ReplayBuffer"""
    dataset_path = pathlib.Path(dataset_path).expanduser().resolve()
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    print(f"Loading dataset from: {dataset_path}")
    
    if str(dataset_path).endswith('.zarr.zip'):
        with zarr.ZipStore(str(dataset_path), mode='r') as zip_store:
            replay_buffer = ReplayBuffer.copy_from_store(
                src_store=zip_store, store=zarr.MemoryStore())
    else:
        replay_buffer = ReplayBuffer.create_from_path(str(dataset_path), mode='r')
    
    print(f"Loaded! Total episodes: {replay_buffer.n_episodes}")
    return replay_buffer


def visualize_episode_trajectory(
        dataset_path: str,
        episode_idx: int = 0,
        output_dir: str = "trajectory_visualizations"
    ):
    """
    可视化指定episode的action轨迹
    
    Args:
        dataset_path: zarr数据集路径
        episode_idx: episode索引
        output_dir: 输出目录
    """
    # 加载数据
    replay_buffer = load_replay_buffer(dataset_path)
    
    if episode_idx >= replay_buffer.n_episodes:
        raise ValueError(f"Episode index {episode_idx} out of range. Total episodes: {replay_buffer.n_episodes}")
    
    # 获取指定episode的数据
    episode_data = replay_buffer.get_episode(episode_idx)
    
    # 获取action
    action = episode_data.get('action', None)
    if action is None:
        raise ValueError("No action data found in dataset")
    
    print(f"\nEpisode {episode_idx}:")
    print(f"  Action shape: {action.shape}")
    print(f"  First action: {action[0]}")
    
    # 提取绝对位置轨迹
    # action = state[t+1] (下一时刻状态)，包含绝对位置，不是相对位移
    # action结构: [左臂9D, 右臂9D, 左臂夹爪1D, 右臂夹爪1D, 头部9D]
    xyz_left = action[:, :3]      # 左臂xyz位置
    xyz_right = action[:, 9:12]   # 右臂xyz位置
    
    print(f"\n  Left arm position (absolute coordinates):")
    print(f"    X: [{xyz_left[:, 0].min():.3f}, {xyz_left[:, 0].max():.3f}]")
    print(f"    Y: [{xyz_left[:, 1].min():.3f}, {xyz_left[:, 1].max():.3f}]")
    print(f"    Z: [{xyz_left[:, 2].min():.3f}, {xyz_left[:, 2].max():.3f}]")
    
    print(f"\n  Right arm position (absolute coordinates):")
    print(f"    X: [{xyz_right[:, 0].min():.3f}, {xyz_right[:, 0].max():.3f}]")
    print(f"    Y: [{xyz_right[:, 1].min():.3f}, {xyz_right[:, 1].max():.3f}]")
    print(f"    Z: [{xyz_right[:, 2].min():.3f}, {xyz_right[:, 2].max():.3f}]")
    
    # 创建输出目录
    output_path = pathlib.Path(output_dir)
    print(f"\nCreating output directory: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # 3D轨迹图 (双臂)
        print("Creating 3D trajectory figure...")
        fig = plt.figure(figsize=(14, 6))
        
        # 3D轨迹
        ax1 = fig.add_subplot(121, projection='3d')
        
        # 绘制左臂轨迹
        ax1.plot(xyz_left[:, 0], xyz_left[:, 1], xyz_left[:, 2], 'b-', linewidth=2.5, label='Left Arm')
        ax1.scatter(xyz_left[0, 0], xyz_left[0, 1], xyz_left[0, 2], color='blue', s=150, marker='o', label='Left Start', edgecolors='darkblue', linewidths=2)
        ax1.scatter(xyz_left[-1, 0], xyz_left[-1, 1], xyz_left[-1, 2], color='cyan', s=150, marker='s', label='Left End', edgecolors='darkblue', linewidths=2)
        
        # 绘制右臂轨迹
        ax1.plot(xyz_right[:, 0], xyz_right[:, 1], xyz_right[:, 2], 'r-', linewidth=2.5, label='Right Arm')
        ax1.scatter(xyz_right[0, 0], xyz_right[0, 1], xyz_right[0, 2], color='red', s=150, marker='o', label='Right Start', edgecolors='darkred', linewidths=2)
        ax1.scatter(xyz_right[-1, 0], xyz_right[-1, 1], xyz_right[-1, 2], color='orange', s=150, marker='s', label='Right End', edgecolors='darkred', linewidths=2)
        
        ax1.set_xlabel('X (m)', fontsize=11)
        ax1.set_ylabel('Y (m)', fontsize=11)
        ax1.set_zlabel('Z (m)', fontsize=11)
        ax1.set_title(f'Episode {episode_idx} - End-Effector 3D Trajectories\n(Absolute Coordinates)', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9, loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # XYZ位置曲线
        ax2 = fig.add_subplot(122)
        timesteps = np.arange(len(xyz_left))
        
        ax2.plot(timesteps, xyz_left[:, 0], 'b-', label='Left X', linewidth=1.8, alpha=0.8)
        ax2.plot(timesteps, xyz_left[:, 1], 'b--', label='Left Y', linewidth=1.8, alpha=0.8)
        ax2.plot(timesteps, xyz_left[:, 2], 'b:', label='Left Z', linewidth=1.8, alpha=0.8)
        
        ax2.plot(timesteps, xyz_right[:, 0], 'r-', label='Right X', linewidth=1.8, alpha=0.8)
        ax2.plot(timesteps, xyz_right[:, 1], 'r--', label='Right Y', linewidth=1.8, alpha=0.8)
        ax2.plot(timesteps, xyz_right[:, 2], 'r:', label='Right Z', linewidth=1.8, alpha=0.8)
        
        ax2.set_xlabel('Timestep', fontsize=11)
        ax2.set_ylabel('Position (m)', fontsize=11)
        ax2.set_title(f'Episode {episode_idx} - End-Effector Position Over Time', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9, ncol=2, loc='best')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path1 = output_path / f"episode_{episode_idx}_trajectory.png"
        print(f"Saving to: {save_path1}")
        fig.savefig(str(save_path1), dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to: {save_path1}")
        plt.close(fig)
        
    except Exception as e:
        print(f"✗ Error saving trajectory figure: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        # 俯视图和侧视图
        print("Creating projection figures...")
        fig = plt.figure(figsize=(16, 5))
        
        # XY平面图（俯视图）
        ax1 = fig.add_subplot(131)
        ax1.plot(xyz_left[:, 0], xyz_left[:, 1], 'b-', linewidth=2.5, label='Left Arm')
        ax1.plot(xyz_right[:, 0], xyz_right[:, 1], 'r-', linewidth=2.5, label='Right Arm')
        ax1.scatter(xyz_left[0, 0], xyz_left[0, 1], color='blue', s=120, marker='o', edgecolors='darkblue', linewidths=2)
        ax1.scatter(xyz_left[-1, 0], xyz_left[-1, 1], color='cyan', s=120, marker='s', edgecolors='darkblue', linewidths=2)
        ax1.scatter(xyz_right[0, 0], xyz_right[0, 1], color='red', s=120, marker='o', edgecolors='darkred', linewidths=2)
        ax1.scatter(xyz_right[-1, 0], xyz_right[-1, 1], color='orange', s=120, marker='s', edgecolors='darkred', linewidths=2)
        ax1.set_xlabel('X (m)', fontsize=11)
        ax1.set_ylabel('Y (m)', fontsize=11)
        ax1.set_title('Top View (XY plane)', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        
        # XZ平面图（侧视图）
        ax2 = fig.add_subplot(132)
        ax2.plot(xyz_left[:, 0], xyz_left[:, 2], 'b-', linewidth=2.5, label='Left Arm')
        ax2.plot(xyz_right[:, 0], xyz_right[:, 2], 'r-', linewidth=2.5, label='Right Arm')
        ax2.scatter(xyz_left[0, 0], xyz_left[0, 2], color='blue', s=120, marker='o', edgecolors='darkblue', linewidths=2)
        ax2.scatter(xyz_left[-1, 0], xyz_left[-1, 2], color='cyan', s=120, marker='s', edgecolors='darkblue', linewidths=2)
        ax2.scatter(xyz_right[0, 0], xyz_right[0, 2], color='red', s=120, marker='o', edgecolors='darkred', linewidths=2)
        ax2.scatter(xyz_right[-1, 0], xyz_right[-1, 2], color='orange', s=120, marker='s', edgecolors='darkred', linewidths=2)
        ax2.set_xlabel('X (m)', fontsize=11)
        ax2.set_ylabel('Z (m)', fontsize=11)
        ax2.set_title('Side View (XZ plane)', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
        
        # YZ平面图
        ax3 = fig.add_subplot(133)
        ax3.plot(xyz_left[:, 1], xyz_left[:, 2], 'b-', linewidth=2.5, label='Left Arm')
        ax3.plot(xyz_right[:, 1], xyz_right[:, 2], 'r-', linewidth=2.5, label='Right Arm')
        ax3.scatter(xyz_left[0, 1], xyz_left[0, 2], color='blue', s=120, marker='o', edgecolors='darkblue', linewidths=2)
        ax3.scatter(xyz_left[-1, 1], xyz_left[-1, 2], color='cyan', s=120, marker='s', edgecolors='darkblue', linewidths=2)
        ax3.scatter(xyz_right[0, 1], xyz_right[0, 2], color='red', s=120, marker='o', edgecolors='darkred', linewidths=2)
        ax3.scatter(xyz_right[-1, 1], xyz_right[-1, 2], color='orange', s=120, marker='s', edgecolors='darkred', linewidths=2)
        ax3.set_xlabel('Y (m)', fontsize=11)
        ax3.set_ylabel('Z (m)', fontsize=11)
        ax3.set_title('Side View (YZ plane)', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.axis('equal')
        
        plt.tight_layout()
        
        save_path2 = output_path / f"episode_{episode_idx}_projections.png"
        print(f"Saving to: {save_path2}")
        fig.savefig(str(save_path2), dpi=150, bbox_inches='tight')
        print(f"✓ Saved projection views to: {save_path2}")
        plt.close(fig)
        
    except Exception as e:
        print(f"✗ Error saving projection figures: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        # 保存轨迹数据为npy文件
        print("Saving trajectory data...")
        left_path = output_path / f"episode_{episode_idx}_left_arm_trajectory.npy"
        right_path = output_path / f"episode_{episode_idx}_right_arm_trajectory.npy"
        
        np.save(str(left_path), xyz_left)
        np.save(str(right_path), xyz_right)
        print(f"✓ Saved left arm trajectory to: {left_path}")
        print(f"✓ Saved right arm trajectory to: {right_path}")
        
    except Exception as e:
        print(f"✗ Error saving npy files: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✓ All done!")
    return xyz_left, xyz_right


if __name__ == "__main__":
    # ==================== 硬编码参数 ====================
    dataset_path = r'C:\Users\yibo\Downloads\umi_base\.cache\q3_shop_bagging_0202\replay_buffer.zarr'
    episode_idx = 0
    output_dir = r'C:\Users\yibo\Downloads\umi_base\trajectory_visualizations'
    # ===================================================
    
    print(f"Output will be saved to: {output_dir}\n")
    
    visualize_episode_trajectory(
        dataset_path=dataset_path,
        episode_idx=episode_idx,
        output_dir=output_dir
    )
