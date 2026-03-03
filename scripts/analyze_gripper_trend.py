import zarr
import numpy as np
import os
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import argparse

def classify_trend(arr, window=5, poly=2, diff_thresh=1e-4):
    """
    对夹爪宽度曲线进行趋势分类：平台、上升、下降
    返回每个点的标签：'flat', 'up', 'down'
    """
    # 使用更大的窗口平滑，适应夹爪动作的缓慢变化
    win = min(max(window, 11), len(arr)//2*2+1)  # 至少11，且为奇数
    if len(arr) < win:
        return np.array(['flat'] * len(arr))
    smooth = savgol_filter(arr, window_length=win, polyorder=poly)
    diff = np.gradient(smooth)
    trend = np.full_like(arr, 'flat', dtype=object)
    trend[diff > diff_thresh] = 'up'
    trend[diff < -diff_thresh] = 'down'
    return trend

def find_trend_windows(trend):
    """
    将趋势标签分段，返回[start, end, label]列表
    """
    # 合并短窗口，减少噪声影响
    min_len = 5  # 最小窗口长度
    windows = []
    if len(trend) == 0:
        return windows
    start = 0
    current = trend[0]
    for i in range(1, len(trend)):
        if trend[i] != current:
            if i - start < min_len and windows:
                # 合并到前一个窗口
                prev = windows.pop()
                start = prev[0]
                current = prev[2]
            windows.append((start, i, current))
            start = i
            current = trend[i]
    if len(trend) - start < min_len and windows:
        # 合并最后一个短窗口
        prev = windows.pop()
        start = prev[0]
        current = prev[2]
    windows.append((start, len(trend), current))
    return windows

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--zarr_path', 
                        type=str, 
                        default="/mnt/data/shenyibo/workspace/umi_base/.cache/q3_choose_block_dh/replay_buffer.zarr", 
                        help='Path to zarr file')
    parser.add_argument('--plot', action='store_true', help='Plot each episode')
    args = parser.parse_args()
    zarr_path = args.zarr_path
    do_plot = args.plot

    if not os.path.exists(zarr_path):
        print(f"Zarr path not found: {zarr_path}")
        quit()
    zarr_root = zarr.open(zarr_path, mode='a')
    gripper_width = zarr_root['data']['left_robot_gripper_width'][:].squeeze(-1)
    episode_ends = zarr_root['meta']['episode_ends'][:]
    print(f"Total episodes: {len(episode_ends)}")
    start = 0
    ok_count = 0
    all_labels = []
    for epi_idx, end in enumerate(episode_ends):
        epi_width = gripper_width[start:end]
        trend = classify_trend(epi_width)
        windows = find_trend_windows(trend)
        max_gripper = np.max(epi_width) if len(epi_width) > 0 else 1.0
        min_drop = max_gripper / 2.0
        merged_windows = []
        i = 0
        while i < len(windows):
            w = windows[i]
            if w[2] in ('up', 'down'):
                val_range = np.abs(epi_width[w[1]-1] - epi_width[w[0]])
                if val_range < min_drop:
                    prev_flat = (len(merged_windows) > 0 and merged_windows[-1][2] == 'flat')
                    next_flat = (i+1 < len(windows) and windows[i+1][2] == 'flat')
                    if prev_flat and next_flat:
                        prev = merged_windows.pop()
                        next_w = windows[i+1]
                        merged_windows.append((prev[0], next_w[1], 'flat'))
                        i += 2
                        continue
                    elif prev_flat:
                        prev = merged_windows.pop()
                        merged_windows.append((prev[0], w[1], 'flat'))
                        i += 1
                        continue
                    elif next_flat:
                        next_w = windows[i+1]
                        merged_windows.append((w[0], next_w[1], 'flat'))
                        i += 2
                        continue
                    else:
                        merged_windows.append((w[0], w[1], 'flat'))
                        i += 1
                        continue
            merged_windows.append(w)
            i += 1
        print(f"Episode {epi_idx}: length={len(epi_width)}")
        for w in merged_windows:
            print(f"  {w[2]}: [{w[0]}, {w[1]}) (len={w[1]-w[0]})")

        trend_seq = [w[2] for w in merged_windows if w[2] in ('up', 'down')]
        ok = True
        for i in range(1, len(trend_seq)):
            if trend_seq[i] == trend_seq[i-1]:
                print(f"[Verify] Warning: Episode {epi_idx} has consecutive '{trend_seq[i]}' at positions {i-1},{i}")
                ok = False
        if ok:
            print(f"[Verify] up/down alternation OK for episode {epi_idx}")
            ok_count += 1

        new_label = np.zeros_like(epi_width)
        valid_windows = merged_windows.copy()
        if valid_windows and valid_windows[0][2] == 'flat':
            valid_windows = valid_windows[1:]
        if valid_windows and valid_windows[-1][2] == 'flat':
            valid_windows = valid_windows[:-1]
        for idx, w in enumerate(valid_windows):
            if w[2] == 'flat':
                prev = valid_windows[idx-1] if idx > 0 else None
                nextw = valid_windows[idx+1] if idx+1 < len(valid_windows) else None
                if prev and nextw and prev[2] == 'down' and nextw[2] == 'up':
                    length = w[1] - w[0]
                    if length > 1:
                        new_label[w[0]:w[1]] = np.linspace(1, 0, length)
                else:
                    new_label[w[0]:w[1]] = 0
            else:
                new_label[w[0]:w[1]] = 0
        all_labels.append(new_label)

        if do_plot:
            plt.figure(figsize=(10, 4))
            plt.plot(epi_width, label='gripper width')
            plt.plot(new_label, label='new label', color='blue', linewidth=2)
            colors = {'flat': 'gray', 'up': 'green', 'down': 'red'}
            for w in merged_windows:
                plt.axvspan(w[0], w[1], color=colors.get(w[2], 'blue'), alpha=0.2, label=w[2] if w[0]==0 else None)
                plt.text((w[0]+w[1])/2, np.max(epi_width), w[2], color=colors.get(w[2],'blue'), fontsize=8, ha='center', va='bottom')
            plt.title(f'Episode {epi_idx}')
            plt.xlabel('Timestep')
            plt.ylabel('Gripper Width / Label')
            plt.legend()
            plt.tight_layout()
            plt.show()
        start = end
    # 拼接所有label，保存到zarr
    left_wrist_mask_rate = np.concatenate(all_labels).reshape(-1, 1)
    if 'left_wrist_mask_rate' in zarr_root['data']:
        del zarr_root['data']['left_wrist_mask_rate']
    zarr_root['data'].create_dataset('left_wrist_mask_rate', data=left_wrist_mask_rate, dtype='float32', chunks=(10000, 1), overwrite=True)
    print(f"Saved left_wrist_mask_rate to zarr, shape: {left_wrist_mask_rate.shape}")
    print(f"\n[Summary] OK episodes: {ok_count}/{len(episode_ends)} ({ok_count/len(episode_ends)*100:.1f}%)")

