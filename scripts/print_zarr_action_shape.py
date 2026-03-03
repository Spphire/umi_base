import zarr
import argparse
import os
import numpy as np

def main(zarr_path):
    if not os.path.exists(zarr_path):
        print(f"Zarr path not found: {zarr_path}")
        return
    zarr_root = zarr.open(zarr_path, mode='a')
    action = zarr_root['data']['action']
    color_type = np.argmax(action[:,10:13], axis=-1)
    color_value = (1 - color_type).reshape(-1,1)
    print(action[0,10:13], color_type[0])
    print(action[3000,10:13], color_type[3000])
    print(f"action shape: {action.shape}")

    new_action = np.concatenate([action[:,:10], color_value], axis=-1)
    print(new_action.shape)

    # 与数据生成脚本保持一致的chunk size
    chunk_size = (10000, new_action.shape[1])
    zarr_root['data'].create_dataset('action11', data=new_action, chunks=chunk_size, dtype='float32', overwrite=True)
    print("action dataset replaced.")

if __name__ == "__main__":
    main("/mnt/data/shenyibo/workspace/umi_base/.cache/q3_choose_block_dh_vqa/replay_buffer.zarr")