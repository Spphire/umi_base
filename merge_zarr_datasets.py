import zarr
import numpy as np
from loguru import logger
import os
import os.path as osp


def merge_zarr_datasets(data_path_list, output_path):
    """
    Merge multiple zarr datasets into one.
    """
    if len(data_path_list) < 2:
        raise ValueError("At least 2 datasets are required for merging")

    # Check all input paths exist
    for path in data_path_list:
        zarr_path = osp.join(path, 'replay_buffer.zarr')
        if not os.path.exists(zarr_path):
            raise FileNotFoundError(f"Zarr dataset not found at {zarr_path}")

    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    output_zarr_path = osp.join(output_path, 'replay_buffer.zarr')

    if os.path.exists(output_zarr_path):
        logger.warning(f"Output path {output_zarr_path} already exists, removing...")
        os.system(f'rm -rf {output_zarr_path}')

    # Load all zarr datasets
    zarr_datasets = []
    for path in data_path_list:
        zarr_path = osp.join(path, 'replay_buffer.zarr')
        zarr_root = zarr.open(zarr_path, mode='r')
        zarr_datasets.append(zarr_root)
        logger.info(f"Loaded dataset from {zarr_path}")

    # Get data keys from first dataset
    first_data = zarr_datasets[0]['data']
    data_keys = list(first_data.keys())
    logger.info(f"Data keys to merge: {data_keys}")

    # Merge data arrays
    merged_data = {}
    for key in data_keys:
        arrays = []
        for ds in zarr_datasets:
            if key in ds['data']:
                arrays.append(np.array(ds['data'][key]))
            else:
                logger.warning(f"Key '{key}' not found in one of the datasets, skipping this key")
                break
        if len(arrays) == len(zarr_datasets):
            merged_data[key] = np.concatenate(arrays, axis=0)
            logger.info(f"Merged '{key}': shape {merged_data[key].shape}")

    # Merge episode_ends with offset adjustment and create dagger mask
    episode_ends_list = []
    dagger_mask_list = []
    offset = 0
    for i, ds in enumerate(zarr_datasets):
        eps_ends = np.array(ds['meta']['episode_ends'])
        episode_ends_list.append(eps_ends + offset)
        offset = episode_ends_list[-1][-1]
        # First dataset episodes are not dagger, others are dagger
        is_dagger = (i > 0)
        dagger_mask_list.append(np.full(len(eps_ends), is_dagger, dtype=bool))
    merged_episode_ends = np.concatenate(episode_ends_list, axis=0)
    merged_dagger_mask = np.concatenate(dagger_mask_list, axis=0)
    logger.info(f"Merged episode_ends: {len(merged_episode_ends)} episodes, total frames: {merged_episode_ends[-1]}")
    logger.info(f"Dagger episodes: {merged_dagger_mask.sum()} / {len(merged_dagger_mask)}")

    # Create output zarr file
    zarr_root = zarr.group(output_zarr_path)
    zarr_data = zarr_root.create_group('data')
    zarr_meta = zarr_root.create_group('meta')

    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)

    # Save merged data
    for key, data in merged_data.items():
        if 'img' in key:
            # Image data: use smaller chunks
            if len(data.shape) == 4:
                chunk_size = (100,) + data.shape[1:]
            else:
                chunk_size = (100,) + data.shape[1:]
            zarr_data.create_dataset(key, data=data, chunks=chunk_size, dtype=data.dtype,
                                     overwrite=True, compressor=compressor)
        else:
            # Non-image data
            if len(data.shape) == 1:
                chunk_size = (10000,)
            else:
                chunk_size = (10000,) + data.shape[1:]
            zarr_data.create_dataset(key, data=data, chunks=chunk_size, dtype=data.dtype,
                                     overwrite=True, compressor=compressor)

    # Save episode_ends and dagger
    zarr_meta.create_dataset('episode_ends', data=merged_episode_ends, chunks=(10000,),
                             dtype='int64', overwrite=True, compressor=compressor)
    zarr_meta.create_dataset('dagger_mask', data=merged_dagger_mask, chunks=(10000,),
                             dtype='bool', overwrite=True, compressor=compressor)

    logger.info('Merged zarr data structure:')
    logger.info(zarr_data.tree())
    logger.info(f"Total frames: {merged_episode_ends[-1]}, Total episodes: {len(merged_episode_ends)}")
    logger.info(f"Saved merged data at {output_zarr_path}")

    return output_zarr_path


if __name__ == '__main__':
    # data_path_list = [
    #     "/home/wendi/Desktop/openpi/data/blocksv2_100",
    #     "/home/wendi/Desktop/openpi/data/blocksv2_dagger_v2_50",
    # ]
    # data_path_list = [
    #     "/home/wendi/Desktop/openpi/data/pourmsg_100",
    #     "/home/wendi/Desktop/openpi/data/pourmsg_dagger_50",
    # ]
    # data_path_list = [
    #     "/home/wendi/Desktop/openpi/data/towel_100",
    #     "/home/wendi/Desktop/openpi/data/towel_dagger_50",
    # ]
    # data_path_list = [
    #     "/home/wendi/Desktop/openpi/data/towel_100",
    #     "/home/wendi/Desktop/openpi/data/towel_dagger_v2_50",
    # ]
    # data_path_list = [
    #     "/home/wendi/Desktop/openpi/data/towelv2_100",
    #     "/home/wendi/Desktop/openpi/data/towelv2_dagger_v3_25",
    # ]
    data_path_list = [
        "/home/wendi/Desktop/openpi/data/towelv2_100",
        "/home/wendi/Desktop/openpi/data/towelv2_dagger_offline_25_clip",
    ]
    # Generate output path by combining input folder names
    folder_names = [osp.basename(p) for p in data_path_list]
    merged_name = "_".join(folder_names)
    output_path = osp.join(osp.dirname(data_path_list[0]), merged_name)

    logger.info(f"Merging {len(data_path_list)} datasets into {output_path}")
    merge_zarr_datasets(data_path_list, output_path)
