import zarr

# 替换为你的 Zarr 文件路径
zarr_path = "/mnt/data/shenyibo/workspace/umi_base/.cache/q3_shop_bagging_0202/replay_buffer.zarr"
z = zarr.open(zarr_path, mode='r')
print(z.keys())
# 检查 action 的 shape 和内容
print("Action shape in Zarr file:", z['action'].shape)
print("Action data (first 10):", z['action'][:10])