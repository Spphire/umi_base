# Repository 整理记录

## 来源

- `server-1024:/mnt/workspace/shenyibo/codex-dit-place-cup-no-tcp-viz`

## 当前状态

- `server-1024` 源目录不是 Git 工作树。
- 原目录约 `48G`，其中主要体积来自 `data/`、`.cache/`、`.pip-cache/`、`.pydeps1022/`、虚拟环境、日志和训练 checkpoint。
- 本仓库整理为源码快照，不包含训练数据、模型权重、虚拟环境、缓存、日志和临时调试产物。

## 已排除的本地/训练产物

- `data/`
- `.cache/`
- `.pip-cache/`
- `.pydeps*/`
- `.venv*/`
- `logs/`
- `debug_sign_accuracy/`
- `*.ckpt`, `*.pt`, `*.pth`, `*.safetensors`, `*.onnx`
- `*.zarr/`, `*.hdf5`, `*.h5`
- `nvidia_output.txt`, `.last_train_log`
- `*.deb`

## 待合并来源

- `dsw22:/mnt/data/users/shenyibo/workspace/codex-dit-place-cup-no-tcp-viz`

当前本机 SSH 配置中 `dsw22` 只有主机名 `dsw22`，没有可解析的 `HostName` 或跳板配置；直接连接失败。拿到可访问地址或 SSH alias 后，应再生成同样的源码快照，与本仓库做文件级 diff 后合并。

## 历史备注

原始 `README.md` 开头包含一个私有 checkpoint 路径和 `scp` 命令，整理时已从首页移除：

```text
/mnt/data/shenyibo/workspace/umi_base/data/outputs/2026.02.24/18.51.58_train_diffusion_unet_timm_q3_choose_block_198_vqa/checkpoints/latest.ckpt
scp -C junjie@8.153.94.10:/mnt/data/shenyibo/workspace/umi_base/data/outputs/2026.02.24/18.51.58_train_diffusion_unet_timm_q3_choose_block_198_vqa/checkpoints/latest.ckpt ~/choose_block_vqa.ckpt
```
