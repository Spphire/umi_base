# Diffusion Policy Training Guide

This repository contains the training and evaluation code used for the lab's
image-conditioned diffusion policies. This guide focuses on the current
dualfold workflow: process Quest 3 and wrist-camera recordings, build a strict
30 Hz Zarr dataset, select Hydra configs, and launch multi-GPU training.

## 1. Current Infrastructure

The canonical H200 checkout is:

```text
/mnt/workspace/shenyibo/diffusion_policy
```

The workspace is shared by the machines reached through SSH ports `4041` and
`4042` on `106.14.2.243`. The active source branch is:

```text
codex/dit-place-cup-no-tcp-viz
```

The Git remote is:

```text
git@github.com:Spphire/umi_base.git
```

Training data, virtual environments, checkpoints, logs, and generated analysis
results are machine-local artifacts. They must not be committed to Git.

## 2. Repository Layout

```text
diffusion_policy/
  config/task/       Observation views, action dimensions, and dataset class
  config/*.yaml      Model, horizon, optimizer, and training settings
  dataset/           Zarr dataset readers and action conversion
  policy/            Diffusion policy implementation
  workspace/         Training loops and checkpoint handling
accelerate/          Multi-GPU launch configurations
analysis/            Reusable offline-analysis scripts
data/outputs/        Training runs; ignored by Git
.cache/              Local Zarr datasets and model caches; ignored by Git
train.py             Hydra training entry point
Makefile             Common single-node launch targets
```

The raw-data alignment and canonical Zarr builder live in a separate Windows
workspace:

```text
W:\vr_align_pipeline
```

## 3. Git Workflow

Always inspect the worktree before pulling or committing:

```bash
cd /mnt/workspace/shenyibo/diffusion_policy
git status --short --branch
git fetch origin
git log --oneline --decorate HEAD..origin/codex/dit-place-cup-no-tcp-viz
```

This training checkout often contains active runs and local datasets. Do not
use `git reset --hard`, and do not add directories such as `data/outputs`,
`.cache`, `.venv*`, `.pydeps*`, `wandb`, or `checkpoints`.

For a normal source-only change:

```bash
git add README.md diffusion_policy/config accelerate scripts
git diff --cached --stat
git diff --cached
git commit -m "Describe the source change"
git push origin codex/dit-place-cup-no-tcp-viz
```

Use explicit paths with `git add`; avoid `git add -A` in the training checkout.

## 4. Environment Setup

### 4.1 Use the existing H200 environment

The known-good shared environment is:

```bash
cd /mnt/workspace/shenyibo/diffusion_policy
source .venv4041/bin/activate
export HYDRA_FULL_ERROR=1
export HF_HUB_OFFLINE=1
```

At the time of writing it uses Python 3.12, PyTorch 2.7.1 with CUDA 12.6,
Accelerate 1.14, Hydra 1.3, and Zarr 2.18. Keep one training run inside one
environment; do not mix this venv with the older `conda umi` environment while
debugging a checkpoint.

Verify the environment before launching a long run:

```bash
python -c "import torch, accelerate, hydra, zarr; print(torch.__version__, torch.version.cuda); print(accelerate.__version__, hydra.__version__, zarr.__version__); print(torch.cuda.device_count())"
python -c "import diffusion_policy; print(diffusion_policy.__file__)"
nvidia-smi
```

`HF_HUB_OFFLINE=1` requires the TIMM/Hugging Face vision backbone to already be
present in the machine cache. Remove that variable only when intentionally
populating the cache on a machine with network access.

### 4.2 Create a fresh training environment

Clone the branch and create a venv from the repository root:

```bash
git clone --branch codex/dit-place-cup-no-tcp-viz git@github.com:Spphire/umi_base.git diffusion_policy
cd diffusion_policy
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

The repository is imported directly from its root and does not currently use
an editable package install. Robot deployment also needs ROS and hardware SDKs;
those are not required for dataset-only training.

For a different CUDA driver, select a matching official PyTorch wheel instead
of copying binary packages from another server. Validate CUDA with:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## 5. Data Processing

### 5.1 Install the Windows VR alignment pipeline

On the recording PC:

```powershell
py -3 -m pip install -r W:\vr_align_pipeline\requirements.txt
```

Quest 3 must be visible to `adb`, and the datacloud service must be reachable.
The pipeline pairs one head session and two wrist sessions by `parent_uuid`.

### 5.2 Pull, pair, align, and repair a recording batch

Use one dataset name consistently for the Quest task tag and datacloud
identifier:

```powershell
py -3 W:\vr_align_pipeline\scripts\run_vr_align_workflow.py `
  --project-root W:\vr_align_pipeline `
  --dataset-name <dataset_name> `
  --task-tag <task_tag> `
  --identifier <datacloud_identifier> `
  --date-prefix <YYYYMMDD> `
  --overwrite `
  --storage-mode hardlink `
  --visualize `
  --visualize-mode outliers `
  --smooth-isolated-spikes
```

Important outputs are:

```text
W:\vr_align_pipeline\data\<dataset_name>\paired
W:\vr_align_pipeline\outputs\<dataset_name>\aligned_trajectories.csv
W:\vr_align_pipeline\outputs\<dataset_name>\final_side_report.csv
W:\vr_align_pipeline\outputs\<dataset_name>\shared_mount_summary.json
```

Review the report and trajectory visualizations before building a training
dataset. The normal accepted categories are:

- `fully_calibrated`: both wrist trajectories passed calibration.
- `mount_repaired_consistent`: a bad wrist was repaired from trusted Quest
  controller motion using a shared mount transform.
- `mount_repaired_exclude`: optional, only include after manual review.
- `mount_repaired_review`: unresolved review category; exclude by default.

### 5.3 Build the canonical strict-30-Hz Zarr

Use `W:\vr_align_pipeline\scripts\build_dualfold_30hz_zarr.py` for new dualfold
datasets. Do not use older converters that shift whole arrays or concatenate
episodes without rebuilding episode boundaries.

Example with three source exports:

```powershell
py -3 W:\vr_align_pipeline\scripts\build_dualfold_30hz_zarr.py `
  --source W:\vr_align_pipeline\exports\dualfold_pink_0608_p1_split `
  --source W:\vr_align_pipeline\exports\dualfold_pink_0610_p1_split `
  --source W:\vr_align_pipeline\exports\dualfold_pink_recovery_0708_split `
  --workspace-dir W:\vr_align_pipeline\collections\dualfold_pink_training `
  --category fully_calibrated `
  --category mount_repaired_consistent `
  --overwrite
```

The output is:

```text
W:\vr_align_pipeline\collections\dualfold_pink_training\zarr\replay_buffer.zarr
```

The converter has deliberately different resampling rules:

- The episode time grid is exactly 30 Hz.
- Pose and gripper signals are interpolated on that grid.
- Wrist trajectories repaired from lower-rate Quest tracking are also
  interpolated; they are not nearest-frame sampled.
- Images use the nearest timestamp because image interpolation is invalid.
- The default maximum pose bracket gap is 0.10 s.
- The default maximum image timestamp gap is 0.12 s.
- Each source session remains a separate episode. `episode_ends` must never
  join the end of one recording to the start of another.
- Absolute `action` and state/target pose values are generated from the same
  resampled trajectory before the dataset class converts actions to relative
  form for training.

The default categories are `fully_calibrated` and
`mount_repaired_consistent`. Specify `--category mount_repaired_exclude`
explicitly only when that category has been approved.

### 5.4 Transfer the Zarr to the training server

Keep datasets under `.cache` so Git never sees them:

```bash
rsync -a --info=progress2 \
  /path/to/dualfold_pink_training/ \
  root@106.14.2.243:/mnt/workspace/shenyibo/diffusion_policy/.cache/dualfold_pink_training/
```

When connecting to the H200 through a nonstandard SSH port, add the appropriate
transport option, for example `-e "ssh -p 4041"`.

The dataset path passed to Hydra must point at the actual Zarr root:

```text
/mnt/workspace/shenyibo/diffusion_policy/.cache/dualfold_pink_training/zarr/replay_buffer.zarr
```

### 5.5 Read-only Zarr smoke check

Run this before training:

```bash
DATASET=/mnt/workspace/shenyibo/diffusion_policy/.cache/dualfold_pink_training/zarr/replay_buffer.zarr
python - "$DATASET" <<'PY'
import sys
import zarr

root = zarr.open_group(sys.argv[1], mode="r")
data = root["data"]
episode_ends = root["meta"]["episode_ends"][:]
print("keys:", sorted(data.array_keys()))
print("action:", data["action"].shape, data["action"].dtype)
print("episodes:", len(episode_ends), "frames:", int(episode_ends[-1]))
assert data["action"].shape[0] == int(episode_ends[-1])
assert all(b > a for a, b in zip([0, *episode_ends[:-1]], episode_ends))
PY
```

For dual-arm training, `action.shape[1]` must be 20. For left-only or
right-only training, the same 20-D Zarr can be used; the dataset config applies
`action_slice: left` or `action_slice: right` and exposes 10 dimensions.

## 6. Choosing Configs

Hydra composes two independent config layers:

1. The task config selects observation keys, action dimensions, and the dataset
   reader.
2. The workspace config selects the model, temporal horizons, optimizer,
   epochs, checkpoint policy, and output naming.

Always pass both explicitly on the command line. Do not rely on defaults in the
Makefile or workspace file.

### 6.1 Task configs

| Training target | Task override | RGB observations | Action |
| --- | --- | --- | --- |
| Dual arm, wrist only | `task=dualfold_2view_20action` | left wrist, right wrist | 20-D |
| Dual arm, wrists + head | `task=dualfold_3view_20action` | left wrist, right wrist, head | 20-D |
| Left arm only | `task=dualfold_left_1view_10action` | left wrist | 10-D left slice |
| Right arm only | `task=dualfold_right_1view_10action` | right wrist | 10-D right slice |
| Dual pick cube, 3 view | `task=dualpickcube_3view_20action` | left wrist, right wrist, head | 20-D |

For a dual-arm sample, the 20 action dimensions are left arm pose/gripper
followed by right arm pose/gripper. Each arm contributes position, 6-D
rotation, and gripper width.

### 6.2 Workspace configs

| Use case | `--config-name` |
| --- | --- |
| Recommended dualfold, shared RGB encoder | `train_diffusion_transformer_timm_dualfold_2frame_30horizon_workspace_sharergb` |
| Two observation frames, separate RGB encoders | `train_diffusion_transformer_timm_dualfold_2frame_30horizon_workspace` |
| One observation frame | `train_diffusion_transformer_timm_dualfold_1frame_30horizon_workspace` |

The recommended shared-RGB workspace currently means:

```text
horizon = 30                 model training/prediction sequence length
n_obs_steps = 2              two observation frames
n_action_steps = 8           runner execution slice, not the training horizon
num_inference_steps = 16      DDIM denoising steps
share_rgb_model = true       one vision backbone shared across camera views
num_epochs = 600
batch_size = 48 per process
mixed_precision = bf16       selected by the Accelerate config
```

Do not confuse `horizon=30` with `num_inference_steps`. The former is the
predicted temporal sequence length; the latter is the number of DDIM denoising
iterations used to produce that sequence.

### 6.3 Hydra overrides

The dataset path should normally be overridden at launch time:

```text
task.dataset.dataset_path=/absolute/path/to/replay_buffer.zarr
```

Common safe overrides are:

```text
exp_name=<descriptive_run_name>
training.num_epochs=600
dataloader.batch_size=48
training.seed=42
```

Hydra stores the fully resolved launch config in the run's `.hydra` directory.
That file is the source of truth when reproducing an old checkpoint.

## 7. Launching Training

All commands below must run from the repository root with the intended Python
environment active.

### 7.1 Eight GPUs with BF16 through Make

```bash
cd /mnt/workspace/shenyibo/diffusion_policy
source .venv4041/bin/activate

make train_acc8_amp \
  TASK=dualfold_2view_20action \
  WKSPACE=train_diffusion_transformer_timm_dualfold_2frame_30horizon_workspace_sharergb \
  EXTRA_ARGS="task.dataset.dataset_path=/mnt/workspace/shenyibo/diffusion_policy/.cache/dualfold_pink_training/zarr/replay_buffer.zarr exp_name=dualfold_pink_training_2view"
```

`make train_acc8_amp` uses `accelerate/8gpu-amp.yaml`, enables BF16, and exports
`HF_HUB_OFFLINE=1` and `HYDRA_FULL_ERROR=1`.

### 7.2 Eight GPUs with BF16 through Accelerate directly

The direct form is easier to audit in a launch script:

```bash
export HYDRA_FULL_ERROR=1
export HF_HUB_OFFLINE=1

python -m accelerate.commands.launch \
  --config_file accelerate/8gpu-amp.yaml \
  train.py \
  --config-name train_diffusion_transformer_timm_dualfold_2frame_30horizon_workspace_sharergb \
  task=dualfold_3view_20action \
  task.dataset.dataset_path=/mnt/workspace/shenyibo/diffusion_policy/.cache/dualfold_pink_training/zarr/replay_buffer.zarr \
  exp_name=dualfold_pink_training_3view
```

### 7.3 Six GPUs with BF16

Use the checked-in six-GPU config:

```bash
python -m accelerate.commands.launch \
  --config_file accelerate/6gpu-amp.yaml \
  train.py \
  --config-name train_diffusion_transformer_timm_dualfold_2frame_30horizon_workspace_sharergb \
  task=dualfold_2view_20action \
  task.dataset.dataset_path=/mnt/workspace/shenyibo/diffusion_policy/.cache/dualfold_pink_training/zarr/replay_buffer.zarr \
  exp_name=dualfold_pink_training_2view_6gpu
```

The file `accelerate/6gpu-amp.yaml` currently selects GPU IDs 0 through 5. Edit
or create another Accelerate config when the free devices are different.

### 7.4 Single-GPU smoke test

Before consuming all GPUs, test config composition and one short train step:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config-name train_diffusion_transformer_timm_dualfold_2frame_30horizon_workspace_sharergb \
  task=dualfold_2view_20action \
  task.dataset.dataset_path=/mnt/workspace/shenyibo/diffusion_policy/.cache/dualfold_pink_training/zarr/replay_buffer.zarr \
  exp_name=smoke_dualfold_2view \
  training.debug=true \
  training.num_epochs=1 \
  training.max_train_steps=2 \
  training.max_val_steps=1 \
  logging.mode=disabled
```

This still creates a small run under `data/outputs`; remove it only after
confirming no useful diagnostics are needed.

### 7.5 Keep a long run alive

Use `tmux` so an SSH disconnect does not terminate training:

```bash
tmux new -s dualfold_2view
cd /mnt/workspace/shenyibo/diffusion_policy
source .venv4041/bin/activate
bash launch_dualfold_pink_2view_4041.sh 2>&1 | tee logs/dualfold_2view.log
```

Detach with `Ctrl-b d`, and reconnect with:

```bash
tmux attach -t dualfold_2view
```

When two servers share `/mnt/workspace`, give simultaneous runs different
`exp_name` values so their output directories cannot collide.

## 8. Monitoring and Outputs

Runs are written under:

```text
data/outputs/YYYY.MM.DD/HH.MM_dit_<task_name>_<exp_name>/
```

Important files are:

```text
.hydra/config.yaml            resolved training configuration
logs.json.txt                 epoch and loss metrics
normalizer.pkl                dataset normalizer used by the run
checkpoints/latest.ckpt       latest checkpoint
checkpoints/epoch=*.ckpt      top-k or periodic checkpoints
```

Monitor the process and GPUs:

```bash
nvidia-smi
watch -n 2 nvidia-smi
ps -ef | grep '[t]rain.py'
tail -f data/outputs/<run>/logs.json.txt
ls -lh data/outputs/<run>/checkpoints
```

A healthy run should have nonzero GPU utilization on all selected devices,
regularly advancing epochs, finite train/validation loss, and a periodically
updated `latest.ckpt`. Allocated memory with 0% utilization usually means the
workers are blocked on data loading, synchronization, or an exception; inspect
the launcher output and all ranks before restarting.

## 9. Resuming a Run

`training.resume=true` loads `checkpoints/latest.ckpt` from the current Hydra
output directory. Therefore, point Hydra back to the original run directory;
otherwise a new timestamped directory will not contain the checkpoint.

```bash
RUN_DIR=/mnt/workspace/shenyibo/diffusion_policy/data/outputs/2026.07.09/01.02_dit_dualfold_2view_20action_example

python -m accelerate.commands.launch \
  --config_file accelerate/8gpu-amp.yaml \
  train.py \
  --config-name train_diffusion_transformer_timm_dualfold_2frame_30horizon_workspace_sharergb \
  task=dualfold_2view_20action \
  task.dataset.dataset_path=/mnt/workspace/shenyibo/diffusion_policy/.cache/dualfold_pink_training/zarr/replay_buffer.zarr \
  training.resume=true \
  training.num_epochs=1200 \
  hydra.run.dir="$RUN_DIR"
```

Before resuming, keep a named copy of a milestone checkpoint when it matters:

```bash
cp "$RUN_DIR/checkpoints/latest.ckpt" "$RUN_DIR/checkpoints/epoch_0600.ckpt"
```

Confirm the resolved dataset path, model config, and world size are unchanged.
Changing `num_epochs` is expected; changing observation keys, action dimensions,
or encoder sharing is not compatible with the old checkpoint.

## 10. Offline Checkpoint Evaluation

The lightweight policy-vs-ground-truth tool accepts explicit paths and episode
ranges:

```bash
CUDA_VISIBLE_DEVICES=0 python visualize_policy_vs_gt_trajectory.py \
  --ckpt data/outputs/<run>/checkpoints/latest.ckpt \
  --dataset /mnt/workspace/shenyibo/diffusion_policy/.cache/dualfold_pink_training/zarr/replay_buffer.zarr \
  --episodes 0:10 \
  --rollout-mode teacher_forcing \
  --no-vis
```

Use `teacher_forcing` to evaluate each prediction against the dataset state at
that time. Use `open_loop` to feed the previous prediction back as the next base
state and expose accumulated drift. Generated summaries and figures should stay
outside Git.

The scripts under `analysis/nedf_frozen_replay_20260615` reproduce the frozen
input, denoising-step, seed-sweep, and quaternion-jump checks used for the NEDF
deployment investigation. Commit the Python scripts, not frozen `.npz` inputs
or generated JSON/TXT reports.

## 11. Exporting a Run for Deployment

Copy the complete run metadata and only `latest.ckpt` when deployment does not
need historical checkpoints:

```bash
rsync -a --info=progress2 \
  --include='checkpoints/' \
  --include='checkpoints/latest.ckpt' \
  --exclude='checkpoints/*.ckpt' \
  data/outputs/<run>/ \
  noe_yiboshen:/home/yiboshen/<deployment_folder>/<run>/
```

Verify both size and checksum after transfer:

```bash
sha256sum data/outputs/<run>/checkpoints/latest.ckpt
ssh noe_yiboshen sha256sum /home/yiboshen/<deployment_folder>/<run>/checkpoints/latest.ckpt
```

The deployment copy should retain `.hydra/config.yaml`, logs, normalizer files,
and other run metadata. Only older `.ckpt` files are excluded.

## 12. Preflight Checklist

Before a full multi-GPU launch, verify all of the following:

- The selected Zarr opens read-only and `episode_ends[-1]` equals total frames.
- The task config observation keys exist in the Zarr.
- Dual-arm action width is 20; single-arm configs use the intended 10-D slice.
- The workspace has `horizon=30`, `n_obs_steps=2`, and the intended RGB-sharing
  setting.
- `task.dataset.dataset_path` resolves to the exact dataset being trained.
- The launch config's process count matches the allocated GPUs.
- No other user is using the selected GPUs.
- The smoke test completes without NaN/Inf loss.
- The run has a unique and descriptive `exp_name`.
- `data/outputs`, `.cache`, logs, environments, and checkpoints remain ignored
  by Git.
