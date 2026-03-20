# Structured Sampling Newcomer Quickstart

Last updated: 2026-03-20  
Branch: `codex/progress-aware-sampling`

## 1. What problem this solves

For tasks like `q3_mouse`, wrist view is strong for grasping, but head view is
critical for post-grasp direction choice. In mixed training, models may overuse
wrist cues and ignore head cues.

Structured sampling addresses this by shaping batch composition:

1. keep wrist-similar samples together (same local manipulation context)
2. prefer samples with larger head-view difference and future-action difference
3. keep full dataset coverage (no hard data dropping)

## 2. High-level pipeline

### Offline candidate building

For each sample `i`, we precompute a ranked candidate list `C(i)`:

1. find wrist-near neighbors with KNN
2. among those neighbors, rank by joint score
   - `score(i,j) = alpha * d_head(i,j) + beta * d_action(i,j)`
3. store top-`M` candidate indices for runtime sampling

### Runtime batch sampling

At training time, each batch is built as:

1. anchor sample
2. structured neighbors from pre-ranked `C(anchor)`
3. fallback fill from unused global samples to keep coverage

This keeps "same wrist but different head/action" pressure in-batch while still
training on the whole dataset across epochs.

## 3. Code map (what changed and where)

### Core sampler logic

1. `diffusion_policy/common/structured_batch_sampler.py`
   - `StructuredCoverageBatchSampler`
   - builds mixed structured+coverage batches
   - guarantees at-most-once usage per epoch

2. `diffusion_policy/common/structured_dataloader.py`
   - `build_train_dataloader(...)`
   - `set_epoch_for_structured_sampler(...)`
   - toggles structured sampler from Hydra config

3. `diffusion_policy/common/structured_sampling_index.py`
   - `pairwise_l2_knn(...)`
   - `compute_future_action_features(...)`
   - `build_joint_ranked_candidates(...)`
   - offline ranking utilities

4. `diffusion_policy/common/structured_sampling_meta.py`
   - metadata/signature compatibility utilities
   - protects cache reuse across mismatched dataset/encoder/sampler settings

### Workspace integration

1. `diffusion_policy/workspace/train_diffusion_unet_timm_workspace.py`
   - training loader switched to `build_train_dataloader(...)`
   - per-epoch `set_epoch_for_structured_sampler(...)`
   - DDP-safe `freeze_encoder` handling
   - head-importance metric improvements

2. `diffusion_policy/workspace/train_diffusion_transformer_timm_workspace.py`
   - same integration and metric fixes as above

### Policy-side metric stability support

1. `diffusion_policy/policy/diffusion_unet_timm_policy.py`
2. `diffusion_policy/policy/diffusion_transformer_timm_policy.py`
   - `predict_action(...)` now accepts optional `generator`
   - allows masked/unmasked comparisons with identical diffusion noise seed

### Data and run tooling

1. `scripts/generate_structured_ring_candidates.py`
   - fallback candidate file generator
   - useful for smoke tests and missing-file recovery

2. `Makefile`
   - `prepare_structured_candidates`
   - `train_acc8_amp_structured`
   - defaults for structured sampling parameters

3. `diffusion_policy/dataset/real_pick_and_place_image_head_dataset.py`
   - legacy wrist random-mask trick removed in dataset path
   - avoids confounding structured-sampling effect

## 4. Minimal run commands

### Recommended (Makefile path)

```bash
make train_acc8_amp_structured \
  TASK=q3_mouse \
  WKSPACE=train_diffusion_unet_timm_single_frame_workspace \
  LOCAL_DATASET_ZARR=/mnt/workspace/users/shenyibo/umi_base/.cache/q3_mouse_dh_train/replay_buffer.zarr
```

### Manual Hydra override path

```bash
accelerate launch --config_file accelerate/8gpu-amp.yaml train.py \
  --config-name train_diffusion_unet_timm_single_frame_workspace \
  task=q3_mouse \
  task.dataset.local_files_only=/mnt/workspace/users/shenyibo/umi_base/.cache/q3_mouse_dh_train/replay_buffer.zarr \
  +dataloader.structured_sampling.enabled=true \
  +dataloader.structured_sampling.candidate_indices_path=/mnt/workspace/users/shenyibo/umi_base/.cache/structured_sampling/q3_mouse/candidate_indices_ring_top192.npy \
  +dataloader.structured_sampling.structured_ratio=0.5 \
  +dataloader.structured_sampling.seed=42
```

## 5. How to verify it is really active

Check one of these:

1. dataloader object type becomes `StructuredCoverageBatchSampler`
2. training starts without missing-candidate errors
3. logs include normal train/val metrics while structured overrides are enabled

For head-importance metrics, note:

1. masked/unmasked comparisons now use same diffusion seed per pair
2. optional override: `training.head_importance_eval_seed=<int>`

## 6. Common failure modes and fixes

1. `candidate_indices first dim must be N=..., got ...`
   - candidate file does not match current dataset length
   - regenerate candidates for the exact zarr path used in training

2. candidate file missing
   - run `make prepare_structured_candidates` first

3. multi-GPU freeze encoder error (`DDP` has no `obs_encoder`)
   - fixed in this branch by unwrapping model before freeze calls

4. noisy or unstable head-importance values
   - use this branch (shared-seed masked/unmasked metric path)
   - prefer trend over many steps, not one-step conclusions

## 7. Suggested newcomer workflow

1. run 1-epoch smoke with structured off (baseline)
2. run 1-epoch smoke with structured on (same config except sampler toggles)
3. compare train/val loss and head-importance trends
4. then launch full training and deployment eval

This is the fastest path to understand whether structured sampling helps a new task.
