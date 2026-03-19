# Progress-Aware Sampling

Last updated: 2026-03-19
Branch: `codex/progress-aware-sampling`

This folder documents sampler design before code changes.
Current highest-priority path is embedding-based cross-view sampling:

- wrist embedding close
- head embedding far
- future-action embedding/action target far (required in joint score)
- analysis encoder signature must match training config

Progress/direction labels are now optional add-ons, not the main path.

## Files

- `design.md`: problem framing, hypotheses, and sampling strategy options
- `embedding-sampler-spec.md`: primary implementation spec for general sampler
- `progress-label-spec.md`: how to derive task progress labels from trajectory data
- `ablation-plan.md`: experiment matrix and acceptance criteria
- `doc-plan.md`: document development plan for this feature branch

## Scope

Current branch already includes v1 implementation skeleton:

- offline metadata utilities (`structured_sampling_meta.py`)
- offline KNN/joint-score utilities (`structured_sampling_index.py`)
- coverage-first structured batch sampler (`structured_batch_sampler.py`)
- train dataloader integration helper (`structured_dataloader.py`)
- workspace integration for UNet/Transformer timm single-frame training

Batch-size-aware default guidance:

- `B_struct = round(batch_size * structured_ratio)`
- candidate pool size `M_pool >= 4 * B_struct` (prefer `6 * B_struct`)
- if class/group cap is used:
  - `max_per_class = max(1, ceil(rho * B_struct))`
  - default `rho = 0.4`

## Runtime Status

Structured batch sampling is integrated and runnable in both timm single-frame workspaces:

- `train_diffusion_unet_timm_workspace.py`
- `train_diffusion_transformer_timm_workspace.py`

Dataset cleanup aligned with this strategy:

- removed legacy wrist random-mask trick in
  `diffusion_policy/dataset/real_pick_and_place_image_head_dataset.py::__getitem__`
- head-view reliance should now come from sampling strategy rather than artificial wrist occlusion

Activation switch:

- `+dataloader.structured_sampling.enabled=true`
- `+dataloader.structured_sampling.candidate_indices_path=...`
- optional:
  - `+dataloader.structured_sampling.structured_ratio=0.5`
  - `+dataloader.structured_sampling.seed=42`

## Verification Evidence

Real-data check (`q3_mouse` local zarr):

- dataset: `C:/Users/yibo/Downloads/umi_base/.cache/q3_mouse_dh_train/replay_buffer.zarr`
- candidate index used in smoke test:
  - `C:/Users/yibo/Downloads/umi_base/.cache/structured_sampling/q3_mouse/candidate_indices_ring_top192.npy`

Sampler type introspection:

1. with structured off:
   - `sampler_obj = None`
   - `batch_sampler_obj = BatchSampler`
2. with structured on:
   - `sampler_obj = StructuredCoverageBatchSampler`
   - `batch_sampler_obj = StructuredCoverageBatchSampler`
   - `is_structured_sampler = True`

Smoke run outputs:

1. baseline (off):
   - `data/outputs/2026.03.18/17.43.01_train_diffusion_unet_timm_q3_mouse_rawresize_sharevit`
2. structured (on):
   - `data/outputs/2026.03.18/18.10.32_train_diffusion_unet_timm_q3_mouse_rawresize_sharevit`

Server robustness note:

1. if candidate file is missing, run:
   - `make prepare_structured_candidates`
2. `make train_acc8_amp_structured` now auto-invokes that preparation step.

DDP freeze-encoder compatibility note:

1. in multi-GPU runs, `freeze_encoder=true` must operate on unwrapped model
2. fixed in all related workspaces by switching from `self.model.obs_encoder` to
   `accelerator.unwrap_model(self.model).obs_encoder` during epoch freeze step
