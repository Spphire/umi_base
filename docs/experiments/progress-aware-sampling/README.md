# Progress-Aware Sampling

Last updated: 2026-03-18
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
