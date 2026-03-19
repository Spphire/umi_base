# umi_base Docs

Last updated: 2026-03-19

This directory stores continuously maintained project notes, focused on:

- Task configs: `q3_mouse`, `q3_hang_cup`
- Primary workspace: `train_diffusion_unet_timm_single_frame_workspace`
- Secondary workspace: `train_diffusion_transformer_timm_single_frame_workspace`

## Structure

- `project-analysis.md`: system-level architecture and train/eval pipeline
- `task-workspace-playbook.md`: daily-use task/workspace cheatsheet and commands
- `experiments/progress-aware-sampling/`: design and execution docs for progress-aware sampling
- `updates/`: date-based running log of changes and findings

## Maintenance Rules

- After each config or training strategy change, append to `updates/YYYY-MM-DD.md`.
- If team aliases differ from real config names, keep a mapping in `task-workspace-playbook.md`.
- Always keep runnable commands with both `--config-name` and `task=...`.
