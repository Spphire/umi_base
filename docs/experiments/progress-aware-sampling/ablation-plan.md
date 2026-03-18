# Ablation Plan: Progress-Aware Sampling

Last updated: 2026-03-18
Branch: `codex/progress-aware-sampling`

## 1. Objective

Evaluate whether embedding-based cross-view sampling improves head-view utilization and reduces directional collapse.

## 2. Fixed Conditions

To keep comparisons fair:

1. Same dataset split
2. Same random seed set (at least 3 seeds recommended)
3. Same training budget (epochs, batch size, optimizer schedule)
4. Same eval protocol and checkpoints for reporting

## 3. Experiment Matrix

## E0. Baseline

- Existing sampler (status quo).

## E1. Embedding Cross-View Sampler (primary)

- Wrist-neighbor + joint-score structured sampling.
- Joint score includes:
  - head embedding distance
  - future-action distance
- Mixed batch strategy (structured + random).
- Candidate pool size should be batch-size aware (`M_pool >= 4 * B_struct`).

## E2. E1 + Weight Tuning (optional)

- Tune `alpha/beta` in joint score.

## E3. E1/E2 + Progress/Direction Label Constraints (optional)

- Add label-based gates only if needed after E1/E2 evidence.

## E-1. Label Quality Dry-Run (only if E3 is enabled)

- Run label extraction only (no training).
- Validate:
  - phase histogram
  - left/right ratio
  - anchor visualization on sampled episodes
  - trend alternation sanity checks

## 4. Metrics

## Primary

1. Rollout success by direction:
   - success_left
   - success_right
   - direction_gap = abs(success_left - success_right)
2. Head-view contribution proxy:
   - `train_action_mse_head_importance`
   - `val_action_mse_head_importance`

## Secondary

1. Overall success rate
2. Train/val loss stability
3. Inference consistency across seeds

## 5. Go/No-Go Criteria

Proceed from E1 to E2 only if:

1. No major regression in total success rate
2. Direction gap does not worsen
3. Structured pair diagnostics are healthy (`wrist-close`, `head-far`, and `action-far`)

Proceed from E2 to E3 only if:

1. Direction gap still materially large
2. Sampling overhead remains manageable
3. No evidence that embedding selection alone is sufficient
4. Label quality dry-run passes

## 6. Logging and Reporting Template

For each run record:

- run id
- config name
- sampler mode (`baseline`, `e1`, `e2`, `e3`)
- seed
- checkpoint used for eval
- success_left / success_right / overall_success
- head_importance train/val
- wrist/head distance stats for structured pairs
- action-distance stats for structured pairs
- candidate pool and cap params (`B`, `p_structured`, `M_pool`, `max_per_class`)
- notes on failure mode

## 7. Minimum Deliverables Per Stage

1. One summary table for all seeds
2. One short conclusion section:
   - what improved
   - what regressed
   - recommendation for next stage
