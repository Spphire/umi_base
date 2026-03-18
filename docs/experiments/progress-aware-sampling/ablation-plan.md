# Ablation Plan: Progress-Aware Sampling

Last updated: 2026-03-18
Branch: `codex/progress-aware-sampling`

## 1. Objective

Evaluate whether progress-aware sampling improves head-view utilization and reduces left/right collapse.

## 2. Fixed Conditions

To keep comparisons fair:

1. Same dataset split
2. Same random seed set (at least 3 seeds recommended)
3. Same training budget (epochs, batch size, optimizer schedule)
4. Same eval protocol and checkpoints for reporting

## 3. Experiment Matrix

## E0. Baseline

- Existing sampler (status quo).

## E1. S1 Phase-Balanced

- Batch has controlled ratio across phase labels.

## E2. S2 Phase + Direction Balanced

- E1 plus left/right balancing in `transport` and `release`.

## E3. S3 Wrist-Similar + Opposite Direction Pairing (optional after E2)

- E2 plus targeted paired/group sampling in decision phases.

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

Proceed from E2 to E3 only if:

1. Direction gap still materially large
2. Sampling overhead remains manageable

## 6. Logging and Reporting Template

For each run record:

- run id
- config name
- sampler mode (`baseline`, `s1`, `s2`, `s3`)
- seed
- checkpoint used for eval
- success_left / success_right / overall_success
- head_importance train/val
- notes on failure mode

## 7. Minimum Deliverables Per Stage

1. One summary table for all seeds
2. One short conclusion section:
   - what improved
   - what regressed
   - recommendation for next stage

