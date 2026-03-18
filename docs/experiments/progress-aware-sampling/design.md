# Design: Progress-Aware Sampling

Last updated: 2026-03-18
Branch: `codex/progress-aware-sampling`

## 1. Problem Statement

Current behavior on `q3_mouse`-style tasks suggests weak head-view usage:

- Human intuition: head view is required for left/right placement decision.
- Model behavior:
  - UNet tends to collapse to mostly one side.
  - Transformer tends to collapse to the opposite side.
- In training diagnostics, head-masking impact is smaller than expected.

This indicates shortcut learning: the policy may rely on wrist-centric signals that are strong during grasping but weak for direction decision.

## 2. Working Hypothesis

The key issue is modality shortcut:

1. Wrist view often explains a large part of training loss.
2. Head view carries decision-critical context but can be ignored.

If we explicitly sample "same wrist context but different head context", the model is pressured to use head information.

## 3. Design Goals

1. Increase effective supervision for direction decision features (head view).
2. Reduce side-collapse behavior in rollout.
3. Keep training pipeline minimally invasive and reversible.

## 4. Proposed Sampling Strategies

## P1. Embedding-Based Cross-View Sampler (highest priority)

For each anchor sample:

1. find nearest neighbors by wrist embedding
2. rank candidates by weighted joint score:
   - head embedding distance
   - future-action distance
3. compose mixed batches with structured and random samples

Expected value:

1. task-agnostic
2. no dependency on direction labels or absolute action coordinates
3. direct pressure on head-view utilization

## P2. Weight Tuning for Joint Score (optional)

Tune weighted combination between head-distance and future-action distance.

Expected value:

1. avoids trivial "head different but supervision almost same" pairs
2. improves stability across tasks

## P3. Progress/Direction Labeling (optional enhancement)

Use progress/direction labels only as secondary constraints after P1 is validated.

Expected value:

1. may improve phase coverage in some tasks
2. should not be the primary dependency for generalization

## 5. Rollout and Training Acceptance Signals

Primary signals:

1. Directional balance in rollout success (left and right both perform).
2. Improvement in head-masking sensitivity metrics:
   - `train_action_mse_head_importance`
   - `val_action_mse_head_importance`

Secondary signals:

1. Overall success rate does not regress.
2. No major instability in train/val loss trends.

## 6. Implementation Order (planned)

1. Build offline wrist/head embedding cache.
2. Integrate `P1` sampler path.
3. Add optional `P2` weight tuning and schedule.
4. Add optional `P3` progress/label constraints only if needed.

Each step should be gated by ablation results before moving to the next.

## 7. Risks

1. Embedding quality may limit neighbor validity.
2. Over-constrained pair selection may reduce diversity.
3. Offline neighbor index may become stale if data distribution changes.

## 8. Non-Goals for this branch stage

1. No model architecture changes.
2. No reward/objective redesign.
3. No production deployment changes before ablation results.

## 9. Data Pipeline Notes from Existing Scripts

Relevant references:

1. `post_process_scripts/post_process_data_vr_mouse.py`
2. `scripts/analyze_gripper_trend.py`
3. `sign_accuracy_pick_place.py`

Decisions for this branch:

1. Keep progress labeling independent from aruco calibration.
   - Current converter already calls `use_aruco_calibration=False`.
   - Aruco logic is considered low value for this single-head/single-arm task and should not block labeling.
2. Treat hand/arm masking (GroundedSAM + SAM) as a separate low-priority module.
   - Do not couple SAM stability work with progress-aware sampling rollout.
   - If revisited, expose a standalone function interface that converter scripts can call in one switch.

## 10. Labeling Strategy Source of Truth (secondary path)

Progress extraction will combine two existing ideas:

1. Trend segmentation from `scripts/analyze_gripper_trend.py`
   - smooth width, classify `flat/up/down`, merge short noisy windows.
2. Closing-window localization from `sign_accuracy_pick_place.py`
   - pick the dominant close segment near width minimum with `min_len` and `min_drop` constraints.

This hybrid approach remains available as an optional enhancement, but not as the primary sampler path.
