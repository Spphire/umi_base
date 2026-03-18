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

The task is phase-structured:

1. Grasping phase: wrist information dominates.
2. Transport/release phase: head information is more important for left/right decision.

If all phases are uniformly mixed in minibatches, optimization may over-focus on wrist-dominant gradients.

## 3. Design Goals

1. Increase effective supervision for direction decision features (head view).
2. Reduce side-collapse behavior in rollout.
3. Keep training pipeline minimally invasive and reversible.

## 4. Proposed Sampling Strategies

## S1. Phase-Balanced Sampling (baseline improvement)

Construct per-sample phase labels and enforce batch quotas:

- `pre_grasp`
- `grasp_transition`
- `transport`
- `release`

Expected value:

- Prevents phase under-representation.
- Improves gradient coverage for transport/release phases.

## S2. Phase + Direction Balanced Sampling

Within key decision phases (`transport`, `release`), enforce left/right balance in each batch.

Expected value:

- Directly counteracts direction collapse.

## S3. Wrist-Similar, Head-Different Contrastive Batching (targeted)

Within decision phases, sample pairs/groups where:

- wrist states are similar (pose and/or wrist feature close),
- direction label differs (left vs right),
- head content differs accordingly.

Expected value:

- Forces the model to use head information when wrist cues are ambiguous.

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

1. Build progress labels and direction labels offline.
2. Integrate `S1` sampler path.
3. Add `S2` balancing constraints.
4. Add optional `S3` targeted pairing.

Each step should be gated by ablation results before moving to the next.

## 7. Risks

1. Incorrect phase labeling can bias sampling in the wrong way.
2. Over-constrained batches may reduce data diversity.
3. S3 may increase data loading complexity and training latency.

## 8. Non-Goals for this branch stage

1. No model architecture changes.
2. No reward/objective redesign.
3. No production deployment changes before ablation results.

