# Document Development Plan (Feature Branch)

Last updated: 2026-03-18
Branch: `codex/progress-aware-sampling`

## 1. Goal

Complete documentation for the progress-aware sampling feature before code implementation.

## 2. Deliverables

1. `design.md`
2. `progress-label-spec.md`
3. `ablation-plan.md`
4. update log entry in `docs/updates/2026-03-18.md`

## 3. Milestones

## M1. Problem and hypothesis alignment

- Document failure mode and expected causality.
- Confirm scope is sampler-focused (not architecture-focused).

## M2. Labeling contract

- Define phase/direction labels and confidence/fallback behavior.
- Freeze v1 schema for code integration.

## M3. Experiment contract

- Freeze baseline and stage-wise ablation plan.
- Define go/no-go criteria and reporting template.

## M4. Ready-for-code review gate

Checklist:

1. Data labeling rules are testable.
2. Sampling strategy is decomposed into `S1 -> S2 -> S3`.
3. Metrics are directly observable in current training/eval stack.

## 4. Review Notes Template

When reviewing docs, comment on:

1. Ambiguous threshold definitions
2. Missing fallback behavior
3. Evaluation ambiguity (especially direction metrics)
4. Runtime complexity risks

## 5. Out of Scope

1. Implementing sampler classes
2. Editing dataloader or dataset code
3. Changing policy architecture

