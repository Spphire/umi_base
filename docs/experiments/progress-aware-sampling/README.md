# Progress-Aware Sampling

Last updated: 2026-03-18
Branch: `codex/progress-aware-sampling`

This folder documents sampler design before code changes.
Current highest-priority path is embedding-based cross-view sampling:

- wrist embedding close
- head embedding far
- future-action embedding/action target far (required in joint score)

Progress/direction labels are now optional add-ons, not the main path.

## Files

- `design.md`: problem framing, hypotheses, and sampling strategy options
- `embedding-sampler-spec.md`: primary implementation spec for general sampler
- `progress-label-spec.md`: how to derive task progress labels from trajectory data
- `ablation-plan.md`: experiment matrix and acceptance criteria
- `doc-plan.md`: document development plan for this feature branch

## Scope

This phase is documentation-first.
No model or dataset code changes are included yet.
