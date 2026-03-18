# Spec: Embedding-Based Cross-View Sampler

Last updated: 2026-03-18
Branch: `codex/progress-aware-sampling`

## 1. Goal

Prioritize a task-agnostic sampler that forces head-view usage without relying on task-specific direction labels.

Core rule for structured samples:

1. wrist embeddings are similar
2. head embeddings are dissimilar

## 2. Why this path is primary

1. Does not depend on absolute action coordinate semantics.
2. Generalizes across tasks better than left/right hand-crafted labels.
3. Directly targets the failure mode: wrist shortcut and weak head dependency.

## 3. Data Unit

Use sequence samples at training horizon granularity (the same unit currently consumed by dataset/sampler).

Per sample store:

- `sample_id`
- `episode_id`
- `timestep`
- `wrist_emb`
- `head_emb`

## 4. Embedding Source

Initial version:

1. Use a frozen encoder for offline embedding extraction.
2. Compute embeddings once and cache.
3. Do not use online-updating train encoder for neighbor search in v1.

Candidate source:

- fixed DINO/timm backbone features from wrist/head image streams.

## 4.1 Encoder Consistency Requirement (mandatory)

The analysis encoder used for sampling artifacts must be consistent with training config.

At minimum, lock and validate:

1. `policy.obs_encoder.model_name`
2. `policy.obs_encoder.pretrained`
3. any checkpoint/version tag used by the timm backbone

Recommended to include in cache signature:

1. workspace config name
2. task name
3. obs encoder config hash (resolved)
4. feature extraction script version

If any of these change, cached embeddings/neighbor indices must be rebuilt.

## 5. Neighbor and Selection Logic

For each anchor sample `i`:

1. Find K nearest neighbors in wrist embedding space.
2. Inside those neighbors, compute joint score with:
   - head distance
   - future-action distance
3. Select top `m` as structured counterparts.

Recommended constraints:

1. Prefer different `episode_id` to avoid near-duplicate temporal neighbors.
2. Exclude very close timesteps from same episode.
3. Keep a distance floor for wrist similarity and head dissimilarity.

Joint scoring (required):

- Let `d_head(i,j)` be normalized head embedding distance.
- Let `d_act(i,j)` be normalized future-action distance.
- Candidate score:
  - `score(i,j) = alpha * d_head(i,j) + beta * d_act(i,j)`
- Select highest-score candidates under wrist-neighbor constraint.

Default starting weights:

- `alpha = 0.5`
- `beta = 0.5`

## 6. Batch Composition

Use mixed batches:

1. `p_structured` fraction from cross-view structured sampler.
2. `1 - p_structured` fraction from regular random sampling.

Default start:

- `p_structured = 0.5`

## 7. Optional Filters

Optional in later stage:

1. Add label-based gates (progress/direction) only as secondary refinement.
2. Add adaptive weighting schedule for `alpha/beta`.

## 8. Output and Cache Format

Suggested cache artifacts:

1. `embeddings_wrist.npy`
2. `embeddings_head.npy`
3. `knn_wrist_indices.npy`
4. `structured_pairs.npy` or grouped index list
5. `meta.json` (encoder signature, config hash, build timestamp, feature version)

## 9. Diagnostics

Before training:

1. Wrist-distance histogram for selected pairs.
2. Head-distance histogram for selected pairs.
3. Random visualization grids for sampled pairs.

During training:

1. Existing head-importance metrics.
2. Direction-collapse proxy from rollout behavior.
3. Distribution of `d_head`, `d_act`, and joint score for selected pairs.

## 10. Non-Goals (v1)

1. No architecture changes.
2. No contrastive loss changes.
3. No dependency on aruco or SAM masking.
