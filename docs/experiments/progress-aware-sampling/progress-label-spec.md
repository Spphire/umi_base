# Spec: Progress Label Extraction

Last updated: 2026-03-18
Branch: `codex/progress-aware-sampling`

## 1. Purpose

Define a reproducible way to assign per-timestep progress labels for sampling.

This spec is designed to be robust and simple for first implementation.

## 2. Input Signals

Required per timestep:

- gripper width (`left_robot_gripper_width`)
- tcp pose (`left_robot_tcp_pose`)
- episode index and timestep index

Optional:

- action sequence
- right-arm channels if bimanual appears later

## 3. Derived Anchors

For each episode, compute:

1. `t_grasp_start`
2. `t_grasp_end`
3. `t_release_start`
4. `t_release_end`

Suggested first-pass logic:

- Smooth gripper width with moving average.
- Detect close event by negative slope + width below `w_close_thresh`.
- Detect release event by positive slope + width above `w_open_thresh`.

Defaults (to tune per dataset):

- `w_close_thresh = 0.035`
- `w_open_thresh = 0.055`
- slope window size: 5-9 steps

## 4. Phase Label Definition

For each timestep `t`:

- `pre_grasp`: `t < t_grasp_start`
- `grasp_transition`: `t_grasp_start <= t <= t_grasp_end`
- `transport`: `t_grasp_end < t < t_release_start`
- `release`: `t_release_start <= t <= t_release_end`

Fallback when anchors fail:

1. Use normalized time split:
   - `[0.00, 0.30)` pre_grasp
   - `[0.30, 0.45)` grasp_transition
   - `[0.45, 0.85)` transport
   - `[0.85, 1.00]` release
2. Mark episode with `label_confidence = low`.

## 5. Direction Label Definition

Direction should represent intended placement side.

First-pass rule:

- Estimate object transport direction from tcp x shift:
  - `delta_x = x_at_release - x_at_grasp`
  - `dir_label = right` if `delta_x > +x_margin`
  - `dir_label = left` if `delta_x < -x_margin`
  - `dir_label = uncertain` otherwise

Default margin:

- `x_margin = 0.02` (meters, tune per setup)

## 6. Output Schema

Suggested per-sample metadata fields:

- `episode_id`
- `timestep`
- `phase_label` in `{pre_grasp, grasp_transition, transport, release}`
- `dir_label` in `{left, right, uncertain}`
- `label_confidence` in `{high, low}`
- `t_norm` in `[0, 1]`

Storage options:

1. Sidecar numpy/json index file keyed by sample index.
2. Embedded fields in zarr metadata.

## 7. Quality Checks

Before enabling sampler logic, validate:

1. Phase distribution across dataset is reasonable.
2. Left/right distribution is not heavily broken by labeling errors.
3. Random episode visualization confirms anchors match actual manipulation stage.

## 8. Versioning

Tag this logic as `progress_spec_v1`.
Any threshold/rule change must bump version and be recorded in update logs.

