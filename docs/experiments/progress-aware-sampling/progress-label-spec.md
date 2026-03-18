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

Suggested first-pass logic (hybrid):

1. Smooth width signal (moving average or Savitzky-Golay).
2. Build trend labels (`flat/up/down`) from derivative sign.
3. Merge short windows to suppress noise.
4. Find dominant closing segment near minimum width.
5. Derive release segment from first stable open trend after transport.

Defaults (to tune per dataset):

- `close_eps = 1e-4`
- `smooth_window = 5` (candidate range 5-11)
- `min_len = 5`
- `min_drop = 0.0` (tune upward if false positives appear)
- optional `w_close_thresh = 0.035`, `w_open_thresh = 0.055` as hard guards

Reference scripts:

1. `scripts/analyze_gripper_trend.py`
2. `sign_accuracy_pick_place.py` (`find_closing_window`)

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

Note:

- Compute direction from the phase anchors (`x_at_grasp`, `x_at_release`), not from episode endpoints.
- If direction is uncertain, keep sample but exclude it from strict left/right balancing quotas.

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
4. Alternation sanity check on gripper trend windows:
   - repeated `up/up` or `down/down` segments should be rare after merge.

## 8. Versioning

Tag this logic as `progress_spec_v1`.
Any threshold/rule change must bump version and be recorded in update logs.

## 9. Integration Boundaries

1. Progress labels must not depend on aruco outputs.
2. Progress labels must not depend on SAM masking availability.
3. If a preprocessing variant is needed (for example SAM masking), keep it as optional side path and preserve the same label schema.
