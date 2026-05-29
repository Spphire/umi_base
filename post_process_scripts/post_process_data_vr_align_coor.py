#!/usr/bin/env python3
"""
Post-process dualfold head / left wrist / right wrist sessions into one coordinate frame.

Model:
    T_A_handle(t) ~= T_A_B * T_B_iphone(t) * T_iphone_handle

The target A frame is a right-handed version of the Unity/Quest head-session
world. Unity LHS poses are converted by mirroring Z; ARKit/iPhone poses are
already treated as right-handed. T_A_B rotation is yaw-only about +Y because
both systems use Y as gravity/up.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import tarfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rotation
from scipy.spatial.transform import Slerp

try:
    import bson
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: bson. Install requirements.txt first.") from exc


DEFAULT_ARCHIVE = Path("dualfold_0511.tar.gz")
UNITY_LHS_TO_RHS = np.diag([1.0, 1.0, -1.0])


@dataclass
class Record:
    path: Path
    uuid: str
    parent_uuid: str
    camera_position: str
    metadata: Dict[str, Any]
    data: Dict[str, Any]


@dataclass
class Triplet:
    parent_uuid: str
    head: Record
    left_wrist: Record
    right_wrist: Record


def bson_loads(raw: bytes) -> Dict[str, Any]:
    try:
        obj = bson.loads(raw)
    except AttributeError:
        obj = bson.decode(raw)
    if not isinstance(obj, dict):
        raise TypeError(f"Unexpected BSON root type: {type(obj)!r}")
    return obj


def extract_lightweight_files(archive: Path, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            if not (member.name.endswith("/metadata.json") or member.name.endswith("/frame_data.bson")):
                continue
            target = out_dir / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            stream = tf.extractfile(member)
            if stream is None:
                continue
            target.write_bytes(stream.read())
            count += 1
    return count


def load_records(raw_dir: Path) -> List[Record]:
    records: List[Record] = []
    for session_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        meta_path = session_dir / "metadata.json"
        bson_path = session_dir / "frame_data.bson"
        if not meta_path.exists() or not bson_path.exists():
            continue
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        data = bson_loads(bson_path.read_bytes())
        records.append(
            Record(
                path=session_dir,
                uuid=str(metadata.get("uuid", session_dir.name)),
                parent_uuid=str(metadata.get("parent_uuid", metadata.get("parentuuid", ""))),
                camera_position=str(metadata.get("camera_position", "")),
                metadata=metadata,
                data=data,
            )
        )
    return records


def build_triplets(records: Sequence[Record]) -> Tuple[List[Triplet], Dict[str, List[Record]]]:
    groups: Dict[str, List[Record]] = defaultdict(list)
    for record in records:
        groups[record.parent_uuid].append(record)

    triplets: List[Triplet] = []
    for parent_uuid, group in sorted(groups.items()):
        by_pos: Dict[str, List[Record]] = defaultdict(list)
        for record in group:
            by_pos[record.camera_position].append(record)
        if by_pos.get("head") and by_pos.get("left_wrist") and by_pos.get("right_wrist"):
            triplets.append(
                Triplet(
                    parent_uuid=parent_uuid,
                    head=by_pos["head"][0],
                    left_wrist=by_pos["left_wrist"][0],
                    right_wrist=by_pos["right_wrist"][0],
                )
            )
    return triplets, groups


def is_valid_pose(pose: Any) -> bool:
    if pose is None:
        return False
    if not isinstance(pose, (list, tuple, np.ndarray)):
        return False
    if len(pose) != 7:
        return False
    arr = np.asarray(pose, dtype=float)
    if not np.all(np.isfinite(arr)):
        return False
    return np.linalg.norm(arr[3:]) > 1e-8


def as_float_array(values: Sequence[Any]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr


def timestamp_stats(timestamps: Sequence[Any], nominal_fps: Optional[float]) -> Dict[str, Any]:
    if timestamps is None or len(timestamps) == 0:
        return {
            "timestamp_count": 0,
            "first_ts": "",
            "last_ts": "",
            "duration_s": "",
            "median_dt_s": "",
            "min_dt_s": "",
            "max_dt_s": "",
            "large_gap_count": "",
            "nonmonotonic_count": "",
        }
    ts = as_float_array(timestamps)
    finite = ts[np.isfinite(ts)]
    if len(finite) == 0:
        return {
            "timestamp_count": len(ts),
            "first_ts": "",
            "last_ts": "",
            "duration_s": "",
            "median_dt_s": "",
            "min_dt_s": "",
            "max_dt_s": "",
            "large_gap_count": "",
            "nonmonotonic_count": "",
        }
    dts = np.diff(finite)
    nominal_dt = 1.0 / nominal_fps if nominal_fps and nominal_fps > 0 else None
    median_dt = float(np.median(dts)) if len(dts) else ""
    if nominal_dt is not None:
        gap_threshold = max(2.0 * nominal_dt, 0.05)
    elif len(dts):
        gap_threshold = max(2.5 * float(np.median(dts)), 0.05)
    else:
        gap_threshold = 0.05
    large_gap_count = int(np.sum(dts > gap_threshold)) if len(dts) else 0
    nonmonotonic_count = int(np.sum(dts <= 0)) if len(dts) else 0
    return {
        "timestamp_count": int(len(ts)),
        "first_ts": float(finite[0]),
        "last_ts": float(finite[-1]),
        "duration_s": float(finite[-1] - finite[0]),
        "median_dt_s": median_dt,
        "min_dt_s": float(np.min(dts)) if len(dts) else "",
        "max_dt_s": float(np.max(dts)) if len(dts) else "",
        "large_gap_count": large_gap_count,
        "nonmonotonic_count": nonmonotonic_count,
    }


def pose_stats(poses: Sequence[Any]) -> Dict[str, Any]:
    if poses is None:
        return {"pose_count": 0, "valid_pose_count": 0, "none_pose_count": 0, "invalid_pose_count": 0}
    valid = 0
    none_count = 0
    invalid = 0
    for pose in poses:
        if pose is None:
            none_count += 1
        elif is_valid_pose(pose):
            valid += 1
        else:
            invalid += 1
    return {
        "pose_count": int(len(poses)),
        "valid_pose_count": int(valid),
        "none_pose_count": int(none_count),
        "invalid_pose_count": int(invalid),
    }


def continuity_rows(records: Sequence[Record]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in records:
        fps = record.metadata.get("fps", "")
        nominal_fps = float(fps) if isinstance(fps, (int, float)) and fps else None
        base = {
            "uuid": record.uuid,
            "parent_uuid": record.parent_uuid,
            "camera_position": record.camera_position,
            "metadata_frames": record.metadata.get("frames", ""),
            "metadata_fps": fps,
            "metadata_duration_s": record.metadata.get("duration", ""),
        }
        if record.camera_position == "head":
            streams = [
                ("leftCameraAccess", "leftCameraAccessTimestamps", "leftCameraPoses"),
                ("leftCameraUnity", "leftCameraUnityTimestamps", "leftCameraPoses"),
                ("rightCameraAccess", "rightCameraAccessTimestamps", "rightCameraPoses"),
                ("rightCameraUnity", "rightCameraUnityTimestamps", "rightCameraPoses"),
                ("leftWrist", "leftCameraAccessTimestamps", "leftWristPoses"),
                ("rightWrist", "leftCameraAccessTimestamps", "rightWristPoses"),
                ("leftWristRaw", "leftCameraAccessTimestamps", "leftWristPosesRaw"),
                ("rightWristRaw", "leftCameraAccessTimestamps", "rightWristPosesRaw"),
            ]
            for stream_name, ts_key, pose_key in streams:
                row = dict(base)
                row.update({"stream": stream_name, "timestamp_key": ts_key, "pose_key": pose_key})
                row.update(timestamp_stats(record.data.get(ts_key, []), nominal_fps))
                row.update(pose_stats(record.data.get(pose_key, [])))
                rows.append(row)
        else:
            row = dict(base)
            row.update({"stream": "arkit", "timestamp_key": "timestamps", "pose_key": "arkitPose"})
            row.update(timestamp_stats(record.data.get("timestamps", []), nominal_fps))
            row.update(pose_stats(record.data.get("arkitPose", [])))
            rows.append(row)
    return rows


def session_rows(records: Sequence[Record], triplets: Sequence[Triplet]) -> List[Dict[str, Any]]:
    triplet_parent_ids = {triplet.parent_uuid for triplet in triplets}
    rows: List[Dict[str, Any]] = []
    for record in records:
        rows.append(
            {
                "uuid": record.uuid,
                "parent_uuid": record.parent_uuid,
                "camera_position": record.camera_position,
                "start_time": record.metadata.get("start_time", ""),
                "duration_s": record.metadata.get("duration", ""),
                "frames": record.metadata.get("frames", ""),
                "fps": record.metadata.get("fps", ""),
                "device_model": (record.metadata.get("device") or {}).get("model", ""),
                "device_identifier": (record.metadata.get("device") or {}).get("identifier", ""),
                "world_map_uuid": record.metadata.get("world_map_uuid", ""),
                "in_complete_triplet": record.parent_uuid in triplet_parent_ids,
                "path": str(record.path),
            }
        )
    return rows


def field_rows(records: Sequence[Record]) -> List[Dict[str, Any]]:
    keys = [
        "timestamps",
        "arkitPose",
        "leftCameraAccessTimestamps",
        "rightCameraAccessTimestamps",
        "leftCameraUnityTimestamps",
        "rightCameraUnityTimestamps",
        "leftCameraPoses",
        "rightCameraPoses",
        "leftWristPoses",
        "rightWristPoses",
        "leftWristPosesRaw",
        "rightWristPosesRaw",
    ]
    rows: List[Dict[str, Any]] = []
    for record in records:
        for key in keys:
            value = record.data.get(key)
            row: Dict[str, Any] = {
                "uuid": record.uuid,
                "parent_uuid": record.parent_uuid,
                "camera_position": record.camera_position,
                "field": key,
                "exists": value is not None,
                "length": len(value) if isinstance(value, (list, tuple)) else "",
            }
            if key == "arkitPose" or "Poses" in key:
                row.update(pose_stats(value if isinstance(value, (list, tuple)) else []))
            if key == "timestamps" or "Timestamps" in key:
                row.update(timestamp_stats(value if isinstance(value, (list, tuple)) else [], record.metadata.get("fps")))
            rows.append(row)
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def quat_xyzw_from_pose(pose: Sequence[float], convention: str) -> np.ndarray:
    if convention == "xyzqwqxqyqz":
        quat = np.asarray([pose[4], pose[5], pose[6], pose[3]], dtype=float)
    elif convention in ("xyzqxqyqzqw", "", None):
        quat = np.asarray([pose[3], pose[4], pose[5], pose[6]], dtype=float)
    else:
        raise ValueError(f"Unsupported pose convention: {convention!r}")
    norm = np.linalg.norm(quat)
    if norm <= 1e-8:
        raise ValueError("Zero-norm quaternion")
    return quat / norm


def pose_to_matrix(
    pose: Sequence[float],
    convention: str,
    unity_lhs_to_rhs: bool = False,
) -> np.ndarray:
    arr = np.asarray(pose, dtype=float)
    quat_xyzw = quat_xyzw_from_pose(arr, convention)
    rot = Rotation.from_quat(quat_xyzw).as_matrix()
    trans = arr[:3].astype(float)
    if unity_lhs_to_rhs:
        trans = UNITY_LHS_TO_RHS @ trans
        rot = UNITY_LHS_TO_RHS @ rot @ UNITY_LHS_TO_RHS
    mat = np.eye(4)
    mat[:3, :3] = rot
    mat[:3, 3] = trans
    return mat


def poses_to_matrices(
    poses: Sequence[Any],
    convention: str,
    unity_lhs_to_rhs: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    mats: List[np.ndarray] = []
    indices: List[int] = []
    for idx, pose in enumerate(poses):
        if not is_valid_pose(pose):
            continue
        try:
            mats.append(pose_to_matrix(pose, convention, unity_lhs_to_rhs=unity_lhs_to_rhs))
            indices.append(idx)
        except Exception:
            continue
    if not mats:
        return np.empty((0, 4, 4)), np.empty((0,), dtype=int)
    return np.stack(mats, axis=0), np.asarray(indices, dtype=int)


def matrix_from_params(params: np.ndarray) -> np.ndarray:
    rotvec = params[:3]
    trans = params[3:6]
    mat = np.eye(4)
    mat[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    mat[:3, 3] = trans
    return mat


def params_from_matrix(mat: np.ndarray) -> np.ndarray:
    params = np.zeros(6)
    params[:3] = Rotation.from_matrix(mat[:3, :3]).as_rotvec()
    params[3:6] = mat[:3, 3]
    return params


def yaw_from_matrix_y(mat: np.ndarray) -> float:
    rot = mat[:3, :3]
    # Nearest rotation about +Y in Frobenius norm.
    return float(math.atan2(rot[0, 2] - rot[2, 0], rot[0, 0] + rot[2, 2]))


def y_aligned_world_transform_from_params(params: np.ndarray) -> np.ndarray:
    yaw = float(params[0])
    trans = params[1:4]
    mat = np.eye(4)
    mat[:3, :3] = Rotation.from_euler("y", yaw).as_matrix()
    mat[:3, 3] = trans
    return mat


def params_from_y_aligned_world_transform(mat: np.ndarray) -> np.ndarray:
    params = np.zeros(4)
    params[0] = yaw_from_matrix_y(mat)
    params[1:4] = mat[:3, 3]
    return params


def matrix_to_pose_qwxyz(mat: np.ndarray) -> Tuple[float, float, float, float, float, float, float]:
    quat_xyzw = Rotation.from_matrix(mat[:3, :3]).as_quat()
    return (
        float(mat[0, 3]),
        float(mat[1, 3]),
        float(mat[2, 3]),
        float(quat_xyzw[3]),
        float(quat_xyzw[0]),
        float(quat_xyzw[1]),
        float(quat_xyzw[2]),
    )


def y_axis_alignment_error(mat: np.ndarray) -> float:
    y_axis = mat[:3, :3] @ np.array([0.0, 1.0, 0.0])
    return float(np.linalg.norm(y_axis - np.array([0.0, 1.0, 0.0])))


def invert_transform(mat: np.ndarray) -> np.ndarray:
    inv = np.eye(4)
    inv[:3, :3] = mat[:3, :3].T
    inv[:3, 3] = -mat[:3, :3].T @ mat[:3, 3]
    return inv


def transform_error(pred: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    delta = invert_transform(pred) @ target
    rotvec = Rotation.from_matrix(delta[:3, :3]).as_rotvec()
    trans = target[:3, 3] - pred[:3, 3]
    return trans, rotvec


def nearest_pairs(
    query_ts: np.ndarray,
    reference_ts: np.ndarray,
    max_dt: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(query_ts) == 0 or len(reference_ts) == 0:
        return np.empty((0,), dtype=int), np.empty((0,), dtype=int), np.empty((0,), dtype=float)
    insert = np.searchsorted(reference_ts, query_ts, side="left")
    lo = np.clip(insert - 1, 0, len(reference_ts) - 1)
    hi = np.clip(insert, 0, len(reference_ts) - 1)
    lo_dt = np.abs(query_ts - reference_ts[lo])
    hi_dt = np.abs(query_ts - reference_ts[hi])
    ref_idx = np.where(lo_dt <= hi_dt, lo, hi)
    dt = np.abs(query_ts - reference_ts[ref_idx])
    keep = dt <= max_dt
    return np.nonzero(keep)[0], ref_idx[keep], dt[keep]


def interpolate_matrices_at(
    mats: np.ndarray,
    ts: np.ndarray,
    query_ts: np.ndarray,
    max_gap_s: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(mats) < 2 or len(ts) < 2 or len(query_ts) == 0:
        return np.empty((0, 4, 4)), np.empty((0,), dtype=int), np.empty((0,))
    order = np.argsort(ts)
    ts_sorted = np.asarray(ts[order], dtype=float)
    mats_sorted = mats[order]
    keep_unique = np.concatenate([[True], np.diff(ts_sorted) > 1e-9])
    ts_sorted = ts_sorted[keep_unique]
    mats_sorted = mats_sorted[keep_unique]
    if len(ts_sorted) < 2:
        return np.empty((0, 4, 4)), np.empty((0,), dtype=int), np.empty((0,))

    insert = np.searchsorted(ts_sorted, query_ts, side="left")
    lo = insert - 1
    hi = insert
    valid = (lo >= 0) & (hi < len(ts_sorted))
    valid_indices = np.nonzero(valid)[0]
    if len(valid_indices) == 0:
        return np.empty((0, 4, 4)), np.empty((0,), dtype=int), np.empty((0,))

    lo = lo[valid_indices]
    hi = hi[valid_indices]
    bracket_gap = ts_sorted[hi] - ts_sorted[lo]
    nearest_gap = np.minimum(np.abs(query_ts[valid_indices] - ts_sorted[lo]), np.abs(ts_sorted[hi] - query_ts[valid_indices]))
    valid_gap = (bracket_gap <= max_gap_s) & (nearest_gap <= max_gap_s)
    valid_indices = valid_indices[valid_gap]
    lo = lo[valid_gap]
    hi = hi[valid_gap]
    if len(valid_indices) == 0:
        return np.empty((0, 4, 4)), np.empty((0,), dtype=int), np.empty((0,))

    alpha = (query_ts[valid_indices] - ts_sorted[lo]) / (ts_sorted[hi] - ts_sorted[lo])
    out = np.repeat(np.eye(4)[None, :, :], len(valid_indices), axis=0)
    out[:, :3, 3] = (1.0 - alpha[:, None]) * mats_sorted[lo, :3, 3] + alpha[:, None] * mats_sorted[hi, :3, 3]
    rots = Rotation.from_matrix(mats_sorted[:, :3, :3])
    slerp = Slerp(ts_sorted, rots)
    out[:, :3, :3] = slerp(query_ts[valid_indices]).as_matrix()
    return out, valid_indices.astype(int), nearest_gap[valid_gap]


def smooth_isolated_pose_spikes(
    mats: np.ndarray,
    rotation_threshold_deg: float,
    translation_threshold_m: float,
    max_neighbor_rotation_deg: float,
    max_neighbor_translation_m: float,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    if len(mats) < 3:
        return mats, []
    smoothed = mats.copy()
    replacements: List[Tuple[int, np.ndarray, np.ndarray, float, float]] = []
    for idx in range(1, len(mats) - 1):
        prev_mat = mats[idx - 1]
        cur_mat = mats[idx]
        next_mat = mats[idx + 1]
        neighbor_rot_deg = float(
            np.degrees(np.linalg.norm(Rotation.from_matrix(prev_mat[:3, :3].T @ next_mat[:3, :3]).as_rotvec()))
        )
        neighbor_trans_m = float(np.linalg.norm(next_mat[:3, 3] - prev_mat[:3, 3]))
        if neighbor_rot_deg > max_neighbor_rotation_deg or neighbor_trans_m > max_neighbor_translation_m:
            continue
        mid_rot = Slerp(
            [0.0, 1.0],
            Rotation.from_matrix(np.stack([prev_mat[:3, :3], next_mat[:3, :3]], axis=0)),
        )([0.5]).as_matrix()[0]
        mid_trans = 0.5 * (prev_mat[:3, 3] + next_mat[:3, 3])
        rot_dev_deg = float(np.degrees(np.linalg.norm(Rotation.from_matrix(mid_rot.T @ cur_mat[:3, :3]).as_rotvec())))
        trans_dev_m = float(np.linalg.norm(cur_mat[:3, 3] - mid_trans))
        if rot_dev_deg > rotation_threshold_deg or trans_dev_m > translation_threshold_m:
            replacements.append((idx, mid_rot, mid_trans, rot_dev_deg, trans_dev_m))

    applied: List[Dict[str, Any]] = []
    for idx, mid_rot, mid_trans, rot_dev_deg, trans_dev_m in replacements:
        smoothed[idx, :3, :3] = mid_rot
        smoothed[idx, :3, 3] = mid_trans
        applied.append(
            {
                "index": int(idx),
                "rotation_deviation_deg": rot_dev_deg,
                "translation_deviation_m": trans_dev_m,
            }
        )
    return smoothed, applied


def smooth_isolated_residual_outliers(
    phone_mats: np.ndarray,
    wrist_mats: np.ndarray,
    calibration: Dict[str, Any],
    rotation_threshold_deg: float,
    translation_threshold_m: float,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    if len(wrist_mats) < 3 or calibration.get("status") != "ok":
        return wrist_mats, []
    x_mat = np.asarray(calibration["ar_world_to_head_rh"], dtype=float)
    y_mat = np.asarray(calibration["phone_to_wrist"], dtype=float)
    trans_errors: List[float] = []
    rot_errors: List[float] = []
    for src, dst in zip(phone_mats, wrist_mats):
        pred = x_mat @ src @ y_mat
        terr, rerr = transform_error(pred, dst)
        trans_errors.append(float(np.linalg.norm(terr)))
        rot_errors.append(float(np.degrees(np.linalg.norm(rerr))))

    smoothed = wrist_mats.copy()
    applied: List[Dict[str, Any]] = []
    for idx in range(1, len(wrist_mats) - 1):
        is_outlier = trans_errors[idx] > translation_threshold_m or rot_errors[idx] > rotation_threshold_deg
        prev_ok = trans_errors[idx - 1] <= translation_threshold_m and rot_errors[idx - 1] <= rotation_threshold_deg
        next_ok = trans_errors[idx + 1] <= translation_threshold_m and rot_errors[idx + 1] <= rotation_threshold_deg
        if not (is_outlier and prev_ok and next_ok):
            continue
        mid_rot = Slerp(
            [0.0, 1.0],
            Rotation.from_matrix(np.stack([wrist_mats[idx - 1, :3, :3], wrist_mats[idx + 1, :3, :3]], axis=0)),
        )([0.5]).as_matrix()[0]
        mid_trans = 0.5 * (wrist_mats[idx - 1, :3, 3] + wrist_mats[idx + 1, :3, 3])
        smoothed[idx, :3, :3] = mid_rot
        smoothed[idx, :3, 3] = mid_trans
        applied.append(
            {
                "sample_idx": int(idx),
                "translation_error_before_m": trans_errors[idx],
                "rotation_error_before_deg": rot_errors[idx],
            }
        )
    return smoothed, applied


def angular_speed_profile(mats: np.ndarray, ts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = min(len(mats), len(ts))
    if n < 3:
        return np.empty((0,)), np.empty((0,))
    times: List[float] = []
    speeds: List[float] = []
    for idx in range(n - 1):
        dt = float(ts[idx + 1] - ts[idx])
        if dt <= 1e-6 or dt > 0.2:
            continue
        delta_rot = mats[idx, :3, :3].T @ mats[idx + 1, :3, :3]
        angle = np.linalg.norm(Rotation.from_matrix(delta_rot).as_rotvec())
        times.append(float(0.5 * (ts[idx] + ts[idx + 1])))
        speeds.append(float(angle / dt))
    if not times:
        return np.empty((0,)), np.empty((0,))
    return np.asarray(times), np.asarray(speeds)


def angular_speed_offset_score(
    phone_t: np.ndarray,
    phone_v: np.ndarray,
    wrist_t: np.ndarray,
    wrist_v: np.ndarray,
    offset_s: float,
) -> Tuple[float, int, float]:
    shifted_phone_t0 = phone_t[0] + offset_s
    shifted_phone_t1 = phone_t[-1] + offset_s
    start = max(float(wrist_t[0]), float(shifted_phone_t0))
    end = min(float(wrist_t[-1]), float(shifted_phone_t1))
    overlap = end - start
    if overlap < 3.0:
        return -1.0, 0, overlap
    grid = np.arange(start, end, 1.0 / 60.0)
    if len(grid) < 30:
        return -1.0, int(len(grid)), overlap
    phone_interp = np.interp(grid - offset_s, phone_t, phone_v)
    wrist_interp = np.interp(grid, wrist_t, wrist_v)
    phone_centered = phone_interp - np.mean(phone_interp)
    wrist_centered = wrist_interp - np.mean(wrist_interp)
    denom = np.linalg.norm(phone_centered) * np.linalg.norm(wrist_centered)
    if denom <= 1e-9:
        return -1.0, int(len(grid)), overlap
    return float(np.dot(phone_centered, wrist_centered) / denom), int(len(grid)), overlap


def estimate_phone_time_offset(
    phone_mats: np.ndarray,
    phone_ts: np.ndarray,
    wrist_mats: np.ndarray,
    wrist_ts: np.ndarray,
    search_s: float,
) -> Dict[str, Any]:
    if search_s <= 0:
        return {
            "method": "disabled",
            "phone_time_offset_to_head_s": 0.0,
            "score": "",
            "score_at_zero": "",
            "overlap_s": "",
            "sample_count": "",
        }
    phone_t, phone_v = angular_speed_profile(phone_mats, phone_ts)
    wrist_t, wrist_v = angular_speed_profile(wrist_mats, wrist_ts)
    if len(phone_t) < 30 or len(wrist_t) < 30:
        return {
            "method": "angular_speed_correlation_failed",
            "phone_time_offset_to_head_s": 0.0,
            "score": "",
            "score_at_zero": "",
            "overlap_s": "",
            "sample_count": "",
        }

    coarse_step = 0.05
    coarse_offsets = np.arange(-search_s, search_s + 0.5 * coarse_step, coarse_step)
    coarse_scores = [angular_speed_offset_score(phone_t, phone_v, wrist_t, wrist_v, float(offset))[0] for offset in coarse_offsets]
    best_coarse = float(coarse_offsets[int(np.argmax(coarse_scores))])

    fine_step = 0.005
    fine_offsets = np.arange(best_coarse - coarse_step, best_coarse + coarse_step + 0.5 * fine_step, fine_step)
    fine_offsets = fine_offsets[(fine_offsets >= -search_s) & (fine_offsets <= search_s)]
    fine_scores = [angular_speed_offset_score(phone_t, phone_v, wrist_t, wrist_v, float(offset))[0] for offset in fine_offsets]
    best_idx = int(np.argmax(fine_scores))
    best_offset = float(fine_offsets[best_idx])
    best_score, sample_count, overlap_s = angular_speed_offset_score(phone_t, phone_v, wrist_t, wrist_v, best_offset)
    zero_score, _, zero_overlap_s = angular_speed_offset_score(phone_t, phone_v, wrist_t, wrist_v, 0.0)
    return {
        "method": "angular_speed_correlation",
        "phone_time_offset_to_head_s": best_offset,
        "score": best_score,
        "score_at_zero": zero_score,
        "overlap_s": overlap_s,
        "overlap_at_zero_s": zero_overlap_s,
        "sample_count": sample_count,
        "search_s": search_s,
    }


def subsample_indices(n: int, max_samples: int) -> np.ndarray:
    if n <= max_samples:
        return np.arange(n, dtype=int)
    return np.unique(np.linspace(0, n - 1, max_samples).round().astype(int))


def relative_motion_pairs(
    phone: np.ndarray,
    wrist: np.ndarray,
    max_pairs: int = 1200,
) -> Tuple[np.ndarray, np.ndarray]:
    n = min(len(phone), len(wrist))
    if n < 8:
        return np.empty((0, 4, 4)), np.empty((0, 4, 4))
    gaps = [1, 2, 4, 8, 16, 32]
    phone_rel: List[np.ndarray] = []
    wrist_rel: List[np.ndarray] = []
    for gap in gaps:
        if n <= gap:
            continue
        pair_count = n - gap
        per_gap_limit = max(1, max_pairs // len(gaps))
        starts = subsample_indices(pair_count, per_gap_limit)
        for start in starts:
            end = int(start + gap)
            phone_rel.append(invert_transform(phone[int(start)]) @ phone[end])
            wrist_rel.append(invert_transform(wrist[int(start)]) @ wrist[end])
    if not phone_rel:
        return np.empty((0, 4, 4)), np.empty((0, 4, 4))
    return np.stack(phone_rel, axis=0), np.stack(wrist_rel, axis=0)


def estimate_mount_from_relative_motions(
    phone: np.ndarray,
    wrist: np.ndarray,
    max_pairs: int = 1200,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    phone_rel, wrist_rel = relative_motion_pairs(phone, wrist, max_pairs=max_pairs)
    if len(phone_rel) < 8:
        return np.eye(4), {
            "relative_motion_status": "insufficient_pairs",
            "relative_motion_pairs": int(len(phone_rel)),
        }

    # From A_handle = A_from_B * B_iphone * Y:
    # inv(A_i) A_j = inv(Y) * inv(B_i) B_j * Y, so B_rel * Y ~= Y * A_rel.
    def residual(params: np.ndarray) -> np.ndarray:
        y_mat = matrix_from_params(params)
        res: List[np.ndarray] = []
        for b_rel, a_rel in zip(phone_rel, wrist_rel):
            pred = b_rel @ y_mat
            target = y_mat @ a_rel
            terr, rerr = transform_error(pred, target)
            res.append(terr)
            res.append(rerr)
        return np.concatenate(res)

    result = least_squares(
        residual,
        np.zeros(6),
        loss="soft_l1",
        f_scale=0.03,
        max_nfev=300,
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
    )
    y_mat = matrix_from_params(result.x)
    res = residual(result.x).reshape(-1, 3)
    return y_mat, {
        "relative_motion_status": "ok",
        "relative_motion_pairs": int(len(phone_rel)),
        "relative_motion_cost": float(result.cost),
        "relative_motion_success": bool(result.success),
        "relative_motion_message": result.message,
        "relative_motion_residual_rms": float(np.sqrt(np.mean(res**2))) if len(res) else "",
    }


def initial_yaw_world_transform(phone: np.ndarray, wrist: np.ndarray, phone_to_wrist: np.ndarray) -> np.ndarray:
    x_candidates = []
    for src, dst in zip(phone, wrist):
        x_candidates.append(dst @ invert_transform(phone_to_wrist) @ invert_transform(src))
    if not x_candidates:
        return np.eye(4)
    yaw_values = np.asarray([yaw_from_matrix_y(mat) for mat in x_candidates])
    yaw = float(math.atan2(np.mean(np.sin(yaw_values)), np.mean(np.cos(yaw_values))))
    translations = np.stack([mat[:3, 3] for mat in x_candidates], axis=0)
    init = np.eye(4)
    init[:3, :3] = Rotation.from_euler("y", yaw).as_matrix()
    init[:3, 3] = np.median(translations, axis=0)
    return init


def estimate_world_and_mount(
    phone_in_ar: np.ndarray,
    wrist_in_head: np.ndarray,
    max_samples: int = 500,
) -> Dict[str, Any]:
    if len(phone_in_ar) != len(wrist_in_head):
        raise ValueError("phone_in_ar and wrist_in_head must have equal length")
    if len(phone_in_ar) < 8:
        raise ValueError("Need at least 8 matched pose samples")

    sample_idx = subsample_indices(len(phone_in_ar), max_samples)
    phone = phone_in_ar[sample_idx]
    wrist = wrist_in_head[sample_idx]

    init_y, relative_report = estimate_mount_from_relative_motions(phone, wrist)
    init_x = initial_yaw_world_transform(phone, wrist, init_y)

    # T_head_wrist ~= X(head_from_ar) * T_ar_phone * Y(phone_to_wrist).
    # Head Unity poses are converted to RHS before this point. Since both worlds
    # use Y for gravity/up, constrain X rotation to yaw-only about Y.
    x0 = np.concatenate([params_from_y_aligned_world_transform(init_x), params_from_matrix(init_y)])

    def residual(params: np.ndarray) -> np.ndarray:
        x_mat = y_aligned_world_transform_from_params(params[:4])
        y_mat = matrix_from_params(params[4:10])
        res: List[np.ndarray] = []
        for src, dst in zip(phone, wrist):
            pred = x_mat @ src @ y_mat
            terr, rerr = transform_error(pred, dst)
            res.append(terr)
            res.append(rerr)
        return np.concatenate(res)

    result = least_squares(
        residual,
        x0,
        loss="soft_l1",
        f_scale=0.03,
        max_nfev=300,
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
    )
    x_mat = y_aligned_world_transform_from_params(result.x[:4])
    y_mat = matrix_from_params(result.x[4:10])

    trans_errors = []
    rot_errors = []
    for src, dst in zip(phone_in_ar, wrist_in_head):
        pred = x_mat @ src @ y_mat
        terr, rerr = transform_error(pred, dst)
        trans_errors.append(np.linalg.norm(terr))
        rot_errors.append(np.linalg.norm(rerr))

    trans_arr = np.asarray(trans_errors)
    rot_arr = np.asarray(rot_errors)
    return {
        "status": "ok",
        "matched_samples": int(len(phone_in_ar)),
        "optimized_samples": int(len(phone)),
        "cost": float(result.cost),
        "success": bool(result.success),
        "message": result.message,
        "constraint": "head_from_ar rotation is yaw-only about +Y after Unity LHS poses are converted to RHS",
        "parameter_count": 10,
        "initialization": "relative hand-eye solve for phone_to_wrist, then yaw-constrained world solve, then joint refinement",
        **relative_report,
        "ar_world_to_head_rh": x_mat.tolist(),
        "ar_world_to_head_yaw_rad": float(result.x[0]),
        "ar_world_to_head_yaw_deg": float(math.degrees(result.x[0])),
        "ar_y_axis_in_head_rh": (x_mat[:3, :3] @ np.array([0.0, 1.0, 0.0])).tolist(),
        "ar_to_head_y_axis_alignment_error": y_axis_alignment_error(x_mat),
        "phone_to_wrist": y_mat.tolist(),
        "translation_rmse_m": float(np.sqrt(np.mean(trans_arr**2))),
        "translation_median_m": float(np.median(trans_arr)),
        "translation_p95_m": float(np.percentile(trans_arr, 95)),
        "rotation_rmse_deg": float(np.degrees(np.sqrt(np.mean(rot_arr**2)))),
        "rotation_median_deg": float(np.degrees(np.median(rot_arr))),
        "rotation_p95_deg": float(np.degrees(np.percentile(rot_arr, 95))),
    }


def choose_wrist_stream(
    head: Record,
    side: str,
    prefer_raw: bool,
    min_valid: int,
) -> Tuple[Optional[str], Dict[str, int]]:
    raw_key = f"{side}WristPosesRaw"
    processed_key = f"{side}WristPoses"
    candidates = [raw_key, processed_key] if prefer_raw else [processed_key, raw_key]
    counts = {key: int(pose_stats(head.data.get(key, [])).get("valid_pose_count", 0)) for key in [processed_key, raw_key]}
    for key in candidates:
        if counts.get(key, 0) >= min_valid:
            return key, counts
    return None, counts


def head_wrist_timestamps(head: Record, wrist_key: str) -> Tuple[np.ndarray, str]:
    n = len(head.data.get(wrist_key, []))
    for key in ("leftCameraAccessTimestamps", "rightCameraAccessTimestamps", "leftCameraUnityTimestamps", "rightCameraUnityTimestamps"):
        ts = head.data.get(key, [])
        if len(ts) == n:
            return as_float_array(ts), key
    return as_float_array(head.data.get("leftCameraAccessTimestamps", []))[:n], "leftCameraAccessTimestamps[:n]"


def calibrate_side(
    triplet: Triplet,
    side: str,
    phone_record: Record,
    prefer_raw: bool,
    min_valid: int,
    max_match_dt: float,
    time_offset_search_s: float,
    smooth_isolated_spikes: bool,
    smooth_rot_spike_deg: float,
    smooth_trans_spike_mm: float,
) -> Dict[str, Any]:
    wrist_key, counts = choose_wrist_stream(triplet.head, side, prefer_raw, min_valid)
    side_report: Dict[str, Any] = {
        "side": side,
        "phone_uuid": phone_record.uuid,
        "wrist_valid_counts": counts,
        "used_wrist_key": wrist_key or "",
    }
    if wrist_key is None:
        side_report.update(
            {
                "status": "missing_head_wrist_poses",
                "reason": "No usable Quest/Unity wrist poses in raw or non-raw wrist streams.",
            }
        )
        return side_report

    head_pose_convention = str(triplet.head.data.get("poseConvention", "xyzqwqxqyqz"))
    wrist_mats, wrist_valid_indices = poses_to_matrices(
        triplet.head.data.get(wrist_key, []),
        head_pose_convention,
        unity_lhs_to_rhs=True,
    )
    wrist_ts_all, wrist_timestamp_key = head_wrist_timestamps(triplet.head, wrist_key)
    side_report["wrist_timestamp_key"] = wrist_timestamp_key
    if len(wrist_ts_all) <= int(np.max(wrist_valid_indices, initial=-1)):
        side_report.update({"status": "bad_head_wrist_timestamps", "reason": "Wrist timestamps do not cover valid wrist poses."})
        return side_report
    wrist_ts = wrist_ts_all[wrist_valid_indices]
    side_report["smooth_isolated_spikes"] = bool(smooth_isolated_spikes)

    phone_mats, phone_valid_indices = poses_to_matrices(
        phone_record.data.get("arkitPose", []),
        "xyzqxqyqzqw",
        unity_lhs_to_rhs=False,
    )
    phone_ts_all = as_float_array(phone_record.data.get("timestamps", []))
    if len(phone_ts_all) <= int(np.max(phone_valid_indices, initial=-1)):
        side_report.update({"status": "bad_phone_timestamps", "reason": "Phone timestamps do not cover valid phone poses."})
        return side_report
    phone_ts = phone_ts_all[phone_valid_indices]

    offset_report = estimate_phone_time_offset(phone_mats, phone_ts, wrist_mats, wrist_ts, time_offset_search_s)
    side_report.update(offset_report)
    offset_score = offset_report.get("score")
    offset_zero_score = offset_report.get("score_at_zero")
    offset_overlap = offset_report.get("overlap_s")
    quality_reasons = []
    if isinstance(offset_score, (int, float)) and offset_score < 0.3:
        quality_reasons.append("low_score")
    if isinstance(offset_score, (int, float)) and isinstance(offset_zero_score, (int, float)) and (offset_score - offset_zero_score) < 0.1:
        quality_reasons.append("weak_score_margin")
    if isinstance(offset_overlap, (int, float)) and offset_overlap < 5.0:
        quality_reasons.append("short_overlap")
    side_report["time_offset_quality"] = "low" if quality_reasons else "ok"
    side_report["time_offset_quality_reason"] = ";".join(quality_reasons)
    phone_time_offset = float(offset_report.get("phone_time_offset_to_head_s") or 0.0)
    phone_interp, wrist_match_idx, interp_gap = interpolate_matrices_at(
        phone_mats,
        phone_ts + phone_time_offset,
        wrist_ts,
        max_gap_s=max_match_dt,
    )
    side_report["matched_samples_before_validity"] = int(len(wrist_match_idx))
    side_report["max_match_dt_s"] = max_match_dt
    side_report["match_dt_median_s"] = float(np.median(interp_gap)) if len(interp_gap) else ""
    side_report["match_dt_max_s"] = float(np.max(interp_gap)) if len(interp_gap) else ""
    if len(wrist_match_idx) < min_valid:
        side_report.update(
            {
                "status": "insufficient_timestamp_matches",
                "reason": f"Only {len(wrist_match_idx)} interpolated matches within {max_match_dt:.3f}s.",
            }
        )
        return side_report

    phone_matched = phone_interp
    wrist_matched = wrist_mats[wrist_match_idx]
    try:
        result = estimate_world_and_mount(phone_matched, wrist_matched)
    except Exception as exc:
        side_report.update({"status": "solve_failed", "reason": str(exc)})
        return side_report
    residual_smoothed_frames: List[Dict[str, Any]] = []
    if smooth_isolated_spikes:
        wrist_smoothed, residual_smoothed_frames = smooth_isolated_residual_outliers(
            phone_matched,
            wrist_matched,
            result,
            rotation_threshold_deg=smooth_rot_spike_deg,
            translation_threshold_m=smooth_trans_spike_mm / 1000.0,
        )
        if residual_smoothed_frames:
            wrist_matched = wrist_smoothed
            try:
                result = estimate_world_and_mount(phone_matched, wrist_matched)
            except Exception as exc:
                side_report.update({"status": "solve_failed_after_smoothing", "reason": str(exc)})
                return side_report
    side_report.update(result)
    side_report["residual_smoothed_frame_count"] = len(residual_smoothed_frames)
    side_report["residual_smoothed_frames"] = residual_smoothed_frames
    x_mat = np.asarray(side_report["ar_world_to_head_rh"], dtype=float)
    y_mat = np.asarray(side_report["phone_to_wrist"], dtype=float)
    residual_smoothed_indices = {int(item["sample_idx"]) for item in residual_smoothed_frames}
    sample_errors: List[Dict[str, Any]] = []
    for local_idx, (src, dst, wrist_idx, gap_s) in enumerate(zip(phone_matched, wrist_matched, wrist_match_idx, interp_gap)):
        pred = x_mat @ src @ y_mat
        aligned_phone = x_mat @ src
        head_via_mount_phone = dst @ invert_transform(y_mat)
        terr, rerr = transform_error(pred, dst)
        aligned_x, aligned_y, aligned_z, aligned_qw, aligned_qx, aligned_qy, aligned_qz = matrix_to_pose_qwxyz(aligned_phone)
        head_via_mount_x, head_via_mount_y, head_via_mount_z, head_via_mount_qw, head_via_mount_qx, head_via_mount_qy, head_via_mount_qz = matrix_to_pose_qwxyz(head_via_mount_phone)
        sample_errors.append(
            {
                "sample_idx": local_idx,
                "wrist_index": int(wrist_idx),
                "timestamp_head_s": float(wrist_ts[wrist_idx]),
                "timestamp_phone_raw_s": float(wrist_ts[wrist_idx] - phone_time_offset),
                "timestamp_phone_aligned_s": float(wrist_ts[wrist_idx]),
                "interp_gap_s": float(gap_s),
                "final_trajectory_source": "phone_session_iphone_camera",
                "aligned_x": aligned_x,
                "aligned_y": aligned_y,
                "aligned_z": aligned_z,
                "aligned_qw": aligned_qw,
                "aligned_qx": aligned_qx,
                "aligned_qy": aligned_qy,
                "aligned_qz": aligned_qz,
                "comparison_trajectory_source": "head_session_iphone_camera_via_mount",
                "head_via_mount_x": head_via_mount_x,
                "head_via_mount_y": head_via_mount_y,
                "head_via_mount_z": head_via_mount_z,
                "head_via_mount_qw": head_via_mount_qw,
                "head_via_mount_qx": head_via_mount_qx,
                "head_via_mount_qy": head_via_mount_qy,
                "head_via_mount_qz": head_via_mount_qz,
                "translation_error_m": float(np.linalg.norm(terr)),
                "rotation_error_deg": float(np.degrees(np.linalg.norm(rerr))),
                "smoothed_pose": local_idx in residual_smoothed_indices,
            }
        )
    side_report["sample_errors"] = sample_errors
    return side_report


def calibration_report(
    triplets: Sequence[Triplet],
    prefer_raw: bool,
    min_valid: int,
    max_match_dt: float,
    time_offset_search_s: float,
    smooth_isolated_spikes: bool,
    smooth_rot_spike_deg: float,
    smooth_trans_spike_mm: float,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "target_frame": "right-handed Quest/Unity head world",
        "coordinate_convention": "Unity/Quest head poses are converted from LHS to RHS by mirroring Z; iPhone ARKit poses are kept RHS.",
        "method": "least_squares with hard gravity constraint: T_head_wrist ~= T_head_from_ar * T_ar_phone * T_phone_to_wrist; R(T_head_from_ar) is yaw-only about +Y",
        "final_aligned_hand_trajectory": "aligned_trajectories.csv contains phone-session iPhone camera poses transformed by T_head_from_ar * T_ar_phone; matched rows also include head-session controller poses converted to iPhone camera by T_head_wrist * inverse(T_phone_to_wrist).",
        "time_matching": "phone timestamps are shifted by an automatically estimated angular-speed offset, then phone poses are interpolated at head wrist timestamps",
        "min_valid": min_valid,
        "max_match_dt_s": max_match_dt,
        "time_offset_search_s": time_offset_search_s,
        "smoothing": {
            "enabled": smooth_isolated_spikes,
            "type": "isolated_residual_spike_midpoint_slerp",
            "rotation_threshold_deg": smooth_rot_spike_deg,
            "translation_threshold_mm": smooth_trans_spike_mm,
        },
        "triplets": [],
    }
    for triplet in triplets:
        left = calibrate_side(
            triplet,
            "left",
            triplet.left_wrist,
            prefer_raw,
            min_valid,
            max_match_dt,
            time_offset_search_s,
            smooth_isolated_spikes,
            smooth_rot_spike_deg,
            smooth_trans_spike_mm,
        )
        right = calibrate_side(
            triplet,
            "right",
            triplet.right_wrist,
            prefer_raw,
            min_valid,
            max_match_dt,
            time_offset_search_s,
            smooth_isolated_spikes,
            smooth_rot_spike_deg,
            smooth_trans_spike_mm,
        )
        report["triplets"].append(
            {
                "parent_uuid": triplet.parent_uuid,
                "head_uuid": triplet.head.uuid,
                "left_wrist_uuid": triplet.left_wrist.uuid,
                "right_wrist_uuid": triplet.right_wrist.uuid,
                "left": left,
                "right": right,
            }
        )
    return report


def head_pose_matrices(head: Record) -> np.ndarray:
    convention = str(head.data.get("poseConvention", "xyzqwqxqyqz"))
    left_mats, _ = poses_to_matrices(head.data.get("leftCameraPoses", []), convention, unity_lhs_to_rhs=True)
    right_mats, _ = poses_to_matrices(head.data.get("rightCameraPoses", []), convention, unity_lhs_to_rhs=True)
    if len(left_mats) and len(right_mats):
        n = min(len(left_mats), len(right_mats))
        mats = left_mats[:n].copy()
        mats[:, :3, 3] = 0.5 * (left_mats[:n, :3, 3] + right_mats[:n, :3, 3])
        return mats
    if len(left_mats):
        return left_mats
    if len(right_mats):
        return right_mats
    return np.empty((0, 4, 4))


def side_phone_camera_matrices(
    phone: Record,
    ar_world_to_head: Optional[np.ndarray],
) -> np.ndarray:
    mats, _ = poses_to_matrices(phone.data.get("arkitPose", []), "xyzqxqyqzqw", unity_lhs_to_rhs=False)
    if not len(mats):
        return np.empty((0, 4, 4))
    if ar_world_to_head is not None:
        mats = np.einsum("ij,njk->nik", ar_world_to_head, mats)
    return mats


def head_session_phone_camera_matrices(
    wrist_mats: np.ndarray,
    phone_to_wrist: Optional[np.ndarray],
) -> np.ndarray:
    if len(wrist_mats) == 0:
        return np.empty((0, 4, 4))
    if phone_to_wrist is None:
        return wrist_mats.copy()
    wrist_to_phone = invert_transform(phone_to_wrist)
    return np.einsum("nij,jk->nik", wrist_mats, wrist_to_phone)


def head_wrist_pose_matrices(head: Record, side: str, preferred_key: str = "") -> Tuple[np.ndarray, str]:
    convention = str(head.data.get("poseConvention", "xyzqwqxqyqz"))
    raw_key = f"{side}WristPosesRaw"
    processed_key = f"{side}WristPoses"
    candidate_keys = [key for key in [preferred_key, raw_key, processed_key] if key]
    for key in dict.fromkeys(candidate_keys):
        mats, _ = poses_to_matrices(head.data.get(key, []), convention, unity_lhs_to_rhs=True)
        if len(mats):
            return mats, key
    return np.empty((0, 4, 4)), ""


def center_local_matrices(mats: np.ndarray) -> np.ndarray:
    if len(mats) == 0:
        return mats
    centered = mats.copy()
    centered[:, :3, 3] -= mats[0:1, :3, 3]
    return centered


def downsample_matrices(mats: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or len(mats) <= max_points:
        return mats
    idx = np.unique(np.linspace(0, len(mats) - 1, max_points).round().astype(int))
    return mats[idx]


def axis_sample_indices(count: int, axis_stride: int) -> np.ndarray:
    if count <= 0:
        return np.empty((0,), dtype=int)
    stride = max(1, axis_stride)
    idx = np.arange(0, count, stride, dtype=int)
    if idx[-1] != count - 1:
        idx = np.append(idx, count - 1)
    return idx


def add_pose_axis_traces(
    traces: List[Any],
    go: Any,
    name: str,
    mats: np.ndarray,
    axis_length: float,
    axis_stride: int,
) -> None:
    if len(mats) == 0 or axis_length <= 0:
        return
    indices = axis_sample_indices(len(mats), axis_stride)
    axis_defs = [
        ("X", "#E63946", 0),
        ("Y", "#2A9D8F", 1),
        ("Z", "#457B9D", 2),
    ]
    for axis_name, color, axis_idx in axis_defs:
        xs: List[Optional[float]] = []
        ys: List[Optional[float]] = []
        zs: List[Optional[float]] = []
        for mat in mats[indices]:
            origin = mat[:3, 3]
            end = origin + axis_length * mat[:3, axis_idx]
            xs.extend([float(origin[0]), float(end[0]), None])
            ys.extend([float(origin[1]), float(end[1]), None])
            zs.extend([float(origin[2]), float(end[2]), None])
        traces.append(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                name=f"{name}_local_{axis_name}",
                line=dict(color=color, width=3),
                opacity=0.85,
            )
        )


def extract_side_info(report: Dict[str, Any], parent_uuid: str, side: str) -> Dict[str, Any]:
    for item in report.get("triplets", []):
        if item.get("parent_uuid") == parent_uuid:
            side_info = item.get(side, {})
            return side_info if isinstance(side_info, dict) else {}
    return {}


def extract_side_calibration(
    report: Dict[str, Any],
    parent_uuid: str,
    side: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    side_info = extract_side_info(report, parent_uuid, side)
    if (
        side_info.get("status") == "ok"
        and side_info.get("ar_world_to_head_rh")
        and side_info.get("phone_to_wrist")
    ):
        return (
            np.asarray(side_info["ar_world_to_head_rh"], dtype=float),
            np.asarray(side_info["phone_to_wrist"], dtype=float),
        )
    return None, None


def make_trajectory_html(
    triplet: Triplet,
    calib_report: Dict[str, Any],
    out_path: Path,
    max_points: int,
    axis_stride: int,
    axis_length: float,
) -> None:
    import plotly.graph_objects as go

    left_info = extract_side_info(calib_report, triplet.parent_uuid, "left")
    right_info = extract_side_info(calib_report, triplet.parent_uuid, "right")
    left_x, left_y = extract_side_calibration(calib_report, triplet.parent_uuid, "left")
    right_x, right_y = extract_side_calibration(calib_report, triplet.parent_uuid, "right")
    left_calibrated = left_x is not None and left_y is not None
    right_calibrated = right_x is not None and right_y is not None
    calibrated = left_calibrated and right_calibrated

    head_mats = head_pose_matrices(triplet.head)
    left_head_mats, _ = head_wrist_pose_matrices(triplet.head, "left", str(left_info.get("used_wrist_key", "")))
    right_head_mats, _ = head_wrist_pose_matrices(triplet.head, "right", str(right_info.get("used_wrist_key", "")))
    left_head_phone_mats = head_session_phone_camera_matrices(left_head_mats, left_y if left_calibrated else None)
    right_head_phone_mats = head_session_phone_camera_matrices(right_head_mats, right_y if right_calibrated else None)
    if left_calibrated:
        left_phone_mats = side_phone_camera_matrices(triplet.left_wrist, left_x)
    else:
        left_phone_mats = side_phone_camera_matrices(triplet.left_wrist, None)
    if right_calibrated:
        right_phone_mats = side_phone_camera_matrices(triplet.right_wrist, right_x)
    else:
        right_phone_mats = side_phone_camera_matrices(triplet.right_wrist, None)

    if not calibrated:
        head_mats = center_local_matrices(head_mats)
        left_head_phone_mats = center_local_matrices(left_head_phone_mats)
        left_phone_mats = center_local_matrices(left_phone_mats)
        right_head_phone_mats = center_local_matrices(right_head_phone_mats)
        right_phone_mats = center_local_matrices(right_phone_mats)

    traces = []
    for name, mats, color, width in [
        ("head", head_mats, "#222222", 6),
        ("left_phone_session_iphone_camera", left_phone_mats, "#006D77", 5),
        ("left_head_session_iphone_camera_via_mount", left_head_phone_mats, "#6BC7D6", 4),
        ("right_phone_session_iphone_camera", right_phone_mats, "#C44536", 5),
        ("right_head_session_iphone_camera_via_mount", right_head_phone_mats, "#F4A261", 4),
    ]:
        plot_mats = downsample_matrices(mats, max_points)
        if len(plot_mats) == 0:
            continue
        pts = plot_mats[:, :3, 3]
        traces.append(
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="lines",
                name=name,
                line=dict(color=color, width=width),
            )
        )
        traces.append(
            go.Scatter3d(
                x=[pts[0, 0]],
                y=[pts[0, 1]],
                z=[pts[0, 2]],
                mode="markers",
                name=f"{name}_start",
                marker=dict(size=4, color=color),
                showlegend=False,
            )
        )
        add_pose_axis_traces(traces, go, name, mats, axis_length=axis_length, axis_stride=axis_stride)

    mode_text = "five calibrated RHS trajectories in head world" if calibrated else "diagnostic only: each trajectory is local-centered, not globally calibrated"
    axis_text = "local axes: red X, green Y, blue Z"
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"{triplet.parent_uuid} - {mode_text}; {axis_text}",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y (gravity/up)",
            zaxis_title="Z",
            aspectmode="data",
        ),
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=0, r=0, t=55, b=0),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")


def write_visualizations(
    triplets: Sequence[Triplet],
    calib_report: Dict[str, Any],
    outputs_dir: Path,
    max_points: int,
    axis_stride: int,
    axis_length: float,
    parent_uuids: Optional[set] = None,
) -> List[Path]:
    html_paths: List[Path] = []
    for triplet in triplets:
        if parent_uuids is not None and triplet.parent_uuid not in parent_uuids:
            continue
        path = outputs_dir / f"trajectory_{triplet.parent_uuid}.html"
        make_trajectory_html(
            triplet,
            calib_report,
            path,
            max_points=max_points,
            axis_stride=axis_stride,
            axis_length=axis_length,
        )
        html_paths.append(path)
    return html_paths


def summarize_calibration(calib: Dict[str, Any]) -> Dict[str, Any]:
    triplets = calib.get("triplets", [])
    side_statuses: Dict[str, int] = defaultdict(int)
    outlier_sides = 0
    for item in triplets:
        for side in ("left", "right"):
            side_info = item.get(side, {})
            side_statuses[str(side_info.get("status", "missing"))] += 1
            if side_info.get("is_outlier_suggested"):
                outlier_sides += 1
    return {
        "triplet_count": len(triplets),
        "side_status_counts": dict(side_statuses),
        "outlier_side_count": outlier_sides,
    }


def classify_outliers(
    calib: Dict[str, Any],
    rmse_mm: float,
    p95_mm: float,
    rot_rmse_deg: float,
    frame_translation_mm: float,
    frame_rotation_deg: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    side_rows: List[Dict[str, Any]] = []
    match_rows: List[Dict[str, Any]] = []
    outlier_rows: List[Dict[str, Any]] = []
    thresholds = {
        "side_translation_rmse_mm": rmse_mm,
        "side_translation_p95_mm": p95_mm,
        "side_rotation_rmse_deg": rot_rmse_deg,
        "frame_translation_mm": frame_translation_mm,
        "frame_rotation_deg": frame_rotation_deg,
    }
    calib["outlier_thresholds"] = thresholds
    for item in calib.get("triplets", []):
        parent_uuid = item.get("parent_uuid", "")
        for side in ("left", "right"):
            side_info = item.get(side, {})
            if not isinstance(side_info, dict):
                continue
            row: Dict[str, Any] = {
                "parent_uuid": parent_uuid,
                "side": side,
                "status": side_info.get("status", "missing"),
            }
            if side_info.get("status") == "ok":
                trans_rmse_mm = 1000.0 * float(side_info.get("translation_rmse_m", 0.0))
                trans_p95_mm = 1000.0 * float(side_info.get("translation_p95_m", 0.0))
                rot_rmse = float(side_info.get("rotation_rmse_deg", 0.0))
                reasons = []
                if trans_rmse_mm > rmse_mm:
                    reasons.append("translation_rmse")
                if trans_p95_mm > p95_mm:
                    reasons.append("translation_p95")
                if rot_rmse > rot_rmse_deg:
                    reasons.append("rotation_rmse")
                side_info["is_outlier_suggested"] = bool(reasons)
                side_info["outlier_reason"] = ";".join(reasons)
                side_info["filter_pass"] = not reasons
                row.update(
                    {
                        "used_wrist_key": side_info.get("used_wrist_key", ""),
                        "wrist_timestamp_key": side_info.get("wrist_timestamp_key", ""),
                        "smooth_isolated_spikes": side_info.get("smooth_isolated_spikes", ""),
                        "residual_smoothed_frame_count": side_info.get("residual_smoothed_frame_count", ""),
                        "time_offset_method": side_info.get("method", ""),
                        "time_offset_quality": side_info.get("time_offset_quality", ""),
                        "time_offset_quality_reason": side_info.get("time_offset_quality_reason", ""),
                        "phone_time_offset_to_head_s": side_info.get("phone_time_offset_to_head_s", ""),
                        "time_offset_score": side_info.get("score", ""),
                        "time_offset_score_at_zero": side_info.get("score_at_zero", ""),
                        "time_offset_overlap_s": side_info.get("overlap_s", ""),
                        "time_offset_sample_count": side_info.get("sample_count", ""),
                        "match_dt_median_s": side_info.get("match_dt_median_s", ""),
                        "match_dt_max_s": side_info.get("match_dt_max_s", ""),
                        "matched_samples": side_info.get("matched_samples", ""),
                        "translation_rmse_mm": trans_rmse_mm,
                        "translation_median_mm": 1000.0 * float(side_info.get("translation_median_m", 0.0)),
                        "translation_p95_mm": trans_p95_mm,
                        "rotation_rmse_deg": rot_rmse,
                        "rotation_median_deg": side_info.get("rotation_median_deg", ""),
                        "rotation_p95_deg": side_info.get("rotation_p95_deg", ""),
                        "is_outlier_suggested": bool(reasons),
                        "outlier_reason": ";".join(reasons),
                        "filter_pass": not reasons,
                    }
                )
                if reasons:
                    side_outlier_row = row.copy()
                    side_outlier_row["outlier_scope"] = "side"
                    outlier_rows.append(side_outlier_row)
                for err in side_info.get("sample_errors", []):
                    frame_reasons = []
                    trans_err_mm = 1000.0 * float(err.get("translation_error_m", 0.0))
                    rot_err_deg = float(err.get("rotation_error_deg", 0.0))
                    if trans_err_mm > frame_translation_mm:
                        frame_reasons.append("frame_translation")
                    if rot_err_deg > frame_rotation_deg:
                        frame_reasons.append("frame_rotation")
                    match_row = {
                        "parent_uuid": parent_uuid,
                        "side": side,
                        **err,
                        "translation_error_mm": trans_err_mm,
                        "is_frame_outlier_suggested": bool(frame_reasons),
                        "frame_outlier_reason": ";".join(frame_reasons),
                    }
                    match_rows.append(match_row)
                    if frame_reasons:
                        frame_outlier_row = match_row.copy()
                        frame_outlier_row["outlier_scope"] = "frame"
                        outlier_rows.append(frame_outlier_row)
            side_rows.append(row)
    return side_rows, match_rows, outlier_rows


def aligned_trajectory_rows(match_rows: Sequence[Dict[str, Any]], side_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    side_filter: Dict[Tuple[str, str], bool] = {}
    side_reason: Dict[Tuple[str, str], str] = {}
    for row in side_rows:
        key = (str(row.get("parent_uuid", "")), str(row.get("side", "")))
        side_filter[key] = bool(row.get("filter_pass", True))
        side_reason[key] = str(row.get("outlier_reason", ""))

    rows: List[Dict[str, Any]] = []
    for row in match_rows:
        key = (str(row.get("parent_uuid", "")), str(row.get("side", "")))
        frame_pass = not bool(row.get("is_frame_outlier_suggested", False))
        side_pass = side_filter.get(key, True)
        rows.append(
            {
                "parent_uuid": row.get("parent_uuid", ""),
                "side": row.get("side", ""),
                "sample_idx": row.get("sample_idx", ""),
                "timestamp_head_s": row.get("timestamp_head_s", ""),
                "timestamp_phone_raw_s": row.get("timestamp_phone_raw_s", ""),
                "timestamp_phone_aligned_s": row.get("timestamp_phone_aligned_s", ""),
                "final_trajectory_source": row.get("final_trajectory_source", "phone_session_iphone_camera"),
                "x": row.get("aligned_x", ""),
                "y": row.get("aligned_y", ""),
                "z": row.get("aligned_z", ""),
                "qw": row.get("aligned_qw", ""),
                "qx": row.get("aligned_qx", ""),
                "qy": row.get("aligned_qy", ""),
                "qz": row.get("aligned_qz", ""),
                "comparison_trajectory_source": row.get("comparison_trajectory_source", "head_session_iphone_camera_via_mount"),
                "comparison_x": row.get("head_via_mount_x", ""),
                "comparison_y": row.get("head_via_mount_y", ""),
                "comparison_z": row.get("head_via_mount_z", ""),
                "comparison_qw": row.get("head_via_mount_qw", ""),
                "comparison_qx": row.get("head_via_mount_qx", ""),
                "comparison_qy": row.get("head_via_mount_qy", ""),
                "comparison_qz": row.get("head_via_mount_qz", ""),
                "side_filter_pass": side_pass,
                "frame_filter_pass": frame_pass,
                "filter_pass": side_pass and frame_pass,
                "side_outlier_reason": side_reason.get(key, ""),
                "frame_outlier_reason": row.get("frame_outlier_reason", ""),
                "translation_error_mm": row.get("translation_error_mm", ""),
                "rotation_error_deg": row.get("rotation_error_deg", ""),
                "smoothed_pose": row.get("smoothed_pose", ""),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--workdir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data-dir", "--input-dir", dest="data_dir", type=Path, default=None)
    parser.add_argument("--outputs-dir", "--output-dir", dest="outputs_dir", type=Path, default=None)
    parser.add_argument("--extract", action="store_true", help="Extract metadata.json and frame_data.bson from the archive.")
    parser.add_argument("--visualize", action="store_true", help="Write draggable Plotly 3D HTML visualizations.")
    parser.add_argument("--visualize-mode", choices=("all", "outliers"), default="outliers", help="When --visualize is set, write all triplets or only aggregate outliers.")
    parser.add_argument("--prefer-processed-wrist", action="store_true", help="Try leftWristPoses/rightWristPoses before Raw streams.")
    parser.add_argument("--min-valid", type=int, default=20)
    parser.add_argument("--max-match-dt", type=float, default=0.05)
    parser.add_argument("--time-offset-search", type=float, default=8.0, help="Search range in seconds for phone-to-head timestamp offset; set 0 to disable.")
    parser.add_argument("--smooth-isolated-spikes", action="store_true", help="Smooth isolated one-frame pose spikes before alignment.")
    parser.add_argument("--smooth-rot-spike-deg", type=float, default=5.0, help="Residual rotation threshold for isolated spike smoothing.")
    parser.add_argument("--smooth-trans-spike-mm", type=float, default=80.0, help="Residual translation threshold for isolated spike smoothing.")
    parser.add_argument("--outlier-rmse-mm", type=float, default=25.0)
    parser.add_argument("--outlier-p95-mm", type=float, default=80.0)
    parser.add_argument("--outlier-rot-rmse-deg", type=float, default=2.0)
    parser.add_argument("--frame-outlier-translation-mm", type=float, default=80.0)
    parser.add_argument("--frame-outlier-rotation-deg", type=float, default=5.0)
    parser.add_argument("--max-plot-points", type=int, default=2500)
    parser.add_argument("--axis-stride", type=int, default=45, help="Draw one local coordinate triad every N pose frames.")
    parser.add_argument("--axis-length", type=float, default=0.08, help="Length of local coordinate axes in meters.")
    args = parser.parse_args()

    workdir = args.workdir
    raw_dir = args.data_dir or (workdir / "data" / "raw")
    outputs_dir = args.outputs_dir or (workdir / "outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    if args.extract or not raw_dir.exists() or not any(raw_dir.glob("*/frame_data.bson")):
        if not args.archive.exists():
            raise FileNotFoundError(args.archive)
        count = extract_lightweight_files(args.archive, raw_dir)
        print(f"extracted_files={count}")

    records = load_records(raw_dir)
    triplets, groups = build_triplets(records)
    print(f"records={len(records)} groups={len(groups)} complete_triplets={len(triplets)}")

    sess_rows = session_rows(records, triplets)
    field_check_rows = field_rows(records)
    cont_rows = continuity_rows(records)
    write_csv(outputs_dir / "session_report.csv", sess_rows)
    write_csv(outputs_dir / "field_report.csv", field_check_rows)
    write_csv(outputs_dir / "continuity_report.csv", cont_rows)

    calib = calibration_report(
        triplets,
        prefer_raw=not args.prefer_processed_wrist,
        min_valid=args.min_valid,
        max_match_dt=args.max_match_dt,
        time_offset_search_s=args.time_offset_search,
        smooth_isolated_spikes=args.smooth_isolated_spikes,
        smooth_rot_spike_deg=args.smooth_rot_spike_deg,
        smooth_trans_spike_mm=args.smooth_trans_spike_mm,
    )
    side_rows, match_rows, outlier_rows = classify_outliers(
        calib,
        rmse_mm=args.outlier_rmse_mm,
        p95_mm=args.outlier_p95_mm,
        rot_rmse_deg=args.outlier_rot_rmse_deg,
        frame_translation_mm=args.frame_outlier_translation_mm,
        frame_rotation_deg=args.frame_outlier_rotation_deg,
    )
    aligned_rows = aligned_trajectory_rows(match_rows, side_rows)
    report_text = json.dumps(calib, indent=2)
    (outputs_dir / "calibration_report.json").write_text(report_text, encoding="utf-8")
    (outputs_dir / "align_report.json").write_text(report_text, encoding="utf-8")
    write_csv(outputs_dir / "side_report.csv", side_rows)
    write_csv(outputs_dir / "matches.csv", match_rows)
    write_csv(outputs_dir / "aligned_trajectories.csv", aligned_rows)
    write_csv(outputs_dir / "outlier_report.csv", outlier_rows)

    html_paths: List[Path] = []
    if args.visualize:
        parent_filter = None
        if args.visualize_mode == "outliers":
            parent_filter = {row["parent_uuid"] for row in outlier_rows}
        html_paths = write_visualizations(
            triplets,
            calib,
            outputs_dir,
            max_points=args.max_plot_points,
            axis_stride=args.axis_stride,
            axis_length=args.axis_length,
            parent_uuids=parent_filter,
        )

    summary = summarize_calibration(calib)
    print(json.dumps(summary, indent=2))
    if html_paths:
        print("html_outputs:")
        for path in html_paths:
            print(path)
    print(f"reports_dir={outputs_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
