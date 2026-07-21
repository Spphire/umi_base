#!/usr/bin/env python3
"""Monitor dualfold training milestones and measure seed-related action jumps.

The monitor watches a run's logs.json.txt for completed epoch milestones. For
each epoch multiple requested by --epoch-interval it waits for
checkpoints/latest.ckpt to become stable, validates the checkpoint's internal
epoch, launches one worker per GPU, and aggregates action prediction statistics
on a fixed validation batch.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_gpus(text: str) -> List[str]:
    gpus = [item.strip() for item in text.split(",") if item.strip()]
    if not gpus:
        raise ValueError("--gpu-ids must contain at least one GPU id")
    return gpus


def build_seed_list(seed_start: int, num_seeds: int) -> List[int]:
    if num_seeds <= 0:
        raise ValueError("--num-seeds must be positive")
    return list(range(seed_start, seed_start + num_seeds))


def split_round_robin(items: Sequence[int], n: int) -> List[List[int]]:
    return [list(items[i::n]) for i in range(n)]


def read_completed_epochs(run_dir: Path) -> List[int]:
    log_path = run_dir / "logs.json.txt"
    if not log_path.is_file():
        return []

    epochs = set()
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            epoch = payload.get("epoch")
            if epoch is None:
                continue
            try:
                epoch_i = int(epoch)
            except (TypeError, ValueError):
                continue
            epochs.add(epoch_i)
    return sorted(epochs)


def milestone_candidates(run_dir: Path, min_epoch: int, epoch_interval: int) -> List[int]:
    out = []
    for epoch in read_completed_epochs(run_dir):
        if epoch < min_epoch:
            continue
        if epoch_interval > 0 and (epoch % epoch_interval) != 0:
            continue
        out.append(epoch)
    return out


def wait_for_stable_file(path: Path, stable_checks: int = 3, sleep_s: float = 5.0) -> None:
    same_count = 0
    last_size = -1
    while same_count < stable_checks:
        if not path.is_file():
            same_count = 0
            last_size = -1
            time.sleep(sleep_s)
            continue
        size = path.stat().st_size
        if size > 1024 and size == last_size:
            same_count += 1
        else:
            same_count = 0
            last_size = size
        time.sleep(sleep_s)


def read_checkpoint_epoch(path: Path) -> int | None:
    import dill
    import torch

    try:
        payload = torch.load(path.open("rb"), pickle_module=dill, map_location="cpu")
    except Exception:
        return None
    pickles = payload.get("pickles", {})
    if "epoch" not in pickles:
        return None
    try:
        return int(dill.loads(pickles["epoch"]))
    except Exception:
        return None


def wait_for_latest_checkpoint(
    run_dir: Path,
    milestone_epoch: int,
    stable_checks: int,
    stable_sleep: float,
    retry_sleep: float,
) -> Path:
    ckpt_path = run_dir / "checkpoints" / "latest.ckpt"
    while True:
        wait_for_stable_file(ckpt_path, stable_checks=stable_checks, sleep_s=stable_sleep)
        ckpt_epoch = read_checkpoint_epoch(ckpt_path)
        if ckpt_epoch is not None and ckpt_epoch >= milestone_epoch:
            return ckpt_path
        print(
            f"[monitor] latest checkpoint epoch={ckpt_epoch} is behind milestone={milestone_epoch}; waiting",
            flush=True,
        )
        time.sleep(retry_sleep)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def finite_stats(values: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "mean": math.nan,
            "std": math.nan,
            "p50": math.nan,
            "p95": math.nan,
            "p99": math.nan,
            "max": math.nan,
        }
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def finite_histogram(values: np.ndarray, bins: Sequence[float]) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    edges = np.asarray(list(bins), dtype=np.float64)
    if arr.size == 0:
        counts = np.zeros(max(len(edges) - 1, 0), dtype=np.int64)
    else:
        counts, _ = np.histogram(arr, bins=edges)
    return {
        "bins": [float(x) for x in edges.tolist()],
        "counts": [int(x) for x in counts.tolist()],
    }


def normalize_vectors(values: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norm, 1e-8)


def ortho6d_to_rotation_matrix_np(rot6d: np.ndarray) -> np.ndarray:
    flat = np.asarray(rot6d, dtype=np.float64).reshape(-1, 6)
    x_raw = flat[:, 0:3]
    y_raw = flat[:, 3:6]
    x = normalize_vectors(x_raw)
    z = normalize_vectors(np.cross(x, y_raw))
    y = np.cross(z, x)
    mat = np.stack([x, y, z], axis=-1)
    return mat.reshape(rot6d.shape[:-1] + (3, 3))


def rotation_angle_deg(rot_a: np.ndarray, rot_b: np.ndarray) -> np.ndarray:
    rel = np.einsum("...ji,...jk->...ik", rot_a, rot_b)
    trace = np.trace(rel, axis1=-2, axis2=-1)
    cos_theta = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def action_rotation_groups(action_dim: int) -> List[Tuple[str, np.ndarray]]:
    if action_dim in (9, 10):
        return [("single", np.arange(3, 9, dtype=np.int64))]
    if action_dim in (18, 20):
        return [
            ("left", np.arange(3, 9, dtype=np.int64)),
            ("right", np.arange(12, 18, dtype=np.int64)),
        ]
    if action_dim == 19:
        return [
            ("left", np.arange(3, 9, dtype=np.int64)),
            ("head", np.arange(13, 19, dtype=np.int64)),
        ]
    if action_dim in (27, 29):
        return [
            ("left", np.arange(3, 9, dtype=np.int64)),
            ("right", np.arange(12, 18, dtype=np.int64)),
            ("head", np.arange(action_dim - 6, action_dim, dtype=np.int64)),
        ]
    return []


def top_rotation_temporal_jump(
    angle_by_part: Dict[str, np.ndarray],
    seeds: np.ndarray,
) -> Dict[str, Any]:
    best: Dict[str, Any] = {}
    best_value = -math.inf
    for part, angles in angle_by_part.items():
        finite = np.where(np.isfinite(angles), angles, -math.inf)
        if finite.size == 0:
            continue
        flat_idx = int(np.argmax(finite))
        value = float(finite.reshape(-1)[flat_idx])
        if value <= best_value:
            continue
        seed_idx, sample_idx, step_idx = np.unravel_index(flat_idx, finite.shape)
        best_value = value
        best = {
            "part": part,
            "seed": int(seeds[seed_idx]),
            "sample_index": int(sample_idx),
            "from_timestep": int(step_idx),
            "to_timestep": int(step_idx + 1),
            "angle_deg": value,
        }
    return best


def robust_rotation_temporal_outliers(
    angle_by_part: Dict[str, np.ndarray],
    seeds: np.ndarray,
    max_items: int = 20,
) -> Dict[str, Any]:
    records = []
    values = []
    for part, angles in angle_by_part.items():
        finite_mask = np.isfinite(angles)
        for seed_idx, sample_idx, step_idx in np.argwhere(finite_mask):
            value = float(angles[seed_idx, sample_idx, step_idx])
            values.append(value)
            records.append((value, part, seed_idx, sample_idx, step_idx))

    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {
            "threshold_deg": math.nan,
            "num_outlier_jumps": 0,
            "top_outlier_jumps": [],
        }

    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    scale = 1.4826 * mad
    threshold = median + 8.0 * scale if scale > 0 else median
    outliers = [item for item in records if item[0] > threshold]
    outliers.sort(key=lambda item: item[0], reverse=True)
    return {
        "threshold_deg": float(threshold),
        "num_outlier_jumps": int(len(outliers)),
        "top_outlier_jumps": [
            {
                "angle_deg": float(value),
                "part": part,
                "seed": int(seeds[seed_idx]),
                "sample_index": int(sample_idx),
                "from_timestep": int(step_idx),
                "to_timestep": int(step_idx + 1),
            }
            for value, part, seed_idx, sample_idx, step_idx in outliers[:max_items]
        ],
    }


def summarize_rotation_temporal_jumps(preds: np.ndarray, seeds: np.ndarray) -> Dict[str, Any]:
    groups = action_rotation_groups(int(preds.shape[-1]))
    bins = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 45.0, 90.0, 180.0]
    if not groups or preds.shape[2] < 2:
        return {
            "units": "degrees",
            "rotation_dim_groups": [
                {"name": name, "dims": [int(x) for x in dims.tolist()]}
                for name, dims in groups
            ],
            "combined": finite_stats(np.asarray([], dtype=np.float64)),
            "histogram": finite_histogram(np.asarray([], dtype=np.float64), bins),
            "by_part": {},
            "top_jump": {},
            "robust_outliers": {
                "threshold_deg": math.nan,
                "num_outlier_jumps": 0,
                "top_outlier_jumps": [],
            },
        }

    angle_by_part: Dict[str, np.ndarray] = {}
    by_part = {}
    for name, dims in groups:
        rot = ortho6d_to_rotation_matrix_np(preds[..., dims])
        angles = rotation_angle_deg(rot[:, :, :-1], rot[:, :, 1:])
        angle_by_part[name] = angles
        by_part[name] = {
            **finite_stats(angles),
            "histogram": finite_histogram(angles, bins),
        }

    combined_angles = np.concatenate(
        [angles.reshape(-1) for angles in angle_by_part.values()],
        axis=0,
    )
    return {
        "units": "degrees",
        "rotation_dim_groups": [
            {"name": name, "dims": [int(x) for x in dims.tolist()]}
            for name, dims in groups
        ],
        "combined": finite_stats(combined_angles),
        "histogram": finite_histogram(combined_angles, bins),
        "by_part": by_part,
        "top_jump": top_rotation_temporal_jump(angle_by_part, seeds),
        "robust_outliers": robust_rotation_temporal_outliers(angle_by_part, seeds),
    }


def top_seed_pair(seed_pair_abs: np.ndarray, seeds: np.ndarray) -> Dict[str, Any]:
    if seed_pair_abs.size == 0:
        return {}
    flat_idx = int(np.argmax(seed_pair_abs))
    pair_idx, sample_idx, step_idx, action_idx = np.unravel_index(flat_idx, seed_pair_abs.shape)
    return {
        "seed_a": int(seeds[pair_idx]),
        "seed_b": int(seeds[pair_idx + 1]),
        "sample_index": int(sample_idx),
        "timestep": int(step_idx),
        "action_dim": int(action_idx),
        "max_abs_delta": float(seed_pair_abs[pair_idx, sample_idx, step_idx, action_idx]),
    }


def robust_seed_outliers(seed_pair_abs: np.ndarray, seeds: np.ndarray) -> Dict[str, Any]:
    if seed_pair_abs.size == 0:
        return {
            "threshold": math.nan,
            "pair_max_abs": [],
            "outlier_pairs": [],
        }
    pair_max = seed_pair_abs.reshape(seed_pair_abs.shape[0], -1).max(axis=1)
    median = float(np.median(pair_max))
    mad = float(np.median(np.abs(pair_max - median)))
    scale = 1.4826 * mad
    threshold = median + 5.0 * scale if scale > 0 else median
    outliers = []
    for idx, value in enumerate(pair_max):
        if value > threshold:
            outliers.append({
                "seed_a": int(seeds[idx]),
                "seed_b": int(seeds[idx + 1]),
                "max_abs_delta": float(value),
            })
    return {
        "threshold": float(threshold),
        "pair_max_abs": [float(x) for x in pair_max],
        "outlier_pairs": outliers,
    }


def summarize_predictions(
    ckpt_path: Path,
    epoch: int,
    result_dir: Path,
    worker_npz_paths: Sequence[Path],
    checkpoint_epoch: int | None,
    source_ckpt_path: Path | None = None,
) -> Dict[str, Any]:
    seed_blocks = []
    pred_blocks = []
    gt_action = None
    for path in worker_npz_paths:
        data = np.load(path)
        seed_blocks.append(data["seeds"].astype(np.int64))
        pred_blocks.append(data["preds"].astype(np.float32))
        if gt_action is None:
            gt_action = data["gt_action"].astype(np.float32)

    seeds = np.concatenate(seed_blocks, axis=0)
    preds = np.concatenate(pred_blocks, axis=0)
    order = np.argsort(seeds)
    seeds = seeds[order]
    preds = preds[order]

    temporal_abs = np.abs(np.diff(preds, axis=2))
    seed_pair_abs = np.abs(np.diff(preds, axis=0))
    seed_pair_l2 = np.sqrt(np.mean(np.square(np.diff(preds, axis=0)), axis=(2, 3)))

    summary: Dict[str, Any] = {
        "created_at_utc": utc_now(),
        "ckpt_path": str(ckpt_path),
        "source_ckpt_path": str(source_ckpt_path) if source_ckpt_path is not None else str(ckpt_path),
        "epoch": int(epoch),
        "checkpoint_epoch": checkpoint_epoch,
        "result_dir": str(result_dir),
        "seeds": [int(x) for x in seeds.tolist()],
        "num_seeds": int(len(seeds)),
        "num_samples": int(preds.shape[1]),
        "horizon": int(preds.shape[2]),
        "action_dim": int(preds.shape[3]),
        "prediction_finite": bool(np.isfinite(preds).all()),
        "temporal_step_abs_delta": finite_stats(temporal_abs),
        "rotation_temporal_angle_deg": summarize_rotation_temporal_jumps(preds, seeds),
        "seed_pair_abs_delta": finite_stats(seed_pair_abs),
        "seed_pair_l2_delta_per_sample": finite_stats(seed_pair_l2),
        "top_seed_pair_jump": top_seed_pair(seed_pair_abs, seeds),
        "robust_seed_pair_outliers": robust_seed_outliers(seed_pair_abs, seeds),
    }
    if gt_action is not None:
        mse = np.mean(np.square(preds - gt_action[None, ...]), axis=(1, 2, 3))
        summary["mse_to_dataset_action_by_seed"] = {
            "seeds": [int(x) for x in seeds.tolist()],
            "mse": [float(x) for x in mse.tolist()],
            "stats": finite_stats(mse),
        }
    return summary


def run_ckpt(
    args: argparse.Namespace,
    ckpt_path: Path,
    epoch: int,
    checkpoint_epoch: int | None = None,
) -> Dict[str, Any]:
    run_dir = Path(args.run_dir).resolve()
    result_dir = run_dir / "seed_jump_stats" / f"epoch={epoch:04d}"
    summary_path = result_dir / "summary.json"
    if summary_path.is_file() and not args.force:
        with summary_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    wait_for_stable_file(ckpt_path, stable_checks=args.stable_checks, sleep_s=args.stable_sleep)
    if checkpoint_epoch is None:
        checkpoint_epoch = read_checkpoint_epoch(ckpt_path)
    result_dir.mkdir(parents=True, exist_ok=True)
    source_ckpt_path = ckpt_path
    if args.copy_ckpt_for_test:
        copy_path = result_dir / "latest_for_seed_jump.ckpt"
        if copy_path.exists() and args.force:
            copy_path.unlink()
        if not copy_path.exists():
            shutil.copy2(ckpt_path, copy_path)
        ckpt_path = copy_path
        checkpoint_epoch = read_checkpoint_epoch(ckpt_path)

    write_json(result_dir / "started.json", {
        "created_at_utc": utc_now(),
        "ckpt_path": str(ckpt_path),
        "source_ckpt_path": str(source_ckpt_path),
        "epoch": epoch,
        "checkpoint_epoch": checkpoint_epoch,
        "args": vars(args),
    })

    gpus = parse_gpus(args.gpu_ids)
    seeds = build_seed_list(args.seed_start, args.num_seeds)
    seed_groups = split_round_robin(seeds, len(gpus))
    worker_npz_paths: List[Path] = []
    processes = []
    for gpu, seed_group in zip(gpus, seed_groups):
        if not seed_group:
            continue
        out_npz = result_dir / f"worker_gpu{gpu}.npz"
        log_path = result_dir / f"worker_gpu{gpu}.log"
        worker_npz_paths.append(out_npz)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "worker",
            "--ckpt",
            str(ckpt_path),
            "--seeds",
            ",".join(str(seed) for seed in seed_group),
            "--num-samples",
            str(args.num_samples),
            "--eval-batch-size",
            str(args.eval_batch_size),
            "--output",
            str(out_npz),
            "--gpu-label",
            str(gpu),
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("HYDRA_FULL_ERROR", "1")
        log_f = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(cmd, cwd=str(repo_root()), env=env, stdout=log_f, stderr=subprocess.STDOUT)
        processes.append((proc, log_f, log_path, gpu))

    errors = []
    for proc, log_f, log_path, gpu in processes:
        rc = proc.wait()
        log_f.close()
        if rc != 0:
            errors.append({"gpu": gpu, "returncode": rc, "log": str(log_path)})
    if errors:
        payload = {
            "created_at_utc": utc_now(),
            "ckpt_path": str(ckpt_path),
            "epoch": epoch,
            "errors": errors,
        }
        write_json(result_dir / "error.json", payload)
        raise RuntimeError(f"seed jump workers failed: {errors}")

    summary = summarize_predictions(
        ckpt_path,
        epoch,
        result_dir,
        worker_npz_paths,
        checkpoint_epoch,
        source_ckpt_path=source_ckpt_path,
    )
    write_json(summary_path, summary)
    append_jsonl(run_dir / "seed_jump_stats" / "summary.jsonl", summary)
    if not args.keep_predictions:
        for path in worker_npz_paths:
            path.unlink(missing_ok=True)
    return summary


def monitor(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir).resolve()
    print(f"[monitor] watching training progress in {run_dir}", flush=True)
    processed = set()
    while True:
        candidates = milestone_candidates(run_dir, args.min_epoch, args.epoch_interval)
        for epoch in candidates:
            summary_path = run_dir / "seed_jump_stats" / f"epoch={epoch:04d}" / "summary.json"
            if epoch in processed:
                continue
            if summary_path.is_file() and not args.force:
                processed.add(epoch)
                continue
            print(f"[monitor] milestone epoch={epoch}; waiting for latest.ckpt", flush=True)
            try:
                ckpt_path = wait_for_latest_checkpoint(
                    run_dir=run_dir,
                    milestone_epoch=epoch,
                    stable_checks=args.stable_checks,
                    stable_sleep=args.stable_sleep,
                    retry_sleep=args.poll_sec,
                )
                checkpoint_epoch = read_checkpoint_epoch(ckpt_path)
                print(
                    f"[monitor] processing milestone={epoch} latest_ckpt_epoch={checkpoint_epoch} ckpt={ckpt_path}",
                    flush=True,
                )
                summary = run_ckpt(args, ckpt_path, epoch, checkpoint_epoch=checkpoint_epoch)
                rot_summary = summary.get("rotation_temporal_angle_deg", {}).get("combined", {})
                print(
                    "[monitor] done milestone={epoch} ckpt_epoch={ckpt_epoch} rot_temporal_p99_deg={rp99:.6g} rot_temporal_max_deg={rmax:.6g} seed_pair_p99={p99:.6g}".format(
                        epoch=epoch,
                        ckpt_epoch=summary.get("checkpoint_epoch"),
                        rp99=rot_summary.get("p99", math.nan),
                        rmax=rot_summary.get("max", math.nan),
                        p99=summary["seed_pair_abs_delta"]["p99"],
                    ),
                    flush=True,
                )
                processed.add(epoch)
            except Exception as exc:
                print(f"[monitor] failed epoch={epoch}: {exc}", flush=True)

        if args.once:
            return
        time.sleep(args.poll_sec)


def worker(args: argparse.Namespace) -> None:
    import dill
    import hydra
    import torch
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader

    root = repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from diffusion_policy.common.pytorch_util import dict_apply
    from diffusion_policy.workspace.train_diffusion_transformer_timm_workspace import (
        TrainDiffusionTransformerTimmWorkspace,
    )

    OmegaConf.register_new_resolver("eval", eval, replace=True)
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    if not seeds:
        raise ValueError("worker received no seeds")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[worker gpu={args.gpu_label}] loading {args.ckpt} on {device}", flush=True)

    with open(args.ckpt, "rb") as f:
        payload = torch.load(f, pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    workspace = TrainDiffusionTransformerTimmWorkspace(cfg)
    workspace.load_payload(payload, exclude_keys=("optimizer", "lr_scheduler"))
    policy = workspace.ema_model if cfg.training.use_ema and workspace.ema_model is not None else workspace.model
    policy.to(device)
    policy.eval()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    val_dataset = dataset.get_validation_dataset()
    sample_dataset = val_dataset if len(val_dataset) > 0 else dataset
    loader = DataLoader(
        sample_dataset,
        batch_size=args.num_samples,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    batch = next(iter(loader))
    obs_cpu = batch["obs"]
    gt_action = batch["action"].detach().cpu().numpy().astype(np.float32)
    num_samples = int(gt_action.shape[0])
    print(
        f"[worker gpu={args.gpu_label}] seeds={seeds} samples={num_samples} batch={args.eval_batch_size}",
        flush=True,
    )

    preds = []
    with torch.inference_mode():
        for seed in seeds:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            seed_preds = []
            for start in range(0, num_samples, args.eval_batch_size):
                end = min(start + args.eval_batch_size, num_samples)
                obs_chunk = dict_apply(
                    {key: value[start:end] for key, value in obs_cpu.items()},
                    lambda x: x.to(device, non_blocking=False),
                )
                out = policy.predict_action(obs_chunk)["action_pred"]
                seed_preds.append(out.detach().cpu().numpy().astype(np.float32))
            preds.append(np.concatenate(seed_preds, axis=0))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        seeds=np.asarray(seeds, dtype=np.int64),
        preds=np.stack(preds, axis=0),
        gt_action=gt_action,
    )
    print(f"[worker gpu={args.gpu_label}] wrote {output}", flush=True)


def add_common_monitor_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--gpu-ids", default="6,7")
    parser.add_argument("--min-epoch", type=int, default=100)
    parser.add_argument("--epoch-interval", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=32)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--poll-sec", type=float, default=60.0)
    parser.add_argument("--stable-checks", type=int, default=3)
    parser.add_argument("--stable-sleep", type=float, default=5.0)
    parser.add_argument("--keep-predictions", action="store_true")
    parser.add_argument("--copy-ckpt-for-test", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--once", action="store_true")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    monitor_parser = subparsers.add_parser("monitor")
    add_common_monitor_args(monitor_parser)

    run_parser = subparsers.add_parser("run")
    add_common_monitor_args(run_parser)
    run_parser.add_argument("--ckpt", required=True)
    run_parser.add_argument("--epoch", type=int, required=True)

    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("--ckpt", required=True)
    worker_parser.add_argument("--seeds", required=True)
    worker_parser.add_argument("--num-samples", type=int, required=True)
    worker_parser.add_argument("--eval-batch-size", type=int, required=True)
    worker_parser.add_argument("--output", required=True)
    worker_parser.add_argument("--gpu-label", default="unknown")

    args = parser.parse_args(argv)
    if args.command == "monitor":
        monitor(args)
    elif args.command == "run":
        summary = run_ckpt(args, Path(args.ckpt).resolve(), args.epoch)
        print(json.dumps(summary, indent=2, sort_keys=True))
    elif args.command == "worker":
        worker(args)
    else:
        parser.error(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
