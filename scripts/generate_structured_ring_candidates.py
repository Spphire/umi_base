import argparse
import json
import os
import sys
from pathlib import Path

import hydra
import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT_DIR = str(Path(__file__).resolve().parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)


def infer_num_samples_from_dataset(
    config_dir: str,
    config_name: str,
    task: str,
    dataset_zarr: str,
) -> int:
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    abs_config_dir = str(Path(config_dir).expanduser().resolve())
    local_zarr = os.path.expanduser(dataset_zarr)

    overrides = [
        f"task={task}",
        f"task.dataset.local_files_only={local_zarr}",
    ]

    with initialize_config_dir(config_dir=abs_config_dir, version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)
    OmegaConf.resolve(cfg)

    dataset = hydra.utils.instantiate(cfg.task.dataset)
    n = len(dataset)
    if n <= 1:
        raise ValueError(f"Dataset is too small for candidate generation, N={n}")
    return int(n)


def main():
    parser = argparse.ArgumentParser(
        description="Generate fallback ring candidates for structured batch sampling."
    )
    parser.add_argument(
        "--dataset_zarr",
        required=True,
        help="Path to replay_buffer.zarr",
    )
    parser.add_argument(
        "--config_dir",
        default="diffusion_policy/config",
        help="Hydra config directory",
    )
    parser.add_argument(
        "--config_name",
        default="train_diffusion_unet_timm_single_frame_workspace",
        help="Hydra train config name",
    )
    parser.add_argument(
        "--task",
        default="q3_mouse",
        help="Hydra task name",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output .npy path for candidate indices [N, M]",
    )
    parser.add_argument(
        "--top_m",
        type=int,
        default=192,
        help="Number of candidates per anchor",
    )
    args = parser.parse_args()

    if args.top_m <= 0:
        raise ValueError(f"--top_m must be > 0, got {args.top_m}")

    n = infer_num_samples_from_dataset(
        config_dir=args.config_dir,
        config_name=args.config_name,
        task=args.task,
        dataset_zarr=args.dataset_zarr,
    )

    m = min(int(args.top_m), n - 1)
    base = np.arange(1, m + 1, dtype=np.int64)
    idx = (np.arange(n, dtype=np.int64)[:, None] + base[None, :]) % n

    out_path = Path(os.path.expanduser(args.output))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), idx)

    meta = {
        "config_dir": str(Path(args.config_dir).expanduser().resolve()),
        "config_name": args.config_name,
        "task": args.task,
        "dataset_zarr": os.path.expanduser(args.dataset_zarr),
        "output": str(out_path),
        "num_samples": int(n),
        "top_m": int(m),
        "construction": "ring-shift",
        "note": "Fallback candidates for structured sampling smoke/stability runs",
    }
    with open(str(out_path) + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[ok] saved candidates: {out_path} shape={idx.shape}")


if __name__ == "__main__":
    main()
