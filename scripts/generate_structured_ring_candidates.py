import argparse
import json
import os
from pathlib import Path

import numpy as np
import zarr


def infer_num_steps(zarr_path: str) -> int:
    root = zarr.open(os.path.expanduser(zarr_path), mode="r")
    if "meta" in root and "episode_ends" in root["meta"]:
        ends = root["meta"]["episode_ends"]
        if len(ends) == 0:
            return 0
        return int(ends[-1])
    if "data" in root:
        for _, arr in root["data"].items():
            return int(arr.shape[0])
    raise ValueError(f"Cannot infer dataset length from zarr: {zarr_path}")


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

    n = infer_num_steps(args.dataset_zarr)
    if n <= 1:
        raise ValueError(f"Dataset is too small for candidate generation, N={n}")

    m = min(int(args.top_m), n - 1)
    base = np.arange(1, m + 1, dtype=np.int64)
    idx = (np.arange(n, dtype=np.int64)[:, None] + base[None, :]) % n

    out_path = Path(os.path.expanduser(args.output))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), idx)

    meta = {
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
