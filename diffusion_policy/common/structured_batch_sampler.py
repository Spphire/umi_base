import math
from typing import Iterator, List, Optional

import numpy as np
import torch


class StructuredCoverageBatchSampler(torch.utils.data.Sampler[List[int]]):
    """
    Coverage-first batch sampler.

    Guarantees each sample appears at most once per epoch, and tries to build
    structured groups by consuming pre-ranked candidate lists for each anchor.
    """

    def __init__(
        self,
        num_samples: int,
        batch_size: int,
        candidate_indices: Optional[np.ndarray] = None,
        structured_ratio: float = 0.5,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
    ):
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if not np.isfinite(structured_ratio):
            raise ValueError("structured_ratio must be finite")
        if structured_ratio < 0.0 or structured_ratio > 1.0:
            raise ValueError(f"structured_ratio must be in [0,1], got {structured_ratio}")

        self.num_samples = int(num_samples)
        self.batch_size = int(batch_size)
        self.structured_ratio = float(structured_ratio)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

        if candidate_indices is not None:
            if candidate_indices.ndim != 2:
                raise ValueError("candidate_indices must be 2D [N, M]")
            if candidate_indices.shape[0] != self.num_samples:
                raise ValueError(
                    f"candidate_indices first dim must be N={self.num_samples}, "
                    f"got {candidate_indices.shape[0]}"
                )
            self.candidate_indices = candidate_indices.astype(np.int64, copy=False)
        else:
            self.candidate_indices = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        if self.drop_last:
            return self.num_samples // self.batch_size
        return int(math.ceil(self.num_samples / self.batch_size))

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        if self.shuffle:
            order = rng.permutation(self.num_samples).astype(np.int64)
        else:
            order = np.arange(self.num_samples, dtype=np.int64)

        unused = np.ones(self.num_samples, dtype=bool)
        fallback_cursor = 0
        # At least anchor must be included in each batch.
        target_structured = max(1, min(self.batch_size, int(round(self.batch_size * self.structured_ratio))))

        for anchor in order:
            anchor = int(anchor)
            if not unused[anchor]:
                continue

            batch: List[int] = [anchor]
            unused[anchor] = False

            if self.candidate_indices is not None and target_structured > 1:
                for c in self.candidate_indices[anchor]:
                    if len(batch) >= target_structured:
                        break
                    c = int(c)
                    if c < 0 or c >= self.num_samples:
                        continue
                    if unused[c]:
                        batch.append(c)
                        unused[c] = False

            while len(batch) < self.batch_size:
                while fallback_cursor < self.num_samples and not unused[int(order[fallback_cursor])]:
                    fallback_cursor += 1
                if fallback_cursor >= self.num_samples:
                    break
                idx = int(order[fallback_cursor])
                batch.append(idx)
                unused[idx] = False
                fallback_cursor += 1

            if len(batch) == self.batch_size:
                yield batch
            elif len(batch) > 0 and not self.drop_last:
                yield batch

