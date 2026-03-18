from typing import Any, Dict, Optional, Tuple

import numpy as np
from torch.utils.data import DataLoader

from diffusion_policy.common.structured_batch_sampler import StructuredCoverageBatchSampler


def _to_plain_dict(cfg: Any) -> Dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return dict(cfg)
    try:
        from omegaconf import DictConfig, OmegaConf

        if isinstance(cfg, DictConfig):
            data = OmegaConf.to_container(cfg, resolve=True)
            return dict(data) if isinstance(data, dict) else {}
    except Exception:
        pass
    return dict(cfg)


def build_train_dataloader(
    dataset,
    dataloader_cfg: Any,
    default_seed: int = 42,
) -> Tuple[DataLoader, Optional[StructuredCoverageBatchSampler]]:
    """
    Build train DataLoader with optional structured batch sampling.

    Expected optional config:
      dataloader.structured_sampling.enabled: bool
      dataloader.structured_sampling.candidate_indices_path: str
      dataloader.structured_sampling.structured_ratio: float
      dataloader.structured_sampling.seed: int
    """
    cfg = _to_plain_dict(dataloader_cfg)
    structured_cfg = cfg.pop("structured_sampling", None)

    if not (isinstance(structured_cfg, dict) and structured_cfg.get("enabled", False)):
        return DataLoader(dataset, **cfg), None

    if "batch_size" not in cfg:
        raise ValueError("dataloader.batch_size is required when structured_sampling is enabled")

    batch_size = int(cfg.pop("batch_size"))
    shuffle = bool(cfg.pop("shuffle", True))
    drop_last = bool(cfg.pop("drop_last", False))
    cfg.pop("sampler", None)
    cfg.pop("batch_sampler", None)

    candidate_indices_path = structured_cfg.get("candidate_indices_path", None)
    if candidate_indices_path is None:
        raise ValueError("structured_sampling.candidate_indices_path is required when enabled")
    candidate_indices = np.load(candidate_indices_path)

    sampler = StructuredCoverageBatchSampler(
        num_samples=len(dataset),
        batch_size=batch_size,
        candidate_indices=candidate_indices,
        structured_ratio=float(structured_cfg.get("structured_ratio", 0.5)),
        shuffle=shuffle,
        drop_last=drop_last,
        seed=int(structured_cfg.get("seed", default_seed)),
    )
    loader = DataLoader(dataset, batch_sampler=sampler, **cfg)
    return loader, sampler


def set_epoch_for_structured_sampler(dataloader, epoch: int) -> bool:
    """
    Best-effort set_epoch for wrapped dataloaders.
    Returns True if a sampler with set_epoch was found.
    """
    updated = False

    def _try_set(obj) -> bool:
        if obj is None:
            return False
        for attr in ("batch_sampler", "sampler"):
            s = getattr(obj, attr, None)
            if hasattr(s, "set_epoch"):
                s.set_epoch(epoch)
                return True
        return False

    updated = _try_set(dataloader) or updated
    inner = getattr(dataloader, "dataloader", None)
    updated = _try_set(inner) or updated
    return updated

