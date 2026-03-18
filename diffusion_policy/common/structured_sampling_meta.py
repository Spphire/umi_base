import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _to_builtin(obj: Any) -> Any:
    """Convert nested config objects to JSON-serializable builtins."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    # Optional OmegaConf support without importing as hard dependency.
    if hasattr(obj, "items") and not isinstance(obj, (str, bytes)):
        try:
            return {str(k): _to_builtin(v) for k, v in obj.items()}
        except Exception:
            pass
    return str(obj)


def stable_json_hash(obj: Any) -> str:
    """SHA256 hash for nested objects after canonical JSON serialization."""
    payload = json.dumps(_to_builtin(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _safe_get(cfg: Any, path: Tuple[str, ...], default: Any = None) -> Any:
    cur = cfg
    for key in path:
        try:
            if isinstance(cur, dict):
                cur = cur[key]
            else:
                cur = getattr(cur, key)
        except Exception:
            return default
    return cur


def extract_encoder_signature(cfg: Any) -> Dict[str, Any]:
    """Build a compact signature that must match between cache and training."""
    return {
        "obs_encoder_target": _safe_get(cfg, ("policy", "obs_encoder", "_target_"), None),
        "model_name": _safe_get(cfg, ("policy", "obs_encoder", "model_name"), None),
        "pretrained": _safe_get(cfg, ("policy", "obs_encoder", "pretrained"), None),
    }


@dataclass
class StructuredSamplingMeta:
    version: str
    task_name: Optional[str]
    workspace_name: Optional[str]
    dataset_signature: str
    encoder_signature: Dict[str, Any]
    sampler_signature: Dict[str, Any]
    cfg_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_meta(
    cfg: Any,
    dataset_signature: str,
    sampler_signature: Dict[str, Any],
    version: str = "structured_sampling_v1",
) -> StructuredSamplingMeta:
    encoder_sig = extract_encoder_signature(cfg)
    cfg_hash = stable_json_hash(_to_builtin(cfg))
    return StructuredSamplingMeta(
        version=version,
        task_name=_safe_get(cfg, ("task_name",), None),
        workspace_name=_safe_get(cfg, ("name",), None),
        dataset_signature=dataset_signature,
        encoder_signature=encoder_sig,
        sampler_signature=_to_builtin(sampler_signature),
        cfg_hash=cfg_hash,
    )


def save_meta(meta_path: str, meta: StructuredSamplingMeta) -> None:
    path = Path(meta_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(meta.to_dict(), f, indent=2, sort_keys=True, ensure_ascii=True)


def load_meta(meta_path: str) -> StructuredSamplingMeta:
    with Path(meta_path).open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return StructuredSamplingMeta(**raw)


def is_meta_compatible(
    existing: StructuredSamplingMeta,
    current: StructuredSamplingMeta,
) -> Tuple[bool, Optional[str]]:
    """Compatibility rule for reusing cache artifacts."""
    if existing.version != current.version:
        return False, f"version mismatch: {existing.version} != {current.version}"
    if existing.dataset_signature != current.dataset_signature:
        return False, "dataset signature mismatch"
    if existing.encoder_signature != current.encoder_signature:
        return False, "encoder signature mismatch"
    if existing.sampler_signature != current.sampler_signature:
        return False, "sampler signature mismatch"
    return True, None

