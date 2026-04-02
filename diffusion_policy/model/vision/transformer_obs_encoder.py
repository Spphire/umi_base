import copy
from typing import Any, Dict, Optional

import timm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import logging

from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin

from diffusion_policy.common.pytorch_util import replace_submodules

logger = logging.getLogger(__name__)


def _resolve_per_key_bool(setting: Any, key: str, default: bool = False) -> bool:
    if isinstance(setting, bool):
        return setting
    if hasattr(setting, "items") and hasattr(setting, "get"):
        if key in setting:
            return bool(setting[key])
        for cfg_key, cfg_value in setting.items():
            if cfg_key == "default":
                continue
            if isinstance(cfg_key, str) and cfg_key in key:
                return bool(cfg_value)
        if "default" in setting:
            return bool(setting.get("default"))
        return bool(default)
    if setting is None:
        return bool(default)
    return bool(setting)

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class TokenAttentionPool(nn.Module):
    """Learnable-query attention pooling over a ViT token sequence.

    Input:  x : [B, N, C]  (patch tokens, no CLS)
    Output: pooled : [B, C]  (caller unsqueezes to [B, 1, C])

    Caches per-call attention weights in ``_last_attn_weights``:
        shape [B, num_heads, 1, N], detached — mean over heads gives a
        [B, N] saliency map suitable for 16×16 heatmap overlay.
    """

    def __init__(self, embed_dim: int, num_heads: int, output_dim: Optional[int] = None):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) / embed_dim ** 0.5)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)
        out = output_dim if (output_dim is not None and output_dim != embed_dim) else None
        self.proj = nn.Linear(embed_dim, out) if out else nn.Identity()
        self._last_attn_weights: Optional[torch.Tensor] = None  # [B, heads, 1, N]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, C]  →  pooled: [B, C]"""
        q = self.query.expand(x.shape[0], -1, -1)  # [B, 1, C]
        pooled, attn_weights = self.attn(
            query=q, key=x, value=x,
            need_weights=True,
            average_attn_weights=False,  # keep per-head: [B, heads, 1, N]
        )
        self._last_attn_weights = attn_weights.detach()
        return self.proj(self.norm(pooled[:, 0, :]))  # [B, C]

    def get_last_attention_map(self) -> Optional[torch.Tensor]:
        """Returns mean-over-heads attention: [B, N]"""
        if self._last_attn_weights is None:
            return None
        return self._last_attn_weights.mean(dim=1).squeeze(1)  # [B, N]


class TransformerObsEncoder(ModuleAttrMixin):
    def __init__(self,
            shape_meta: dict,
            model_name: str='vit_base_patch16_clip_224.openai',
            global_pool: str='',
            transforms: list=None,
            n_emb: int=768,
            pretrained: bool=False,
            frozen: Any=False,
            # replace BatchNorm with GroupNorm
            use_group_norm: bool=False,
            # use single rgb model for all rgb inputs
            share_rgb_model: bool=False,
            feature_aggregation: Any=None,
            downsample_ratio: int=32
        ):
        """
        Assumes rgb input: B,T,C,H,W
        Assumes low_dim input: B,T,D
        """
        super().__init__()
        
        rgb_keys = list()
        low_dim_keys = list()
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_projection_map = nn.ModuleDict()
        key_shape_map = dict()
        key_frozen_map: Dict[str, bool] = {}
        key_tokens_per_step_map: Dict[str, int] = {}

        assert global_pool == ''
        model = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            global_pool=global_pool, # '' means no pooling
            num_classes=0            # remove classification layer
        )
        self.model_name = model_name
        
        feature_dim = None
        if model_name.startswith('resnet'):
            # the last layer is nn.Identity() because num_classes is 0
            # second last layer is AdaptivePool2d, which is also identity because global_pool is empty
            if downsample_ratio == 32:
                modules = list(model.children())[:-2]
                model = torch.nn.Sequential(*modules)
                feature_dim = 512
            elif downsample_ratio == 16:
                modules = list(model.children())[:-3]
                model = torch.nn.Sequential(*modules)
                feature_dim = 256
            else:
                raise NotImplementedError(f"Unsupported downsample_ratio: {downsample_ratio}")
        elif model_name.startswith('convnext'):
            # the last layer is nn.Identity() because num_classes is 0
            # second last layer is AdaptivePool2d, which is also identity because global_pool is empty
            if downsample_ratio == 32:
                modules = list(model.children())[:-2]
                model = torch.nn.Sequential(*modules)
                feature_dim = 1024
            else:
                raise NotImplementedError(f"Unsupported downsample_ratio: {downsample_ratio}")

        if use_group_norm and not pretrained:
            model = replace_submodules(
                root_module=model,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=(x.num_features // 16) if (x.num_features % 16 == 0) else (x.num_features // 8), 
                    num_channels=x.num_features)
            )
            
        # ---------------------------------------------------------------------------
        # Per-key feature aggregation
        # ---------------------------------------------------------------------------
        # feature_aggregation can be:
        #   str / None : same strategy for every rgb key
        #     ViT  → 'cls' | 'attn_pool' | None (keep all tokens)
        #     CNN  → 'avg' | 'max' | 'soft_attention' | 'spatial_embedding' | ...
        #   dict       : per-key overrides, e.g.
        #     {'default': 'cls', 'left_wrist_img': 'cls', 'left_eye_img': 'attn_pool'}
        # ---------------------------------------------------------------------------
        self.feature_aggregation = feature_aggregation          # kept for CNN compat
        self.key_aggregation_map: Dict[str, Optional[str]] = {} # filled per-key below
        self.key_pool_module_map = nn.ModuleDict()              # TokenAttentionPool per key
        self._last_attention_maps: Dict[str, torch.Tensor] = {} # [B, N] heatmap cache

        # CNN-only legacy global module (soft_attention)
        _global_agg_str = feature_aggregation if isinstance(feature_aggregation, str) else None
        if (not model_name.startswith('vit')) and _global_agg_str == 'soft_attention':
            self.attention = nn.Sequential(
                nn.Linear(feature_dim, 1, bias=False),
                nn.Softmax(dim=1)
            )
        
        image_shape = None
        obs_shape_meta = shape_meta['obs']
        rgb_obs_keys = list()
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_obs_keys.append(key)
                assert image_shape is None or image_shape == shape[1:]
                image_shape = shape[1:]
        if share_rgb_model and hasattr(frozen, "items") and hasattr(frozen, "get"):
            resolved_frozen = {
                key: _resolve_per_key_bool(frozen, key, default=False)
                for key in rgb_obs_keys
            }
            if len(set(resolved_frozen.values())) > 1:
                raise ValueError(
                    "share_rgb_model=True is incompatible with per-key frozen overrides. "
                    "Set share_rgb_model=False to freeze head and fine-tune wrist independently."
                )
        if transforms is not None and not isinstance(transforms[0], torch.nn.Module):
            assert transforms[0].type == 'RandomCrop'
            ratio = transforms[0].ratio
            transforms = [
                torchvision.transforms.RandomCrop(size=int(image_shape[0] * ratio)),
                torchvision.transforms.Resize(size=image_shape[0], antialias=True)
            ] + transforms[1:]
        transform = nn.Identity() if transforms is None else torch.nn.Sequential(*transforms)

        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            key_shape_map[key] = shape
            if type == 'rgb':
                rgb_keys.append(key)

                this_model = model if share_rgb_model else copy.deepcopy(model)
                key_is_frozen = _resolve_per_key_bool(frozen, key, default=False)
                if key_is_frozen:
                    assert pretrained
                    for param in this_model.parameters():
                        param.requires_grad = False
                key_model_map[key] = this_model
                key_frozen_map[key] = key_is_frozen

                # Determine per-key aggregation strategy
                #
                # Supported behavior:
                # 1) feature_aggregation is str / None:
                #    use the same aggregation for all rgb obs keys.
                # 2) feature_aggregation is dict:
                #    - exact key match has highest priority
                #    - otherwise, try substring match where cfg_key is contained in obs key
                #    - if no match, fallback to 'default' (if provided), else None
                is_mapping_agg = hasattr(feature_aggregation, 'items') and hasattr(feature_aggregation, 'get')
                if is_mapping_agg:
                    key_agg = None

                    # priority 1: exact match
                    if key in feature_aggregation:
                        key_agg = feature_aggregation[key]
                    else:
                        # priority 2: substring match
                        # e.g. cfg key 'wrist' matches obs key 'left_wrist_img'
                        for cfg_key, cfg_agg in feature_aggregation.items():
                            if cfg_key == 'default':
                                continue
                            if isinstance(cfg_key, str) and cfg_key in key:
                                key_agg = cfg_agg
                                break

                    # priority 3: default fallback
                    if key_agg is None:
                        key_agg = feature_aggregation.get('default', None)
                else:
                    key_agg = feature_aggregation
                if model_name.startswith('vit') and key_agg not in (None, 'cls', 'attn_pool'):
                    logger.warn(
                        f'[{key}] ViT does not support aggregation "{key_agg}", falling back to "cls"'
                    )
                    key_agg = 'cls'
                self.key_aggregation_map[key] = key_agg

                # check if we need feature projection
                with torch.no_grad():
                    example_img = torch.zeros((1,)+tuple(shape))
                    example_feature_map = this_model(example_img)
                    # For attn_pool: build TokenAttentionPool from the backbone output dim
                    if model_name.startswith('vit') and key_agg == 'attn_pool':
                        patch_tokens = example_feature_map[:, 1:, :]  # skip CLS
                        embed_dim = patch_tokens.shape[-1]
                        num_heads = max(1, embed_dim // 64)
                        pool_mod = TokenAttentionPool(embed_dim=embed_dim, num_heads=num_heads)
                        self.key_pool_module_map[key] = pool_mod
                        example_features = pool_mod(patch_tokens).unsqueeze(1)  # [1, 1, C]
                    else:
                        example_features = self.aggregate_feature(key, example_feature_map)
                    feature_shape = example_features.shape
                    feature_size = feature_shape[-1]
                    key_tokens_per_step_map[key] = int(feature_shape[1])
                proj = nn.Identity()
                if feature_size != n_emb:
                    proj = nn.Linear(in_features=feature_size, out_features=n_emb)
                key_projection_map[key] = proj

                this_transform = transform
                key_transform_map[key] = this_transform
            elif type == 'low_dim':
                dim = np.prod(shape)
                proj = nn.Identity()
                if dim != n_emb:
                    proj = nn.Linear(in_features=dim, out_features=n_emb)
                key_projection_map[key] = proj
                key_tokens_per_step_map[key] = 1

                low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
        
        feature_map_shape = [x // downsample_ratio for x in image_shape]
            
        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)

        self.n_emb = n_emb
        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.key_projection_map = key_projection_map
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map
        self.key_frozen_map = key_frozen_map
        self.key_tokens_per_step_map = key_tokens_per_step_map
        self.key_token_count_map: Dict[str, int] = {}
        self.key_token_slice_map: Dict[str, slice] = {}
        token_cursor = 0
        for key in self.rgb_keys + self.low_dim_keys:
            horizon = int(obs_shape_meta[key]['horizon'])
            token_count = int(self.key_tokens_per_step_map[key]) * horizon
            self.key_token_count_map[key] = token_count
            self.key_token_slice_map[key] = slice(token_cursor, token_cursor + token_count)
            token_cursor += token_count
        self.total_obs_tokens = token_cursor

        self.record_feature_stats = False
        self._last_feature_map = dict()

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def enable_feature_recording(self, enabled: bool = True):
        self.record_feature_stats = enabled

    def pop_feature_grad_norms(self, norm_type: float = 2.0):
        result = {}
        for key, feat in self._last_feature_map.items():
            if feat is None or feat.grad is None:
                continue
            grad = feat.grad
            if grad.ndim > 1:
                grad_norm = grad.norm(p=norm_type, dim=-1).mean().item()
            else:
                grad_norm = grad.norm(p=norm_type).item()
            result[key] = grad_norm
        self._last_feature_map = dict()
        return result

    def get_param_grad_norms(self, norm_type: float = 2.0):
        result = {}
        for key, module in self.key_model_map.items():
            total = 0.0
            count = 0
            for p in module.parameters():
                if p.grad is None:
                    continue
                total += p.grad.norm(p=norm_type).item()
                count += 1
            if count > 0:
                result[key] = total / count
        return result

    # ------------------------------------------------------------------
    # Attention heatmap API
    # ------------------------------------------------------------------

    def get_last_attention_maps(self) -> Dict[str, torch.Tensor]:
        """Return cached attention maps from the last forward() call.

        Keys are obs keys that use ``'attn_pool'`` aggregation.
        Values are tensors of shape ``[B, N_patches]`` (mean over heads).

        N_patches = (H / patch_size) * (W / patch_size),
        e.g. 256 for ViT-B/14 at 224 px.

        Reshape to (H/patch_size, W/patch_size) and upsample to 224×224
        for a spatial heatmap overlay::

            attn = encoder.get_last_attention_maps()['left_eye_img']  # [B, 256]
            heat = attn.reshape(B, 16, 16)
            heat = F.interpolate(heat.unsqueeze(1).float(), (224, 224),
                                 mode='bilinear', align_corners=False).squeeze(1)
        """
        return dict(self._last_attention_maps)

    def pop_last_attention_maps(self) -> Dict[str, torch.Tensor]:
        """Return and clear cached attention maps."""
        result = dict(self._last_attention_maps)
        self._last_attention_maps.clear()
        return result

    # ------------------------------------------------------------------

    def get_token_slices(self, include_time_token: bool = False,
            time_token_name: str = '__time__') -> Dict[str, slice]:
        result = {
            key: self.key_token_slice_map[key]
            for key in (self.rgb_keys + self.low_dim_keys)
        }
        if include_time_token:
            result[time_token_name] = slice(self.total_obs_tokens, self.total_obs_tokens + 1)
        return result

    def get_token_layout(self, include_time_token: bool = False,
            time_token_name: str = '__time__') -> Dict[str, Dict[str, int]]:
        layout = dict()
        for key, token_slice in self.get_token_slices(
                include_time_token=include_time_token,
                time_token_name=time_token_name).items():
            layout[key] = {
                'start': int(token_slice.start),
                'end': int(token_slice.stop),
                'count': int(token_slice.stop - token_slice.start)
            }
        return layout

    def aggregate_feature(self, key: str, feature: torch.Tensor) -> torch.Tensor:
        """Aggregate backbone output for *key* into shape [B, N_out, C].

        Strategy is read from ``self.key_aggregation_map[key]``:

        * ``'cls'``       → [B, 1, C]   CLS token
        * ``'attn_pool'`` → [B, 1, C]   learned attention pool over patch tokens;
                            writes [B, N_patches] map to _last_attention_maps[key]
        * ``None``        → [B, N, C]   all ViT tokens (or flattened CNN spatial)
        """
        if self.model_name.startswith('vit'):
            agg = self.key_aggregation_map.get(key, None)
            if agg == 'cls':
                return feature[:, [0], :]           # [B, 1, C]
            if agg == 'attn_pool':
                pool = self.key_pool_module_map[key]
                patch_tokens = feature[:, 1:, :]    # skip CLS → [B, N_patches, C]
                pooled = pool(patch_tokens)          # [B, C]
                attn_map = pool.get_last_attention_map()  # [B, N_patches]
                if attn_map is not None:
                    self._last_attention_maps[key] = attn_map
                return pooled.unsqueeze(1)           # [B, 1, C]
            # None → return all tokens
            return feature                           # [B, N_all, C]

        # ---- CNN (ResNet / ConvNext) – legacy global behavior ----
        fa = self.feature_aggregation if isinstance(self.feature_aggregation, str) else None
        assert len(feature.shape) == 4
        if fa == 'attention_pool_2d':
            return self.attention_pool_2d(feature)

        feature = torch.flatten(feature, start_dim=-2)  # B, C, H*W
        feature = torch.transpose(feature, 1, 2)        # B, H*W, C

        if fa == 'avg':
            return torch.mean(feature, dim=1, keepdim=True)
        elif fa == 'max':
            return torch.amax(feature, dim=1, keepdim=True)
        elif fa == 'soft_attention':
            weight = self.attention(feature)
            return torch.sum(feature * weight, dim=1, keepdim=True)
        elif fa == 'spatial_embedding':
            return torch.mean(feature * self.spatial_embedding, dim=1, keepdim=True)
        else:
            return feature
        
    def forward(self, obs_dict):
        self._last_attention_maps.clear()
        embeddings = list()
        batch_size = next(iter(obs_dict.values())).shape[0]

        # process rgb input
        for key in self.rgb_keys:
            img = obs_dict[key]
            B, T = img.shape[:2]
            assert B == batch_size
            assert img.shape[2:] == self.key_shape_map[key]
            img = img.reshape(B*T, *img.shape[2:])
            # NaN/Inf check for input
            if not torch.isfinite(img).all():
                img_cpu = img.detach().float().cpu()
                print(f"[NaN Debug][{key}] input reshape min={img_cpu.min().item():.6f} max={img_cpu.max().item():.6f} finite={torch.isfinite(img_cpu).all().item()}", flush=True)
                raise RuntimeError(f"[{key}] NaN/Inf detected in input after reshape")
            img = self.key_transform_map[key](img)
            # NaN/Inf check after transform
            if not torch.isfinite(img).all():
                img_cpu = img.detach().float().cpu()
                print(f"[NaN Debug][{key}] after transform min={img_cpu.min().item():.6f} max={img_cpu.max().item():.6f} finite={torch.isfinite(img_cpu).all().item()}", flush=True)
                raise RuntimeError(f"[{key}] NaN/Inf detected after transform")
            raw_feature = self.key_model_map[key](img)
            # NaN/Inf check after backbone
            if not torch.isfinite(raw_feature).all():
                raw_cpu = raw_feature.detach().float().cpu()
                print(f"[NaN Debug][{key}] after backbone min={raw_cpu.min().item():.6f} max={raw_cpu.max().item():.6f} finite={torch.isfinite(raw_cpu).all().item()}", flush=True)
                raise RuntimeError(f"[{key}] NaN/Inf detected after backbone")
            feature = self.aggregate_feature(key, raw_feature)
            # NaN/Inf check after aggregate_feature
            if not torch.isfinite(feature).all():
                feat_cpu = feature.detach().float().cpu()
                print(f"[NaN Debug][{key}] after aggregate_feature min={feat_cpu.min().item():.6f} max={feat_cpu.max().item():.6f} finite={torch.isfinite(feat_cpu).all().item()}", flush=True)
                raise RuntimeError(f"[{key}] NaN/Inf detected after aggregate_feature")
            emb = self.key_projection_map[key](feature)
            # NaN/Inf check after projection
            if not torch.isfinite(emb).all():
                emb_cpu = emb.detach().float().cpu()
                print(f"[NaN Debug][{key}] after projection min={emb_cpu.min().item():.6f} max={emb_cpu.max().item():.6f} finite={torch.isfinite(emb_cpu).all().item()}", flush=True)
                raise RuntimeError(f"[{key}] NaN/Inf detected after projection")
            assert len(emb.shape) == 3 and emb.shape[0] == B * T and emb.shape[-1] == self.n_emb
            emb = emb.reshape(B,-1,self.n_emb)
            if self.record_feature_stats and emb.requires_grad:
                emb.retain_grad()
                self._last_feature_map[key] = emb
            embeddings.append(emb)

        # process lowdim input
        for key in self.low_dim_keys:
            data = obs_dict[key]
            B, T = data.shape[:2]
            assert B == batch_size
            assert data.shape[2:] == self.key_shape_map[key]
            data = data.reshape(B,T,-1)
            emb = self.key_projection_map[key](data)
            # NaN/Inf check for lowdim branch
            if not torch.isfinite(emb).all():
                emb_cpu = emb.detach().float().cpu()
                print(f"[NaN Debug][{key}] lowdim after projection min={emb_cpu.min().item():.6f} max={emb_cpu.max().item():.6f} finite={torch.isfinite(emb_cpu).all().item()}", flush=True)
                raise RuntimeError(f"[{key}] NaN/Inf detected in lowdim branch after projection")
            assert emb.shape[-1] == self.n_emb
            if self.record_feature_stats and emb.requires_grad:
                emb.retain_grad()
                self._last_feature_map[key] = emb
            embeddings.append(emb)

        # concatenate all features along t
        result = torch.cat(embeddings, dim=1)
        # NaN/Inf check after concat
        if not torch.isfinite(result).all():
            result_cpu = result.detach().float().cpu()
            print(f"[NaN Debug][transformer_obs_encoder] concat result min={result_cpu.min().item():.6f} max={result_cpu.max().item():.6f} finite={torch.isfinite(result_cpu).all().item()}", flush=True)
            raise RuntimeError("[transformer_obs_encoder] NaN/Inf detected after feature concat")
        return result

    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            this_obs = torch.zeros(
                (1, attr['horizon']) + shape, 
                dtype=self.dtype,
                device=self.device)
            example_obs_dict[key] = this_obs
        example_output = self.forward(example_obs_dict)
        assert len(example_output.shape) == 3
        assert example_output.shape[0] == 1

        return example_output.shape


def test():
    import hydra
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    import math
    OmegaConf.register_new_resolver("eval", eval, replace=True)

    with hydra.initialize(version_base=None, config_path='../../../diffusion_policy/config'):
        cfg = hydra.compose(config_name='HOMMI')
        OmegaConf.resolve(cfg)

    encoder = instantiate(cfg.policy.obs_encoder)

    # ------------------------------------------------------------------ #
    # 1. Inspect encoder internal state
    # ------------------------------------------------------------------ #
    print("\n[TEST] rgb_keys           :", encoder.rgb_keys)
    print("[TEST] low_dim_keys       :", encoder.low_dim_keys)
    print("[TEST] key_aggregation_map:", encoder.key_aggregation_map)
    print("[TEST] key_pool_module_map:", list(encoder.key_pool_module_map.keys()))

    # Every rgb key must have an entry in key_aggregation_map
    for key in encoder.rgb_keys:
        assert key in encoder.key_aggregation_map, \
            f"[{key}] missing from key_aggregation_map"

    # Every attn_pool key must have a matching TokenAttentionPool module
    for key, agg in encoder.key_aggregation_map.items():
        if agg == 'attn_pool':
            assert key in encoder.key_pool_module_map, \
                f"[{key}] uses attn_pool but no TokenAttentionPool module found"
            pool_mod = encoder.key_pool_module_map[key]
            assert type(pool_mod).__name__ == 'TokenAttentionPool', \
                f"[{key}] pool module type is {type(pool_mod).__name__}, expected TokenAttentionPool"
            assert hasattr(pool_mod, 'get_last_attention_map'), \
                f"[{key}] pool module missing get_last_attention_map()"
        else:
            assert key not in encoder.key_pool_module_map, \
                f"[{key}] agg={agg} but unexpected pool module exists"

    # ------------------------------------------------------------------ #
    # 2. Dummy forward pass
    # ------------------------------------------------------------------ #
    obs_shape_meta = cfg.task.shape_meta['obs']
    obs_dict = {}
    for key, attr in obs_shape_meta.items():
        shape = tuple(attr['shape'])
        horizon = int(attr['horizon'])
        obs_dict[key] = torch.zeros((1, horizon) + shape, dtype=torch.float32)

    out = encoder(obs_dict)
    print("[TEST] encoder output shape:", tuple(out.shape))
    assert out.ndim == 3 and out.shape[0] == 1 and out.shape[-1] == encoder.n_emb, \
        f"unexpected output shape {out.shape}"

    # ------------------------------------------------------------------ #
    # 3. Attention map sanity check for attn_pool keys
    # ------------------------------------------------------------------ #
    attn_maps = encoder.get_last_attention_maps()
    attn_pool_keys = [k for k, v in encoder.key_aggregation_map.items() if v == 'attn_pool']
    print("[TEST] attn_pool keys      :", attn_pool_keys)
    print("[TEST] cached attn keys    :", list(attn_maps.keys()))

    for key in attn_pool_keys:
        assert key in attn_maps, f"[{key}] missing attention map after forward()"
        amap = attn_maps[key]
        assert amap.ndim == 2, \
            f"[{key}] attention map shape should be [B*T, N_patches], got {amap.shape}"
        N = amap.shape[1]
        g = int(math.isqrt(N))
        assert g * g == N, \
            f"[{key}] N_patches={N} is not a perfect square; cannot reshape to grid"
        print(f"[TEST]   {key}: attn shape={tuple(amap.shape)}, "
              f"grid={g}x{g}, value range=[{amap.min():.4f}, {amap.max():.4f}]")

    print("[TEST] all checks passed.")

    # ------------------------------------------------------------------ #
    # 4. ONNX export  (visualise with Netron — https://netron.app)
    # ------------------------------------------------------------------ #
    import os

    class _EncoderWrapper(nn.Module):
        """Wraps TransformerObsEncoder for ONNX export.

        ONNX requires plain tensor inputs, not dicts.  This wrapper fixes the
        key order as ``rgb_keys + low_dim_keys`` (both already sorted) and
        reconstructs the obs dict internally so the full graph is traced.

        Input names in Netron will match the obs-key names.
        """
        def __init__(self, enc):
            super().__init__()
            self.enc = enc
            # fixed positional order: rgb first, then low-dim (mirrors forward())
            self.keys: list = enc.rgb_keys + enc.low_dim_keys

        def forward(self, *args):
            obs = {k: v for k, v in zip(self.keys, args)}
            return self.enc(obs)

    wrapper = _EncoderWrapper(encoder).eval()
    dummy_inputs = tuple(obs_dict[k] for k in wrapper.keys)
    input_names  = list(wrapper.keys)
    output_names = ["embeddings"]

    onnx_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "encoder_debug.onnx"
    )
    print(f"\n[ONNX] exporting to: {onnx_path}")

    # NOTE:
    # torchvision antialiased resize maps to aten::_upsample_bilinear2d_aa,
    # which is not supported by torch.onnx (opset 17) in some PyTorch builds.
    # For architecture visualization, we bypass input transforms during export.
    _backup_transform_map = {k: v for k, v in encoder.key_transform_map.items()}
    try:
        for k in encoder.rgb_keys:
            encoder.key_transform_map[k] = nn.Identity()

        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                dummy_inputs,
                onnx_path,
                input_names=input_names,
                output_names=output_names,
                opset_version=17,
                do_constant_folding=True,
            )
    finally:
        for k, v in _backup_transform_map.items():
            encoder.key_transform_map[k] = v

    print("[ONNX] export done.")
    print("[ONNX] input keys (positional order):", input_names)
    print("[ONNX] open encoder_debug.onnx with Netron: https://netron.app")


if __name__ == "__main__":
    test()
