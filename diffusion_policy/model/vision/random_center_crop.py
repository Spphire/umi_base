import math
from typing import Sequence, Union
import torch
from torch import Tensor
from torch import nn
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

FillType = Union[int, float, Sequence[int], Sequence[float]]


class RandomCenterCrop(nn.Module):
    """Randomly zooms an image by sampling a center crop ratio."""

    def __init__(
        self,
        ratio_min: float,
        ratio_max: float,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        fill: FillType = 0,
        antialias: bool = True,
        obs_horizon_step: int = 2,
    ) -> None:
        super().__init__()
        if ratio_min <= 0 or ratio_max <= 0:
            raise ValueError("crop ratios must be positive.")
        if ratio_min > ratio_max:
            raise ValueError("ratio_min must be <= ratio_max.")
        if obs_horizon_step <= 0:
            raise ValueError("temporal_group_size must be positive.")
        self.ratio_min = ratio_min
        self.ratio_max = ratio_max
        self.interpolation = interpolation
        self.fill = fill
        self.antialias = antialias
        self.obs_horizon_step = obs_horizon_step

    def forward(self, img: Union[Tensor, "PIL.Image.Image"]) -> Union[Tensor, "PIL.Image.Image"]:
        if isinstance(img, torch.Tensor):
            if img.ndim < 3:
                raise ValueError("Tensor image must have at least 3 dims (C, H, W).")
            total = img.shape[0]
            horizon = self.obs_horizon_step
            if total % horizon != 0:
                raise ValueError(
                    f"Tensor batch of size {total} must be divisible by temporal_group_size {horizon}."
                )
            num_groups = total // horizon
            ratios = torch.empty(num_groups, device=img.device).uniform_(self.ratio_min, self.ratio_max)
            ratios = ratios.repeat_interleave(horizon)
            cropped = [
                self._apply(frame, ratio.item()) for frame, ratio in zip(img, ratios)
            ]
            return torch.stack(cropped, dim=0)

        ratio = torch.empty(1).uniform_(self.ratio_min, self.ratio_max).item()
        return self._apply(img, ratio)

    def _apply(self, img, ratio: float):
        height, width = F.get_image_size(img)
        if ratio <= 1.0:
            crop_h = max(1, math.floor(height * ratio))
            crop_w = max(1, math.floor(width * ratio))
            if crop_h == height and crop_w == width:
                return img
            cropped = F.center_crop(img, [crop_h, crop_w])
            return F.resize(
                cropped,
                [height, width],
                interpolation=self.interpolation,
                antialias=self.antialias,
            )
        scaled_h = max(1, math.floor(height / ratio))
        scaled_w = max(1, math.floor(width / ratio))
        resized = F.resize(
            img,
            [scaled_h, scaled_w],
            interpolation=self.interpolation,
            antialias=self.antialias,
        )
        pad_h = height - scaled_h
        pad_w = width - scaled_w
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        padding = [left, top, right, bottom]
        return F.pad(resized, padding, fill=self.fill)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(ratio_min={self.ratio_min}, "
            f"ratio_max={self.ratio_max}, interpolation={self.interpolation}, "
            f"fill={self.fill}, antialias={self.antialias}, "
            f"obs_horizon_step={self.obs_horizon_step})"
        )
