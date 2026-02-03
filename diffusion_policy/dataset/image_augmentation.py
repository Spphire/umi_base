# -*- coding: utf-8 -*-
"""
Image Data Augmentation Pipeline

Scenario:
- Cross-camera (Quest -> Hikvision)
- MP4 / RTSP compression involved
- Mechanical arm manipulating objects
- Incremental Learning (IL)
"""

import cv2
import numpy as np
import random
import os


def slight_rotate(img, max_angle=8):
    h, w = img.shape[:2]
    angle = random.uniform(-max_angle, max_angle)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )


def adjust_brightness_contrast_gamma(
    img,
    brightness_range=(-20, 20),
    contrast_range=(0.9, 1.1),
    gamma_range=(0.9, 1.1)
):
    img = img.astype(np.float32)
    brightness = random.uniform(*brightness_range)
    contrast = random.uniform(*contrast_range)
    gamma = random.uniform(*gamma_range)
    img = img * contrast + brightness
    img = np.clip(img, 0, 255)
    img = np.power(img / 255.0, gamma) * 255.0
    return img.astype(np.uint8)


def channel_gain(img, gain_range=(0.95, 1.05)):
    img = img.astype(np.float32)
    gains = np.random.uniform(gain_range[0], gain_range[1], size=3)
    for c in range(3):
        img[:, :, c] *= gains[c]
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


def slight_gaussian_blur(img, kernel_choices=(3, 5)):
    k = random.choice(kernel_choices)
    return cv2.GaussianBlur(img, (k, k), 0)


def jpeg_compression(img, quality_range=(50, 90)):
    quality = random.randint(*quality_range)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, enc = cv2.imencode(".jpg", img, encode_param)
    if not success:
        return img
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def slight_translation(img, max_shift_ratio=0.05):
    h, w = img.shape[:2]
    max_dx = int(w * max_shift_ratio)
    max_dy = int(h * max_shift_ratio)
    dx = random.randint(-max_dx, max_dx)
    dy = random.randint(-max_dy, max_dy)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(
        img, M, (w, h),
        borderMode=cv2.BORDER_REFLECT_101
    )

def _augment_single_frame(img):
    if random.random() < 0.8:
        img = slight_rotate(img)
    if random.random() < 0.3:
        img = slight_translation(img)
    img = adjust_brightness_contrast_gamma(img)
    if random.random() < 0.5:
        img = channel_gain(img)
    if random.random() < 0.3:
        img = slight_gaussian_blur(img)
    if random.random() < 0.3:
        img = jpeg_compression(img)
    return img

def apply_image_augmentation(img):
    """
    img: HWC uint8 或 THWC uint8
    返回同维度 uint8
    """

    # 如果是多帧，逐帧增强
    if img.ndim == 4:  # T H W C
        return np.stack([_augment_single_frame(f) for f in img], axis=0)
    else:  # 单帧 H W C
        return _augment_single_frame(img)
    
def batch_resize_thwc(
    imgs,
    target_size=224,
    mode="pad"  # "pad" or "crop"
):
    """
    imgs: THWC uint8
    mode:
        - "pad"  : 等比缩放 + 补黑边 (letterbox)
        - "crop" : 等比缩放 + 中心裁剪
    return: THWC uint8 (T, target_size, target_size, C)
    """
    assert imgs.ndim == 4, "Input must be THWC"
    T, H, W, C = imgs.shape

    out = np.zeros((T, target_size, target_size, C), dtype=imgs.dtype)

    for t in range(T):
        img = imgs[t]

        scale = target_size / min(H, W) if mode == "crop" else target_size / max(H, W)
        new_w = int(round(W * scale))
        new_h = int(round(H * scale))

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        if mode == "pad":
            # letterbox: 居中补黑边
            pad_x = (target_size - new_w) // 2
            pad_y = (target_size - new_h) // 2

            out[t, pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        elif mode == "crop":
            # center crop
            start_x = (new_w - target_size) // 2
            start_y = (new_h - target_size) // 2

            out[t] = resized[
                start_y:start_y + target_size,
                start_x:start_x + target_size
            ]
        else:
            raise ValueError(f"Unknown mode: {mode}")

    return out



if __name__ == "__main__":
    input_image_path = "test_input.jpg"
    output_dir = "augmentation_results"
    os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(input_image_path)
    if img is None:
        raise FileNotFoundError(f"Failed to load image: {input_image_path}")

    for i in range(5):
        aug_img = apply_image_augmentation(img)
        cv2.imwrite(
            os.path.join(output_dir, f"augmented_{i}.jpg"),
            aug_img
        )

    print("Augmentation test finished. Results saved to:", output_dir)
