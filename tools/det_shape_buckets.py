"""仅供冻结语料实验使用的 N/H/W 归并；不改变生产 det 的预处理和后处理。"""

from __future__ import annotations

import math
import time
from typing import Any

SHAPE_STRATEGIES: dict[str, tuple[int | None, int | None]] = {
    "F": (1, None),
    "G": (4, None),
    "H": (None, 64),
    "I": (None, 128),
}


def project_shape(shape: list[int], policy: str) -> dict[str, Any]:
    """预先计算真实送入网络的形状、拆批数量和额外像素工作量。"""
    n, c, height, width = shape
    batch, alignment = SHAPE_STRATEGIES.get(policy, (None, None))
    target_batch = batch or n
    target_height = math.ceil(height / alignment) * alignment if alignment else height
    target_width = math.ceil(width / alignment) * alignment if alignment else width
    chunks = math.ceil(n / target_batch)
    shapes = [[target_batch, c, target_height, target_width] for _ in range(chunks)]
    return {
        "original_shape": shape,
        "network_shapes": shapes,
        "real_images": n,
        "processed_images": target_batch * chunks,
        "original_pixels": n * height * width,
        "processed_pixels": target_batch * chunks * target_height * target_width,
        "spatial_alignment": alignment,
        "batch_bucket": batch,
        "spatial_pad_normalized_value": 0.0 if alignment else None,
        "batch_pad": "repeat_last_real_image" if target_batch * chunks > n else None,
    }


def process_bucketed_batch(detector: Any, batch_items: list[Any], policy: str) -> tuple[list[Any], float]:
    """保留原图 resize/坐标，只在网络入口拆批补齐，并在 CPU 恢复原概率图后执行一次 DB 后处理。"""
    import numpy as np
    import torch

    if not batch_items:
        return [], 0.0
    started = time.perf_counter()
    images = np.ascontiguousarray(np.stack([item[1] for item in batch_items]))
    shape = list(images.shape)
    projection = project_shape(shape, policy)
    n, _, height, width = shape
    batch, _, padded_height, padded_width = projection["network_shapes"][0]
    maps = []
    padding_seconds = 0.0
    for offset in range(0, n, batch):
        padding_started = time.perf_counter()
        real = images[offset : offset + batch]
        count = len(real)
        padded = np.zeros((batch, shape[1], padded_height, padded_width), dtype=images.dtype)
        padded[:count, :, :height, :width] = real
        if count < batch:
            # eval 网络不存在跨样本归一化，补齐样本的输出会在 CPU 丢弃。
            padded[count:] = padded[count - 1]
        padding_seconds += time.perf_counter() - padding_started
        with torch.inference_mode():
            inputs = detector._to_inference_dtype(torch.from_numpy(padded).to(detector.device))
            outputs = detector.net(inputs)
        predictions = detector._build_det_preds(outputs)["maps"]
        if predictions.shape[:2] != (batch, 1) or predictions.shape[2:] != (padded_height, padded_width):
            raise ValueError(f"归并后的概率图形状不符合 DBHead 约定：{predictions.shape}")
        # 先回传 CPU，再裁切，避免额外 MPS slice 图缓存和错误的后处理缩放。
        maps.append(np.ascontiguousarray(predictions[:count, :, :height, :width]))
        del predictions, inputs, outputs, padded
    concatenation_started = time.perf_counter()
    restored = np.concatenate(maps, axis=0)
    padding_seconds += time.perf_counter() - concatenation_started
    detector.bucket_padding_seconds += padding_seconds
    if restored.shape != (n, 1, height, width):
        raise ValueError("归并后的概率图未恢复原始 N/H/W")
    if detector.capture_maps:
        detector.last_maps = restored
    boxes = detector._postprocess_det_batch(
        {"maps": restored}, np.stack([item[2] for item in batch_items]), [item[3] for item in batch_items]
    )
    elapsed = time.perf_counter() - started
    return [(box, elapsed / n) for box in boxes], elapsed
