"""仅供本机 CPU det 实验使用的真实尾批拆分，不修改生产调用路径。"""

from __future__ import annotations

import time
from typing import Any

CPU_MICROBATCH_STRATEGIES: dict[str, tuple[int, int]] = {
    "O": (16, 1),
    "P": (16, 2),
    "Q": (16, 4),
    "R": (16, 8),
    "S": (1, 2),
    "T": (1, 4),
    "U": (1, 8),
}


def project_cpu_microbatch(shape: list[int], policy: str) -> dict[str, Any]:
    """明确计算阈值分流及每个真实 microbatch 的 NCHW，尾批不复制虚假样本。"""
    threshold, cap = CPU_MICROBATCH_STRATEGIES[policy]
    n, channels, height, width = shape
    if any(value < 1 for value in shape):
        raise ValueError("CPU microbatch 输入形状必须为正")
    split = n >= threshold and n > cap
    shapes = [[min(cap, n - offset), channels, height, width] for offset in range(0, n, cap)] if split else [shape]
    return {
        "original_shape": shape,
        "network_shapes": shapes,
        "real_images": n,
        "processed_images": n,
        "original_pixels": n * height * width,
        "processed_pixels": n * height * width,
        "spatial_alignment": None,
        "spatial_pad_normalized_value": None,
        "batch_pad": None,
        "batch_bucket": cap,
        "split_threshold": threshold,
        "split_required": split,
    }


def process_cpu_microbatch(detector: Any, batch_items: list[Any], policy: str) -> tuple[list[Any], float]:
    """按真实 N 切连续数组视图、串行前向，恢复原概率图顺序后执行一次原后处理。"""
    import numpy as np
    import torch

    if not batch_items:
        return [], 0.0
    started = time.perf_counter()
    images = np.ascontiguousarray(np.stack([item[1] for item in batch_items]))
    n, _, height, width = images.shape
    projection = project_cpu_microbatch(list(images.shape), policy)
    if not projection["split_required"] or detector.device != "cpu":
        raise ValueError("拆分函数只接受已满足阈值的 CPU 输入；未拆批时应直接调用原实现")
    cap = projection["batch_bucket"]
    maps = []
    organization_seconds = 0.0
    for offset in range(0, n, cap):
        preparation_started = time.perf_counter()
        chunk = np.ascontiguousarray(images[offset : offset + cap])
        organization_seconds += time.perf_counter() - preparation_started
        with torch.inference_mode():
            inputs = detector._to_inference_dtype(torch.from_numpy(chunk).to(detector.device))
            outputs = detector.net(inputs)
        predictions = detector._build_det_preds(outputs)["maps"]
        if predictions.shape != (len(chunk), 1, height, width):
            raise ValueError(f"CPU microbatch 概率图形状异常：{predictions.shape}")
        maps.append(predictions)
        del predictions, inputs, outputs, chunk
    concatenation_started = time.perf_counter()
    restored = np.concatenate(maps, axis=0)
    organization_seconds += time.perf_counter() - concatenation_started
    # 复用实验计时字段记录拆批组织成本；本策略没有 padding 操作。
    detector.bucket_padding_seconds += organization_seconds
    if restored.shape != (n, 1, height, width):
        raise ValueError("CPU microbatch 未恢复原 N/H/W")
    if detector.capture_maps:
        detector.last_maps = restored
    boxes = detector._postprocess_det_batch(
        {"maps": restored}, np.stack([item[2] for item in batch_items]), [item[3] for item in batch_items]
    )
    elapsed = time.perf_counter() - started
    return [(box, elapsed / n) for box in boxes], elapsed
