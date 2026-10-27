"""通过静态设备事实与已有 OCR 校准，推荐 API 服务开放的档位集合。"""

from __future__ import annotations

from dataclasses import dataclass
import os
import sys

from loguru import logger

from ...types import TIERS, Tier
from .cpu_threads import effective_cpu_count


@dataclass(frozen=True)
class TierHardware:
    """与设备探测隔离的推荐依据，便于完整覆盖各类硬件组合。"""

    cpu_count: int
    advanced_gpu: bool = False
    reason: str = "cpu"


def tiers_for_hardware(hardware: TierHardware) -> frozenset[Tier]:
    """按已确定的工程阈值计算集合，达标 GPU 同时开放所有较低档位。"""
    if hardware.advanced_gpu:
        return frozenset(TIERS)
    if hardware.cpu_count >= 4:
        return frozenset(("flash", "basic", "standard"))
    return frozenset(("flash",))


def _apple_metal_available() -> bool:
    """仅在 Apple Silicon 探测时惰性查询 Metal，不加载推理模型。"""
    from .platform import is_apple_silicon_cpu

    if not is_apple_silicon_cpu():
        return False
    import torch

    return torch.backends.mps.is_available()


def _advanced_gpu_reason() -> str | None:
    """复用可见设备及兼容策略，集显只读取证据而不触发自动校准。"""
    backend = os.getenv("MINERU_LLAMA_CPP_BACKEND", "auto")
    if backend == "cpu":
        return None
    if sys.platform == "darwin":
        # ggml 的 Metal GPU 类型不能区分独显与集显，只采用已确认的 Apple Silicon 规则。
        return "apple_silicon_metal" if backend in {"auto", "metal"} and _apple_metal_available() else None
    if sys.platform not in {"linux", "win32"}:
        return None

    from .llama_device import discover_llama_devices

    devices = tuple(d for d in discover_llama_devices() if backend == "auto" or d.backend == backend)
    if any(not device.integrated for device in devices):
        return "discrete_gpu"
    if not devices:
        return None

    from .onnx_calibration_cache import calibrated_gpu_is_faster
    from .onnx_device import ONNX_GPU_TIME_RATIO_THRESHOLDS

    if sys.platform == "win32":
        from .windows_gpu import discover_gpus, preferred_gpu

        adapter = preferred_gpu(discover_gpus())
        threshold = ONNX_GPU_TIME_RATIO_THRESHOLDS["dml"]
    else:
        from .webgpu import discover_webgpu_devices, select_webgpu_device

        adapter = select_webgpu_device(discover_webgpu_devices())
        threshold = ONNX_GPU_TIME_RATIO_THRESHOLDS["webgpu"]
    if adapter is None or threshold is None:
        return None
    matches = [d for d in devices if (d.vendor_id, d.device_id) == (adapter.vendor_id, adapter.device_id)]
    if len(matches) != 1:
        logger.info("GPU tier recommendation uses CPU rules: OCR adapter identity is not uniquely matched")
        return None
    if calibrated_gpu_is_faster(adapter, threshold):
        return "ocr_calibrated_gpu"
    logger.info("GPU tier recommendation uses CPU rules: no matching valid OCR evidence of a faster GPU")
    return None


def detect_tier_hardware() -> TierHardware:
    """探测失败时保守使用 CPU 条件，推荐过程不下载模型或保存配置。"""
    try:
        count = effective_cpu_count()
    except OSError as error:
        logger.warning("CPU availability discovery failed: {}", error)
        count = 1
    try:
        reason = _advanced_gpu_reason()
    except Exception as error:
        logger.warning("GPU tier recommendation unavailable; using CPU rules: {}", error)
        return TierHardware(count, reason="gpu_discovery_unavailable")
    return TierHardware(count, bool(reason), reason or "cpu")


def recommend_tiers() -> frozenset[Tier]:
    """每次调用读取当前硬件事实；API 服务只在启动阶段调用一次。"""
    hardware = detect_tier_hardware()
    tiers = tiers_for_hardware(hardware)
    logger.info(
        "Recommended API tiers: {} (available CPUs: {}, reason: {})",
        ",".join(tier for tier in TIERS if tier in tiers),
        hardware.cpu_count,
        hardware.reason,
    )
    return tiers


__all__ = ["TierHardware", "detect_tier_hardware", "recommend_tiers", "tiers_for_hardware"]
