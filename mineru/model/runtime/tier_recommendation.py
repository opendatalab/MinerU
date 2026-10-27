"""通过硬件初选与启动阶段的 OCR 设备校准，确定 API 服务开放档位。"""

from __future__ import annotations

from dataclasses import dataclass
import os
import sys
from typing import TYPE_CHECKING

from loguru import logger

from ...types import TIERS, Tier
from .cpu_threads import effective_cpu_count
from .device import resolve_small_model_backend, resolve_torch_device
from .onnx_calibration_cache import CalibrationInfo, inspect_calibration_record

if TYPE_CHECKING:
    from ...config import VlmConfig


@dataclass(frozen=True)
class CalibrationDecision:
    """描述本次推荐中的校准状态，避免把设备可用性当成实测胜出。"""

    status: str
    reason: str
    evidence: CalibrationInfo | None = None


@dataclass(frozen=True)
class TierRecommendation:
    """API 自动启动和硬件命令共同消费的推荐报告。"""

    ceiling: Tier
    pdf_image_tiers: tuple[Tier, ...]
    default_request_tier: Tier
    reason: str
    cpu_count: int
    small_backend: str
    vlm_engine: str
    device: str
    calibration: CalibrationDecision
    warnings: tuple[str, ...] = ()
    schema_version: int = 1


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
    """只读探测当前硬件与校准证据；失败时使用 CPU 条件，不下载模型或保存配置。"""
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


def recommend_tiers(*, vlm_config: VlmConfig | None = None) -> TierRecommendation:
    """先分流实际小模型后端；Torch 免测速，ONNX 初选达标后同步完成既有选型。"""
    from ...config import get_config
    from ..vlm.selector import resolve_vlm_engine

    settings = vlm_config if vlm_config is not None else get_config().model.vlm
    settings.validate_environment()
    small_backend = resolve_small_model_backend()
    vlm_engine = "http-client" if settings.server_url else resolve_vlm_engine(settings.engine)
    warnings: list[str] = []
    device = "cpu"
    calibration = CalibrationDecision("skipped", "not_required")
    if small_backend == "torch":
        choice = resolve_torch_device()
        try:
            count = effective_cpu_count()
        except OSError as error:
            count = 1
            warnings.append(f"CPU availability discovery failed: {error}")
        hardware = TierHardware(count, choice.accelerated, choice.reason)
        device = choice.device
        calibration = CalibrationDecision("skipped", "torch_backend")
        if choice.warning:
            warnings.append(choice.warning)
    else:
        hardware = detect_tier_hardware()
    tiers = tiers_for_hardware(hardware)
    if small_backend == "onnx" and "basic" in tiers and sys.platform in {"win32", "linux"}:
        logger.info("Automatic API tier selection is resolving the ONNX device before finalizing tiers")
        try:
            from .onnx_device import cached_onnx_device, resolve_onnx_device

            # 复用模型运行时的配置、锁和缓存；显式设备选择及不兼容 GPU 不强制测速。
            cached = cached_onnx_device()
            onnx_choice = resolve_onnx_device()
            device = f"{onnx_choice.backend}:gpu" if onnx_choice.device == "gpu" else "cpu"
            evidence = inspect_calibration_record((onnx_choice.adapter,) if onnx_choice.adapter is not None else None)
            status = "skipped"
            if onnx_choice.reason == "rec_calibration":
                status = "reused" if cached is not None else "completed"
            elif onnx_choice.reason in {
                "selection_failed",
                "calibration_unstable",
                "directml_unavailable",
                "calibration_pending",
            }:
                status = "failed"
                warnings.append(f"ONNX selection used CPU: {onnx_choice.reason}")
            elif onnx_choice.reason in {"configured_cpu", "configured_gpu"}:
                status = "reused" if evidence.matches_environment and evidence.decision_valid else "skipped"
            calibration = CalibrationDecision(status, onnx_choice.reason, evidence)
        except Exception as error:
            logger.warning("Startup ONNX selection unavailable; recommending tiers from current evidence: {}", error)
            warnings.append(f"ONNX selection unavailable: {error}")
            calibration = CalibrationDecision("failed", "selection_unavailable")
        # 校准会写入新证据或撤销旧证据，必须在发布能力前重新计算，不能等到下次启动。
        hardware = detect_tier_hardware()
        tiers = tiers_for_hardware(hardware)
    if hardware.reason == "gpu_discovery_unavailable":
        warnings.append("GPU discovery unavailable; using CPU recommendation rules")
    logger.info(
        "Recommended API tiers: {} (available CPUs: {}, reason: {})",
        ",".join(tier for tier in TIERS if tier in tiers),
        hardware.cpu_count,
        hardware.reason,
    )
    ordered = tuple(tier for tier in TIERS if tier in tiers)
    default = next(tier for tier in ("standard", "basic", "flash") if tier in tiers)
    return TierRecommendation(
        ordered[-1],
        ordered,
        default,
        hardware.reason,
        hardware.cpu_count,
        small_backend,
        vlm_engine,
        device,
        calibration,
        tuple(warnings),
    )


__all__ = [
    "CalibrationDecision",
    "TierHardware",
    "TierRecommendation",
    "detect_tier_hardware",
    "recommend_tiers",
    "tiers_for_hardware",
]
