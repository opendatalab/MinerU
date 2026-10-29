# Copyright (c) Opendatalab. All rights reserved.
"""通过小模型后端的实际设备选择与可用 CPU 数确定 API 服务开放档位。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

from ...types import TIERS, Tier, select_default_task_tier
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
    # 保留 JSON 字段兼容性；自动推荐不再查询或消费显存容量。
    gpu_memory: None = None


@dataclass(frozen=True)
class TierHardware:
    """与设备选型隔离的推荐依据，便于覆盖 CPU 阈值和 GPU 结果。"""

    cpu_count: int
    advanced_gpu: bool = False
    reason: str = "cpu"


def tiers_for_hardware(hardware: TierHardware) -> frozenset[Tier]:
    """GPU 开放全部档位；CPU 至少 4 核开放 Basic，至少 12 核开放 Standard。"""
    if hardware.advanced_gpu:
        return frozenset(TIERS)
    if hardware.cpu_count >= 12:
        return frozenset(TIERS[:3])
    if hardware.cpu_count >= 4:
        return frozenset(TIERS[:2])
    return frozenset(("flash",))


def recommend_tiers(*, vlm_config: VlmConfig | None = None) -> TierRecommendation:
    """复用小模型后端的实际设备结论，不额外探测 GPU 库存、显存或性能。"""
    from ...config import get_config
    from ..vlm.selector import resolve_vlm_engine

    settings = vlm_config if vlm_config is not None else get_config().model.vlm
    settings.validate_environment()
    small_backend = resolve_small_model_backend()
    vlm_engine = "http-client" if settings.server_url else resolve_vlm_engine(settings.engine)
    warnings: list[str] = []
    try:
        count = effective_cpu_count()
    except OSError as error:
        count = 1
        warnings.append(f"CPU availability discovery failed: {error}")

    device = "cpu"
    if small_backend == "torch":
        choice = resolve_torch_device()
        hardware = TierHardware(count, choice.accelerated, choice.reason)
        device = choice.device
        calibration = CalibrationDecision("skipped", "torch_backend")
        if choice.warning:
            warnings.append(choice.warning)
    else:
        logger.info("Automatic API tier selection is resolving the ONNX device before finalizing tiers")
        try:
            from .onnx_device import cached_onnx_device, resolve_onnx_device

            # 与 llama 自动模式复用相同入口；缓存、显式配置和免校准规则均由 ONNX 选型负责。
            cached = cached_onnx_device()
            onnx_choice = resolve_onnx_device()
            accelerated = onnx_choice.device == "gpu"
            device = f"{onnx_choice.backend}:gpu" if accelerated else "cpu"
            reason = "ocr_calibrated_gpu" if accelerated and onnx_choice.reason == "rec_calibration" else onnx_choice.reason
            hardware = TierHardware(count, accelerated, reason)
            try:
                evidence = inspect_calibration_record((onnx_choice.adapter,) if onnx_choice.adapter is not None else None)
            except Exception as error:
                evidence = None
                warnings.append(f"ONNX calibration diagnostics unavailable: {error}")
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
            elif onnx_choice.reason == "configured_cpu" and evidence is not None:
                status = "reused" if evidence.matches_environment and evidence.decision_valid else "skipped"
            calibration = CalibrationDecision(status, onnx_choice.reason, evidence)
        except Exception as error:
            logger.warning("Startup ONNX selection unavailable; using CPU recommendation rules: {}", error)
            warnings.append(f"ONNX selection unavailable: {error}")
            calibration = CalibrationDecision("failed", "selection_unavailable")
            hardware = TierHardware(count, reason="selection_unavailable")

    tiers = tiers_for_hardware(hardware)
    ordered = tuple(tier for tier in TIERS if tier in tiers)
    logger.info(
        "Recommended API tiers: {} (available CPUs: {}, reason: {})",
        ",".join(ordered),
        hardware.cpu_count,
        hardware.reason,
    )
    default = select_default_task_tier(tiers)
    assert default is not None  # 硬件推荐始终包含 Flash，必定存在请求默认档位。
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
    "recommend_tiers",
    "tiers_for_hardware",
]
