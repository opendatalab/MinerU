# Copyright (c) Opendatalab. All rights reserved.
"""通过现有设备选型与选定 GPU 的内存预算，确定 API 服务开放档位。"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from loguru import logger

from ...types import TIERS, Tier, select_default_task_tier
from .cpu_threads import effective_cpu_count
from .device import resolve_small_model_backend, resolve_torch_device
from .gpu import is_discrete_gpu, match_gpu_identities
from .gpu_memory import GpuMemory, native_gpu_memory, torch_gpu_memory
from .onnx_calibration_cache import CalibrationInfo, inspect_calibration_record

if TYPE_CHECKING:
    from ...config import VlmConfig
    from .onnx_device import OnnxSelection


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
    gpu_memory: GpuMemory | None = None


@dataclass(frozen=True)
class TierHardware:
    """与设备探测隔离的推荐依据，便于完整覆盖各类硬件组合。"""

    cpu_count: int
    advanced_gpu: bool = False
    reason: str = "cpu"
    gpu_memory: GpuMemory | None = None


def tiers_for_hardware(hardware: TierHardware) -> frozenset[Tier]:
    """按已确定的工程阈值计算集合，达标 GPU 同时开放所有较低档位。"""
    if hardware.advanced_gpu and hardware.gpu_memory is not None and hardware.gpu_memory.sufficient:
        return frozenset(TIERS)
    if hardware.cpu_count >= 4:
        return frozenset(("flash", "basic"))
    return frozenset(("flash",))


def _apple_metal_available() -> bool:
    """仅在 Apple Silicon 探测时惰性查询 Metal，不加载推理模型。"""
    from .platform import is_apple_silicon_cpu

    if not is_apple_silicon_cpu():
        return False
    import torch

    return torch.backends.mps.is_available()


def _gpu_hardware(count: int, confirmed: bool, reason: str, memory: GpuMemory) -> TierHardware:
    """把设备确认和容量阈值合并为一次可解释的推荐。"""
    if confirmed and not memory.sufficient:
        reason = "gpu_memory_unknown" if memory.memory_bytes is None else "gpu_memory_insufficient"
    return TierHardware(count, confirmed and memory.sufficient, reason, memory)


def _onnx_gpu_hardware(count: int, choice: OnnxSelection | None = None) -> TierHardware:
    """初选只读候选和旧证据；最终推荐严格采用本次选型，CPU 或失败不能被库存升级。"""
    from ...config import get_config

    if choice is not None and (choice.device != "gpu" or choice.adapter is None):
        return TierHardware(count, reason=choice.reason)
    backend = os.getenv("MINERU_LLAMA_CPP_BACKEND", "auto")
    if backend == "cpu":
        return TierHardware(count, reason="configured_cpu")
    if sys.platform == "darwin":
        if backend in {"auto", "metal"} and _apple_metal_available():
            return _gpu_hardware(count, True, "apple_silicon_metal", torch_gpu_memory("mps"))
        return TierHardware(count)
    if sys.platform not in {"linux", "win32"}:
        return TierHardware(count)
    if choice is None and get_config().model.onnx_device == "cpu":
        return TierHardware(count, reason="configured_cpu")

    from .llama_device import discover_llama_devices

    devices = tuple(d for d in discover_llama_devices() if backend == "auto" or d.backend == backend)
    if not devices:
        return TierHardware(count, reason="no_visible_gpu")

    if choice is not None:
        adapter = choice.adapter
    elif sys.platform == "win32":
        from .windows_gpu import discover_gpus, preferred_gpu

        adapter = preferred_gpu(discover_gpus())
    else:
        from .webgpu import discover_webgpu_devices, select_webgpu_device

        adapter = select_webgpu_device(discover_webgpu_devices())
    if adapter is None:
        return TierHardware(count, reason="no_eligible_gpu")
    matched = match_gpu_identities((adapter.identity,), [device.identity for device in devices])[0]
    if matched is None:
        return TierHardware(count, reason="gpu_identity_unconfirmed")
    memory = native_gpu_memory(adapter.identity, device=f"{choice.backend if choice else 'onnx'}:gpu")
    if choice is not None:
        # 性能判断属于 ONNX 自动选型；已选 GPU 仅需满足内存门槛，不再要求历史校准证据。
        reason = "ocr_calibrated_gpu" if choice.reason == "rec_calibration" else choice.reason
        return _gpu_hardware(count, True, reason, memory)
    if get_config().model.onnx_device == "gpu":
        # 初选保留内存候选，让低 CPU 配额也能进入实际设备选型，配置本身不确认设备可用。
        return TierHardware(count, reason="configured_gpu", gpu_memory=memory)

    from .onnx_calibration_cache import calibrated_gpu_is_faster
    from .onnx_device import ONNX_GPU_TIME_RATIO_THRESHOLDS
    from .webgpu import WebGPUDevice

    threshold = ONNX_GPU_TIME_RATIO_THRESHOLDS["webgpu" if sys.platform == "linux" else "dml"]
    if threshold is None:
        return TierHardware(count, reason="no_eligible_gpu")
    discrete = (is_discrete_gpu(adapter) or is_discrete_gpu(devices[matched])) and (
        not isinstance(adapter, WebGPUDevice) or adapter.vulkan_device == (adapter.vendor_id, adapter.device_id)
    )
    calibrated = calibrated_gpu_is_faster(adapter, threshold)
    reason = "discrete_gpu" if discrete else "ocr_calibrated_gpu" if calibrated else "gpu_calibration_required"
    return _gpu_hardware(count, discrete or calibrated, reason, memory)


def detect_tier_hardware() -> TierHardware:
    """只读探测当前硬件与校准证据；失败时使用 CPU 条件，不下载模型或保存配置。"""
    try:
        count = effective_cpu_count()
    except OSError as error:
        logger.warning("CPU availability discovery failed: {}", error)
        count = 1
    try:
        return _onnx_gpu_hardware(count)
    except Exception as error:
        logger.warning("GPU tier recommendation unavailable; using CPU rules: {}", error)
        return TierHardware(count, reason="gpu_discovery_unavailable")


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
        hardware = (
            _gpu_hardware(count, True, choice.reason, torch_gpu_memory(choice.device))
            if choice.accelerated
            else TierHardware(count, reason=choice.reason)
        )
        device = choice.device
        calibration = CalibrationDecision("skipped", "torch_backend")
        if choice.warning:
            warnings.append(choice.warning)
    else:
        hardware = detect_tier_hardware()
        if hardware.gpu_memory is not None and hardware.gpu_memory.device == "mps":
            device = "mps"
            calibration = CalibrationDecision("skipped", "apple_silicon_metal")
    tiers = tiers_for_hardware(hardware)
    gpu_candidate = hardware.gpu_memory is not None and hardware.gpu_memory.sufficient
    if small_backend == "onnx" and ("basic" in tiers or gpu_candidate) and sys.platform in {"win32", "linux"}:
        logger.info("Automatic API tier selection is resolving the ONNX device before finalizing tiers")
        try:
            from .onnx_device import cached_onnx_device, resolve_onnx_device

            # 复用模型运行时的配置、锁和缓存；显式设备选择及不兼容 GPU 不强制测速。
            cached = cached_onnx_device()
            onnx_choice = resolve_onnx_device()
            device = f"{onnx_choice.backend}:gpu" if onnx_choice.device == "gpu" else "cpu"
            hardware = _onnx_gpu_hardware(hardware.cpu_count, onnx_choice)
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
            hardware = replace(hardware, advanced_gpu=False, reason="selection_unavailable")
        # 以实际选择形成最终结果，不能重新扫描库存覆盖本次 CPU 回退。
        tiers = tiers_for_hardware(hardware)
    if hardware.reason == "gpu_discovery_unavailable":
        warnings.append("GPU discovery unavailable; using CPU recommendation rules")
    if hardware.gpu_memory is not None and hardware.gpu_memory.warning:
        warnings.append(hardware.gpu_memory.warning)
    if hardware.reason == "gpu_identity_unconfirmed":
        warnings.append("GPU device identity could not be uniquely matched; using CPU recommendation rules")
    logger.info(
        "Recommended API tiers: {} (available CPUs: {}, reason: {})",
        ",".join(tier for tier in TIERS if tier in tiers),
        hardware.cpu_count,
        hardware.reason,
    )
    ordered = tuple(tier for tier in TIERS if tier in tiers)
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
        gpu_memory=hardware.gpu_memory,
    )


__all__ = [
    "CalibrationDecision",
    "TierHardware",
    "TierRecommendation",
    "detect_tier_hardware",
    "recommend_tiers",
    "tiers_for_hardware",
]
