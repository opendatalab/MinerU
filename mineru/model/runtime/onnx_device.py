# Copyright (c) Opendatalab. All rights reserved.
"""Process-local ONNX selection, with a single persisted cpu/gpu preference."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys
import threading
from typing import Literal

from filelock import FileLock
from loguru import logger

from .windows_gpu import GpuAdapter, discover_gpus, preferred_gpu
from .webgpu import WebGPUDevice, discover_webgpu_devices, select_webgpu_device
from .gpu import is_discrete_gpu

# Require at least 20% lower GPU inference time; simplified to 0.8 at the user's request.
# Evidence: docs/next/onnx-selection-validation.md. Each backend can be calibrated independently.
ONNX_GPU_TIME_RATIO_THRESHOLDS: dict[str, float | None] = {"dml": 0.8, "webgpu": 0.8}
METACOMMANDS: dict[str, dict[str, bool]] = {
    "intel": {"layout": False, "mfr": True, "ocr_rec": False, "unet": False},
    "amd": {"layout": True, "mfr": True, "ocr_rec": True, "unet": True},
    "nvidia": {"layout": True, "mfr": True, "ocr_rec": False, "unet": False},
}


@dataclass(frozen=True)
class OnnxSelection:
    """Immutable actual route. Adapter identity is valid only in this process."""

    device: Literal["cpu", "gpu"] = "cpu"
    adapter: GpuAdapter | WebGPUDevice | None = None
    reason: str = "cpu"
    backend: Literal["dml", "webgpu"] | None = None


_lock = threading.RLock()
_selections: dict[tuple[int, str, str, str, bool], OnnxSelection] = {}


def cached_onnx_device(*, allow_download: bool = True) -> OnnxSelection | None:
    """只读查询当前配置的进程内选择，用于区分本次选型与已缓存结果。"""
    from ...config import get_config, get_config_file_path, get_config_source

    key = (
        os.getpid(),
        get_config_file_path(),
        get_config().model.onnx_device,
        get_config_source("model.onnx_device"),
        allow_download,
    )
    with _lock:
        return _selections.get(key)


def metacommands_enabled(vendor: str, model_kind: str) -> bool | None:
    """None leaves an unknown vendor on the ORT default."""
    return METACOMMANDS.get(vendor, {}).get(model_kind)


def choose_gpu(cpu_elapsed: float, gpu_elapsed: float, threshold: float) -> bool:
    """Compare valid elapsed times with the calibrated, conservatively bounded threshold."""
    import math

    if not all(math.isfinite(value) and value > 0 for value in (cpu_elapsed, gpu_elapsed, threshold)):
        raise ValueError("Calibration times and threshold must be finite and positive")
    if threshold >= 1:
        raise ValueError("GPU threshold must be below 1")
    return gpu_elapsed / cpu_elapsed <= threshold


def inspect_onnx_candidate(
    requested: str,
    *,
    backend: Literal["dml", "webgpu"] | None,
    adapter: GpuAdapter | WebGPUDevice | None = None,
    provider_available: bool = True,
    has_devices: bool = False,
) -> OnnxSelection:
    """按已采集的库存解释免校准路径；需测速时只报告待校准，不探测、下载或写配置。"""
    if requested == "cpu":
        return OnnxSelection(reason="configured_cpu")
    if backend is None:
        return OnnxSelection(reason="unsupported_platform")
    if backend == "dml" and not provider_available:
        return OnnxSelection(reason="directml_unavailable")
    if backend == "webgpu" and has_devices and adapter is None:
        return OnnxSelection(reason="webgpu_architecture_policy", backend=backend)
    if adapter is None:
        return OnnxSelection(reason="no_hardware_gpu")
    # Linux 只有唯一确认的 Vulkan 独显且会话带物理约束，才与 DML 独显一样免校准。
    discrete = is_discrete_gpu(adapter) and (
        not isinstance(adapter, WebGPUDevice) or adapter.vulkan_device == (adapter.vendor_id, adapter.device_id)
    )
    if requested == "gpu" or discrete:
        return OnnxSelection("gpu", adapter, "configured_gpu" if requested == "gpu" else "discrete_gpu", backend)
    if ONNX_GPU_TIME_RATIO_THRESHOLDS[backend] is None:
        return OnnxSelection(reason="calibration_pending")
    return OnnxSelection(reason="calibration_required", backend=backend)


def _select(
    requested: str, intra_op_num_threads: int, inter_op_num_threads: int, *, allow_download: bool = True
) -> tuple[OnnxSelection, bool]:
    """复用只读候选规则，只有需要测速时才进入模型准备和校准流程。"""
    backend: Literal["dml", "webgpu"] | None = (
        "webgpu" if sys.platform == "linux" else "dml" if sys.platform == "win32" else None
    )
    if requested == "cpu" or backend is None:
        if requested == "gpu":
            logger.warning("ONNX GPU selection requires Windows DirectML or Linux WebGPU; using CPU")
        return inspect_onnx_candidate(requested, backend=backend), False
    has_devices = False
    if backend == "webgpu":
        devices = discover_webgpu_devices()
        has_devices = bool(devices)
        adapter = select_webgpu_device(devices)
        if devices and adapter is None:
            logger.info("Linux WebGPU device policy selected CPU; no eligible visible GPU")
    else:
        import onnxruntime as ort

        if "DmlExecutionProvider" not in ort.get_available_providers():
            logger.warning("DirectML provider unavailable; using CPU without saving a performance decision")
            return inspect_onnx_candidate(requested, backend=backend, provider_available=False), False
        adapter = preferred_gpu(discover_gpus(classify=requested == "auto"))
    selection = inspect_onnx_candidate(requested, backend=backend, adapter=adapter, has_devices=has_devices)
    if selection.reason != "calibration_required":
        if selection.reason == "calibration_pending":
            logger.warning("ONNX {} threshold has not completed calibration; using CPU without saving", backend)
        # 驱动发现失败不能作为 CPU 性能胜出的证据，保留原有 DML 空库存持久化规则。
        return selection, selection.reason == "no_hardware_gpu" and requested == "auto" and backend == "dml"
    assert adapter is not None
    threshold = ONNX_GPU_TIME_RATIO_THRESHOLDS[backend]
    assert threshold is not None
    from ..registry import MINERU_4_MODELS_ONNX
    from .onnx_calibration import run_calibration

    # Download follows the normal model lifecycle; it is outside the 15-second worker timeout.
    model = MINERU_4_MODELS_ONNX.ocr_rec.ensure() if allow_download else MINERU_4_MODELS_ONNX.ocr_rec.ensure(source="local")
    result = run_calibration(
        str(model),
        adapter,
        intra_op_num_threads=intra_op_num_threads,
        inter_op_num_threads=inter_op_num_threads,
        threshold=threshold,
    )
    logger.bind(onnx_calibration=result).info("ONNX rec calibration: {}", result)
    if result["decision_valid"] is not True:
        logger.warning("ONNX calibration remained unstable after extended sampling; using CPU without saving")
        return OnnxSelection(reason="calibration_unstable", backend=backend), False
    use_gpu = choose_gpu(result["cpu_elapsed"], result["gpu_elapsed"], threshold)
    return OnnxSelection("gpu" if use_gpu else "cpu", adapter if use_gpu else None, "rec_calibration", backend), True


def _select_and_save(
    requested: str, source: str, intra_op_num_threads: int, inter_op_num_threads: int, *, allow_download: bool = True
) -> OnnxSelection:
    from ...config import read_onnx_device_from_file, update_config_file

    if requested == "auto" and source != "env":
        requested = read_onnx_device_from_file()
    selection, completed = _select(requested, intra_op_num_threads, inter_op_num_threads, allow_download=allow_download)
    if completed and requested == "auto" and source != "env":
        try:
            update_config_file({"model": {"onnx_device": selection.device}}, only_if_onnx_auto=True)
        except OSError as error:
            logger.warning("ONNX choice is process-local because config could not be saved: {}", error)
    return selection


def resolve_onnx_device(
    *, intra_op_num_threads: int = 0, inter_op_num_threads: int = 0, allow_download: bool = True
) -> OnnxSelection:
    """在首次模型使用或 CPU/GPU 候选达标的 API 自动推荐时选型；导入和只读诊断不调用。"""
    from ...config import get_config, get_config_file_path, get_config_source

    requested = get_config().model.onnx_device
    source = get_config_source("model.onnx_device")
    path = get_config_file_path()
    key = (os.getpid(), path, requested, source, allow_download)
    with _lock:
        if key in _selections:
            return _selections[key]
        try:
            if requested != "auto" or sys.platform not in {"win32", "linux"}:
                selection = _select_and_save(
                    requested, source, intra_op_num_threads, inter_op_num_threads, allow_download=allow_download
                )
            else:
                try:
                    Path(path).parent.mkdir(parents=True, exist_ok=True)
                    selection_lock = FileLock(f"{path}.onnx-selection.lock", timeout=30)
                    selection_lock.acquire()
                except OSError:
                    selection = _select_and_save(
                        requested, source, intra_op_num_threads, inter_op_num_threads, allow_download=allow_download
                    )
                else:
                    try:
                        selection = _select_and_save(
                            requested, source, intra_op_num_threads, inter_op_num_threads, allow_download=allow_download
                        )
                    finally:
                        selection_lock.release()
        except Exception as error:
            logger.warning("ONNX device selection failed; using CPU for this process: {}", error)
            selection = OnnxSelection(reason="selection_failed")
        _selections[key] = selection
        logger.info("ONNX device selected: {}", selection)
        return selection


def confirm_gpu_session(selection: OnnxSelection) -> None:
    """独显自动选型仅在真实 DML/WebGPU 会话成功后保存，初始化回退不写入 GPU 偏好。"""
    if selection.reason != "discrete_gpu":
        return
    from ...config import get_config, get_config_source, update_config_file

    if get_config().model.onnx_device == "auto" and get_config_source("model.onnx_device") != "env":
        try:
            update_config_file({"model": {"onnx_device": "gpu"}}, only_if_onnx_auto=True)
        except OSError as error:
            logger.warning("GPU is usable but config could not be saved: {}", error)


__all__ = [
    "METACOMMANDS",
    "ONNX_GPU_TIME_RATIO_THRESHOLDS",
    "OnnxSelection",
    "choose_gpu",
    "cached_onnx_device",
    "confirm_gpu_session",
    "inspect_onnx_candidate",
    "metacommands_enabled",
    "resolve_onnx_device",
]
