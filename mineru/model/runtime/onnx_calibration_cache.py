# Copyright (c) Opendatalab. All rights reserved.
"""保存可验证的 OCR 校准证据，供服务启动推荐和只读诊断复用。"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import tempfile
from typing import Any, Literal

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .cpu_threads import effective_cpu_count
from .webgpu import WebGPUDevice
from .windows_gpu import GpuAdapter


class CalibrationRecord(BaseModel):
    """区分真实校准结果与手动设备偏好，并限制损坏记录的读取范围。"""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    schema_version: Literal[1] = 1
    fingerprint: dict[str, Any]
    cpu_elapsed: float = Field(gt=0)
    gpu_elapsed: float = Field(gt=0)
    threshold: float = Field(gt=0, lt=1)
    outputs_match: Literal[True]
    decision_valid: bool
    model_sha256: str


@dataclass(frozen=True)
class CalibrationInfo:
    """只读校准摘要；未知环境匹配状态与明确不匹配分开表达。"""

    path: str
    status: str
    matches_environment: bool | None = None
    decision_valid: bool | None = None
    outputs_match: bool | None = None
    cpu_elapsed: float | None = None
    gpu_elapsed: float | None = None
    ratio: float | None = None
    threshold: float | None = None
    model_sha256: str | None = None
    error: str | None = None


def inspect_calibration_record(adapters: tuple[GpuAdapter | WebGPUDevice, ...] | None = None) -> CalibrationInfo:
    """读取记录并与已探测的物理设备比较，不创建目录、不测速或修改文件。"""
    path = calibration_record_path()
    try:
        record = CalibrationRecord.model_validate_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return CalibrationInfo(str(path), "missing")
    except (OSError, ValueError) as error:
        return CalibrationInfo(str(path), "invalid", error=str(error))
    matches = None if adapters is None else any(record.fingerprint == _fingerprint(adapter) for adapter in adapters)
    return CalibrationInfo(
        str(path),
        "present",
        matches,
        record.decision_valid,
        record.outputs_match,
        record.cpu_elapsed,
        record.gpu_elapsed,
        record.gpu_elapsed / record.cpu_elapsed,
        record.threshold,
        record.model_sha256,
    )


def calibration_record_path() -> Path:
    """记录随当前配置文件隔离，读取时不创建目录或锁文件。"""
    from ...config import get_config_file_path

    return Path(f"{get_config_file_path()}.onnx-calibration.json")


def _runtime_version(distribution: str) -> str | None:
    """仅查询安装元数据，不导入或初始化推理运行时。"""
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _threads(configured: int, variable: str, default: int) -> int:
    """镜像 ONNX 校准的线程优先级，避免读取记录时创建 ORT 会话。"""
    if configured > 0:
        return configured
    try:
        value = int(os.getenv(variable, "0"))
    except ValueError:
        value = 0
    return value if value > 0 else default


def _fingerprint(
    adapter: GpuAdapter | WebGPUDevice, intra_op_num_threads: int = 0, inter_op_num_threads: int = 0
) -> dict[str, Any]:
    """绑定物理设备、后端版本及 CPU 配置；忽略进程内设备序号和排序。"""
    # 显式保持既有证据格式，分类来源等新增诊断字段不参与性能证据的身份指纹。
    identity: dict[str, Any] = {"vendor_id": adapter.vendor_id, "device_id": adapter.device_id}
    if isinstance(adapter, GpuAdapter):
        identity.update(luid=adapter.luid, name=adapter.name, uma=adapter.unified_memory)
    else:
        identity.update(metadata=adapter.metadata, vulkan_device=adapter.vulkan_device)
    backend = "webgpu" if isinstance(adapter, WebGPUDevice) else "dml"
    # JSON 往返统一 metadata 等元组，确保读写指纹完全一致。
    return json.loads(
        json.dumps(
            {
                "adapter": identity,
                "backend": backend,
                "runtime": _runtime_version("onnxruntime" if backend == "webgpu" else "onnxruntime-directml"),
                "ep": _runtime_version("onnxruntime-ep-webgpu") if backend == "webgpu" else None,
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "cpu_count": effective_cpu_count(),
                "intra_threads": _threads(intra_op_num_threads, "MINERU_INTRA_OP_NUM_THREADS", 4),
                "inter_threads": _threads(inter_op_num_threads, "MINERU_INTER_OP_NUM_THREADS", 1),
            }
        )
    )


def save_calibration_record(adapter: GpuAdapter | WebGPUDevice, result: dict[str, Any]) -> None:
    """只保存已由校准 worker 验证的结果，写入失败不改变实际设备选择。"""
    temporary: str | None = None
    try:
        record = CalibrationRecord(
            fingerprint=_fingerprint(adapter, result["intra_threads"], result["inter_threads"]),
            cpu_elapsed=result["cpu_elapsed"],
            gpu_elapsed=result["gpu_elapsed"],
            threshold=result["decision_threshold"],
            outputs_match=result["outputs_match"],
            decision_valid=result["decision_valid"],
            model_sha256=result["model_sha256"],
        )
        path = calibration_record_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".onnx-calibration.", dir=path.parent)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(record.model_dump_json())
        os.replace(temporary, path)
    except (OSError, ValueError, KeyError, TypeError) as error:
        logger.warning("Cannot save ONNX calibration evidence: {}", error)
    finally:
        if temporary is not None:
            try:
                Path(temporary).unlink(missing_ok=True)
            except OSError as error:
                logger.warning("Cannot remove temporary ONNX calibration evidence: {}", error)


def invalidate_calibration_record() -> None:
    """新一轮实际校准开始时撤销旧证据，失败结果不能继续沿用旧的 GPU 胜出结论。"""
    try:
        calibration_record_path().unlink(missing_ok=True)
    except OSError as error:
        logger.warning("Cannot invalidate ONNX calibration evidence: {}", error)


def calibrated_gpu_is_faster(adapter: GpuAdapter | WebGPUDevice, threshold: float) -> bool:
    """仅接受当前设备和运行环境匹配、输出一致且稳定的真实校准记录。"""
    try:
        record = CalibrationRecord.model_validate_json(calibration_record_path().read_text(encoding="utf-8"))
        return (
            record.fingerprint == _fingerprint(adapter)
            and record.threshold == threshold
            and record.decision_valid
            and record.gpu_elapsed / record.cpu_elapsed <= threshold
        )
    except (OSError, ValueError, TypeError, ValidationError):
        return False


__all__ = [
    "CalibrationInfo",
    "calibrated_gpu_is_faster",
    "calibration_record_path",
    "inspect_calibration_record",
    "invalidate_calibration_record",
    "save_calibration_record",
]
