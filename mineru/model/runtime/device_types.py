# Copyright (c) Opendatalab. All rights reserved.
"""不加载推理后端的设备选型值类型。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .webgpu import WebGPUDevice
    from .windows_gpu import GpuAdapter


@dataclass(frozen=True)
class TorchDeviceSelection:
    """记录经过可用性检查的 Torch 设备，不包含模型性能结论。"""

    device: str = "cpu"
    reason: str = "torch_cpu"
    warning: str | None = None

    @property
    def accelerated(self) -> bool:
        """将已验证的非 CPU 设备归为加速模式。"""
        return self.device.split(":", 1)[0] != "cpu"


@dataclass(frozen=True)
class OnnxSelection:
    """保存 ONNX 具体路由，适配器身份只在本进程有效。"""

    device: Literal["cpu", "gpu"] = "cpu"
    adapter: GpuAdapter | WebGPUDevice | None = None
    reason: str = "cpu"
    backend: Literal["dml", "webgpu"] | None = None


@dataclass(frozen=True)
class RuntimeDevice:
    """从同一具体选型派生默认后端和 CPU/GPU 模式，避免冗余状态不一致。"""

    selection: OnnxSelection | TorchDeviceSelection

    @property
    def backend(self) -> Literal["onnx", "torch"]:
        """返回具体选型所属的小模型后端。"""
        return "torch" if isinstance(self.selection, TorchDeviceSelection) else "onnx"

    @property
    def device(self) -> Literal["cpu", "gpu"]:
        """统一把 Torch 非 CPU 加速器及 ONNX GPU 路由映射为 gpu。"""
        if isinstance(self.selection, TorchDeviceSelection):
            return "gpu" if self.selection.accelerated else "cpu"
        return self.selection.device

    @property
    def is_cpu(self) -> bool:
        """供策略消费者判断是否采用 CPU 模式。"""
        return self.device == "cpu"

    @property
    def model_device(self) -> str:
        """Torch 使用具体设备字符串，ONNX 的包装层保持 CPU 字符串并单独传递适配器。"""
        return self.selection.device if isinstance(self.selection, TorchDeviceSelection) else "cpu"


__all__ = ["OnnxSelection", "RuntimeDevice", "TorchDeviceSelection"]
