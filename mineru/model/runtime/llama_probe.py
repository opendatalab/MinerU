# Copyright (c) Opendatalab. All rights reserved.
"""在独立进程读取已安装 ggml 的真实可见设备，避免提前初始化宿主的 Vulkan 状态。"""

from __future__ import annotations

import ctypes as ct
from contextlib import ExitStack
from dataclasses import asdict, dataclass, replace
import importlib.util
import json
import os
from pathlib import Path
import sys

from .gpu import GpuClassification, GpuIdentity, match_gpu_identities, with_gpu_classification
from .vulkan import VulkanDevice, enumerate_vulkan_devices

PROBE_MARKER = "MINERU_LLAMA_DEVICES="


@dataclass(frozen=True)
class LlamaDevice(GpuClassification):
    """ggml 注册设备名称和经过唯一匹配的 Vulkan PCI 身份。"""

    name: str
    description: str
    backend: str
    vendor_id: int | None = None
    device_id: int | None = None
    vulkan_index: int | None = None
    luid: str | None = None

    @property
    def identity(self) -> GpuIdentity:
        """提供原生唯一匹配得到的物理身份，不使用可变的 VulkanN 注册名称。"""
        return GpuIdentity(self.vendor_id, self.device_id, luid=self.luid, name=self.description)


def _classify_devices(devices: tuple[LlamaDevice, ...], physical: tuple[VulkanDevice, ...]) -> tuple[LlamaDevice, ...]:
    """ggml 只确认可见性，分类来自 Linux Vulkan 或 Windows D3D12 的唯一原生匹配。"""
    vulkan_indices = [index for index, device in enumerate(devices) if device.backend == "vulkan"]
    matches = match_gpu_identities(
        [devices[index].identity for index in vulkan_indices], [device.identity for device in physical], allow_name=True
    )
    result = list(devices)
    for index, matched in zip(vulkan_indices, matches):
        if matched is None:
            continue
        native = physical[matched]
        result[index] = replace(
            result[index], vendor_id=native.vendor_id, device_id=native.device_id, vulkan_index=native.index
        )
        if sys.platform == "linux":
            result[index] = with_gpu_classification(result[index], native)
    if sys.platform == "win32":
        from .windows_gpu import discover_gpus

        try:
            adapters = discover_gpus()
        except (OSError, RuntimeError):
            adapters = ()
        matches = match_gpu_identities(
            [result[index].identity for index in vulkan_indices], [adapter.identity for adapter in adapters], allow_name=True
        )
        for index, matched in zip(vulkan_indices, matches):
            if matched is not None:
                adapter = adapters[matched]
                result[index] = with_gpu_classification(
                    replace(result[index], vendor_id=adapter.vendor_id, device_id=adapter.device_id, luid=adapter.luid), adapter
                )
    return tuple(result)


def probe_devices() -> tuple[LlamaDevice, ...]:
    """仅供子进程使用；不加载模型，不运行 shader，不猜测原生设备排序。"""
    spec = importlib.util.find_spec("mineru_llama_cpp")
    if spec is None or spec.origin is None:
        raise ImportError("mineru-llama-cpp is not installed")
    root = Path(spec.origin).resolve().parent
    with ExitStack() as stack:
        if sys.platform == "win32":
            for folder in (root / "bin", root / "lib"):
                if folder.is_dir():
                    stack.enter_context(os.add_dll_directory(str(folder)))
        return _probe_library(root)


def _probe_library(root: Path) -> tuple[LlamaDevice, ...]:
    """直接加载 ggml 注册库，避免 Python 扩展中的另一份 C++ 运行时参与探测。"""
    names = ("ggml.dll", "libggml.dll") if sys.platform == "win32" else ("libggml.so.0", "libggml.so")
    path = next((root / folder / name for folder in ("lib", "bin") for name in names if (root / folder / name).is_file()), None)
    if path is None:
        raise RuntimeError("Cannot find the packaged ggml device registry library")
    library = ct.CDLL(str(path))
    load = library.ggml_backend_load_all_from_path
    load.argtypes, load.restype = [ct.c_char_p], None
    count = library.ggml_backend_dev_count
    count.argtypes, count.restype = [], ct.c_size_t
    get = library.ggml_backend_dev_get
    get.argtypes, get.restype = [ct.c_size_t], ct.c_void_p
    kind = library.ggml_backend_dev_type
    kind.argtypes, kind.restype = [ct.c_void_p], ct.c_int
    name = library.ggml_backend_dev_name
    name.argtypes, name.restype = [ct.c_void_p], ct.c_char_p
    description = library.ggml_backend_dev_description
    description.argtypes, description.restype = [ct.c_void_p], ct.c_char_p
    registry = library.ggml_backend_dev_backend_reg
    registry.argtypes, registry.restype = [ct.c_void_p], ct.c_void_p
    backend = library.ggml_backend_reg_name
    backend.argtypes, backend.restype = [ct.c_void_p], ct.c_char_p
    load(str(root / "bin").encode())
    try:
        physical = enumerate_vulkan_devices()
    except (OSError, RuntimeError):
        physical = ()
    result = []
    for index in range(count()):
        device = get(index)
        device_type = kind(device)
        if device_type not in (1, 2):  # Public ggml GPU / IGPU enum, excluding CPU and accelerators.
            continue
        label = description(device).decode()
        backend_name = backend(registry(device)).decode().lower()
        result.append(
            LlamaDevice(
                name(device).decode(),
                label,
                "metal" if backend_name == "mtl" else backend_name,
            )
        )
    return _classify_devices(tuple(result), physical)


def main() -> None:
    """单行 JSON 协议与原生库可能打印的诊断信息分离。"""
    print(PROBE_MARKER + json.dumps([asdict(device) for device in probe_devices()]), flush=True)


__all__ = ["LlamaDevice", "PROBE_MARKER"]


if __name__ == "__main__":
    main()
