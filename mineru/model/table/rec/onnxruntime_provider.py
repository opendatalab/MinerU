# Copyright (c) Opendatalab. All rights reserved.
from typing import Any, List, Sequence, Tuple

from loguru import logger

from mineru.utils.config_reader import get_device


CPU_PROVIDER = "CPUExecutionProvider"
CUDA_PROVIDER = "CUDAExecutionProvider"
CPU_PROVIDER_OPTS = {
    "arena_extend_strategy": "kSameAsRequested",
}
CUDA_PROVIDER_OPTS = {
    "cudnn_conv_algo_search": "HEURISTIC",
}


def _normalize_device(device: object) -> str:
    """归一化 MinerU 设备名，兼容 cuda:0 这类带索引的写法。"""
    if not isinstance(device, str):
        return ""
    return device.split(":", 1)[0].strip().lower()


def _build_cpu_provider() -> Tuple[str, dict[str, Any]]:
    """构建 CPU provider 配置，避免复用可变的模块级字典。"""
    return (CPU_PROVIDER, dict(CPU_PROVIDER_OPTS))


def _build_cuda_provider() -> Tuple[str, dict[str, Any]]:
    return (CUDA_PROVIDER, dict(CUDA_PROVIDER_OPTS))


def build_table_onnx_providers(
    available_providers: Sequence[str],
) -> List[Tuple[str, dict[str, Any]]]:
    """根据 MinerU 当前设备为表格 ONNX 模型选择 onnxruntime providers。"""
    cpu_provider = _build_cpu_provider()
    cuda_provider = _build_cuda_provider()
    device = _normalize_device(get_device())

    # 只有 MinerU 设备明确为 CUDA 时才尝试 CUDAExecutionProvider，保持默认 CPU 行为。
    if device != "cuda":
        return [cpu_provider]

    if CUDA_PROVIDER in available_providers:
        return [cuda_provider, cpu_provider]

    return [cpu_provider]
