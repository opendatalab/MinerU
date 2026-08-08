# Copyright (c) Opendatalab. All rights reserved.
from typing import Any, List, Sequence, Tuple

from loguru import logger

from mineru.utils.config_reader import get_device


CPU_PROVIDER = "CPUExecutionProvider"
CUDA_PROVIDER = "CUDAExecutionProvider"
COREML_PROVIDER = "CoreMLExecutionProvider"
MPS_PROVIDER = "MPSExecutionProvider"
CPU_PROVIDER_OPTS = {
    "arena_extend_strategy": "kSameAsRequested",
}
CUDA_PROVIDER_OPTS = {
    "cudnn_conv_algo_search": "HEURISTIC",
}
# 注意：旧版 onnxruntime 的 CoreML EP 不识别 ComputeUnits 等选项，
# 只使用各版本都支持的 ModelFormat，避免 EP 静默回退到 CPU。
COREML_PROVIDER_OPTS = {
    "ModelFormat": "MLProgram",
}
MPS_PROVIDER_OPTS = {
    "enable_mps_fallback": 1,
}


def _provider_name(provider: Any) -> str:
    """从 provider 描述（字符串或 (名称, 选项) 元组）中取出 EP 名称。"""
    if isinstance(provider, str):
        return provider
    return str(provider[0])


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


def _build_coreml_provider() -> Tuple[str, dict[str, Any]]:
    """macOS 上 onnxruntime 官方 wheel 内置的 CoreML EP。"""
    return (COREML_PROVIDER, dict(COREML_PROVIDER_OPTS))


def _build_mps_provider() -> Tuple[str, dict[str, Any]]:
    """onnxruntime 较新版本提供的 Metal (MPS) EP。"""
    return (MPS_PROVIDER, dict(MPS_PROVIDER_OPTS))


def build_table_onnx_providers(
    available_providers: Sequence[str],
) -> List[Tuple[str, dict[str, Any]]]:
    """根据 MinerU 当前设备为表格 ONNX 模型选择 onnxruntime providers。"""
    cpu_provider = _build_cpu_provider()
    cuda_provider = _build_cuda_provider()
    device = _normalize_device(get_device())

    # CUDA 设备：只有明确为 CUDA 且 EP 可用时才尝试 CUDAExecutionProvider，保持默认 CPU 行为。
    if device == "cuda":
        if CUDA_PROVIDER in available_providers:
            return [cuda_provider, cpu_provider]
        return [cpu_provider]

    # Apple Silicon（MPS 设备）使用统一内存：优先 CoreML EP，其次 MPS EP，兜底 CPU。
    if device == "mps":
        accelerated_provider = None
        if COREML_PROVIDER in available_providers:
            accelerated_provider = _build_coreml_provider()
        elif MPS_PROVIDER in available_providers:
            accelerated_provider = _build_mps_provider()
        if accelerated_provider is not None:
            return [accelerated_provider, cpu_provider]
        return [cpu_provider]

    return [cpu_provider]


def create_table_onnx_session(model_path: str, sess_options: Any, available_providers: Sequence[str]) -> Any:
    """创建表格 ONNX 推理会话，加速 EP 不可用时自动降级。

    CoreML/MPS 等 EP 对部分模型算子或 provider 选项支持不完整，直接创建会话
    可能抛错，也可能被 onnxruntime 静默回退到 CPU。这里依次尝试：
    带选项的加速 EP -> 不带选项的加速 EP -> CPU，保证表格结构识别始终可用。
    """
    from onnxruntime import InferenceSession

    cpu_provider = _build_cpu_provider()
    providers = build_table_onnx_providers(available_providers)
    candidates: List[Any] = []
    if providers:
        candidates.append(providers)
        candidates.append([name for name, _ in providers])
    candidates.append([cpu_provider])

    last_exc: Exception | None = None
    for candidate in candidates:
        try:
            session = InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=candidate,
            )
        except Exception as exc:
            last_exc = exc
            continue
        # 校验加速 EP 是否真正生效（onnxruntime 可能静默回退到 CPU）。
        provider_name = _provider_name(candidate[0])
        if provider_name == CPU_PROVIDER or session.get_providers()[0] == provider_name:
            return session

    logger.warning(
        f"Failed to create table ONNX session with providers {providers}, "
        f"falling back to CPU: {last_exc}"
    )
    return InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=[cpu_provider],
    )
