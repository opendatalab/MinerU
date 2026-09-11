# Copyright (c) Opendatalab. All rights reserved.
"""独立于小模型后端的 VLM 引擎选择。"""

from typing import Literal, TypeAlias

from loguru import logger

from ..runtime.device import get_device, module_available
from ..runtime.platform import is_linux_environment, is_windows_environment

VlmEngineName: TypeAlias = Literal["llama-cpp", "mlx", "lmdeploy", "vllm"]
VlmEngine: TypeAlias = Literal["llama-cpp-engine", "mlx-engine", "lmdeploy-engine", "vllm-engine", "vllm-async-engine"]
DEFAULT_VLM_ENGINE = "llama-cpp-engine"
_ENGINE_BACKENDS: dict[VlmEngineName, VlmEngine] = {
    "llama-cpp": "llama-cpp-engine",
    "mlx": "mlx-engine",
    "lmdeploy": "lmdeploy-engine",
    "vllm": "vllm-engine",
}
VLM_REQUIRED_MODULES: dict[VlmEngineName, tuple[str, ...]] = {
    "llama-cpp": ("mineru_llama_cpp",),
    "vllm": ("vllm",),
    "lmdeploy": ("lmdeploy", "qwen_vl_utils"),
    "mlx": ("mlx", "mlx_vlm"),
}


def resolve_vlm_engine(engine: str | None = None) -> VlmEngineName:
    """按显式配置或平台能力选择引擎；显式名称不要求下载端安装引擎。"""
    from ...config import config

    selected = config.model.vlm.engine if engine is None else engine
    if selected == "llama-cpp":
        return "llama-cpp"
    if selected == "vllm":
        return "vllm"
    if selected == "lmdeploy":
        return "lmdeploy"
    if selected == "mlx":
        return "mlx"
    if selected != "auto":
        raise ValueError(f"Unsupported VLM engine '{selected}'. Expected one of: auto, llama-cpp, vllm, lmdeploy, mlx.")
    # macOS 和未知平台固定优先 llama；不探测或自动导入 MLX。
    if not (is_linux_environment() or is_windows_environment()):
        return "llama-cpp"
    if get_device().split(":")[0] == "cpu":
        return "llama-cpp"
    if is_linux_environment() and module_available("vllm"):
        return "vllm"
    if module_available("lmdeploy"):
        return "lmdeploy"
    return "llama-cpp"


def get_vlm_engine(inference_engine: str | None = None, is_async: bool = False) -> VlmEngine:
    """将统一引擎选择转换为客户端后端名，异步选项仅影响 vLLM。"""
    selected = resolve_vlm_engine(inference_engine)
    backend = "vllm-async-engine" if selected == "vllm" and is_async else _ENGINE_BACKENDS[selected]
    logger.info(f"Using {backend} as the inference engine for VLM.")
    return backend


__all__ = ["DEFAULT_VLM_ENGINE", "VLM_REQUIRED_MODULES", "VlmEngine", "VlmEngineName", "get_vlm_engine", "resolve_vlm_engine"]
