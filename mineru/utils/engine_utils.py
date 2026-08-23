# Copyright (c) Opendatalab. All rights reserved.
from typing import Literal, TypeAlias

from loguru import logger

from .check_sys_env import is_linux_environment, is_mac_environment, is_mac_os_version_supported, is_windows_environment

VlmEngine: TypeAlias = Literal[
    "llama-cpp-engine",
    "mlx-engine",
    "lmdeploy-engine",
    "vllm-engine",
    "vllm-async-engine",
]

DEFAULT_VLM_ENGINE = "llama-cpp-engine"


def get_vlm_engine(inference_engine: Literal["auto"], is_async: bool = False) -> VlmEngine:
    """
    自动选择或验证 VLM 推理引擎

    Args:
        inference_engine: 指定的引擎名称或 'auto' 进行自动选择
        is_async: 是否使用异步引擎(仅对 vllm 有效)

    Returns:
        最终选择的引擎名称
    """
    # 根据操作系统自动选择引擎
    if is_windows_environment():
        engine = _select_windows_engine()
    elif is_linux_environment():
        engine = _select_linux_engine(is_async)
    elif is_mac_environment():
        engine = _select_mac_engine()
    else:
        logger.warning("Unknown operating system, falling back to transformers")
        engine = DEFAULT_VLM_ENGINE

    logger.info(f"Using {engine} as the inference engine for VLM.")
    return engine


def _select_windows_engine() -> VlmEngine:
    """Windows 平台引擎选择"""
    try:
        import lmdeploy  # type: ignore

        return "lmdeploy-engine"
    except ImportError:
        return DEFAULT_VLM_ENGINE


def _select_linux_engine(is_async: bool) -> VlmEngine:
    """Linux 平台引擎选择"""
    try:
        import vllm  # type: ignore

        if is_async:
            return "vllm-async-engine"
        else:
            return "vllm-engine"
    except ImportError:
        try:
            import lmdeploy  # type: ignore

            return "lmdeploy-engine"
        except ImportError:
            return DEFAULT_VLM_ENGINE


def _select_mac_engine() -> VlmEngine:
    """macOS 平台引擎选择"""
    try:
        if is_mac_os_version_supported():
            import mlx_vlm  # type: ignore

            return "mlx-engine"
    except ImportError:
        pass
    return DEFAULT_VLM_ENGINE
