#  Copyright (c) Opendatalab. All rights reserved.
from loguru import logger

from mineru.utils.check_sys_env import is_mac_os_version_supported, is_windows_environment, is_mac_environment, \
    is_linux_environment


def get_vlm_engine(inference_engine: str, is_async: bool = False) -> str:
    """
    自动选择或验证 VLM 推理引擎

    Args:
        inference_engine: 指定的引擎名称或 'auto' 进行自动选择
        is_async: 是否使用异步引擎(仅对 vllm 有效)

    Returns:
        最终选择的引擎名称
    """
    if inference_engine == 'auto':
        # 根据操作系统自动选择引擎
        if is_windows_environment():
            inference_engine = _select_windows_engine()
        elif is_linux_environment():
            inference_engine = _select_linux_engine(is_async)
        elif is_mac_environment():
            inference_engine = _select_mac_engine()
        else:
            logger.warning("Unknown operating system, falling back to transformers")
            inference_engine = 'transformers'

    formatted_engine = _format_engine_name(inference_engine)
    logger.info(f"Using {formatted_engine} as the inference engine for VLM.")
    return formatted_engine


def _select_windows_engine() -> str:
    """Windows 平台引擎选择"""
    try:
        import lmdeploy
        return 'lmdeploy'
    except ImportError:
        return 'transformers'


def _select_linux_engine(is_async: bool) -> str:
    """Linux 平台引擎选择"""
    try:
        import vllm
        return 'vllm-async' if is_async else 'vllm'
    except ImportError:
        try:
            import lmdeploy
            return 'lmdeploy'
        except ImportError:
            return 'transformers'


def _select_mac_engine() -> str:
    """macOS 平台引擎选择"""
    try:
        from mlx_vlm import load as mlx_load
        if is_mac_os_version_supported():
            return 'mlx'
        else:
            return 'transformers'
    except ImportError:
        return 'transformers'


def _format_engine_name(engine: str) -> str:
    """统一格式化引擎名称"""
    if engine != 'transformers':
        return f"{engine}-engine"
    return engine
