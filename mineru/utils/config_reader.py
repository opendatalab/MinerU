# Copyright (c) Opendatalab. All rights reserved.
import os

from loguru import logger


def get_device() -> str:
    device_mode = os.getenv("MINERU_DEVICE_MODE", None)
    if device_mode is not None:
        return device_mode
    try:
        import torch
    except ImportError:
        return "cpu"
    try:
        if torch.cuda.is_available():  # type: ignore
            return "cuda"
    except Exception:
        pass
    try:
        if torch.backends.mps.is_available():  # type: ignore
            return "mps"
    except Exception:
        pass
    try:
        import torch_npu

        if torch_npu.npu.is_available():  # type: ignore
            return "npu"
    except Exception:
        pass
    try:
        if torch.gcu.is_available():  # type: ignore
            return "gcu"
    except Exception:
        pass
    try:
        if torch.musa.is_available():  # type: ignore
            return "musa"
    except Exception:
        pass
    try:
        if torch.mlu.is_available():  # type: ignore
            return "mlu"
    except Exception:
        pass
    try:
        if torch.sdaa.is_available():  # type: ignore
            return "sdaa"
    except Exception:
        pass
    return "cpu"


def get_model_stack() -> str:
    """返回模型推理技术栈：``"light"`` 或 ``"full"``。

    优先读取 ``config.model.stack``（支持环境变量 ``MINERU_MODEL_STACK``）。
    当值为 ``"auto"`` 时，根据 ``get_device()`` 的结果自动选择：
    device 为 ``"cpu"`` 时用 ``"light"``（onnxruntime / llama.cpp），否则用 ``"full"``（PyTorch / transformers）。
    """
    from ..config import config

    stack = config.model.stack
    if stack in ("light", "full"):
        return stack
    # auto: cpu → light, 其他 → full
    device = get_device()
    return "light" if device == "cpu" else "full"


def get_processing_window_size(default: int = 64) -> int:
    value = os.getenv("MINERU_PROCESSING_WINDOW_SIZE")
    if value is None:
        return default
    try:
        window_size = int(value)
    except ValueError:
        logger.warning(f"Invalid MINERU_PROCESSING_WINDOW_SIZE value: {value}, use default {default}")
        return default
    return max(1, window_size)


def get_max_concurrent_requests(default: int = 3) -> int:
    if default <= 0:
        raise ValueError(f"default max_concurrent_requests must be a positive integer, got {default}")
    value = os.getenv("MINERU_API_MAX_CONCURRENT_REQUESTS")
    if value is None:
        return default
    try:
        max_concurrent_requests = int(value)
    except ValueError as exc:
        raise ValueError(f"Invalid MINERU_API_MAX_CONCURRENT_REQUESTS value: {value}. Expected a positive integer.") from exc
    if max_concurrent_requests <= 0:
        raise ValueError(f"Invalid MINERU_API_MAX_CONCURRENT_REQUESTS value: {value}. Expected a positive integer.")
    return max_concurrent_requests
