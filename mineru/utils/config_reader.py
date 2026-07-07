# Copyright (c) Opendatalab. All rights reserved.
import json
import os
from typing import Any

from loguru import logger

# 定义配置文件名常量，保留给旧日志文案使用；实际读取路径通过函数动态解析环境变量。
CONFIG_FILE_NAME = os.getenv("MINERU_TOOLS_CONFIG_JSON", "mineru.json")


def get_tools_config_file_path() -> str:
    """获取 MinerU 工具配置文件路径，支持环境变量指定绝对或相对路径。"""
    config_file_name = os.getenv("MINERU_TOOLS_CONFIG_JSON", "mineru.json")
    if os.path.isabs(config_file_name):
        return config_file_name
    return os.path.join(os.path.expanduser("~"), config_file_name)


def read_config() -> dict[str, Any] | None:
    config_file = get_tools_config_file_path()

    if not os.path.exists(config_file):
        # logger.warning(f'{config_file} not found, using default configuration')
        return None
    else:
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config


def get_configured_model_source(default: str | None = None) -> str | None:
    """读取配置文件中的固定模型来源配置，auto 或缺失时返回默认值。"""
    supported_sources = {"huggingface", "modelscope"}
    config = read_config()
    if config is None:
        return default

    model_source = config.get("model-source")
    if model_source is None:
        return default
    if not isinstance(model_source, str):
        logger.warning(f"'model-source' in {get_tools_config_file_path()} must be a string, use {default} as default")
        return default

    normalized_model_source = model_source.strip().lower()
    if not normalized_model_source:
        return default
    if normalized_model_source == "auto":
        return default
    if normalized_model_source in supported_sources:
        return normalized_model_source

    logger.warning(
        f"Unsupported 'model-source' in {get_tools_config_file_path()}: {model_source}, use {default} as default"
    )
    return default


def get_s3_config(bucket_name: str) -> tuple[str, str, str]:
    """~/magic-pdf.json 读出来."""
    config = read_config()

    bucket_info = config.get("bucket_info")
    if bucket_name not in bucket_info:
        access_key, secret_key, storage_endpoint = bucket_info["[default]"]
    else:
        access_key, secret_key, storage_endpoint = bucket_info[bucket_name]

    if access_key is None or secret_key is None or storage_endpoint is None:
        raise Exception(f"ak, sk or endpoint not found in {CONFIG_FILE_NAME}")

    # logger.info(f"get_s3_config: ak={access_key}, sk={secret_key}, endpoint={storage_endpoint}")

    return access_key, secret_key, storage_endpoint


def get_s3_config_dict(path: str) -> dict[str, str]:
    access_key, secret_key, storage_endpoint = get_s3_config(get_bucket_name(path))
    return {"ak": access_key, "sk": secret_key, "endpoint": storage_endpoint}


def get_bucket_name(path: str) -> str:
    bucket, key = parse_bucket_key(path)
    return bucket


def parse_bucket_key(s3_full_path: str) -> tuple[str, str]:
    """
    输入 s3://bucket/path/to/my/file.txt
    输出 bucket, path/to/my/file.txt
    """
    s3_full_path = s3_full_path.strip()
    if s3_full_path.startswith("s3://"):
        s3_full_path = s3_full_path[5:]
    if s3_full_path.startswith("/"):
        s3_full_path = s3_full_path[1:]
    bucket, key = s3_full_path.split("/", 1)
    return bucket, key


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


def get_ocr_det_mask_inline_formula_enable(enable: bool) -> bool:
    enable_env = os.getenv("MINERU_OCR_DET_MASK_INLINE_FORMULA_ENABLE")
    enable = enable if enable_env is None else enable_env.lower() == "true"
    return enable


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


def get_latex_delimiter_config() -> dict[str, Any] | None:
    config = read_config()
    if config is None:
        return None
    latex_delimiter_config = config.get("latex-delimiter-config", None)
    if latex_delimiter_config is None:
        # logger.warning(f"'latex-delimiter-config' not found in {CONFIG_FILE_NAME}, use 'None' as default")
        return None
    else:
        return latex_delimiter_config


def get_llm_aided_config() -> dict[str, Any] | None:
    config = read_config()
    if config is None:
        return None
    llm_aided_config = config.get("llm-aided-config", None)
    if llm_aided_config is None:
        # logger.warning(f"'llm-aided-config' not found in {CONFIG_FILE_NAME}, use 'None' as default")
        return None
    else:
        return llm_aided_config


def get_local_models_dir() -> dict[str, str] | None:
    config = read_config()
    if config is None:
        return None
    models_dir = config.get("models-dir")
    if models_dir is None:
        logger.warning(f"'models-dir' not found in {CONFIG_FILE_NAME}, use None as default")
    return models_dir
