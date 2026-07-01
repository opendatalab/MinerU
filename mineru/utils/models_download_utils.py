# Copyright (c) Opendatalab. All rights reserved.
import json
import os
from functools import lru_cache

from huggingface_hub import snapshot_download as hf_snapshot_download
from loguru import logger
from modelscope import snapshot_download as ms_snapshot_download
import requests

from .config_reader import get_configured_model_source, get_local_models_dir, get_tools_config_file_path
from .enum_class import ModelPath

MODEL_SOURCE_ENV_VAR = "MINERU_MODEL_SOURCE"
CONFIG_TEMPLATE_URL = "https://gcore.jsdelivr.net/gh/opendatalab/MinerU@master/mineru.template.json"
MINERU_CONFIG_VERSION = "1.3.2"
HUGGINGFACE_MODELS_PAGE_URL = "https://huggingface.co/models"
HUGGINGFACE_MODELS_PAGE_TIMEOUT = 3
HUGGINGFACE_MODELS_PAGE_MAX_ATTEMPTS = 2
REMOTE_MODEL_SOURCES = ("huggingface", "modelscope")


def download_json(url: str) -> dict:
    """下载 JSON 文件并返回解析后的内容。"""
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def is_config_version_outdated(config_version: object) -> bool:
    """判断本地配置版本是否低于当前模板版本。"""

    def version_tuple(version: object) -> tuple[int, ...]:
        """将版本号字符串转换为可比较的整数元组。"""
        parts = []
        for part in str(version).split("."):
            parts.append(int(part) if part.isdigit() else 0)
        return tuple(parts)

    current_version = version_tuple(config_version)
    target_version = version_tuple(MINERU_CONFIG_VERSION)
    max_len = max(len(current_version), len(target_version))
    current_version += (0,) * (max_len - len(current_version))
    target_version += (0,) * (max_len - len(target_version))
    return current_version < target_version


def merge_config_dict(base_config: dict, override_config: dict, skip_keys: set[str] | None = None) -> dict:
    """递归合并配置字典，用 override_config 覆盖 base_config 并保留新模板字段。"""
    skip_keys = skip_keys or set()
    merged_config = dict(base_config)
    for key, value in override_config.items():
        if key in skip_keys:
            continue
        base_value = merged_config.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            merged_config[key] = merge_config_dict(base_value, value, skip_keys=skip_keys)
        else:
            merged_config[key] = value
    return merged_config


def download_and_modify_json(url: str, local_filename: str, modifications: dict) -> None:
    """下载或读取 JSON 配置，并按 modifications 合并更新后写回。"""
    if os.path.exists(local_filename):
        with open(local_filename, encoding="utf-8") as f:
            data = json.load(f)
        config_version = data.get("config_version", "0.0.0")
        if is_config_version_outdated(config_version):
            template_data = download_json(url)
            data = merge_config_dict(template_data, data, skip_keys={"config_version"})
    else:
        data = download_json(url)

    data = merge_config_dict(data, modifications)

    os.makedirs(os.path.dirname(local_filename) or ".", exist_ok=True)
    with open(local_filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def persist_resolved_model_source(model_source: str) -> None:
    """将 auto 解析出的实际模型来源写入配置文件，避免后续启动受网络波动影响。"""
    if model_source not in REMOTE_MODEL_SOURCES:
        return
    try:
        download_and_modify_json(
            CONFIG_TEMPLATE_URL,
            get_tools_config_file_path(),
            {"model-source": model_source},
        )
    except Exception as exc:
        logger.warning(f"Failed to persist resolved model source '{model_source}': {exc}")


def normalize_download_relative_path(relative_path: str, repo_mode: str) -> str:
    """按仓库模式规范化下载相对路径，保持 pipeline 与 VLM 根路径语义。"""
    if repo_mode == "pipeline":
        return relative_path.strip("/")
    if repo_mode == "vlm":
        if relative_path == "/":
            return relative_path
        return relative_path.strip("/")
    raise ValueError(f"Unsupported repo_mode: {repo_mode}, must be 'pipeline' or 'vlm'")


def read_existing_tools_config() -> dict | None:
    """读取已存在的工具配置文件；缺失、读取失败或格式异常时返回 None。"""
    config_file = get_tools_config_file_path()
    if not os.path.exists(config_file):
        return None
    try:
        with open(config_file, encoding="utf-8") as f:
            config = json.load(f)
    except Exception as exc:
        logger.warning(f"Failed to read model config from {config_file}: {exc}")
        return None
    if not isinstance(config, dict):
        logger.warning(f"Model config in {config_file} must be a JSON object.")
        return None
    return config


def get_configured_repo_model_root(config: dict, repo_mode: str) -> str | None:
    """从配置中读取指定仓库模式的模型根目录，缺失或类型错误时返回 None。"""
    models_dir = config.get("models-dir")
    if not isinstance(models_dir, dict):
        return None
    model_root = models_dir.get(repo_mode)
    if not isinstance(model_root, str):
        return None
    model_root = model_root.strip()
    if not model_root:
        return None
    return os.path.expanduser(model_root)


def build_configured_model_path(model_root: str, relative_path: str) -> str:
    """根据模型根目录和下载相对路径拼出本地校验路径。"""
    if relative_path in ("", "/"):
        return model_root
    return os.path.join(model_root, relative_path)


def is_existing_model_path_usable(model_path: str) -> bool:
    """判断本地模型路径是否可复用：文件可直接用，目录必须包含实际文件。"""
    if os.path.isfile(model_path):
        return True
    if not os.path.isdir(model_path):
        return False
    for _root, _dirs, files in os.walk(model_path):
        if files:
            return True
    return False


def get_existing_configured_model_root(repo_mode: str, relative_path: str) -> str | None:
    """如果配置中的模型根目录已包含本次所需模型，则返回该根目录以跳过远端下载。"""
    config = read_existing_tools_config()
    if config is None:
        return None

    model_root = get_configured_repo_model_root(config, repo_mode)
    if model_root is None:
        return None

    local_model_path = build_configured_model_path(model_root, relative_path)
    if is_existing_model_path_usable(local_model_path):
        logger.debug(f"Use configured local {repo_mode} model path: {local_model_path}")
        return model_root
    return None


def persist_downloaded_model_config(model_source: str, repo_mode: str, model_root: str) -> None:
    """远端模型下载成功后，创建或更新配置文件中的模型根目录和实际模型来源。"""
    if model_source not in REMOTE_MODEL_SOURCES:
        return
    config_file = get_tools_config_file_path()
    try:
        download_and_modify_json(
            CONFIG_TEMPLATE_URL,
            config_file,
            {
                "models-dir": {
                    repo_mode: model_root,
                },
                "model-source": model_source,
            },
        )
    except Exception as exc:
        logger.warning(f"Failed to persist downloaded {repo_mode} model config to {config_file}: {exc}")


@lru_cache(maxsize=1)
def resolve_auto_model_source() -> str:
    """通过 Hugging Face 模型列表页探测 auto 应该使用的实际模型来源。"""
    last_error = None
    for _ in range(HUGGINGFACE_MODELS_PAGE_MAX_ATTEMPTS):
        try:
            response = requests.get(
                HUGGINGFACE_MODELS_PAGE_URL,
                timeout=HUGGINGFACE_MODELS_PAGE_TIMEOUT,
            )
            if 200 <= response.status_code < 400:
                return "huggingface"
            last_error = f"status_code={response.status_code}"
        except Exception as exc:
            last_error = str(exc)

    logger.warning(f"Failed to access {HUGGINGFACE_MODELS_PAGE_URL}: {last_error}, fallback to modelscope.")
    return "modelscope"


def resolve_model_source(model_source: str | None = None, allow_auto: bool = False) -> str:
    """将环境变量或配置文件中的模型来源解析为实际可下载的来源。"""
    if model_source is None:
        model_source = os.getenv(MODEL_SOURCE_ENV_VAR)
        if isinstance(model_source, str) and model_source.strip().lower() == "auto":
            raise ValueError(
                f"{MODEL_SOURCE_ENV_VAR}=auto is not supported. "
                f"Unset {MODEL_SOURCE_ENV_VAR} to use auto detection once, "
                "or set it to huggingface/modelscope/local."
            )
    if model_source is None:
        model_source = get_configured_model_source()
    if model_source is None:
        model_source = "auto"
        allow_auto = True

    if not isinstance(model_source, str):
        logger.warning(f"Unsupported model source type: {type(model_source)}, fallback to auto.")
        model_source = "auto"
        allow_auto = True

    normalized_model_source = model_source.strip().lower()
    if normalized_model_source == "local":
        return "local"
    if normalized_model_source == "auto":
        if not allow_auto:
            raise ValueError(
                "model source auto is only supported for internal default detection "
                "or explicit download command selection."
            )
        resolved_model_source = resolve_auto_model_source()
        persist_resolved_model_source(resolved_model_source)
        return resolved_model_source
    if normalized_model_source in REMOTE_MODEL_SOURCES:
        return normalized_model_source

    logger.warning(f"Unsupported model source: {model_source}, fallback to auto.")
    resolved_model_source = resolve_auto_model_source()
    persist_resolved_model_source(resolved_model_source)
    return resolved_model_source


@lru_cache(maxsize=None)
def _snapshot_download_cached(model_source: str, repo_mode: str, repo: str, relative_path: str) -> str:
    """按进程缓存远端 snapshot_download 结果，减少重复缓存检查和 Fetching 日志。"""
    if model_source == "huggingface":
        snapshot_download = hf_snapshot_download
    elif model_source == "modelscope":
        snapshot_download = ms_snapshot_download
    else:
        raise ValueError(f"未知的仓库类型: {model_source}")

    if repo_mode == "pipeline":
        return snapshot_download(repo, allow_patterns=[relative_path, relative_path + "/*"])

    if repo_mode == "vlm":
        # VLM 整仓下载和局部路径下载都参与缓存，但保持原有 allow_patterns 行为。
        if relative_path == "/":
            return snapshot_download(repo)
        return snapshot_download(repo, allow_patterns=[relative_path, relative_path + "/*"])

    raise ValueError(f"Unsupported repo_mode: {repo_mode}, must be 'pipeline' or 'vlm'")


def auto_download_and_get_model_root_path(relative_path: str, repo_mode: str = "pipeline") -> str:
    """
    支持文件或目录的可靠下载。
    - 如果输入文件: 返回本地文件绝对路径
    - 如果输入目录: 返回本地缓存下与 relative_path 同结构的相对路径字符串
    :param repo_mode: 指定仓库模式，'pipeline' 或 'vlm'
    :param relative_path: 文件或目录相对路径
    :return: 本地文件绝对路径或相对路径
    """
    model_source = resolve_model_source()

    if model_source == "local":
        local_models_config = get_local_models_dir()
        if local_models_config is None:
            raise ValueError("Local model paths are not configured.")
        root_path = local_models_config.get(repo_mode, None)
        if not root_path:
            raise ValueError(f"Local path for repo_mode '{repo_mode}' is not configured.")
        return root_path

    # 建立仓库模式到路径的映射
    repo_mapping = {
        "pipeline": {
            "huggingface": ModelPath.pipeline_root_hf,
            "modelscope": ModelPath.pipeline_root_modelscope,
        },
        "vlm": {
            "huggingface": ModelPath.vlm_root_hf,
            "modelscope": ModelPath.vlm_root_modelscope,
        },
    }

    if repo_mode not in repo_mapping:
        raise ValueError(f"Unsupported repo_mode: {repo_mode}, must be 'pipeline' or 'vlm'")

    # model_source 已解析为实际远端来源后，再选择对应仓库。
    repo = repo_mapping[repo_mode][model_source]

    relative_path = normalize_download_relative_path(relative_path, repo_mode)
    configured_model_root = get_existing_configured_model_root(repo_mode, relative_path)
    if configured_model_root is not None:
        return configured_model_root

    cache_dir = _snapshot_download_cached(model_source, repo_mode, repo, relative_path)

    if not cache_dir:
        raise FileNotFoundError(f"Failed to download model: {relative_path} from {repo}")
    persist_downloaded_model_config(model_source, repo_mode, cache_dir)
    return cache_dir


if __name__ == "__main__":
    path1 = "models/README.md"
    root = auto_download_and_get_model_root_path(path1)
    print("本地文件绝对路径:", os.path.join(root, path1))
