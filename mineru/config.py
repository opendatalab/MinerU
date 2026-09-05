"""MinerU startup configuration."""

from __future__ import annotations

import copy
import logging
import os
import re
import socket
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, get_args, get_origin
from urllib.parse import urlsplit, urlunsplit

import yaml
from filelock import FileLock
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_core import to_jsonable_python

_logger = logging.getLogger(__name__)

MINERU_HOME_ENV = "MINERU_HOME"
MINERU_CONFIG_ENV = "MINERU_CONFIG"
MINERU_ENV_PREFIX = "MINERU_"

AutoBool = Literal["auto"] | bool
ConfigSource = Literal["default", "file", "env"]
ModelSource = Literal["auto", "huggingface", "modelscope", "local"]

_INTERPOLATION_RE = re.compile(r"\$\{(\w+)(?::-([^${}]*))?\}")


def _mineru_home() -> str:
    configured = os.getenv(MINERU_HOME_ENV)
    if configured not in (None, ""):
        return os.path.expanduser(configured)
    return os.path.join(os.path.expanduser("~"), ".mineru")


def _default_path(path1: str, /, *paths: str) -> str:
    return os.path.join(_mineru_home(), path1, *paths)


def _uds_available() -> bool:
    try:
        socket.socket(socket.AF_UNIX, socket.SOCK_STREAM).close()
        return True
    except Exception:
        return False


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] in ('"', "'") and value[-1] == value[0]:
        return value[1:-1]
    return value


def _interpolation_replace(match: re.Match[str]) -> str:
    var_name, raw_default = match.group(1), match.group(2)
    value = os.environ.get(var_name)
    if value is not None:
        return value
    if raw_default is not None:
        return _strip_quotes(raw_default)
    raise ValueError(
        f"Environment variable {var_name!r} is referenced in MinerU config but is not set and has no default value."
    )


def _substitute(value: str, max_depth: int = 20) -> str:
    for _ in range(max_depth):
        substituted = _INTERPOLATION_RE.sub(_interpolation_replace, value)
        if substituted == value:
            return value
        value = substituted
    raise ValueError(f"MinerU config interpolation did not converge after {max_depth} passes.")


def _interpolate_env(value: Any) -> Any:
    """Recursively substitute ${VAR} and ${VAR:-default} placeholders."""
    if isinstance(value, str):
        return _substitute(value)
    if isinstance(value, dict):
        return {key: _interpolate_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_interpolate_env(item) for item in value]
    return value


def _load_config(config_file: str) -> dict[str, Any]:
    with open(config_file, encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    if isinstance(raw, dict):
        return _interpolate_env(raw)
    _logger.warning("MinerU config file [%s] is empty or invalid.", config_file)
    return {}


def _resolve_config_file() -> tuple[str, bool]:
    config_file = os.getenv(MINERU_CONFIG_ENV)
    if config_file not in (None, ""):
        config_file = os.path.expanduser(config_file)
        if not os.path.isfile(config_file):
            raise FileNotFoundError(f"MinerU config file [{config_file}] does not exist.")
        return config_file, True

    default_config_file = _default_path("config.yaml")
    exists = os.path.isfile(default_config_file)
    if not exists:
        _logger.debug(
            "MinerU config file not found. Default path is %s. Use %s to specify a custom path.",
            default_config_file,
            MINERU_CONFIG_ENV,
        )
    return default_config_file, exists


def _collect_path(remaining: str, model_class: type[BaseModel]) -> list[str] | None:
    """Greedily match an UPPER_CASE env suffix to a pydantic field path."""
    remaining_lower = remaining.lower()
    fields = sorted(model_class.model_fields.keys(), key=len, reverse=True)
    for field_name in fields:
        if remaining_lower == field_name:
            return [field_name]
        if remaining_lower.startswith(field_name + "_"):
            annotation = model_class.model_fields[field_name].annotation
            if isinstance(annotation, type) and issubclass(annotation, BaseModel):
                sub_remaining = remaining[len(field_name) + 1 :]
                sub_path = _collect_path(sub_remaining, annotation)
                if sub_path is not None:
                    return [field_name, *sub_path]
    return None


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _collect_env_overrides(prefix: str = MINERU_ENV_PREFIX) -> tuple[dict[str, Any], set[tuple[str, ...]]]:
    prefix_upper = prefix.upper()
    overrides: dict[str, Any] = {}
    paths: set[tuple[str, ...]] = set()

    for key, value in os.environ.items():
        if not key.startswith(prefix_upper):
            continue
        remaining = key[len(prefix_upper) :]
        if not remaining or remaining == "CONFIG":
            continue
        path = _collect_path(remaining, Config)
        if path is None:
            continue
        node = overrides
        for part in path[:-1]:
            node = node.setdefault(part, {})
        node[path[-1]] = value
        paths.add(tuple(path))

    return overrides, paths


def _model_annotation(annotation: Any) -> type[BaseModel] | None:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    origin = get_origin(annotation)
    if origin is None:
        return None
    for arg in get_args(annotation):
        if isinstance(arg, type) and issubclass(arg, BaseModel):
            return arg
    return None


def _default_source_paths(model_class: type[BaseModel], prefix: tuple[str, ...] = ()) -> set[tuple[str, ...]]:
    paths: set[tuple[str, ...]] = set()
    for field_name, field in model_class.model_fields.items():
        sub_model = _model_annotation(field.annotation)
        current = (*prefix, field_name)
        if sub_model is not None:
            paths.update(_default_source_paths(sub_model, current))
        else:
            paths.add(current)
    return paths


def _configured_source_paths(
    data: dict[str, Any],
    model_class: type[BaseModel],
    prefix: tuple[str, ...] = (),
) -> set[tuple[str, ...]]:
    paths: set[tuple[str, ...]] = set()
    for key, value in data.items():
        if key not in model_class.model_fields:
            continue
        field = model_class.model_fields[key]
        sub_model = _model_annotation(field.annotation)
        current = (*prefix, key)
        if sub_model is not None and isinstance(value, dict):
            paths.update(_configured_source_paths(value, sub_model, current))
        else:
            paths.add(current)
    return paths


def _normalize_config_source_path(path: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(path, str):
        return tuple(part for part in path.split(".") if part)
    return tuple(path)


@dataclass(frozen=True)
class LoadedConfig:
    config: "Config"
    sources: dict[tuple[str, ...], ConfigSource]
    config_file: str
    config_file_exists: bool


def _load_effective_config() -> LoadedConfig:
    config_file, config_file_exists = _resolve_config_file()
    raw_config = _load_config(config_file) if config_file_exists else {}
    sources: dict[tuple[str, ...], ConfigSource] = dict.fromkeys(_default_source_paths(Config), "default")
    for path in _configured_source_paths(raw_config, Config):
        sources[path] = "file"

    base_config = Config(**raw_config)
    overrides, env_paths = _collect_env_overrides()
    if overrides:
        base_config = Config(**_deep_merge(to_jsonable_python(base_config), overrides))
        for path in env_paths:
            sources[path] = "env"
    return LoadedConfig(
        config=base_config,
        sources=sources,
        config_file=config_file,
        config_file_exists=config_file_exists,
    )


def get_config_source(path: str | Sequence[str]) -> ConfigSource:
    return _loaded_config.sources.get(_normalize_config_source_path(path), "default")


def get_config_file_path() -> str:
    return _loaded_config.config_file


def get_config_file_exists() -> bool:
    return _loaded_config.config_file_exists


def update_config_file(patch: dict[str, Any]) -> None:
    config_file = get_config_file_path()
    config_dir = os.path.dirname(config_file)
    if config_dir:
        os.makedirs(config_dir, exist_ok=True)

    lock = FileLock(f"{config_file}.lock")
    with lock:
        if os.path.exists(config_file):
            with open(config_file, encoding="utf-8") as file:
                loaded = yaml.safe_load(file)
            current = loaded if isinstance(loaded, dict) else {}
        else:
            current = {}

        updated = _deep_merge(current, patch)
        fd, tmp_path = tempfile.mkstemp(prefix=".config.", suffix=".tmp", dir=config_dir or None)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                yaml.safe_dump(updated, file, allow_unicode=True, sort_keys=False)
            os.replace(tmp_path, config_file)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class UDSConfig(BaseModel):
    enabled: AutoBool = "auto"
    path: str = _default_path("doclib.sock")
    permission: int = 0o600


class TCPConfig(BaseModel):
    enabled: AutoBool = "auto"
    host: str = "127.0.0.1"
    port: int = 15980
    strict_port: bool = False
    port_probe_count: int = Field(default=100, ge=1)
    backlog: int = 128
    timeout: int = 600


class LogConfig(BaseModel):
    dir: str = _default_path("logs")
    app_path: str | None = None
    access_path: str | None = None
    stdout_path: str | None = None
    stderr_path: str | None = None
    parse_server_stdout_path: str | None = None
    parse_server_stderr_path: str | None = None
    level: str = "info"

    @property
    def resolved_app_path(self) -> str:
        return self.app_path or os.path.join(self.dir, "doclib.log")

    @property
    def resolved_access_path(self) -> str:
        return self.access_path or os.path.join(self.dir, "doclib.access.log")

    @property
    def resolved_stdout_path(self) -> str:
        return self.stdout_path or os.path.join(self.dir, "doclib.stdout.log")

    @property
    def resolved_stderr_path(self) -> str:
        return self.stderr_path or os.path.join(self.dir, "doclib.stderr.log")

    @property
    def resolved_parse_server_stdout_path(self) -> str:
        return self.parse_server_stdout_path or os.path.join(self.dir, "doclib.parse-server.stdout.log")

    @property
    def resolved_parse_server_stderr_path(self) -> str:
        return self.parse_server_stderr_path or os.path.join(self.dir, "doclib.parse-server.stderr.log")


class SQLiteConfig(BaseModel):
    path: str = _default_path("doclib.db")
    busy_timeout_ms: int = Field(default=5000, ge=0)
    lock_retry_attempts: int = Field(default=3, ge=0)
    lock_retry_base_delay_ms: int = Field(default=50, ge=0)
    mmap_size: int = 268435456
    cache_size: int = -20000
    wal_autocheckpoint: int = 1000
    journal_size_limit: int = 33_554_432
    temp_store: str = "memory"
    synchronous: str = "NORMAL"


class ManagedParseServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 16580
    strict_port: bool = False
    port_probe_count: int = Field(default=100, ge=1)


class LatexDelimiterConfig(BaseModel):
    """单组 LaTeX 左右定界符配置。"""

    left: str = Field(min_length=1)
    right: str = Field(min_length=1)


def _default_display_latex_delimiter() -> LatexDelimiterConfig:
    """构造缺省行间公式定界符。"""
    return LatexDelimiterConfig(left="$$", right="$$")


def _default_inline_latex_delimiter() -> LatexDelimiterConfig:
    """构造缺省行内公式定界符。"""
    return LatexDelimiterConfig(left="$", right="$")


class LatexDelimitersConfig(BaseModel):
    """Markdown 行内与行间公式定界符配置。"""

    display: LatexDelimiterConfig = Field(default_factory=_default_display_latex_delimiter)
    inline: LatexDelimiterConfig = Field(default_factory=_default_inline_latex_delimiter)


class RenderConfig(BaseModel):
    """MiddleJson 输出渲染配置。"""

    latex_delimiters: LatexDelimitersConfig = Field(default_factory=LatexDelimitersConfig)


class LLMAidedFeaturesConfig(BaseModel):
    """LLM 辅助后处理的独立功能开关。"""

    title_leveling: bool = False
    cross_page_table_cell_merge: bool = False


class LLMAidedConfig(BaseModel):
    """OpenAI-compatible LLM 辅助后处理配置。"""

    api_key: str = Field(default="", repr=False)
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: str = "qwen3.5-plus"
    enable_thinking: bool | None = None
    max_concurrency: int = Field(default=16, ge=1)
    features: LLMAidedFeaturesConfig = Field(default_factory=LLMAidedFeaturesConfig)

    @field_validator("max_concurrency", mode="before")
    @classmethod
    def _reject_boolean_max_concurrency(cls, value: Any) -> Any:
        """拒绝被 Python 视作整数的布尔并发数，同时保留环境变量字符串转换。"""
        if isinstance(value, bool):
            raise ValueError("llm_aided.max_concurrency must be a positive integer")
        return value

    @model_validator(mode="after")
    def _validate_enabled_credentials(self) -> "LLMAidedConfig":
        """任一 LLM 功能启用时要求连接参数均为非空字符串。"""
        enabled = self.features.title_leveling or self.features.cross_page_table_cell_merge
        if not enabled:
            return self

        for field_name in ("api_key", "base_url", "model"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"llm_aided.{field_name} must be a non-empty string when an LLM feature is enabled")
        return self


class VlmConfig(BaseModel):
    """所有本地解析入口共享的远程 VLM 连接配置；空地址表示使用本地引擎。"""

    model_config = {"hide_input_in_errors": True}

    server_url: str = ""
    api_key: str = Field(default="", repr=False)
    model: str = ""
    http_timeout: int = Field(default=600, ge=1)
    max_concurrency: int = Field(default=100, ge=1)

    @field_validator("api_key", "model")
    @classmethod
    def _strip_connection_value(cls, value: str) -> str:
        """清理连接字段首尾空白，保持显式空值可覆盖全局配置。"""
        return value.strip()

    @field_validator("server_url")
    @classmethod
    def _normalize_server_url(cls, value: str) -> str:
        """保留代理前缀并移除末尾 v1，以尾斜线阻止底层客户端丢弃路径。"""
        value = value.strip()
        if not value:
            return ""
        try:
            parts = urlsplit(value)
            valid = (
                parts.scheme in {"http", "https"}
                and bool(parts.hostname)
                and parts.username is None
                and parts.password is None
                and not parts.query
                and not parts.fragment
                and not any(char.isspace() for char in value)
            )
            _ = parts.port
        except ValueError:
            valid = False
        if not valid:
            raise ValueError("model.vlm.server_url must be an HTTP(S) service URL without credentials, query or fragment")
        path = parts.path.rstrip("/")
        if path.endswith("/v1"):
            path = path[:-3]
        return urlunsplit((parts.scheme, parts.netloc, path + "/", "", ""))

    @field_validator("http_timeout", "max_concurrency", mode="before")
    @classmethod
    def _reject_boolean_limit(cls, value: Any) -> Any:
        """拒绝布尔数值，同时允许配置环境变量提供整数文本。"""
        if isinstance(value, bool):
            raise ValueError("VLM timeout and concurrency must be positive integers")
        return value

    def validate_environment(self) -> None:
        """远程推理前拒绝会隐式覆盖显式配置的旧环境变量，错误不包含凭据。"""
        if not self.server_url:
            return
        for env_name, expected, field in (
            ("MINERU_VL_API_KEY", self.api_key, "api_key"),
            ("MINERU_VL_MODEL_NAME", self.model, "model"),
        ):
            # 底层只清理 API Key；模型名按原始环境变量值读取。
            actual = os.getenv(env_name, "")
            if field == "api_key":
                actual = actual.strip()
            if actual and actual != expected:
                raise ValueError(f"{env_name} conflicts with model.vlm.{field}; unset the legacy environment variable")


class ModelConfig(BaseModel):
    base_dir: str = _default_path("models")
    source: str = "auto"
    stack: str = "auto"  # "auto" | "light" | "full"
    vlm: VlmConfig = Field(default_factory=VlmConfig)


class DoclibConfig(BaseModel):
    """Doclib startup configuration.

    Only configuration needed before the doclib server starts belongs here.
    Runtime configuration that can be read from SQLite stays in the config
    service.
    """

    uds: UDSConfig = Field(default_factory=UDSConfig)
    tcp: TCPConfig = Field(default_factory=TCPConfig)
    log: LogConfig = Field(default_factory=LogConfig)
    sqlite: SQLiteConfig = Field(default_factory=SQLiteConfig)
    endpoint_path: str = _default_path("doclib.endpoint.json")
    data_dir: str = _default_path("doclib")
    managed_parse_server: ManagedParseServerConfig = Field(default_factory=ManagedParseServerConfig)
    ingest_workers: int = 2
    parse_workers: int = 2
    scan_interval_sec: int = 300
    device_check_interval_sec: int = 5
    ingest_lock_timeout_sec: int = 60
    parse_lock_timeout_sec: int = 1800
    scan_lock_timeout_sec: int = 1800
    compaction_interval_sec: int = 3600
    parse_server_health_check_interval_sec: int = 30
    parse_server_probe_timeout_sec: int = 10
    parse_server_startup_grace_sec: int = 30
    parse_server_startup_timeout_sec: int = 600
    parse_server_stop_timeout_sec: int = 10

    @property
    def resolved_uds_enabled(self) -> bool:
        if self.uds.enabled == "auto":
            return _uds_available()
        return self.uds.enabled

    @property
    def resolved_tcp_enabled(self) -> bool:
        if self.tcp.enabled == "auto":
            return not _uds_available()
        return self.tcp.enabled


class Config(BaseModel):
    """Top-level MinerU startup configuration."""

    doclib: DoclibConfig = Field(default_factory=DoclibConfig)
    render: RenderConfig = Field(default_factory=RenderConfig)
    llm_aided: LLMAidedConfig = Field(default_factory=LLMAidedConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)


_loaded_config = _load_effective_config()
config = _loaded_config.config


def PatchedConfig(**kwargs: Any) -> Config:
    merged = _deep_merge(to_jsonable_python(config), kwargs)
    return Config(**merged)


__all__ = [
    "AutoBool",
    "ConfigSource",
    "LoadedConfig",
    "ModelConfig",
    "VlmConfig",
    "ModelSource",
    "config",
    "Config",
    "DoclibConfig",
    "LLMAidedConfig",
    "LLMAidedFeaturesConfig",
    "LatexDelimiterConfig",
    "LatexDelimitersConfig",
    "RenderConfig",
    "TCPConfig",
    "LogConfig",
    "SQLiteConfig",
    "UDSConfig",
    "PatchedConfig",
    "get_config_source",
    "get_config_file_path",
    "get_config_file_exists",
    "update_config_file",
    "MINERU_HOME_ENV",
    "MINERU_CONFIG_ENV",
    "MINERU_ENV_PREFIX",
]
