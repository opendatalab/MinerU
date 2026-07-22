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

import yaml
from filelock import FileLock
from pydantic import BaseModel, Field
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


class ModelConfig(BaseModel):
    base_dir: str = _default_path("models")
    source: str = "auto"


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
    parse_server_health_check_interval_sec: int = 60
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
    model: ModelConfig = Field(default_factory=ModelConfig)
    # render: RenderConfig


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
    "ModelSource",
    "config",
    "Config",
    "DoclibConfig",
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
