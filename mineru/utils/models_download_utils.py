# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Sequence
from typing import Literal

from filelock import FileLock
from huggingface_hub import snapshot_download as hf_snapshot_download
from loguru import logger
from modelscope import snapshot_download as ms_snapshot_download
import requests

from ..config import config, get_config_source, update_config_file
from .model_registry import ModelPath, ModelRepo, model_path_exists

MODEL_SOURCE_ENV_VAR = "MINERU_MODEL_SOURCE"
_HUGGINGFACE_MODELS_PAGE_URL = "https://huggingface.co/models"
_HUGGINGFACE_MODELS_PAGE_TIMEOUT = 3
_HUGGINGFACE_MODELS_PAGE_MAX_ATTEMPTS = 2
_REMOTE_MODEL_SOURCES = ("huggingface", "modelscope")
DOWNLOAD_MODEL_SOURCES = ("auto", *_REMOTE_MODEL_SOURCES)
ResolvedRemoteModelSource = Literal["huggingface", "modelscope"]
ResolvedModelSource = Literal["huggingface", "modelscope", "local"]


@dataclass(frozen=True)
class ModelReadyResult:
    ready: bool
    root: Path
    missing_paths: list[str] = field(default_factory=list)


def _normalize_source(raw_source: object, *, strict: bool) -> str:
    source = str(raw_source or "auto").strip().lower()
    if source in {"auto", "huggingface", "modelscope", "local"}:
        return source
    if strict:
        expected = ", ".join(DOWNLOAD_MODEL_SOURCES)
        raise ValueError(f"Unsupported model source '{raw_source}'. Expected one of: {expected}.")
    logger.warning(f"Unsupported model source {raw_source!r}; falling back to auto.")
    return "auto"


def _resolve_auto_model_source() -> ResolvedRemoteModelSource:
    """Resolve auto source by probing Hugging Face reachability."""
    last_error = None
    for _ in range(_HUGGINGFACE_MODELS_PAGE_MAX_ATTEMPTS):
        try:
            response = requests.get(
                _HUGGINGFACE_MODELS_PAGE_URL,
                timeout=_HUGGINGFACE_MODELS_PAGE_TIMEOUT,
            )
            if 200 <= response.status_code < 400:
                return "huggingface"
            last_error = f"status_code={response.status_code}"
        except Exception as exc:
            last_error = str(exc)

    logger.warning("Failed to access %s: %s, fallback to modelscope.", _HUGGINGFACE_MODELS_PAGE_URL, last_error)
    return "modelscope"


def _persist_resolved_model_source(model_source: str) -> None:
    if model_source not in _REMOTE_MODEL_SOURCES:
        return
    try:
        update_config_file({"model": {"source": model_source}})
    except Exception as exc:
        logger.warning(f"Failed to persist resolved model source {model_source!r}: {exc}")


def resolve_model_source(
    model_source: str | None = None,
    *,
    allow_auto: bool = False,
    local_as_auto: bool = False,
) -> ResolvedModelSource:
    """Resolve the effective model source.

    Explicit ``model_source`` is CLI input and is validated strictly. When no
    explicit source is supplied, ``config.model.source`` may come from config
    file, environment, or defaults; invalid values are treated as auto.
    """
    explicit_source = model_source is not None
    raw_source = model_source if explicit_source else config.model.source
    normalized_source = _normalize_source(raw_source, strict=explicit_source)

    if explicit_source and normalized_source == "local":
        raise ValueError("--source does not support 'local'. Use auto, huggingface, or modelscope.")

    if normalized_source == "local":
        if not local_as_auto:
            return "local"
        normalized_source = "auto"

    if normalized_source in _REMOTE_MODEL_SOURCES:
        return normalized_source  # type: ignore[return-value]

    if normalized_source != "auto":
        raise ValueError(f"Unsupported model source: {raw_source}")
    if not allow_auto:
        raise ValueError("model source auto is only supported when auto resolution is explicitly allowed.")

    resolved_model_source = _resolve_auto_model_source()
    if not explicit_source and raw_source == "auto" and get_config_source("model.source") in {"default", "file"}:
        _persist_resolved_model_source(resolved_model_source)
    return resolved_model_source


def _snapshot_download(model_source: ResolvedRemoteModelSource, repo: ModelRepo, patterns: list[str] | None) -> str:
    repo_id = repo.repos[model_source]
    local_dir = repo.local_dir()
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {"local_dir": str(local_dir)}
    if patterns is not None:
        kwargs["allow_patterns"] = patterns
    if model_source == "huggingface":
        return hf_snapshot_download(repo_id, **kwargs)
    return ms_snapshot_download(repo_id, **kwargs)


def _path_patterns(paths: list[str]) -> list[str]:
    patterns: list[str] = []
    for path in paths:
        normalized = path.strip("/")
        if not normalized:
            continue
        patterns.extend([normalized, f"{normalized}/*"])
    return patterns


def _relative_paths(repo: ModelRepo, paths: Sequence[str | ModelPath]) -> list[str]:
    relative_paths: list[str] = []
    for path in paths:
        if isinstance(path, ModelPath):
            if path.repo is not repo:
                raise ValueError(
                    f"Model path {path.relative_path!r} belongs to repo {path.repo.name!r}, not {repo.name!r}."
                )
            relative_paths.append(path.relative_path)
        else:
            relative_paths.append(str(path).strip("/"))
    return relative_paths


def _verify_paths(repo: ModelRepo, paths: Sequence[str | ModelPath]) -> ModelReadyResult:
    model_paths = [path if isinstance(path, ModelPath) else repo.path(path) for path in paths]
    missing = [path.relative_path for path in model_paths if not model_path_exists(path)]
    return ModelReadyResult(ready=not missing, root=repo.local_dir(), missing_paths=missing)


def _raise_not_ready(repo: ModelRepo, result: ModelReadyResult) -> None:
    missing = ", ".join(result.missing_paths)
    raise FileNotFoundError(f"Model repo {repo.name} is not ready under {result.root}; missing: {missing}")


def download_model_repo(repo: ModelRepo, *, source: str | None = None, local_as_auto: bool = False) -> Path:
    resolved_source = resolve_model_source(source, allow_auto=True, local_as_auto=local_as_auto)
    if resolved_source == "local":
        result = verify_model_repo(repo)
        if not result.ready:
            _raise_not_ready(repo, result)
        return result.root

    relative_paths = [path.relative_path for path in repo.required_paths()] if repo.download_mode == "required_paths" else None
    lock_path = repo.lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock_path)):
        _snapshot_download(resolved_source, repo, None if relative_paths is None else _path_patterns(relative_paths))
        result = verify_model_repo(repo)
        if not result.ready:
            _raise_not_ready(repo, result)
        return result.root


def download_model_files(
    repo: ModelRepo,
    paths: Sequence[str | ModelPath],
    *,
    source: str | None = None,
    local_as_auto: bool = False,
) -> Path:
    if not paths:
        return repo.local_dir()

    resolved_source = resolve_model_source(source, allow_auto=True, local_as_auto=local_as_auto)
    if resolved_source == "local":
        result = _verify_paths(repo, paths)
        if not result.ready:
            _raise_not_ready(repo, result)
        return result.root

    relative_paths = _relative_paths(repo, paths)
    lock_path = repo.lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock_path)):
        _snapshot_download(resolved_source, repo, _path_patterns(relative_paths))
        result = _verify_paths(repo, paths)
        if not result.ready:
            _raise_not_ready(repo, result)
        return result.root


def verify_model_repo(repo: ModelRepo) -> ModelReadyResult:
    return _verify_paths(repo, list(repo.required_paths()))


__all__ = [
    "DOWNLOAD_MODEL_SOURCES",
    "MODEL_SOURCE_ENV_VAR",
    "ModelReadyResult",
    "download_model_files",
    "download_model_repo",
    "resolve_model_source",
    "verify_model_repo",
]
