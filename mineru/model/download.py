# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import requests
from filelock import FileLock
from huggingface_hub import HfApi
from huggingface_hub import snapshot_download as hf_snapshot_download
from huggingface_hub.utils import filter_repo_objects
from loguru import logger

from ..config import get_config, get_config_source, update_config_file
from ._download_manifest import (
    MODEL_COMPLETE_MARKER,
    is_legacy_marker,
    is_payload_path,
    legacy_payload_missing,
    manifest_files_for_path,
    missing_manifest_files,
    read_manifest,
    write_manifest,
)
from ._manifest_upgrade import ManifestUpgrade, schedule_manifest_upgrade

if TYPE_CHECKING:
    from ..config import ModelSource

DownloadMode = Literal["full", "required_paths"]
_RemovedMarker = dict[str, int] | Literal["legacy"] | None


def _join_model_path(*parts: str) -> str:
    """连接并校验模型仓库内的相对路径片段。"""
    cleaned: list[str] = []
    for raw in parts:
        value = str(raw).replace("\\", "/").strip("/")
        if not value:
            continue
        cleaned.extend(part for part in value.split("/") if part and part != ".")
    if any(part == ".." for part in cleaned):
        raise ValueError("Model paths must not contain '..' segments.")
    return "/".join(cleaned)


@dataclass(frozen=True)
class ModelPath:
    """表示模型仓库中的一个具名相对路径。"""

    repo: "ModelRepo"
    name: str
    relative_path: str

    def path(self, relative_path: str, /, *children: str) -> "ModelPath":
        """基于当前路径派生一个更深的模型路径。"""
        joined = _join_model_path(self.relative_path, relative_path, *children)
        name = _join_model_path(self.name, relative_path, *children)
        return ModelPath(repo=self.repo, name=name, relative_path=joined)

    def local_path(self) -> Path:
        """返回该模型路径在本地模型根目录中的位置。"""
        return self.repo.local_dir() / self.relative_path

    def ensure(self, *, source: "ModelSource | None" = None) -> Path:
        """确保当前模型路径可用，并在返回前安排旧标记的后台迁移。"""
        download_model_files(self.repo, [self], source=source, reuse_ready=True)
        _schedule_legacy_upgrade(self.repo, [self])
        return self.local_path()


@dataclass(frozen=True)
class ModelRepo:
    """定义一个可从多来源获取的模型仓库。"""

    name: str
    repos: dict[str, str]
    local_name: str = ""
    paths: dict[str, str] = field(default_factory=dict)
    download_mode: DownloadMode = "full"

    def __post_init__(self) -> None:
        """缺少本地名称时使用稳定仓库名。"""
        if not self.local_name:
            object.__setattr__(self, "local_name", self.name)

    def __getattr__(self, name: str) -> ModelPath:
        """允许通过具名字段读取仓库预定义路径。"""
        try:
            return self.named_path(name)
        except KeyError as exc:
            raise AttributeError(name) from exc

    def named_path(self, name: str) -> ModelPath:
        """返回仓库中已声明的具名模型路径。"""
        return ModelPath(repo=self, name=name, relative_path=self.paths[name])

    def path(self, relative_path: str, /, *children: str) -> ModelPath:
        """按任意安全相对路径构造模型路径。"""
        joined = _join_model_path(relative_path, *children)
        return ModelPath(repo=self, name=joined, relative_path=joined)

    def required_paths(self) -> tuple[ModelPath, ...]:
        """返回仓库声明的全部必需路径。"""
        return tuple(self.named_path(name) for name in self.paths)

    def local_dir(self) -> Path:
        """返回仓库在 MinerU 模型目录中的本地根路径。"""
        return Path(get_config().model.base_dir).expanduser() / self.local_name

    def lock_path(self) -> Path:
        """返回仓库下载互斥锁的稳定路径。"""
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.local_name).strip("._") or "model"
        return Path(get_config().model.base_dir).expanduser() / ".locks" / f"{safe_name}.lock"

    def ensure(self, *, source: "ModelSource | None" = None) -> Path:
        """确保完整仓库可用并返回根目录，旧标记的联网迁移不阻塞当前使用。"""
        root = download_model_repo(self, source=source, reuse_ready=True)
        _schedule_legacy_upgrade(self, _repo_completion_paths(self))
        return root


MODEL_SOURCE_ENV_VAR = "MINERU_MODEL_SOURCE"
_HUGGINGFACE_MODELS_PAGE_URL = "https://huggingface.co/models"
_HUGGINGFACE_MODELS_PAGE_TIMEOUT = 3
_HUGGINGFACE_MODELS_PAGE_MAX_ATTEMPTS = 2
_REMOTE_MODEL_SOURCES = ("huggingface", "modelscope")
DOWNLOAD_MODEL_SOURCES = ("auto", *_REMOTE_MODEL_SOURCES)
ResolvedRemoteModelSource = Literal["huggingface", "modelscope"]
ResolvedModelSource = Literal["huggingface", "modelscope", "local"]


def ms_snapshot_download(repo_id: str, *, local_dir: str, allow_patterns: list[str] | None = None) -> str:
    """仅在实际使用 ModelScope 时导入 SDK，避免 HF/本地路径触发其 Torch 探测。"""
    from modelscope import snapshot_download

    return snapshot_download(repo_id, local_dir=local_dir, allow_patterns=allow_patterns)


def ms_model_files(repo_id: str) -> dict[str, int]:
    """按下载器默认版本读取 ModelScope 预期文件，仅在使用该来源时加载 SDK。"""
    from modelscope.hub.api import HubApi
    from modelscope.utils.constant import DEFAULT_MODEL_REVISION

    files = HubApi().get_model_files(repo_id, revision=DEFAULT_MODEL_REVISION, recursive=True)
    return {item["Path"]: item["Size"] for item in files if item.get("Type") != "tree"}


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
    explicit source is supplied, ``get_config().model.source`` may come from config
    file, environment, or defaults; invalid values are treated as auto.
    """
    explicit_source = model_source is not None
    raw_source = model_source if explicit_source else get_config().model.source
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


def _source_repo_id(repo: ModelRepo, model_source: str) -> str:
    """校验仓库是否已发布到所选来源，避免错误回退到旧仓库。"""
    if model_source not in repo.repos:
        available = ", ".join(repo.repos)
        raise ValueError(f"Model repo {repo.name} is not available from {model_source}. Available sources: {available}.")
    return repo.repos[model_source]


def _snapshot_download(model_source: ResolvedRemoteModelSource, repo: ModelRepo, patterns: list[str] | None) -> dict[str, int]:
    """按远程预期清单下载并校验快照，返回可用于本地复用检查的文件及大小。"""
    repo_id = _source_repo_id(repo, model_source)
    local_dir = repo.local_dir()
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    if model_source == "huggingface":
        info = HfApi().model_info(repo_id, files_metadata=True)
        files = {item.rfilename: item.size for item in info.siblings or ()}
    else:
        files = ms_model_files(repo_id)
    expected_files: dict[str, int] = {}
    for name in filter_repo_objects(files, allow_patterns=patterns):
        if not is_payload_path(name):
            continue
        size = files[name]
        if type(size) is not int or size < 0:
            raise ValueError(f"Missing or invalid model file size from {model_source}: {name!r}")
        expected_files[name] = size
    if not expected_files:
        raise FileNotFoundError(f"{model_source} snapshot for {repo.name} has no files matching the requested paths.")
    # 下载器可能凭借自己的缓存元数据跳过已有文件；先移除大小不符的失效文件，保证增量修复实际执行。
    for name in missing_manifest_files(local_dir, expected_files):
        target = local_dir / name
        if target.is_file() or target.is_symlink():
            target.unlink()
    requested_files = list(expected_files)
    if model_source == "huggingface":
        hf_snapshot_download(repo_id, local_dir=str(local_dir), allow_patterns=requested_files, revision=info.sha)
    else:
        ms_snapshot_download(repo_id, local_dir=str(local_dir), allow_patterns=requested_files)
    missing_files = missing_manifest_files(local_dir, expected_files)
    if missing_files:
        missing = ", ".join(missing_files)
        raise FileNotFoundError(f"{model_source} snapshot for {repo.name} is incomplete; missing or invalid: {missing}")
    return expected_files


def _path_patterns(paths: list[str]) -> list[str] | None:
    """将请求路径展开为下载模式；仓库根路径表示完整快照。"""
    if "" in paths:
        return None
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
                raise ValueError(f"Model path {path.relative_path!r} belongs to repo {path.repo.name!r}, not {repo.name!r}.")
            relative_paths.append(path.relative_path)
        else:
            relative_paths.append(str(path).strip("/"))
    return relative_paths


def _verify_paths(repo: ModelRepo, paths: Sequence[str | ModelPath]) -> ModelReadyResult:
    """统一检查文件和目录，并保留清单给出的具体失效文件路径。"""
    model_paths = [path if isinstance(path, ModelPath) else repo.path(path) for path in paths]
    missing = list(dict.fromkeys(name for path in model_paths for name in _model_path_missing(path)))
    return ModelReadyResult(ready=not missing, root=repo.local_dir(), missing_paths=missing)


def _model_path_missing(path: ModelPath) -> list[str]:
    """验证请求范围内的完成清单，目录继承祖先清单时也必须包含自身文件。"""
    local_path = path.local_path()
    repo_root = path.repo.local_dir()
    try:
        local_path.relative_to(repo_root)
    except ValueError:
        return [path.relative_path]
    is_file = local_path.is_file()
    if local_path == repo_root and is_file:
        return [MODEL_COMPLETE_MARKER]
    directory = local_path.parent if is_file else local_path
    legacy_marker_found = False
    for parent in (directory, *directory.parents):
        marker = parent / MODEL_COMPLETE_MARKER
        if marker.is_file():
            files = read_manifest(marker)
            marker_name = marker.relative_to(repo_root).as_posix()
            if files is None:
                if is_legacy_marker(marker):
                    # 先查找祖先的新版清单，避免旧子标记绕过其中记录的严格文件检查。
                    legacy_marker_found = True
                elif not is_file:
                    # 损坏或未知版本的清单不能降级为旧标记；独立文件继续遵守文件路径契约。
                    return [marker_name if local_path.is_dir() or local_path == repo_root else path.relative_path]
            else:
                selected = manifest_files_for_path(
                    files, local_path.relative_to(parent).as_posix() if parent != local_path else ""
                )
                if selected:
                    return [
                        (parent / name).relative_to(repo_root).as_posix() for name in missing_manifest_files(parent, selected)
                    ]
                if not is_file:
                    return [path.relative_path or marker_name]
        if parent == repo_root:
            break
    if legacy_marker_found:
        return _legacy_model_path_missing(path)
    if is_file:
        try:
            if local_path.stat().st_size > 0:
                return []
        except OSError:
            pass
        return [path.relative_path]
    if local_path == repo_root:
        return [MODEL_COMPLETE_MARKER]
    return [path.relative_path]


def _legacy_model_path_missing(path: ModelPath) -> list[str]:
    """旧标记按本地必要路径和真实权重只读检查，迁移由 ensure 返回前另行安排。"""
    root = path.repo.local_dir()
    local_path = path.local_path()
    missing = legacy_payload_missing(local_path)
    if local_path.is_dir():
        for required in path.repo.required_paths():
            if required.local_path() != local_path and required.local_path().is_relative_to(local_path):
                missing.extend(legacy_payload_missing(required.local_path()))
    return list(dict.fromkeys(item.relative_to(root).as_posix() if item != root else "<payload>" for item in missing))


def _verify_downloaded_paths(repo: ModelRepo, paths: Sequence[str | ModelPath], files: dict[str, int]) -> ModelReadyResult:
    """下载后要求每个请求路径均被远程清单覆盖，已有本地残留不能替代缺失文件。"""
    model_paths = [path if isinstance(path, ModelPath) else repo.path(path) for path in paths]
    missing = missing_manifest_files(repo.local_dir(), files)
    missing.extend(
        path.relative_path or "<payload>" for path in model_paths if not manifest_files_for_path(files, path.relative_path)
    )
    return ModelReadyResult(ready=not missing, root=repo.local_dir(), missing_paths=missing)


def _repo_completion_paths(repo: ModelRepo) -> list[ModelPath]:
    if repo.download_mode == "full":
        return [repo.path("")]
    return list(repo.required_paths())


def _schedule_legacy_upgrade(repo: ModelRepo, paths: Sequence[ModelPath]) -> None:
    """使用后捕获旧标记范围，在下载锁外安排元数据迁移，不影响当前模型加载。"""
    try:
        root = repo.local_dir().absolute()
        markers: set[Path] = set()
        for path in (*paths, *repo.required_paths()):
            local_path = path.local_path().absolute()
            directory = local_path if local_path.is_dir() else local_path.parent
            if not directory.is_relative_to(root):
                continue
            for parent in (directory, *directory.parents):
                marker = parent / MODEL_COMPLETE_MARKER
                if is_legacy_marker(marker):
                    markers.add(marker)
                if parent == root:
                    break
        # 有父标记时按完整父范围迁移，子目录请求不会被误记成整仓完整清单。
        selected = tuple(
            marker
            for marker in sorted(markers)
            if not any(marker != other and marker.is_relative_to(other.parent) for other in markers)
        )
        if selected:
            schedule_manifest_upgrade(
                ManifestUpgrade(root=root, lock_path=repo.lock_path().absolute(), repos=dict(repo.repos), markers=selected)
            )
    except Exception as exc:
        logger.debug(f"Could not schedule legacy model manifest upgrade for {repo.name}: {exc}")


def _remove_directory_markers(paths: Sequence[ModelPath]) -> dict[Path, _RemovedMarker]:
    """下载前撤销请求范围及祖先标记，并保存可在成功后重新验证的原清单。"""
    removed_markers: dict[Path, _RemovedMarker] = {}
    for path in paths:
        local_path = path.local_path()
        repo_root = path.repo.local_dir()
        directory = local_path if local_path.is_dir() or local_path == repo_root else local_path.parent
        for directory in (directory, *directory.parents):
            marker = directory / MODEL_COMPLETE_MARKER
            if marker.is_file():
                removed_markers[marker] = "legacy" if is_legacy_marker(marker) else read_manifest(marker)
                marker.unlink()
            if directory == repo_root:
                break
    return removed_markers


def _mark_directories_complete(paths: Sequence[ModelPath], files: dict[str, int]) -> None:
    """仅用本次下载确认的文件清单原子发布请求目录的完成标记。"""
    manifests: dict[Path, dict[str, int]] = {}
    for path in paths:
        local_path = path.local_path()
        if local_path.is_dir():
            selected = manifest_files_for_path(files, path.relative_path)
            prefix = f"{path.relative_path}/" if path.relative_path else ""
            manifests[local_path] = {name.removeprefix(prefix): size for name, size in selected.items()}
    # 整个目录已被重新确认时，其文件祖先中的旧子清单应由新清单接管，避免离线加载仍命中旧标记。
    removed: set[Path] = set()
    for root, manifest in manifests.items():
        for name in manifest:
            directory = (root / name).parent
            while directory != root:
                marker = directory / MODEL_COMPLETE_MARKER
                if marker not in removed and marker.is_file():
                    marker.unlink()
                    removed.add(marker)
                directory = directory.parent
    for root, manifest in manifests.items():
        write_manifest(root / MODEL_COMPLETE_MARKER, manifest)


def _restore_directory_markers(
    repo: ModelRepo,
    markers: dict[Path, _RemovedMarker],
    paths: Sequence[ModelPath],
    files: dict[str, int],
) -> None:
    """合并局部更新后重新验证父清单，其他成员缺失时不恢复父目录就绪状态。"""
    for marker, previous in markers.items():
        if marker.is_file() or previous is None:
            continue
        root = marker.parent
        root_name = root.relative_to(repo.local_dir()).as_posix() if root != repo.local_dir() else ""
        if previous == "legacy":
            if not _legacy_model_path_missing(repo.path(root_name)):
                marker.touch(exist_ok=True)
            continue
        prefix = f"{root_name}/" if root_name else ""
        updated = dict(previous)
        for path in paths:
            if not path.local_path().is_relative_to(root):
                continue
            relative_path = path.local_path().relative_to(root).as_posix() if path.local_path() != root else ""
            for name in manifest_files_for_path(updated, relative_path):
                del updated[name]
        updated.update({name.removeprefix(prefix): size for name, size in manifest_files_for_path(files, root_name).items()})
        if updated and not missing_manifest_files(root, updated):
            write_manifest(marker, updated)


def _raise_not_ready(repo: ModelRepo, result: ModelReadyResult) -> None:
    """离线校验失败时报告真实路径，并提示修复文件或完成标记。"""
    missing = ", ".join(result.missing_paths)
    raise FileNotFoundError(
        f"Model repo {repo.name} is not ready under {result.root}; missing or invalid: {missing}. "
        f"Run: mineru-kit models download {repo.name} to repair files or completion markers."
    )


def download_model_repo(
    repo: ModelRepo,
    *,
    source: str | None = None,
    local_as_auto: bool = False,
    reuse_ready: bool = False,
) -> Path:
    """在仓库锁内验证或补齐请求资源，成功后发布可离线验证的完成清单。"""
    completion_paths = _repo_completion_paths(repo)
    relative_paths = [path.relative_path for path in completion_paths] if repo.download_mode == "required_paths" else None
    lock_path = repo.lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock_path)):
        if reuse_ready:
            ready_result = verify_model_repo(repo)
            if ready_result.ready:
                return ready_result.root

        resolved_source = resolve_model_source(source, allow_auto=True, local_as_auto=local_as_auto)
        if resolved_source == "local":
            result = verify_model_repo(repo)
            if not result.ready:
                _raise_not_ready(repo, result)
            return result.root

        _source_repo_id(repo, resolved_source)
        removed_markers = _remove_directory_markers(completion_paths)
        files = _snapshot_download(resolved_source, repo, None if relative_paths is None else _path_patterns(relative_paths))
        downloaded_result = _verify_downloaded_paths(repo, completion_paths, files)
        if not downloaded_result.ready:
            _raise_not_ready(repo, downloaded_result)
        _mark_directories_complete(completion_paths, files)
        _restore_directory_markers(repo, removed_markers, completion_paths, files)
        return downloaded_result.root


def download_model_files(
    repo: ModelRepo,
    paths: Sequence[str | ModelPath],
    *,
    source: str | None = None,
    local_as_auto: bool = False,
    reuse_ready: bool = False,
) -> Path:
    """按请求范围增量下载，并与整仓入口使用相同的就绪判定和标记生命周期。"""
    if not paths:
        return repo.local_dir()

    model_paths = [path if isinstance(path, ModelPath) else repo.path(path) for path in paths]
    relative_paths = _relative_paths(repo, paths)
    lock_path = repo.lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock_path)):
        if reuse_ready:
            ready_result = _verify_paths(repo, model_paths)
            if ready_result.ready:
                return ready_result.root

        resolved_source = resolve_model_source(source, allow_auto=True, local_as_auto=local_as_auto)
        if resolved_source == "local":
            result = _verify_paths(repo, model_paths)
            if not result.ready:
                _raise_not_ready(repo, result)
            return result.root

        _source_repo_id(repo, resolved_source)
        removed_markers = _remove_directory_markers(model_paths)
        files = _snapshot_download(resolved_source, repo, _path_patterns(relative_paths))
        downloaded_result = _verify_downloaded_paths(repo, model_paths, files)
        if not downloaded_result.ready:
            _raise_not_ready(repo, downloaded_result)
        _mark_directories_complete(model_paths, files)
        _restore_directory_markers(repo, removed_markers, model_paths, files)
        return downloaded_result.root


def verify_model_repo(repo: ModelRepo) -> ModelReadyResult:
    """仅通过本地文件与完成清单检查仓库，不触发网络请求或修改模型目录。"""
    return _verify_paths(repo, _repo_completion_paths(repo))


def model_path_exists(path: ModelPath) -> bool:
    """与仓库检查一致地判断请求文件或目录是否就绪。"""
    return not _model_path_missing(path)


__all__ = [
    "DOWNLOAD_MODEL_SOURCES",
    "DownloadMode",
    "MODEL_COMPLETE_MARKER",
    "MODEL_SOURCE_ENV_VAR",
    "ModelPath",
    "ModelReadyResult",
    "ModelRepo",
    "download_model_files",
    "download_model_repo",
    "model_path_exists",
    "resolve_model_source",
    "verify_model_repo",
]
