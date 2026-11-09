# Copyright (c) Opendatalab. All rights reserved.
"""旧完成标记的尽力而为后台迁移，只查询元数据并原子更新标记。"""

from __future__ import annotations

import io
import os
import pickle
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from urllib.parse import quote

import requests
from filelock import FileLock
from huggingface_hub import HfApi
from loguru import logger

from ._download_manifest import (
    is_legacy_marker,
    is_payload_path,
    legacy_payload_missing,
    manifest_files_for_path,
    missing_manifest_files,
    write_manifest,
)

_QUERY_CONNECT_TIMEOUT = 2.0
_QUERY_TIMEOUT = 5.0
_RETRY_COOLDOWN = 300.0
_MAX_ACTIVE_UPGRADES = 2
_MAX_METADATA_BYTES = 8 * 1024 * 1024
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40,64}\Z")


@dataclass(frozen=True)
class ManifestUpgrade:
    """冻结模型目录及仓库信息，后台任务不再读取调用线程的配置上下文。"""

    root: Path
    lock_path: Path
    repos: dict[str, str]
    markers: tuple[Path, ...]


@dataclass(frozen=True)
class _CachedRevision:
    """记录查询版本及缓存文件版本，兼容 ModelScope 按文件保存不同 commit 的格式。"""

    revision: str
    files: dict[str, str]


class _CacheUnpickler(pickle.Unpickler):
    """仅兼容 ModelScope 缓存中的基础容器，禁止反序列化任意 Python 对象。"""

    def find_class(self, module: str, name: str) -> object:
        """拒绝缓存中引用的全局对象和可执行构造器。"""
        raise pickle.UnpicklingError("Model cache metadata must contain only plain values.")


def _cached_revisions(request: ManifestUpgrade) -> dict[str, _CachedRevision]:
    """读取旧下载版本；ModelScope 多 commit 时必须额外比对每个文件的缓存版本。"""
    scopes = [marker.parent.relative_to(request.root).as_posix() for marker in request.markers]
    scopes = ["" if value == "." else value for value in scopes]
    files: dict[str, dict[str, str]] = {"huggingface": {}, "modelscope": {}}
    metadata_root = request.root / ".cache/huggingface/download"
    if "huggingface" in request.repos and metadata_root.is_dir():
        for metadata in metadata_root.rglob("*.metadata"):
            name = metadata.relative_to(metadata_root).as_posix().removesuffix(".metadata")
            if not is_payload_path(name) or not any(manifest_files_for_path({name: 0}, scope) for scope in scopes):
                continue
            try:
                with metadata.open(encoding="utf-8") as stream:
                    revision = stream.readline(256).strip()
            except (OSError, ValueError):
                continue
            if _COMMIT_PATTERN.fullmatch(revision):
                files["huggingface"][name] = revision
    cache = request.root / ".msc"
    if "modelscope" in request.repos and cache.is_file():
        try:
            with cache.open("rb") as stream:
                content = stream.read(_MAX_METADATA_BYTES + 1)
            records = _CacheUnpickler(io.BytesIO(content)).load() if len(content) <= _MAX_METADATA_BYTES else None
            if isinstance(records, list):
                for record in records:
                    if not isinstance(record, dict):
                        continue
                    name, revision = record.get("Path"), record.get("Revision")
                    if (
                        isinstance(name, str)
                        and is_payload_path(name)
                        and isinstance(revision, str)
                        and _COMMIT_PATTERN.fullmatch(revision)
                        and any(manifest_files_for_path({name: 0}, scope) for scope in scopes)
                    ):
                        files["modelscope"][name] = revision
        except (OSError, ValueError, EOFError, pickle.UnpicklingError):
            pass
    revisions: dict[str, _CachedRevision] = {}
    hf_versions = set(files["huggingface"].values())
    if len(hf_versions) == 1:
        revisions["huggingface"] = _CachedRevision(next(iter(hf_versions)), files["huggingface"])
    ms_versions = set(files["modelscope"].values())
    if ms_versions:
        revision = next(iter(ms_versions)) if len(ms_versions) == 1 else "master"
        if len(ms_versions) > 1:
            try:
                with (request.root / ".mv").open(encoding="utf-8") as stream:
                    cached_version = stream.readline(1024)
                match = re.match(r"Revision:([^,\r\n]+)", cached_version)
                if match:
                    revision = match.group(1)
            except (OSError, ValueError):
                pass
        revisions["modelscope"] = _CachedRevision(revision, files["modelscope"])
    return revisions


def _remote_manifest(
    source: str, repo_id: str, revision: str, *, cached_file_revisions: dict[str, str] | None = None
) -> dict[str, int]:
    """用短超时查询指定旧版本的远程清单，不创建目录、不下载或删除模型文件。"""
    if source == "huggingface":
        info = HfApi().model_info(repo_id, revision=revision, files_metadata=True, timeout=_QUERY_TIMEOUT)
        if info.sha != revision:
            return {}
        files = {item.rfilename: item.size for item in info.siblings or ()}
    elif source == "modelscope":
        # 迁移只访问公开仓库元数据，避免导入 ModelScope 推理 SDK 或触发其下载流程。
        url = f"https://modelscope.cn/api/v1/models/{quote(repo_id, safe='/')}/repo/files"
        with requests.get(
            url,
            params={"Revision": revision, "Recursive": "True"},
            timeout=(_QUERY_CONNECT_TIMEOUT, _QUERY_TIMEOUT),
        ) as response:
            response.raise_for_status()
            value = response.json()
        if not isinstance(value, dict) or value.get("Code") != 200:
            return {}
        data = value.get("Data")
        entries = data.get("Files") if isinstance(data, dict) else None
        # 服务端可能截断大型递归列表；无法证明列表完整时不执行自动迁移。
        if not isinstance(entries, list) or len(entries) >= 1000:
            return {}
        if cached_file_revisions:
            remote_revisions = {
                item["Path"]: item.get("Revision") for item in entries if isinstance(item, dict) and item.get("Type") != "tree"
            }
            if any(remote_revisions.get(name) != value for name, value in cached_file_revisions.items()):
                return {}
        files = {item["Path"]: item["Size"] for item in entries if isinstance(item, dict) and item.get("Type") != "tree"}
    else:
        return {}
    payload: dict[str, int] = {}
    for name, size in files.items():
        if not isinstance(name, str) or not is_payload_path(name):
            continue
        if type(size) is not int or size < 0:
            return {}
        payload[name] = size
    return payload


def _upgrade_markers(request: ManifestUpgrade) -> None:
    """联网取得历史预期集合，核验成功后在下载锁内再次复核并原子替换旧标记。"""
    for source, cached in _cached_revisions(request).items():
        repo_id = request.repos.get(source)
        if repo_id is None:
            continue
        try:
            if source == "modelscope":
                files = _remote_manifest(source, repo_id, cached.revision, cached_file_revisions=cached.files)
            else:
                files = _remote_manifest(source, repo_id, cached.revision)
        except Exception as exc:
            logger.debug(f"Model manifest upgrade query failed for {repo_id}: {exc}")
            continue
        if not files:
            continue
        for marker in request.markers:
            scope = marker.parent.relative_to(request.root).as_posix() if marker.parent != request.root else ""
            selected = manifest_files_for_path(files, scope)
            prefix = f"{scope}/" if scope else ""
            manifest = {name.removeprefix(prefix): size for name, size in selected.items()}
            if not manifest or missing_manifest_files(marker.parent, manifest):
                continue
            # 网络查询不占用下载锁；锁竞争时让位给正常使用，后续冷却结束后再尝试。
            with FileLock(str(request.lock_path), timeout=0):
                if not is_legacy_marker(marker):
                    continue
                if legacy_payload_missing(marker.parent) or missing_manifest_files(marker.parent, manifest):
                    continue
                write_manifest(marker, manifest)


class _UpgradeScheduler:
    """限制并发和失败重试频率，使用守护线程保证离线查询不拖慢退出。"""

    def __init__(self) -> None:
        """只准备进程内同步状态，创建实例不会联网或启动线程。"""
        self.lock = threading.Lock()
        self.active: dict[Path, threading.Thread] = {}
        self.retry_after: dict[Path, float] = {}
        self.fork_handler_registered = False

    def _after_fork(self) -> None:
        """子进程重建锁和任务状态，避免继承已消失的后台线程。"""
        self.lock = threading.Lock()
        self.active = {}
        self.retry_after = {}

    def schedule(self, request: ManifestUpgrade) -> threading.Thread | None:
        """每仓库同时启动一个任务，失败后冷却，所有启动错误均留给旧协议正常使用。"""
        with self.lock:
            if request.root in self.active or monotonic() < self.retry_after.get(request.root, 0):
                return None
            if len(self.active) >= _MAX_ACTIVE_UPGRADES:
                return None
            if not self.fork_handler_registered and hasattr(os, "register_at_fork"):
                os.register_at_fork(after_in_child=self._after_fork)
                self.fork_handler_registered = True
            worker = threading.Thread(target=self._run, args=(request,), name="mineru-manifest-upgrade", daemon=True)
            self.active[request.root] = worker
            try:
                worker.start()
            except Exception:
                self.active.pop(request.root, None)
                self.retry_after[request.root] = monotonic() + _RETRY_COOLDOWN
                return None
            return worker

    def _run(self, request: ManifestUpgrade) -> None:
        """后台迁移异常只记录调试信息，始终释放并发配额且保留失败冷却。"""
        try:
            _upgrade_markers(request)
        except Exception as exc:
            logger.debug(f"Model manifest upgrade skipped for {request.root.name}: {exc}")
        finally:
            with self.lock:
                self.active.pop(request.root, None)
                self.retry_after[request.root] = monotonic() + _RETRY_COOLDOWN


_SCHEDULER = _UpgradeScheduler()


def schedule_manifest_upgrade(request: ManifestUpgrade) -> threading.Thread | None:
    """仅在存在旧标记时安排迁移，不使用或改写 model.source，包括 local 配置。"""
    if not any(is_legacy_marker(marker) for marker in request.markers):
        return None
    return _SCHEDULER.schedule(request)


__all__ = ["ManifestUpgrade", "schedule_manifest_upgrade"]
