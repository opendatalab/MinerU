# Copyright (c) Opendatalab. All rights reserved.
"""模型下载完成清单的本地读写与文件校验。"""

from __future__ import annotations

import json
import os
import stat
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath, PureWindowsPath

MODEL_COMPLETE_MARKER = ".mineru_complete"
MODEL_FILES_MANIFEST = ".mineru_files.json"
_MANIFEST_VERSION = 1
_PROVIDER_METADATA_DIR_NAMES = {".cache", "._____temp"}
_PROVIDER_METADATA_FILE_NAMES = {MODEL_COMPLETE_MARKER, MODEL_FILES_MANIFEST, ".msc", ".mdl", ".mv"}
_WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pth", ".pt", ".onnx", ".gguf", ".pdiparams"}


def is_payload_path(value: str) -> bool:
    """只接受仓库内规范的相对文件路径，并排除下载器元数据。"""
    path = PurePosixPath(value)
    return bool(
        value
        and path.parts
        and "\\" not in value
        and not any(ord(character) < 32 for character in value)
        and not path.is_absolute()
        and not PureWindowsPath(value).drive
        and path.as_posix() == value
        and ".." not in path.parts
        and path.name not in _PROVIDER_METADATA_FILE_NAMES
        and not any(part in _PROVIDER_METADATA_DIR_NAMES for part in path.parts)
        and not path.name.startswith(f"{MODEL_COMPLETE_MARKER}.")
    )


def manifest_files_for_path(files: Mapping[str, int], relative_path: str) -> dict[str, int]:
    """选取请求文件或目录覆盖的清单条目，保持原有相对路径。"""
    if not relative_path:
        return dict(files)
    prefix = f"{relative_path}/"
    return {name: size for name, size in files.items() if name == relative_path or name.startswith(prefix)}


def read_manifest(marker: Path) -> dict[str, int] | None:
    """读取版本化清单；没有新版清单时由调用者单独判断是否为旧空标记。"""
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(value, dict) or type(value.get("version")) is not int or value["version"] != _MANIFEST_VERSION:
        return None
    files = value.get("files")
    if not isinstance(files, dict) or not files:
        return None
    if any(
        not isinstance(name, str) or not is_payload_path(name) or type(size) is not int or size < 0
        for name, size in files.items()
    ):
        return None
    return files


def is_legacy_marker(marker: Path) -> bool:
    """仅将历史下载器写入的零字节常规文件识别为旧完成标记。"""
    try:
        info = marker.stat()
    except OSError:
        return False
    return stat.S_ISREG(info.st_mode) and info.st_size == 0


def _file_has_content(path: Path) -> bool:
    """通过文件类型和大小排除空权重、目录和失效软链接。"""
    try:
        info = path.stat()
    except OSError:
        return False
    return stat.S_ISREG(info.st_mode) and info.st_size > 0


def _legacy_index_missing(index: Path) -> list[Path]:
    """利用旧模型自带的权重索引检查全部分片，不依赖远端或扫描生成的清单。"""
    try:
        value = json.loads(index.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return [index]
    weight_map = value.get("weight_map") if isinstance(value, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        return [index]
    shards = list(weight_map.values())
    if any(not isinstance(name, str) or not is_payload_path(name) for name in shards):
        return [index]
    return [index.parent / name for name in dict.fromkeys(shards) if not _file_has_content(index.parent / name)]


def legacy_payload_missing(path: Path) -> list[Path]:
    """兼容旧标记时检查真实权重和索引分片；README、配置及缓存不能单独证明可用。"""
    if path.is_file():
        return [] if _file_has_content(path) else [path]
    if not path.is_dir():
        return [path]
    weights: list[Path] = []
    missing: list[Path] = []
    for root, directories, names in os.walk(path):
        directories[:] = sorted(name for name in directories if name not in _PROVIDER_METADATA_DIR_NAMES)
        for name in sorted(names):
            candidate = Path(root) / name
            if name == MODEL_COMPLETE_MARKER and candidate.parent != path:
                files = read_manifest(candidate)
                if files is not None:
                    missing.extend(candidate.parent / item for item in missing_manifest_files(candidate.parent, files))
                elif not is_legacy_marker(candidate):
                    missing.append(candidate)
                continue
            if not is_payload_path(candidate.relative_to(path).as_posix()):
                continue
            if candidate.suffix.lower() in _WEIGHT_SUFFIXES:
                weights.append(candidate)
                if not _file_has_content(candidate):
                    missing.append(candidate)
            elif name.endswith((".safetensors.index.json", ".bin.index.json")):
                missing.extend(_legacy_index_missing(candidate))
    if not weights and not missing:
        missing.append(path)
    return list(dict.fromkeys(missing))


def missing_manifest_files(root: Path, files: Mapping[str, int]) -> list[str]:
    """返回清单中缺失、不是有效文件或大小发生变化的路径。"""
    missing: list[str] = []
    for name, size in files.items():
        try:
            info = (root / name).stat()
        except OSError:
            missing.append(name)
            continue
        if not stat.S_ISREG(info.st_mode) or info.st_size != size:
            missing.append(name)
    return missing


def write_manifest(marker: Path, files: Mapping[str, int]) -> None:
    """在同一目录原子发布清单，避免中断时留下半写入的完成标记。"""
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=marker.parent,
            prefix=f"{MODEL_COMPLETE_MARKER}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump({"version": _MANIFEST_VERSION, "files": dict(files)}, temporary, ensure_ascii=False, sort_keys=True)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.replace(marker)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


__all__ = [
    "MODEL_COMPLETE_MARKER",
    "MODEL_FILES_MANIFEST",
    "is_legacy_marker",
    "is_payload_path",
    "legacy_payload_missing",
    "manifest_files_for_path",
    "missing_manifest_files",
    "read_manifest",
    "write_manifest",
]
