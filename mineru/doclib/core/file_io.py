"""File I/O utilities: SHA-256, stat, metadata extraction."""

from __future__ import annotations

import asyncio
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from typing import TypedDict

from docvortex.schema import DocumentMetadata
from ...filetypes import TEXT_EXTENSIONS, IMAGE_EXTENSIONS

# Metadata truncation limits
TRUNC_TITLE = 500
TRUNC_AUTHOR = 200
TRUNC_SUBJECT = 1000
TRUNC_KEYWORDS = 1000


class MetadataExtractionError(Exception):
    def __init__(self, code: str, message: str) -> None:
        """保留 Doclib 的公开错误码和消息。"""
        super().__init__(message)
        self.code = code


# ── SHA-256 ────────────────────────────────────────────────────────


def _sha256_sync(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


async def compute_sha256(filepath: str) -> str:
    return await asyncio.to_thread(_sha256_sync, filepath)


# ── file stat ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class FileStat:
    size_bytes: int
    mtime_ms: int


async def get_file_stat(filepath: str) -> FileStat:
    def _stat() -> FileStat:
        st = os.stat(filepath)
        return FileStat(size_bytes=st.st_size, mtime_ms=int(st.st_mtime * 1000))

    return await asyncio.to_thread(_stat)


# ── metadata extraction ────────────────────────────────────────────


class DoclibMetadata(TypedDict):
    """映射现有数据库列及入库阶段的属性诊断。"""

    page_count: int | None
    title: str | None
    author: str | None
    subject: str | None
    keywords: str | None
    language: str | None
    is_image_based: int
    error_code: str | None
    error_msg: str | None


def metadata_to_doclib(metadata: DocumentMetadata | None) -> DoclibMetadata:
    """将共享源属性投影为产品字段，截断仅发生在数据库边界。"""
    result: DoclibMetadata = {
        "page_count": None,
        "title": None,
        "author": None,
        "subject": None,
        "keywords": None,
        "language": None,
        "is_image_based": 0,
        "error_code": None,
        "error_msg": None,
    }
    if metadata is None or metadata.document is None:
        return result
    properties = metadata.document
    result.update(
        title=properties.title,
        author="; ".join(properties.authors) or None,
        subject=properties.subject,
        keywords=", ".join(properties.keywords) or None,
        language=properties.languages[0] if properties.languages else None,
        page_count=properties.page_count,
    )
    if metadata.file_suffix in {"doc", "docx", "rtf"}:
        # 源文件声明的布局页数不改变重排版文档的既有调度口径。
        result["page_count"] = 1
    elif metadata.file_suffix in {"odt", "ods", "odp"}:
        result["page_count"] = result["page_count"] or 1
    for field, limit in (
        ("title", TRUNC_TITLE),
        ("author", TRUNC_AUTHOR),
        ("subject", TRUNC_SUBJECT),
        ("keywords", TRUNC_KEYWORDS),
    ):
        value = result[field]
        if isinstance(value, str):
            result[field] = value[:limit]
    return result


async def extract_metadata(filepath: str) -> DoclibMetadata:
    """在线程中调用 DocVortex 公共接口；纯文本与图片维持原有入库行为。"""
    from docvortex import extract_metadata as extract
    from docvortex.errors import DocumentError

    if Path(filepath).suffix.lower().lstrip(".") in TEXT_EXTENSIONS | IMAGE_EXTENSIONS:
        return metadata_to_doclib(None)
    try:
        extracted = await asyncio.to_thread(extract, filepath)
    except DocumentError as exc:
        raise MetadataExtractionError(exc.code, str(exc)) from exc
    result = metadata_to_doclib(extracted.metadata)
    if extracted.diagnostics:
        result["error_code"] = "read_metadata_failed"
        result["error_msg"] = "; ".join(item.message for item in extracted.diagnostics)[:500]
    return result


__all__ = [
    "DoclibMetadata",
    "MetadataExtractionError",
    "FileStat",
    "compute_sha256",
    "get_file_stat",
    "extract_metadata",
    "metadata_to_doclib",
]
