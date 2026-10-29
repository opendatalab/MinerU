"""Gradio 宿主可替换的文档读写边界，显式隔离 PDFium 与产物持久化。"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ...parser.base import ParseResult
from . import artifacts, page_range
from .artifacts import RunArtifacts
from .page_range import PdfPageMetadata


class GradioDocumentIO(Protocol):
    """定义界面和转换服务共享的同步文档操作，不约束宿主的进程实现。"""

    def pdf_page_metadata(self, file_path: str | None) -> PdfPageMetadata:
        """读取上传文件页数，返回可序列化的文件身份、页数和错误。"""
        ...

    def effective_page_range(
        self,
        path: str | Path | None,
        raw_page_range: str | None,
        *,
        max_pages: int | None = None,
    ) -> str:
        """校验实际页范围及页数上限，保留既有范围规范化语义。"""
        ...

    def persist_parse_result(
        self,
        result: ParseResult,
        source_path: Path,
        *,
        output_root: Path,
        page_range: str,
    ) -> RunArtifacts:
        """物化解析结果及布局预览，只返回路径和可序列化的产物描述。"""
        ...


class LocalGradioDocumentIO:
    """普通 WebUI 的进程内实现，实例化时不打开文档或创建目录。"""

    def pdf_page_metadata(self, file_path: str | None) -> PdfPageMetadata:
        """委托现有页数读取实现，沿用共享 PDFium 锁和错误格式。"""
        return page_range.pdf_page_metadata(file_path)

    def effective_page_range(
        self,
        path: str | Path | None,
        raw_page_range: str | None,
        *,
        max_pages: int | None = None,
    ) -> str:
        """委托现有选页逻辑，保持整本解析和最大页数限制。"""
        return page_range.effective_page_range(path, raw_page_range, max_pages=max_pages)

    def persist_parse_result(
        self,
        result: ParseResult,
        source_path: Path,
        *,
        output_root: Path,
        page_range: str,
    ) -> RunArtifacts:
        """委托现有持久化实现，不在适配层保留解析树或 PDF 对象。"""
        return artifacts.persist_parse_result(result, source_path, output_root=output_root, page_range=page_range)


__all__ = ["GradioDocumentIO", "LocalGradioDocumentIO"]
