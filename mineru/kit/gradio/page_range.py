"""Gradio 专用的 PDF 页数预检与单次选页限制，不改变通用解析接口。"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from ...errors import InvalidRequestError
from ...parser.page_range import count_pages_in_range, expand_page_range, normalize_page_range_input
from ...types import Tier


class PdfPageMetadata(TypedDict):
    """只传递可序列化的文件标识、页数和错误，不保留 PDFium 对象。"""

    path: str
    page_count: int
    error: str


def validate_max_pages(max_pages: int | None) -> None:
    """校验启动配置，None 表示不限制，显式值只接受正整数。"""
    if max_pages is not None and (type(max_pages) is not int or max_pages <= 0):
        raise ValueError("max_pages must be a positive integer or None")


def read_pdf_page_count(path: str | Path) -> int:
    """在共享 PDFium 锁内读取页数，并在成功或异常时及时释放文档。"""
    import pypdfium2 as pdfium

    from ...model.flash.pdf.pdfium import pdfium_guard

    try:
        with pdfium_guard():
            document = pdfium.PdfDocument(str(path))
            try:
                page_count = len(document)
            finally:
                document.close()
        if page_count <= 0:
            raise ValueError("document has no available pages")
    except Exception as exc:
        raise InvalidRequestError(
            "page_range_invalid",
            "无法读取 PDF 页数，请检查文件是否损坏或需要密码。",
            "page_range",
        ) from exc
    return page_count


def pdf_page_metadata(file_path: str | None) -> PdfPageMetadata:
    """为上传事件读取 PDF 页数；保留文件标识，供前端丢弃过期响应。"""
    metadata: PdfPageMetadata = {"path": str(file_path or ""), "page_count": 0, "error": ""}
    if file_path and Path(file_path).suffix.lower() == ".pdf":
        try:
            metadata["page_count"] = read_pdf_page_count(file_path)
        except InvalidRequestError as exc:
            metadata["error"] = str(exc)
    return metadata


def effective_page_range(
    path: str | Path | None,
    raw_page_range: str | None,
    *,
    tier: Tier,
    max_pages: int | None = None,
) -> str:
    """按真实 PDF 页数校验 Gradio 请求，防止绕过前端提交超限范围。"""
    validate_max_pages(max_pages)
    if tier == "flash" or not path or Path(path).suffix.lower() != ".pdf":
        return ""
    normalized = normalize_page_range_input(raw_page_range)
    page_count = read_pdf_page_count(path)
    if not normalized and max_pages is not None:
        normalized = normalize_page_range_input(f"1-{min(page_count, max_pages)}")
    resolved = expand_page_range(normalized, page_count)
    selected_count = count_pages_in_range(resolved)
    if max_pages is not None and selected_count > max_pages:
        raise InvalidRequestError(
            "page_range_invalid",
            f"单次最多解析 {max_pages} 页，当前选择了 {selected_count} 页。",
            "page_range",
        )
    return normalized


__all__ = ["PdfPageMetadata", "effective_page_range", "pdf_page_metadata", "read_pdf_page_count", "validate_max_pages"]
